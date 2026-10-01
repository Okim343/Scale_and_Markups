"""CLI entry point for the homogeneous-scalability counterfactual.

Reads a prior :mod:`counterfactuals.scalability_sorting` run's outputs
(``sorting_objects.yaml``, ``sorting_economies.parquet``) for the baseline and
shuffle-average ``lambda_K`` legs, and computes only the *homogeneous* leg
itself: a per-sector harmonic-mean collapse of alpha
(:func:`~counterfactuals.homogeneous_rts.homogenize.homogenize_alpha_primitive`)
that holds each sector's mean curvature ``kappa_bar_j`` fixed while zeroing
within-sector alpha dispersion, plus an economy-wide cost-weighted robustness
variant. Because ``log(1+lambda) = (1-beta)*Delta_W`` exactly
(:func:`steady_state.welfare.welfare_metrics.consumption_equivalent_lambda`),
the three legs decompose ``Delta_base`` additively into a "common markup"
term, a within-sector "dispersion" term, and a "sorting" term.

Everything under ``steady_state/`` and both ``fixed_input_welfare`` modules
are imported unchanged; this package only adds the homogenizing transforms in
``homogenize.py``. Since the sorting run and this run are decoupled processes
sharing only a base draw built from the same seed, a consistency guard
(``consistency_check.yaml``) recomputes the baseline MARKET economy and diffs
it against the stored sorting-run baseline row before anything else runs.

Run from ``04_Code/``, after a ``scalability_sorting`` run has produced its
outputs::

    python -m counterfactuals.homogeneous_rts.run_homogeneous_rts \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from steady_state.__main__ import (
    _params_for_command,
    _pooled_inputs,
    _write_yaml,
    load_config,
)
from steady_state.model.normalization import euler_R, solve_pooled_ge
from steady_state.model.pool import draw_pool
from steady_state.model.pricing import PricingRegime
from steady_state.simulation.cross_section import build_firm_panel
from steady_state.welfare.welfare_metrics import (
    aggregate_markup_cw,
    consumption_equivalent_lambda,
    steady_state_welfare,
)

from counterfactuals.fixed_input_welfare.decomposition import (
    fixed_capital_decomposition,
)
from counterfactuals.fixed_input_welfare.fixed_capital_planner import (
    solve_fixed_capital_planner,
)

from .homogenize import (
    homogenize_alpha_cost_weighted,
    homogenize_alpha_primitive,
    sector_mean_inv_alpha,
)

_VARIANTS = ("primitive", "cost_weighted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.homogeneous_rts.run_homogeneous_rts",
        description="Homogeneous-scalability counterfactual: per-sector "
                    "curvature-preserving collapse of alpha, decomposing the "
                    "scalability_sorting run's markup-reallocation loss into "
                    "common-markup / dispersion / sorting terms.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--calibration", default="out_results/calib_pooled/calibration_pooled.yaml",
        help="calibration YAML (must match the scalability_sorting run's).",
    )
    parser.add_argument(
        "--sorting-dir", default=None,
        help="directory holding sorting_objects.yaml / sorting_economies.parquet "
             "(default out_results/counterfactuals/scalability_sorting).",
    )
    parser.add_argument("--out-dir", default=None)
    return parser


def _solve_market(draw, params, cfg, inputs, wf, initial):
    return solve_pooled_ge(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        regime=PricingRegime.MARKET, active_mask=draw.active_mask, initial=initial,
    )


def _solve_fixed_k(draw, params, cfg, inputs, me):
    return solve_fixed_capital_planner(
        draw.alpha, draw.v, params, K_target=me.K_agg,
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        n_firms_cs=inputs.n_firms_cs, K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        regime=PricingRegime.PLANNER, market_solver_kwargs=cfg.market.to_kwargs(),
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask,
        initial=(me.w, me.X_market,
                 euler_R(cfg.parameters.beta, cfg.parameters.delta_K)),
    )


def _corr_av(draw) -> float:
    """Pearson corr(alpha, v) over active firms (pooled); see permute.py."""
    mask = np.asarray(draw.active_mask, dtype=bool)
    a = np.asarray(draw.alpha, dtype=np.float64)[mask]
    v = np.asarray(draw.v, dtype=np.float64)[mask]
    if a.size < 2 or np.std(a) == 0.0 or np.std(v) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, v)[0, 1])


def _log1p_mean_se(values: np.ndarray) -> tuple[float, float]:
    x = np.log1p(np.asarray(values, dtype=np.float64))
    mean = float(np.mean(x)) if x.size else float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
    return mean, se


def _rel_diff(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    beta, phi = cfg.parameters.beta, cfg.parameters.phi

    sorting_dir = (Path(args.sorting_dir) if args.sorting_dir
                   else cfg.out_results_dir / "counterfactuals" / "scalability_sorting")
    with open(sorting_dir / "sorting_objects.yaml") as stream:
        objects = yaml.safe_load(stream)
    econ = pd.read_parquet(sorting_dir / "sorting_economies.parquet")

    M = int(objects["M"])
    seed = int(objects["seed"])
    chi_baseline_stored = float(objects["chi_baseline"])
    reanchored = bool(objects.get("reanchored_yhat_baseline_median", False))
    lambda_K_baseline = float(objects["lambda_K_baseline"])

    base_row = econ[econ["mode"] == "baseline"].iloc[0]
    shuf_rows = econ[econ["mode"] == "shuffle"]
    if shuf_rows.empty:
        raise ValueError(f"no shuffle rows found in {sorting_dir / 'sorting_economies.parquet'}")

    # --- rebuild base_draw identically to run_scalability_sorting.py ---
    support = (inputs.alpha_support if inputs.alpha_support is not None
               else np.linspace(0.6, 1.20, 500))
    base_draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N, M=M,
        H=params.H, rng=seed, v_min=params.v_min,
    )

    me_base = _solve_market(base_draw, params, cfg, inputs, wf, initial)

    if reanchored:
        y_hat_star = float(me_base.y_anchor)
        params = dataclasses.replace(params, y_hat=y_hat_star)
        me_base = _solve_market(base_draw, params, cfg, inputs, wf,
                                (me_base.w, me_base.X_market))

    chi_baseline = float(me_base.chi)

    # --- consistency guard ---
    checks = {
        "C": (float(me_base.C), float(base_row["C"])),
        "K": (float(me_base.K_agg), float(base_row["K"])),
        "R": (float(me_base.R), float(base_row["R"])),
        "y_anchor": (float(me_base.y_anchor), float(base_row["y_anchor"])),
        "corr_av": (_corr_av(base_draw), float(base_row["corr_av"])),
        "chi_baseline": (chi_baseline, chi_baseline_stored),
    }
    consistency = {
        key: {
            "recomputed": actual, "stored": expected,
            "rel_diff": _rel_diff(actual, expected),
        }
        for key, (actual, expected) in checks.items()
    }
    bad = {k: v for k, v in consistency.items() if v["rel_diff"] > 1e-6}
    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.out_results_dir / "counterfactuals" / "homogeneous_rts")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(out_dir / "consistency_check.yaml",
                {"sorting_dir": str(sorting_dir), "checks": consistency})
    if bad:
        raise AssertionError(
            f"recomputed base_draw/me_base does not match {sorting_dir} baseline "
            f"row within 1e-6 rel tol: {bad}. The homogeneous_rts and "
            f"scalability_sorting runs are decoupled -- rerun scalability_sorting "
            f"with matching --calibration/--M/--seed, or check for calibration/"
            f"config drift."
        )

    print(f"consistency guard passed (max rel_diff="
          f"{max(v['rel_diff'] for v in consistency.values()):.2e})")

    # --- homogeneous legs ---
    variant_rows: list[dict] = []
    decomps: dict[str, dict] = {}
    fp_primitive = None
    draw_primitive = None
    for variant in _VARIANTS:
        if variant == "primitive":
            hom_draw = homogenize_alpha_primitive(base_draw)
        else:
            hom_draw = homogenize_alpha_cost_weighted(base_draw, me_base)

        me_hom = _solve_market(hom_draw, params, cfg, inputs, wf,
                               (me_base.w, me_base.X_market))
        if not me_hom.converged:
            me_hom = _solve_market(hom_draw, params, cfg, inputs, wf, initial)
        fp_hom = _solve_fixed_k(hom_draw, params, cfg, inputs, me_hom)

        decomp = fixed_capital_decomposition(
            me_hom, fp_hom, pe=None, chi_me=chi_baseline, beta=beta, phi=phi,
            n_firms_cs=inputs.n_firms_cs,
        )
        decomp_own = fixed_capital_decomposition(
            me_hom, fp_hom, pe=None, chi_me=float(me_hom.chi), beta=beta, phi=phi,
            n_firms_cs=inputs.n_firms_cs,
        )
        decomps[variant] = decomp

        active_mask = np.asarray(hom_draw.active_mask, dtype=bool)
        alpha_active = np.asarray(hom_draw.alpha, dtype=np.float64)[active_mask]
        variant_rows.append({
            "variant": variant,
            "alpha_hom_min": float(np.min(alpha_active)),
            "alpha_hom_max": float(np.max(alpha_active)),
            "alpha_hom_mean": float(np.mean(alpha_active)),
            "lambda_K": float(decomp["lambda_reallocation"]),
            "W_market": float(decomp["W_market"]),
            "W_planner_fixed_k": float(decomp["W_planner_fixed_k"]),
            "mean_inv_alpha": float(np.mean(1.0 / alpha_active)),
            "lambda_K_ownchi": float(decomp_own["lambda_reallocation"]),
            "chi_own": float(me_hom.chi),
            "converged": bool(me_hom.converged and fp_hom.converged),
            "residual": float(fp_hom.residual),
        })
        print(f"[{variant:>13}] alpha_hom in "
              f"[{variant_rows[-1]['alpha_hom_min']:.4g}, "
              f"{variant_rows[-1]['alpha_hom_max']:.4g}]  "
              f"lambda_K={variant_rows[-1]['lambda_K']:+.6g}  "
              f"converged={variant_rows[-1]['converged']}")

        support_lo, support_hi = float(support.min()), float(support.max())
        if not (support_lo <= variant_rows[-1]["alpha_hom_min"] <= support_hi
                and support_lo <= variant_rows[-1]["alpha_hom_max"] <= support_hi):
            warnings.warn(
                f"{variant} alpha_hom falls outside the alpha support "
                f"[{support_lo}, {support_hi}]", RuntimeWarning, stacklevel=2,
            )
        if not variant_rows[-1]["converged"]:
            warnings.warn(f"{variant} leg did not fully converge", RuntimeWarning,
                          stacklevel=2)

        if variant == "primitive":
            fp_primitive, draw_primitive = fp_hom, hom_draw

    variants_df = pd.DataFrame(variant_rows)

    # --- Delta = log(1+lambda) decomposition (primitive variant only) ---
    lambda_K_hom_primitive = float(decomps["primitive"]["lambda_reallocation"])
    lambda_K_hom_cost_weighted = float(decomps["cost_weighted"]["lambda_reallocation"])

    delta_base = float(np.log1p(lambda_K_baseline))
    delta_shuffle, delta_shuffle_se = _log1p_mean_se(shuf_rows["lambda_K"].to_numpy())
    delta_hom = float(np.log1p(lambda_K_hom_primitive))
    delta_hom_cost_weighted = float(np.log1p(lambda_K_hom_cost_weighted))

    markup_common = delta_hom
    dispersion = delta_shuffle - delta_hom
    sorting = delta_base - delta_shuffle
    identity_residual = markup_common + dispersion + sorting - delta_base
    assert abs(identity_residual) < 1e-9, (
        f"markup_common+dispersion+sorting != Delta_base "
        f"(residual={identity_residual:.3e})"
    )

    # --- curvature disclosure ---
    kappa_base = sector_mean_inv_alpha(base_draw)
    kappa_hom_primitive = sector_mean_inv_alpha(draw_primitive)
    hom_cw_draw = homogenize_alpha_cost_weighted(base_draw, me_base)
    kappa_hom_cost_weighted = sector_mean_inv_alpha(hom_cw_draw)
    curvature_disclosure = {
        "mean_inv_alpha_baseline_mean": float(np.nanmean(kappa_base)),
        "mean_inv_alpha_hom_primitive_mean": float(np.nanmean(kappa_hom_primitive)),
        "mean_inv_alpha_hom_cost_weighted_mean": float(np.nanmean(kappa_hom_cost_weighted)),
        "max_abs_diff_baseline_vs_hom_primitive": float(
            np.nanmax(np.abs(kappa_base - kappa_hom_primitive))
        ),
    }

    decomposition = {
        "provenance": {
            "M": M, "seed": seed, "chi_baseline": chi_baseline,
            "y_hat": float(params.y_hat), "sorting_dir": str(sorting_dir),
        },
        "Delta_base": delta_base,
        "Delta_shuffle": delta_shuffle,
        "Delta_shuffle_se": delta_shuffle_se,
        "Delta_hom_primitive": delta_hom,
        "Delta_hom_cost_weighted": delta_hom_cost_weighted,
        "lambda_K_baseline": lambda_K_baseline,
        "lambda_K_hom_primitive": lambda_K_hom_primitive,
        "lambda_K_hom_cost_weighted": lambda_K_hom_cost_weighted,
        "decomposition": {
            "markup_common": markup_common,
            "dispersion": dispersion,
            "sorting": sorting,
            "identity_residual": identity_residual,
        },
        "curvature_disclosure": curvature_disclosure,
        "note_cost_weighted": "robustness variant only (different question: "
                              "holds the aggregate MC elasticity fixed, not "
                              "each sector's mean curvature); not part of the "
                              "markup_common/dispersion/sorting split.",
    }

    _write_yaml(out_dir / "homogeneous_rts_decomposition.yaml", decomposition)
    variants_df.to_parquet(out_dir / "homogeneous_economies.parquet", index=False)
    build_firm_panel(
        fp_primitive, tilde_alpha=draw_primitive.tilde_alpha,
        regime="planner_fixed_k_homogeneous_primitive",
    ).to_parquet(out_dir / "sim_panel_homogeneous_primitive.parquet", index=False)

    print(f"\nDelta_base={delta_base:.6g}  Delta_shuffle={delta_shuffle:.6g} "
          f"(se {delta_shuffle_se:.3g})  Delta_hom={delta_hom:.6g}")
    print(f"markup_common={markup_common:+.6g}  dispersion={dispersion:+.6g}  "
          f"sorting={sorting:+.6g}  (identity_residual={identity_residual:.2e})")
    print(f"\nwrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
