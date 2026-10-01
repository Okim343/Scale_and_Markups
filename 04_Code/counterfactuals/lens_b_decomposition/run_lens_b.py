"""CLI entry point for the Lens-B (dispersion x level) decomposition.

Standalone sibling of
:func:`counterfactuals.scale_channel_decomposition.run_scale_channel.main`. It
runs AFTER the scale-channel run and REUSES that run's cached welfare levels --
the full free-capital planner ``W_planner`` and the market ``W_market`` -- from
``scale_channel_economies.parquet``. The expensive planner leg is therefore NOT
re-solved. The only NEW solves are the 14 UNIFORM legs (``regime=UNIFORM``,
``mu = mu_bar``), from which the DISPERSION leg ``lambda_dispersion`` is obtained;
the LEVEL leg falls out residually on the log scale as
``Delta_level = Delta_total - Delta_dispersion`` (Lens-B multiplicative identity
``(1+lambda_disp)(1+lambda_level) = 1+lambda_total``).

Minimal design (time-boxed)
---------------------------
The market active mask and ``wf_reference`` are *exogenous passthroughs* in
:func:`steady_state.model.normalization.solve_pooled_ge` ("no selection"), so
they are reconstructed for free from the base draw + calibration -- no MARKET
re-solve is needed for the 12 arrangements whose ``mu_bar`` (= cost-weighted
market markup ``mu_cw_market``) is already cached in the parquet. Only:

* the BASELINE market is solved once (to build the cost-weighted homogeneous
  draw ``homogenize_alpha_cost_weighted(base_draw, me_base)`` and to warm the
  UNIFORM chain), and
* the two HOMOGENEOUS-alpha markets are solved (their ``mu_cw_market`` is NaN in
  the cached parquet, so ``mu_bar`` must be recomputed).

=> ~3 MARKET solves + 14 UNIFORM solves, 0 PLANNER solves.

Everything is anchored on the *cached* ``W_market``/``W_planner`` (both on the
frozen ``chi_baseline`` from provenance), and the UNIFORM economy is a
deterministic fixpoint on the bit-identical reconstructed draw, so both Lens-B
identities close to machine precision with NO cross-run drift guard. ``M``/
``seed``/``shuffle_seed`` are inherited verbatim from the scale-channel
provenance (no ``--M``/``--seed`` flags): the base draw must be bit-identical to
the cached run for the cache reuse to be valid.

Run from ``04_Code/``, after ``scale_channel_decomposition`` has produced its
outputs::

    python -m counterfactuals.lens_b_decomposition.run_lens_b \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml
"""

from __future__ import annotations

import argparse
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
from steady_state.model.normalization import solve_pooled_ge
from steady_state.model.pool import draw_pool
from steady_state.model.pricing import PricingRegime
from steady_state.welfare.welfare_metrics import (
    aggregate_markup_cw,
    consumption_equivalent_lambda,
    steady_state_welfare,
)

from counterfactuals.homogeneous_rts.homogenize import (
    homogenize_alpha_cost_weighted,
    homogenize_alpha_primitive,
)
from counterfactuals.scalability_sorting.permute import (
    active_corr_av,
    permute_alpha,
)

from .decomposition import (
    identity_residual,
    per_draw_level_delta,
    three_channel_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.lens_b_decomposition.run_lens_b",
        description="Lens-B decomposition: put the dispersion x level split of "
                    "the market->planner welfare gain on the alpha-arrangement "
                    "axis, reusing the cached scale-channel planner/market "
                    "welfare (no planner re-solve).",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--calibration",
        default="out_results/calib_pooled/calibration_pooled.yaml",
        help="calibration YAML (must match the scale-channel run's).",
    )
    parser.add_argument(
        "--scale-channel-dir", default=None, dest="scale_channel_dir",
        help="directory holding scale_channel_economies.parquet and "
             "scale_channel_decomposition.yaml (default "
             "out_results/counterfactuals/scale_channel_decomposition).",
    )
    parser.add_argument("--out-dir", default=None)
    return parser


# --- solver legs --------------------------------------------------------------

def _solve_market(draw, params, cfg, inputs, wf, initial):
    """MARKET leg -- baseline (cost-weighted hom draw + warm anchor) and the two
    homogeneous arrangements (their mu_bar is NaN in the cached parquet)."""
    return solve_pooled_ge(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        regime=PricingRegime.MARKET, active_mask=draw.active_mask, initial=initial,
    )


def _solve_uniform(draw, params, cfg, inputs, *, mu_bar, wf, initial):
    """The single new leg per arrangement: UNIFORM markup mu ≡ mu_bar, free
    capital, on THIS arrangement's exogenous mask.

    ``active_mask`` and ``wf_reference`` are exogenous passthroughs
    (normalization.solve_pooled_ge: "no selection"), reconstructed from the draw
    and calibration -- identical to what the market/planner legs used in the
    scale-channel run, so W_uniform is on the same economy as the cached
    W_market/W_planner. Warm-started from ``initial`` (the chained previous
    uniform, or the baseline/hom market point).
    """
    return solve_pooled_ge(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(),
        regime=PricingRegime.UNIFORM, mu_bar=float(mu_bar),
        wf_reference=wf, active_mask=draw.active_mask, initial=initial,
    )


# --- provenance + cache readers ----------------------------------------------

def _load_provenance(scale_channel_yaml: Path) -> dict:
    """Inherit ``M``/``seed``/``shuffle_seed``/``chi_baseline``/... verbatim from
    the scale-channel run's ``provenance`` block."""
    with open(scale_channel_yaml) as stream:
        doc = yaml.safe_load(stream)
    prov = doc["provenance"]
    return {
        "M": int(prov["M"]),
        "seed": int(prov["seed"]),
        "shuffle_seed": int(prov["shuffle_seed"]),
        "n_shuffle": int(prov["n_shuffle"]),
        "chi_baseline": float(prov["chi_baseline"]),
        "y_hat": float(prov["y_hat"]),
        "beta": float(prov["beta"]),
        "phi": float(prov["phi"]),
        "design": str(prov.get("design", "")),
    }


def _read_scale_channel_cache(parquet: Path) -> dict:
    """Read the cached scale-channel welfare, keyed by ``(label, perm_seed)``.

    Provides ``W_market`` (cached), ``W_planner`` (the full free-capital planner
    -- reused, never re-solved here), the cached ``lambda_total``/``delta_total``
    (cross-check), and ``mu_cw_market`` (the UNIFORM target ``mu_bar``; NaN for
    the two homogeneous arrangements, which are recomputed).
    """
    df = pd.read_parquet(parquet)
    cache: dict = {}
    for _, r in df.iterrows():
        label = str(r["label"])
        ps = r["perm_seed"]
        key = ((label, int(ps)) if label == "shuffle" and not pd.isna(ps)
               else (label, None))
        cache[key] = {
            "W_market": float(r["W_market_cached"]),
            "W_planner": float(r["W_planner"]),
            "lambda_total_cached": float(r["lambda_total"]),
            "delta_total_cached": float(r["delta_total"]),
            "mu_cw_market": float(r["mu_cw_market"]),  # may be NaN (hom rows)
            "corr_av_cached": float(r["corr_av"]),
        }
    return cache


# --- arrangement family (reproduce the cached shuffle-seed sequence) ---------

def _arrangement_family(base_draw, me_base, shuffle_seed, n_shuffle):
    """Yield ``(label, perm_seed, draw)`` for the 14 arrangements.

    The shuffle perm_seed sequence and per-shuffle RNG are reproduced EXACTLY as
    in ``run_scalability_sorting`` / ``run_scale_channel`` (``perm_seed =
    shuffle_seed + i``; RNG seeded with ``[shuffle_seed, i]``) so the shuffle
    rows align 1:1 with the cached parquet for the per-draw ``Delta_level``
    differencing.
    """
    yield "baseline", None, permute_alpha(base_draw, "identity")
    for i in range(int(n_shuffle)):
        perm_seed = int(shuffle_seed) + i
        rng = np.random.default_rng([int(shuffle_seed), i])
        yield "shuffle", perm_seed, permute_alpha(base_draw, "shuffle", rng)
    yield "reverse", None, permute_alpha(base_draw, "reverse")
    yield "hom_primitive", None, homogenize_alpha_primitive(base_draw)
    yield ("hom_cost_weighted", None,
           homogenize_alpha_cost_weighted(base_draw, me_base))


# --- per-arrangement row ------------------------------------------------------

def _rel_diff(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _economy_row(label, perm_seed, draw, ue, cached, *, chi_baseline, beta, phi,
                 n_firms_cs, mu_bar) -> dict:
    """One row of ``lens_b_economies.parquet``.

    ``lambda_dispersion`` and ``lambda_total`` are anchored on the CACHED
    ``W_market``; ``lambda_level`` on the cached ``W_planner``. With
    ``W_uniform`` the new UNIFORM solve (same frozen ``chi_baseline``), the three
    exponentials telescope and the log identity ``Delta_level = Delta_total -
    Delta_dispersion`` closes to machine precision.
    """
    W_market = cached["W_market"]
    W_planner = cached["W_planner"]
    W_uniform = steady_state_welfare(ue.C, ue.L_agg, chi_baseline, beta, phi)

    lambda_total = consumption_equivalent_lambda(W_market, W_planner, beta)
    lambda_disp = consumption_equivalent_lambda(W_market, W_uniform, beta)
    lambda_level = consumption_equivalent_lambda(W_uniform, W_planner, beta)

    delta_total = float(np.log1p(lambda_total))
    delta_disp = float(np.log1p(lambda_disp))
    delta_level = delta_total - delta_disp  # canonical residual (Lens-B identity)

    return {
        "label": label,
        "perm_seed": (float(perm_seed) if perm_seed is not None else float("nan")),
        "corr_av": active_corr_av(draw),
        "W_market_cached": float(W_market),
        "W_uniform": float(W_uniform),
        "W_planner_cached": float(W_planner),
        "mu_bar": float(mu_bar),
        "mu_cw_uniform": float(aggregate_markup_cw(ue, n_firms_cs=n_firms_cs)),
        "lambda_total": float(lambda_total),
        "lambda_dispersion": float(lambda_disp),
        "lambda_level": float(lambda_level),
        "delta_total": delta_total,
        "delta_dispersion": delta_disp,
        "delta_level": float(delta_level),
        # cross-check the recomputed total against the cached one
        "lambda_total_cached": float(cached["lambda_total_cached"]),
        "delta_total_cached_gap": float(delta_total - cached["delta_total_cached"]),
        # termwise-vs-direct closure of Delta_level (acceptance check)
        "delta_level_termwise_gap": float(delta_level - np.log1p(lambda_level)),
        "identity_residual": identity_residual(lambda_disp, lambda_level, lambda_total),
        "uniform_converged": bool(ue.converged),
    }


# --- 3x3 assembly -------------------------------------------------------------

def _log1p_mean_se(values: np.ndarray) -> tuple[float, float]:
    """Mean +/- SE on the ``log1p`` scale (matches homogeneous_rts / scale_channel)."""
    x = np.log1p(np.asarray(values, dtype=np.float64))
    mean = float(np.mean(x)) if x.size else float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
    return mean, se


def _channel_block(split: dict, delta_base: float, *, se: float | None) -> dict:
    """One 3x3 row: values, percentage shares of delta_base, SEs on the
    shuffle-derived (heterogeneity, sorting) terms."""
    def share(x: float) -> float:
        return float(100.0 * x / delta_base) if abs(delta_base) > 1e-30 else float("nan")
    block = {
        "common_alpha": {"value": split["common_alpha"],
                         "share_pct": share(split["common_alpha"])},
        "heterogeneity": {"value": split["heterogeneity"],
                          "share_pct": share(split["heterogeneity"])},
        "sorting": {"value": split["sorting"],
                    "share_pct": share(split["sorting"])},
        "delta_base": float(delta_base),
    }
    if se is not None:
        block["heterogeneity"]["se"] = float(se)
        block["sorting"]["se"] = float(se)
    return block


def _assemble_3x3(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)

    def one(label: str):
        sub = df[df["label"] == label]
        return sub.iloc[0] if not sub.empty else None

    base = one("baseline")
    rev = one("reverse")
    hom_p = one("hom_primitive")
    hom_cw = one("hom_cost_weighted")
    shuf = df[df["label"] == "shuffle"].sort_values("perm_seed")

    lam_tot_shuf = shuf["lambda_total"].to_numpy()
    lam_disp_shuf = shuf["lambda_dispersion"].to_numpy()

    # --- Delta_total row (cached; reproduces the scale-channel delta_total) ---
    dtot_base = float(base["delta_total"])
    dtot_shuf_mean, dtot_shuf_se = _log1p_mean_se(lam_tot_shuf)
    dtot_hom = float(hom_p["delta_total"])
    total_split = three_channel_split(dtot_base, dtot_shuf_mean, dtot_hom)

    # --- Delta_dispersion row (the only genuinely new estimand) ---
    dd_base = float(base["delta_dispersion"])
    dd_shuf_mean, dd_shuf_se = _log1p_mean_se(lam_disp_shuf)
    dd_hom = float(hom_p["delta_dispersion"])
    disp_split = three_channel_split(dd_base, dd_shuf_mean, dd_hom)

    # --- Delta_level row: termwise (Delta_total - Delta_dispersion) ---
    level_split = {ch: total_split[ch] - disp_split[ch] for ch in total_split}
    _, dlevel_shuf_se = per_draw_level_delta(
        np.log1p(lam_tot_shuf), np.log1p(lam_disp_shuf)
    )
    dlevel_base = dtot_base - dd_base

    # termwise-vs-per-draw closure of the shuffle-derived level terms
    dlevel_shuf_mean = float(np.mean(np.log1p(lam_tot_shuf) - np.log1p(lam_disp_shuf)))
    dlevel_hom = dtot_hom - dd_hom
    level_split_perdraw = three_channel_split(dlevel_base, dlevel_shuf_mean, dlevel_hom)
    termwise_gap = max(abs(level_split[ch] - level_split_perdraw[ch])
                       for ch in level_split)

    three_by_three = {
        "delta_total": _channel_block(total_split, dtot_base, se=dtot_shuf_se),
        "delta_dispersion": _channel_block(disp_split, dd_base, se=dd_shuf_se),
        "delta_level": _channel_block(level_split, dlevel_base, se=dlevel_shuf_se),
        "delta_level_termwise_gap": float(termwise_gap),
    }

    def arrangement(row):
        if row is None:
            return None
        return {
            "lambda_total": float(row["lambda_total"]),
            "lambda_dispersion": float(row["lambda_dispersion"]),
            "lambda_level": float(row["lambda_level"]),
            "mu_bar": float(row["mu_bar"]),
        }

    per_arrangement = {
        "baseline": arrangement(base),
        "reverse": arrangement(rev),
        "hom_primitive": arrangement(hom_p),
        "hom_cost_weighted": arrangement(hom_cw),
        "shuffle_mean": {
            "lambda_total": float(np.mean(lam_tot_shuf)),
            "lambda_total_se": (float(np.std(lam_tot_shuf, ddof=1) / np.sqrt(lam_tot_shuf.size))
                                if lam_tot_shuf.size > 1 else float("nan")),
            "lambda_dispersion": float(np.mean(lam_disp_shuf)),
            "lambda_dispersion_se": (float(np.std(lam_disp_shuf, ddof=1) / np.sqrt(lam_disp_shuf.size))
                                     if lam_disp_shuf.size > 1 else float("nan")),
            "lambda_level_mean_perdraw": float(np.mean(
                np.expm1(np.log1p(lam_tot_shuf) - np.log1p(lam_disp_shuf)))),
            "delta_level_se_perdraw": dlevel_shuf_se,
            "n_shuffle": int(shuf.shape[0]),
        },
    }

    # robustness: primitive vs cost-weighted common-alpha DISPERSION term
    common_alpha_disp_primitive = disp_split["common_alpha"]  # uses hom_primitive
    common_alpha_disp_cost_weighted = (
        float(hom_cw["delta_dispersion"]) if hom_cw is not None else float("nan")
    )
    robustness = {
        "common_alpha_dispersion_primitive": float(common_alpha_disp_primitive),
        "common_alpha_dispersion_cost_weighted": float(common_alpha_disp_cost_weighted),
        "note": "If the dispersion common-alpha term is stable across the "
                "primitive (per-sector harmonic-mean) and cost-weighted "
                "(aggregate-elasticity-preserving) hom constructions, the "
                "curvature-level confound is not driving it. Expected milder "
                "than the scale case (dispersion removal drives no K expansion).",
    }

    return {
        "three_by_three": three_by_three,
        "per_arrangement": per_arrangement,
        "robustness": robustness,
        "diagnostics": {
            "identity_residual_max": float(df["identity_residual"].abs().max()),
            "delta_level_termwise_gap_max": float(df["delta_level_termwise_gap"].abs().max()),
            "delta_total_cached_gap_max": float(df["delta_total_cached_gap"].abs().max()),
            "mu_cw_uniform_max_abs_dev_from_target": float(
                (df["mu_cw_uniform"] - df["mu_bar"]).abs().max()),
            "all_uniform_converged": bool(df["uniform_converged"].all()),
        },
    }


# --- orchestration ------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    if args.calibration is None:
        print("warning: no --calibration given; using config initial params.")
    beta, phi = cfg.parameters.beta, cfg.parameters.phi

    scale_channel_dir = (
        Path(args.scale_channel_dir) if args.scale_channel_dir
        else cfg.out_results_dir / "counterfactuals" / "scale_channel_decomposition")
    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.out_results_dir / "counterfactuals" / "lens_b_decomposition")

    prov = _load_provenance(scale_channel_dir / "scale_channel_decomposition.yaml")
    cache = _read_scale_channel_cache(scale_channel_dir / "scale_channel_economies.parquet")

    # base draw bit-identical to the scale-channel run (verbatim M/seed)
    M, seed = prov["M"], prov["seed"]
    chi_baseline = prov["chi_baseline"]

    # invariant: frozen calibrated y_hat (never re-anchored)
    assert float(params.y_hat) == prov["y_hat"], (
        f"params.y_hat={float(params.y_hat)} != cached y_hat={prov['y_hat']}; "
        "calibration drift would break the cache-reuse comparability."
    )

    support = (inputs.alpha_support if inputs.alpha_support is not None
               else np.linspace(0.6, 1.20, 500))
    base_draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N, M=M,
        H=params.H, rng=seed, v_min=params.v_min,
    )

    # baseline MARKET (needed for the cost-weighted hom draw + UNIFORM warm anchor)
    me_base = _solve_market(base_draw, params, cfg, inputs, wf, initial)
    chi_drift = _rel_diff(float(me_base.chi), chi_baseline)
    print(f"baseline chi={float(me_base.chi):.6g} (cached {chi_baseline:.6g}, "
          f"rel_diff={chi_drift:.2e})  K_base={me_base.K_agg:.6g}  "
          f"y_hat={float(params.y_hat):.6g}  "
          f"corr_av={active_corr_av(base_draw):+.4f}  M={M}")
    if chi_drift > 1e-4:
        print(f"WARNING: recomputed baseline chi differs from the cached "
              f"chi_baseline by {chi_drift:.2e}; welfare uses the cached "
              f"chi_baseline for consistency with W_market/W_planner. This "
              f"suggests calibration/config drift vs the scale-channel run.")

    warm = (me_base.w, me_base.X_market)  # chained UNIFORM warm start

    rows: list[dict] = []
    for label, perm_seed, draw in _arrangement_family(
        base_draw, me_base, prov["shuffle_seed"], prov["n_shuffle"]
    ):
        key = (label, perm_seed if label == "shuffle" else None)
        if key not in cache:
            raise KeyError(
                f"arrangement {key} not found in scale_channel_economies.parquet; "
                "the Lens-B run and the cached scale-channel run are decoupled "
                "(check --calibration / provenance / shuffle_seed alignment)."
            )
        cached = cache[key]

        mu_bar = cached["mu_cw_market"]
        uniform_warm = warm
        if np.isnan(mu_bar):  # hom arrangements: mu_bar not cached -> recompute
            me_hom = _solve_market(draw, params, cfg, inputs, wf, warm)
            if not me_hom.converged:
                me_hom = _solve_market(draw, params, cfg, inputs, wf, initial)
            mu_bar = aggregate_markup_cw(me_hom, n_firms_cs=inputs.n_firms_cs)
            uniform_warm = (me_hom.w, me_hom.X_market)

        ue = _solve_uniform(draw, params, cfg, inputs,
                            mu_bar=mu_bar, wf=wf, initial=uniform_warm)
        if not ue.converged:  # retry from the baseline market point, then cold
            ue = _solve_uniform(draw, params, cfg, inputs,
                                mu_bar=mu_bar, wf=wf, initial=warm)
            if not ue.converged:
                ue = _solve_uniform(draw, params, cfg, inputs,
                                    mu_bar=mu_bar, wf=wf, initial=(1.0, 1.0))
        if ue.converged:
            warm = (ue.w, ue.X_market)  # advance the chain only on success

        row = _economy_row(label, perm_seed, draw, ue, cached,
                           chi_baseline=chi_baseline, beta=beta, phi=phi,
                           n_firms_cs=inputs.n_firms_cs, mu_bar=mu_bar)
        rows.append(row)
        print(f"[{label:>17}] corr_av={row['corr_av']:+.4f}  "
              f"mu_bar={row['mu_bar']:.6g}  "
              f"lam_tot={row['lambda_total']:+.6g}  "
              f"lam_disp={row['lambda_dispersion']:+.6g}  "
              f"lam_level={row['lambda_level']:+.6g}  "
              f"id_res={row['identity_residual']:+.1e}  "
              f"dtot_gap={row['delta_total_cached_gap']:+.1e}  "
              f"conv={row['uniform_converged']}")

    econ = pd.DataFrame(rows)
    assembled = _assemble_3x3(rows)

    decomposition = {
        "provenance": {
            "M": M, "seed": seed, "shuffle_seed": prov["shuffle_seed"],
            "n_shuffle": prov["n_shuffle"], "chi_baseline": chi_baseline,
            "y_hat": float(params.y_hat), "beta": float(beta), "phi": float(phi),
            "design": "reuse_cached_planner_minimal",
            "source_parquet": str(scale_channel_dir / "scale_channel_economies.parquet"),
            "note": "UNIFORM legs solved per arrangement; W_market/W_planner "
                    "(and thus lambda_total) reused from the scale-channel run "
                    "-- planner NOT re-solved. Delta_level = Delta_total - "
                    "Delta_dispersion (residual).",
        },
        **assembled,
        "caveats": [
            "The DISPERSION leg is expected to be genuinely sorting-sensitive "
            "(sorting reshapes which firms bear the wedge dispersion) -- unlike "
            "the scale-channel's a-priori orthogonality hypothesis. A non-trivial "
            "sorting share in Delta_dispersion means the cost of dispersion "
            "depends on WHICH technologies carry the wedges.",
            "Homogeneous-alpha confound: the common-alpha column inherits the "
            "curvature-level confound; see the robustness block (primitive vs "
            "cost-weighted). Expected milder than the scale case.",
            "Untargeted mu-dispersion frontier: absolute lambda levels ride the "
            "uncalibrated markup tail, which bites the DISPERSION leg most "
            "directly; the relative 3x3 under common random numbers is the clean "
            "object.",
            "Minimal design: mask/wf_reference reconstructed from the draw "
            "(exogenous passthroughs); only the baseline + 2 homogeneous markets "
            "are solved. No cross-run drift guard -- W_market/W_planner are read "
            "from cache and the UNIFORM economy is deterministic on the "
            "bit-identical reconstructed draw.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    econ.to_parquet(out_dir / "lens_b_economies.parquet", index=False)
    _write_yaml(out_dir / "lens_b_decomposition.yaml", decomposition)

    tbt = assembled["three_by_three"]
    diag = assembled["diagnostics"]
    print("\n--- 3x3 (Delta = log(1+lambda)) ---")
    for row_name in ("delta_total", "delta_dispersion", "delta_level"):
        blk = tbt[row_name]
        print(f"{row_name:>16}: common_alpha={blk['common_alpha']['value']:+.6g} "
              f"({blk['common_alpha']['share_pct']:.1f}%)  "
              f"heterogeneity={blk['heterogeneity']['value']:+.6g} "
              f"({blk['heterogeneity']['share_pct']:.1f}%)  "
              f"sorting={blk['sorting']['value']:+.6g} "
              f"({blk['sorting']['share_pct']:.1f}%)")
    print(f"\nDelta_dispersion SORTING term = "
          f"{tbt['delta_dispersion']['sorting']['value']:+.6g} "
          f"(se {tbt['delta_dispersion']['sorting'].get('se', float('nan')):.3g})")
    print(f"identity_residual_max={diag['identity_residual_max']:.2e}  "
          f"delta_level_termwise_gap_max={diag['delta_level_termwise_gap_max']:.2e}  "
          f"delta_total_cached_gap_max={diag['delta_total_cached_gap_max']:.2e}  "
          f"mu_uniform_dev_max={diag['mu_cw_uniform_max_abs_dev_from_target']:.2e}")
    print(f"\nwrote outputs to {out_dir}")
    if not diag["all_uniform_converged"]:
        print("WARNING: at least one UNIFORM leg did not fully converge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
