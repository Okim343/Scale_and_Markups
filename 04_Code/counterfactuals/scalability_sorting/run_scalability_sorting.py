"""CLI entry point for the scalability-sorting counterfactual.

Thin wrapper mirroring
:func:`counterfactuals.fixed_input_welfare.run_fixed_capital_planner.main`. It
builds one calibrated base :class:`~steady_state.model.pool.PoolDraw`, then loops
over a family of marginal-preserving permutations of the alpha<->v assignment
(``identity`` baseline + ``n_shuffle`` random shuffles + one ``reverse``). For
each economy it re-solves the MARKET leg, pins aggregate capital to that
economy's own realized ``K``, solves the fixed-capital-envelope PLANNER, and
runs the *unchanged* ``fixed_capital_decomposition`` with ``chi`` held at the
**baseline MARKET** value for every economy (plan sec. 4.1 -- the governing
convention). The free-capital PLANNER leg is skipped (Objects A/B/C need only
MARKET + FIXED-K).

Everything under ``steady_state/`` and both ``fixed_input_welfare`` modules are
imported unchanged; this package only adds ``permute_alpha``.

Run from ``04_Code/``::

    python -m counterfactuals.scalability_sorting.run_scalability_sorting \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml \\
      --M 10000 --n-shuffle 10
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

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

from .permute import active_corr_av, permute_alpha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.scalability_sorting.run_scalability_sorting",
        description="Scalability-sorting counterfactual: marginal-preserving "
                    "permutations of the alpha<->v assignment, each run through "
                    "the fixed-capital-envelope planner pipeline.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--calibration", default=None,
        help="calibration YAML; required in practice "
             "(e.g. out_results/calib_pooled/calibration_pooled.yaml).",
    )
    parser.add_argument("--M", type=int)
    parser.add_argument("--seed", type=int, help="base-draw seed")
    parser.add_argument("--n-shuffle", type=int, default=10)
    parser.add_argument(
        "--shuffle-seed", type=int, default=20260709,
        help="base seed for the permutation RNGs (shuffle economy i uses "
             "[shuffle_seed, i]).",
    )
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--modes", default="baseline,shuffle,reverse",
        help="comma-separated subset of baseline,shuffle,reverse (default all).",
    )
    parser.add_argument(
        "--reanchor-yhat-baseline-median", action="store_true",
        help="re-anchor the scalability technology at the typical operating "
             "scale of the baseline MARKET economy: set y_hat = median active "
             "output of the baseline market (its y_anchor), then hold that same "
             "y_hat fixed for every economy/regime. Robustness check on the "
             "reference scale that pins where alpha stops being a size penalty "
             "(MC ~ (y/y_hat)^(1/alpha-1)).",
    )
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


def _economy_row(mode, perm_seed, draw, me, fp, decomp, decomp_own, n_firms_cs):
    """One row of sorting_economies.parquet plus the chi-consistency cross-check."""
    return {
        "mode": mode,
        "perm_seed": perm_seed,
        "corr_av": active_corr_av(draw),
        "C": float(me.C), "L": float(me.L_agg), "K": float(me.K_agg),
        "R": float(me.R),
        "y_anchor": float(me.y_anchor), "y_sw": float(me.y_sw),
        "W_market": decomp["W_market"],
        "W_planner_fixed_k": decomp["W_planner_fixed_k"],
        "lambda_K": decomp["lambda_reallocation"],
        "K_target_gap": decomp["K_target_gap"],
        "mu_cw_planner_fixed_k": decomp["mu_cw_planner_fixed_k"],
        "mu_cw_market": float(aggregate_markup_cw(me, n_firms_cs=n_firms_cs)),
        # own-chi cross-check (plan sec. 4.1): lambda_K must match `lambda_K`
        # to solver tolerance (chi cancels within an economy), while W levels
        # differ from the baseline-chi ones for non-baseline economies.
        "lambda_K_ownchi": decomp_own["lambda_reallocation"],
        "W_market_ownchi": decomp_own["W_market"],
        "W_planner_fixed_k_ownchi": decomp_own["W_planner_fixed_k"],
        "chi_own": float(me.chi),
        "market_converged": bool(me.converged),
        "fixed_k_converged": bool(fp.converged),
        "fixed_k_residual": float(fp.residual),
    }


def _ce_stats(base_val, other_vals, beta):
    """Mean +/- SE of CE(base, other) across the shuffle economies."""
    ce = np.array(
        [consumption_equivalent_lambda(base_val, o, beta) for o in other_vals],
        dtype=np.float64,
    )
    mean = float(np.mean(ce)) if ce.size else float("nan")
    se = float(np.std(ce, ddof=1) / np.sqrt(ce.size)) if ce.size > 1 else float("nan")
    return mean, se, ce


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    if args.calibration is None:
        print("warning: no --calibration given; using config initial params.")

    M = int(args.M or cfg.monte_carlo.M_final)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    beta, phi = cfg.parameters.beta, cfg.parameters.phi

    support = (inputs.alpha_support if inputs.alpha_support is not None
               else np.linspace(0.6, 1.20, 500))
    base_draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N, M=M,
        H=params.H, rng=seed, v_min=params.v_min,
    )

    # --- baseline MARKET (cold start from calibration warm point) -> chi_baseline
    me_base = _solve_market(base_draw, params, cfg, inputs, wf, initial)
    y_hat_used = float(params.y_hat)

    # Optional reference-scale re-anchoring (plan robustness check): pin the
    # scalability technology's anchor to the baseline market's typical operating
    # scale (its median active output = y_anchor), then hold that SAME y_hat for
    # every economy/regime. y_hat feeds every firm's MC via
    # (y/y_hat)^(1/alpha-1), so this genuinely re-prices; we re-solve the
    # baseline market under the new anchor so chi_baseline is consistent.
    if args.reanchor_yhat_baseline_median:
        y_hat_star = float(me_base.y_anchor)
        print(f"re-anchor: calibrated y_hat={params.y_hat:.6g} -> "
              f"baseline-market median y_anchor={y_hat_star:.6g} "
              f"(sales-weighted mean y_sw={me_base.y_sw:.6g} for contrast)")
        params = dataclasses.replace(params, y_hat=y_hat_star)
        y_hat_used = y_hat_star
        me_base = _solve_market(base_draw, params, cfg, inputs, wf,
                                (me_base.w, me_base.X_market))

    chi_baseline = float(me_base.chi)
    warm = (me_base.w, me_base.X_market)
    print(f"baseline chi={chi_baseline:.6g}  K_base={me_base.K_agg:.6g}  "
          f"y_hat={y_hat_used:.6g}  corr_av={active_corr_av(base_draw):.4f}")

    # --- assemble the economy family ---
    plan: list[tuple[str, int | None]] = []
    if "baseline" in modes:
        plan.append(("baseline", None))
    if "shuffle" in modes:
        plan.extend(("shuffle", i) for i in range(args.n_shuffle))
    if "reverse" in modes:
        plan.append(("reverse", None))

    rows: list[dict] = []
    panels: dict[str, tuple] = {}  # mode/key -> (fp, draw) for panel export
    for mode, idx in plan:
        if mode == "baseline":
            draw, me, perm_seed = base_draw, me_base, None
        else:
            perm_seed = None if mode == "reverse" else int(args.shuffle_seed) + idx
            rng = None if mode == "reverse" else np.random.default_rng(
                [int(args.shuffle_seed), idx]
            )
            draw = permute_alpha(base_draw, mode, rng)
            me = _solve_market(draw, params, cfg, inputs, wf, warm)
            if not me.converged:  # reverse is the poorest warm start; retry cold
                me = _solve_market(draw, params, cfg, inputs, wf, initial)

        fp = _solve_fixed_k(draw, params, cfg, inputs, me)

        decomp = fixed_capital_decomposition(
            me, fp, pe=None, chi_me=chi_baseline, beta=beta, phi=phi,
            n_firms_cs=inputs.n_firms_cs,
        )
        decomp_own = fixed_capital_decomposition(
            me, fp, pe=None, chi_me=float(me.chi), beta=beta, phi=phi,
            n_firms_cs=inputs.n_firms_cs,
        )
        rows.append(_economy_row(mode, perm_seed, draw, me, fp, decomp,
                                  decomp_own, inputs.n_firms_cs))
        label = mode if mode != "shuffle" else f"shuffle{idx}"
        print(f"[{label:>10}] corr_av={rows[-1]['corr_av']:+.4f}  "
              f"K_gap={rows[-1]['K_target_gap']:+.2e}  "
              f"mu_cw_fk={rows[-1]['mu_cw_planner_fixed_k']:.6g}  "
              f"lambda_K={rows[-1]['lambda_K']:+.6g}  "
              f"mkt_conv={rows[-1]['market_converged']} "
              f"fk_conv={rows[-1]['fixed_k_converged']}")
        # keep panels for baseline, reverse, and the first shuffle
        if mode in ("baseline", "reverse") or (mode == "shuffle" and idx == 0):
            panels[label if mode == "shuffle" else mode] = (fp, draw)

    econ = pd.DataFrame(rows)

    # --- Objects A / B / C (plan sec. 4) ---
    base = econ[econ["mode"] == "baseline"]
    shuf = econ[econ["mode"] == "shuffle"]
    rev = econ[econ["mode"] == "reverse"]

    objects: dict = {"M": M, "seed": seed, "shuffle_seed": int(args.shuffle_seed),
                     "n_shuffle": int(len(shuf)), "chi_baseline": chi_baseline,
                     "y_hat": y_hat_used,
                     "reanchored_yhat_baseline_median":
                         bool(args.reanchor_yhat_baseline_median),
                     "modes": modes}

    if not base.empty and not shuf.empty:
        W_mkt_base = float(base["W_market"].iloc[0])
        W_fkp_base = float(base["W_planner_fixed_k"].iloc[0])
        lam_base = float(base["lambda_K"].iloc[0])

        a_mean, a_se, _ = _ce_stats(W_mkt_base, shuf["W_market"].to_numpy(), beta)
        b_mean, b_se, _ = _ce_stats(W_fkp_base, shuf["W_planner_fixed_k"].to_numpy(), beta)
        lam_shuf = shuf["lambda_K"].to_numpy()
        lam_shuf_mean = float(np.mean(lam_shuf))
        lam_shuf_se = (float(np.std(lam_shuf, ddof=1) / np.sqrt(lam_shuf.size))
                       if lam_shuf.size > 1 else float("nan"))

        objects.update({
            "object_A_market_sorting_CE_mean": a_mean,
            "object_A_market_sorting_CE_se": a_se,
            "object_B_planner_sorting_CE_mean": b_mean,
            "object_B_planner_sorting_CE_se": b_se,
            "lambda_K_baseline": lam_base,
            "lambda_K_shuffle_mean": lam_shuf_mean,
            "lambda_K_shuffle_se": lam_shuf_se,
            # Object C (headline): how much larger the markup cost is because
            # alpha is positively sorted with z.
            "object_C_markup_sorting_premium": lam_base - lam_shuf_mean,
            "object_C_se": lam_shuf_se,
        })
    if not rev.empty:
        objects["lambda_K_reverse"] = float(rev["lambda_K"].iloc[0])
        if not base.empty:
            objects["object_C_reverse_bound"] = (
                float(base["lambda_K"].iloc[0]) - float(rev["lambda_K"].iloc[0])
            )

    # --- write artifacts ---
    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.out_results_dir / "counterfactuals" / "scalability_sorting")
    out_dir.mkdir(parents=True, exist_ok=True)

    econ.to_parquet(out_dir / "sorting_economies.parquet", index=False)
    _write_yaml(out_dir / "sorting_objects.yaml", objects)
    for key, (fp, draw) in panels.items():
        build_firm_panel(
            fp, tilde_alpha=draw.tilde_alpha, regime=f"planner_fixed_k_{key}",
        ).to_parquet(out_dir / f"sim_panel_{key}.parquet", index=False)

    print(f"\nwrote outputs to {out_dir}")
    if "object_C_markup_sorting_premium" in objects:
        print(f"Object C (markup-sorting premium) = "
              f"{objects['object_C_markup_sorting_premium']:+.6g} "
              f"(SE {objects['object_C_se']:.3g})")
    all_conv = bool(econ["market_converged"].all() and econ["fixed_k_converged"].all())
    if not all_conv:
        print("WARNING: at least one leg did not fully converge; "
              "check fixed_k_residual / consider bounding log R (see README).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
