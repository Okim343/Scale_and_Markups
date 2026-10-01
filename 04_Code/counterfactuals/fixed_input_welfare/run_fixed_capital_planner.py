"""CLI entry point for the fixed-capital-envelope planner counterfactual.

Mirrors :func:`steady_state.__main__.cmd_welfare`'s call graph, reusing the
following **unchanged** (imported directly, nothing under ``steady_state/`` is
modified): ``load_config``, ``_pooled_inputs``, ``_params_for_command``,
``_write_yaml`` (from ``steady_state.__main__``), ``draw_pool``,
``solve_pooled_ge``/``euler_R`` (from ``steady_state.model.normalization``),
``build_firm_panel`` (from ``steady_state.simulation.cross_section``).

The only new logic: solve MARKET -> read ``K_target = me.K_agg`` -> call
:func:`solve_fixed_capital_planner` warm-started from
``(me.w, me.X_market, euler_R(...))`` on MARKET's active mask / wf_reference
(the same warm-start pattern ``solve_market_and_planner`` uses for PLANNER) ->
optionally solve the full PLANNER too -> build the table/decomposition -> write
outputs.

Why re-run MARKET from scratch rather than reusing ``sim_panel_market.parquet``:
that panel carries no ``w``/``R``/``X_market``/``wf_reference`` to warm-start the
new 3-unknown solve, so reconstructing state from it would mean re-deriving
those from the calibration YAML anyway -- at which point re-solving is simpler
and safer, and decouples this script from any prior run's on-disk artifacts.
Cost is ~4-5 min extra at M=10000, and buys full reproducibility. As a cheap
cross-check the script writes ``market_recompute_check.yaml`` (its freshly
solved MARKET's ``C, L, K, R, mu_cw``) to eyeball-diff against the production
``welfare_decomposition.parquet`` market row.

Run from ``04_Code/``::

    python -m counterfactuals.fixed_input_welfare.run_fixed_capital_planner \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml --M 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
from steady_state.welfare.welfare_metrics import aggregate_markup_cw

from .decomposition import fixed_capital_decomposition, fixed_capital_table
from .fixed_capital_planner import solve_fixed_capital_planner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.fixed_input_welfare.run_fixed_capital_planner",
        description="Fixed-capital-envelope planner counterfactual "
                    "(PLANNER pricing with K pinned to MARKET's realized K).",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--calibration", default=None,
        help="calibration YAML; in practice required "
             "(e.g. out_results/calib_pooled/calibration_pooled.yaml) -- there "
             "is no silent fallback to a missing default path.",
    )
    parser.add_argument("--M", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--skip-full-planner", action="store_true",
        help="skip the full free-capital PLANNER leg (faster; only reports "
             "lambda_reallocation, no lambda_scale / identity check).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    if args.calibration is None:
        print("warning: no --calibration given; using config initial params. "
              "The production run used "
              "out_results/calib_pooled/calibration_pooled.yaml.")
    M = int(args.M or cfg.monte_carlo.M_final)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)

    support = (inputs.alpha_support if inputs.alpha_support is not None
               else np.linspace(0.6, 1.20, 500))
    draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N, M=M,
        H=params.H, rng=seed, v_min=params.v_min,
    )

    # --- MARKET leg (re-solved from scratch; frozen technology, regime=MARKET),
    #     identical to the market leg produced by solve_market_and_planner. ---
    me = solve_pooled_ge(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        regime=PricingRegime.MARKET,
        active_mask=draw.active_mask, initial=initial,
    )
    K_target = me.K_agg

    # --- FIXED_K planner: PLANNER pricing, K pinned to K_target via R free. ---
    fp = solve_fixed_capital_planner(
        draw.alpha, draw.v, params, K_target=K_target,
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        n_firms_cs=inputs.n_firms_cs, K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        regime=PricingRegime.PLANNER,
        market_solver_kwargs=cfg.market.to_kwargs(),
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask,
        initial=(me.w, me.X_market, euler_R(cfg.parameters.beta, cfg.parameters.delta_K)),
    )

    # --- optional full free-capital PLANNER leg (for lambda_scale / identity). ---
    pe = None
    if not args.skip_full_planner:
        pe = solve_pooled_ge(
            draw.alpha, draw.v, params, beta=cfg.parameters.beta,
            delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
            K_switch=cfg.participation.K_switch,
            love_of_variety=cfg.participation.love_of_variety,
            market_solver_kwargs=cfg.market.to_kwargs(),
            regime=PricingRegime.PLANNER,
            wf_reference=me.participation.wf_reference,
            active_mask=me.participation.active_mask,
            initial=(me.w, me.X_market),
        )

    table = fixed_capital_table(me, fp, pe, n_firms_cs=inputs.n_firms_cs)
    decomp = fixed_capital_decomposition(
        me, fp, pe, chi_me=me.chi, beta=cfg.parameters.beta,
        phi=cfg.parameters.phi, n_firms_cs=inputs.n_firms_cs,
    )

    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.out_results_dir / "counterfactuals" / "fixed_capital_planner")
    out_dir.mkdir(parents=True, exist_ok=True)

    build_firm_panel(
        fp, tilde_alpha=draw.tilde_alpha, regime="planner_fixed_k",
    ).to_parquet(out_dir / "sim_panel_planner_fixed_k.parquet", index=False)
    table.to_parquet(out_dir / "fixed_capital_decomposition.parquet", index=False)

    welfare = {**decomp, "M": M, "seed": seed}
    _write_yaml(out_dir / "fixed_capital_welfare.yaml", welfare)

    market_mu_cw = aggregate_markup_cw(me, n_firms_cs=inputs.n_firms_cs)
    _write_yaml(out_dir / "market_recompute_check.yaml", {
        "C": float(me.C), "L": float(me.L_agg), "K": float(me.K_agg),
        "R": float(me.R), "mu_cw": float(market_mu_cw),
    })

    print(f"wrote outputs to {out_dir}")
    print(f"K_target={K_target:.6g}  K_target_gap={decomp['K_target_gap']:.3e}  "
          f"mu_cw_planner_fixed_k={decomp['mu_cw_planner_fixed_k']:.6g}")
    print(f"R_market={decomp['R_market']:.6g}  "
          f"R_planner_fixed_k={decomp['R_planner_fixed_k']:.6g}")
    print(f"lambda_reallocation={decomp['lambda_reallocation']:.6g}")
    if pe is not None:
        print(f"lambda_scale={decomp['lambda_scale']:.6g}  "
              f"lambda_total={decomp['lambda_total']:.6g}  "
              f"identity_residual={decomp['identity_residual']:.3e}")
    if not (me.converged and fp.converged and (pe is None or pe.converged)):
        print("WARNING: at least one leg did not fully converge; "
              "check residuals / consider bounding log R (see README).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
