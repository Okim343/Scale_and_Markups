"""Step 03: fixed-input planner welfare chain at calibrated parameters.

Ported from ``04_Code/counterfactuals/fixed_input_welfare/run_fixed_capital_planner.py``.
On the draws and exogenous Poisson mask of ``02_welfare.py`` (same M and seed
defaults, so ME matches its market leg), solves

    ME -> PE_I (K, L, M fixed) -> PE_K (K, L fixed, M free) -> PE (L fixed)

where PE_I pins aggregate capital and materials at the market levels by freeing
the rental rate R and a shadow materials price P_mat, PE_K is the 04_Code
fixed-capital planner (R free, materials free at P = 1), and PE is the free
planner of ``02_welfare.py``. Reports the legs lambda_realloc (ME -> PE_I),
lambda_materials (PE_I -> PE_K) and lambda_capital (PE_K -> PE), their
Delta = ln(1 + lambda) versions and the identity
(1+l_r)(1+l_m)(1+l_c) = 1 + lambda_total. chi is held at the ME value.

Reads   <paths.bundle_dir>/{F_alpha.parquet, aggregate_moments.yaml}, <paths.params_file>
Writes  <out-dir>/fixed_input_regimes_<window>.parquet               per-regime aggregates
        <out-dir>/fixed_input_welfare_<window>.yaml                  lambdas, Deltas, R, P_mat, gaps, timings
        <out-dir>/sim_panel_planner_fixed_input_<window>.parquet     PE_I firm panel
        <out-dir>/sim_panel_planner_fixed_k_<window>.parquet         PE_K firm panel
        <out-dir>/fixed_input_manifest_<window>.yaml                 provenance
(default <out-dir> = <paths.out_results_dir>/fixed_input)

    python 03_fixed_input_welfare.py [--config config_simulation.yaml] [--params FILE] [--M N]
        [--seed S] [--out-dir D] [--skip-full-planner] [--skip-fixed-k]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from orbis_model.config import load_config, setup_logger
from orbis_model.model.pool import alpha_cap, draw_pool
from orbis_model.params import build_provenance, file_sha256, load_simulation_inputs, write_yaml
from orbis_model.simulation.cross_section import build_firm_panel
from orbis_model.welfare import fixed_input_decomposition, fixed_input_table, solve_fixed_input_chain

SIM_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_simulation.yaml")
    parser.add_argument("--params", default=None, help="override paths.params_file")
    parser.add_argument("--M", type=int, default=None, help="override welfare.M")
    parser.add_argument("--seed", type=int, default=None, help="override monte_carlo.master_seed")
    parser.add_argument("--out-dir", default=None,
                        help="default: <paths.out_results_dir>/fixed_input")
    parser.add_argument("--skip-full-planner", action="store_true",
                        help="skip PE (no lambda_capital, lambda_total or identity check)")
    parser.add_argument("--skip-fixed-k", action="store_true",
                        help="skip PE_K (PE_I warm-starts from ME; reports lambda_materials_capital)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logger("fixed_input", cfg.log_level)
    t0 = time.time()
    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, logger)
    M = int(args.M or cfg.welfare_M)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else cfg.out_results_dir / "fixed_input")
    fixed_cfg = cfg.raw.get("fixed_input", {}) or {}
    tol = float(fixed_cfg.get("tol", 1e-5))
    max_nfev = int(fixed_cfg.get("max_nfev", 40))
    tag = primitives.window_tag
    draw = draw_pool(
        primitives.alpha_support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
        M=M, H=params.H, rng=seed, v_min=params.v_min,
        alpha_max=alpha_cap(params.gamma, cfg.parameters.alpha_cap_factor),
    )
    chain = solve_fixed_input_chain(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=primitives.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        active_mask=draw.active_mask, initial=initial, tol=tol, max_nfev=max_nfev,
        fixed_k=not args.skip_fixed_k, full_planner=not args.skip_full_planner,
    )
    me, pe_i, pe_k, pe = chain.me, chain.pe_i, chain.pe_k, chain.pe
    decomp = fixed_input_decomposition(
        me, pe_i, pe_k, pe, chi_me=me.chi, beta=cfg.parameters.beta, phi=cfg.parameters.phi,
    )
    table = fixed_input_table(me, pe_i, pe_k, pe, n_firms_cs=primitives.n_firms_cs,
                              R_euler=chain.R_euler)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"fixed_input_regimes_{tag}.parquet", index=False)
    build_firm_panel(pe_i, tilde_alpha=draw.tilde_alpha, regime="planner_fixed_input").to_parquet(
        out_dir / f"sim_panel_planner_fixed_input_{tag}.parquet", index=False)
    if pe_k is not None:
        build_firm_panel(pe_k, tilde_alpha=draw.tilde_alpha, regime="planner_fixed_k").to_parquet(
            out_dir / f"sim_panel_planner_fixed_k_{tag}.parquet", index=False)
    converged = {row.regime: bool(row.converged) for row in table.itertuples()}
    serializable = {
        **decomp,
        "R_euler": chain.R_euler,
        "wf_reference": me.participation.wf_reference,
        "converged": converged,
        "residual": {row.regime: float(row.residual) for row in table.itertuples()},
        "solver": {"tol": tol, "max_nfev": max_nfev, "stats": chain.stats},
        "wall_s": {label: round(sec, 2) for label, sec in chain.wall_s.items()},
    }
    write_yaml(out_dir / f"fixed_input_welfare_{tag}.yaml", serializable)
    provenance = build_provenance(
        bundle_dir=cfg.bundle_dir, configs={"simulation": cfg.path},
        code_dirs=(SIM_DIR,), script="05_Simulation/03_fixed_input_welfare.py",
    )
    write_yaml(out_dir / f"fixed_input_manifest_{tag}.yaml", {
        **provenance,
        "params_file": {"path": str(params_path), "sha256": file_sha256(params_path)},
        "M": M, "seed": seed, "runtime_s": round(time.time() - t0, 1),
    })

    lambdas = " ".join(f"{k}={v:.6g}" for k, v in decomp.items() if k.startswith("lambda_"))
    logger.info(lambdas)
    if pe is not None:
        logger.info(f"identity residual={decomp['identity_residual']:.3e} "
                    f"(Delta: {decomp['Delta_identity_residual']:.3e})")
    for row in table.itertuples():
        logger.info(f"{row.regime:>20}: R={row.R:.6g} P_mat={row.P_mat:.6g} "
                    f"K_gap={row.K_target_gap:.2e} M_gap={row.M_target_gap:.2e} "
                    f"C={row.C:.6g} nfev={row.n_func_evals} "
                    f"wall={chain.wall_s[row.regime]:.1f}s")
    if not all(converged.values()):
        logger.warning(f"not every regime converged: {converged}")
    logger.info(f"wrote {out_dir} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
