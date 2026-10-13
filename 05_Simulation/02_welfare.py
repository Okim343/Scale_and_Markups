"""Step 02: market vs planner welfare at calibrated parameters.

Ported from ``cmd_welfare`` in ``04_Code/steady_state/__main__.py``. Solves the
market and planner allocations on common draws and the same exogenous Poisson
mask, inserts the uniform-markup allocation (mu_bar = market mu_cw) and writes
the two-channel decomposition (lambda_total, lambda_level, lambda_dispersion).
The params file must have been produced on the bundle in ``paths.bundle_dir``.

Reads   <paths.bundle_dir>/{F_alpha.parquet, aggregate_moments.yaml}, <paths.params_file>
Writes  <out-dir>/sim_panel_market_<window>.parquet        market-regime firm panel
        <out-dir>/sim_panel_planner_<window>.parquet       planner-regime firm panel (mu = 1)
        <out-dir>/welfare_decomposition_<window>.parquet   per-regime aggregates
        <out-dir>/welfare_<window>.yaml                    lambdas, W_*, mu_bar, identity residual
        <out-dir>/welfare_manifest_<window>.yaml           provenance

    python 02_welfare.py [--config config_simulation.yaml] [--params FILE] [--M N] [--seed S] [--out-dir D]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from orbis_model.config import load_config, setup_logger
from orbis_model.model.pool import draw_pool
from orbis_model.params import build_provenance, file_sha256, load_simulation_inputs, write_yaml
from orbis_model.simulation.cross_section import build_firm_panel
from orbis_model.welfare import solve_market_and_planner, two_channel_decomposition, welfare_decomposition_table

SIM_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_simulation.yaml")
    parser.add_argument("--params", default=None, help="override paths.params_file")
    parser.add_argument("--M", type=int, default=None, help="override welfare.M")
    parser.add_argument("--seed", type=int, default=None, help="override monte_carlo.master_seed")
    parser.add_argument("--out-dir", default=None, help="override paths.out_results_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logger("welfare", cfg.log_level)
    t0 = time.time()
    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, logger)
    M = int(args.M or cfg.welfare_M)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir
    tag = primitives.window_tag
    draw = draw_pool(
        primitives.alpha_support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
        M=M, H=params.H, rng=seed, v_min=params.v_min,
    )
    pair = solve_market_and_planner(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=primitives.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        active_mask=draw.active_mask, initial=initial,
    )
    decomp = two_channel_decomposition(
        pair.me, pair.pe, params=params, chi_me=pair.me.chi,
        beta=cfg.parameters.beta, phi=cfg.parameters.phi,
        delta_K=cfg.parameters.delta_K, n_firms_cs=primitives.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    build_firm_panel(pair.me, tilde_alpha=draw.tilde_alpha, regime="market").to_parquet(
        out_dir / f"sim_panel_market_{tag}.parquet", index=False)
    build_firm_panel(pair.pe, tilde_alpha=draw.tilde_alpha, regime="planner").to_parquet(
        out_dir / f"sim_panel_planner_{tag}.parquet", index=False)
    welfare_decomposition_table(
        pair.me, pair.pe, decomp["uniform"], n_firms_cs=primitives.n_firms_cs,
    ).to_parquet(out_dir / f"welfare_decomposition_{tag}.parquet", index=False)
    serializable = {k: float(v) for k, v in decomp.items() if k != "uniform"}
    serializable["wf_reference"] = pair.me.participation.wf_reference
    serializable["converged"] = {
        "market": bool(pair.me.converged), "planner": bool(pair.pe.converged),
        "uniform": bool(decomp["uniform"].converged),
    }
    write_yaml(out_dir / f"welfare_{tag}.yaml", serializable)
    provenance = build_provenance(
        bundle_dir=cfg.bundle_dir, configs={"simulation": cfg.path},
        code_dirs=(SIM_DIR,), script="05_Simulation/02_welfare.py",
    )
    write_yaml(out_dir / f"welfare_manifest_{tag}.yaml", {
        **provenance,
        "params_file": {"path": str(params_path), "sha256": file_sha256(params_path)},
        "M": M, "seed": seed, "runtime_s": round(time.time() - t0, 1),
    })
    logger.info(f"lambda_total={decomp['lambda_total']:.6g} lambda_level={decomp['lambda_level']:.6g} "
                f"lambda_dispersion={decomp['lambda_dispersion']:.6g} mu_bar={decomp['mu_bar']:.6g}; "
                f"identity residual={decomp['identity_residual']:.3e}")
    logger.info(f"converged market/planner/uniform = {serializable['converged']}; "
                f"wrote {out_dir} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
