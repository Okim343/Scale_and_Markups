"""Step 01: simulate the pooled cross section at calibrated parameters.

Ported from ``cmd_simulate`` in ``04_Code/steady_state/__main__.py``. Draws the
common random numbers at the params-file vector (frozen ``v_min``), solves the
pooled GE (P = 1, L = 1) on the exogenous Poisson mask and writes the firm
panel and the model moments. The params file must have been produced on the
bundle in ``paths.bundle_dir`` (sha256 check).

Reads   <paths.bundle_dir>/{F_alpha.parquet, aggregate_moments.yaml}, <paths.params_file>
Writes  <out-dir>/sim_panel_<window>.parquet         rectangular firm panel (M x H rows)
        <out-dir>/sim_moments_<window>.parquet       pooled model moments
        <out-dir>/sim_economy_moments_<window>.yaml  moments plus GE aggregates
        <out-dir>/simulate_manifest_<window>.yaml    provenance

    python 01_simulate.py [--config config_simulation.yaml] [--params FILE] [--M N] [--seed S] [--out-dir D]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from orbis_model.config import load_config, setup_logger
from orbis_model.params import build_provenance, file_sha256, load_simulation_inputs, write_yaml
from orbis_model.simulation.cross_section import run_simulation
from orbis_model.simulation.moments import economy_moments, pooled_moments_table

SIM_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_simulation.yaml")
    parser.add_argument("--params", default=None, help="override paths.params_file")
    parser.add_argument("--M", type=int, default=None, help="override monte_carlo.M")
    parser.add_argument("--seed", type=int, default=None, help="override monte_carlo.master_seed")
    parser.add_argument("--out-dir", default=None, help="override paths.out_results_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logger("simulate", cfg.log_level)
    t0 = time.time()
    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, logger)
    M = int(args.M or cfg.monte_carlo.M)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir
    tag = primitives.window_tag
    sim = run_simulation(
        primitives, params, M=M, master_seed=seed,
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        initial=initial,
        out_dir=out_dir, panel_filename=f"sim_panel_{tag}.parquet",
    )
    pooled_moments_table(sim, primitives).to_parquet(out_dir / f"sim_moments_{tag}.parquet", index=False)
    moments = economy_moments(sim, primitives)
    write_yaml(out_dir / f"sim_economy_moments_{tag}.yaml", moments)
    provenance = build_provenance(
        bundle_dir=cfg.bundle_dir, configs={"simulation": cfg.path},
        code_dirs=(SIM_DIR,), script="05_Simulation/01_simulate.py",
    )
    write_yaml(out_dir / f"simulate_manifest_{tag}.yaml", {
        **provenance,
        "params_file": {"path": str(params_path), "sha256": file_sha256(params_path)},
        "M": M, "seed": seed, "runtime_s": round(time.time() - t0, 1),
    })
    for key in ("mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
                "alpha_rts_gap_top5_bottom50", "mu_cw_alpha", "mean_active_count"):
        logger.info(f"  {key:<28s} {moments[key]: .6g}")
    logger.info(f"simulated {M} markets (window {tag}); GE converged={moments['converged']}, "
                f"residual={moments['normalization_residual']:.2e}; wrote {out_dir} "
                f"({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
