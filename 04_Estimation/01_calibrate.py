"""Step 01: calibrate (xi, N, gamma, eta, rho_bar) to the Orbis Spain targets.

Ported from ``cmd_calibrate`` in ``04_Code/steady_state/__main__.py``. One
joint ``trf`` least-squares fit through the anchored GE solve (which also pins
``v_min``), then the post-calibration sunk entry cost. Initial values, bounds,
targeted moments and weights come from ``config_estimation.yaml``; the model
settings from the simulation config it names. Nothing is read from 04_Code.

Reads   <paths.bundle_dir>/{F_alpha.parquet, aggregate_moments.yaml, manifest.yaml}
Writes  <out-dir>/calibrated_params.yaml          parameters, reference prices, fit, provenance
        <out-dir>/calibration_manifest.yaml       provenance plus output sha256
        <out-dir>/calibration_fit_<window>.csv    model vs target per targeted moment
        <out-dir>/calibration_checkpoint.jsonl    every objective evaluation (crash recovery)

    python 01_calibrate.py [--config config_estimation.yaml] [--M N] [--seed S] [--max-nfev K] [--out-dir D]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from orbis_estimation import calibrate_pooled, compute_sunk_entry_cost
from orbis_estimation.config import initial_params, load_config
from orbis_estimation.diagnostics import calibration_fit_table
from orbis_estimation.targets import load_targets
from orbis_model.config import setup_logger
from orbis_model.params import build_provenance, file_sha256, write_params_file, write_yaml
from orbis_model.primitives import load_primitives

EST_DIR = Path(__file__).resolve().parent
SIM_DIR = EST_DIR.parent / "05_Simulation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_estimation.yaml")
    parser.add_argument("--M", type=int, default=None, help="override monte_carlo.M")
    parser.add_argument("--seed", type=int, default=None, help="override monte_carlo.seed")
    parser.add_argument("--max-nfev", type=int, default=None, help="override inner.max_nfev")
    parser.add_argument("--out-dir", default=None, help="override paths.out_results_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    sim = cfg.simulation
    logger = setup_logger("calibrate", cfg.log_level)
    t0 = time.time()
    primitives = load_primitives(cfg.bundle_dir, H=sim.parameters.H)
    targets = load_targets(cfg.bundle_dir)
    initial = initial_params(cfg, primitives)
    M = int(args.M or cfg.M)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    max_nfev = int(args.max_nfev or cfg.max_nfev)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"bundle {cfg.bundle_dir} (window {primitives.window_tag}); M={M}, seed={seed}, "
                f"max_nfev={max_nfev}; targeted {list(cfg.moment_keys)}")
    logger.info(f"initial (xi, N, gamma, eta, rho_bar) = ({initial.xi:.6g}, {initial.N:.6g}, "
                f"{initial.gamma:.6g}, {initial.eta:.6g}, {initial.rho_bar:.6g}); "
                f"rho_bar bounds {cfg.bounds['rho_bar']}")
    result = calibrate_pooled(
        primitives, targets, M=M, rng=seed,
        initial=(initial.xi, initial.N, initial.gamma, initial.eta, initial.rho_bar),
        xi_bounds=cfg.bounds["xi"],
        N_bounds=cfg.bounds["N"],
        eta_bounds=cfg.bounds["eta"],
        gamma_gap_bounds=cfg.bounds["gamma_gap"],
        rho_bar_bounds=cfg.bounds["rho_bar"],
        moment_keys=cfg.moment_keys,
        weights=cfg.weights,
        beta=sim.parameters.beta, delta_K=sim.parameters.delta_K,
        K_switch=sim.participation.K_switch,
        love_of_variety=sim.participation.love_of_variety,
        market_solver_kwargs=sim.market.to_kwargs(),
        ftol=cfg.ftol, xtol=cfg.xtol,
        max_nfev=max_nfev,
        checkpoint_path=out_dir / "calibration_checkpoint.jsonl",
    )
    sunk_entry_cost = compute_sunk_entry_cost(
        result.equilibrium, beta=sim.parameters.beta, varphi=cfg.varphi
    )
    provenance = build_provenance(
        bundle_dir=cfg.bundle_dir,
        configs={"estimation": cfg.path, "simulation": sim.path},
        code_dirs=(EST_DIR, SIM_DIR), script="04_Estimation/01_calibrate.py",
    )
    provenance.update({"M": M, "seed": seed, "max_nfev": max_nfev})
    params_path = write_params_file(
        out_dir / "calibrated_params.yaml", result.params,
        reference={
            "w": result.reference.w, "R": result.reference.R,
            "X_market": result.reference.X_market,
            "wf_reference": result.reference.wf_reference,
            "Ed_full_pool": result.reference.Ed_full_pool,
            "P": result.reference.P, "L": result.reference.L,
        },
        provenance=provenance,
        entry={
            "sunk_entry_cost_F": sunk_entry_cost,
            "varphi": cfg.varphi,
            "note": "post-calibration EMX free-entry backout; welfare-only, "
                    "does not enter the static cross-section",
        },
        fit={
            "success": result.success, "cost": result.cost, "nfev": result.nfev,
            "message": result.message,
            "moments": {
                k: {
                    "model": float(result.model_moments[k]),
                    "target": float(result.target_moments[k]),
                    "weight": float(cfg.weights[k]),
                }
                for k in result.moment_keys
            },
            "diagnostics": {
                "mu_cw_alpha": float(result.model_moments["mu_cw_alpha"]),
                "mean_active_count": float(result.model_moments["mean_active_count"]),
                "mean_firms_per_sector_data": primitives.mean_firms_per_sector,
                "y_sw": float(result.equilibrium.y_sw),
                "y_anchor": float(result.equilibrium.y_anchor),
                "corr_alpha_log_sales": float(result.model_moments["corr_alpha_log_sales"]),
                "corr_alpha_log_sales_partial_z": float(result.model_moments["corr_alpha_log_sales_partial_z"]),
                "corr_alpha_log_sales_partial_v": float(result.model_moments["corr_alpha_log_sales_partial_v"]),
            },
        },
        identification={
            "eta_jacobian_column_norm": result.eta_jacobian_column_norm,
            "eta_standard_error": float(result.standard_errors[3]),
            "jacobian_rank": result.jacobian_rank,
            "jacobian_condition": result.jacobian_condition,
            "share_std": float(result.model_moments["share_std"]),
            "share_range": float(result.model_moments["share_range"]),
        },
    )
    fit_path = out_dir / f"calibration_fit_{primitives.window_tag}.csv"
    calibration_fit_table(result).to_csv(fit_path, index=False)
    write_yaml(out_dir / "calibration_manifest.yaml", {
        **provenance,
        "runtime_s": round(time.time() - t0, 1),
        "outputs": {
            path.name: {"sha256": file_sha256(path)} for path in (params_path, fit_path)
        },
    })
    for k in result.moment_keys:
        logger.info(f"  {k:<28s} model {result.model_moments[k]: .6g}  "
                    f"target {result.target_moments[k]: .6g}  weight {cfg.weights[k]:g}")
    logger.info(f"calibrated (xi, N, gamma, eta, rho_bar, v_min) = "
                f"({result.xi:.6g}, {result.N:.6g}, {result.gamma:.6g}, {result.eta:.6g}, "
                f"{result.rho_bar:.6g}, {result.v_min:.6g}); cost={result.cost:.6g}, "
                f"nfev={result.nfev}, success={result.success}")
    lo, hi = cfg.bounds["N"]
    if result.N > 0.99 * hi:
        logger.warning(f"N={result.N:.6g} is at its upper bound {hi:g}")
    logger.info(f"anchor (median) y_anchor={result.equilibrium.y_anchor:.6g} "
                f"(y_sw={result.equilibrium.y_sw:.6g})")
    logger.info(f"wrote {params_path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
