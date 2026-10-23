"""Step 01 (market-free variant): calibrate (xi, N, gamma, rho_bar) with eta fixed.

Parallel to ``01_calibrate.py``, which it leaves untouched. Targets ``mu_cw`` and
the RTS gap (bundle) and the Zipf tail index of firm sales at the top 5% and top 2%
(``zipf_targets.yaml`` from ``04_Moments/01_code/12_zipf_targets.py``); ``eta`` is
fixed in ``config_estimation_zipf.yaml``. One ``trf`` fit through the anchored GE
solve (:func:`orbis_estimation.zipf.calibrate_zipf`), then the post-calibration sunk
entry cost. The model settings come from the simulation config the estimation config
names. Design: ``02_Drafts/ARCHIVE/md_files/ORBIS_migration/ORBIS_ZIPF_CALIBRATION.md``.

Reads   <paths.bundle_dir>/{F_alpha.parquet, aggregate_moments.yaml, manifest.yaml}, <paths.zipf_targets>
Writes  <out-dir>/calibrated_params_zipf.yaml          params-file schema (05_Simulation --params)
        <out-dir>/calibration_manifest_zipf.yaml       provenance plus output sha256
        <out-dir>/calibration_fit_zipf_<window>.csv    model vs target per targeted moment
        <out-dir>/calibration_identification_zipf.yaml Jacobian, elasticities, condition numbers
        <out-dir>/calibration_checkpoint_zipf.jsonl    every objective evaluation

    python 01_calibrate_zipf.py [--config config_estimation_zipf.yaml] [--M N] [--seed S]
                                [--max-nfev K] [--out-dir D]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from orbis_estimation import compute_sunk_entry_cost
from orbis_estimation.zipf import (
    PARAMETER_NAMES, calibrate_zipf, load_zipf_config, load_zipf_targets, model_kwargs,
    validation_moments,
)
from orbis_model.config import setup_logger
from orbis_model.params import build_provenance, file_sha256, write_params_file, write_yaml
from orbis_model.primitives import load_primitives

EST_DIR = Path(__file__).resolve().parent
SIM_DIR = EST_DIR.parent / "05_Simulation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_estimation_zipf.yaml")
    parser.add_argument("--M", type=int, default=None, help="override monte_carlo.M")
    parser.add_argument("--seed", type=int, default=None, help="override monte_carlo.seed")
    parser.add_argument("--max-nfev", type=int, default=None, help="override inner.max_nfev")
    parser.add_argument("--out-dir", default=None, help="override paths.out_results_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_zipf_config(args.config)
    sim = cfg.simulation
    logger = setup_logger("calibrate_zipf", cfg.log_level)
    t0 = time.time()
    primitives = load_primitives(cfg.bundle_dir, H=sim.parameters.H)
    targets = load_zipf_targets(cfg.bundle_dir, cfg.zipf_targets)
    M = int(args.M or cfg.M)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    max_nfev = int(args.max_nfev or cfg.max_nfev)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = model_kwargs(sim)
    iv = cfg.initial_values
    logger.info(f"bundle {cfg.bundle_dir} (window {primitives.window_tag}); M={M}, seed={seed}, "
                f"max_nfev={max_nfev}; eta fixed at {cfg.eta}; targeted {list(cfg.moment_keys)}")
    logger.info(f"model settings: {kw}")
    logger.info("targets: " + ", ".join(f"{k}={targets[k]:.6g}" for k in cfg.moment_keys))
    logger.info("initial: " + ", ".join(f"{k}={iv[k]:.6g}" for k in PARAMETER_NAMES))

    result = calibrate_zipf(
        primitives, targets, M=M, rng=seed, eta=cfg.eta,
        initial=[iv[k] for k in PARAMETER_NAMES], bounds=cfg.bounds, weights=cfg.weights,
        ftol=cfg.ftol, xtol=cfg.xtol, max_nfev=max_nfev, fd_step=cfg.fd_step,
        checkpoint_path=out_dir / "calibration_checkpoint_zipf.jsonl", logger=logger, **kw,
    )
    sunk_entry_cost = compute_sunk_entry_cost(result.equilibrium, beta=sim.parameters.beta, varphi=cfg.varphi)
    validation = validation_moments(
        result.equilibrium, result.params, alpha_support=primitives.alpha_support,
        alpha_cap_factor=sim.parameters.alpha_cap_factor,
    )
    provenance = build_provenance(
        bundle_dir=cfg.bundle_dir,
        configs={"estimation": cfg.path, "simulation": sim.path},
        code_dirs=(EST_DIR, SIM_DIR), script="04_Estimation/01_calibrate_zipf.py",
    )
    provenance.update({"M": M, "seed": seed, "max_nfev": max_nfev,
                       "zipf_targets": {"path": str(cfg.zipf_targets), "sha256": file_sha256(cfg.zipf_targets)}})
    m = result.model_moments
    params_path = write_params_file(
        out_dir / "calibrated_params_zipf.yaml", result.params,
        reference={
            "w": result.reference.w, "R": result.reference.R, "X_market": result.reference.X_market,
            "wf_reference": result.reference.wf_reference, "Ed_full_pool": result.reference.Ed_full_pool,
            "P": result.reference.P, "L": result.reference.L,
        },
        provenance=provenance,
        model={"alpha_cap_factor": sim.parameters.alpha_cap_factor, "alpha_draw": "truncated"},
        calibration={"variant": "zipf", "eta_fixed": cfg.eta,
                     "calibrated": list(PARAMETER_NAMES), "targets": list(cfg.moment_keys)},
        entry={
            "sunk_entry_cost_F": sunk_entry_cost, "varphi": cfg.varphi,
            "note": "post-calibration EMX free-entry backout; welfare-only, "
                    "does not enter the static cross-section",
        },
        fit={
            "success": result.success, "status": result.status, "cost": result.cost,
            "nfev": result.nfev, "n_evals": result.n_evals, "message": result.message,
            "ge_converged": bool(result.equilibrium.converged),
            "ge_residual": float(result.equilibrium.residual),
            "at_bound": list(result.at_bound),
            "moments": {
                k: {"model": float(m[k]), "target": float(result.target_moments[k]),
                    "weight": float(cfg.weights[k])}
                for k in result.moment_keys
            },
            "untargeted": {
                k: {"model": float(m[k]), "target": float(result.target_moments[k])}
                for k in ("zeta_top1", "cr4", "cr20", "top1pct", "top5pct", "emx_slope")
            },
            "diagnostics": {
                "mu_cw_alpha": float(m["mu_cw_alpha"]),
                "mean_active_count": float(m["mean_active_count"]),
                "mean_firms_per_sector_data": primitives.mean_firms_per_sector,
                "y_sw": float(result.equilibrium.y_sw), "y_anchor": float(result.equilibrium.y_anchor),
                **{k: float(m[k]) for k in ("zeta_hill_top5", "zeta_hill_top2", "zeta_hill_top1",
                                            "corr_alpha_log_sales") if k in m},
            },
        },
        identification={
            "jacobian_condition": result.jacobian_condition,
            "elasticity_condition": result.elasticity_condition,
            "zeta_block_condition": result.zeta_block_condition,
        },
    )
    fit = pd.DataFrame([
        {"moment": k, "model": float(m[k]), "target": float(result.target_moments[k]),
         "relative_error": float(m[k] / result.target_moments[k] - 1.0),
         "weighted_residual": float(r), "weight": float(cfg.weights[k])}
        for k, r in zip(result.moment_keys, result.residuals, strict=True)
    ])
    fit_path = out_dir / f"calibration_fit_zipf_{primitives.window_tag}.csv"
    fit.to_csv(fit_path, index=False)
    cols = ["xi", "N", "gamma", "rho_bar"]
    ident_path = out_dir / "calibration_identification_zipf.yaml"
    write_yaml(ident_path, {
        "note": "jacobian_x: d(weighted residual)/d(log xi, log N, log(gamma-eta), logit rho_bar); "
                "elasticities: d log m / d log p for xi, N, gamma and d log m / d rho_bar",
        "moment_keys": list(result.moment_keys), "columns": cols,
        "jacobian_x": np.asarray(result.jacobian_x).tolist(),
        "elasticities": np.asarray(result.elasticities).tolist(),
        "jacobian_condition": result.jacobian_condition,
        "elasticity_condition": result.elasticity_condition,
        "zeta_block_condition": result.zeta_block_condition,
        "validation": {k: float(v) for k, v in validation.items()},
    })
    write_yaml(out_dir / "calibration_manifest_zipf.yaml", {
        **provenance, "runtime_s": round(time.time() - t0, 1),
        "outputs": {p.name: {"sha256": file_sha256(p)} for p in (params_path, fit_path, ident_path)},
    })
    logger.info("fit:\n" + fit.to_string(index=False))
    logger.info("elasticities (rows moments, cols xi N gamma rho_bar):\n"
                + pd.DataFrame(result.elasticities, index=result.moment_keys, columns=cols).round(3).to_string())
    logger.info(f"calibrated xi={result.params.xi:.6g} N={result.params.N:.6g} gamma={result.params.gamma:.6g} "
                f"eta={result.params.eta:g} rho_bar={result.params.rho_bar:.6g} v_min={result.params.v_min:.6g}; "
                f"cost={result.cost:.4g}, nfev={result.nfev}, evals={result.n_evals}, success={result.success} "
                f"({result.message}); GE converged={result.equilibrium.converged}; cond(J)={result.jacobian_condition:.3g}")
    if result.at_bound:
        logger.warning(f"parameters at a bound: {list(result.at_bound)}")
    logger.info(f"wrote {params_path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
