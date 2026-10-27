"""Step 03: recalibrate a restricted-heterogeneity economy to the baseline Zipf targets.

The restricted economies of Table 2 (``hom_alpha``: capability only; ``hom_v``:
scalability only; ``hom_both``: neither) are refit to the same four targets,
weights, bounds, M and seed as ``01_calibrate_zipf.py`` over their own free
parameters (:mod:`orbis_estimation.restricted`). The moment function is the
canonical :func:`orbis_estimation.zipf.zipf_moments_at_params`; the restriction
is in its inputs (``xi = inf`` for common v, a one-node alpha support for common
alpha).

Three modes:
    --gate0   the baseline economy at the job 3162 parameters must reproduce that
              file's fitted moments exactly (identity of the restricted path; also
              checks that the worker count does not move the numbers)
    --scan    the objective on the config grid at scan_M (starting values)
    default   trf fit; restarts replay the checkpoint, and the submit script
              restarts the process on exit code 75 (evaluation timeout)

Reads   <zipf_config> (bundle, targets, simulation config, M, seed, weights, bounds),
        <paths.baseline_params> (sha256-checked)
Writes  <out-dir>/<economy>/calibrated_params_<economy>.yaml   params-file schema plus restriction block
        <out-dir>/<economy>/calibration_manifest_<economy>.yaml provenance plus output sha256
        <out-dir>/<economy>/calibration_fit_<economy>.csv       model vs target, every target
        <out-dir>/<economy>/calibration_checkpoint_<economy>.jsonl  every evaluation (replayed on restart)
        <out-dir>/<economy>/scan_<economy>.csv (+ scan checkpoint)   with --scan
        <out-dir>/gate0_w<workers>.json                         with --gate0

    python 03_calibrate_restricted.py --economy {hom_alpha,hom_v,hom_both} [--scan | --gate0]
        [--config FILE] [--initial-from-scan] [--M N] [--seed S] [--max-nfev K]
        [--workers W] [--out-dir D]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from orbis_estimation import compute_sunk_entry_cost
from orbis_estimation.restricted import (
    ALL_PARAMETERS, ECONOMIES, Checkpoint, Evaluator, Watchdog, calibrate_restricted,
    economy_moments_at_params, encode, fingerprint, scan_points,
)
from orbis_estimation.zipf import load_zipf_config, load_zipf_targets, model_kwargs, validation_moments
from orbis_model.config import setup_logger
from orbis_model.model.normalization import PooledParams
from orbis_model.params import build_provenance, file_sha256, write_params_file, write_yaml
from orbis_model.primitives import load_primitives

EST_DIR = Path(__file__).resolve().parent
SIM_DIR = EST_DIR.parent / "05_Simulation"
DEFAULT_CONFIG = EST_DIR / "config_estimation_restricted.yaml"
UNTARGETED = ("zeta_top1", "cr4", "cr20", "top1pct", "top5pct", "emx_slope")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--economy", required=True, choices=[k for k in ECONOMIES if k != "baseline"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--scan", action="store_true", help="evaluate the config grid at scan_M")
    mode.add_argument("--gate0", action="store_true", help="identity check at the baseline parameters")
    p.add_argument("--config", default=None, help="default: config_estimation_restricted.yaml")
    p.add_argument("--initial-from-scan", action="store_true", help="start the fit at the best scan point")
    p.add_argument("--M", type=int, default=None, help="override M (fit) or scan_M (scan)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-nfev", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="override solver.parallel.n_jobs")
    p.add_argument("--out-dir", default=None, help="override paths.out_results_dir")
    return p


def _resolve(path: str, anchor: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (anchor / p).resolve()


def load_restricted_config(path: str | None) -> dict:
    cfg_path = Path(path).resolve() if path else DEFAULT_CONFIG
    raw = yaml.safe_load(cfg_path.read_text())
    paths = raw["paths"]
    return {
        "path": cfg_path, "raw": raw,
        "zipf_config": _resolve(paths["zipf_config"], cfg_path.parent),
        "baseline_params": _resolve(paths["baseline_params"], cfg_path.parent),
        "out_results_dir": _resolve(paths["out_results_dir"], cfg_path.parent),
        "baseline_sha": str(raw["baseline_params_sha256"]),
        "scan_M": int(raw.get("scan_M", 1000)),
        "economies": raw.get("economies") or {},
        "resilience": raw.get("resilience") or {},
        "log_level": str(raw.get("log_level", "INFO")),
    }


def baseline_values(path: Path, expected_sha: str, primitives) -> tuple[dict, PooledParams]:
    """The job 3162 file, sha-checked, and its parameters as the fit's ``decode`` built them (v_min = 1)."""
    sha = file_sha256(path)
    if sha != expected_sha:
        raise ValueError(f"{path} has sha256 {sha}; the config pins {expected_sha}")
    data = yaml.safe_load(path.read_text())
    p = data["parameters"]
    if (float(p["a"]), int(p["H"]), float(p["phi_v"])) != (primitives.a, primitives.H, primitives.phi_v):
        raise ValueError("baseline params a/H/phi_v differ from the bundle and config")
    params = PooledParams(xi=float(p["xi"]), N=float(p["N"]), gamma=float(p["gamma"]), eta=float(p["eta"]),
                          rho_bar=float(p["rho_bar"]), a=primitives.a, H=primitives.H, phi_v=primitives.phi_v)
    return data, params


def gate0(zcfg, rcfg, primitives, targets, kw, factor, *, M, seed, workers, out_dir, logger) -> int:
    """The restricted path in baseline mode reproduces the job 3162 fitted moments exactly."""
    data, params = baseline_values(rcfg["baseline_params"], rcfg["baseline_sha"], primitives)
    prov = data["provenance"]
    if (int(prov["M"]), int(prov["seed"])) != (M, seed):
        raise ValueError(f"gate 0 needs the baseline's M and seed {prov['M'], prov['seed']}; got {M, seed}")
    t0 = time.time()
    model, _, eq, _, _ = economy_moments_at_params(ECONOMIES["baseline"], params, primitives, M=M, rng=seed,
                                                   alpha_cap_factor=factor, **kw)
    saved = {k: v["model"] for k, v in data["fit"]["moments"].items()}
    saved.update({k: v["model"] for k, v in data["fit"]["untargeted"].items()})
    fresh = {k: float(model[k]) for k in saved}
    saved["v_min"], fresh["v_min"] = float(data["parameters"]["v_min"]), float(eq.v_min)
    rel = {k: abs(fresh[k] - float(saved[k])) / max(abs(float(saved[k])), 1e-300) for k in saved}
    payload = {"exact": all(fresh[k] == float(saved[k]) for k in saved),
               "max_relative_difference": max(rel.values()), "relative_differences": rel,
               "fresh": fresh, "saved": saved, "workers": workers, "M": M, "seed": seed,
               "baseline_params": str(rcfg["baseline_params"]), "secs": round(time.time() - t0, 1),
               "ge_converged": bool(eq.converged)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gate0_w{workers}.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"gate 0: exact={payload['exact']} max relative difference "
                f"{payload['max_relative_difference']:.3e} ({payload['secs']:.0f} s, {workers} workers)")
    return 0 if payload["exact"] else 1


def make_evaluator(economy, *, zcfg, primitives, targets, kw, factor, fixed, M, seed, ckpt_path, header,
                   resilience, logger) -> Evaluator:
    rho_bounds = tuple(zcfg.bounds["rho_bar"])
    run = {**header, "economy": economy.label, "M": M, "seed": seed, "fixed": fixed,
           "weights": dict(zcfg.weights), "bounds": {k: list(v) for k, v in zcfg.bounds.items()},
           "targets": {k: float(targets[k]) for k in zcfg.weights}}
    ckpt = Checkpoint(ckpt_path, fingerprint(run), run)
    ev = Evaluator(economy=economy, primitives=primitives, targets=targets, weights=zcfg.weights,
                   fixed=fixed, eta=zcfg.eta, rho_bar_bounds=rho_bounds, M=M, rng=seed,
                   alpha_cap_factor=factor, model_kw=kw, checkpoint=ckpt, logger=logger)
    ev.watchdog = Watchdog(factor=float(resilience.get("timeout_factor", 4.0)),
                           min_s=float(resilience.get("timeout_min_s", 900)),
                           first_s=float(resilience.get("timeout_first_s", 5400)),
                           on_timeout=ev.record_timeout, logger=logger)
    return ev


def run_scan(ev: Evaluator, grid, initial, *, out_csv: Path, logger) -> None:
    points = scan_points(grid, ev.economy.free, initial)
    rows = []
    for k, point in enumerate(points):
        x = encode(point, ev.economy.free, eta=ev.eta, rho_bar_bounds=ev.rho_bar_bounds)
        arr = ev.residual(x, tag=f"scan{k}")
        record = ev.checkpoint.get(x) or {}
        row = {**point, "cost": 0.5 * float(arr @ arr), "status": record.get("status"),
               **{f"res_{key}": float(r) for key, r in zip(ev.moment_keys, arr)}}
        row.update({f"m_{key}": val for key, val in (record.get("moments") or {}).items()
                    if key in (*ev.moment_keys, *UNTARGETED, "mu_cw_alpha", "n_active_firms")})
        for extra in ("alpha_common", "v_min", "secs"):
            if extra in record:
                row[extra] = record[extra]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    best = min(rows, key=lambda r: r["cost"])
    logger.info(f"scan: {len(rows)} points ({ev.replayed} replayed) -> {out_csv}; best "
                + ", ".join(f"{k}={best[k]:.5g}" for k in ev.economy.free) + f" cost={best['cost']:.4g}")


def write_outputs(result, *, economy, zcfg, rcfg, primitives, targets, factor, fixed, initial, M, seed,
                  max_nfev, out_dir, logger, t0) -> None:
    sim = zcfg.simulation
    m = result.model_moments
    eq = result.equilibrium
    validation = validation_moments(eq, result.params, alpha_support=result.primitives.alpha_support,
                                    alpha_cap_factor=factor)
    provenance = build_provenance(
        bundle_dir=zcfg.bundle_dir,
        configs={"restricted": rcfg["path"], "estimation": zcfg.path, "simulation": sim.path},
        code_dirs=(EST_DIR, SIM_DIR), script="04_Estimation/03_calibrate_restricted.py",
    )
    provenance.update({
        "M": M, "seed": seed, "max_nfev": max_nfev,
        "zipf_targets": {"path": str(zcfg.zipf_targets), "sha256": file_sha256(zcfg.zipf_targets)},
        "baseline_params": {"path": str(rcfg["baseline_params"]), "sha256": rcfg["baseline_sha"]},
    })
    restriction = {
        "economy": economy.label, "common_alpha": economy.common_alpha, "common_v": economy.common_v,
        "alpha_common": float(result.primitives.alpha_support[0]) if economy.common_alpha else None,
        "alpha_common_rule": ("harmonic mean of the F^alpha nodes below alpha_cap(gamma)"
                              if economy.common_alpha else None),
        "v_rule": "xi = inf: v = v_min for every firm" if economy.common_v else None,
        "rho_bar_rule": "pinned at 0 (no role with a common primitive)",
        "fixed": {k: fixed[k] for k in ALL_PARAMETERS
                  if k not in economy.free and not (k == "xi" and economy.common_v)
                  and not (k == "rho_bar" and (economy.common_alpha or economy.common_v))},
    }
    params_path = write_params_file(
        out_dir / f"calibrated_params_{economy.label}.yaml", result.params,
        reference={"w": result.reference.w, "R": result.reference.R, "X_market": result.reference.X_market,
                   "wf_reference": result.reference.wf_reference, "Ed_full_pool": result.reference.Ed_full_pool,
                   "P": result.reference.P, "L": result.reference.L},
        provenance=provenance,
        model={"alpha_cap_factor": factor, "alpha_draw": "truncated"},
        calibration={"variant": "zipf_restricted", "economy": economy.label, "eta_fixed": zcfg.eta,
                     "calibrated": list(economy.free), "targets": list(zcfg.moment_keys),
                     "initial": {k: float(initial[k]) for k in economy.free}},
        restriction=restriction,
        entry={"sunk_entry_cost_F": compute_sunk_entry_cost(eq, beta=sim.parameters.beta, varphi=zcfg.varphi),
               "varphi": zcfg.varphi,
               "note": "post-calibration EMX free-entry backout; welfare-only"},
        fit={
            "success": result.success, "status": result.status, "cost": result.cost, "nfev": result.nfev,
            "n_evals": result.n_evals, "n_replayed": result.n_replayed, "message": result.message,
            "ge_converged": bool(eq.converged), "ge_residual": float(eq.residual),
            "at_bound": list(result.at_bound), "structure": dict(result.structure),
            "moments": {k: {"model": float(m[k]), "target": float(targets[k]), "weight": float(zcfg.weights[k]),
                            "relative_error": float(m[k] / targets[k] - 1.0)} for k in result.moment_keys},
            "untargeted": {k: {"model": float(m[k]), "target": float(targets[k])} for k in UNTARGETED},
            "diagnostics": {"mu_cw_alpha": float(m["mu_cw_alpha"]),
                            "mean_active_count": float(m["mean_active_count"]),
                            "y_anchor": float(eq.y_anchor), **{k: float(v) for k, v in validation.items()}},
        },
        identification={"jacobian_x": np.asarray(result.jacobian_x).tolist(),
                         "jacobian_condition": result.jacobian_condition,
                         "columns": list(economy.free), "moment_keys": list(result.moment_keys)},
    )
    fit = pd.DataFrame([
        {"moment": k, "model": float(m[k]), "target": float(targets[k]),
         "relative_error": float(m[k] / targets[k] - 1.0), "weighted_residual": float(r),
         "weight": float(zcfg.weights[k])}
        for k, r in zip(result.moment_keys, result.residuals, strict=True)
    ])
    fit_path = out_dir / f"calibration_fit_{economy.label}.csv"
    fit.to_csv(fit_path, index=False)
    write_yaml(out_dir / f"calibration_manifest_{economy.label}.yaml", {
        **provenance, "runtime_s": round(time.time() - t0, 1),
        "outputs": {p.name: {"sha256": file_sha256(p)} for p in (params_path, fit_path)},
    })
    logger.info("fit:\n" + fit.to_string(index=False))
    logger.info(f"{economy.label}: " + " ".join(f"{k}={getattr(result.params, k):.6g}" for k in ALL_PARAMETERS)
                + f" v_min={result.params.v_min:.6g} cost={result.cost:.4g} evals={result.n_evals} "
                f"(replayed {result.n_replayed}) success={result.success} ({result.message}); "
                f"structure passed={result.structure['passed']}")
    if result.at_bound:
        logger.warning(f"parameters at a bound: {list(result.at_bound)}")
    if not result.structure["passed"]:
        raise RuntimeError(f"{economy.label}: restriction structure check failed: {result.structure}")
    logger.info(f"wrote {params_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rcfg = load_restricted_config(args.config)
    zcfg = load_zipf_config(rcfg["zipf_config"])
    economy = ECONOMIES[args.economy]
    logger = setup_logger(f"restricted_{economy.label}", rcfg["log_level"])
    t0 = time.time()
    sim = zcfg.simulation
    primitives = load_primitives(zcfg.bundle_dir, H=sim.parameters.H)
    targets = load_zipf_targets(zcfg.bundle_dir, zcfg.zipf_targets)
    kw = model_kwargs(sim)
    factor = kw.pop("alpha_cap_factor")
    kw["market_solver_kwargs"] = dict(kw["market_solver_kwargs"])
    if args.workers is not None:
        kw["market_solver_kwargs"]["parallel_n_jobs"] = int(args.workers)
    workers = kw["market_solver_kwargs"].get("parallel_n_jobs")
    seed = int(args.seed if args.seed is not None else zcfg.seed)
    out_root = Path(args.out_dir).resolve() if args.out_dir else rcfg["out_results_dir"]

    if args.gate0:
        return gate0(zcfg, rcfg, primitives, targets, kw, factor, M=int(args.M or zcfg.M), seed=seed,
                     workers=workers, out_dir=out_root, logger=logger)

    _, base = baseline_values(rcfg["baseline_params"], rcfg["baseline_sha"], primitives)
    spec = rcfg["economies"].get(economy.label) or {}
    fixed = {k: float(getattr(base, k)) for k in ALL_PARAMETERS}
    initial = {**fixed, **{k: float(v) for k, v in (spec.get("initial") or {}).items()}}
    out_dir = out_root / economy.label
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {"baseline_sha256": rcfg["baseline_sha"], "workers": workers, "alpha_cap_factor": factor}
    resilience = rcfg["resilience"]
    logger.info(f"{economy.label}: free {list(economy.free)}; common alpha {economy.common_alpha}, "
                f"common v {economy.common_v}; targets {list(zcfg.moment_keys)}; workers {workers}")

    if args.scan:
        M = int(args.M or rcfg["scan_M"])
        ev = make_evaluator(economy, zcfg=zcfg, primitives=primitives, targets=targets, kw=kw, factor=factor,
                            fixed=fixed, M=M, seed=seed, ckpt_path=out_dir / f"scan_checkpoint_{economy.label}.jsonl",
                            header={**header, "mode": "scan", "initial": initial}, resilience=resilience,
                            logger=logger)
        run_scan(ev, spec.get("scan") or {}, initial, out_csv=out_dir / f"scan_{economy.label}.csv", logger=logger)
        return 0

    if args.initial_from_scan:
        scan = pd.read_csv(out_dir / f"scan_{economy.label}.csv")
        best = scan.loc[scan["cost"].idxmin()]
        initial.update({k: float(best[k]) for k in economy.free})
        logger.info("initial from scan: " + ", ".join(f"{k}={initial[k]:.6g}" for k in economy.free))
    M = int(args.M or zcfg.M)
    max_nfev = int(args.max_nfev or zcfg.max_nfev)
    ev = make_evaluator(economy, zcfg=zcfg, primitives=primitives, targets=targets, kw=kw, factor=factor,
                        fixed=fixed, M=M, seed=seed,
                        ckpt_path=out_dir / f"calibration_checkpoint_{economy.label}.jsonl",
                        header={**header, "mode": "fit", "initial": {k: initial[k] for k in economy.free},
                                "max_nfev": max_nfev, "fd_step": zcfg.fd_step},
                        resilience=resilience, logger=logger)
    result = calibrate_restricted(ev, initial=initial, bounds=zcfg.bounds, ftol=zcfg.ftol, xtol=zcfg.xtol,
                                  max_nfev=max_nfev, fd_step=zcfg.fd_step)
    write_outputs(result, economy=economy, zcfg=zcfg, rcfg=rcfg, primitives=primitives, targets=targets,
                  factor=factor, fixed=fixed, initial=initial, M=M, seed=seed, max_nfev=max_nfev,
                  out_dir=out_dir, logger=logger, t0=t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
