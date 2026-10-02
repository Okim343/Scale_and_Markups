"""Finish a validated sorting-curve run with independent arrangement workers.

The ordinary driver must have written a passing baseline first. Stop its Slurm
job after that checkpoint before launching this driver in the same output folder.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd
from steady_state.__main__ import _params_for_command, _pooled_inputs, load_config
from steady_state.model.pool import draw_pool
from counterfactuals.scalability_sorting.permute import partial_permute
from . import run_sorting_curve as runner
from .figure import write_artifacts


def remaining_plan(prov):
    plan = [(f"identity_q{q:.2f}_d{d}", "identity", q, d)
            for q in (.2, .4, .6, .8) for d in (range(2) if q == .4 else range(1))]
    plan += [(f"reverse_q{q:.2f}_d0", "reverse", q, 0) for q in (.33, .67)]
    if prov["smoke"]:
        plan += [("reverse", "reverse", 0., 0), ("shuffle_smoke", "identity", 1., 0)]
    if prov["rho_crosscheck"]:
        plan += [(f"rho_{rho:.2f}", "rho", rho, 0) for rho in (.45, .70)]
    return [(idx, *spec) for idx, spec in enumerate(plan, start=1)]


def compact_inputs(draw, params):
    """Remove only globally inactive trailing padding after the full CRN draw."""
    used = np.flatnonzero(np.any(draw.active_mask, axis=0))
    width = max(int(used[-1]) + 1, int(np.ceil(params.N)))
    if np.any(draw.active_mask[:, width:]):
        raise AssertionError("attempt to omit active firms")
    compact = dataclasses.replace(draw, **{key: np.ascontiguousarray(getattr(draw, key)[:, :width])
                  for key in ("alpha", "v", "tilde_alpha", "active_mask")})
    for key in ("alpha", "v", "tilde_alpha"):
        if not np.array_equal(getattr(draw, key)[draw.active_mask],
                              getattr(compact, key)[compact.active_mask]):
            raise AssertionError(f"active primitive changed: {key}")
    return compact, dataclasses.replace(params, H=width)


def initialize(config, calibration, prov, compact=False, solver_workers=None):
    global _context
    cfg = load_config(config)
    cfg = dataclasses.replace(cfg, market=dataclasses.replace(cfg.market,
           parallel=dataclasses.replace(cfg.market.parallel, n_jobs=solver_workers or prov["workers"])))
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, calibration)
    if hashlib.sha256(inputs.alpha_support.tobytes()).hexdigest() != prov["support_sha256"]:
        raise RuntimeError("parallel resume alpha support changed")
    if params.y_hat != prov["y_hat"]:
        raise AssertionError("parallel resume y_hat drift")
    base = draw_pool(inputs.alpha_support, xi=params.xi, rho_bar=params.rho_bar,
                     N=params.N, M=prov["M"], H=params.H, rng=prov["seed"], v_min=params.v_min)
    _context = (cfg, inputs, params, wf, initial, base, prov, compact)


def solve_spec(spec):
    idx, label, mode, q, d = spec
    cfg, inputs, params, wf, initial, base, prov, compact = _context
    perm_seed = [int(prov["shuffle_seed"]), idx, d]
    if mode == "rho":
        draw = draw_pool(inputs.alpha_support, xi=params.xi, rho_bar=q,
                         N=params.N, M=prov["M"], H=params.H,
                         rng=prov["seed"], v_min=params.v_min)
        for field in ("alpha", "tilde_alpha", "active_mask"):
            if not np.array_equal(getattr(base, field), getattr(draw, field)):
                raise AssertionError(f"rho CRN drift: {field}")
    else:
        draw = partial_permute(base, mode, q, np.random.default_rng(perm_seed))
    full_H = params.H
    if compact:
        draw, params = compact_inputs(draw, params)
    row = runner.solve_row(label, draw, params, cfg, inputs, wf, initial, prov["chi_baseline"])
    row.update(draw_pool_H=full_H, numerical_solver_H=params.H, solver_workers=cfg.market.parallel.n_jobs)
    row.update(kind="rho" if mode == "rho" else "partial", base_mode=mode,
               q=None if mode == "rho" else q, perm_seed=json.dumps(perm_seed),
               rho_bar=q if mode == "rho" else float(params.rho_bar))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="out_results/counterfactuals/sorting_curve")
    p.add_argument("--config")
    p.add_argument("--calibration", default="out_results/calib_pooled/calibration_pooled.yaml")
    p.add_argument("--arrangement-workers", type=int, default=4)
    p.add_argument("--fig-dir")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--solver-workers", type=int)
    args = p.parse_args()
    out = Path(args.out_dir)
    prov = json.loads((out / "run_manifest.json").read_text())["provenance"]
    hashes = {"config_sha256": Path(args.config or "steady_state/config.yaml"),
              "calibration_sha256": Path(args.calibration), "runner_sha256": Path(runner.__file__)}
    for key, path in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != prov[key]:
            raise RuntimeError(f"parallel resume input changed: {key}")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    solver_workers = args.solver_workers or prov["workers"]
    if solver_workers < 1 or args.arrangement_workers < 1:
        raise ValueError("worker counts must be positive")
    if args.arrangement_workers * solver_workers > cpus:
        raise RuntimeError("arrangement workers times solver workers exceeds allocated CPUs")
    if args.compact:
        validation = json.loads((out / "compact_validation.json").read_text())
        if not validation["passed"] or validation["runner_sha256"] != prov["runner_sha256"]:
            raise RuntimeError("compact solver baseline validation is missing or failed")
        prov = {**prov, "numerical_optimization": {"globally_inactive_trailing_columns_omitted": True,
                "full_pool_H": prov["calibrated_params"]["H"], "solver_workers": solver_workers,
                "arrangement_workers": args.arrangement_workers, "validation_file": "compact_validation.json"}}
    cfg = load_config(args.config)
    root = cfg.out_results_dir / "counterfactuals"
    rows = [] if prov["smoke"] else [r for r in runner.cached_rows(root, prov) if r["label"] != "baseline"]
    baseline = json.loads((out / "checkpoint_baseline.json").read_text())
    if not prov["smoke"]:
        cache_base = next(r for r in runner.cached_rows(root, prov) if r["label"] == "baseline")
        runner.validate_baseline(baseline, cache_base, prov["chi_baseline"], out)
    rows.append(baseline)
    pending = []
    plan = remaining_plan(prov)
    for spec in plan:
        path = out / f"checkpoint_{spec[1]}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            pending.append(spec)
    fig_dir = Path(args.fig_dir) if args.fig_dir else (out / "figures" if prov["smoke"] else cfg.out_figs_dir / "counterfactuals")

    def publish(complete):
        frame = pd.DataFrame(rows)
        tmp = out / "sorting_curve_economies.tmp.parquet"
        frame.to_parquet(tmp, index=False)
        tmp.replace(out / "sorting_curve_economies.parquet")
        write_artifacts(frame, prov, out, fig_dir, complete=complete)

    publish(not pending)
    with ProcessPoolExecutor(max_workers=args.arrangement_workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.calibration, prov, args.compact, solver_workers)) as pool:
        futures = {pool.submit(solve_spec, spec): spec for spec in pending}
        for future in as_completed(futures):
            row = future.result()
            runner.atomic_json(out / f"checkpoint_{row['label']}.json", row)
            rows.append(row)
            publish(len(rows) == len(plan) + 1 + (0 if prov["smoke"] else 11))
            print(f"[{row['label']}] CHECKPOINT lambda_total={row['lambda_total']:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
