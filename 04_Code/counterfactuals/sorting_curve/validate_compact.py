"""Validate dropping globally inactive numerical padding at production M."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from . import run_remaining_parallel as parallel
from . import run_sorting_curve as runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="out_results/counterfactuals/sorting_curve")
    p.add_argument("--config")
    p.add_argument("--calibration", default="out_results/calib_pooled/calibration_pooled.yaml")
    p.add_argument("--solver-workers", type=int, default=4)
    args = p.parse_args()
    out = Path(args.out_dir)
    prov = json.loads((out / "run_manifest.json").read_text())["provenance"]
    start = time.perf_counter()
    parallel.initialize(args.config, args.calibration, prov, True, args.solver_workers)
    cfg, inputs, params, wf, initial, base, _, _ = parallel._context
    compact, solver_params = parallel.compact_inputs(base, params)
    row = runner.solve_row("baseline_compact", compact, solver_params, cfg, inputs, wf, initial, prov["chi_baseline"])
    reference = next(r for r in runner.cached_rows(cfg.out_results_dir / "counterfactuals", prov) if r["label"] == "baseline")
    with tempfile.TemporaryDirectory() as tmp:
        runner.validate_baseline(row, reference, prov["chi_baseline"], Path(tmp))
        guard = json.loads((Path(tmp) / "validation.json").read_text())
    previous = json.loads((out / "checkpoint_baseline.json").read_text())
    fields = [k for k in previous if k.startswith(("W_", "C_", "delta_", "lambda_"))]
    gaps = {key: abs(row[key]-previous[key])/max(abs(previous[key]), 1e-30) for key in fields}
    if max(gaps.values()) > 1e-6:
        raise RuntimeError("compact baseline differs from full-width baseline")
    # A nonbaseline check also verifies permutations plus the numerical path.
    spec = next(s for s in parallel.remaining_plan(prov) if s[1] == "identity_q0.20_d0")
    partial = parallel.solve_spec(spec)
    previous_partial = json.loads((out / "checkpoint_identity_q0.20_d0.json").read_text())
    partial_gaps = {key: abs(partial[key]-previous_partial[key])/max(abs(previous_partial[key]), 1e-30) for key in fields}
    if max(partial_gaps.values()) > 1e-6:
        raise RuntimeError("compact partial arrangement differs from full-width solve")
    payload = {**guard, "runner_sha256": prov["runner_sha256"], "pool_H": params.H,
               "numerical_H": solver_params.H, "elapsed_seconds": time.perf_counter()-start,
               "full_width_baseline_relative_differences": gaps,
               "full_width_q020_relative_differences": partial_gaps, "solver_workers": args.solver_workers}
    runner.atomic_json(out / "compact_validation.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
