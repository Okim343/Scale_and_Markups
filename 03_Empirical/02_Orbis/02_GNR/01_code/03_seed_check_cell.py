"""Check whether a cell's GNR Step 2 solution depends on the multistart draw.

Step 2 is just-identified (5 moments, 5 parameters), so the GMM estimate sets every moment
to zero and the objective N g'g/2 to 0. With identity weighting the objective can be very
flat, and a multistart can stop at a point that is not a solution. For each seed this
reruns Step 2 (same Step 1, same sample, only the Sobol starting points change) and reports:
  fval            N g'Wg/2 at the polished solution (0 at the GMM estimate)
  J               N g' S^-1 g with S = cov(eta * IV): the efficiently weighted test of
                  the moments at that point; chi2(5) 5% critical value 11.07
  elasticities    mean kelast / lelast / rts and RTS P10 / P90 on the Step-2 sample

Needs 01_run_gnr_cell.py to have run for the cell (reuses its step1_results.json).
Writes <output.intermediary_dir>/cells/<ind2d>/seed_check.json.

    python 03_seed_check_cell.py --cell 42 --seeds 1982 7 123 9999 [--workers 16]
"""

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parent
CHI2_5_95 = 11.0705


def load_driver():
    spec = importlib.util.spec_from_file_location("gnr_cell_driver", CODE / "01_run_gnr_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # also puts 00_GNR_core on sys.path
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1982, 7, 123, 9999])
    ap.add_argument("--config", type=Path, default=CODE / "config_gnr_orbis_es.yaml")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    drv = load_driver()
    import gnr_model as gm
    from gnr_estimate import build_step2_sample, run_step2, step2_arrays
    from gnr_panel import construct_gnr_vars, trim_panel
    from gnr_postprocess import reconstruct_firm_year

    config = yaml.safe_load(args.config.read_text())
    if args.workers:
        config["step2"]["n_workers"] = args.workers
    out = drv.resolve(config["output"]["intermediary_dir"]) / "cells" / str(args.cell)
    logger = drv.setup_logger(f"{args.cell}-seeds", out / "seed_check.log", "INFO")
    s1 = json.loads((out / "step1_results.json").read_text())

    df, _ = drv.load_cell(config, args.cell, logger)
    df = construct_gnr_vars(df, logger, build_materials_alt1=False, id_col=drv.CORE_ID)
    est = trim_panel(df, config, logger)
    sub, counts = build_step2_sample(est, np.asarray(s1["gamma_raw"]), False, logger)
    rt, lrt, p2, lp2, iv, dummies, _ = step2_arrays(sub, s1, logger)
    n, h2size, w_eye = len(rt), int(config["step2"]["h2size"]), np.eye(iv.shape[1])

    results = []
    for seed in args.seeds:
        cfg2 = copy.deepcopy(config["step2"])
        cfg2["seed"] = seed
        t0 = time.time()
        s2 = run_step2(sub, s1, cfg2, logger)
        a = np.asarray(s2["alpha"])
        fval, _, eta = gm.step2_objective_nested(a, rt, lrt, p2, lp2, iv, dummies, h2size,
                                                 w_eye, return_full=True)
        h = eta[:, None] * iv
        g = h.mean(axis=0)
        J = float(n * g @ np.linalg.solve(np.cov(h.T), g))
        s2["sample_counts"] = counts
        fy = reconstruct_firm_year(est.copy(), s1, s2, config["postprocess"]["rts_flag_range"], logger)
        fy = fy[fy["in_step2_sample"]]
        r = {"seed": seed, "fval": float(fval), "J": J, "J_rejects_5pct": J > CHI2_5_95,
             "max_abs_moment_in_se": float(np.max(np.abs(g) / (h.std(axis=0) / np.sqrt(n)))),
             "near_best": s2["local_solution_diagnostics"]["near_best_count"],
             "n_finite": s2["local_solution_diagnostics"]["n_finite"],
             **{k: s2["local_solution_diagnostics"][k]
                for k in ("n_exact_roots", "n_distinct_roots", "root_counts", "n_groups",
                          "chosen_group", "chosen_root")
                if k in s2["local_solution_diagnostics"]},
             "chosen_group_starts": (s2["local_solution_diagnostics"]["root_groups"]
                                     [s2["local_solution_diagnostics"]["chosen_group"]]["starts"]
                                     if "root_groups" in s2["local_solution_diagnostics"] else None),
             "alpha": s2["alpha"],
             "kelast_mean": float(fy["kelast"].mean()), "lelast_mean": float(fy["lelast"].mean()),
             "melast_mean": float(fy["melast"].mean()), "rts_mean": float(fy["rts"].mean()),
             "rts_p10": float(fy["rts"].quantile(0.1)), "rts_p90": float(fy["rts"].quantile(0.9)),
             "runtime_s": round(time.time() - t0, 1)}
        results.append(r)
        logger.info(f"seed {seed}: fval {r['fval']:.3g}, J {J:.2f}, k {r['kelast_mean']:.4f}, "
                    f"l {r['lelast_mean']:.4f}, RTS {r['rts_mean']:.4f}; roots "
                    f"{r.get('n_distinct_roots', '?')}, groups {r.get('n_groups', '?')}, chosen group "
                    f"{r.get('chosen_group', '?')} ({r['chosen_group_starts']} starts) "
                    f"({r['runtime_s']:.0f}s)")
        (out / "seed_check.json").write_text(json.dumps(
            {"cell": args.cell, "n_step2": n, "chi2_5_crit": CHI2_5_95, "results": results}, indent=2))


if __name__ == "__main__":
    main()
