"""Two diagnostics of GNR Step 2 for one cell: stalls vs local minima, and exact root solving.

Diagnostic 1 (are the multistart "local minima" real?). Runs the est_gnr.m-style multistart
(search: matlab, one seed) keeping every endpoint, and evaluates at each endpoint, and at the
polished solutions of earlier MATLAB / Python runs, the exact gradient of the objective
Q = N g'g/2 and its Hessian (central differences of the exact gradient):
  grad_ratio   |grad Q|_inf over fminunc's own stopping threshold 1e-6 * max(1, |grad Q(x0)|_inf)
               (> 1: fminunc's first-order test is not met, so the point is not stationary)
  hess_pd      Hessian positive definite (a strict local minimum if also stationary)
  newton_frac  predicted Newton decrease (1/2) grad' H^-1 grad over Q (large: a Newton step
               would remove most of the objective, so the point is far from a minimum)
  J            efficiently weighted moment test, chi2(5) 5% critical value 11.07

Diagnostic 2 (does the moment system have a root, and is it unique?). For each seed's start
set, runs search: exact (standardized k / l, Levenberg-Marquardt on g = 0 with the exact
Jacobian) and reports how many endpoints are exact roots, how many distinct roots, and the
elasticities at each.

Needs 01_run_gnr_cell.py to have run for the cell (reuses its step1_results.json).
Writes <output.intermediary_dir>/cells/<ind2d>/step2_diagnostics.{json,md}.

    python 05_step2_diagnostics.py --cell 42 --seeds 1982 7 123 9999 --workers 64 \
        [--matlab-diag-dir DIR]   # DIR/seed_<s>/gnr_diag.json from the MATLAB runs
"""

import argparse
import copy
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parent
CHI2_5_95 = 11.0705
FMINUNC_OPT_TOL = 1e-6


def load_driver():
    spec = importlib.util.spec_from_file_location("gnr_cell_driver", CODE / "01_run_gnr_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # also puts 00_GNR_core on sys.path
    return mod


def _point_metrics(args):
    """Gradient / Hessian / J at one point (runs in a worker; data in gnr_model._WORKER)."""
    import gnr_model as gm

    a, x0 = args
    rt, lrt, p2, lp2, iv, dum, h2size = gm._WORKER["payload"]
    n = len(rt)

    def grad(x):
        g, jm, _, _ = gm.step2_moments_jacobian(x, rt, lrt, p2, lp2, iv, dum, h2size)
        return n * jm.T @ g, g

    gq, g = grad(a)
    q = float(n * g @ g / 2.0)
    hess = np.empty((5, 5))
    for j in range(5):
        h = 1e-5 * max(1.0, abs(a[j]))
        e = np.zeros(5)
        e[j] = h
        hess[:, j] = (grad(a + e)[0] - grad(a - e)[0]) / (2 * h)
    hess = (hess + hess.T) / 2
    eig = np.linalg.eigvalsh(hess)
    threshold = None
    if x0 is not None:
        threshold = FMINUNC_OPT_TOL * max(1.0, float(np.max(np.abs(grad(x0)[0]))))
    _, _, eta, _ = gm.step2_moments_jacobian(a, rt, lrt, p2, lp2, iv, dum, h2size)
    hm = eta[:, None] * iv
    j_eff = float(n * g @ np.linalg.solve(np.cov(hm.T), g))
    try:
        newton = float(0.5 * gq @ np.linalg.solve(hess, gq))
    except np.linalg.LinAlgError:
        newton = np.nan
    return {"Q": q, "grad_inf": float(np.max(np.abs(gq))),
            "grad_ratio": None if threshold is None else float(np.max(np.abs(gq)) / threshold),
            "hess_min_eig": float(eig[0]), "hess_pd": bool(eig[0] > 0),
            "newton_frac": newton / q if q > 0 else np.nan, "J": j_eff}


def elasticities(drv, est, s1, a, rt, lrt, p2, lp2, iv, dum, h2size, n_step2, inds, logger):
    import gnr_model as gm
    from gnr_postprocess import reconstruct_firm_year

    _, delta, _ = gm.step2_objective_nested(np.asarray(a), rt, lrt, p2, lp2, iv, dum, h2size,
                                            np.eye(5), return_full=True)
    s2 = {"alpha": list(a), "delta": delta.tolist(), "h2size": h2size,
          "sample_counts": {"n_step2": n_step2},
          "delta_layout": {"ind_dummy_categories": inds}}
    fy = reconstruct_firm_year(est.copy(), s1, s2, [0.0, 3.0], logger)
    fy = fy[fy["in_step2_sample"]]
    return {"kelast": float(fy["kelast"].mean()), "lelast": float(fy["lelast"].mean()),
            "rts": float(fy["rts"].mean()), "rts_p10": float(fy["rts"].quantile(0.1)),
            "rts_p90": float(fy["rts"].quantile(0.9))}


def summarize(rows, key):
    v = np.asarray([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])])
    return {"min": float(v.min()), "p50": float(np.median(v)), "max": float(v.max())} if len(v) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1982, 7, 123, 9999])
    ap.add_argument("--config", type=Path, default=CODE / "config_gnr_orbis_es.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--matlab-diag-dir", type=Path, default=None)
    args = ap.parse_args()

    drv = load_driver()
    import gnr_model as gm
    from gnr_estimate import build_step2_sample, run_step2, step2_arrays
    from gnr_panel import construct_gnr_vars, trim_panel

    config = yaml.safe_load(args.config.read_text())
    out = drv.resolve(config["output"]["intermediary_dir"]) / "cells" / str(args.cell)
    logger = drv.setup_logger(f"{args.cell}-diag", out / "step2_diagnostics.log", "INFO")
    s1 = json.loads((out / "step1_results.json").read_text())
    df, _ = drv.load_cell(config, args.cell, logger)
    df = construct_gnr_vars(df, logger, build_materials_alt1=False, id_col=drv.CORE_ID)
    est = trim_panel(df, config, logger)
    sub, counts = build_step2_sample(est, np.asarray(s1["gamma_raw"]), False, logger)
    rt, lrt, p2, lp2, iv, dum, inds = step2_arrays(sub, s1, logger)
    h2size = int(config["step2"]["h2size"])
    payload = (rt, lrt, p2, lp2, iv, dum, h2size)
    result = {"cell": args.cell, "n_step2": len(rt), "seeds": args.seeds}

    # ---------------- Diagnostic 1 ----------------
    t0 = time.time()
    cfg2 = copy.deepcopy(config["step2"])
    cfg2["seed"] = args.seeds[0]
    x_design = np.column_stack([np.ones(len(rt)), -p2])
    linguess = np.linalg.lstsq(x_design, rt, rcond=None)[0][1:]
    sobol = gm.gensobol(int(cfg2["n_sobol"]), gm.P2SIZE, *map(float, cfg2["sobol_bounds"]),
                        seed=int(cfg2["seed"]))
    starts = np.vstack([linguess, np.full(gm.P2SIZE, 1e-4), sobol[: int(cfg2["n_starts"])]])
    ends = gm.parallel_multistart(gm._step2_solve_matlab, starts,
                                  (rt, lrt, p2, lp2, iv, dum, h2size, np.eye(5)), args.workers)
    jobs = [(np.asarray(x), x0) for (f, x, c), x0 in zip(ends, starts) if np.isfinite(f)]
    conv = [c for (f, x, c) in ends if np.isfinite(f)]
    named = []
    if args.matlab_diag_dir:
        for sd in args.seeds:
            pth = args.matlab_diag_dir / f"seed_{sd}" / "gnr_diag.json"
            if pth.exists():
                named.append((f"MATLAB polished, seed {sd}", np.asarray(json.loads(pth.read_text())["alpha"], dtype=float)))
    sc = out / "seed_check.json"
    if sc.exists():
        for r in json.loads(sc.read_text())["results"]:
            named.append((f"Python matlab-mode polished, seed {r['seed']}", np.asarray(r["alpha"], dtype=float)))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=gm._init_worker,
                             initargs=(payload,)) as pool:
        end_metrics = list(pool.map(_point_metrics, jobs))
        named_metrics = list(pool.map(_point_metrics, [(a, None) for _, a in named]))
    for m, c in zip(end_metrics, conv):
        m["fminunc_converged"] = bool(c)
    stationary = [m for m in end_metrics if m["grad_ratio"] <= 1.0]
    true_min = [m for m in stationary if m["hess_pd"]]
    result["diagnostic_1"] = {
        "seed": args.seeds[0], "n_endpoints": len(end_metrics),
        "n_fminunc_converged": int(sum(conv)),
        "n_stationary_by_fminunc_test": len(stationary),
        "n_stationary_and_hessian_pd": len(true_min),
        "n_hessian_pd": int(sum(m["hess_pd"] for m in end_metrics)),
        "grad_ratio": summarize(end_metrics, "grad_ratio"),
        "newton_frac": summarize(end_metrics, "newton_frac"),
        "Q": summarize(end_metrics, "Q"), "J": summarize(end_metrics, "J"),
        "best_endpoints": sorted(end_metrics, key=lambda m: m["Q"])[:5],
        "polished_points": [{"name": nm, **m} for (nm, _), m in zip(named, named_metrics)],
        "runtime_s": round(time.time() - t0, 1),
    }
    logger.info(f"diagnostic 1: {len(end_metrics)} endpoints, {len(stationary)} pass fminunc's "
                f"gradient test, {len(true_min)} are stationary with PD Hessian "
                f"({time.time() - t0:.0f}s)")
    (out / "step2_diagnostics.json").write_text(json.dumps(result, indent=2))

    # ---------------- Diagnostic 2 ----------------
    result["diagnostic_2"] = []
    for sd in args.seeds:
        t0 = time.time()
        cfg2 = copy.deepcopy(config["step2"])
        cfg2.update(seed=sd, search="exact", n_workers=args.workers)
        s2 = run_step2(sub, s1, cfg2, logger)
        d = s2["local_solution_diagnostics"]
        roots = []
        for a, cnt in zip(d["root_alphas"], d["root_counts"]):
            roots.append({"count": cnt, "alpha": a,
                          **elasticities(drv, est, s1, a, rt, lrt, p2, lp2, iv, dum, h2size,
                                         counts["n_step2"], inds, logger)})
        rec = {"seed": sd, "n_finite": d["n_finite"], "n_exact_roots": d["n_exact_roots"],
               "n_distinct_roots": d["n_distinct_roots"], "J_min": d["J_min"],
               "J_p50": d["J_p50"], "roots": roots, "runtime_s": round(time.time() - t0, 1)}
        if not roots:
            rec["best_endpoint"] = {"alpha": s2["alpha"], **elasticities(
                drv, est, s1, s2["alpha"], rt, lrt, p2, lp2, iv, dum, h2size,
                counts["n_step2"], inds, logger)}
        result["diagnostic_2"].append(rec)
        logger.info(f"diagnostic 2, seed {sd}: {d['n_exact_roots']} exact roots of {d['n_finite']}, "
                    f"{d['n_distinct_roots']} distinct; "
                    + "; ".join(f"root x{r['count']}: k {r['kelast']:.4f} l {r['lelast']:.4f} "
                                f"RTS {r['rts']:.4f}" for r in roots)
                    + f" ({time.time() - t0:.0f}s)")
        (out / "step2_diagnostics.json").write_text(json.dumps(result, indent=2))

    # ---------------- Summary ----------------
    d1 = result["diagnostic_1"]
    lines = [f"# GNR Step 2 diagnostics, cell {args.cell} ({len(rt):,} Step-2 rows)", "",
             "## 1. Multistart endpoints (search: matlab, seed "
             f"{d1['seed']})", "",
             f"- endpoints: {d1['n_endpoints']}; fminunc-converged: {d1['n_fminunc_converged']}",
             f"- pass fminunc's own gradient test (grad_ratio <= 1): {d1['n_stationary_by_fminunc_test']}",
             f"- stationary with positive-definite Hessian (true local minima): {d1['n_stationary_and_hessian_pd']}",
             f"- grad_ratio min / median / max: {d1['grad_ratio']}",
             f"- Newton-predicted decrease as share of Q, min / median / max: {d1['newton_frac']}",
             "", "Polished solutions of earlier runs:", "",
             "| point | Q | grad_inf | Hessian PD | Newton frac | J |", "|---|---|---|---|---|---|"]
    for m in d1["polished_points"]:
        lines.append(f"| {m['name']} | {m['Q']:.4g} | {m['grad_inf']:.3g} | {m['hess_pd']} | "
                     f"{m['newton_frac']:.3g} | {m['J']:.2f} |")
    lines += ["", "## 2. Exact root solving (search: exact)", "",
              "| seed | exact roots / endpoints | distinct roots | root: starts, capital, labor, RTS mean (P10-P90) |",
              "|---|---|---|---|"]
    for r in result["diagnostic_2"]:
        desc = "; ".join(f"{x['count']} starts: k {x['kelast']:.3f}, l {x['lelast']:.3f}, "
                         f"RTS {x['rts']:.3f} ({x['rts_p10']:.2f}-{x['rts_p90']:.2f})" for x in r["roots"])
        if not r["roots"]:
            b = r["best_endpoint"]
            desc = f"no root; best J {r['J_min']:.3g}: RTS {b['rts']:.3f}"
        lines.append(f"| {r['seed']} | {r['n_exact_roots']} / {r['n_finite']} | {r['n_distinct_roots']} | {desc} |")
    (out / "step2_diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
