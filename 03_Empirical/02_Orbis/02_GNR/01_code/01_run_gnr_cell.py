"""Estimate GNR for one Orbis Spain NAICS2 cell (Hubmer et al. 2025: within country x NAICS2).

Reads the cleaned GNR panel (config input.source), keeps one `ind2d` cell, and runs the shared
estimator in 03_Empirical/00_GNR_core: logs and consecutive-year lags, est_gnr.m trimming
within the cell, Step 1 NLS, Step 2 nested GMM (no industry dummies: one cell), and the
firm-year reconstruction.

Writes to <output.intermediary_dir>/cells/<ind2d>/ (gitignored):
    step1_results.json, step2_results.json   parameters + diagnostics, with a fingerprint
    firm_year_rts.parquet                    trimmed rows; Step-2 objects NaN outside Step 2
    cell_summary.json                        counts, runtimes, elasticity / RTS moments
    run.log

Restartable: a finished cell is skipped, and a saved step is reused, when its fingerprint
(cell, content of the cell's input rows, mapping/panel/trimming/step settings) is unchanged.
The fingerprint does not depend on file paths or timestamps, so results move between machines.
--force reruns.

    python 01_run_gnr_cell.py --cell 21 [--config config_gnr_orbis_es.yaml] [--workers 8] [--force]
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(__file__).resolve().parent
PROJECT_ROOT = CODE.parents[3]
GNR_CORE = PROJECT_ROOT / "03_Empirical" / "00_GNR_core"
if str(GNR_CORE) not in sys.path:
    sys.path.insert(0, str(GNR_CORE))

from gnr_estimate import build_step2_sample, run_step1, run_step2  # noqa: E402
from gnr_panel import construct_gnr_vars, trim_panel  # noqa: E402
from gnr_postprocess import reconstruct_firm_year  # noqa: E402

# The core keeps the Compustat id name internally; it is renamed back to bvdid on output.
CORE_ID = "gvkey"
LEVELS = {"revenue": "rev_lvl", "capital": "cap_lvl", "materials": "mat_lvl",
          "labor": "lab_lvl", "wage_bill": "wagebill_lvl"}
OUT_COLS = ["year", "ind2d", "eps", "melast", "kelast", "lelast", "omega", "lomega", "eta",
            "rts", "alpha_gross_output", "in_step2_sample", "rts_outlier_flag",
            "s", "r", "k", "m", "l"]


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_logger(cell, log_path, level):
    logger = logging.getLogger(f"GNR-{cell}")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def data_hash(df):
    """Content hash of a cell's input rows (row order and float width do not matter)."""
    d = df.sort_values([CORE_ID, "year"]).reset_index(drop=True)
    d = d.astype({c: "float64" for c in LEVELS.values()})
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()[:16]


def fingerprint(config, cell, dhash, blocks):
    """Hash of everything a step depends on (worker count excluded: it does not change results)."""
    def strip(b):
        return {k: v for k, v in b.items() if k != "n_workers"} if isinstance(b, dict) else b
    payload = {"cell": cell, "data": dhash, **{b: strip(config[b]) for b in blocks}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_json(path, fp):
    """Saved step results if present and computed with the same fingerprint."""
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("fingerprint") == fp:
            return d
    return None


def load_cell(config, cell, logger):
    mp, pan = config["mapping"], config["panel"]
    src = resolve(config["input"]["source"])
    cols = [mp["id"], mp["year"], mp["ind"]] + [mp[k] for k in LEVELS]
    df = pd.read_parquet(src, columns=cols, filters=[(mp["ind"], "==", cell)])
    df = df.rename(columns={mp["id"]: CORE_ID, mp["year"]: "year", mp["ind"]: "ind2d",
                            **{mp[k]: v for k, v in LEVELS.items()}})
    df = df[df["year"].between(int(pan["start_year"]), int(pan["end_year"]))]
    df["year"] = df["year"].astype(int)
    df["ind2d"] = df["ind2d"].astype(int)
    df[CORE_ID] = df[CORE_ID].astype(str)
    if df.duplicated([CORE_ID, "year"]).any():
        raise ValueError("duplicate (firm, year) rows in the cell; fix the cleaning stage")
    logger.info(f"cell {cell}: {len(df):,} rows, {df[CORE_ID].nunique():,} firms, "
                f"years {df['year'].min()}-{df['year'].max()} from {src.name}")
    return df, src


def moments(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = np.percentile(x, [10, 50, 90])
    return {"mean": float(x.mean()), "sd": float(x.std()), "p10": float(q[0]),
            "p50": float(q[1]), "p90": float(q[2]), "share_neg": float((x < 0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=int, required=True, help="ind2d code, e.g. 21 or 31 (= 31-33)")
    ap.add_argument("--config", type=Path, default=CODE / "config_gnr_orbis_es.yaml")
    ap.add_argument("--workers", type=int, default=None, help="override step1/step2 n_workers")
    ap.add_argument("--force", action="store_true", help="rerun even if results are cached")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    cell = args.cell
    if cell not in config["cells"]["list"]:
        raise ValueError(f"cell {cell} is not in cells.list {config['cells']['list']}")
    if args.workers:
        config["step1"]["n_workers"] = config["step2"]["n_workers"] = args.workers
    out = resolve(config["output"]["intermediary_dir"]) / "cells" / str(cell)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cell, out / "run.log", config.get("log_level", "INFO"))

    t_start = time.time()
    df, _ = load_cell(config, cell, logger)
    dhash = data_hash(df)
    base = ["mapping", "panel", "trimming"]
    fp1 = fingerprint(config, cell, dhash, base + ["step1"])
    fp2 = fingerprint(config, cell, dhash, base + ["step1", "step2"])
    summary_path = out / "cell_summary.json"
    if not args.force and (out / "firm_year_rts.parquet").exists() and load_json(summary_path, fp2):
        logger.info(f"cell {cell}: finished with the same inputs and settings; skipping (--force reruns)")
        return
    n_cell = len(df)
    df = construct_gnr_vars(df, logger, build_materials_alt1=bool(config["panel"]["build_materials_alt1"]),
                            id_col=CORE_ID)
    n_lag = int(df["lag_is_consecutive"].sum())
    est = trim_panel(df, config, logger)
    years_kept = sorted(est["year"].unique().tolist())

    # --- Step 1 -------------------------------------------------------------------
    s1_path = out / "step1_results.json"
    s1 = None if args.force else load_json(s1_path, fp1)
    if s1:
        logger.info("=== Step 1: reusing saved results (same fingerprint) ===")
        t1 = s1["runtime_s"]
    else:
        logger.info("=== Step 1: NLS share regression ===")
        t0 = time.time()
        s1 = run_step1(est, config["step1"], logger)
        t1 = time.time() - t0
        s1.update(fingerprint=fp1, runtime_s=t1, cell=cell)
        s1_path.write_text(json.dumps(s1, indent=2))

    # --- Step 2 -------------------------------------------------------------------
    require_wages = bool(config["panel"].get("require_wages", False))
    s2_path = out / "step2_results.json"
    s2 = None if args.force else load_json(s2_path, fp2)
    if s2:
        logger.info("=== Step 2: reusing saved results (same fingerprint) ===")
        t2 = s2["runtime_s"]
    else:
        logger.info("=== Step 2: nested GMM ===")
        t0 = time.time()
        sub, counts = build_step2_sample(est, np.asarray(s1["gamma_raw"]), require_wages, logger)
        s2 = run_step2(sub, s1, config["step2"], logger)
        t2 = time.time() - t0
        s2.update(sample_counts=counts, require_wages=require_wages, fingerprint=fp2,
                  runtime_s=t2, cell=cell)
        s2_path.write_text(json.dumps(s2, indent=2))

    # --- Firm-year objects ------------------------------------------------------------
    est = reconstruct_firm_year(est, s1, s2, config["postprocess"]["rts_flag_range"], logger)
    fy = est[[CORE_ID] + OUT_COLS].rename(columns={CORE_ID: "bvdid"})
    fy.to_parquet(out / "firm_year_rts.parquet", index=False)

    s2s = fy[fy["in_step2_sample"]]
    summary = {
        "cell": cell, "fingerprint": fp2, "data_hash": dhash, "pipeline_version": config.get("pipeline_version"),
        "counts": {"cell_rows": n_cell, "with_consecutive_lag": n_lag, "trimmed": int(len(est)),
                   "step2": int(len(s2s)), "firms_step2": int(s2s["bvdid"].nunique()),
                   "years_kept": years_kept},
        "runtime_s": {"step1": round(t1, 1), "step2": round(t2, 1),
                      "total": round(time.time() - t_start, 1)},
        "step1": {k: s1[k] for k in ["fval", "fval_multistart_best", "n_starts",
                                    "n_feasible_solutions", "bigeps"]},
        "step2": {"fval": s2["fval"], "fval_multistart_best": s2["fval_multistart_best"],
                  **{k: v for k, v in s2["local_solution_diagnostics"].items()
                     if k in ("n_finite", "near_best_count", "n_exact_roots", "n_distinct_roots",
                              "root_counts", "J_estimate", "root_found", "selection_rule",
                              "group_tol", "chosen_root", "n_groups", "chosen_group",
                              "root_groups")},
                  "search": s2.get("search", "extended")},
        "materials_share": moments(np.exp(s2s["s"])),
        "melast": moments(s2s["melast"]), "lelast": moments(s2s["lelast"]),
        "kelast": moments(s2s["kelast"]), "rts": moments(s2s["rts"]),
        "rts_outliers": int(fy["rts_outlier_flag"].sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"cell {cell} done in {summary['runtime_s']['total']:.0f}s: step-2 rows "
        f"{summary['counts']['step2']:,}; mean elasticities m {summary['melast']['mean']:.3f}, "
        f"l {summary['lelast']['mean']:.3f}, k {summary['kelast']['mean']:.3f}; "
        f"RTS mean {summary['rts']['mean']:.3f} (P10 {summary['rts']['p10']:.3f}, "
        f"P90 {summary['rts']['p90']:.3f})")


if __name__ == "__main__":
    main()
