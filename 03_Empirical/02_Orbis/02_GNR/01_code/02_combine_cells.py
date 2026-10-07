"""Stack finished per-cell GNR results into one Orbis Spain firm-year panel.

Reads  <output.intermediary_dir>/cells/<ind2d>/{firm_year_rts.parquet, cell_summary.json}
Writes <output.intermediary_dir>/firm_year_rts.parquet   all finished cells (gitignored)
       <output.intermediary_dir>/cell_diagnostics.csv    one row per cell
       <output.intermediary_dir>/gnr_cells_report.md     per-cell table vs Hubmer et al. Table A.5,
                                                         plus the alternative step-2 root groups
       <output.outdata_dir>/{firm_year_rts.parquet, gnr_cells_report.md, cell_diagnostics.csv,
                             manifest.yaml}              the final GNR estimates (gitignored), only
                                                         when every cell is finished with the
                                                         current config (fingerprints match)

Missing cells are listed, not an error, so this can run while the array is incomplete.

    python 02_combine_cells.py [--config config_gnr_orbis_es.yaml]
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(__file__).resolve().parent
PROJECT_ROOT = CODE.parents[3]
SECTOR_LABEL = {31: "31-33", 44: "44-45", 48: "48-49"}
# Hubmer et al. Table A.5, Spain (all NAICS2 cells pooled; firm-year moments).
HUBMER_ES = {"melast": (0.45, 0.16, 0.43, 0.76), "lelast": (0.46, 0.21, 0.47, 0.69),
             "kelast": (0.04, 0.01, 0.04, 0.07), "rts": (0.95, 0.85, 0.95, 1.00)}


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_driver():
    spec = importlib.util.spec_from_file_location("gnr_cell_driver", CODE / "01_run_gnr_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_state():
    def git(*a):
        return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD") or None, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
                "code_dirty": bool(git("status", "--porcelain", "--", "03_Empirical/00_GNR_core",
                                       "03_Empirical/02_Orbis/02_GNR/01_code"))}
    except OSError:
        return {"commit": None, "branch": None, "code_dirty": None}


def write_outdata(config, config_path, inter, panel, rows, fps):
    """Copy the combined panel and reports to output.outdata_dir with a manifest."""
    out = resolve(config["output"]["outdata_dir"])
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "firm_year_rts.parquet", index=False)
    for f in ("gnr_cells_report.md", "cell_diagnostics.csv"):
        shutil.copy2(inter / f, out / f)
    lo, hi = config["combine"]["window"]
    w = panel[panel["in_step2_sample"] & panel["year"].between(lo, hi)]
    manifest = {
        "description": "Orbis Spain GNR gross-output production function, estimated per NAICS2 "
                       "cell (ind2d); firm-year elasticities and returns to scale.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **git_state(),
        "pipeline_version": config.get("pipeline_version"),
        "config": (str(config_path.resolve().relative_to(PROJECT_ROOT))
                   if config_path.resolve().is_relative_to(PROJECT_ROOT) else str(config_path)),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input": config["input"]["source"],
        "cells": [int(c) for c in config["cells"]["list"]],
        "cell_fingerprints": {str(c): fp for c, fp in fps.items()},
        "step2_search": config["step2"].get("search"),
        "root_selection": config["step2"].get("root_selection"),
        "rows": {"total": int(len(panel)), "in_step2_sample": int(panel["in_step2_sample"].sum()),
                 f"in_step2_sample_{lo}_{hi}": int(len(w)), "firms_step2": int(w["bvdid"].nunique())},
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "columns": {
            "bvdid": "Orbis firm id", "year": "fiscal year", "ind2d": "NAICS2 cell (31 = 31-33, "
            "44 = 44-45, 48 = 48-49)", "s, r, k, m, l": "logs: materials share of revenue, "
            "deflated revenue, capital (tfas), materials, labor (wage bill)",
            "eps, melast": "step 1 residual and materials elasticity (all trimmed rows)",
            "kelast, lelast, omega, lomega, eta": "step 2 objects (NaN outside the step 2 sample)",
            "rts = alpha_gross_output": "melast + lelast + kelast",
            "in_step2_sample": "row has a consecutive lag and a feasible lagged share polynomial",
            "rts_outlier_flag": f"rts outside {config['postprocess']['rts_flag_range']} "
                                "(flagged, not dropped)"},
        "pooled_step2_moments": {v: {"mean": float(w[v].mean()), "p10": float(w[v].quantile(0.1)),
                                     "p50": float(w[v].median()), "p90": float(w[v].quantile(0.9))}
                                 for v in ["melast", "lelast", "kelast", "rts"]},
        "license": "Firm-level derivative of WRDS Orbis data: never commit or publish.",
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    print(f"\nfinal estimates written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CODE / "config_gnr_orbis_es.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    inter = resolve(config["output"]["intermediary_dir"])
    lo, hi = config["combine"]["window"]

    frames, rows, missing, alt, fps = [], [], [], [], {}
    for cell in config["cells"]["list"]:
        d = inter / "cells" / str(cell)
        if not ((d / "firm_year_rts.parquet").exists() and (d / "cell_summary.json").exists()):
            missing.append(cell)
            continue
        fy = pd.read_parquet(d / "firm_year_rts.parquet")
        frames.append(fy)
        sm = json.loads((d / "cell_summary.json").read_text())
        fps[cell] = sm["fingerprint"]
        w = fy[fy["in_step2_sample"] & fy["year"].between(lo, hi)]
        row = {"sector": SECTOR_LABEL.get(cell, str(cell)), "step2_rows": len(w),
               "firms": w["bvdid"].nunique(), "runtime_min": sm["runtime_s"]["total"] / 60,
               "s1_fval": sm["step1"]["fval"], "s2_fval": sm["step2"]["fval"],
               "s2_near_best": f"{sm['step2']['near_best_count']}/{sm['step2']['n_finite']}"}
        for v in ["melast", "lelast", "kelast", "rts"]:
            row[f"{v}_mean"] = w[v].mean()
            row[f"{v}_p10"] = w[v].quantile(0.1)
            row[f"{v}_p90"] = w[v].quantile(0.9)
        st2 = sm["step2"]
        row["root_found"] = st2.get("root_found")
        row["n_roots"] = st2.get("n_distinct_roots")
        row["n_groups"] = st2.get("n_groups")
        if st2.get("root_groups"):
            starts = st2["root_groups"][st2["chosen_group"]]["starts"]
        elif st2.get("root_counts"):
            starts = max(st2["root_counts"])  # rule most_starts: the root reached most often
        else:
            starts = None
        row["root_starts"] = f"{starts}/{st2['n_finite']}" if starts is not None else ""
        if len(st2.get("root_groups", [])) > 1:
            for gi, g in enumerate(st2["root_groups"]):
                e = g["elasticities"]
                alt.append({"sector": row["sector"], "group": gi,
                            "chosen": "yes" if gi == st2["chosen_group"] else "",
                            "roots": ",".join(map(str, g["members"])), "starts": g["starts"],
                            **{f"{v}_mean": e[v]["mean"] for v in ["melast", "lelast", "kelast"]},
                            "rts_mean": e["rts"]["mean"], "rts_p10": e["rts"]["p10"],
                            "rts_p90": e["rts"]["p90"], "share_l_neg": e["lelast"]["share_neg"],
                            "share_k_neg": e["kelast"]["share_neg"]})
        row["J_est"] = st2.get("J_estimate", np.nan)
        sc_path = d / "seed_check.json"
        if sc_path.exists():
            sc = json.loads(sc_path.read_text())["results"]
            rts = [r["rts_mean"] for r in sc]
            row["seeds"] = len(sc)
            row["seed_rts_range"] = max(rts) - min(rts)
            row["seed_roots"] = ",".join(str(r.get("n_distinct_roots", "?")) for r in sc)
            row["seed_groups"] = ",".join(str(r.get("n_groups", "?")) for r in sc)
        row["share_m_neg"] = (w["melast"] < 0).mean()
        row["share_l_neg"] = (w["lelast"] < 0).mean()
        row["share_k_neg"] = (w["kelast"] < 0).mean()
        rows.append(row)

    if not frames:
        print(f"no finished cells under {inter / 'cells'}")
        return
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(inter / "firm_year_rts.parquet", index=False)
    diag = pd.DataFrame(rows)
    diag.to_csv(inter / "cell_diagnostics.csv", index=False)

    w = panel[panel["in_step2_sample"] & panel["year"].between(lo, hi)]
    def fmt(x):
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "NO"
        if isinstance(x, float):
            return f"{x:.2g}" if 0 < abs(x) < 1e-3 else f"{x:,.3f}"
        return f"{x:,}" if isinstance(x, (int, np.integer)) else str(x)
    cols = ["sector", "step2_rows", "firms", "runtime_min", "root_found", "n_roots", "n_groups",
            "root_starts", "J_est", "seeds", "seed_rts_range", "seed_roots", "seed_groups",
            "melast_mean",
            "lelast_mean", "kelast_mean", "rts_mean", "rts_p10", "rts_p90", "share_l_neg",
            "share_k_neg"]
    lines = ["# Orbis Spain GNR by NAICS2 cell", "",
             f"Step-2 firm-years, {lo}-{hi}. Finished cells: {len(rows)} of "
             f"{len(config['cells']['list'])}" + (f"; missing: {missing}" if missing else "") + ".", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    lines += ["", "Pooled over finished cells vs Hubmer et al. Table A.5, Spain (all cells):", "",
              "| object | ours mean | ours P10 | ours P50 | ours P90 | Hubmer mean | P10 | P50 | P90 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v, ref in HUBMER_ES.items():
        q = w[v].quantile([0.1, 0.5, 0.9]).tolist()
        lines.append(f"| {v} | {w[v].mean():.3f} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | "
                     + " | ".join(f"{x:.2f}" for x in ref) + " |")
    if alt:
        acols = ["sector", "group", "chosen", "roots", "starts", "melast_mean", "lelast_mean",
                 "kelast_mean", "rts_mean", "rts_p10", "rts_p90", "share_l_neg", "share_k_neg"]
        lines += ["", "Step-2 root groups in cells with more than one economically distinct root "
                  "(robustness; the chosen group is the baseline). Moments on the full step-2 "
                  "sample of the cell; starts out of the finite root solves:", "",
                  "| " + " | ".join(acols) + " |", "|" + "---|" * len(acols)]
        lines += ["| " + " | ".join(fmt(r[c]) for c in acols) + " |" for r in alt]
    (inter / "gnr_cells_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    drv = load_driver()
    blocks = ["mapping", "panel", "trimming", "step1", "step2"]
    stale = []
    for c, fp in fps.items():
        df, _ = drv.load_cell(config, c, drv.logging.getLogger("combine"))
        if fp != drv.fingerprint(config, c, drv.data_hash(df), blocks):
            stale.append(c)
    if missing or stale:
        print(f"\noutdata not written: missing cells {missing}, cells run with other settings {stale}")
    else:
        write_outdata(config, args.config, inter, panel, rows, fps)


if __name__ == "__main__":
    main()
