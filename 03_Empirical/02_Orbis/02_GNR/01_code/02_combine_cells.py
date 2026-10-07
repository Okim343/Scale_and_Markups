"""Stack finished per-cell GNR results into one Orbis Spain firm-year panel.

Reads  <output.intermediary_dir>/cells/<ind2d>/{firm_year_rts.parquet, cell_summary.json}
Writes <output.intermediary_dir>/firm_year_rts.parquet   all finished cells (gitignored)
       <output.intermediary_dir>/cell_diagnostics.csv    one row per cell
       <output.intermediary_dir>/gnr_cells_report.md     per-cell table vs Hubmer et al. Table A.5

Missing cells are listed, not an error, so this can run while the array is incomplete.

    python 02_combine_cells.py [--config config_gnr_orbis_es.yaml]
"""

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CODE / "config_gnr_orbis_es.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    inter = resolve(config["output"]["intermediary_dir"])
    lo, hi = config["combine"]["window"]

    frames, rows, missing = [], [], []
    for cell in config["cells"]["list"]:
        d = inter / "cells" / str(cell)
        if not ((d / "firm_year_rts.parquet").exists() and (d / "cell_summary.json").exists()):
            missing.append(cell)
            continue
        fy = pd.read_parquet(d / "firm_year_rts.parquet")
        frames.append(fy)
        sm = json.loads((d / "cell_summary.json").read_text())
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
        row["root_starts"] = (f"{st2['root_counts'][0]}/{st2['n_finite']}"
                              if st2.get("root_counts") else "")
        row["J_est"] = st2.get("J_estimate", np.nan)
        sc_path = d / "seed_check.json"
        if sc_path.exists():
            sc = json.loads(sc_path.read_text())["results"]
            rts = [r["rts_mean"] for r in sc]
            row["seeds"] = len(sc)
            row["seed_rts_range"] = max(rts) - min(rts)
            row["seed_roots"] = ",".join(str(r.get("n_distinct_roots", "?")) for r in sc)
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
    cols = ["sector", "step2_rows", "firms", "runtime_min", "root_found", "n_roots",
            "root_starts", "J_est", "seeds", "seed_rts_range", "seed_roots", "melast_mean",
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
    (inter / "gnr_cells_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
