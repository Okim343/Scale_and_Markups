"""Build the age/size analysis panel: clean panel + extra cost items + cost split.

Reads (never writes) the GNR pipeline's clean panel; writes only to 06_Age_Size/01_intermediary.

Cost split on operating revenue (opre), from the expense side:
    oth = oope - staf - depr        other external costs (services, rents, utilities, ...)
    va  = opre - mate - oth         operating value added
    op  = va - staf - depr          operating profit / loss (= Orbis oppl when cost = mate)
so opre = mate + oth + staf + depr + op exactly. Kept: nature-of-expense accounts, where
Orbis cost of sales equals materials (|cost - mate| < 1% of opre), and oth >= 0.
Orbis's own added value (av) is not used: it is built from net profit, so it includes
financial and extraordinary items; it is kept only as a cross-check.

Two samples:
    baseline   depr reported (firm-years with missing depr dropped)
    zero_fill  missing depr set to 0 (robustness)

    python3 01_code/02_build_panel.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CLEAN = HERE.parent / "01_Cleaning" / "01_intermediary" / "orbis_es_clean_panel.parquet"
EXTRA = HERE / "00_indata" / "orbis_es_extra_costs.parquet"
OUT = HERE / "01_intermediary"
KEYS = ["bvdid", "conscode", "filing_type", "closdate"]
TOL = 0.01


def cost_split(d, zero_fill):
    d = d.copy()
    if zero_fill:
        d["depr"] = d["depr"].fillna(0.0)
    d = d[d[["depr", "oope", "oppl", "cost"]].notna().all(axis=1)]
    d = d[(d["cost"] - d["mate"]).abs() / d["opre"] < TOL]
    d["oth"] = d["oope"] - d["staf"] - d["depr"]
    d["va"] = d["opre"] - d["mate"] - d["oth"]
    d["op"] = d["va"] - d["staf"] - d["depr"]
    return d[d["oth"] >= 0]


def main():
    cols = KEYS + ["year", "naics4", "sector_id", "excluded_sector", "inc_year",
                   "rev", "opre", "mate", "staf", "tfas", "empl", "rev_D"]
    d = pd.read_parquet(CLEAN, columns=cols)
    d = d[(d["excluded_sector"] == 0) & d["year"].between(2015, 2023) & (d["tfas"] > 0)
          & (d["opre"] > 0)]
    d["closdate"] = pd.to_datetime(d["closdate"])
    x = pd.read_parquet(EXTRA)
    x["closdate"] = pd.to_datetime(x["closdate"])
    for c in KEYS[1:3]:
        d[c] = d[c].astype(str)
        x[c] = x[c].astype(str)
    n0 = len(d)
    d = d.merge(x, on=KEYS, how="left", validate="one_to_one")
    d["defl"] = d["rev_D"] / d["rev"]            # HICP factor to 2019 EUR, as in the clean panel

    log = [f"base firm-years (retained sectors, 2015-2023, tfas > 0, opre > 0): {n0:,}",
           f"matched extra items: oope {d['oope'].notna().mean():.4f}, "
           f"depr {d['depr'].notna().mean():.4f}, oppl {d['oppl'].notna().mean():.4f}, "
           f"cost {d['cost'].notna().mean():.4f}",
           f"nature-of-expense (|cost - mate| < 1% of opre): "
           f"{((d['cost'] - d['mate']).abs() / d['opre'] < TOL).mean():.4f}"]
    for name, zf in [("baseline", False), ("zero_fill", True)]:
        s = cost_split(d, zf)
        s.drop(columns=["rev_D"]).to_parquet(OUT / f"panel_{name}.parquet", index=False)
        log.append(f"{name}: {len(s):,} firm-years ({len(s) / n0:.3f} of base), "
                   f"{s['bvdid'].nunique():,} firms; operating P/L matches Orbis oppl within 1%: "
                   f"{((s['op'] - s['oppl']).abs() / s['opre'] < TOL).mean():.4f}; "
                   f"va within 1% of Orbis av: {((s['va'] - s['av']).abs() / s['opre'] < TOL).mean():.4f}")
    (OUT / "build_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
