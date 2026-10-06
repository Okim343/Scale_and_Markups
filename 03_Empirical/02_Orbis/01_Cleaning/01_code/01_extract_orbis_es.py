"""Extract Spanish Orbis financials and industry codes from WRDS (spec: ORBIS_SPAIN_EXTRACTION.md).

Pulls exactly the GNR / markup / concentration inputs for Spanish unconsolidated
(U1, U2) industrial-company accounts, all size tiers, closing years 2014-2024.
Chunked by size tier x year and restartable: finished chunks are skipped.

Credentials come from ~/.pgpass and the WRDS_USERNAME env var; nothing is stored here.

    python 03_Empirical/02_Orbis/01_Cleaning/01_code/01_extract_orbis_es.py

Writes (all gitignored except the report):
    00_indata/raw_chunks/fin_{tier}_{year}.parquet, firms_{tier}.parquet
    00_indata/orbis_es_financials.parquet
    00_indata/orbis_es_firms.parquet
    00_indata/extraction_report.md
"""

import os
import time
from pathlib import Path

import pandas as pd
import wrds

IN_DIR = Path(__file__).resolve().parents[1] / "00_indata"
CHUNK_DIR = IN_DIR / "raw_chunks"
LIB = "bvd"
TIERS = ["l", "m", "s"]
YEARS = range(2014, 2025)
KEYS = ["bvdid", "conscode", "filing_type", "closdate"]
FIN_COLS = KEYS + ["closdate_year", "nr_months",
                   "turn", "opre", "mate", "staf", "tfas", "fias", "empl"]
MONEY = ["turn", "opre", "mate", "staf", "tfas", "fias", "empl"]
CODES = ["naceccod2", "nacepcod2", "naicsccod2017", "naicspcod2017"]


def fin_sql(tier, year):
    return (f"SELECT {', '.join(FIN_COLS)} FROM {LIB}.ob_ind_g_fins_{tier} "
            f"WHERE contact_ctryiso = 'ES' AND conscode IN ('U1', 'U2') "
            f"AND closdate_year = {year}")


def firm_sql(tier):
    # One row per firm: several classification rows per firm, so collapse server side.
    # Core/primary codes should be unique per firm; n_* counts let us verify that.
    aggs = ", ".join(f"max({c}) AS {c}, count(DISTINCT {c}) AS n_{c}" for c in CODES)
    return (f"SELECT bvdid, {aggs}, "
            f"max(CASE WHEN natclass = 'CNAE 2009' THEN natpcod END) AS cnae2009_pcod "
            f"FROM {LIB}.ob_industry_classifications_{tier} "
            f"WHERE contact_ctryiso = 'ES' GROUP BY bvdid")


def fetch(db, sql, path, label):
    if path.exists():
        print(f"skip {label} (exists)")
        return
    t0 = time.time()
    df = db.raw_sql(sql)
    df.to_parquet(path, index=False)
    print(f"{label}: {len(df):,} rows in {time.time() - t0:.0f}s")


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
    for tier in TIERS:
        fetch(db, firm_sql(tier), CHUNK_DIR / f"firms_{tier}.parquet", f"firms {tier}")
        for year in YEARS:
            fetch(db, fin_sql(tier, year), CHUNK_DIR / f"fin_{tier}_{year}.parquet",
                  f"fin {tier} {year}")
    db.close()

    fin = pd.concat(
        [pd.read_parquet(CHUNK_DIR / f"fin_{t}_{y}.parquet").assign(size_tier=t)
         for t in TIERS for y in YEARS], ignore_index=True)
    fin["closdate"] = pd.to_datetime(fin["closdate"])
    fin["closdate_year"] = fin["closdate_year"].astype("Int16")
    for c in ["conscode", "filing_type", "nr_months", "size_tier"]:
        fin[c] = fin[c].astype("category")

    firms = pd.concat([pd.read_parquet(CHUNK_DIR / f"firms_{t}.parquet").assign(size_tier=t)
                       for t in TIERS], ignore_index=True)
    firms = firms[firms["bvdid"].isin(fin["bvdid"].unique())]

    fin.to_parquet(IN_DIR / "orbis_es_financials.parquet", index=False)
    firms.to_parquet(IN_DIR / "orbis_es_firms.parquet", index=False)
    write_report(fin, firms)


def write_report(fin, firms):
    out = ["# Extraction report: Orbis Spain", "",
           f"Source: `{LIB}.ob_ind_g_fins_{{l,m,s}}`, `{LIB}.ob_industry_classifications_{{l,m,s}}`; "
           "ES, conscode U1/U2, closdate_year 2014-2024.", ""]

    fin_dups = fin.duplicated(KEYS).sum()
    firm_dups = firms["bvdid"].duplicated().sum()
    out += ["## Size and keys", "",
            f"- financials: {len(fin):,} rows, {fin['bvdid'].nunique():,} firms",
            f"- firms: {len(firms):,} rows, {firms['bvdid'].nunique():,} firms",
            f"- duplicate financial keys {KEYS}: {fin_dups:,}",
            f"- duplicate bvdid in firms (same firm in several tiers): {firm_dups:,}",
            f"- financial firms matched in firms file: "
            f"{fin['bvdid'].isin(firms['bvdid']).mean():.4f}",
            f"- firm-years with more than one account (several filing types/closdates): "
            f"{fin.duplicated(['bvdid', 'closdate_year']).sum():,}",
            f"- bvdid not starting with 'ES': {(~fin['bvdid'].str.startswith('ES')).sum():,}", ""]

    out += ["## Industry-code coverage (share of firms)", ""]
    for c in CODES + ["cnae2009_pcod"]:
        out.append(f"- {c}: {firms[c].notna().mean():.4f}")
    for c in CODES:
        out.append(f"- firms with more than one distinct {c}: {(firms[f'n_{c}'] > 1).sum():,}")
    out.append("")

    pos = fin[MONEY].gt(0)
    rev = fin["turn"].fillna(fin["opre"]).gt(0)
    fin["gnr_all"] = rev & pos["mate"] & pos["staf"] & pos["tfas"]
    fin["m12"] = fin["nr_months"].astype(str).eq("12")
    by = fin.assign(**{f"{c}_pos": pos[c] for c in MONEY}).groupby(
        ["closdate_year", "size_tier"], observed=True)
    tab = by.agg(n=("bvdid", "size"), m12=("m12", "mean"), gnr_all=("gnr_all", "mean"),
                 **{f"{c}_pos": (f"{c}_pos", "mean") for c in MONEY}).round(3)
    out += ["## By year and tier (shares count missing as not positive)", "",
            tab.reset_index().to_markdown(index=False), ""]

    usable = fin[fin["gnr_all"] & fin["m12"]].groupby("closdate_year").size()
    out += ["## Usable firm-years (all GNR inputs > 0, 12-month accounts)", "",
            usable.rename("n").reset_index().to_markdown(index=False),
            f"\nTotal: {usable.sum():,}", ""]

    out += ["## Files", ""]
    for f in ["orbis_es_financials.parquet", "orbis_es_firms.parquet"]:
        out.append(f"- {f}: {(IN_DIR / f).stat().st_size / 1e6:.0f} MB")
    out += ["", "## SQL (per tier and year)", "", "```sql", fin_sql("{tier}", "{year}"),
            firm_sql("{tier}"), "```", ""]

    (IN_DIR / "extraction_report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
