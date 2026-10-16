"""Pull the extra cost items for the age/size analysis from WRDS Orbis.

Separate from the GNR / markup pipeline: reads nothing from it and writes only to
06_Age_Size/00_indata. Same account filter as 01_Cleaning/01_code/01_extract_orbis_es.py
(Spain, unconsolidated U1/U2, closing years 2014-2024), so rows match its accounts on
the key (bvdid, conscode, filing_type, closdate).

    oope  other operating expenses = staff costs + depreciation + other external costs
    depr  depreciation and amortisation
    oppl  operating P/L = opre - cost - oope
    cost  cost of sales (equals mate for nature-of-expense filers)
    av    Orbis added value (built from net profit, so it includes financial items)

    WRDS_USERNAME=ahmed_ahmed python3 01_code/01_extract_extra_costs.py </dev/null

Chunked by size tier x year and restartable: finished chunks are skipped.
"""

import os
import time
from pathlib import Path

import pandas as pd
import wrds

OUT_DIR = Path(__file__).resolve().parents[1] / "00_indata"
CHUNK_DIR = OUT_DIR / "raw_chunks"
LIB = "bvd"
TIERS = ["l", "m", "s"]
YEARS = range(2014, 2025)
KEYS = ["bvdid", "conscode", "filing_type", "closdate"]
COLS = KEYS + ["oope", "depr", "oppl", "cost", "av"]


def sql(tier, year):
    return (f"SELECT {', '.join(COLS)} FROM {LIB}.ob_ind_g_fins_{tier} "
            f"WHERE contact_ctryiso = 'ES' AND conscode IN ('U1', 'U2') "
            f"AND closdate_year = {year}")


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
    for tier in TIERS:
        for year in YEARS:
            path = CHUNK_DIR / f"extra_{tier}_{year}.parquet"
            if path.exists():
                print(f"skip {tier} {year} (exists)")
                continue
            t0 = time.time()
            df = db.raw_sql(sql(tier, year))
            df.to_parquet(path, index=False)
            print(f"{tier} {year}: {len(df):,} rows in {time.time() - t0:.0f}s", flush=True)
    db.close()

    extra = pd.concat([pd.read_parquet(CHUNK_DIR / f"extra_{t}_{y}.parquet")
                       for t in TIERS for y in YEARS], ignore_index=True)
    extra["closdate"] = pd.to_datetime(extra["closdate"])
    n0 = len(extra)
    extra = extra.drop_duplicates(KEYS)
    extra.to_parquet(OUT_DIR / "orbis_es_extra_costs.parquet", index=False)
    print(f"wrote orbis_es_extra_costs.parquet: {len(extra):,} rows "
          f"({n0 - len(extra):,} duplicate keys dropped)")


if __name__ == "__main__":
    main()
