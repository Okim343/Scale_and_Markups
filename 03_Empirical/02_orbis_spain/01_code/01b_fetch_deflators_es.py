"""Fetch Spanish price indices from the Eurostat dissemination API (public, no credentials).

Baseline deflator (Hubmer et al. App. C.3): HICP all-items, every money variable.
Robustness "sector" deflators: A64 gross-value-added deflator (revenue and materials),
section-level GFCF deflator (capital), GDP deflator (wage bill).

Eurostat has no Spanish output (P1) or intermediate-consumption (P2) deflator at A64:
Spain transmits P1/P2 in nama_10_a64 only at current prices, and no supply-use tables at
previous-year prices. The VA deflator is therefore the only industry price index at A64.
See ORBIS_SPAIN_CLEANING.md.

    python 03_Empirical/02_orbis_spain/01_code/01b_fetch_deflators_es.py [--refresh]

Writes (public data; ignored by the repo-wide *indata*/ rule, rerun this script to rebuild):
    00_indata/deflators/{table}_ES.json     raw JSON-stat downloads
    00_indata/deflators/deflators_es.csv    tidy: series, code, year, index_raw, index_2019
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

DEFL_DIR = Path(__file__).resolve().parents[1] / "00_indata" / "deflators"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
SINCE = 2010
BASE_YEAR = 2019

# series name -> (table, filters, code dimension). The code dimension is the one kept
# as `code` in the tidy file (NACE industry, or a constant for aggregate series).
QUERIES = {
    "hicp": ("prc_hicp_aind", {"coicop": "CP00", "unit": "INX_A_AVG"}, None),
    "gdp_defl": ("nama_10_gdp", {"na_item": "B1GQ", "unit": "PD15_NAC"}, None),
    "va_defl": ("nama_10_a64", {"na_item": "B1G", "unit": "PD15_NAC"}, "nace_r2"),
    "gfcf_defl": ("nama_10_a64_p5",
                  {"na_item": "P51G", "asset10": "N11G", "unit": "PD15_NAC"}, "nace_r2"),
}


def fetch(table, filters, refresh):
    """Download one filtered table as JSON-stat; reuse the saved file unless refresh."""
    path = DEFL_DIR / f"{table}_ES.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    params = {"geo": "ES", "sinceTimePeriod": SINCE, "lang": "EN", **filters}
    for attempt in range(8):
        r = requests.get(f"{API}/{table}", params=params, timeout=180)
        d = r.json()
        # Large extractions are prepared asynchronously; poll until ready.
        if "error" in d and "ASYNCHRONOUS" in str(d["error"]):
            time.sleep(20)
            continue
        if "error" in d or r.status_code != 200:
            raise RuntimeError(f"Eurostat {table}: HTTP {r.status_code} {d.get('error')}")
        path.write_text(json.dumps(d))
        return d
    raise RuntimeError(f"Eurostat {table}: still asynchronous after retries")


def jsonstat_to_frame(d):
    """Flatten a JSON-stat 2.0 dataset into one row per non-missing cell."""
    ids, sizes = d["id"], d["size"]
    labels = [sorted(d["dimension"][k]["category"]["index"].items(), key=lambda x: x[1])
              for k in ids]
    rows = []
    for flat, value in d["value"].items():
        flat, pos = int(flat), []
        for s in reversed(sizes):
            pos.append(flat % s)
            flat //= s
        pos = pos[::-1]
        rows.append({k: labels[i][pos[i]][0] for i, k in enumerate(ids)} | {"value": value})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download saved tables")
    args = parser.parse_args()
    DEFL_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for series, (table, filters, code_dim) in QUERIES.items():
        df = jsonstat_to_frame(fetch(table, filters, args.refresh))
        df = df.assign(series=series, source=table,
                       code=df[code_dim] if code_dim else "TOTAL",
                       year=df["time"].astype(int), index_raw=df["value"].astype(float))
        frames.append(df[["series", "source", "code", "year", "index_raw"]])
        print(f"{series:10s} {table:16s} {len(df):5d} cells, "
              f"{df['code'].nunique() if code_dim else 1} codes, "
              f"years {df['year'].min()}-{df['year'].max()}")

    out = pd.concat(frames, ignore_index=True)
    base = (out[out["year"] == BASE_YEAR].set_index(["series", "code"])["index_raw"]
            .rename("base"))
    out = out.join(base, on=["series", "code"])
    n_nobase = int(out["base"].isna().sum())
    out = out.dropna(subset=["base"])
    out["index_2019"] = 100 * out["index_raw"] / out["base"]
    out = out.drop(columns="base").sort_values(["series", "code", "year"])
    out.to_csv(DEFL_DIR / "deflators_es.csv", index=False)
    print(f"wrote deflators_es.csv ({len(out)} rows; {n_nobase} rows without a "
          f"{BASE_YEAR} base dropped)")


if __name__ == "__main__":
    main()
