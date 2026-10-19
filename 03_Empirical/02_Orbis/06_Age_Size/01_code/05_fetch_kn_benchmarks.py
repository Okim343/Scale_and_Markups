"""Fetch the national-accounts benchmark for the Karabarbounis-Neiman comparison.

Spanish non-financial corporations (sector S11) from Eurostat annual sector accounts
(nasa_10_nf_tr), current prices: compensation of employees (D1), gross value added (B1G),
other taxes and subsidies on production (D29, D39). KN's corporate labour share is D1 / B1G.
Public data, no credentials. Writes 06_Age_Size/00_indata/benchmarks/.

    python3 01_code/05_fetch_kn_benchmarks.py [--refresh]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "00_indata" / "benchmarks"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
TABLE = "nasa_10_nf_tr"
FILTERS = {"geo": "ES", "sector": "S11", "unit": "CP_MEUR", "na_item": ["D1", "B1G", "D29", "D39"],
           "sinceTimePeriod": 2008, "lang": "EN"}


def fetch(refresh):
    path = OUT / f"{TABLE}_ES_S11.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    for _ in range(8):
        r = requests.get(f"{API}/{TABLE}", params=FILTERS, timeout=180)
        d = r.json()
        if "error" in d and "ASYNCHRONOUS" in str(d["error"]):
            time.sleep(20)
            continue
        if "error" in d or r.status_code != 200:
            raise RuntimeError(f"Eurostat {TABLE}: HTTP {r.status_code} {d.get('error')}")
        path.write_text(json.dumps(d))
        return d
    raise RuntimeError(f"Eurostat {TABLE}: still asynchronous after retries")


def jsonstat_to_frame(d):
    """Flatten a JSON-stat 2.0 dataset into one row per non-missing cell."""
    ids, sizes = d["id"], d["size"]
    labels = [sorted(d["dimension"][k]["category"]["index"].items(), key=lambda x: x[1]) for k in ids]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = jsonstat_to_frame(fetch(args.refresh))
    df["year"] = df["time"].astype(int)
    tidy = df.pivot_table(index=["year", "direct"], columns="na_item", values="value").reset_index()
    tidy.to_csv(OUT / "na_s11_es.csv", index=False)
    print(tidy.to_string(index=False))


if __name__ == "__main__":
    main()
