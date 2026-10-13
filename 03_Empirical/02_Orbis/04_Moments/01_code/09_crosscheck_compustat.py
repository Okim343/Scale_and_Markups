"""Step 09: cross-check the step 02/03/05 functions on the Compustat S1 file (read only).

Runs the Orbis functions on the Compustat S1 intermediate with the Compustat choices (market
sector_id, nominal sale, cogs weights, the already-trimmed S1 markup so no extra trim, window
compustat.window) and compares with the published Compustat
values (compustat.expected) at compustat.tolerance. Writes nothing; prints PASS or FAIL per
moment and exits non-zero on any failure.

Reads   <compustat.intermediary_dir>/s1_compustat_firmyear.parquet

    python 09_crosscheck_compustat.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import importlib
import sys

import pandas as pd

from moments_common import load_config, resolve, setup_logger

panel = importlib.import_module("02_build_market_panel")
pooled = importlib.import_module("03_pooled_targets")
emx = importlib.import_module("05_emx_slope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("crosscheck", config.get("log_level", "INFO"))
    cs = config["compustat"]
    inter = resolve(cs["intermediary_dir"])
    y0, y1 = map(int, cs["window"])

    s1 = pd.read_parquet(inter / "s1_compustat_firmyear.parquet",
                         columns=["year", "sector_id", "sale", "cogs", "markup"])
    s1["year"] = s1["year"].astype(int)
    my = panel.market_year(s1, "sector_id", sales="sale", weight="cogs", mu="markup")
    p, _ = pooled.pool(my, (y0, y1), market="sector_id")
    W = emx.wide(my.rename(columns={"sector_id": "market"}))
    slope = emx.ols(emx.long_diff(W, y0, y1))

    got = {"cr4": p["cr4"], "cr20": p["cr20"], "top1pct": p["top1pct"], "top5pct": p["top5pct"],
           "n_firms_cs": p["n_firms_cs"], "emx_slope": slope["b_hat"]}
    tol, ok = float(cs["tolerance"]), True
    for k, exp in cs["expected"].items():
        passed = abs(got[k] - exp) <= tol
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {k:26s} got {got[k]:.7f}  expected {exp:.7f}  "
              f"diff {got[k] - exp:+.2e}")
    logger.info(f"Compustat cross-check {y0}-{y1}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
