"""Step 03: pooled concentration targets and firm counts.

Ports Compustat 02_pooled_targets.py:127-169 and 08_assemble_bundle.py:536-538. Over a window:
  per market, the window mean of n, tot_sale, cr4, cr20, top1pct and top5pct;
  pooled cr4/cr20/top1pct/top5pct = window-mean tot_sale-weighted mean across markets;
  n_firms_cs = sum of the window-mean n (D10: literal, no rescaling);
  mean_firms_per_sector = tot_sale-weighted mean of the window-mean n;
  n_markets = number of markets (manifest only).
The bundle uses markets.concentration (naics4, D1) over active_window; every market definition
and the robustness window are also computed for the report.

Reads   <output.intermediary_dir>/market_year.parquet
Writes  <output.intermediary_dir>/pooled_targets.json   pooled values by market definition x window
        <output.intermediary_dir>/per_market.csv        window means per market

    python 03_pooled_targets.py [--config config_moments_orbis_es.yaml]
"""

import argparse

import numpy as np
import pandas as pd

from moments_common import dirs, load_config, setup_logger, write_json

MEASURES = ["cr4", "cr20", "top1pct", "top5pct"]


def pool(my, window, market="market"):
    """Window means per market and their sales-weighted pooling; returns (pooled, per_market)."""
    y0, y1 = window
    per = (my[my["year"].between(y0, y1)]
           .groupby(market, as_index=False)[["n", "tot_sale", *MEASURES]].mean())
    w = per["tot_sale"].to_numpy(float) / per["tot_sale"].sum()
    if not np.isclose(w.sum(), 1.0):
        raise RuntimeError("pooling weights do not sum to one")
    pooled = {k: float(np.dot(w, per[k])) for k in MEASURES}
    pooled.update(n_firms_cs=float(per["n"].sum()), mean_firms_per_sector=float(np.dot(w, per["n"])),
                  median_firms_per_market=float(per["n"].median()), n_markets=int(len(per)),
                  tot_sale=float(per["tot_sale"].sum()))
    return pooled, per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("pooled_targets", config.get("log_level", "INFO"))
    inter, _ = dirs(config)
    my = pd.read_parquet(inter / "market_year.parquet")
    windows = {"active": config["active_window"], "robustness": config["robustness_window"]}

    pooled, tables = {}, []
    for m, (tag, win) in [(m, w) for m in config["markets"]["emx_robustness"] for w in windows.items()]:
        p, per = pool(my[my["market_def"] == m], win)
        pooled.setdefault(m, {})[tag] = {"window": [int(x) for x in win], **p}
        tables.append(per.assign(market_def=m, window=f"{win[0]}-{win[1]}"))
        logger.info(f"{m:7s} {win[0]}-{win[1]}: " + ", ".join(
            f"{k} {p[k]:.4f}" for k in MEASURES) + f", n_firms_cs {p['n_firms_cs']:,.1f}, "
            f"mean_firms_per_sector {p['mean_firms_per_sector']:,.1f}, n_markets {p['n_markets']}")

    conc = config["markets"]["concentration"]
    result = {"concentration_market": conc, "active_window": config["active_window"],
              "bundle": pooled[conc]["active"], "by_market": pooled}
    write_json(inter / "pooled_targets.json", result)
    per_market = pd.concat(tables, ignore_index=True)
    per_market[["market_def", "window", "market", "n", "tot_sale", *MEASURES]].to_csv(
        inter / "per_market.csv", index=False)
    logger.info(f"bundle market: {conc}, {config['active_window']}; "
                "wrote pooled_targets.json and per_market.csv")


if __name__ == "__main__":
    main()
