"""Step 02: per (year, market) firm counts, concentration, HHI and V-weighted markup.

Ports Compustat 01_load_compustat.py:163-190 (markup trims) and 02_pooled_targets.py:64-110
(within-market shares and concentration). For every market definition in
markets.emx_robustness (ind2d, naics4, nace3, cnae4) and every year of the markup file:
  n, tot_sale = sum rev_D, tot_V = sum V_D;
  s = rev_D / tot_sale within (year, market); hhi = sum s^2;
  cr4, cr20 = cumulative sales share at rank min(k, n);
  top1pct, top5pct at k = max(1, ceil(0.01 n)), max(1, ceil(0.05 n));
  mu_cw_iy = sum V_D mu / sum V_D over the trimmed rows, mu = markup_wi after the S1 trims
    (D13: drop outside the hard bounds, then clip to the within-year p1/p99 of the survivors
    of all markets); used only by emx_slope;
  low_firm_flag = n < markets.low_firm_threshold.
The concentration universe is the whole markup sample (D2); the trims never drop rows from it.

Reads   <input.markups_dir>/firm_year_markups.parquet
Writes  <output.intermediary_dir>/market_year.parquet   one row per (market_def, year, market)

    python 02_build_market_panel.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import time

import numpy as np
import pandas as pd

from moments_common import dirs, load_config, resolve, setup_logger


def trim_markups(mu, year, hard, winsor_pct):
    """S1 trims (01_load_compustat.py:163-190): hard-bound mask, then within-year clipping.

    Returns (mu clipped, keep mask); rows outside the hard bounds are masked out and do not
    enter the year quantiles.
    """
    keep = mu.between(*hard)
    q = mu[keep].groupby(year[keep]).quantile(list(winsor_pct)).unstack()
    return mu.clip(lower=year.map(q[winsor_pct[0]]), upper=year.map(q[winsor_pct[1]])), keep


def market_year(df, market, sales="rev_D", weight="V_D", mu="mu", keep=None, low_firm=20):
    """Within-(year, market) aggregates of 02_pooled_targets.py:_aggregate_cell, vectorized.

    df needs year, the market column, sales, weight and mu; keep masks the rows entering
    mu_cw_iy (default all). Shares and concentration use every row.
    """
    keys = ["year", market]
    d = pd.DataFrame({"year": df["year"].to_numpy(), market: df[market].to_numpy(),
                      "sale": df[sales].to_numpy(float), "w": df[weight].to_numpy(float)})
    keep = np.ones(len(d), bool) if keep is None else np.asarray(keep)
    d["wmu"] = np.where(keep, d["w"] * df[mu].to_numpy(float), 0.0)
    d["wk"] = np.where(keep, d["w"], 0.0)
    d = d.sort_values(keys + ["sale"], ascending=[True, True, False], kind="stable")
    g = d.groupby(keys, sort=False)
    d["n"] = g["sale"].transform("size")
    d["tot"] = g["sale"].transform("sum")
    d["rank"] = g.cumcount() + 1
    d["cum"] = g["sale"].cumsum() / d["tot"]
    d["s"] = d["sale"] / d["tot"]
    d["s2"] = d["s"] ** 2

    out = g.agg(n=("sale", "size"), tot_sale=("sale", "sum"), tot_V=("w", "sum"), s_sum=("s", "sum"),
                hhi=("s2", "sum"), wmu=("wmu", "sum"), wk=("wk", "sum"))
    if not np.allclose(out["s_sum"], 1.0):
        raise RuntimeError(f"{market}: within-market shares do not sum to one")
    out["mu_cw_iy"] = out["wmu"] / out["wk"]
    n = d["n"].to_numpy()
    # Literal ports of the Compustat ranks, float ceil included.
    ranks = {"cr4": np.minimum(4, n), "cr20": np.minimum(20, n),
             "top1pct": np.maximum(1, np.ceil(0.01 * n)), "top5pct": np.maximum(1, np.ceil(0.05 * n))}
    for name, k in ranks.items():
        out[name] = d.loc[d["rank"].to_numpy() == k].set_index(keys)["cum"]
    out["low_firm_flag"] = out["n"] < low_firm
    return out.drop(columns=["s_sum", "wmu", "wk"]).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("market_panel", config.get("log_level", "INFO"))
    t0 = time.time()
    inter, _ = dirs(config)
    mk, trim = config["markets"], config["s1_markup_trim"]

    cols = ["bvdid", "year", *mk["emx_robustness"], "rev_D", "V_D", config["markup"]["col"]]
    df = pd.read_parquet(resolve(config["input"]["markups_dir"]) / "firm_year_markups.parquet",
                         columns=cols)
    logger.info(f"markup sample: {len(df):,} firm-years, {df['year'].min()}-{df['year'].max()}")

    mu = df[config["markup"]["col"]]
    df["mu"], keep = trim_markups(mu, df["year"], trim["hard"], trim["winsor_pct"])
    logger.info(f"S1 trims: {int((~keep).sum()):,} rows outside {trim['hard']}; "
                f"{int((keep & (df['mu'] != mu)).sum()):,} clipped to the within-year "
                f"{trim['winsor_pct']} quantiles")

    frames = []
    for m in mk["emx_robustness"]:
        # Firm-years without a code for this market definition leave its panel (cnae4 only).
        has = df[m].notna().to_numpy()
        if not has.all():
            logger.warning(f"{m}: {int((~has).sum()):,} firm-years without a code dropped "
                           f"({df.loc[~has, 'rev_D'].sum() / df['rev_D'].sum():.2%} of sales)")
        my = market_year(df[has], m, keep=keep[has], low_firm=int(mk["low_firm_threshold"]))
        if my["n"].sum() != has.sum():
            raise RuntimeError(f"{m}: market counts do not add up to the firm-years")
        frames.append(my.rename(columns={m: "market"}).assign(market_def=m))
        logger.info(f"{m}: {my[m].nunique()} markets, "
                    f"{len(my):,} market-years, {int(my['low_firm_flag'].sum()):,} with n < "
                    f"{mk['low_firm_threshold']}")
    out = pd.concat(frames, ignore_index=True)
    out["market"] = out["market"].astype(str)
    out = out[["market_def", "year", "market", "n", "tot_sale", "tot_V", "cr4", "cr20", "top1pct",
               "top5pct", "hhi", "mu_cw_iy", "low_firm_flag"]]
    out.to_parquet(inter / "market_year.parquet", index=False)
    logger.info(f"wrote market_year.parquet: {len(out):,} rows ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
