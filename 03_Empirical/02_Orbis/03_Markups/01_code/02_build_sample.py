"""Step 02: DLEU estimation and markup sample for Orbis Spain.

Follows 01_Create_Data.do and the top of 02_Produce_Thetas_st.do
(02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MARKUPS.md, D1-D6):
  1. Clean panel, the 14 GNR cells, years first_year..markup_years[1] (2014 is lag-only).
  2. V = mate + staf (COGS analogue); s_g = rev / V. One deflator (HICP), so s_g is nominal.
  3. Trim 1: keep p1 < s_g < p99, Stata percentiles by year pooled over cells.
  4. Capital: keep tfas > 0 (D4). This is where the DLEU code drops rows with missing capital
     (costshare1 missing).
  5. User cost (r_t - pi_t + 12) / 100 with the EONIA-basis rate and Spanish GDP-deflator
     inflation; kexp = usercost * tfas_D.
  6. Trim 2: costshare1 = V / (V + kexp); drop 0 or missing, keep p1 <= cs1 <= p99 by year.
     costshare2 needs SG&A and is skipped (D1).
  7. Lags after trimming, consecutive years only (Stata L. after xtset): c, k, K, and
     i = ln(K - 0.9 L.K) (missing when investment <= 0), then L.i.

Reads   <input.panel>, <input.deflators>, <user_cost.rates_dir>/overnight_rate_annual.csv
Writes  <output.intermediary_dir>/markup_sample.parquet   firm-years after both trims, with lags
        <output.intermediary_dir>/trim_waterfall.csv      counts by year x cell at each step
        <output.intermediary_dir>/trim_cutoffs.csv        p1/p99 cutoffs by year, user cost by year

    python 02_build_sample.py [--config config_markups_orbis_es.yaml]
"""

import argparse

import numpy as np
import pandas as pd

from markups_common import load_config, resolve, setup_logger, stata_pctile


def user_cost_table(config):
    """Annual usercost = (r - pi + premium) / 100, with its components."""
    uc = config["user_cost"]
    rate = pd.read_csv(resolve(uc["rates_dir"]) / "overnight_rate_annual.csv")
    defl = pd.read_csv(resolve(config["input"]["deflators"]))
    g = defl[(defl["series"] == uc["inflation_series"]) & (defl["code"] == "TOTAL")]
    g = g.sort_values("year").set_index("year")["index_raw"]
    infl = (100 * (g / g.shift(1) - 1)).rename("inflation").reset_index()
    t = rate[["year", "rate_eonia_basis"]].merge(infl, on="year", how="inner")
    t["usercost"] = (t["rate_eonia_basis"] - t["inflation"] + uc["premium_pp"]) / 100
    return t


def trim_by_year(df, col, lo_p, hi_p, strict):
    """Boolean keep mask: col inside its by-year Stata percentiles; returns (mask, cutoffs)."""
    cuts = df.groupby("year")[col].agg(lo=lambda x: stata_pctile(x, lo_p),
                                       hi=lambda x: stata_pctile(x, hi_p))
    lo = df["year"].map(cuts["lo"])
    hi = df["year"].map(cuts["hi"])
    keep = (df[col] > lo) & (df[col] < hi) if strict else (df[col] >= lo) & (df[col] <= hi)
    return keep.to_numpy(), cuts


def add_lags(df, id_col):
    """Stata-style one-year lags on the trimmed panel (missing unless the t-1 row exists)."""
    df = df.sort_values([id_col, "year"]).reset_index(drop=True)
    prev_ok = (df[id_col].eq(df[id_col].shift(1)) & df["year"].eq(df["year"].shift(1) + 1)).to_numpy()

    def lag(s):
        return s.shift(1).where(prev_ok)

    df["c_lag"] = lag(df["c"])
    df["k_lag"] = lag(df["k"])
    return df, lag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("sample", config.get("log_level", "INFO"))
    inp, smp, tr, th = config["input"], config["sample"], config["trimming"], config["theta"]
    id_col = inp["id_col"]
    if tr["by"] != "year":
        raise ValueError("only trimming.by = year (the DLEU code) is implemented")
    inter = resolve(config["output"]["intermediary_dir"])
    inter.mkdir(parents=True, exist_ok=True)

    y0, y1 = int(smp["first_year"]), int(smp["markup_years"][1])
    rev_col, cap_col, v_cols = smp["revenue"], smp["capital"], list(smp["variable_input"])
    cols = list(dict.fromkeys([id_col, "year", "ind2d", "excluded_sector", rev_col, cap_col,
                               *v_cols, *smp["keep_cols"]]))
    df = pd.read_parquet(resolve(inp["panel"]), columns=cols)
    df = df[df["year"].between(y0, y1) & df["ind2d"].isin(smp["cells"])].copy()
    if (df["excluded_sector"] != 0).any():
        raise RuntimeError("excluded sectors found inside the retained cells")
    df = df.drop(columns="excluded_sector")
    logger.info(f"panel {y0}-{y1}, {len(smp['cells'])} cells: {len(df):,} firm-years")

    df["rev_D"] = df[rev_col]
    df["V_D"] = df[v_cols].sum(axis=1, min_count=len(v_cols))
    df["cap_D"] = df[cap_col]
    if not ((df["rev_D"] > 0) & (df["V_D"] > 0)).all():
        raise RuntimeError("non-positive revenue or V: the cleaned panel should exclude these")
    df["s_g"] = df["rev_D"] / df["V_D"]

    wf = df.groupby(["year", "ind2d"]).size().rename("n_start").to_frame()
    lo_p, hi_p = tr["pct"]

    # Trim 1: sale / cogs, strict, by year (01_Create_Data.do section 04).
    keep, cut1 = trim_by_year(df, "s_g", lo_p, hi_p, strict=tr["sale_v_strict"])
    wf["drop_trim_sale_v"] = df[~keep].groupby(["year", "ind2d"]).size()
    df = df[keep]

    # Capital: tfas > 0 (D4), at the point where DLEU drops missing costshare1.
    keep = (df["cap_D"] > 0).to_numpy()
    wf["drop_capital"] = df[~keep].groupby(["year", "ind2d"]).size()
    logger.info(f"capital: tfas missing {int(df['cap_D'].isna().sum()):,}, "
                f"zero {int((df['cap_D'] == 0).sum()):,}, negative {int((df['cap_D'] < 0).sum()):,}")
    df = df[keep].copy()

    # User cost and capital expenditure.
    uct = user_cost_table(config)
    missing = set(range(y0, y1 + 1)) - set(uct["year"])
    if missing:
        raise RuntimeError(f"no user cost for years {sorted(missing)}")
    df = df.merge(uct[["year", "usercost"]], on="year", how="left")
    df["kexp"] = df["usercost"] * df["cap_D"]

    # Trim 2: costshare1, inclusive bounds, by year (02_Produce_Thetas_st.do section 01).
    df["cs1"] = df["V_D"] / (df["V_D"] + df["kexp"])
    keep = (df["cs1"].notna() & (df["cs1"] != 0)).to_numpy()
    df = df[keep]
    keep, cut2 = trim_by_year(df, "cs1", lo_p, hi_p, strict=not tr["costshare_inclusive"])
    wf["drop_trim_costshare"] = df[~keep].groupby(["year", "ind2d"]).size()
    df = df[keep].copy()

    # Production-function variables and lags after trimming.
    df["r"] = np.log(df["rev_D"])
    df["c"] = np.log(df["V_D"])
    df["k"] = np.log(df["cap_D"])
    df, lag = add_lags(df, id_col)
    df["inv"] = df["cap_D"] - (1 - th["depreciation"]) * lag(df["cap_D"])
    df["i"] = np.log(df["inv"].where(df["inv"] > 0))
    df["i_lag"] = lag(df["i"])
    df["has_lag"] = df["c_lag"].notna()
    df["has_lag_inv"] = df["i_lag"].notna()

    wf["n_final"] = df.groupby(["year", "ind2d"]).size()
    wf["n_has_lag"] = df.groupby(["year", "ind2d"])["has_lag"].sum()
    wf["n_inv_nonpos"] = df[df["inv"].notna() & ~(df["inv"] > 0)].groupby(["year", "ind2d"]).size()
    wf["n_has_lag_inv"] = df.groupby(["year", "ind2d"])["has_lag_inv"].sum()
    wf = wf.fillna(0).astype(int).reset_index()
    wf.to_csv(inter / "trim_waterfall.csv", index=False)

    cuts = (cut1.add_prefix("s_g_").join(cut2.add_prefix("cs1_")).reset_index()
            .merge(uct, on="year", how="left"))
    cuts.to_csv(inter / "trim_cutoffs.csv", index=False)

    out_cols = [id_col, "year", "ind2d", *smp["keep_cols"], "rev_D", "V_D", "cap_D", "usercost",
                "kexp", "s_g", "cs1", "r", "c", "k", "c_lag", "k_lag", "inv", "i", "i_lag",
                "has_lag", "has_lag_inv"]
    out_cols = list(dict.fromkeys(out_cols))
    df[out_cols].to_parquet(inter / "markup_sample.parquet", index=False)

    tot = wf.drop(columns="ind2d").groupby("year").sum()
    logger.info("waterfall by year:\n" + tot.to_string())
    logger.info("cutoffs and user cost:\n" + cuts.round(4).to_string(index=False))
    logger.info(f"wrote markup_sample.parquet: {len(df):,} firm-years, "
                f"{int(df['has_lag_inv'].sum()):,} with L.i")


if __name__ == "__main__":
    main()
