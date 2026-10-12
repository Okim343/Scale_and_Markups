"""Step 04: firm-year markups, comparison with the GNR markup, manifest and report.

markup_wi = theta_wi_c * rev_D / V_D (DLEU 03_Calibration_Moments.do: theta_WI1_ct * sale/cogs)
markup_cs = theta_cs   * rev_D / V_D (paper Section VI.A)
on every firm-year of the trimmed sample in sample.markup_years (2014 is lag-only).

The aggregate moments belong to 04_Moments; the report here gives per cell x year and pooled
diagnostics: thetas, sales-, V- and (V + kexp)-weighted mean markups, unweighted and
sales-weighted P10/P50/P90 (DLEU III.B weights firms by sales share), trim counts, and the GNR
materials-based markup melast * rev / mate on the same firm-years
(02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MARKUPS.md, D6c).

Reads   <output.intermediary_dir>/{markup_sample,thetas}.parquet, trim_waterfall.csv
        <compare.gnr_outdata_dir>/firm_year_rts.parquet
Writes  <output.outdata_dir>/firm_year_markups.parquet   one row per firm-year (Section 6 of the md)
        <output.outdata_dir>/manifest.yaml               provenance, row counts
        <output.outdata_dir>/markups_report.md           tables
        <output.outdata_dir>/cell_year_summary.csv       the per cell x year table

    python 04_build_markups.py [--config config_markups_orbis_es.yaml]
"""

import argparse
import datetime

import numpy as np
import pandas as pd
import yaml

from markups_common import (PROJECT_ROOT, file_sha256, git_state, load_config, resolve,
                            setup_logger, weighted_quantile)

OUT_COLS = ["bvdid", "year", "ind2d", "naics4", "nace3", "cnae4", "rev_D", "V_D", "mate_D",
            "staf_D", "cap_D", "kexp", "usercost", "theta_wi_c", "theta_wi_k", "theta_imputed",
            "theta_cs", "markup_wi", "markup_cs", "in_iv_sample", "has_lag", "has_lag_inv"]


def summarize(g, mcol="markup_wi"):
    """Markup moments of one group (within-group weights, as DLEU within year)."""
    m, s, v, vk = g[mcol], g["rev_D"], g["V_D"], g["V_D"] + g["kexp"]
    q = m.quantile([0.1, 0.5, 0.9]).to_numpy()
    wq = weighted_quantile(m, s, [0.1, 0.5, 0.9])
    return pd.Series({"n": len(g), "mu_sales_w": np.average(m, weights=s),
                      "mu_V_w": np.average(m, weights=v), "mu_Vk_w": np.average(m, weights=vk),
                      "var_sales_w": np.average((m - np.average(m, weights=s)) ** 2, weights=s),
                      "p10": q[0], "p50": q[1], "p90": q[2],
                      "p10_sw": wq[0], "p50_sw": wq[1], "p90_sw": wq[2]})


def md_table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            x = r[c]
            if isinstance(x, (float, np.floating)):
                cells.append(f"{x:.{floatfmt}f}")
            elif isinstance(x, (int, np.integer)):
                cells.append(f"{x:,}" if abs(x) >= 10000 else str(x))
            else:
                cells.append(str(x))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    logger = setup_logger("markups", config.get("log_level", "INFO"))
    smp, th = config["sample"], config["theta"]
    inter = resolve(config["output"]["intermediary_dir"])
    outd = resolve(config["output"]["outdata_dir"])
    outd.mkdir(parents=True, exist_ok=True)
    m0, m1 = map(int, smp["markup_years"])
    c0, c1 = map(int, th["centres"])

    df = pd.read_parquet(inter / "markup_sample.parquet")
    thetas = pd.read_parquet(inter / "thetas.parquet")
    wf = pd.read_csv(inter / "trim_waterfall.csv")

    df = df[df["year"].between(m0, m1)].merge(
        thetas[["ind2d", "year", "theta_wi_c", "theta_wi_k", "theta_imputed", "theta_cs"]],
        on=["ind2d", "year"], how="left", validate="many_to_one")
    if df["theta_wi_c"].isna().any():
        raise RuntimeError("firm-years without a theta")
    df["markup_wi"] = df["theta_wi_c"] * df["rev_D"] / df["V_D"]
    df["markup_cs"] = df["theta_cs"] * df["rev_D"] / df["V_D"]
    reg = ["r", "c", "c_lag", "k", "i_lag", "k_lag"]
    df["in_iv_sample"] = np.isfinite(df[reg]).all(axis=1) & df["year"].between(c0, c1)

    # mate_D and staf_D come from keep_cols; check the V identity.
    if not np.allclose(df["mate_D"] + df["staf_D"], df["V_D"]):
        raise RuntimeError("V_D != mate_D + staf_D")
    out = df[OUT_COLS].sort_values(["bvdid", "year"]).reset_index(drop=True)
    out.to_parquet(outd / "firm_year_markups.parquet", index=False)
    logger.info(f"wrote firm_year_markups.parquet: {len(out):,} firm-years, "
                f"{out['bvdid'].nunique():,} firms")

    # --- Per cell x year ---------------------------------------------------------------
    cy = df.groupby(["ind2d", "year"]).apply(summarize, include_groups=False).reset_index()
    cy_cs = (df.groupby(["ind2d", "year"]).apply(lambda g: np.average(g["markup_cs"], weights=g["rev_D"]),
                                                  include_groups=False).rename("mu_cs_sales_w").reset_index())
    cy = (thetas[["ind2d", "year", "theta_wi_c", "theta_wi_k", "se_wi_c", "n_iv", "theta_imputed",
                  "theta_cs"]].merge(cy, on=["ind2d", "year"]).merge(cy_cs, on=["ind2d", "year"])
          .merge(wf[["ind2d", "year", "n_start", "drop_trim_sale_v", "drop_capital",
                     "drop_trim_costshare"]], on=["ind2d", "year"], how="left"))
    cy["n"] = cy["n"].astype(int)
    cy.to_csv(outd / "cell_year_summary.csv", index=False)

    # --- Pooled by year ------------------------------------------------------------------
    py = df.groupby("year").apply(summarize, include_groups=False)
    py["mu_cs_sales_w"] = df.groupby("year").apply(
        lambda g: np.average(g["markup_cs"], weights=g["rev_D"]), include_groups=False)
    py["theta_wi_sales_w"] = df.groupby("year").apply(
        lambda g: np.average(g["theta_wi_c"], weights=g["rev_D"]), include_groups=False)
    py["n"] = py["n"].astype(int)
    wfy = wf.drop(columns="ind2d").groupby("year").sum()
    py = py.join(wfy[["n_start", "drop_trim_sale_v", "drop_capital", "drop_trim_costshare"]]).reset_index()
    py_avg = py.drop(columns="year").mean(numeric_only=True)

    # --- Pooled by cell over the window (average of the yearly within-cell figures) ------
    pc = cy.groupby("ind2d")[["theta_wi_c", "theta_cs", "mu_sales_w", "mu_V_w", "p50", "p90",
                              "p50_sw", "p90_sw", "mu_cs_sales_w"]].mean()
    pc["n_firm_years"] = cy.groupby("ind2d")["n"].sum()
    pc = pc.reset_index()

    # --- GNR comparison on the same firm-years --------------------------------------------
    gnr_dir = resolve(config["compare"]["gnr_outdata_dir"])
    gnr = pd.read_parquet(gnr_dir / "firm_year_rts.parquet", columns=["bvdid", "year", "melast"])
    cmp = df.merge(gnr, on=["bvdid", "year"], how="inner")
    cmp = cmp[np.isfinite(cmp["melast"])]
    cmp["markup_gnr"] = cmp["melast"] * cmp["rev_D"] / cmp["mate_D"]

    def gsum(g):
        sw = g["rev_D"]
        return pd.Series({
            "n": len(g), "mu_wi_sales_w": np.average(g["markup_wi"], weights=sw),
            "mu_gnr_sales_w": np.average(g["markup_gnr"], weights=sw),
            "p50_wi": g["markup_wi"].median(), "p50_gnr": g["markup_gnr"].median(),
            "spearman": g["markup_wi"].corr(g["markup_gnr"], method="spearman")})
    gy = cmp.groupby("year").apply(gsum, include_groups=False).reset_index()
    gc = cmp.groupby("ind2d").apply(gsum, include_groups=False).reset_index()
    gy["n"], gc["n"] = gy["n"].astype(int), gc["n"].astype(int)
    logger.info(f"GNR comparison: {len(cmp):,} common firm-years "
                f"({len(cmp) / len(df):.1%} of the markup sample)")

    # --- Worldscope level check (loose) ----------------------------------------------------
    ws = config["compare"]["worldscope"]
    wy = df[df["year"] == int(ws["year"])]
    ratio = wy["rev_D"] / wy["V_D"]

    def sw(x, g):
        return np.average(x, weights=g["rev_D"])
    ws_rows = [
        {"series": f"Worldscope Spain ({ws['source']})", "n_firms": "", "markup_sales_w": ws["markup_sales_w"]},
        {"series": "Orbis, baseline (theta_wi_c)", "n_firms": len(wy), "markup_sales_w": sw(wy["markup_wi"], wy)},
        {"series": f"Orbis, constant theta = {ws['constant_theta']}", "n_firms": len(wy),
         "markup_sales_w": ws["constant_theta"] * sw(ratio, wy)},
    ]
    for n in ws["top_n_firms"]:
        top = wy.nlargest(int(n), "rev_D")
        ws_rows += [
            {"series": f"Orbis, top {n} firms by sales, baseline", "n_firms": len(top),
             "markup_sales_w": sw(top["markup_wi"], top)},
            {"series": f"Orbis, top {n} firms by sales, constant theta = {ws['constant_theta']}",
             "n_firms": len(top), "markup_sales_w": ws["constant_theta"] * sw(top["rev_D"] / top["V_D"], top)},
        ]
    wst = pd.DataFrame(ws_rows)
    logger.info(f"Worldscope check, {ws['year']}:\n" + wst.round(3).to_string(index=False))

    # --- Manifest -------------------------------------------------------------------------
    commit, branch, dirty = git_state()
    panel_path = resolve(config["input"]["panel"])
    rates_path = resolve(config["user_cost"]["rates_dir"]) / "overnight_rate_annual.csv"
    manifest = {
        "description": "Orbis Spain firm-year markups following De Loecker, Eeckhout and Unger (2020): "
                       "V = mate + staf, rolling 5-year Wooldridge-IV elasticity by NAICS2 cell; "
                       "cost-share alternative. Decisions: "
                       "02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MARKUPS.md.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit, "branch": branch, "code_dirty": dirty,
        "pipeline_version": config["pipeline_version"],
        "config": str(config_path.resolve().relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "input": config["input"]["panel"], "input_sha256": file_sha256(panel_path),
        "rates_sha256": file_sha256(rates_path),
        "years": [m0, m1], "cells": [int(c) for c in smp["cells"]],
        "rows": int(len(out)), "firms": int(out["bvdid"].nunique()),
        "rows_in_iv_sample": int(out["in_iv_sample"].sum()),
        "rows_theta_imputed": int(out["theta_imputed"].sum()),
        "rows_by_year": {int(k): int(v) for k, v in out.groupby("year").size().items()},
        "columns": OUT_COLS,
        "weights_available": {"sales": "rev_D", "V": "V_D", "V_plus_capital_cost": "V_D + kexp"},
    }
    (outd / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))

    # --- Report ---------------------------------------------------------------------------
    cuts = pd.read_csv(inter / "trim_cutoffs.csv")
    yr_cols = ["year", "n", "theta_wi_sales_w", "mu_sales_w", "mu_V_w", "mu_Vk_w", "mu_cs_sales_w",
               "p10", "p50", "p90", "p10_sw", "p50_sw", "p90_sw", "n_start", "drop_trim_sale_v",
               "drop_capital", "drop_trim_costshare"]
    cy_cols = ["ind2d", "year", "theta_wi_c", "theta_wi_k", "se_wi_c", "n_iv", "theta_cs", "n",
               "mu_sales_w", "mu_V_w", "p10", "p50", "p90", "p10_sw", "p50_sw", "p90_sw",
               "drop_trim_sale_v", "drop_capital", "drop_trim_costshare"]
    cyr = cy[cy_cols].copy()
    cyr["year"] = cyr["year"].astype(str) + np.where(cy["theta_imputed"], "*", "")
    lines = [
        "# Orbis Spain DLEU markups: report",
        "",
        f"Generated {manifest['created_utc']} from commit `{commit}` (code dirty: {dirty}). "
        "Decisions: `02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MARKUPS.md`. "
        "markup_wi = theta_wi_c * rev / (mate + staf).",
        "",
        f"Firm-years {len(out):,} ({out['bvdid'].nunique():,} firms), {m0}-{m1}, 14 cells; "
        f"{int(out['in_iv_sample'].sum()):,} enter the IV.",
        "",
        "## Pooled by year",
        "",
        "Weights within year over all cells. `_sw` = sales-weighted percentile. Trim counts include "
        "every cell; `drop_capital` is tfas <= 0 or missing.",
        "",
        md_table(py[yr_cols]),
        "",
        "Averages of the yearly figures, 2015-2023: "
        + ", ".join(f"{k} {py_avg[k]:.3f}" for k in ["mu_sales_w", "mu_V_w", "mu_Vk_w",
                                                     "mu_cs_sales_w", "p50", "p90", "p90_sw"]) + ".",
        "",
        "## Pooled by cell (average of the yearly within-cell figures)",
        "",
        md_table(pc),
        "",
        "## Comparison with the GNR materials-based markup (same firm-years)",
        "",
        f"markup_gnr = melast * rev / mate from `02_GNR/03_outdata` (GNR trims differ, so this is "
        f"the intersection: {len(cmp):,} firm-years, {len(cmp) / len(df):.1%} of the markup sample). "
        "Sales weights within the group.",
        "",
        md_table(gy),
        "",
        md_table(gc),
        "",
        f"## Loose level check: Worldscope Spain, {ws['year']}",
        "",
        "Worldscope covers mostly listed firms with consolidated statements, true COGS, all sectors, "
        "and US DLEU elasticities; this sample is unconsolidated Orbis accounts, V = mate + staf, 14 "
        "cells, Spanish elasticities. Only the level is comparable (the paper's +"
        f"{ws['change_since_1980']} is relative to 1980). The bridges change one ingredient at a time: "
        "the elasticity (DLEU's constant) and the firm set (largest firms by sales).",
        "",
        md_table(wst),
        "",
        "## Trim cutoffs and user cost by year",
        "",
        md_table(cuts, 4),
        "",
        "## Per cell x year",
        "",
        "`*` = theta imputed from the first window (2016). Trim counts by cell for the year.",
        "",
        md_table(cyr),
        "",
    ]
    (outd / "markups_report.md").write_text("\n".join(lines))
    logger.info("pooled by year:\n" + py[yr_cols[:13]].round(3).to_string(index=False))
    logger.info("pooled by cell:\n" + pc.round(3).to_string(index=False))
    logger.info("GNR comparison by year:\n" + gy.round(3).to_string(index=False))
    logger.info("GNR comparison by cell:\n" + gc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
