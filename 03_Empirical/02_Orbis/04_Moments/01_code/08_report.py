"""Step 08: moments report, Orbis Spain bundle next to the Compustat bundle.

Reads   <output.outdata_dir>/{aggregate_moments.yaml, manifest.yaml}
        <output.intermediary_dir>/{pooled_targets.json, year_aggregates.csv, aggregate_markup.json,
        emx_slope_moment.yaml, emx_slope_robustness.csv, alpha_moments.json, alpha_moments_by_cell.csv}
        <compustat.outdata_dir>/aggregate_moments.yaml
Writes  <output.outdata_dir>/moments_report.md

    python 08_report.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from moments_common import dirs, load_config, resolve, setup_logger

KEY_TEXT = {
    "mu_cw": "Cost-weighted aggregate markup. Per year, the (V_D + kexp)-weighted mean of the DLEU "
             "markup_wi = theta_wi_c rev_D / V_D over the 03_Markups sample (no further trims); then the "
             "simple mean over 2015-2019. V = mate + staf holds all labour and has no SG&A split, so the "
             "value is the analogue of Compustat's SG&A-inclusive markup (1.185), not of its COGS markup "
             "(1.440). The 2015 thetas are imputed from the 2016 window. The Spain run fits the model's "
             "pure markup mu_cw to this value with markup_target: cogs.",
    "cr4": "Sales share of the 4 largest firms within each (year, NAICS4 market) of the markup "
           "sample; window mean per market; weighted across markets by window-mean market sales.",
    "cr20": "As cr4 with the 20 largest firms (all firms when a market has fewer).",
    "top1pct": "As cr4 with the largest max(1, ceil(0.01 n)) firms; loader-required, weight 0 in the "
               "current calibration.",
    "top5pct": "As top1pct with ceil(0.05 n).",
    "emx_slope": "Slope of the 2015 to 2019 change in the NAICS2 cell inverse V-weighted markup (S1 "
                 "trims: [0.1, 10], then within-year 1/99) on the change in cell HHI, 14 cells, OLS "
                 "with a constant, HC1. Concentration is measured at NAICS4 but the slope at NAICS2, "
                 "where theta varies; see the robustness and decomposition tables.",
    "a": "Capital share in value added, 1/3 as EMX (p.alpha = 1/3). Not estimated: Orbis has no value "
         "added and the GNR kelast / (kelast + lelast) (diagnostic below) is implausibly low.",
    "phi_v": "Value-added weight of the gross-output technology: 1 - mean(melast) / mean(rts) over "
             "GNR step-2 firm-years 2015-2019.",
    "n_firms_cs": "Sum over NAICS4 markets of the window-mean firm count in the markup sample; literal, "
                  "no rescaling.",
    "mean_firms_per_sector": "Window-mean-sales-weighted mean of the NAICS4 window-mean firm counts; "
                             "the model's initial N if its config sets none.",
    "alpha_rts_gap_top5_bottom50": "The new rho_bar target (Hubmer, Chan, Ozkan, Salgado and Hong, "
                                   "Table II): mean demeaned firm-mean rts of the top 5% minus the "
                                   "bottom 50% of the within-NAICS4 distribution of firm-mean log "
                                   "revenue, 2015-2019, no winsorization. Needs a new model moment.",
}


def md_table(df, floatfmt=4):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):          # keeps int columns int, unlike iterrows
        cells = []
        for x in r.values():
            if isinstance(x, (float, np.floating)):
                cells.append("" if np.isnan(x) else f"{x:,.1f}" if abs(x) >= 1000 else f"{x:.{floatfmt}f}")
            elif isinstance(x, (int, np.integer)) and not isinstance(x, bool):
                cells.append(f"{x:,}" if abs(x) >= 10000 else str(x))
            else:
                cells.append(str(x))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def flat(moments):
    """Bundle keys as one flat mapping (targets first)."""
    return {**moments.get("targets", {}), **{k: v for k, v in moments.items()
                                            if not isinstance(v, dict)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("report", config.get("log_level", "INFO"))
    inter, outd = dirs(config)
    moments = yaml.safe_load((outd / "aggregate_moments.yaml").read_text())
    man = yaml.safe_load((outd / "manifest.yaml").read_text())
    cs = yaml.safe_load((resolve(config["compustat"]["outdata_dir"]) / "aggregate_moments.yaml").read_text())
    conc = json.loads((inter / "pooled_targets.json").read_text())
    agg = json.loads((inter / "aggregate_markup.json").read_text())
    ya = pd.read_csv(inter / "year_aggregates.csv")
    emx = yaml.safe_load((inter / "emx_slope_moment.yaml").read_text())
    rob = pd.read_csv(inter / "emx_slope_robustness.csv")
    am = json.loads((inter / "alpha_moments.json").read_text())
    cells = pd.read_csv(inter / "alpha_moments_by_cell.csv")
    a6, r6 = am["active"], am["robustness"]
    tot = pd.read_parquet(inter / "market_year.parquet").groupby("market_def")[["n", "tot_sale"]].sum()
    lost = {m: (int(tot.loc["ind2d", "n"] - r["n"]), 1 - r["tot_sale"] / tot.loc["ind2d", "tot_sale"])
            for m, r in tot.iterrows() if r["n"] < tot.loc["ind2d", "n"]}

    # --- Bundle keys vs Compustat ----------------------------------------------------------
    o, c = flat(moments), flat(cs)
    keys = pd.DataFrame({"key": list(o), "Orbis Spain 2015-2019": [o[k] for k in o],
                         "Compustat 2010-2019": [c.get(k, np.nan) for k in o]})
    keys.loc[len(keys)] = ["mu_cw_sga (Compustat only)", np.nan, c["mu_cw_sga"]]

    # --- Concentration ---------------------------------------------------------------------
    ct = pd.DataFrame([{"market": m, "window": "-".join(map(str, v["window"])),
                        **{k: v[k] for k in ["cr4", "cr20", "top1pct", "top5pct", "n_firms_cs",
                                             "mean_firms_per_sector", "median_firms_per_market"]},
                        "n_markets": v["n_markets"]}
                       for m, ws in conc["by_market"].items() for v in ws.values()])

    # --- Markups ------------------------------------------------------------------------------
    mw = pd.DataFrame([{"window": "-".join(map(str, v["window"])), "role": k,
                        **{x: v[x] for x in ["mu_cw", "mu_V_w", "mu_sales_w"]}}
                       for k, v in agg["windows"].items()])

    # --- EMX ------------------------------------------------------------------------------------
    rob["cell"] = rob["b_hat"].map("{:+.3f}".format) + " (" + rob["se_hc1"].map("{:.3f}".format) + ")"
    rob["variant"] = rob["weight"] + ", " + rob["sample"]
    rt = (rob.pivot_table(index=["market", "spec", "years", "fe"], columns="variant", values="cell",
                          aggfunc="first", sort=False).reset_index().fillna(""))
    rt.columns.name = None
    nobs = rob[(rob["weight"] == "none") & (rob["sample"] == "all")].set_index(
        ["market", "spec", "years", "fe"])["n_obs"]
    rt["n (all)"] = [nobs.get(tuple(r)) for r in rt[["market", "spec", "years", "fe"]].to_numpy()]
    dec = emx["decomposition"]
    dt = pd.DataFrame([{"level": lvl, "regression": k, "b_hat": v["b_hat"], "se_hc1": v["se_hc1"],
                        "n": v["n_obs"]} for lvl in ["ind2d", "naics4"] for k, v in dec[lvl].items()
                       if isinstance(v, dict)])

    # --- Alpha moments ---------------------------------------------------------------------------
    def get(d, path):
        for k in path.split("."):
            d = d[k]
        return d
    paths = {"alpha_rts_gap_top5_bottom50": "alpha_rts_gap_top5_bottom50",
             "phi_v": "phi_v", "a_gnr = kelast / (kelast + lelast)": "a_gnr",
             **{f"mean {k}": f"gnr_means.{k}" for k in ["melast", "kelast", "lelast", "rts"]}}
    at = pd.DataFrame([{"moment": k, "2015-2019": get(a6, p), "2015-2023": get(r6, p)}
                       for k, p in paths.items()])
    labels = {"firm_year_within_ind2d_year": "firm-years, within ind2d-year (Salgado literal)",
              "firm_year_within_naics4_year": "firm-years, within naics4-year",
              "firm_mean_within_ind2d": "firm means, within ind2d",
              "firm_mean_within_naics4": "firm means, within naics4 (target)"}
    gt = pd.DataFrame([{"basis": lab, **{f"{t} {w}": d["rts_gap_variants"][k][t]
                                         for w, d in [("2015-19", a6), ("2015-23", r6)]
                                         for t in ["top1", "top5", "top10"]}}
                       for k, lab in labels.items()])
    prof = pd.concat({k: pd.DataFrame(v).set_index("bin") for k, v in a6["profiles"].items()}, axis=1)
    prof.columns = [f"{a}: {b}" for a, b in prof.columns]
    prof = prof.reset_index()
    bs = a6["rts_gap_bootstrap"]

    lines = [
        "# Orbis Spain calibration moments: report",
        "",
        f"Generated from commit `{man['commit']}` (code dirty: {man['code_dirty']}), bundle created "
        f"{man['created_utc']}. Decisions: `02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md`. "
        f"Active window {config['active_window'][0]}-{config['active_window'][1]}; concentration market "
        f"{man['markets']['concentration']} ({man['markets']['n_markets']} markets); emx market "
        f"{man['markets']['emx']}; markup_target {man['markup_target']}.",
        "",
        "## Bundle keys, Orbis Spain vs Compustat",
        "",
        "Compustat values from `01_Compustat/01_industry_alpha/03_outdata/aggregate_moments.yaml`. "
        "Compustat mu_cw is the COGS markup; its SG&A-inclusive mu_cw_sga is the comparable object for "
        "the Orbis V-basis markup.",
        "",
        md_table(keys),
        "",
        "## Concentration by market and window",
        "",
        "Markup sample, 2015-2023. Pooled with window-mean market sales weights. "
        + "".join(f"{m} leaves out {n:,} firm-years without a code ({s:.1%} of sales). "
                  for m, (n, s) in lost.items()),
        "",
        md_table(ct),
        "",
        "## Aggregate markup by year and weight",
        "",
        "markup_wi, no trims. `mu_cw` = (V + kexp)-weighted (bundle), `mu_V_w` = V-weighted, "
        "`mu_sales_w` = sales-weighted. 2015 uses the 2016-window theta (imputed).",
        "",
        md_table(ya),
        "",
        md_table(mw),
        "",
        "## EMX slope",
        "",
        f"Primary: {emx['market']} long difference {emx['window'][0]}-{emx['window'][1]}: "
        f"**{emx['value']:+.4f}** (HC1 se {emx['se']:.4f}, n {emx['n']}, R2 {emx['r_squared']:.3f}). "
        "Compustat: -0.5096 (NAICS2, 2010-2019).",
        "",
        "Robustness, b (HC1 se). `ld` = long difference; `ld_2yr_avg` = two-year averaged endpoints; "
        "`stacked_Nyr` = stacked N-year differences over 2015-2023. Columns: weighting (none, or WLS "
        "on start-year market sales) and sample (all markets, or markets with at least "
        f"{config['markets']['low_firm_threshold']} firms in every year).",
        "",
        md_table(rt),
        "",
        f"Decomposition ({dec['window'][0]}-{dec['window'][1]}). At ind2d, 1/mu = (1/theta)(V/rev) on the "
        "untrimmed sample: `theta_part` and `cost_share_part` are the two terms of Delta(1/mu) with "
        "endpoint-mean weights (they add up to `total_untrimmed`); `d_inv_theta` and `d_V_over_rev` "
        "are the raw changes. At naics4: `within_cell_fe` adds ind2d dummies; `between_cell_means` "
        "regresses start-year-sales-weighted cell means of the naics4 changes across the 14 cells.",
        "",
        md_table(dt),
        "",
        f"Share of the naics4 Var(Delta HHI) within cells: "
        f"{dec['naics4']['share_var_dhhi_within_cells']:.3f}; sd of Delta HHI "
        f"{dec['naics4']['sd_dhhi']:.3f} at naics4 vs {dec['naics4']['sd_dhhi_ind2d']:.3f} at ind2d.",
        "",
        "## Alpha moments",
        "",
        f"GNR step-2 firm-years with finite rts: {a6['firm_years']:,} ({a6['firms']:,} firms) in 2015-2019; "
        f"{r6['firm_years']:,} ({r6['firms']:,} firms) in 2015-2023.",
        "",
        md_table(at),
        "",
        "RTS gap variants (mean demeaned rts of the top share minus the bottom 50% of the within-group "
        "revenue distribution). Firm-bootstrap se of the target "
        f"({bs['reps']} reps, resampling bvdid): {bs['se']:.4f}.",
        "",
        md_table(gt),
        "",
        "Within-market revenue profile, 2015-2019: mean demeaned rts, melast and lelast by vigintile of "
        "the within-group revenue rank (bin 1 = smallest 5%). `target` = firm means within naics4; "
        "`firm_year_within_ind2d_year` = Salgado's basis.",
        "",
        md_table(prof),
        "",
        "Per NAICS2 cell, 2015-2019: firm units (firm means within naics4) and their RTS gap, "
        "restricted to the cell; GNR elasticity means over firm-years.",
        "",
        md_table(cells, 3),
        "",
        "## Definitions",
        "",
        *[f"**{k}** ({o[k]:.6g}). {KEY_TEXT[k]}\n" for k in KEY_TEXT],
    ]
    (outd / "moments_report.md").write_text("\n".join(lines))
    logger.info(f"wrote {outd / 'moments_report.md'}")


if __name__ == "__main__":
    main()
