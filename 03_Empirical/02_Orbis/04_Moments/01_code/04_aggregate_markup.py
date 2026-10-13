"""Step 04: aggregate cost-weighted markup mu_cw.

Ports Compustat 00_build_markup_firm_year.py:355-373 (per-year cost-weighted mean) and
07_aggregate_moments.py:76-84 (simple mean over the window years). D4: per year,
  mu_cw_t = sum (V_D + kexp) markup_wi / sum (V_D + kexp)
on the markup file as is (no S1 trims, D13); mu_cw = mean of mu_cw_t over active_window.
Compustat weights cogs + kexp + xsga; Orbis has no SG&A and V = mate + staf already holds all
labour, so the value is the analogue of the SG&A-inclusive markup (markup_target: cogs).
V-weighted and sales-weighted series, the robustness window and the years with an estimated
(not imputed) theta are reported alongside.

Reads   <input.markups_dir>/firm_year_markups.parquet
Writes  <output.intermediary_dir>/aggregate_markup.json   window means by weight
        <output.intermediary_dir>/year_aggregates.csv     the yearly series

    python 04_aggregate_markup.py [--config config_moments_orbis_es.yaml]
"""

import argparse

import pandas as pd

from moments_common import dirs, load_config, resolve, setup_logger, write_json


def yearly(df, mu, weights):
    """Per-year weighted means of mu, one column per named weight column."""
    g = df.assign(**{f"_{k}": df[w] * df[mu] for k, w in weights.items()}).groupby("year")
    s = g[[f"_{k}" for k in weights] + list(dict.fromkeys(weights.values()))].sum()
    out = pd.DataFrame({f"mu_{k}": s[f"_{k}"] / s[w] for k, w in weights.items()})
    out["n"] = g.size()
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("aggregate_markup", config.get("log_level", "INFO"))
    inter, _ = dirs(config)
    col = config["markup"]["col"]
    if config["markup"]["cost_weight"] != "V_D + kexp":
        raise ValueError("markup.cost_weight: only 'V_D + kexp' is implemented")

    df = pd.read_parquet(resolve(config["input"]["markups_dir"]) / "firm_year_markups.parquet",
                         columns=["year", col, "rev_D", "V_D", "kexp"])
    df["Vk"] = df["V_D"] + df["kexp"]
    ya = yearly(df, col, {"cw": "Vk", "V_w": "V_D", "sales_w": "rev_D"})
    ya.to_csv(inter / "year_aggregates.csv", index=False)

    windows = {"active": config["active_window"], "robustness": config["robustness_window"],
               "theta_estimated": config["markup"]["report_years"]}
    res = {"markup_col": col, "cost_weight": config["markup"]["cost_weight"], "windows": {}}
    for tag, (y0, y1) in windows.items():
        w = ya[ya["year"].between(y0, y1)]
        res["windows"][tag] = {"window": [int(y0), int(y1)], "n_years": int(len(w)),
                               **{c: float(w[c].mean()) for c in ["mu_cw", "mu_V_w", "mu_sales_w"]}}
    res["mu_cw"] = res["windows"]["active"]["mu_cw"]
    write_json(inter / "aggregate_markup.json", res)
    logger.info("yearly aggregates:\n" + ya.round(4).to_string(index=False))
    for tag, r in res["windows"].items():
        logger.info(f"{tag:16s} {r['window']}: mu_cw {r['mu_cw']:.4f}, V-weighted {r['mu_V_w']:.4f}, "
                    f"sales-weighted {r['mu_sales_w']:.4f}")
    logger.info("wrote aggregate_markup.json and year_aggregates.csv")


if __name__ == "__main__":
    main()
