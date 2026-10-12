"""Step 06: time series of the aggregate markup, sales- and cost-weighted (as Global Markups Fig. 1).

mu_t = sum_i w_it * markup_wi_it within year t over all cells, with w = rev_D / sum(rev_D)
(sales, DLEU baseline) or w = V_D / sum(V_D) (cost: V = mate + staf, the COGS analogue; DLEU
mu_cogs_w). House style: dashed annual series and a solid centred 3-year moving average
(the series has 9 years, so ticks every 2 years rather than by decade).

Reads   <output.outdata_dir>/firm_year_markups.parquet
Writes  <output.outdata_dir>/aggregate_markup.png and aggregate_markup.csv (gitignored)

    python 06_plot_aggregate_markup.py [--config config_markups_orbis_es.yaml]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402

from markups_common import load_config, resolve, setup_logger  # noqa: E402

MA_WINDOW = 3


def set_style():
    # EB Garamond lives in the user font folder, which matplotlib's cache may not have seen.
    for f in (Path.home() / "Library" / "Fonts").glob("EBGaramond-*.ttf"):
        fm.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = ["EB Garamond", "Garamond", "serif"]
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.titlelocation"] = "left"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("plot", config.get("log_level", "INFO"))
    outd = resolve(config["output"]["outdata_dir"])

    df = pd.read_parquet(outd / "firm_year_markups.parquet", columns=["year", "rev_D", "V_D", "markup_wi"])
    agg = pd.DataFrame({
        "mu_sales_w": df.groupby("year").apply(lambda g: np.average(g["markup_wi"], weights=g["rev_D"]),
                                               include_groups=False),
        "mu_cost_w": df.groupby("year").apply(lambda g: np.average(g["markup_wi"], weights=g["V_D"]),
                                              include_groups=False),
    })
    for c in ["mu_sales_w", "mu_cost_w"]:
        agg[f"{c}_ma{MA_WINDOW}"] = agg[c].rolling(MA_WINDOW, center=True, min_periods=MA_WINDOW).mean()
    agg.index.name = "year"
    agg.reset_index().to_csv(outd / "aggregate_markup.csv", index=False)

    set_style()
    years = agg.index.to_numpy()
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for col, color, label in [("mu_sales_w", "red", "Sales weighted"),
                              ("mu_cost_w", "green", "Cost weighted")]:
        ax.plot(years, agg[col], color=color, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.75)
        ax.plot(years, agg[f"{col}_ma{MA_WINDOW}"], color=color, linewidth=2.5, linestyle="-", label=label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)
    ax.set_title("Aggregate markup, Spain", fontsize=16)
    ax.set_xticks(np.arange(years.min(), years.max() + 1, 2))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    vals = agg[["mu_sales_w", "mu_cost_w"]].to_numpy()
    lo, hi = vals.min(), vals.max()
    pad = 0.06 * (hi - lo) if hi > lo else 0.02
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(outd / "aggregate_markup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("aggregate markup:\n" + agg.round(3).to_string())
    logger.info(f"wrote {outd / 'aggregate_markup.png'}")


if __name__ == "__main__":
    main()
