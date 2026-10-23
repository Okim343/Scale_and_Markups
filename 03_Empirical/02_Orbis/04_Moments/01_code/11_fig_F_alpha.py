"""Step 11: figure of the pooled scalability distribution F(alpha) and the model's alpha cap.

Plots the CDF of the firm units behind F(alpha) (window-mean GNR rts per firm, before the
winsorization of 01_build_F_alpha.py) against the 500-node support in 03_outdata, and marks the
cap 0.99 gamma / (gamma - 1) that the model applies to the draws, with gamma from the calibrated
parameters.

Reads   <gnr.outdata_dir>/firm_year_rts.parquet
        <output.outdata_dir>/F_alpha.parquet
        04_Estimation/out_results/calibrated_params.yaml        gamma, model.alpha_cap_factor
Writes  02_Drafts/figures/orbis/fig_F_alpha_orbis.pdf

    python 11_fig_F_alpha.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from moments_common import PROJECT_ROOT, dirs, load_config, resolve, setup_logger

FIG_STEM = PROJECT_ROOT / "02_Drafts" / "figures" / "orbis" / "fig_F_alpha_orbis"
PARAMS = PROJECT_ROOT / "04_Estimation" / "out_results" / "calibrated_params.yaml"
HOUSE_STYLE = PROJECT_ROOT / "05_Simulation" / "orbis_model" / "figures" / "house_style.py"


def house_style():
    spec = importlib.util.spec_from_file_location("house_style", HOUSE_STYLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def firm_units(config):
    """Window-mean alpha per firm, as steps 1-2 of 01_build_F_alpha.py."""
    g, fa = config["gnr"], config["F_alpha"]
    fy = pd.read_parquet(resolve(g["outdata_dir"]) / "firm_year_rts.parquet",
                         columns=[g["id_col"], "year", "in_step2_sample", fa["alpha_col"]])
    fy = fy[fy["in_step2_sample"] & np.isfinite(fy[fa["alpha_col"]]) & fy["year"].between(*fa["window"])]
    return fy.groupby(g["id_col"])[fa["alpha_col"]].mean().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    config, _ = load_config(ap.parse_args().config)
    logger = setup_logger("11_fig_F_alpha")

    units = np.sort(firm_units(config))
    nodes = pd.read_parquet(dirs(config)[1] / "F_alpha.parquet").sort_values("alpha")
    p = yaml.safe_load(PARAMS.read_text())
    gamma = p["parameters"]["gamma"]
    cap = p["model"]["alpha_cap_factor"] * gamma / (gamma - 1)
    at_cap = float(nodes.loc[nodes["alpha"] >= cap, "weight"].sum() / nodes["weight"].sum())
    logger.info(f"{len(units):,} firms; cap {cap:.4f} (gamma {gamma:.4f}); node mass at cap {at_cap:.4f}")

    hs = house_style()
    hs.configure_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(units, np.arange(1, len(units) + 1) / len(units), color="#c98a3c", linewidth=1.4,
            label="Firms (window-mean $\\alpha$)")
    ax.step(nodes["alpha"], nodes["weight"].cumsum() / nodes["weight"].sum(), where="post",
            color="#2b3a67", linewidth=1.4, label="Support (500 nodes, winsorized at 1/99)")
    ax.axvline(cap, color="0.55", linestyle="--", linewidth=0.9)
    ax.text(cap, 0.04, f" cap $0.99\\,\\gamma/(\\gamma-1) = {cap:.3f}$", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7.5, color="0.35")
    ax.axvline(1.0, color="0.7", linewidth=0.8)
    ax.set_xlim(0.6, 1.4)
    ax.set_xlabel("Returns to scale $\\alpha$", fontsize=9)
    ax.set_ylabel("Cumulative share of firms", fontsize=9)
    hs.style_axes(ax, tick_labelsize=8, axis_linewidth=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    hs.save_pdf_and_png(fig, FIG_STEM, save_png=False)
    plt.close(fig)
    logger.info(f"wrote {FIG_STEM.relative_to(PROJECT_ROOT)}.pdf")


if __name__ == "__main__":
    main()
