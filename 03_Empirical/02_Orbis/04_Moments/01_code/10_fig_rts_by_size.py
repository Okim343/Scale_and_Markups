"""Step 10: returns to scale and the materials elasticity by firm size (Hubmer et al. 2025, Fig. 2).

Sample: GNR step-2 firm-years with a finite rts in the active window, as 06_alpha_moments.py.
Firm-years are ranked by log sales r into 20 equal-width percentile bins, either pooled across
cells within year or within (ind2d, year), as Hubmer et al.
  (a) mean rts by bin; pooled ranks demeaned by the pooled average, within-industry ranks by the
      (ind2d, year) average.
  (b) mean melast and materials revenue share exp(s) by within-industry bin, both demeaned by the
      (ind2d, year) average.
The within-industry rts curve is the `firm_year_within_ind2d_year` profile of moments_report.md;
the rho_bar target uses firm means within naics4 instead (the `target` profile).

Reads   <gnr.outdata_dir>/firm_year_rts.parquet
Writes  <output.intermediary_dir>/rts_by_size_bins.csv      bin means behind the figure
        02_Drafts/figures/orbis/fig_rts_by_size_orbis.pdf   house style of the model figures

    python 10_fig_rts_by_size.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moments_common import PROJECT_ROOT, dirs, load_config, resolve, setup_logger

BINS = 20
FIG_STEM = PROJECT_ROOT / "02_Drafts" / "figures" / "orbis" / "fig_rts_by_size_orbis"
# Loaded by path so the figure shares the model figures' style without importing the model package.
HOUSE_STYLE = PROJECT_ROOT / "05_Simulation" / "orbis_model" / "figures" / "house_style.py"


def house_style():
    spec = importlib.util.spec_from_file_location("house_style", HOUSE_STYLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(config):
    g, am = config["gnr"], config["alpha_moments"]
    fy = pd.read_parquet(resolve(g["outdata_dir"]) / "firm_year_rts.parquet",
                         columns=["year", "ind2d", am["sample"], am["alpha_col"], "r", "s", "melast"])
    fy = fy[fy[am["sample"]] & np.isfinite(fy[am["alpha_col"]])]
    fy = fy[fy["year"].between(*config["active_window"])]
    return fy.rename(columns={am["alpha_col"]: "rts"}).assign(mshare=lambda d: np.exp(d["s"]))


def binned(fy, group, cols, demean_group):
    """Bin means of `cols` over BINS equal-width percentiles of r ranked within `group`; each
    column is demeaned by its `demean_group` average (overall average when None)."""
    pct = fy.groupby(group)["r"].rank(pct=True)
    b = np.clip(np.ceil(pct * BINS), 1, BINS).astype(int)
    dm = {c: fy[c] - (fy.groupby(demean_group)[c].transform("mean") if demean_group else fy[c].mean())
          for c in cols}
    out = pd.DataFrame({"bin": b, **dm}).groupby("bin").mean()
    out["pct_mid"] = (out.index - 0.5) * 100 / BINS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    config, _ = load_config(ap.parse_args().config)
    logger = setup_logger("10_fig_rts_by_size")
    fy = load(config)
    logger.info(f"{len(fy):,} firm-years, {config['active_window']}")

    within = ["ind2d", "year"]
    pct = fy.groupby(within)["r"].rank(pct=True)
    dm = fy["rts"] - fy.groupby(within)["rts"].transform("mean")
    logger.info(f"RTS gap top 5% - bottom 50%: {dm[pct > 0.95].mean() - dm[pct <= 0.5].mean():.5f}")
    pooled = binned(fy, ["year"], ["rts"], None)
    inner = binned(fy, within, ["rts", "melast", "mshare"], within)
    table = pooled.add_prefix("pooled_").join(inner.add_prefix("within_"))
    table.to_csv(dirs(config)[0] / "rts_by_size_bins.csv")

    hs = house_style()
    hs.configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    line = dict(linewidth=1.4, markersize=3.5, markeredgecolor="white")
    ax1.plot(pooled["pct_mid"], pooled["rts"], "-o", color="#2b3a67", label="Pooled", **line)
    ax1.plot(inner["pct_mid"], inner["rts"], "-o", color="#8b3a3a", label="Within-industry", **line)
    ax1.set_title("Returns to scale", loc="left", fontsize=10)
    ax1.set_xlabel("Percentile of the sales distribution", fontsize=9)
    ax1.set_ylabel("Demeaned returns to scale", fontsize=9)
    ax2.plot(inner["pct_mid"], inner["melast"], "-o", color="#2b3a67", label="Output elasticity", **line)
    ax2.plot(inner["pct_mid"], inner["mshare"], "-o", color="#8b3a3a", label="Revenue share", **line)
    ax2.set_title("Intermediate inputs", loc="left", fontsize=10)
    ax2.set_xlabel("Within-industry percentile of the sales distribution", fontsize=9)
    ax2.set_ylabel("Demeaned elasticity and revenue share", fontsize=9)
    for ax in (ax1, ax2):
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.set_xlim(0, 100)
        hs.style_axes(ax, tick_labelsize=8, axis_linewidth=0.8)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    hs.save_pdf_and_png(fig, FIG_STEM, save_png=False)
    plt.close(fig)
    logger.info(f"wrote {FIG_STEM.relative_to(PROJECT_ROOT)}.pdf")
    logger.info("\n" + table.round(4).to_string())


if __name__ == "__main__":
    main()
