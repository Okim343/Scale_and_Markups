"""Step 09: channel figure for the refit economies of Table 2.

Stacks the Lens B (dispersion, level) and Lens A (misallocation, materials,
capital) legs of Delta_total = ln(1 + lambda) for each economy written by
08_recalibrated_anatomy.py, one horizontal bar per economy and lens. Each lens
sums exactly to Delta_total, so bar lengths are the economies' totals and the
labels inside segments are shares of them. Colors and leg names follow the
sorting lens figure (``sorting.readout``).

Reads   <in-dir>/recalibrated_legs.csv
Writes  <fig-stem>.pdf/.png

    python 09_recalibrated_channels_figure.py --in-dir DIR --fig-stem STEM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from orbis_model.counterfactuals.sorting.readout import COLORS, NAMES
from orbis_model.figures.house_style import configure_style, save_pdf_and_png, style_axes

ROOT = Path(__file__).resolve().parent
ECONOMIES = (("hom_both", "Homogenous"), ("hom_alpha", r"$\nu$-Economy"),
             ("hom_v", r"$\alpha$-Economy"), ("baseline", r"$(\alpha,\nu)$-Economy"))
PANELS = (("Panel A. Lens B", ("disp", "level")),
          ("Panel B. Lens A", ("realloc", "materials", "capital")))
LABEL_MIN = 0.018   # segments narrower than this (Delta units) carry no share label
IDENTITY_TOL = 1e-10


def plot_channels(legs: pd.DataFrame, fig_stem: Path) -> None:
    legs = legs.set_index("economy")
    for lens in ("identity_A", "identity_B"):
        if legs[lens].abs().max() > IDENTITY_TOL:
            raise ValueError(f"{lens} fails: the legs do not sum to Delta_total")
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True, sharex=True)
    ypos = range(len(ECONOMIES))[::-1]
    xmax = 1.18 * legs["Delta_total"].max()
    for ax, (title, keys) in zip(axes, PANELS):
        for y, (label, _) in zip(ypos, ECONOMIES):
            row, left = legs.loc[label], 0.0
            for k in keys:
                width = float(row[f"Delta_{k}"])
                ax.barh(y, width, left=left, height=0.62, color=COLORS[k], alpha=0.85,
                        edgecolor="white", linewidth=0.8, label=NAMES[k] if y == ypos[0] else None)
                if width >= LABEL_MIN:
                    ax.text(left + width / 2, y, f"{100 * row[f'share_{k}']:.0f}%", ha="center",
                            va="center", fontsize=7, color="white")
                left += width
            ax.text(left + 0.004, y, f"{row['Delta_total']:.3f}", ha="left", va="center",
                    fontsize=7.5, color="0.25")
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(r"$\Delta = \ln(1+\lambda)$", fontsize=9)
        style_axes(ax, tick_labelsize=8, axis_linewidth=0.8)
        ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    axes[0].set_yticks(list(ypos), [name for _, name in ECONOMIES])
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    save_pdf_and_png(fig, fig_stem)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default=ROOT.parent / "out_results" / "zipf" / "recalibrated_anatomy_M5000")
    p.add_argument("--fig-stem", default=ROOT.parent / "02_Drafts" / "figures" / "orbis"
                   / "fig_recalibrated_channels_orbis_zipf")
    args = p.parse_args(argv)
    plot_channels(pd.read_csv(Path(args.in_dir) / "recalibrated_legs.csv"), Path(args.fig_stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
