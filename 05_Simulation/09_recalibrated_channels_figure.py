"""Step 09: bridge figure for the refit economies of Table 2.

Walks Delta_total = ln(1 + lambda) from the homogenous economy to the calibrated
baseline in the steps of Table 2's Panel B: common level (Delta_hom), productivity
dispersion (Delta_nu - Delta_hom), RTS dispersion (Delta_alpha - Delta_hom) and
the joint interaction (Delta_base - Delta_nu - Delta_alpha + Delta_hom). Every
step is stacked by the legs of one lens, and the split is exact leg by leg since
each lens sums to Delta_total in every economy. Panel A stacks the dispersion
and level legs, Panel B the misallocation, materials and capital legs; colors
and leg names follow the sorting lens figure (``sorting.readout``).

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
STEPS = (("common", "Common\nlevel"), ("nu", "Productivity\ndispersion"), ("alpha", "RTS\ndispersion"),
         ("inter", "Joint\ninteraction"), ("total", "Calibrated\nbaseline"))
PANELS = (("Panel A. Dispersion and level", ("disp", "level")),
          ("Panel B. Misallocation, materials and capital", ("realloc", "materials", "capital")))
IDENTITY_TOL = 1e-10
WIDTH = 0.62


def step_values(legs: pd.DataFrame, leg: str) -> dict[str, float]:
    """Panel B split of one leg: the four steps sum to the calibrated leg."""
    h, n, a, b = (float(legs.loc[e, f"Delta_{leg}"]) for e in ("hom_both", "hom_alpha", "hom_v", "baseline"))
    return {"common": h, "nu": n - h, "alpha": a - h, "inter": b - n - a + h, "total": b}


def plot_bridge(legs: pd.DataFrame, fig_stem: Path) -> None:
    legs = legs.set_index("economy")
    for lens in ("identity_A", "identity_B"):
        if legs[lens].abs().max() > IDENTITY_TOL:
            raise ValueError(f"{lens} fails: the legs do not sum to Delta_total")
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    ymax = 1.08 * legs.loc["baseline", "Delta_total"]
    for ax, (title, keys) in zip(axes, PANELS):
        vals = {k: step_values(legs, k) for k in keys}
        level = 0.0   # running total the next step starts from
        for i, (step, _) in enumerate(STEPS):
            start = 0.0 if step in ("common", "total") else level
            up, down = start, start
            for k in keys:
                v = vals[k][step]
                ax.bar(i, v, bottom=up if v >= 0 else down, width=WIDTH, color=COLORS[k], alpha=0.85,
                       edgecolor="white", linewidth=0.8, label=NAMES[k] if i == 0 else None)
                if v >= 0:
                    up += v
                else:
                    down += v
            change = sum(vals[k][step] for k in keys)
            text = f"{change:.3f}" if step in ("common", "total") else f"+{change:.3f}"
            ax.text(i, up + 0.004, text, ha="center", va="bottom", fontsize=7, color="0.25")
            if step != "total":
                level = start + change if step != "common" else change
                ax.plot([i + WIDTH / 2, i + 1 - WIDTH / 2], [level, level], color="0.6", linewidth=0.7,
                        linestyle=":")
        ax.set_ylim(0, ymax)
        ax.set_xticks(range(len(STEPS)), [name for _, name in STEPS])
        ax.set_title(title, loc="left", fontsize=10)
        style_axes(ax, tick_labelsize=8, axis_linewidth=0.8)
        ax.tick_params(axis="x", length=0, labelsize=7.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False, fontsize=7.5)
    axes[0].set_ylabel(r"$\Delta = \ln(1+\lambda)$", fontsize=9)
    save_pdf_and_png(fig, fig_stem)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default=ROOT.parent / "out_results" / "zipf" / "recalibrated_anatomy_M5000")
    p.add_argument("--fig-stem", default=ROOT.parent / "02_Drafts" / "figures" / "orbis"
                   / "fig_recalibrated_channels_orbis_zipf")
    args = p.parse_args(argv)
    plot_bridge(pd.read_csv(Path(args.in_dir) / "recalibrated_legs.csv"), Path(args.fig_stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
