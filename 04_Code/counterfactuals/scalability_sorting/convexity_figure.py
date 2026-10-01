"""Convexity figure: for all four welfare legs (reallocation, scale, dispersion,
level), the actual path from reverse-sorted to baseline vs. the linear-benchmark
chord between them, overlaid on one set of axes.

Each leg's Delta is min-max normalized to [0, 1] (0 at reverse, 1 at baseline)
so all four share the same two endpoints and the same dashed linear benchmark --
only the sorting-eliminated ``shuffle`` point differs across legs, making its
distance below the benchmark directly comparable leg to leg.

Reads the cached ``scale_channel_decomposition.yaml`` and
``lens_b_decomposition.yaml`` outputs (SPEC sec. 6-family); no new solves.
Purely a presentation aid for the "is this just linear in correlation?"
appendix slide: if Delta scaled linearly in corr(alpha, nu), every leg's
shuffle point would sit exactly on the dashed chord. They instead cluster
well below it, near the flat end -- almost all of the welfare movement
happens on the positive-correlation half, in every leg.

Run from ``04_Code/``::

    python -m counterfactuals.scalability_sorting.convexity_figure \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from steady_state.__main__ import load_config
from steady_state.figures.house_style import configure_style, save_pdf_and_png, style_axes

# (label, color, source dict key in `sources`, per_arrangement endpoint lambda key,
#  three_by_three block key for the cached hom/shuffle/base Delta split)
LEGS = [
    ("Reallocation", "#2b3a67", "scale_channel", "lambda_K", "delta_K"),
    ("Scale", "#c98a3c", "scale_channel", "lambda_scale", "delta_scale"),
    ("Dispersion", "#1f7a72", "lens_b", "lambda_dispersion", "delta_dispersion"),
    ("Level", "#8b3a3a", "lens_b", "lambda_level", "delta_level"),
]


def _leg_deltas(pa: dict, tbt: dict, endpoint_key: str, tbt_key: str) -> tuple[float, float, float]:
    """(Delta_reverse, Delta_shuffle, Delta_base), reading the cached hom/shuffle/base
    split from ``three_by_three`` (matches the Slide 20/22 tables exactly -- the SCALE
    and LEVEL legs are CRN per-draw-differenced, not log1p of an averaged lambda, per
    SPEC sec. 6-iii) and only computing Delta_reverse fresh (a single deterministic
    draw, no averaging subtlety).
    """
    block = tbt[tbt_key]
    d_hom = float(block["common_alpha"]["value"])
    d_shuffle = d_hom + float(block["heterogeneity"]["value"])
    d_base = float(block["delta_base"])
    d_reverse = float(np.log1p(pa["reverse"][endpoint_key]))
    return d_reverse, d_shuffle, d_base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.scalability_sorting.convexity_figure")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    scd_path = (cfg.out_results_dir / "counterfactuals" / "scale_channel_decomposition"
                / "scale_channel_decomposition.yaml")
    with open(scd_path) as stream:
        scale_channel = yaml.safe_load(stream)

    lbd_path = (cfg.out_results_dir / "counterfactuals" / "lens_b_decomposition"
                / "lens_b_decomposition.yaml")
    with open(lbd_path) as stream:
        lens_b = yaml.safe_load(stream)

    sources = {
        "scale_channel": (scale_channel["per_arrangement"], scale_channel["three_by_three"]),
        "lens_b": (lens_b["per_arrangement"], lens_b["three_by_three"]),
    }

    econ_path = (cfg.out_results_dir / "counterfactuals" / "scalability_sorting"
                 / "sorting_economies.parquet")
    econ = pd.read_parquet(econ_path)
    corr_reverse = float(econ.loc[econ["mode"] == "reverse", "corr_av"].iloc[0])
    corr_base = float(econ.loc[econ["mode"] == "baseline", "corr_av"].iloc[0])
    corr_shuffle = float(econ.loc[econ["mode"] == "shuffle", "corr_av"].mean())
    frac_corr = (corr_shuffle - corr_reverse) / (corr_base - corr_reverse)

    configure_style()
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # one shared linear benchmark: after normalizing each leg to [0,1], every
    # leg's chord runs through the same two endpoints.
    ax.plot([corr_reverse, corr_base], [0.0, 1.0], linestyle="--", color="0.6",
            linewidth=1.6, zorder=1, label="Linear benchmark")

    print(f"corr: reverse={corr_reverse:+.4f} shuffle={corr_shuffle:+.4f} base={corr_base:+.4f}"
          f"  ({100*frac_corr:.1f}% of the way)")
    max_frac_delta = 0.0
    for label, color, source, endpoint_key, tbt_key in LEGS:
        pa, tbt = sources[source]
        d_reverse, d_shuffle, d_base = _leg_deltas(pa, tbt, endpoint_key, tbt_key)
        frac_delta = (d_shuffle - d_reverse) / (d_base - d_reverse)
        max_frac_delta = max(max_frac_delta, frac_delta)

        xs = [corr_reverse, corr_shuffle, corr_base]
        ys = [0.0, frac_delta, 1.0]
        ax.plot(xs, ys, linestyle="-", color=color, linewidth=2.1, zorder=2,
                marker="o", markersize=6.5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=1.0,
                label=f"{label} ({100*frac_delta:.0f}%)")

        print(f"{label:13s} Delta: reverse={d_reverse:.5f} shuffle={d_shuffle:.5f} "
              f"base={d_base:.5f}  -> {100*frac_delta:.1f}% of the way")

    ax.annotate("reverse (NAM)", xy=(corr_reverse, 0.0), xytext=(10, 14),
                textcoords="offset points", fontsize=11, ha="left", va="bottom",
                color="0.25")
    ax.annotate("baseline", xy=(corr_base, 1.0), xytext=(-10, -4),
                textcoords="offset points", fontsize=11, ha="right", va="top",
                color="0.25")
    ax.annotate("shuffle", xy=(corr_shuffle, max_frac_delta), xytext=(0, 12),
                textcoords="offset points", fontsize=11, ha="center",
                va="bottom", color="0.25")

    ax.set_xlabel(r"$\mathrm{corr}(\alpha,\nu)$", fontsize=13)
    ax.set_ylabel("Fraction of total welfare movement\n(reverse = 0, baseline = 1)",
                  fontsize=11)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.0%}"))
    style_axes(ax)
    ax.legend(fontsize=10, loc="upper left", frameon=False)

    out_dir = Path(args.out_dir) if args.out_dir else cfg.out_figs_dir / "counterfactuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "sorting_convexity_overlay"
    save_pdf_and_png(fig, stem)
    plt.close(fig)
    print(f"\nwrote figure to {stem}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
