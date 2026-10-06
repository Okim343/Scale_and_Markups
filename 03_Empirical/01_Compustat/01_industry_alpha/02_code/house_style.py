"""Matplotlib house-style helpers for empirical bundle figures.

Adapted from the shared house-style helper so the empirical plotting scripts
reuse the same styling logic as the steady-state figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, get_data_path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


CMR10_PATH = Path(get_data_path()) / "fonts" / "ttf" / "cmr10.ttf"
EB_GARAMOND_NAME = "EB Garamond"
EB_GARAMOND_PATH = Path.home() / "Library" / "Fonts" / "EBGaramond-Regular.ttf"


def configure_style() -> str:
    """Apply the house style and return the selected font family."""
    if EB_GARAMOND_PATH.exists():
        font_manager.fontManager.addfont(str(EB_GARAMOND_PATH))
        family = EB_GARAMOND_NAME
    elif CMR10_PATH.exists():
        font_manager.fontManager.addfont(str(CMR10_PATH))
        family = "CMR10"
    else:
        family = "serif"

    plt.rcParams["font.family"] = family
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["axes.titleweight"] = "regular"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.constrained_layout.use"] = True
    return family


def make_figure(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (10.0, 6.0),
    sharex: bool = False,
    sharey: bool = False,
):
    """Create a white figure matching the reference proportions."""
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    fig.patch.set_facecolor("white")
    axes_arr = axes if hasattr(axes, "ravel") else [axes]
    for ax in axes_arr.ravel() if hasattr(axes_arr, "ravel") else axes_arr:
        ax.set_facecolor("white")
    return fig, axes


def style_axes(
    ax: Axes,
    *,
    y_as_percent: bool = False,
    x_nbins: int | None = None,
    y_nbins: int | None = None,
    tick_labelsize: int = 12,
    axis_linewidth: float = 1.2,
) -> None:
    """Apply the minimalist axis treatment used by the house style."""
    if x_nbins is not None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=x_nbins, prune=None))
    if y_nbins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins, prune=None))
    if y_as_percent:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    ax.tick_params(labelsize=tick_labelsize, width=axis_linewidth)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(axis_linewidth)
        ax.spines[side].set_color("black")


def move_ylabel_to_top(ax: Axes, text: str) -> None:
    """Place the y-axis label at the top-left rather than centered."""
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        clip_on=False,
    )


def save_pdf_and_png(
    fig: Figure,
    output_stem: str | Path,
    *,
    save_pdf: bool = True,
    save_png: bool = True,
) -> None:
    """Save the styled figure as PDF first, with optional PNG companion."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if save_pdf:
        fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    if save_png:
        fig.savefig(output_stem.with_suffix(".png"), format="png", bbox_inches="tight", dpi=180)
