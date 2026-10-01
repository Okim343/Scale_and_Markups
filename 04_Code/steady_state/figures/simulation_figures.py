"""Figures built directly from pooled simulation and welfare outputs.

This module is intentionally small for now. It recreates the archived
``welfare_lambda_contribution_grid`` figure for the current pooled output
schema by reading the full market panel directly, instead of requiring the
cluster-side ``welfare_firm_grid.csv`` extract used by the old sector model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
import yaml

from ..config import DEFAULT_CONFIG_PATH, load_config
from .house_style import configure_style, make_figure, save_pdf_and_png, style_axes


N_SCAL_MU_BINS = 5


def _default_results_dir() -> Path:
    return load_config(DEFAULT_CONFIG_PATH).out_results_dir


def default_output_dir(results_dir: Path) -> Path:
    """Put welfare-style figures under ``out_figs/welfare/<results-tag>``."""
    name = Path(results_dir).name
    tag = name[len("welfare_"):] if name.startswith("welfare_") else name
    return load_config(DEFAULT_CONFIG_PATH).out_figs_dir / "welfare" / tag


def _default_market_panel(results_dir: Path) -> Path:
    panel = results_dir / "sim_panel_market.parquet"
    if panel.exists():
        return panel
    try:
        return next(results_dir.glob("sim_panel_*.parquet"))
    except StopIteration as exc:
        raise FileNotFoundError(
            f"No sim_panel_market.parquet or sim_panel_*.parquet found in {results_dir}"
        ) from exc


def _default_planner_panel(results_dir: Path) -> Path | None:
    panel = results_dir / "sim_panel_planner.parquet"
    return panel if panel.exists() else None


def _load_welfare_scalars(results_dir: Path) -> dict[str, float]:
    path = results_dir / "welfare.yaml"
    if not path.exists():
        candidates = sorted(results_dir.glob("welfare_*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No welfare.yaml or welfare_*.yaml found in {results_dir}")
        path = candidates[0]

    with open(path) as stream:
        raw = yaml.safe_load(stream)

    # Current pooled schema is flat. The old sector schema nested the same
    # scalars under welfare/allocations; accepting both makes the script useful
    # for comparisons against archived outputs.
    welfare = raw.get("welfare", raw)
    allocations = raw.get("allocations", {})
    uniform = allocations.get("uniform", {}) if isinstance(allocations, dict) else {}
    mu_bar = welfare.get("mu_bar", uniform.get("mu_bar"))
    if mu_bar is None:
        raise KeyError(f"Could not find mu_bar in {path}")

    required = ("lambda_total", "lambda_level", "lambda_dispersion")
    missing = [key for key in required if key not in welfare]
    if missing:
        raise KeyError(f"Missing welfare scalar(s) in {path}: {', '.join(missing)}")

    return {
        "lambda_total": float(welfare["lambda_total"]),
        "lambda_level": float(welfare["lambda_level"]),
        "lambda_dispersion": float(welfare["lambda_dispersion"]),
        "mu_bar": float(mu_bar),
    }


def _equal_mass_edges(x: np.ndarray, n: int) -> np.ndarray:
    xs = np.sort(np.asarray(x, dtype=float))
    if xs.size == 0:
        raise ValueError("Cannot build bins from an empty array")
    cw = np.linspace(0.0, 1.0, xs.size)
    edges = np.interp(np.linspace(0.0, 1.0, n + 1), cw, xs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_label_precision(edges: np.ndarray) -> int:
    finite = np.asarray(edges[np.isfinite(edges)], dtype=float)
    if finite.size < 2:
        return 2
    min_step = float(np.min(np.diff(np.unique(finite))))
    if not np.isfinite(min_step) or min_step <= 0.0:
        return 2
    return min(max(2, int(np.ceil(-np.log10(min_step))) + 1), 6)


def _alpha_rank_percentile(tilde_alpha: np.ndarray) -> np.ndarray:
    """Return the 0-1 alpha rank percentile from the stored panel column."""
    values = np.asarray(tilde_alpha, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return values
    return ndtr(values)


def build_scalability_markup_grid(
    panel_path: Path | str,
    *,
    n_bins: int = N_SCAL_MU_BINS,
) -> pd.DataFrame:
    """Build the firm-cell grid used by the lambda contribution heatmap.

    Rows are active firms from the market allocation. The pooled model gives
    each firm row uniform population weight, so unweighted sums reproduce the
    old extract's share objects.
    """
    panel_path = Path(panel_path)
    columns = ["tilde_alpha", "active", "markup", "sales", "cost"]
    df = pd.read_parquet(panel_path, columns=columns)
    df = df[df["active"].astype(bool)].copy()
    if df.empty:
        raise ValueError(f"No active firms found in {panel_path}")

    df["wedge"] = df["sales"] - df["cost"]
    wedge = df["wedge"].to_numpy(dtype=float)
    cost = df["cost"].to_numpy(dtype=float)
    sales = df["sales"].to_numpy(dtype=float)
    markup = df["markup"].to_numpy(dtype=float)

    alpha_rank = _alpha_rank_percentile(df["tilde_alpha"].to_numpy(dtype=float))
    scal_edges = np.linspace(0.0, 1.0, n_bins + 1)
    scal_edges[0], scal_edges[-1] = -np.inf, np.inf
    scal_bin = np.clip(np.digitize(alpha_rank, scal_edges[1:-1]), 0, n_bins - 1)

    mu_edges = _equal_mass_edges(markup, n_bins)
    mu_bin = np.clip(np.digitize(markup, mu_edges[1:-1]), 0, n_bins - 1)
    mu_precision = _bin_label_precision(mu_edges[1:-1])

    tot_wedge = float(np.nansum(wedge))
    tot_cost = float(np.nansum(cost))
    tot_sales = float(np.nansum(sales))
    if not np.isfinite(tot_wedge) or abs(tot_wedge) <= 1e-15:
        raise ValueError("Aggregate markup wedge is zero or non-finite")
    if not np.isfinite(tot_cost) or abs(tot_cost) <= 1e-15:
        raise ValueError("Aggregate cost is zero or non-finite")
    if not np.isfinite(tot_sales) or abs(tot_sales) <= 1e-15:
        raise ValueError("Aggregate sales is zero or non-finite")

    scal_pct = np.linspace(0, 100, n_bins + 1)
    rows = []
    for scal_idx in range(n_bins):
        for mu_idx in range(n_bins):
            sel = (scal_bin == scal_idx) & (mu_bin == mu_idx)
            lo, hi = mu_edges[mu_idx], mu_edges[mu_idx + 1]
            lo_label = float(np.nanmin(markup)) if not np.isfinite(lo) else lo
            hi_label = float(np.nanmax(markup)) if not np.isfinite(hi) else hi
            mu_label = (
                f"Q{mu_idx + 1}\n>={lo_label:.{mu_precision}f}"
                if mu_idx == n_bins - 1
                else f"Q{mu_idx + 1}\n{lo_label:.{mu_precision}f}-{hi_label:.{mu_precision}f}"
            )
            rows.append(
                {
                    "scal_bin": scal_idx,
                    "mu_bin": mu_idx,
                    "scal_label": f"{scal_pct[scal_idx]:.0f}-{scal_pct[scal_idx + 1]:.0f}%",
                    "mu_label": mu_label,
                    "wedge_share": float(np.nansum(wedge[sel]) / tot_wedge),
                    "cost_share": float(np.nansum(cost[sel]) / tot_cost),
                    "sales_share": float(np.nansum(sales[sel]) / tot_sales),
                    "mean_markup": float(np.nanmean(markup[sel])) if np.any(sel) else np.nan,
                    "mu_lo": float(lo),
                    "mu_hi": float(hi),
                }
            )
    return pd.DataFrame(rows)


def lambda_contribution_grid(
    grid: pd.DataFrame,
    welfare: dict[str, float],
    out_stem: Path | str,
) -> None:
    """Render the archived lambda-contribution heatmap for pooled outputs."""
    mu_bar = float(welfare["mu_bar"])
    gain_level = float(np.log1p(welfare["lambda_level"]))
    gain_dispersion = float(np.log1p(welfare["lambda_dispersion"]))
    gain_total = gain_level + gain_dispersion
    if abs(gain_total) <= 1e-15:
        raise ValueError("Total log welfare gain is zero")

    g = grid.copy()
    level_weights = g["wedge_share"].to_numpy(dtype=float)
    level_weights = level_weights / level_weights.sum()

    mu_cell = g["mean_markup"].to_numpy(dtype=float)
    dispersion_weights = np.nan_to_num(
        g["cost_share"].to_numpy(dtype=float) * (mu_cell - mu_bar) ** 2
    )
    if dispersion_weights.sum() > 0:
        dispersion_weights = dispersion_weights / dispersion_weights.sum()

    cell_gain = level_weights * gain_level + dispersion_weights * gain_dispersion
    g["pct_total"] = 100.0 * cell_gain / gain_total

    n_scal = int(g["scal_bin"].max()) + 1
    n_mu = int(g["mu_bin"].max()) + 1
    z = np.zeros((n_mu, n_scal))
    for _, row in g.iterrows():
        z[int(row["mu_bin"]), int(row["scal_bin"])] = float(row["pct_total"])

    fig, ax = make_figure(figsize=(9.0, 7.5))
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="magma_r")
    threshold = z.max() * 0.5 if z.size else 0.0
    for i in range(n_mu):
        for j in range(n_scal):
            ax.text(
                j,
                i,
                f"{z[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if z[i, j] > threshold else "black",
            )

    scal_labels = (
        g.drop_duplicates("scal_bin").sort_values("scal_bin")["scal_label"].tolist()
    )
    mu_labels = g.drop_duplicates("mu_bin").sort_values("mu_bin")["mu_label"].tolist()
    ax.set_xticks(range(n_scal))
    ax.set_yticks(range(n_mu))
    ax.set_xticklabels(scal_labels, fontsize=10)
    ax.set_yticklabels(mu_labels, fontsize=10)
    ax.set_xlabel("Scalability quantile (alpha rank)", fontsize=12)
    ax.set_ylabel("Markup mu quintile", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Share of total welfare gain $\lambda_{\mathrm{total}}$ (%)", fontsize=11)
    ax.text(
        0.0,
        1.05,
        "Contribution to the welfare cost of markups, by firm cell",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    ax.text(
        0.0,
        1.005,
        "Sufficient-statistic approximation "
        f"(cells sum to 100% of lambda_total={100 * float(welfare['lambda_total']):.0f}%) "
        "- not a GE counterfactual",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)


def build_markup_series(panel_path: Path | str, *, weight_col: str | None = None) -> np.ndarray:
    """Return the markup of every active firm in a simulated panel.

    If ``weight_col`` is given (e.g. ``"cost"``), also return that column as a
    second array aligned with the markups, for use as histogram/moment weights.
    """
    panel_path = Path(panel_path)
    columns = ["active", "markup"] + ([weight_col] if weight_col else [])
    df = pd.read_parquet(panel_path, columns=columns)
    df = df[df["active"].astype(bool)]
    if df.empty:
        raise ValueError(f"No active firms found in {panel_path}")
    markup = df["markup"].to_numpy(dtype=float)
    if weight_col is None:
        return markup
    return markup, df[weight_col].to_numpy(dtype=float)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    mean = _weighted_mean(x, w)
    return float(np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w)))


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    x_sorted, w_sorted = x[order], w[order]
    cum_weight = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cum_weight, cum_weight[-1] / 2.0))
    return float(x_sorted[idx])


def markup_distribution(
    markup: np.ndarray,
    out_stem: Path | str,
    *,
    regime_label: str = "market",
    n_bins: int = 60,
    reference_lines: dict[str, float] | None = None,
    weights: np.ndarray | None = None,
    weight_label: str = "cost",
) -> None:
    """Render linear- and log-count histogram panels of firm markups.

    The pooled market equilibrium concentrates >99% of firms within a narrow
    band of the mean but has a long thin right tail (highly scalable, highly
    concentrated markets). The linear panel shows the true relative mass (the
    tail is nearly invisible there); the log panel keeps that tail visible
    rather than flattened to a single spike.

    If ``weights`` is given (e.g. per-firm ``cost``), each firm contributes its
    weight rather than a unit count, so the histogram and summary statistics
    reflect economic mass (matching the ``mu_cw`` cost-weighted aggregate
    convention) instead of a simple firm count.
    """
    markup = np.asarray(markup, dtype=float)
    if weights is None:
        mean, median, std = float(np.mean(markup)), float(np.median(markup)), float(np.std(markup))
        mass_label = "Firm count"
    else:
        weights = np.asarray(weights, dtype=float)
        mean, median, std = _weighted_mean(markup, weights), _weighted_median(markup, weights), _weighted_std(markup, weights)
        mass_label = f"{weight_label.capitalize()}-weighted mass"
    bin_edges = np.linspace(markup.min(), markup.max(), n_bins + 1)

    fig, axes = make_figure(ncols=2, figsize=(15.0, 6.0))
    for ax, yscale, ylabel in ((axes[0], "linear", mass_label), (axes[1], "log", f"{mass_label} (log scale)")):
        ax.hist(markup, bins=bin_edges, weights=weights, color="#7B2D8B", edgecolor="white", linewidth=0.3)
        ax.set_yscale(yscale)
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.2, label=f"mean = {mean:.4f}")
        ax.axvline(median, color="black", linestyle=":", linewidth=1.2, label=f"median = {median:.4f}")
        for label, value in (reference_lines or {}).items():
            ax.axvline(value, color="#1b7f3a", linestyle="-", linewidth=1.5, label=label)
        style_axes(ax)
        ax.set_xlabel(r"Markup $\mu$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{yscale} scale", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")

    weighted_note = f", {weight_label}-weighted" if weights is not None else ""
    axes[0].text(
        0.0, 1.12,
        f"Distribution of markups across active firms ({regime_label} equilibrium{weighted_note})",
        transform=axes[0].transAxes, ha="left", va="bottom", fontsize=15,
    )
    axes[0].text(
        0.0, 1.06,
        f"mean={mean:.4f}  median={median:.4f}  std={std:.4f}  max={markup.max():.4f}  "
        f"(n={markup.size:,} active firms)",
        transform=axes[0].transAxes, ha="left", va="bottom", fontsize=10, color="0.35",
    )
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)


def generate_simulation_figures(
    results_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    *,
    panel: Path | str | None = None,
) -> Path:
    """Generate the currently implemented pooled simulation figures."""
    results_dir = Path(results_dir) if results_dir else _default_results_dir()
    out_dir = Path(out_dir) if out_dir else default_output_dir(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    welfare = _load_welfare_scalars(results_dir)
    panel_path = Path(panel) if panel else _default_market_panel(results_dir)
    grid = build_scalability_markup_grid(panel_path)
    lambda_contribution_grid(grid, welfare, out_dir / "welfare_lambda_contribution_grid")

    market_markup, market_cost = build_markup_series(panel_path, weight_col="cost")
    reference_lines = {}
    planner_panel_path = _default_planner_panel(results_dir)
    if planner_panel_path is not None:
        planner_markup, planner_cost = build_markup_series(planner_panel_path, weight_col="cost")
        planner_mean = _weighted_mean(planner_markup, planner_cost)
        reference_lines[f"planner (efficient) = {planner_mean:.4f}"] = planner_mean
    markup_distribution(
        market_markup, out_dir / "markup_distribution",
        reference_lines=reference_lines, weights=market_cost,
    )
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render pooled simulation/welfare figures from full panel outputs."
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--panel", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out = generate_simulation_figures(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        panel=args.panel,
    )
    print(f"wrote simulation figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
