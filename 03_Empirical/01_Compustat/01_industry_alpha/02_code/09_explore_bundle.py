"""
09 — Diagnostic figures and summary from the pooled calibration bundle.

This script now focuses on three objects that directly bridge the empirical
pipeline to the simulation inputs:

  (1) how the pooled empirical α distribution is transformed into the model's
      discretized/clipped F(α);
  (2) how α co-moves with GNR productivity ω in the firm cross section; and
  (3) how α co-moves with observed markups in the retained Compustat sample.

It also keeps the aggregate-markup stability figure because that target still
anchors the pooled bundle.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from house_style import configure_style, make_figure, move_ylabel_to_top, save_pdf_and_png, style_axes
from utils import COMPUSTAT_TO_SECTOR_ID, PATHS, ensure_dir, load_config, setup_logger


configure_style()

POOLED_BLUE = "#4477AA"
POOLED_RED = "#C44E52"
POOLED_GREEN = "#228833"
POOLED_GOLD = "#CCB974"
POOLED_GRAY = "#6C6C6C"
OKABE_BLUE = "#0072B2"
OKABE_VERMILLION = "#D55E00"


def _winsorize_within(df: pd.DataFrame, col: str, by: str, pct) -> pd.Series:
    lo, hi = pct

    def _clip(s: pd.Series) -> pd.Series:
        a, b = s.quantile([lo, hi])
        return s.clip(lower=a, upper=b)

    return df.groupby(by)[col].transform(_clip)


def _load_f_alpha_inputs(config: dict, agg_moments: dict, logger):
    cfg = config["F_alpha"]
    alpha_col = cfg.get("alpha_col", "alpha_hybrid_sector_a")
    collapse = cfg.get("collapse", "firm_mean")
    y0 = int(agg_moments["active_window"]["start"])
    y1 = int(agg_moments["active_window"]["end"])

    s1 = pd.read_parquet(PATHS.intermediary / "s1_compustat_firmyear.parquet", columns=["sector_id"])
    retained_sector_ids = sorted(s1["sector_id"].dropna().unique().tolist())

    fy = pd.read_parquet(
        PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet",
        columns=["gvkey", "year", "ind2d", alpha_col],
    ).dropna(subset=[alpha_col])
    fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)].copy()
    fy["sector_id"] = fy["ind2d"].astype(int).map(COMPUSTAT_TO_SECTOR_ID)
    fy = fy.dropna(subset=["sector_id"])
    fy = fy[fy["sector_id"].isin(retained_sector_ids)].copy()

    if collapse == "firm_mean":
        pooled = fy.groupby("gvkey", as_index=False).agg(alpha_unit=(alpha_col, "mean"))
        alpha_source = "window-mean alpha by firm"
    else:
        pooled = fy.rename(columns={alpha_col: "alpha_unit"})[["alpha_unit"]]
        alpha_source = "firm-year alpha"

    winsor_pct = cfg.get("winsor_pct", [0.01, 0.99])
    lo_q, hi_q = np.quantile(pooled["alpha_unit"], winsor_pct)
    pooled["alpha_winsor"] = pooled["alpha_unit"].clip(lo_q, hi_q)
    logger.info(
        f"F(alpha) source cross-section: {len(pooled):,} units; "
        f"winsor bounds [{winsor_pct[0]:.2f}, {winsor_pct[1]:.2f}] -> "
        f"[{lo_q:.4f}, {hi_q:.4f}]"
    )
    return pooled, alpha_source, (float(lo_q), float(hi_q))


def _prepare_alpha_omega_firm_cross_section(config: dict, corr_payload: dict) -> pd.DataFrame:
    cfg = config.get("alpha_omega_corr", {})
    winsor_pct = cfg.get("winsor_pct", [0.01, 0.99])
    alpha_col = config["F_alpha"].get("alpha_col", "alpha_hybrid_sector_a")
    y0, y1 = corr_payload["window"]

    fy = pd.read_parquet(
        PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet",
        columns=["gvkey", "year", "ind2d", alpha_col, "omega"],
    ).dropna(subset=[alpha_col, "omega"])
    fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)].copy()
    firms = fy.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=(alpha_col, "mean"),
        omega=("omega", "mean"),
    )
    firms["alpha"] = _winsorize_within(firms, "alpha", "ind2d", winsor_pct)
    firms["omega"] = _winsorize_within(firms, "omega", "ind2d", winsor_pct)
    firms["alpha_rank"] = firms.groupby("ind2d")["alpha"].rank(pct=True)
    firms["omega_rank"] = firms.groupby("ind2d")["omega"].rank(pct=True)
    return firms


def _prepare_alpha_omega_sales_firm_cross_section(config: dict, agg_moments: dict) -> pd.DataFrame:
    """Firm cross-section of (alpha, omega, log sales), prepared exactly as in
    06b_alpha_sales_corr_hybrid.py: S4b hybrid panel merged onto S1 real sales,
    active window, firm collapse, within-sector winsorization; plus within-sector
    demeaned values and percentile ranks for the figures."""
    cfg = config.get("alpha_sales_corr", config.get("alpha_omega_corr", {}))
    winsor_pct = cfg.get("winsor_pct", [0.01, 0.99])
    alpha_col = config["F_alpha"].get("alpha_col", "alpha_hybrid_sector_a")
    y0 = int(agg_moments["active_window"]["start"])
    y1 = int(agg_moments["active_window"]["end"])

    fy = pd.read_parquet(
        PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet",
        columns=["gvkey", "year", "ind2d", alpha_col, "omega"],
    ).dropna(subset=[alpha_col, "omega"])
    sales = pd.read_parquet(
        PATHS.intermediary / "s1_compustat_firmyear.parquet",
        columns=["gvkey", "year", "sale_D"],
    )
    sales["year"] = sales["year"].astype(fy["year"].dtype)
    fy = fy.merge(sales, on=["gvkey", "year"], how="inner")
    fy = fy[fy["sale_D"] > 0.0]
    fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)].copy()

    fy["log_sales"] = np.log(fy["sale_D"])
    firms = fy.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=(alpha_col, "mean"),
        omega=("omega", "mean"),
        log_sales=("log_sales", "mean"),
    )
    for col in ("alpha", "omega", "log_sales"):
        firms[col] = _winsorize_within(firms, col, "ind2d", winsor_pct)
        firms[f"{col}_dm"] = firms[col] - firms.groupby("ind2d")[col].transform("mean")
        firms[f"{col}_rk"] = firms.groupby("ind2d")[col].rank(pct=True)
    return firms


def fig_alpha_omega_sales_heatmap(firms: pd.DataFrame, out_stem, *, min_cell: int = 10):
    """Mean within-sector demeaned log sales over (alpha, omega) rank deciles."""
    d = firms.copy()
    d["a_dec"] = np.ceil(d["alpha_rk"] * 10).clip(1, 10).astype(int)
    d["o_dec"] = np.ceil(d["omega_rk"] * 10).clip(1, 10).astype(int)

    mean_ls = d.pivot_table(
        index="o_dec", columns="a_dec", values="log_sales_dm", aggfunc="mean"
    ).reindex(index=range(1, 11), columns=range(1, 11))
    counts = d.pivot_table(
        index="o_dec", columns="a_dec", values="log_sales_dm", aggfunc="size"
    ).reindex(index=range(1, 11), columns=range(1, 11))
    mean_ls = mean_ls.where(counts >= min_cell)

    vmax = np.nanmax(np.abs(mean_ls.to_numpy()))
    fig, ax = make_figure(figsize=(7.4, 6.0))
    im = ax.imshow(
        mean_ls.to_numpy(),
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        extent=(0.5, 10.5, 0.5, 10.5),
    )
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_xlabel(r"Scalability $\alpha$ decile (within sector)", fontsize=12)
    ax.set_ylabel(r"Productivity $\ln\hat{z}$ decile (within sector)", fontsize=12)
    ax.set_title("Firm size across scalability and productivity", loc="left", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean log real sales (within-sector, demeaned)", fontsize=11)
    cbar.outline.set_visible(False)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)


def fig_rankrank_sales(firms: pd.DataFrame, out_stem, *, n_bins: int = 20):
    """Mean within-sector sales rank by alpha-rank and omega-rank bins."""
    fig, ax = make_figure(figsize=(8.0, 5.0))

    for col, color, name in (
        ("alpha_rk", OKABE_BLUE, r"$\alpha$ (scalability)"),
        ("omega_rk", OKABE_VERMILLION, r"$\omega$ (TFPQ)"),
    ):
        x = firms[col].to_numpy()
        y = firms["log_sales_rk"].to_numpy()
        bins = pd.qcut(x, n_bins, duplicates="drop")
        g = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin", observed=True)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(0.0, 1.0, 50)
        ax.scatter(g["x"].mean(), g["y"].mean(), s=40, color=color, zorder=3,
                   label=f"{name}, slope = {slope:.2f}")
        ax.plot(xs, slope * xs + intercept, color=color, lw=1.4, ls="--", zorder=2)

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Within-sector percentile rank of firm characteristic", fontsize=12)
    ax.set_ylabel("Mean within-sector sales rank", fontsize=12)
    ax.set_title("Rank-rank relationship with firm size", loc="left", fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=11, loc="upper left")
    style_axes(ax)
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)


def fig_alpha_by_sales_percentile(firms: pd.DataFrame, out_stem, *, n_bins: int = 20):
    """Mean demeaned alpha by sales-percentile bin, pooled vs within-sector ranking.
    Unit-free on both axes (alpha is an elasticity, sales enter only through ranks)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pooled_rk = firms["log_sales"].rank(pct=True)
    fig, ax = make_figure(figsize=(10.0, 6.0))

    for x, a, color, name in (
        (pooled_rk, firms["alpha"] - firms["alpha"].mean(), OKABE_BLUE, "Pooled ranking"),
        (firms["log_sales_rk"], firms["alpha_dm"], OKABE_VERMILLION, "Within-sector ranking"),
    ):
        b = pd.cut(x, edges, include_lowest=True, labels=False)
        g = pd.DataFrame({"x": x, "a": a, "b": b}).groupby("b")
        ax.plot(g["x"].mean(), g["a"].mean(), color=color, lw=1.8, marker="o", ms=7, label=name)

    ax.axhline(0.0, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Firm sales percentile", fontsize=12)
    ax.set_ylabel(r"Mean scalability $\alpha$ (demeaned)", fontsize=12)
    ax.set_title("Scalability by firm size", loc="left", fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    style_axes(ax)
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)


def _prepare_alpha_markup_firm_cross_section(config: dict, agg_moments: dict) -> pd.DataFrame:
    alpha_col = config["F_alpha"].get("alpha_col", "alpha_hybrid_sector_a")
    y0 = int(agg_moments["active_window"]["start"])
    y1 = int(agg_moments["active_window"]["end"])

    rts = pd.read_parquet(
        PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet",
        columns=["gvkey", "year", "ind2d", alpha_col],
    ).dropna(subset=[alpha_col])
    rts = rts[(rts["year"] >= y0) & (rts["year"] <= y1)].copy()

    comp = pd.read_parquet(
        PATHS.intermediary / "s1_compustat_firmyear.parquet",
        columns=["gvkey", "year", "sector_id", "markup"],
    ).dropna(subset=["markup"])
    comp = comp[(comp["year"] >= y0) & (comp["year"] <= y1)].copy()
    comp = comp[comp["markup"] > 0]

    merged = rts.merge(comp, on=["gvkey", "year"], how="inner")
    merged["log_markup"] = np.log(merged["markup"])
    firms = merged.groupby(["sector_id", "gvkey"], as_index=False).agg(
        alpha=(alpha_col, "mean"),
        markup=("markup", "mean"),
        log_markup=("log_markup", "mean"),
    )
    firms["alpha_rank"] = firms.groupby("sector_id")["alpha"].rank(pct=True)
    firms["markup_rank"] = firms.groupby("sector_id")["log_markup"].rank(pct=True)
    return firms


def _binned_profile(df: pd.DataFrame, x: str, y: str, n_bins: int = 20) -> pd.DataFrame:
    work = df[[x, y]].dropna().copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    work["bin"] = pd.cut(
        work[x],
        bins=edges,
        include_lowest=True,
        labels=False,
    )
    prof = work.groupby("bin", as_index=False).agg(
        x_mean=(x, "mean"),
        y_mean=(y, "mean"),
        n=(y, "size"),
    )
    return prof.dropna()


def fig_aggregate_timeseries(
    year_agg: pd.DataFrame,
    agg_moments: dict,
    out_stem,
    *,
    mu_key: str = "agg_mu_cw",
    title: str = "Aggregate markup target across windows",
):
    fig, ax = make_figure(figsize=(10, 6))

    ax.plot(
        year_agg["year"],
        year_agg[mu_key],
        color=POOLED_BLUE,
        lw=2.2,
    )
    aw = agg_moments["active_window"]
    ax.axvspan(aw["start"], aw["end"], color=POOLED_GRAY, alpha=0.12)

    sens = agg_moments["sensitivity_panel"]
    colors = {"primary": POOLED_RED, "robust_a": POOLED_GRAY, "robust_b": POOLED_GREEN}
    for label, w in sens.items():
        ax.hlines(
            w[mu_key],
            w["start"],
            w["end"],
            color=colors.get(label, "black"),
            lw=2.0,
            alpha=0.95,
        )
        ax.text(
            w["end"] + 0.25,
            w[mu_key],
            f"{label}: {w[mu_key]:.3f}",
            color=colors.get(label, "black"),
            fontsize=10,
            va="center",
        )

    style_axes(ax, x_nbins=8, y_nbins=5)
    ax.set_xlabel("Year")
    move_ylabel_to_top(ax, "Aggregate markup")
    ax.set_title(title, loc="left", fontsize=14)
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)


def fig_alpha_raw_vs_winsorized_vs_clipped(
    pooled_alpha: pd.DataFrame,
    F_alpha: pd.DataFrame,
    alpha_source: str,
    winsor_bounds: tuple[float, float],
    out_stem,
):
    fig, ax_r = make_figure(figsize=(10.0, 5.8))

    raw = pooled_alpha["alpha_unit"].to_numpy(dtype=float)
    wins = pooled_alpha["alpha_winsor"].to_numpy(dtype=float)
    nodes = F_alpha.sort_values("u").copy()
    alpha_clip_lo = float(nodes["alpha"].min())
    alpha_clip_hi = float(nodes["alpha"].max())

    cdf_raw = np.cumsum(nodes["weight"].to_numpy()) / nodes["weight"].sum()
    ax_r.step(nodes["alpha_raw"], cdf_raw, where="post", color=POOLED_GOLD, lw=2.0, label="equal-mass nodes (pre-shrink)")
    if "alpha_shrunk" in nodes.columns:
        ax_r.step(nodes["alpha_shrunk"], cdf_raw, where="post", color=POOLED_BLUE, lw=2.0, label="after mean-preserving shrink")
    ax_r.step(nodes["alpha"], cdf_raw, where="post", color=POOLED_RED, lw=2.2, label="final support (shrink → clip)")
    ax_r.axvline(alpha_clip_lo, color=POOLED_RED, lw=1.2, ls="--")
    ax_r.axvline(alpha_clip_hi, color=POOLED_RED, lw=1.2, ls="--")
    style_axes(ax_r, y_nbins=5)
    ax_r.set_xlabel(r"$\alpha$")
    move_ylabel_to_top(ax_r, r"$F(\alpha)$")
    ax_r.set_title("Support construction used by the simulation: winsorize → shrink → clip", loc="left", fontsize=13)
    ax_r.legend(loc="lower right", fontsize=9)

    n_clip = int(nodes["alpha_clipped_flag"].sum())
    fig.suptitle(
        f"Pooled $F(\\alpha)$ pipeline: {alpha_source}; {len(nodes)} equal-mass nodes, {n_clip} clipped",
        x=0.01,
        ha="left",
        fontsize=14,
    )
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)

    return {
        "mean_raw": float(np.mean(raw)),
        "mean_winsor": float(np.mean(wins)),
        "clip_lo": alpha_clip_lo,
        "clip_hi": alpha_clip_hi,
        "winsor_lo": winsor_bounds[0],
        "winsor_hi": winsor_bounds[1],
        "n_nodes": int(len(nodes)),
        "n_clipped": n_clip,
        "q": {
            0.10: float(np.interp(0.10, nodes["u"], nodes["alpha"])),
            0.50: float(np.interp(0.50, nodes["u"], nodes["alpha"])),
            0.90: float(np.interp(0.90, nodes["u"], nodes["alpha"])),
        },
    }


def fig_alpha_omega_rank_dependence(firms: pd.DataFrame, corr_payload: dict, out_stem):
    fig, axes = make_figure(nrows=1, ncols=2, figsize=(11.5, 5.6))
    ax_l, ax_r = axes

    ax_l.scatter(
        firms["alpha_rank"],
        firms["omega_rank"],
        s=9,
        color=POOLED_BLUE,
        alpha=0.12,
        edgecolors="none",
        rasterized=True,
    )
    slope, intercept = np.polyfit(firms["alpha_rank"], firms["omega_rank"], 1)
    fit_x = np.array([0.0, 1.0])
    ax_l.plot(fit_x, intercept + slope * fit_x, color=POOLED_RED, lw=2.2)
    style_axes(ax_l, y_nbins=5)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.set_xlabel(r"Within-sector $\alpha$ percentile")
    move_ylabel_to_top(ax_l, r"Within-sector $\omega$ percentile")
    ax_l.set_title("Firm-level rank dependence", loc="left", fontsize=13)
    ax_l.text(
        0.02,
        0.03,
        f"Firm units: {len(firms):,}; OLS slope: {slope:+.3f}",
        transform=ax_l.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    per_sector = pd.DataFrame.from_dict(corr_payload["per_sector"], orient="index").reset_index(names="ind2d")
    per_sector = per_sector.dropna(subset=["rho_spearman"]).sort_values("n_firms")
    ax_r.hlines(
        per_sector["ind2d"].astype(str),
        xmin=0.0,
        xmax=per_sector["rho_spearman"],
        color=POOLED_GRAY,
        lw=1.0,
        alpha=0.8,
    )
    ax_r.scatter(
        per_sector["rho_spearman"],
        per_sector["ind2d"].astype(str),
        s=np.sqrt(per_sector["n_firms"]) * 6,
        color=POOLED_GREEN,
        alpha=0.9,
    )
    pooled_s = float(corr_payload["pooled_within_spearman"])
    copula_rho = float(corr_payload["implied_gaussian_copula_rho"])
    split_avg = float(corr_payload["split_sample"]["average"])
    ax_r.axvline(pooled_s, color=POOLED_RED, lw=2.0, ls="--")
    style_axes(ax_r, y_nbins=5)
    ax_r.set_xlabel("Spearman correlation")
    move_ylabel_to_top(ax_r, "NAICS2 sector")
    ax_r.set_title("Sector correlations around the pooled target", loc="left", fontsize=13)
    ax_r.text(
        0.02,
        0.02,
        (
            f"Pooled within-sector Spearman = {pooled_s:+.3f}\n"
            f"Implied Gaussian-copula $\\bar\\rho$ = {copula_rho:+.3f}\n"
            f"Split-sample average = {split_avg:+.3f}"
        ),
        transform=ax_r.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )

    fig.suptitle("Rank dependence between scalability and productivity", x=0.01, ha="left", fontsize=14)
    save_pdf_and_png(fig, out_stem)
    plt.close(fig)

    return {
        "pooled_spearman": pooled_s,
        "copula_rho": copula_rho,
        "split_avg": split_avg,
        "n_firms": int(corr_payload["n_firm_units"]),
    }


def write_summary(agg_moments, manifest, f_alpha_stats, alpha_omega_stats, out_path):
    aw = agg_moments["active_window"]
    targets = agg_moments["targets"]

    L: list[str] = []
    L.append("# Pooled Calibration Bundle — Diagnostics")
    L.append("")
    L.append(
        f"**Window:** {aw['start']}–{aw['end']}  ·  "
        f"**Pipeline version:** {agg_moments['pipeline_version']}  ·  "
        f"**Schema:** {manifest.get('schema', 'pooled-bundle')}"
    )
    L.append("")
    L.append("The figures below focus on the objects that bridge the cleaned data to the simulation inputs.")
    L.append("")

    L.append("## 1. Pooled calibration targets")
    L.append("")
    L.append("| Moment | Value | Identifies |")
    L.append("|---|---|---|")
    rows = [
        ("Aggregate cost-weighted markup $\\bar\\mu^{cw}$", targets["mu_cw"], ".4f", "$\\gamma$"),
        ("CR4 sales share", targets["cr4"], ".4f", "sector-size weighted concentration"),
        ("CR20 sales share", targets["cr20"], ".4f", "sector-size weighted concentration"),
        ("Top-1% sales share", targets["top1pct"], ".4f", "right-tail / $\\sigma_z$"),
        ("Top-5% sales share", targets["top5pct"], ".4f", "right-tail / $\\sigma_z$"),
        ("EMX oligopoly slope", targets["emx_slope"], "+.4f", "$\\eta$ given $\\gamma$"),
    ]
    for name, val, fmt, ident in rows:
        L.append(f"| {name} | **{format(float(val), fmt)}** | {ident} |")
    L.append("")

    L.append("## 2. Aggregate markup stability")
    L.append("")
    L.append("*The cost-weighted markup target remains stable across sample windows.* See `fig5_aggregate_markup_timeseries.pdf`.")
    L.append("")

    L.append("## 3. Pooled $F(\\alpha)$ input pipeline")
    L.append("")
    q = f_alpha_stats["q"]
    hybrid_alpha = agg_moments.get("hybrid_alpha", {})
    shrink_target = hybrid_alpha.get("shrink_target", {})
    L.append("| Statistic | Value |")
    L.append("|---|---|")
    L.append(f"| Mean raw pooled $\\alpha$ | {f_alpha_stats['mean_raw']:.4f} |")
    L.append(f"| Mean winsorized pooled $\\alpha$ | {f_alpha_stats['mean_winsor']:.4f} |")
    if "shrink_lambda" in hybrid_alpha:
        L.append(f"| Mean-preserving shrink $\\lambda$ | {float(hybrid_alpha['shrink_lambda']):.2f} |")
    if shrink_target:
        ref_bits = []
        if "target_sd" in shrink_target:
            ref_bits.append(f"target sd {float(shrink_target['target_sd']):.3f}")
        if "reference_sd" in shrink_target:
            ref_bits.append(f"reference sd {float(shrink_target['reference_sd']):.3f}")
        if "reference_p99" in shrink_target:
            ref_bits.append(f"reference P99 {float(shrink_target['reference_p99']):.2f}")
        L.append(
            "| Shrink target | "
            f"{shrink_target.get('source', 'external RTS evidence')}: "
            f"{shrink_target.get('moment', 'RTS dispersion')}"
            + (f" ({'; '.join(ref_bits)})" if ref_bits else "")
            + " |"
        )
    lower_clip_rationale = hybrid_alpha.get("lower_clip_rationale", {})
    if lower_clip_rationale:
        L.append(
            "| Lower clip rationale | "
            f"{lower_clip_rationale.get('source', 'external RTS evidence')}: "
            "conservative support floor, not a firm-level lower-tail target"
            + (
                f"; lowest reported industry-average RTS "
                f"{float(lower_clip_rationale['lowest_reported_industry_average_rts']):.2f}"
                if "lowest_reported_industry_average_rts" in lower_clip_rationale
                else ""
            )
            + " |"
        )
    L.append(f"| Winsor bounds | [{f_alpha_stats['winsor_lo']:.4f}, {f_alpha_stats['winsor_hi']:.4f}] |")
    L.append(f"| Final clip support | [{f_alpha_stats['clip_lo']:.4f}, {f_alpha_stats['clip_hi']:.4f}] |")
    L.append(f"| Equal-mass nodes | {f_alpha_stats['n_nodes']} |")
    L.append(f"| Clipped nodes | {f_alpha_stats['n_clipped']} |")
    L.append(f"| P10 / P50 / P90 of final support | {q[0.10]:.4f} / {q[0.50]:.4f} / {q[0.90]:.4f} |")
    L.append("")
    L.append("This figure makes the simulation input explicit: raw pooled RTS estimates are winsorized, discretized, and then clipped to the admissible support used by the solver. See `fig2_alpha_raw_vs_winsorized_vs_clipped.pdf`.")
    L.append("")

    L.append("## 4. $\\alpha$–$\\omega$ rank dependence")
    L.append("")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    L.append(f"| Pooled within-sector Spearman | **{alpha_omega_stats['pooled_spearman']:+.4f}** |")
    L.append(f"| Implied Gaussian-copula $\\bar\\rho$ | **{alpha_omega_stats['copula_rho']:+.4f}** |")
    L.append(f"| Split-sample average | {alpha_omega_stats['split_avg']:+.4f} |")
    L.append(f"| Firm units | {alpha_omega_stats['n_firms']:,} |")
    L.append("")
    L.append("The right panel reports the pooled within-sector rank dependence used to discipline the ex-ante alignment restriction; the left panel shows the pooled retained-sample sales profile of $\\omega$ and $\\alpha$. See `fig3_alpha_omega_rank_dependence.pdf`.")
    L.append("")

    flags = agg_moments.get("flags", {})
    if flags:
        L.append("## 5. Flags carried from the bundle")
        L.append("")
        for k, v in flags.items():
            L.append(f"- **{k}:** {v}")
        L.append("")

    out_path.write_text("\n".join(L))


_STALE_PREFIXES = (
    "fig1_markup_share_scatter",
    "fig_F_alpha_pooled",
    "fig4_alpha_markup_rank_dependence",
    "fig2_sector_calibration_targets",
    "fig3_firm_quadrant_decomp",
    "fig3b_aggregate_alpha_mu",
    "fig4_F_alpha_quadrature",
    "fig6_quadrant_amplification",
)


def cleanup_stale(out_dir, logger) -> None:
    for prefix in _STALE_PREFIXES:
        for p in out_dir.glob(prefix + "*"):
            logger.info(f"  removing stale artifact: {p.name}")
            p.unlink()


def main() -> None:
    config = load_config()
    logger = setup_logger("EXPLORE", "INFO")

    out_dir = ensure_dir(PATHS.emp_root / "04_figures")
    logger.info(f"writing to {out_dir.relative_to(PATHS.project_root)}")

    F_alpha = pd.read_parquet(PATHS.outdata / "F_alpha.parquet")
    with open(PATHS.outdata / "aggregate_moments.yaml") as f:
        agg_moments = yaml.safe_load(f)
    with open(PATHS.outdata / "manifest.yaml") as f:
        manifest = yaml.safe_load(f)
    with open(PATHS.outdata / "alpha_omega_corr_hybrid.json") as f:
        alpha_omega_corr = json.load(f)

    year_agg = pd.read_parquet(PATHS.intermediary / "s7_year_aggregates.parquet")
    pooled_alpha, alpha_source, winsor_bounds = _load_f_alpha_inputs(config, agg_moments, logger)
    alpha_omega_firms = _prepare_alpha_omega_firm_cross_section(config, alpha_omega_corr)
    cleanup_stale(out_dir, logger)

    logger.info("fig 5: aggregate markup timeseries")
    fig_aggregate_timeseries(year_agg, agg_moments, out_dir / "fig5_aggregate_markup_timeseries")

    logger.info("fig 5b: aggregate SG&A-inclusive markup timeseries")
    fig_aggregate_timeseries(
        year_agg,
        agg_moments,
        out_dir / "fig5b_aggregate_markup_sga_timeseries",
        mu_key="agg_mu_xsga_cw",
        title="Aggregate SG&A-inclusive markup target across windows",
    )

    logger.info("fig 2: pooled alpha raw vs winsorized vs clipped")
    f_alpha_stats = fig_alpha_raw_vs_winsorized_vs_clipped(
        pooled_alpha,
        F_alpha,
        alpha_source,
        winsor_bounds,
        out_dir / "fig2_alpha_raw_vs_winsorized_vs_clipped",
    )

    logger.info("fig 3: alpha-omega rank dependence")
    alpha_omega_stats = fig_alpha_omega_rank_dependence(
        alpha_omega_firms,
        alpha_omega_corr,
        out_dir / "fig3_alpha_omega_rank_dependence",
    )

    logger.info("figs: alpha-omega-sales heatmap and rank-rank profile")
    alpha_omega_sales_firms = _prepare_alpha_omega_sales_firm_cross_section(config, agg_moments)
    fig_alpha_omega_sales_heatmap(
        alpha_omega_sales_firms, out_dir / "fig_heatmap_alpha_omega_logsales"
    )
    fig_rankrank_sales(alpha_omega_sales_firms, out_dir / "fig_rankrank_sales")
    fig_alpha_by_sales_percentile(alpha_omega_sales_firms, out_dir / "fig_alpha_by_sales_percentile")

    logger.info("summary.md")
    write_summary(
        agg_moments,
        manifest,
        f_alpha_stats,
        alpha_omega_stats,
        out_dir / "summary.md",
    )

    for p in sorted(out_dir.glob("*")):
        logger.info(f"  {p.name:46s} {p.stat().st_size / 1e3:7.1f} KB")


if __name__ == "__main__":
    sys.exit(main())
