"""Sorting-sweep figure, summary yaml and a source-keyed results.md.

Ported from ``04_Code/counterfactuals/sorting_curve/figure.py``. The curve is
the copula sweep only (one solved economy per ``rho_bar``). Permutation and
homogeneous-alpha economies enter the summary and the table, not the figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .sweep import LEGS, REGIMES

NAMES = {"realloc": "Misallocation", "materials": "Materials", "capital": "Capital",
         "total": "Total", "disp": "Dispersion", "level": "Level"}
COLORS = {"realloc": "#2b3a67", "materials": "#6b8e23", "capital": "#c98a3c",
          "total": "#222222", "disp": "#1f7a72", "level": "#8b3a3a"}
DIAG_COLUMNS = ("mu_cw_ME", "K_PE_over_K_ME", "cap_sales_share_ME", "top_share_median_ME", "cr4_ME")


def copula_curve(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.kind.eq("copula")].sort_values("corr_av").reset_index(drop=True)


PLOT_CORR_GAP = 0.02  # sweep points this close to the calibrated corr duplicate it visually


def plotted_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Copula curve for the figures: drops sweep points on top of the calibrated economy."""
    curve = copula_curve(df)
    base_corr = df.set_index("label").loc["baseline", "corr_av"]
    near = curve.label.ne("baseline") & (curve.corr_av - base_corr).abs().lt(PLOT_CORR_GAP)
    return curve[~near].reset_index(drop=True)


def sorting_split(df: pd.DataFrame) -> dict:
    """Delta_base = Delta_hom + (Delta_shuf - Delta_hom) + (Delta_base - Delta_shuf), per leg.

    ``Delta_shuf`` averages the shuffle replicates on the Delta scale. The
    common-alpha term uses the sector harmonic mean; the cost-weighted variant
    is reported alongside.
    """
    labels = set(df.label)
    shuffles = df[df.label.str.startswith("shuffle_")]
    if not {"baseline", "hom_sector"} <= labels or shuffles.empty:
        return {}
    base = df.set_index("label").loc["baseline"]
    hom = df.set_index("label").loc["hom_sector"]
    hom_cost = df.set_index("label").loc["hom_cost"] if "hom_cost" in labels else None
    out = {"n_shuffle": int(len(shuffles))}
    for leg in LEGS:
        key = f"Delta_{leg}"
        d_base, d_hom = float(base[key]), float(hom[key])
        d_shuf = float(shuffles[key].mean())
        entry = {
            "Delta_base": d_base, "Delta_shuf": d_shuf, "Delta_hom": d_hom,
            "common": d_hom, "heterogeneity": d_shuf - d_hom, "sorting": d_base - d_shuf,
            "Delta_shuf_replicate_range": float(shuffles[key].max() - shuffles[key].min()),
        }
        if d_base != 0.0:
            entry.update(share_common=d_hom / d_base, share_heterogeneity=(d_shuf - d_hom) / d_base,
                         share_sorting=(d_base - d_shuf) / d_base)
        if hom_cost is not None:
            entry["Delta_hom_cost"] = float(hom_cost[key])
        out[leg] = entry
    return out


def copula_vs_permutation(df: pd.DataFrame) -> list[dict]:
    """Gap of each permutation point to the copula curve interpolated at its correlation.

    Points outside the copula curve's correlation range are reported without a gap.
    """
    curve = copula_curve(df)
    out = []
    for _, row in df[df.kind.eq("permutation")].sort_values("corr_av").iterrows():
        inside = bool(curve.corr_av.min() <= row.corr_av <= curve.corr_av.max())
        entry = {"label": row.label, "corr_av": float(row.corr_av), "inside_copula_range": inside}
        for leg in LEGS:
            key = f"Delta_{leg}"
            entry[f"gap_{key}"] = (float(row[key] - np.interp(row.corr_av, curve.corr_av, curve[key]))
                                   if inside else None)
        out.append(entry)
    return out


def secant_slopes(df: pd.DataFrame) -> dict:
    curve = copula_curve(df)
    if len(curve) < 2:
        return {}
    dx = np.diff(curve.corr_av.to_numpy())
    return {leg: [float(s) for s in np.diff(curve[f"Delta_{leg}"].to_numpy()) / dx] for leg in LEGS}


def summarize(df: pd.DataFrame, complete: bool) -> dict:
    regs = list(REGIMES)
    return {
        "complete": bool(complete),
        "n_rows": int(len(df)),
        "all_converged": bool(df[[f"converged_{k}" for k in regs]].all().all()),
        "max_labor_deviation": float(df[[f"L_{k}" for k in regs]].sub(1.0).abs().max().max()),
        "max_consumption_welfare_gap": float(
            df[[f"consumption_gap_{leg}" for leg in LEGS]].abs().max().max()),
        "max_identity_residual": float(df[["identity_A", "identity_B"]].abs().max().max()),
        "secant_slopes": secant_slopes(df),
        "sorting_split": sorting_split(df),
        "copula_vs_permutation": copula_vs_permutation(df),
    }


def plot_curve(df: pd.DataFrame, fig_stem: Path) -> bool:
    """Plot welfare gains and lifetime welfare at every copula sweep point.

    Both lifetime-welfare series use calibrated market welfare as their common
    reference. Saved W values include discounted utility and labor disutility.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ...figures.house_style import configure_style, save_pdf_and_png, style_axes

    curve = plotted_curve(df)
    if len(curve) < 2:
        return False
    base = df.set_index("label").loc["baseline"]
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    ax.plot(curve.corr_av, curve.lambda_total, "-o", color=COLORS["total"], linewidth=1.4,
            markersize=3.5, markeredgecolor="white", label=NAMES["total"])
    ax.set_title("Panel A. Welfare gain", loc="left", fontsize=10)
    style_axes(ax, y_as_percent=True, tick_labelsize=8, axis_linewidth=0.8)
    ax.set_ylabel(r"Consumption-equivalent $\lambda$", fontsize=9)

    ax = axes[1]
    for key, name, color in (("ME", "Market", "#2b3a67"), ("PE", "Planner", "#c98a3c")):
        ax.plot(curve.corr_av, curve[f"W_{key}"] - base.W_ME, "-o", color=color,
                linewidth=1.4, markersize=3.5, markeredgecolor="white", label=name)
    ax.set_title("Panel B. Lifetime welfare", loc="left", fontsize=10)
    ax.set_ylabel("Difference from calibrated market", fontsize=9)
    style_axes(ax, tick_labelsize=8, axis_linewidth=0.8)

    for ax in axes:
        ax.axvline(base.corr_av, color="0.55", linestyle="--", linewidth=0.9)
        ax.text(base.corr_av, 1.0, " calibrated", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5, color="0.35")
        ax.set_xlabel(r"$\mathrm{corr}(\alpha,\nu)$ over active firms", fontsize=9)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    save_pdf_and_png(fig, fig_stem)
    plt.close(fig)
    return True


def plot_lenses(df: pd.DataFrame, fig_stem: Path) -> bool:
    """Stack the Lens B (dispersion, level) and Lens A (reallocation, materials, capital)
    legs of Delta_total at every copula sweep point.

    In Delta = ln(1 + lambda) units each lens sums exactly to Delta_total.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ...figures.house_style import configure_style, save_pdf_and_png, style_axes

    curve = plotted_curve(df)
    if len(curve) < 2:
        return False
    base = df.set_index("label").loc["baseline"]
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    panels = (("Panel A. Dispersion and level", ("disp", "level")),
              ("Panel B. Misallocation, materials and capital", ("realloc", "materials", "capital")))
    for ax, (title, legs) in zip(axes, panels):
        ax.stackplot(curve.corr_av, *[curve[f"Delta_{k}"] for k in legs],
                     colors=[COLORS[k] for k in legs], alpha=0.85, labels=[NAMES[k] for k in legs],
                     edgecolor="white", linewidth=0.6)
        ax.plot(curve.corr_av, curve.Delta_total, "-o", color=COLORS["total"], linewidth=1.2,
                markersize=3.0, markeredgecolor="white", label=NAMES["total"])
        ax.axvline(base.corr_av, color="0.55", linestyle="--", linewidth=0.9)
        ax.text(base.corr_av, 1.0, " calibrated", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5, color="0.35")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(r"$\mathrm{corr}(\alpha,\nu)$ over active firms", fontsize=9)
        style_axes(ax, tick_labelsize=8, axis_linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False, fontsize=7.5)
    axes[0].set_ylabel(r"$\Delta = \ln(1+\lambda)$", fontsize=9)
    save_pdf_and_png(fig, fig_stem)
    plt.close(fig)
    return True


def plot_levels(df: pd.DataFrame, fig_stem: Path) -> bool:
    """ln C of the market and the planner relative to the baseline, against corr(alpha, v).

    With L = 1 and chi frozen, welfare is ln C / (1 - beta) up to a constant common to
    all arrangements, so the vertical gap between the two lines minus its baseline
    value is the change in Delta_total.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ...figures.house_style import configure_style, save_pdf_and_png, style_axes

    curve = plotted_curve(df)
    if len(curve) < 2:
        return False
    perm = df[df.kind.eq("permutation")]
    base = df.set_index("label").loc["baseline"]
    lines = (("ME", "Market", "#2b3a67"), ("PE", "Planner", "#c98a3c"))
    configure_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    rel = lambda frame, key: np.log(frame[f"C_{key}"] / base[f"C_{key}"])
    ax.fill_between(curve.corr_av, rel(curve, "ME"), rel(curve, "PE"), color="0.88", zorder=0)
    for key, name, color in lines:
        ax.plot(curve.corr_av, rel(curve, key), "-o", color=color, linewidth=2.0, markersize=5.5,
                markeredgecolor="white", label=name)
        ax.scatter(perm.corr_av, rel(perm, key), s=42, facecolors="none", edgecolors=color,
                   linewidths=1.4, zorder=4)
    ax.scatter([], [], s=42, facecolors="none", edgecolors="0.3", linewidths=1.4, label="Permutations")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(base.corr_av, color="0.55", linestyle="--", linewidth=1.2)
    ax.text(base.corr_av, 1.0, " calibrated", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10, color="0.35")
    ax.set_xlabel(r"$\mathrm{corr}(\alpha,\nu)$ over active firms", fontsize=12)
    ax.set_ylabel(r"$\ln C - \ln C_{\mathrm{calibrated}}$", fontsize=12)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save_pdf_and_png(fig, fig_stem)
    plt.close(fig)
    return True


def write_artifacts(df: pd.DataFrame, provenance: dict, out_dir: Path, fig_dir: Path,
                    *, complete: bool) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    summary = summarize(df, complete)
    summary["figure_written"] = plot_curve(df, fig_dir / "sorting_curve")
    summary["levels_figure_written"] = plot_levels(df, fig_dir / "sorting_levels")
    summary["lenses_figure_written"] = plot_lenses(df, fig_dir / "sorting_lenses")
    tmp = out_dir / "sorting.yaml.tmp"
    tmp.write_text(yaml.safe_dump({"provenance": provenance, "summary": summary}, sort_keys=False))
    tmp.replace(out_dir / "sorting.yaml")
    (out_dir / "results.md").write_text(results_markdown(df, provenance, summary))
    return summary


def _pct(x) -> str:
    return "" if x is None or pd.isna(x) else f"{100 * float(x):.4f}"


def results_markdown(df: pd.DataFrame, provenance: dict, summary: dict) -> str:
    lines = [
        "# Sorting sweep (Orbis Spain)", "",
        ("The planned sweep is complete." if summary["complete"]
         else "The sweep is incomplete; everything below is provisional."), "",
        f"M = {provenance['M']}, seed = {provenance['seed']}, calibrated rho_bar = "
        f"{provenance['rho_bar_calibrated']:.6f}. "
        + ("This is a smoke run; it is not a result. " if provenance.get("smoke") else ""), "",
        "## Sources", "",
        "Every number below comes from `sorting_economies.parquet` (one row per arrangement, "
        "selected by `label`; column headings are exact keys, lambdas in percent) or from "
        "`sorting.yaml:summary`. `sorting.yaml:provenance` records seeds, the frozen calibration "
        "and input hashes; `validation.json` records the baseline gate.", "",
        "## Arrangements", "",
        "| label | kind | corr_av | " + " | ".join(f"lambda_{leg} (%)" for leg in LEGS)
        + " | " + " | ".join(DIAG_COLUMNS) + " |",
        "|---|---|---:|" + "---:|" * (len(LEGS) + len(DIAG_COLUMNS)),
    ]
    order = df.assign(_x=df.corr_av.fillna(np.inf)).sort_values(["_x", "label"])
    for _, row in order.iterrows():
        corr = "" if pd.isna(row.corr_av) else f"{row.corr_av:+.4f}"
        diag = " | ".join("" if pd.isna(row[c]) else f"{row[c]:.4f}" for c in DIAG_COLUMNS)
        lines.append(f"| {row.label} | {row.kind} | {corr} | "
                     + " | ".join(_pct(row[f'lambda_{leg}']) for leg in LEGS) + f" | {diag} |")
    lines += ["", "Copula points re-draw v at a new rho_bar with alpha, tilde_alpha, eps and the "
              "active set as common random numbers, so the v marginal is equal in distribution "
              "only. Permutation points keep both marginals bitwise. Under `hom_sector` the "
              "correlation comes from variation across sectors only; under `hom_cost` alpha is "
              "constant and the correlation is undefined. chi is frozen at the baseline market "
              "value (`chi`) and the anchor (`v_min`, `y_hat`) at the calibration.", ""]

    split = summary.get("sorting_split") or {}
    if split:
        lines += ["## Sorting split", "",
                  "Delta_base = common (Delta_hom) + heterogeneity (Delta_shuf - Delta_hom) + "
                  "sorting (Delta_base - Delta_shuf), per leg; Delta_shuf averages "
                  f"{split['n_shuffle']} shuffle replicates "
                  "(`sorting.yaml:summary.sorting_split.<leg>`).", "",
                  "| leg | Delta_base | common | heterogeneity | sorting | sorting share | "
                  "shuffle replicate range | Delta_hom_cost |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for leg in LEGS:
            e = split[leg]
            share = "" if "share_sorting" not in e else f"{100 * e['share_sorting']:.2f}%"
            cost = "" if "Delta_hom_cost" not in e else f"{e['Delta_hom_cost']:.6f}"
            lines.append(f"| {leg} | {e['Delta_base']:.6f} | {e['common']:.6f} | "
                         f"{e['heterogeneity']:.6f} | {e['sorting']:.6f} | {share} | "
                         f"{e['Delta_shuf_replicate_range']:.2e} | {cost} |")
        lines.append("")

    slopes = summary.get("secant_slopes") or {}
    if slopes:
        lines += ["## Shape of the copula curve", "",
                  "Secant slopes dDelta/dcorr between consecutive copula points, in order of "
                  "increasing correlation (`sorting.yaml:summary.secant_slopes.<leg>`). Segments "
                  "between points are interpolations; finite secants cannot establish a kink.", ""]
        for leg in LEGS:
            lines.append(f"- {NAMES[leg]}: " + ", ".join(f"{s:.4f}" for s in slopes[leg]))
        lines.append("")

    checks = summary.get("copula_vs_permutation") or []
    if checks:
        lines += ["## Copula versus permutation", "",
                  "Gap of each permutation point to the copula curve linearly interpolated at "
                  "its realized correlation, on the Delta scale "
                  "(`sorting.yaml:summary.copula_vs_permutation[i]`). Correlation alone does not "
                  "pin the joint distribution, so a gap is descriptive.", ""]
        for i, c in enumerate(checks):
            if c["inside_copula_range"]:
                gaps = ", ".join(f"{leg} {c[f'gap_Delta_{leg}']:+.5f}" for leg in LEGS)
                lines.append(f"- [{i}] {c['label']} (corr {c['corr_av']:+.4f}): {gaps}")
            else:
                lines.append(f"- [{i}] {c['label']} (corr {c['corr_av']:+.4f}): outside the "
                             "copula range, no interpolated gap")
        lines.append("")

    lines += ["## Checks", "",
              f"All solves converged: {summary['all_converged']} (`summary.all_converged`). "
              f"Maximum |L - 1| {summary['max_labor_deviation']:.3e}; maximum consumption versus "
              f"welfare log gap {summary['max_consumption_welfare_gap']:.3e}; maximum lens identity "
              f"residual {summary['max_identity_residual']:.3e} (`summary.max_labor_deviation`, "
              "`summary.max_consumption_welfare_gap`, `summary.max_identity_residual`).", ""]
    return "\n".join(lines)
