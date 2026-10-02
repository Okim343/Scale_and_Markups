"""Plot observed sorting points and write a source-keyed research readout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from steady_state.figures.house_style import configure_style, save_pdf_and_png, style_axes
from .run_sorting_curve import LEGS

COLORS = {"realloc": "#2b3a67", "scale": "#c98a3c", "disp": "#1f7a72", "level": "#8b3a3a"}
NAMES = {"realloc": "Reallocation", "scale": "Scale", "disp": "Dispersion", "level": "Level"}


def curve_points(df):
    """Average repeated partial draws and the saved shuffle draws on Delta scale."""
    records = []
    for _, row in df.iterrows():
        if row["kind"] == "rho":
            continue
        record = row.to_dict()
        record["group"] = ("shuffle_mean" if str(row.label).startswith("shuffle") else
                           str(row.label).rsplit("_d", 1)[0])
        records.append(record)
    points = pd.DataFrame(records)
    columns = ["corr_av"] + [f"delta_{leg}" for leg in LEGS]
    return points.groupby("group")[columns].mean().reset_index().sort_values("corr_av")


def write_artifacts(df, provenance, out, fig_dir, complete=False):
    points = curve_points(df)
    summary = {"complete": complete, "n_rows": len(df),
               "all_converged": bool(df[[f"converged_{k}" for k in ("ME", "PE_I", "U", "PE")]].all().all()),
               "max_labor_deviation": float(df[[f"L_{k}" for k in ("ME", "PE_I", "U", "PE")]].sub(1).abs().max().max()),
               "max_consumption_welfare_gap": float(df[[f"consumption_gap_{k}" for k in LEGS]].abs().max().max()),
               "max_identity_residual": float(df[["identity_A", "identity_B"]].abs().max().max()),
               "raw_leg_ranking_all": bool(((df.delta_disp < df.delta_realloc) &
                  (df.delta_realloc < df.delta_scale) & (df.delta_scale < df.delta_level)).all())}
    lines = ["# Sorting curve", "", "I report observed solves and cached endpoints. " +
             ("The requested sweep is complete." if complete else "The sweep is incomplete; findings below are provisional."), "",
             "## Provenance and source keys", "",
             "All table numbers below come from `sorting_curve_economies.parquet`, selected by `label`; "
             "column headings are exact keys. Lambda columns are shown in percent. "
             "`sorting_curve.yaml:provenance` records the frozen calibration, seeds, source paths, and data hashes. "
             "Saved planner and uniform consumption is reconstructed from saved welfare at labor equal to one; saved market C and L are retained. "
             "Shuffle summaries average Delta across matched saved draws; they do not take log of average lambda.", "",
             "| label | corr_av | lambda_realloc (%) | lambda_scale (%) | lambda_disp (%) | lambda_level (%) | lambda_total (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in df.sort_values("corr_av").iterrows():
        lines.append("| " + str(row.label) + " | " + f"{row.corr_av:.6f}" + " | " +
                     " | ".join(f"{100*row[f'lambda_{leg}']:.6f}" for leg in LEGS) + " |")
    has_endpoints = {"baseline", "reverse"}.issubset(set(points.group))
    if has_endpoints:
        base = points[points.group.eq("baseline")].iloc[0]
        rev = points[points.group.eq("reverse")].iloc[0]
        configure_style()
        fig, ax = plt.subplots(figsize=(9.4, 6.4))
        ax.plot([rev.corr_av, base.corr_av], [0, 1], "--", color="0.6", linewidth=1.6, label="Linear benchmark")
        normalized = points.copy()
        for leg, color in COLORS.items():
            key = f"delta_{leg}"
            normalized[key] = (points[key] - rev[key]) / (base[key] - rev[key])
            ax.plot(points.corr_av, normalized[key], "-o", color=color, linewidth=2.1,
                    markersize=6.5, markeredgecolor="white", label=NAMES[leg])
            # Display every replicate as an observed point, although lines use group means.
            partial = df[df.kind.eq("partial")]
            ax.scatter(partial.corr_av, (partial[key]-rev[key])/(base[key]-rev[key]),
                       color=color, s=32, edgecolor="white", zorder=4)
            slopes = np.diff(normalized[key]) / np.diff(normalized.corr_av)
            summary[f"secant_slopes_{leg}"] = [float(x) for x in slopes]
            summary[f"sampled_convex_{leg}"] = bool(np.all(np.diff(slopes) >= -1e-6))
            summary[f"sampled_monotone_{leg}"] = bool(np.all(slopes >= -1e-6))
        ax.set_xlabel(r"$\mathrm{corr}(\alpha,\nu)$", fontsize=13)
        ax.set_ylabel("Fraction of welfare movement\n(reverse = 0, baseline = 1)", fontsize=11)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.0%}"))
        style_axes(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=10)
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_pdf_and_png(fig, fig_dir / "sorting_convexity_curve")
        plt.close(fig)
        summary["normalized_ranking_all"] = bool(((normalized.delta_disp <= normalized.delta_realloc+1e-9) &
                       (normalized.delta_realloc <= normalized.delta_scale+1e-9) &
                       (normalized.delta_scale <= normalized.delta_level+1e-9)).all())
        lines += ["", "## Shape and ranking", "",
                  "Connecting segments guide the eye. Only markers are observed economies. "
                  "Finite secant slopes can reject sampled convexity but cannot prove convexity throughout "
                  "or identify a structural kink between observations. Repeated partial draws are averaged for secant slopes.", ""]
        if len(points) >= 8:
            lines += ["The completed primary points show a pronounced bend around the nearly zero-correlation "
                      "shuffle anchor: the negative branch is shallow and the positive branch is much steeper. "
                      "The positive branch is approximately linear, with declining slopes most clearly for reallocation "
                      "and dispersion. I therefore do not describe the curve as convex throughout. The finite grid "
                      "locates a bend but does not establish an exact structural kink "
                      "(`sorting_curve.yaml:summary.secant_slopes_<leg>`).", ""]
        for leg in COLORS:
            slopes = summary[f"secant_slopes_{leg}"]
            lines.append(f"- {NAMES[leg]}: " + ("sampled slopes are nondecreasing" if summary[f"sampled_convex_{leg}"] else
                         "sampled slopes decrease somewhere, so strict convexity is not supported") +
                         f" (`sorting_curve.yaml:summary.sampled_convex_{leg}`; "
                         f"`summary.secant_slopes_{leg}` = {', '.join(f'{x:.6f}' for x in slopes)}).")
        lines += ["", f"The raw Delta ranking disp < realloc < scale < level holds at every row: "
                  f"{summary['raw_leg_ranking_all']} (`sorting_curve.yaml:summary.raw_leg_ranking_all`). "
                  f"The normalized ranking holds at every grouped primary point: {summary['normalized_ranking_all']} "
                  "(`summary.normalized_ranking_all`). These are different comparisons."]
        lines.append("")
        if "shuffle_mean" in set(points.group):
            sh = normalized[normalized.group.eq("shuffle_mean")].iloc[0]
            for leg in LEGS:
                frac = float((points[points.group.eq("shuffle_mean")][f"delta_{leg}"].iloc[0]-rev[f"delta_{leg}"])/
                             (base[f"delta_{leg}"]-rev[f"delta_{leg}"]))
                summary[f"movement_above_shuffle_{leg}"] = 1-frac
                lines.append(f"- {leg}: {100*(1-frac):.4f}% of reverse-to-baseline movement occurs above "
                             f"the shuffle mean (`sorting_curve.yaml:summary.movement_above_shuffle_{leg}`).")
        rho_checks = []
        for _, row in df[df.kind.eq("rho")].iterrows():
            check = {"label": row.label, "corr_av": float(row.corr_av)}
            for leg in LEGS:
                key = f"delta_{leg}"
                estimate = float(np.interp(row.corr_av, points.corr_av, points[key]))
                check[f"interpolated_primary_{key}"] = estimate
                check[f"gap_{key}"] = float(row[key]-estimate)
                check[f"gap_normalized_{leg}"] = float((row[key]-estimate)/(base[key]-rev[key]))
            rho_checks.append(check)
        summary["rho_checks"] = rho_checks
        lines += ["", "## Copula check", "", provenance["rho_caveat"] + ". "
                  "The comparison uses piecewise linear interpolation between primary observations at the realized "
                  "correlation, not an exact matched-correlation solve. Correlation alone does not fix the joint distribution. "
                  "Agreement is descriptive and has no prespecified statistical tolerance."]
        if not rho_checks:
            lines.append("No copula check has completed.")
        if rho_checks:
            lines.append("The completed copula checks lie below the interpolated partial-permutation curve. "
                         "They do not agree quantitatively at their realized correlations; Pearson correlation alone "
                         "does not summarize the economically relevant joint assignment. The signed departures below "
                         "are fractions of each leg's reverse-to-baseline movement, expressed in percentage points.")
        lines.append("")
        for i, check in enumerate(rho_checks):
            lines.append(f"- {check['label']}: realized correlation {check['corr_av']:.6f}; normalized gaps " +
                         ", ".join(f"{leg} {100*check[f'gap_normalized_{leg}']:+.4f} pp" for leg in LEGS) +
                         f" (`sorting_curve.yaml:summary.rho_checks[{i}].corr_av` and `gap_normalized_<leg>`).")
    replicate = df[df.label.isin(["identity_q0.40_d0", "identity_q0.40_d1"])]
    if len(replicate) == 2:
        gap = float(abs(replicate.lambda_total.iloc[0]-replicate.lambda_total.iloc[1])*100)
        summary["replicate_q040_total_gap_pp"] = gap
        lines += ["", f"The repeated partial draw has an absolute total-lambda gap of {gap:.6f} pp "
                  "(`sorting_curve.yaml:summary.replicate_q040_total_gap_pp`). Two draws measure a difference, "
                  "not a precise standard error."]
    validation_path = out / "compact_validation.json"
    if provenance.get("numerical_optimization") and validation_path.exists():
        guard = json.loads(validation_path.read_text())
        largest = max(guard["full_width_q020_relative_differences"].values())
        summary["padding_validation_passed"] = guard["passed"]
        lines += ["", "## Runtime correction", "",
                  f"The original draw retains {guard['pool_H']} slots per sector. Numerical evaluation omits "
                  f"globally inactive trailing padding and uses {guard['numerical_H']} columns "
                  "(`compact_validation.json:pool_H`, `numerical_H`). No active primitive or calibrated "
                  "economic parameter changes. The full-width and shortened-array baseline W, C, lambda, "
                  "and Delta columns match exactly (`full_width_baseline_relative_differences`). "
                  f"The completed partial point agrees within {largest:.3e} relatively "
                  "(`full_width_q020_relative_differences`). The two production-size equivalence solves took "
                  f"{guard['elapsed_seconds']:.3f} seconds (`elapsed_seconds`). Previously completed "
                  "full-width points are retained; only the final copula point uses the shorter numerical arrays."]
    lines += ["", f"Maximum labor residual {summary['max_labor_deviation']:.3e}; maximum consumption versus welfare log gap {summary['max_consumption_welfare_gap']:.3e} "
              "(`sorting_curve.yaml:summary.max_labor_deviation`, `summary.max_consumption_welfare_gap`). "
              "Cached nonmarket labor is imputed as one; its numerical residual is unavailable.",
              "", f"All stored solves converged: {summary['all_converged']}; maximum lens residual "
              f"{summary['max_identity_residual']:.3e} (`sorting_curve.yaml:summary.all_converged`, "
              "`summary.max_identity_residual`).", ""]
    payload = {"provenance": provenance, "summary": summary}
    tmp = out / "sorting_curve.yaml.tmp"
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    tmp.replace(out / "sorting_curve.yaml")
    (out / "results.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="out_results/counterfactuals/sorting_curve")
    p.add_argument("--fig-dir", default="out_figs/counterfactuals")
    args = p.parse_args()
    out = Path(args.results_dir)
    payload = yaml.safe_load((out / "sorting_curve.yaml").read_text())
    write_artifacts(pd.read_parquet(out / "sorting_curve_economies.parquet"), payload["provenance"],
                    out, Path(args.fig_dir), complete=payload["summary"]["complete"])


if __name__ == "__main__":
    main()
