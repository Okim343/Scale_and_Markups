"""
Stage S7 — Aggregate macro moments, markup-share slope, and alpha-rank diagnostics.

Reads:
  00_indata/03_Compustat/markup_cost_weighted.dta   (aggregate cost-weighted)
  01_intermediary/s2_firm_year_shares.parquet       (firm-year markups + s_ji)

Writes:
  01_intermediary/s7_year_aggregates.parquet
      Year-level aggregate cost-weighted markups for sensitivity.

  01_intermediary/s7_slope_regression.json
      Global calibration moments:
        - aggregate_markup_cost_weighted   (target on γ)
        - aggregate_markup_sensitivity     (active + robustness windows)
        - markup_share_slope_loglog        (target on η given γ)
        - markup_share_slope_by_sector     (per-sector diagnostic)
        - markup_share_decile_curve        (nonparametric EMX-style curve)

Identification map (per pipeline_plan.md §5):
  γ ← aggregate_markup_cost_weighted via μ_min = γ/(γ-1)
  η ← markup_share_slope = -(1/η - 1/γ)  given γ

Run:
  python 07_aggregate_moments.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import PATHS, ensure_dir, load_config, setup_logger


def _to_python(obj):
    """Convert numpy scalars / arrays to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def load_year_aggregates(logger, year_floor: int) -> pd.DataFrame:
    """Load the pre-aggregated cost-weighted aggregate markup series."""
    cw = pd.read_stata(PATHS.compustat_markup_cw)
    cw["year"] = cw["year"].astype(int)
    merged = cw[cw["year"] >= year_floor].copy()
    merged = merged.rename(
        columns={
            "markup": "agg_mu_cw",
            "markup_xsga": "agg_mu_xsga_cw",
        }
    )
    merged["year"] = merged["year"].astype("int16")
    for c in ["agg_mu_cw", "agg_mu_xsga_cw"]:
        merged[c] = merged[c].astype("float32")
    logger.info(
        f"  aggregate markup series: {merged['year'].min()}–{merged['year'].max()}, "
        f"{len(merged)} years"
    )
    return merged


def time_average(year_agg: pd.DataFrame, start: int, end: int) -> dict:
    win = year_agg[(year_agg["year"] >= start) & (year_agg["year"] <= end)]
    return {
        "start": start,
        "end": end,
        "n_years": int(len(win)),
        "agg_mu_cw": float(win["agg_mu_cw"].mean()),
        "agg_mu_xsga_cw": float(win["agg_mu_xsga_cw"].mean()),
    }


def slope_regression(sub: pd.DataFrame, logger) -> dict:
    """Pooled markup-share slope with sector_id + year FE, HC1 SE."""
    model = smf.ols(
        "log_markup ~ log_s + C(sector_id) + C(year)",
        data=sub,
    ).fit(cov_type="HC1")

    out = {
        "spec": "log(μ_jt) ~ log(s_jt) + C(sector_id) + C(year); HC1 SE",
        "coef": float(model.params["log_s"]),
        "se_hc1": float(model.bse["log_s"]),
        "t_stat": float(model.tvalues["log_s"]),
        "p_value": float(model.pvalues["log_s"]),
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
    }
    logger.info(
        f"  pooled slope log(μ) on log(s) = {out['coef']:.4f} "
        f"(HC1 SE {out['se_hc1']:.4f}, t={out['t_stat']:.2f}); "
        f"n={out['n_obs']:,d}, R²={out['r_squared']:.3f}"
    )
    return out


def slope_by_sector(sub: pd.DataFrame, logger, min_n: int = 50) -> dict:
    """Per-sector slope, dropping sectors with too few obs."""
    per_sector = {}
    for sid, gdf in sub.groupby("sector_id", observed=True):
        if len(gdf) < min_n:
            continue
        try:
            m = smf.ols(
                "log_markup ~ log_s + C(year)", data=gdf
            ).fit(cov_type="HC1")
            per_sector[str(sid)] = {
                "coef": float(m.params["log_s"]),
                "se_hc1": float(m.bse["log_s"]),
                "n_obs": int(m.nobs),
                "r_squared": float(m.rsquared),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  per-sector regression failed for {sid}: {exc}")
    if per_sector:
        ps_df = (
            pd.DataFrame(per_sector)
            .T.sort_values("coef")
            .round(4)
            [["coef", "se_hc1", "n_obs"]]
        )
        logger.info(f"  per-sector slopes (active window):\n{ps_df.to_string()}")
    return per_sector


def decile_curve(sub: pd.DataFrame, logger) -> list[dict]:
    """Nonparametric markup-share curve by equal-mass within-sector share deciles.

    Deciles are formed on the pooled active-window distribution of s_ji.
    Mean and median markup are reported per decile.
    """
    sub = sub.copy()
    sub["share_decile"] = pd.qcut(
        sub["s_ji"], q=10, labels=False, duplicates="drop"
    )
    curve = (
        sub.groupby("share_decile", observed=True)
        .agg(
            n=("markup", "size"),
            mean_s=("s_ji", "mean"),
            p50_s=("s_ji", "median"),
            mean_markup=("markup", "mean"),
            p50_markup=("markup", "median"),
        )
        .reset_index()
        .sort_values("share_decile")
    )
    logger.info(f"  markup-share decile curve:\n{curve.round(5).to_string(index=False)}")
    return [
        {k: _to_python(v) for k, v in row.items()}
        for row in curve.to_dict(orient="records")
    ]


def main() -> None:
    config = load_config()
    logger = setup_logger("S7", config.get("log_level", "INFO"))
    aw_key = config["active_window"]
    aw_start = int(config["windows"][aw_key]["start"])
    aw_end = int(config["windows"][aw_key]["end"])
    year_floor = int(config["s1_keep_from_year"])
    logger.info(
        f"active window {aw_key} = [{aw_start}, {aw_end}]; year floor {year_floor}"
    )

    # --- 1. Aggregate markup time series + sensitivity ---------------------
    logger.info("reading aggregate markup time series")
    year_agg = load_year_aggregates(logger, year_floor)

    out_year = ensure_dir(PATHS.intermediary) / "s7_year_aggregates.parquet"
    year_agg.to_parquet(out_year, index=False, compression="snappy")
    logger.info(
        f"wrote {out_year.relative_to(PATHS.project_root)} "
        f"({out_year.stat().st_size / 1e3:.2f} KB)"
    )

    sensitivity = {
        label: time_average(year_agg, int(w["start"]), int(w["end"]))
        for label, w in config["windows"].items()
    }
    aw = sensitivity[aw_key]
    logger.info(
        f"  active-window cost-weighted aggregate markup: {aw['agg_mu_cw']:.4f} "
        f"(COGS) / {aw['agg_mu_xsga_cw']:.4f} (SG&A-inclusive)"
    )

    # --- 2. Markup-share slope regression ---------------------------------
    logger.info("loading firm-year shares from S2 for slope regression")
    firm = pd.read_parquet(PATHS.intermediary / "s2_firm_year_shares.parquet")
    sub = firm[(firm["year"] >= aw_start) & (firm["year"] <= aw_end)].copy()
    sub = sub[(sub["markup"] > 0) & (sub["s_ji"] > 0)].copy()
    sub["log_markup"] = np.log(sub["markup"].astype("float64"))
    sub["log_s"] = np.log(sub["s_ji"].astype("float64"))
    logger.info(
        f"  regression sample: {len(sub):,d} firm-years across "
        f"{sub['sector_id'].nunique()} sectors, "
        f"{sub['year'].nunique()} years"
    )

    pooled = slope_regression(sub, logger)
    by_sector = slope_by_sector(sub, logger)

    # --- 3. Nonparametric decile curve ------------------------------------
    deciles = decile_curve(sub, logger)

    # --- Assemble and write JSON ------------------------------------------
    out = {
        "active_window": {"key": aw_key, "start": aw_start, "end": aw_end},
        "aggregate_markup_cost_weighted": aw["agg_mu_cw"],
        "aggregate_markup_xsga_cost_weighted": aw["agg_mu_xsga_cw"],
        "aggregate_markup_sensitivity": sensitivity,
        "markup_share_slope_loglog": pooled,
        "markup_share_slope_by_sector": by_sector,
        "markup_share_decile_curve": deciles,
    }
    out = _to_python(out)

    out_json = ensure_dir(PATHS.intermediary) / "s7_slope_regression.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        f"wrote {out_json.relative_to(PATHS.project_root)} "
        f"({out_json.stat().st_size / 1e3:.2f} KB)"
    )


if __name__ == "__main__":
    sys.exit(main())
