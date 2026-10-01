"""
Stage R1 — Build the GNR estimator input panel from curated Compustat.

Reads (config: input.source):
  01_industry_alpha/00_indata/03_Compustat/markup_firm_year.dta

Writes:
  02_intermediary/gnr_panel.parquet          (raw constructed panel with lags)
  02_intermediary/gnr_panel_trimmed.parquet  (final estimation sample)

Chosen v1 mapping (fixed by hubmer_rts_implementation.md):
  id = gvkey, year = fiscal year, ind = ind2d
  r = log(sale_D), k = log(capital_D), m = log(cogs_D), l = log(emp)
  w = log(xlr_D / emp) when available (OPTIONAL in the baseline)
  s = m - r
  lags: firm-level one-period lags within gvkey (consecutive years only)
Robustness columns:
  m_alt1 = log(cogs_D - xlr_D) when positive; s_alt1 = m_alt1 - r

Trimming (mirrors est_gnr.m):
  - exp(s) in share_bounds (default [0.05, 0.95])
  - percentile trim on r, k, l, m, s (default keep >= 1st pctile, <= 100th)
  - industries with fewer than ind_min_obs observations dropped
  - years with fewer than year_min_obs observations dropped
  Industry/year counts are computed on the post-trim sample (divergence
  from MATLAB, which mixed pre/post-trim counts; see README.md).

Run:
  python 01_build_gnr_panel.py [--config config_smoke.yaml]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import (
    PATHS,
    ensure_dir,
    intermediary_dir,
    parse_config_arg,
    resolve_path,
    setup_logger,
)

RAW_OUT = "gnr_panel.parquet"
TRIM_OUT = "gnr_panel_trimmed.parquet"

# Core GNR fields the estimator needs to be non-missing (wages excluded:
# the baseline does not require w/lw for identification).
CORE_FIELDS = ["s", "ls", "r", "lr", "k", "lk", "m", "lm", "l", "ll"]


def safe_log(x: pd.Series, name: str, logger) -> pd.Series:
    """log(x) for x > 0; NaN otherwise. Logs how many values were invalid."""
    pos = x > 0
    n_bad = int((x.notna() & ~pos).sum())
    if n_bad:
        logger.info(f"    {name}: {n_bad:,} non-positive values set to NaN before log")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out[pos] = np.log(x[pos].astype("float64"))
    return out


def build_panel(config: dict, logger) -> pd.DataFrame:
    mp = config["mapping"]
    cfg_panel = config["panel"]
    src = resolve_path(config, "input.source")
    logger.info(f"reading {src.relative_to(PATHS.project_root)}")

    needed = [
        mp["id"], mp["year"], mp["ind"],
        mp["revenue"], mp["capital"], mp["materials"], mp["labor"], mp["wage_bill"],
    ]
    df = pd.read_stata(src, columns=needed)

    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"source file is missing required columns: {missing_cols}")

    df = df.rename(
        columns={
            mp["id"]: "gvkey",
            mp["year"]: "year",
            mp["ind"]: "ind2d",
            mp["revenue"]: "rev_lvl",
            mp["capital"]: "cap_lvl",
            mp["materials"]: "mat_lvl",
            mp["labor"]: "lab_lvl",
            mp["wage_bill"]: "wagebill_lvl",
        }
    )
    logger.info(f"  raw rows: {len(df):,}")
    min_raw = cfg_panel.get("min_raw_rows")
    if min_raw is not None and len(df) < int(min_raw):
        raise ValueError(
            f"raw input has {len(df):,} rows, below panel.min_raw_rows="
            f"{int(min_raw):,}. This usually means GNR is being fed an "
            "aggregation-trimmed markup panel instead of the broad Compustat "
            "firm-year universe."
        )

    # --- basic identifiers ---------------------------------------------------
    df = df.dropna(subset=["year", "ind2d"])
    df["year"] = df["year"].astype(int)
    df["ind2d"] = df["ind2d"].astype(int)
    df["gvkey"] = df["gvkey"].astype(str)
    df = df[df["ind2d"] > 0]

    start_year = int(cfg_panel["start_year"])
    df = df[df["year"] >= start_year]
    logger.info(f"  rows with year >= {start_year}: {len(df):,}")
    min_post = cfg_panel.get("min_post_start_rows")
    if min_post is not None and len(df) < int(min_post):
        raise ValueError(
            f"post-{start_year} input has {len(df):,} rows, below "
            f"panel.min_post_start_rows={int(min_post):,}. Check that the "
            "source file has not already been QJE cost-share trimmed."
        )

    n_dupes = int(df.duplicated(["gvkey", "year"]).sum())
    if n_dupes:
        raise ValueError(
            f"{n_dupes} duplicate (gvkey, year) rows in source — fix upstream "
            "before estimating (lags would be ill-defined)."
        )

    # Optional firm subsampling (smoke tests only).
    frac = cfg_panel.get("sample_firms_frac")
    if frac:
        rng = np.random.default_rng(int(cfg_panel.get("sample_seed", 0)))
        firms = df["gvkey"].unique()
        keep = rng.choice(firms, size=max(1, int(len(firms) * float(frac))), replace=False)
        df = df[df["gvkey"].isin(set(keep))]
        logger.warning(
            f"  SUBSAMPLED to {float(frac):.0%} of firms "
            f"({len(keep):,} firms, {len(df):,} rows) — smoke-test mode"
        )

    # --- logs ------------------------------------------------------------------
    logger.info("  constructing logs (non-positive values -> NaN):")
    df["r"] = safe_log(df["rev_lvl"], "r = log(sale_D)", logger)
    df["k"] = safe_log(df["cap_lvl"], "k = log(capital_D)", logger)
    df["m"] = safe_log(df["mat_lvl"], "m = log(cogs_D)", logger)
    df["l"] = safe_log(df["lab_lvl"], "l = log(emp)", logger)

    # w = log(xlr_D / emp), optional auxiliary field
    wage_per_emp = df["wagebill_lvl"] / df["lab_lvl"]
    df["w"] = safe_log(wage_per_emp, "w = log(xlr_D / emp)", logger)
    df["has_w"] = df["w"].notna()
    logger.info(f"    wage coverage: {df['has_w'].mean():.1%} of rows have w")

    df["s"] = df["m"] - df["r"]

    # Robustness: m_alt1 = log(cogs_D - xlr_D) when positive
    if cfg_panel.get("build_materials_alt1", True):
        diff = df["mat_lvl"] - df["wagebill_lvl"]
        n_nonpos = int((diff.notna() & (diff <= 0)).sum())
        logger.info(
            f"  m_alt1 robustness: cogs_D - xlr_D non-positive for "
            f"{n_nonpos:,} rows (set to NaN); defined for "
            f"{int((diff > 0).sum()):,} rows"
        )
        df["m_alt1"] = safe_log(diff, "m_alt1 = log(cogs_D - xlr_D)", logger)
        df["s_alt1"] = df["m_alt1"] - df["r"]

    # --- firm-level one-period lags (consecutive years only) -------------------
    df = df.sort_values(["gvkey", "year"]).reset_index(drop=True)
    grp = df.groupby("gvkey", sort=False)
    lag_year = grp["year"].shift(1)
    consecutive = lag_year == df["year"] - 1
    lag_map = {"s": "ls", "r": "lr", "k": "lk", "m": "lm", "l": "ll", "w": "lw"}
    if "m_alt1" in df.columns:
        lag_map.update({"m_alt1": "lm_alt1", "s_alt1": "ls_alt1"})
    for src_col, lag_col in lag_map.items():
        df[lag_col] = grp[src_col].shift(1).where(consecutive)
    df["lag_is_consecutive"] = consecutive.fillna(False)
    logger.info(
        f"  rows with a consecutive prior firm-year: "
        f"{int(df['lag_is_consecutive'].sum()):,} / {len(df):,}"
    )

    return df


def trim_panel(df: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """Estimation-sample restrictions, mirroring est_gnr.m."""
    cfg_trim = config["trimming"]
    require_wages = bool(config["panel"].get("require_wages", False))

    fields = list(CORE_FIELDS)
    if require_wages:
        fields += ["w", "lw"]
        logger.warning(
            "  require_wages=true: imposing non-missing w and lw "
            "(this cuts the sample sharply; baseline default is false)"
        )
    est = df.dropna(subset=fields).copy()
    logger.info(f"  complete-case estimation sample (core fields): {len(est):,}")
    if len(est) == 0:
        raise ValueError("estimation sample is empty after complete-case filter")

    # 1) materials-share band: exp(s) in [lo, hi]
    lo, hi = map(float, cfg_trim["share_bounds"])
    share = np.exp(est["s"])
    sel = (share >= lo) & (share <= hi)
    logger.info(
        f"  share band exp(s) in [{lo}, {hi}]: drop {int((~sel).sum()):,}, "
        f"keep {int(sel.sum()):,}"
    )
    est = est[sel]

    # 2) percentile trim on r, k, l, m, s (MATLAB mincut/maxcut, default 1/100)
    mincut = float(cfg_trim["pctile_mincut"])
    maxcut = float(cfg_trim["pctile_maxcut"])
    if mincut > 0 or maxcut < 100:
        sel = pd.Series(True, index=est.index)
        for v in ["r", "k", "l", "m", "s"]:
            lo_v, hi_v = np.percentile(est[v], [mincut, maxcut])
            sel &= (est[v] >= lo_v) & (est[v] <= hi_v)
        logger.info(
            f"  percentile trim [{mincut}, {maxcut}] on r,k,l,m,s: "
            f"drop {int((~sel).sum()):,}, keep {int(sel.sum()):,}"
        )
        est = est[sel]

    # 3) minimum industry size, then minimum year size (post-trim counts)
    ind_min = int(cfg_trim["ind_min_obs"])
    counts = est["ind2d"].value_counts()
    small_inds = sorted(counts[counts < ind_min].index.tolist())
    if small_inds:
        n_drop = int(est["ind2d"].isin(small_inds).sum())
        logger.info(
            f"  industries below ind_min_obs={ind_min}: {small_inds} "
            f"(drop {n_drop:,} rows)"
        )
        est = est[~est["ind2d"].isin(small_inds)]

    year_min = int(cfg_trim["year_min_obs"])
    ycounts = est["year"].value_counts()
    small_years = sorted(ycounts[ycounts < year_min].index.tolist())
    if small_years:
        n_drop = int(est["year"].isin(small_years).sum())
        logger.info(
            f"  years below year_min_obs={year_min}: {small_years} "
            f"(drop {n_drop:,} rows)"
        )
        est = est[~est["year"].isin(small_years)]

    logger.info(f"  final estimation sample: {len(est):,} rows")
    min_final = config["panel"].get("min_final_rows")
    if min_final is not None and len(est) < int(min_final):
        raise ValueError(
            f"final estimation sample has {len(est):,} rows, below "
            f"panel.min_final_rows={int(min_final):,}. This guardrail is meant "
            "to catch accidental row-universe changes before GNR estimation."
        )
    if len(est) == 0:
        raise ValueError("estimation sample is empty after trimming")

    # Summary diagnostics
    logger.info(
        f"  industries retained ({est['ind2d'].nunique()}): "
        f"{sorted(est['ind2d'].unique().tolist())}"
    )
    logger.info(
        f"  years: {int(est['year'].min())}-{int(est['year'].max())}; "
        f"firms: {est['gvkey'].nunique():,}; "
        f"wage coverage in sample: {est['has_w'].mean():.1%}"
    )
    return est.reset_index(drop=True)


def main() -> None:
    config = parse_config_arg("Stage R1 — build the GNR estimator input panel")
    logger = setup_logger("R1", config.get("log_level", "INFO"))

    df = build_panel(config, logger)
    out_dir = ensure_dir(intermediary_dir(config))

    raw_path = out_dir / RAW_OUT
    df.to_parquet(raw_path, index=False, compression="snappy")
    logger.info(
        f"wrote {raw_path.relative_to(PATHS.project_root)} "
        f"({raw_path.stat().st_size / 1e6:.1f} MB; {len(df):,} rows)"
    )

    est = trim_panel(df, config, logger)
    trim_path = out_dir / TRIM_OUT
    est.to_parquet(trim_path, index=False, compression="snappy")
    logger.info(
        f"wrote {trim_path.relative_to(PATHS.project_root)} "
        f"({trim_path.stat().st_size / 1e6:.1f} MB; {len(est):,} rows)"
    )


if __name__ == "__main__":
    sys.exit(main())
