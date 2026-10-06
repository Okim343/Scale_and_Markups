"""
GNR panel construction and estimation-sample trimming, shared by every data source.

Moved out of 01_Compustat/00_Hubmer_RTS/01_code/01_build_gnr_panel.py (stage R1),
logic unchanged. The caller supplies levels already renamed to the canonical names
rev_lvl, cap_lvl, mat_lvl, lab_lvl, wagebill_lvl plus an id column, `year` and
`ind2d`; construct_gnr_vars adds the logs, the share and the consecutive-year lags,
and trim_panel applies the est_gnr.m trimming.

Field names follow the MATLAB code:
  r = log revenue, k = log capital, m = log materials, l = log labor,
  w = log(wage bill / labor) (optional), s = m - r,
  lr, lk, lm, ll, ls, lw = one-period lags within the id (consecutive years only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

def construct_gnr_vars(
    df: pd.DataFrame, logger, build_materials_alt1: bool = True, id_col: str = "gvkey"
) -> pd.DataFrame:
    """Logs, materials share and firm-level one-period lags (consecutive years only)."""
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
    if build_materials_alt1:
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
    df = df.sort_values([id_col, "year"]).reset_index(drop=True)
    grp = df.groupby(id_col, sort=False)
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
    """Estimation-sample restrictions, mirroring est_gnr.m. Reads config["trimming"]
    and config["panel"] (require_wages, min_final_rows)."""
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