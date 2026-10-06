"""
Stage S1 — Compustat firm-year cleaning.

Reads:
  00_indata/03_Compustat/markup_firm_year.dta

Writes:
  01_intermediary/s1_compustat_firmyear.parquet

Logic (per pipeline_plan.md §4 Stage S1):
  1. Keep canonical columns.
  2. Drop rows with missing markup, missing ind2d, or non-positive sale/cogs.
  3. Drop NAICS2 ∈ excluded set from config (52, 53, 62, 81, 92, 99).
  4. Drop years < s1_keep_from_year (default 1997); downstream stages further
     filter to the active window. This keeps S1 invariant under window changes.
  5. Apply markup trim: hard bounds + within-year winsorization.
  6. Cast types, write parquet.

Run:
  python 01_load_compustat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    PATHS,
    active_window,
    ensure_dir,
    load_config,
    map_to_sector_id,
    setup_logger,
)


# Columns retained from the raw Compustat-derived markup panel.
# Output elasticities (theta_*) and deflated quantities are kept because
# Stage S2 uses real_sale / real_cogs for time-averaging and Stage S7
# uses markups for the slope regression.
KEEP_COLS = [
    "gvkey",
    "year",
    "ind2d",
    "ind3d",
    "ind4d",
    "sale",
    "cogs",
    "xsga",
    "ppegt",
    "emp",
    "markup",
    "markup_xsga",
    "theta_WI1_ct",
    "theta_WI2_ct",
    "weight_s",
]

# Deflated counterparts (from the upstream DLE pipeline); kept if present.
DEFLATED_COLS = [
    "sale_D",
    "cogs_D",
    "xsga_D",
    "mkvalt_D",
    "dividend_D",
    "capital_D",
    "xlr_D",
]

OUT_NAME = "s1_compustat_firmyear.parquet"


def _log_drop(logger, df_before: pd.DataFrame, df_after: pd.DataFrame, reason: str) -> None:
    """Helper to log how many rows were removed by a filtering step."""
    n_before = len(df_before)
    n_after = len(df_after)
    n_drop = n_before - n_after
    pct = 100.0 * n_drop / n_before if n_before else 0.0
    logger.info(f"  drop {reason:<50s} {n_drop:>9,d} rows  ({pct:5.2f}%) -> {n_after:>9,d} remain")


def load_raw(logger) -> pd.DataFrame:
    """Read the .dta file and keep only the canonical column set."""
    path = PATHS.compustat_markup_firmyear
    logger.info(f"reading {path.relative_to(PATHS.project_root)}")
    df = pd.read_stata(path)
    logger.info(f"  raw shape: {df.shape}")

    # Some deflated columns may not be present in every vintage of the panel.
    optional_present = [c for c in DEFLATED_COLS if c in df.columns]
    keep = KEEP_COLS + optional_present
    missing_required = [c for c in KEEP_COLS if c not in df.columns]
    if missing_required:
        raise KeyError(
            f"required columns missing from {path.name}: {missing_required}"
        )
    df = df[keep].copy()
    logger.info(f"  kept {len(keep)} cols (incl. {len(optional_present)} deflated)")
    return df


def drop_invalid(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Drop rows that are unusable for downstream stages."""
    n0 = len(df)
    logger.info(f"row filtering (start: {n0:,d})")

    before = df
    df = df.dropna(subset=["markup"])
    _log_drop(logger, before, df, "missing markup")

    before = df
    df = df.dropna(subset=["ind2d"])
    _log_drop(logger, before, df, "missing ind2d")

    before = df
    df = df[df["sale"] > 0]
    _log_drop(logger, before, df, "non-positive sale")

    before = df
    df = df[df["cogs"] > 0]
    _log_drop(logger, before, df, "non-positive cogs")

    # Year must be present and finite for window filtering to work.
    before = df
    df = df.dropna(subset=["year"])
    _log_drop(logger, before, df, "missing year")

    return df


def filter_sectors_and_years(
    df: pd.DataFrame, config: dict, logger
) -> pd.DataFrame:
    """Apply NAICS2 exclusion and the S1-level year floor."""
    naics_exclude = set(int(x) for x in config["naics_exclude"])
    year_floor = int(config["s1_keep_from_year"])

    before = df
    df = df[~df["ind2d"].astype(int).isin(naics_exclude)]
    _log_drop(
        logger, before, df, f"NAICS2 in {sorted(naics_exclude)}"
    )

    before = df
    df = df[df["year"] >= year_floor]
    _log_drop(logger, before, df, f"year < {year_floor}")

    # Diagnostic: how many sectors remain, and how many year-sector cells.
    n_sectors = df["ind2d"].astype(int).nunique()
    n_year_sector = df.groupby(["year", "ind2d"]).ngroups
    logger.info(
        f"  retained {n_sectors} NAICS2 sectors across "
        f"{int(df['year'].min())}–{int(df['year'].max())} "
        f"({n_year_sector:,d} year-sector cells)"
    )
    return df


def trim_markups(df: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """Hard bounds first, then within-year winsorization on the survivors."""
    tcfg = config["markup_trim"]
    lo, hi = float(tcfg["hard_lower"]), float(tcfg["hard_upper"])

    before = df
    df = df[(df["markup"] >= lo) & (df["markup"] <= hi)]
    _log_drop(logger, before, df, f"markup outside [{lo}, {hi}]")

    if tcfg.get("apply_winsor", True):
        pct_lo, pct_hi = tcfg["winsor_pct"]
        # Year-level percentiles, broadcast back to rows.
        bounds = (
            df.groupby("year")["markup"]
            .quantile([pct_lo, pct_hi])
            .unstack()
            .rename(columns={pct_lo: "lo", pct_hi: "hi"})
        )
        df = df.merge(bounds, left_on="year", right_index=True, how="left")
        n_clipped = int(((df["markup"] < df["lo"]) | (df["markup"] > df["hi"])).sum())
        df["markup"] = df["markup"].clip(lower=df["lo"], upper=df["hi"])
        df = df.drop(columns=["lo", "hi"])
        logger.info(
            f"  winsorized {n_clipped:,d} markup obs to within-year "
            f"[{pct_lo:.2f}, {pct_hi:.2f}] quantiles"
        )

    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast NAICS and year to compact integer types; gvkey stays as str.

    Also adds the canonical `sector_id` column (KLEMS/BDS native granularity).
    Downstream stages key on `sector_id`, not `ind2d`.
    """
    df = df.copy()
    df["year"] = df["year"].astype("int16")
    df["ind2d"] = df["ind2d"].astype("int8")
    if "ind3d" in df.columns:
        df["ind3d"] = df["ind3d"].astype("Int16")
    if "ind4d" in df.columns:
        df["ind4d"] = df["ind4d"].astype("Int32")
    df["gvkey"] = df["gvkey"].astype(str)
    df["sector_id"] = map_to_sector_id(df["ind2d"]).astype("string")
    if df["sector_id"].isna().any():
        bad = sorted(df.loc[df["sector_id"].isna(), "ind2d"].unique().tolist())
        raise ValueError(
            f"ind2d values with no sector_id mapping in COMPUSTAT_TO_SECTOR_ID: {bad}"
        )
    return df


def summarize(df: pd.DataFrame, logger) -> None:
    """Diagnostic summary of the final panel."""
    logger.info("final panel summary")
    logger.info(f"  rows                : {len(df):,d}")
    logger.info(f"  unique gvkeys       : {df['gvkey'].nunique():,d}")
    logger.info(f"  years               : {df['year'].min()}–{df['year'].max()}")
    logger.info(f"  NAICS2 sectors      : {sorted(df['ind2d'].unique().tolist())}")
    logger.info(
        f"  sector_id (collapsed): "
        f"{sorted(df['sector_id'].dropna().unique().tolist())}"
    )
    logger.info(f"  markup p1/p50/p99   : "
                f"{df['markup'].quantile(0.01):.3f} / "
                f"{df['markup'].quantile(0.50):.3f} / "
                f"{df['markup'].quantile(0.99):.3f}")
    # Show how 2010–2019 (the active baseline window) looks against the full panel.
    sub = df[(df["year"] >= 2010) & (df["year"] <= 2019)]
    logger.info(
        f"  2010–2019 subset    : {len(sub):,d} rows, "
        f"{sub['gvkey'].nunique():,d} firms"
    )


def main() -> None:
    config = load_config()
    logger = setup_logger("S1", config.get("log_level", "INFO"))

    aw_start, aw_end = active_window(config)
    logger.info(
        f"pipeline version {config.get('pipeline_version')}; "
        f"active window {aw_start}-{aw_end} "
        f"(S1 keeps from {config['s1_keep_from_year']} onward)"
    )

    df = load_raw(logger)
    df = drop_invalid(df, logger)
    df = filter_sectors_and_years(df, config, logger)
    df = trim_markups(df, config, logger)
    df = cast_types(df)
    summarize(df, logger)

    out_dir = ensure_dir(PATHS.intermediary)
    out_path = out_dir / OUT_NAME
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"wrote {out_path.relative_to(PATHS.project_root)} "
                f"({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
