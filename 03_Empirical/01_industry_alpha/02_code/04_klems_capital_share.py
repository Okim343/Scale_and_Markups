"""
Stage S4 — KLEMS capital share and cross-sector expenditure weights.

Reads:
  00_indata/klems_labor_share_naics2_with_VA_1997_2023.csv

Writes:
  01_intermediary/s4_klems_sector_year.parquet

Logic (per pipeline_plan.md §4 Stage S4):
  1. Read KLEMS NAICS2 panel (1997-2023).
  2. KLEMS uses combined codes for some sectors: '31-33' (Manufacturing),
     '48-49' (Transportation). We keep them at native granularity here and
     defer the Compustat-to-KLEMS sector-mapping decision to Stage S8.
  3. Drop sectors fully contained in `naics_exclude`. Keep combined groups
     with `partial_overlap_flag = True` when some constituent NAICS2 is
     excluded (currently affects '48-49' since NAICS-49 was added to
     the exclude list in Stage S2).
  4. Compute a_iy = 1 - LS_VA, clip to config `a_i_clip` (default [0.05, 0.7]).
  5. Compute va_share_iy, go_share_iy as within-year shares across the
     retained KLEMS group set.

Coverage warnings to log:
  - KLEMS in this file is MISSING retail entirely (no '44', '45', or '44-45').
    Compustat retains 44 and 45; they will need a_i fallback in Stage S8.

No window filter is applied here; downstream uses active_window for time avg.

Run:
  python 04_klems_capital_share.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import PATHS, ensure_dir, load_config, setup_logger


S4_OUT = "s4_klems_sector_year.parquet"


# Map KLEMS sector code -> constituent individual NAICS2 codes.
# Built from the unique values seen in the KLEMS file.
KLEMS_GROUP_TO_NAICS2: dict[str, list[int]] = {
    "11": [11], "21": [21], "22": [22], "23": [23],
    "31-33": [31, 32, 33],
    "42": [42],
    "48-49": [48, 49],
    "51": [51], "52": [52], "53": [53], "54": [54],
    "56": [56], "61": [61], "62": [62],
    "71": [71], "72": [72], "81": [81],
}


def overlap_with_retained(components: list[int], retained: set[int]) -> bool:
    return bool(set(components) & retained)


def has_excluded(components: list[int], excluded: set[int]) -> bool:
    return bool(set(components) & excluded)


def main() -> None:
    config = load_config()
    logger = setup_logger("S4", config.get("log_level", "INFO"))

    naics_exclude = set(int(x) for x in config["naics_exclude"])
    a_lo, a_hi = config["a_i_clip"]
    year_floor = int(config["s1_keep_from_year"])

    # --- Determine retained NAICS2 from S1 output --------------------------
    s1_path = PATHS.intermediary / "s1_compustat_firmyear.parquet"
    if not s1_path.exists():
        raise FileNotFoundError(
            f"Stage S1 output missing at {s1_path}; run 01_load_compustat.py first."
        )
    s1 = pd.read_parquet(s1_path, columns=["ind2d"])
    retained_compustat = set(int(x) for x in s1["ind2d"].unique())
    logger.info(f"  Compustat-retained NAICS2: {sorted(retained_compustat)}")
    logger.info(f"  config naics_exclude     : {sorted(naics_exclude)}")

    # --- Read KLEMS --------------------------------------------------------
    path = PATHS.klems_csv
    logger.info(f"reading {path.relative_to(PATHS.project_root)}")
    df = pd.read_csv(path)
    logger.info(f"  raw shape: {df.shape}")

    df["sector_id"] = df["naics_target"].astype(str)
    unknown = set(df["sector_id"].unique()) - set(KLEMS_GROUP_TO_NAICS2.keys())
    if unknown:
        logger.warning(f"  unknown KLEMS sector codes: {sorted(unknown)}")
    df = df[df["sector_id"].isin(KLEMS_GROUP_TO_NAICS2)].copy()

    # --- Year floor (KLEMS starts 1997 so this is a no-op in practice) -----
    before = len(df)
    df = df[df["year"] >= year_floor]
    logger.info(f"  year >= {year_floor}: {before:,d} -> {len(df):,d}")

    # --- Filter KLEMS groups that overlap retained Compustat sectors -------
    df["components"] = df["sector_id"].map(KLEMS_GROUP_TO_NAICS2)
    df["overlap"] = df["components"].apply(
        lambda c: overlap_with_retained(c, retained_compustat)
    )
    df["partial_overlap_flag"] = df["components"].apply(
        lambda c: has_excluded(c, naics_exclude)
    )
    before = len(df)
    df = df[df["overlap"]].copy()
    n_partial = int(df["partial_overlap_flag"].sum())
    logger.info(
        f"  retained KLEMS groups (overlap with Compustat-retained): "
        f"{before:,d} -> {len(df):,d}"
    )
    if n_partial:
        partial_groups = sorted(
            df.loc[df["partial_overlap_flag"], "sector_id"].unique().tolist()
        )
        logger.warning(
            f"  partial-overlap KLEMS groups: {partial_groups} "
            f"(some constituent NAICS2 is excluded; treat with caution)"
        )

    # Diagnostic: retained Compustat NAICS2 that DO NOT appear in any KLEMS group.
    klems_covered_naics: set[int] = set()
    for grp in df["sector_id"].unique():
        klems_covered_naics |= set(KLEMS_GROUP_TO_NAICS2[grp])
    uncovered = sorted(retained_compustat - klems_covered_naics)
    if uncovered:
        logger.warning(
            f"  Compustat NAICS2 with NO KLEMS coverage: {uncovered} "
            f"(will need a_i fallback = {config['a_i_fallback']} in Stage S8)"
        )

    # --- Compute a_iy with clipping ----------------------------------------
    df["a_iy_raw"] = (1.0 - df["LS_VA"]).astype("float32")
    df["a_iy"] = df["a_iy_raw"].clip(lower=a_lo, upper=a_hi).astype("float32")
    df["a_iy_clipped_flag"] = (df["a_iy"] != df["a_iy_raw"]).astype(bool)
    n_clip = int(df["a_iy_clipped_flag"].sum())
    if n_clip:
        clipped_rows = (
            df.loc[df["a_iy_clipped_flag"], ["year", "sector_id", "a_iy_raw"]]
            .sort_values("a_iy_raw")
        )
        logger.warning(
            f"  clipped {n_clip} a_iy values outside [{a_lo}, {a_hi}]"
        )
        logger.warning(
            f"  most extreme clipped cells:\n{clipped_rows.head(5).to_string(index=False)}"
        )

    # --- Cross-sector shares (within retained KLEMS groups, by year) -------
    year_va = df.groupby("year")["value_added"].transform("sum")
    year_go = df.groupby("year")["gross_output"].transform("sum")
    df["va_share_iy"] = (df["value_added"] / year_va).astype("float32")
    df["go_share_iy"] = (df["gross_output"] / year_go).astype("float32")

    # --- Final tidy --------------------------------------------------------
    df = df.drop(columns=["components", "overlap", "naics_target"])
    df["year"] = df["year"].astype("int16")
    df["LS_VA"] = df["LS_VA"].astype("float32")
    df["LS_GO"] = df["LS_GO"].astype("float32")

    out_cols = [
        "year",
        "sector_id",
        "partial_overlap_flag",
        "lab_comp",
        "gross_output",
        "value_added",
        "LS_VA",
        "LS_GO",
        "a_iy_raw",
        "a_iy",
        "a_iy_clipped_flag",
        "va_share_iy",
        "go_share_iy",
    ]
    df_out = df[out_cols].copy()

    # --- Summary -----------------------------------------------------------
    aw_start = int(config["windows"][config["active_window"]]["start"])
    aw_end = int(config["windows"][config["active_window"]]["end"])
    sub = df_out[(df_out["year"] >= aw_start) & (df_out["year"] <= aw_end)]
    logger.info(f"  active window {aw_start}-{aw_end} rows: {len(sub):,d}")

    by_sector = (
        sub.groupby("sector_id")
        .agg(
            ls_va=("LS_VA", "mean"),
            a_i=("a_iy", "mean"),
            va_share=("va_share_iy", "mean"),
            go_share=("go_share_iy", "mean"),
            partial=("partial_overlap_flag", "first"),
        )
        .round(4)
        .sort_values("va_share", ascending=False)
    )
    logger.info(f"  active-window sector means:\n{by_sector.to_string()}")

    # Sanity: VA shares should sum to ~1 each year over the retained set
    yearly_va_sum = sub.groupby("year")["va_share_iy"].sum()
    yearly_go_sum = sub.groupby("year")["go_share_iy"].sum()
    logger.info(
        f"  yearly VA share sums: min={yearly_va_sum.min():.4f}, "
        f"max={yearly_va_sum.max():.4f}"
    )
    logger.info(
        f"  yearly GO share sums: min={yearly_go_sum.min():.4f}, "
        f"max={yearly_go_sum.max():.4f}"
    )

    # --- Write -------------------------------------------------------------
    out_path = ensure_dir(PATHS.intermediary) / S4_OUT
    df_out.to_parquet(out_path, index=False, compression="snappy")
    logger.info(
        f"wrote {out_path.relative_to(PATHS.project_root)} "
        f"({out_path.stat().st_size / 1e3:.2f} KB)"
    )


if __name__ == "__main__":
    sys.exit(main())
