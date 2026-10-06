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

from utils import (  # utils puts 03_Empirical/00_GNR_core on sys.path
    PATHS,
    ensure_dir,
    intermediary_dir,
    parse_config_arg,
    resolve_path,
    setup_logger,
)
from gnr_panel import construct_gnr_vars, trim_panel

RAW_OUT = "gnr_panel.parquet"
TRIM_OUT = "gnr_panel_trimmed.parquet"

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

    df = construct_gnr_vars(
        df, logger, build_materials_alt1=bool(cfg_panel.get("build_materials_alt1", True))
    )
    return df


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
