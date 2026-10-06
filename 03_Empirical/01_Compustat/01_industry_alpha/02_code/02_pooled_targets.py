"""
Stage S2 — Within-NAICS2 concentration/markup, then POOLED scalar targets.

NAICS2 (collapsed to `sector_id`) is the empirical "market" at which
within-market concentration is measured. This stage measures concentration and
the cost-weighted markup WITHIN each (year, sector_id) cell, then COLLAPSES
those measures to single pooled scalars by taking the sector-sales-weighted mean
ACROSS sector_id (after time-averaging over the active window). The pooled
scalars are what the structural model consumes — there is no sector dimension
in the bundle.

Reads:
  01_intermediary/s1_compustat_firmyear.parquet

Writes:
  01_intermediary/s2_firm_year_shares.parquet
      Firm-year panel from S1 with `s_ji` (within-(year, sector_id) sales
      share) appended. Feeds S7's markup-share slope regression and S8b's
      HHI panel. (NAICS2 retained here purely as calibration data.)

  01_intermediary/s2_sector_year_compustat.parquet
      Long table, one row per (year, sector_id): n_firms_cs, tot_sale_cs,
      tot_cogs_cs, mu_cw_iy, cr4/cr20/top1pct/top5pct (within-sector), and
      low_firm_flag. Feeds the pooled collapse below and S8b's emx_slope.

  01_intermediary/s2_pooled_targets.json
      The POOLED scalar concentration targets (sector-sales-weighted mean across
      sector_id of the window-averaged within-sector measures) plus the pooled
      total firm count. These are copied verbatim into aggregate_moments.yaml
      by Stage S8.

Run:
  python 02_pooled_targets.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from utils import PATHS, active_window, ensure_dir, load_config, setup_logger


# Diagnostic threshold: sector-years with fewer than this many firms are flagged.
LOW_FIRM_THRESHOLD = 20

S1_IN = "s1_compustat_firmyear.parquet"
S2_SECTOR_OUT = "s2_sector_year_compustat.parquet"
S2_FIRM_OUT = "s2_firm_year_shares.parquet"
S2_POOLED_OUT = "s2_pooled_targets.json"

# The four within-market concentration measures, with their pooled target names.
CONCENTRATION_COLS = {
    "cr4_sales_cs_iy": "cr4",
    "cr20_sales_cs_iy": "cr20",
    "top1pct_cs_iy": "top1pct",
    "top5pct_cs_iy": "top5pct",
}


def add_within_sector_shares(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Append s_ji = sale_j / Σ_h sale_h within each (year, sector_id)."""
    sector_total_sale = df.groupby(["year", "sector_id"])["sale"].transform("sum")
    df = df.copy()
    df["s_ji"] = (df["sale"] / sector_total_sale).astype("float32")
    n_zero = int((df["s_ji"] <= 0).sum())
    if n_zero:
        logger.warning(f"  {n_zero} firm-years have non-positive within-sector share")
    return df


def _aggregate_cell(group: pd.DataFrame) -> pd.Series:
    """Within-(year, sector_id) concentration and cost-weighted markup."""
    sales = group["sale"].to_numpy()
    cogs = group["cogs"].to_numpy()
    mu = group["markup"].to_numpy()

    n = sales.size
    tot_sale = sales.sum()
    tot_cogs = cogs.sum()

    mu_cw = float((cogs * mu).sum() / tot_cogs)

    order = np.argsort(-sales)
    cum_share = np.cumsum(sales[order]) / tot_sale

    cr4 = float(cum_share[min(3, n - 1)])
    cr20 = float(cum_share[min(19, n - 1)])

    # Top 1% / 5% use ceiling to guarantee at least 1 firm.
    k1 = max(1, int(np.ceil(0.01 * n)))
    k5 = max(1, int(np.ceil(0.05 * n)))
    top1 = float(cum_share[k1 - 1])
    top5 = float(cum_share[k5 - 1])

    return pd.Series(
        {
            "n_firms_cs": np.int32(n),
            "tot_sale_cs": float(tot_sale),
            "tot_cogs_cs": float(tot_cogs),
            "mu_cw_iy": np.float32(mu_cw),
            "cr4_sales_cs_iy": np.float32(cr4),
            "cr20_sales_cs_iy": np.float32(cr20),
            "top1pct_cs_iy": np.float32(top1),
            "top5pct_cs_iy": np.float32(top5),
        }
    )


def compute_sector_year(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Build the (year, sector_id) within-market long table."""
    logger.info("computing within-market (year, sector_id) aggregates")
    out = (
        df.groupby(["year", "sector_id"], group_keys=False)
        .apply(_aggregate_cell, include_groups=False)
        .reset_index()
    )
    out["low_firm_flag"] = (out["n_firms_cs"] < LOW_FIRM_THRESHOLD).astype(bool)
    out["year"] = out["year"].astype("int16")
    out["sector_id"] = out["sector_id"].astype("string")
    return out


def collapse_to_pooled(sector_df: pd.DataFrame, window: tuple[int, int], logger) -> dict:
    """Time-average within-market measures, then pool across sector_id.

    The pooled scalar for each concentration measure is the sector-sales-weighted
    mean ACROSS sector_id of the window-averaged WITHIN-sector measure:

        pooled_cr4 = Σ_s sales̄_s · c̄r4_s / Σ_s sales̄_s

    where bars denote the simple mean over active-window years. This mirrors the
    EMX Cournot calibration analogue, which pools within-sector concentration
    moments with sector expenditure/size weights.
    """
    s, e = window
    sub = sector_df[(sector_df["year"] >= s) & (sector_df["year"] <= e)].copy()
    # Window mean per sector_id of each within-market measure + size diagnostics.
    value_cols = ["n_firms_cs", "tot_sale_cs", *CONCENTRATION_COLS.keys()]
    per_sector = sub.groupby("sector_id", as_index=False)[value_cols].mean()

    weights = per_sector["tot_sale_cs"].to_numpy(dtype=float)
    wsum = weights.sum()
    pooled: dict[str, float] = {}
    for col, target_name in CONCENTRATION_COLS.items():
        vals = per_sector[col].to_numpy(dtype=float)
        pooled[target_name] = float(np.dot(weights, vals) / wsum)
    # Pooled population/size diagnostics.
    pooled["n_firms_cs"] = float(per_sector["n_firms_cs"].sum())
    pooled["tot_sale_cs"] = float(wsum)
    pooled["n_markets"] = int(len(per_sector))

    logger.info("  pooled concentration targets (sector-sales-weighted across sector_id):")
    for name in ("cr4", "cr20", "top1pct", "top5pct"):
        logger.info(f"    {name:9s} = {pooled[name]:.6f}")
    logger.info(f"    n_firms_cs (total) = {pooled['n_firms_cs']:.4f} over {pooled['n_markets']} markets")
    return {
        "active_window": {"start": s, "end": e},
        "weighting": "tot_sale_cs-weighted mean across sector_id of window-averaged "
        "within-sector measures",
        "targets": {k: pooled[k] for k in ("cr4", "cr20", "top1pct", "top5pct")},
        "n_firms_cs": pooled["n_firms_cs"],
        "tot_sale_cs": pooled["tot_sale_cs"],
        "n_markets": pooled["n_markets"],
        "per_sector": per_sector.round(6).to_dict(orient="records"),
    }


def main() -> None:
    config = load_config()
    logger = setup_logger("S2", config.get("log_level", "INFO"))
    aw = active_window(config)

    in_path = PATHS.intermediary / S1_IN
    logger.info(f"reading {in_path.relative_to(PATHS.project_root)}")
    df = pd.read_parquet(in_path)
    logger.info(f"  shape: {df.shape}")

    firm_df = add_within_sector_shares(df, logger)
    sector_df = compute_sector_year(firm_df, logger)
    pooled = collapse_to_pooled(sector_df, aw, logger)

    out_dir = ensure_dir(PATHS.intermediary)

    firm_out = out_dir / S2_FIRM_OUT
    firm_df.to_parquet(firm_out, index=False, compression="snappy")
    logger.info(
        f"wrote {firm_out.relative_to(PATHS.project_root)} "
        f"({firm_out.stat().st_size / 1e6:.2f} MB)"
    )

    sector_out = out_dir / S2_SECTOR_OUT
    sector_df.to_parquet(sector_out, index=False, compression="snappy")
    logger.info(
        f"wrote {sector_out.relative_to(PATHS.project_root)} "
        f"({sector_out.stat().st_size / 1e6:.2f} MB)"
    )

    pooled_out = out_dir / S2_POOLED_OUT
    with open(pooled_out, "w") as f:
        json.dump(pooled, f, indent=2)
    logger.info(
        f"wrote {pooled_out.relative_to(PATHS.project_root)} "
        f"({pooled_out.stat().st_size / 1e3:.2f} KB)"
    )


if __name__ == "__main__":
    sys.exit(main())
