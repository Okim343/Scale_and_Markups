"""
Stage S4b — Build the firm-year HYBRID GNR-KLEMS alpha panel.

The canonical model-facing firm-level scalability object is

    alpha_hybrid_sector_a = melast + kelast + kelast * (1 - a_sector) / a_sector

where `melast`/`kelast` are the GNR gross-output elasticities from the published
firm-year panel and `a_sector` is the active-window KLEMS value-added capital
share of the firm's sector (falling back to the aggregate KLEMS `a_bar` when a
sector has no KLEMS coverage). The imputed labor elasticity is

    lelast_klems_sector_a = kelast * (1 - a_sector) / a_sector,

so by construction kelast / (kelast + lelast_klems_sector_a) = a_sector.

The original GNR `lelast` (Compustat/headcount-contaminated labor measurement)
is preserved ONLY as the diagnostic column `lelast_gnr_raw`; it never feeds the
model-facing alpha, the copula (S6), or the alpha-sales diagnostics.

Reads:
  00_indata/04_salgado_data/firm_year_rts.parquet  (GNR melast/kelast/lelast/omega)
  01_intermediary/s1_compustat_firmyear.parquet     (sector_id merge)
  01_intermediary/s4_klems_sector_year.parquet      (active-window sector a_iy)

Writes:
  01_intermediary/s4b_firm_year_alpha_hybrid.parquet

Run:
  python 04b_build_hybrid_alpha.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import (
    COMPUSTAT_TO_SECTOR_ID,
    PATHS,
    active_window,
    ensure_dir,
    load_config,
    pooled_capital_share,
    setup_logger,
)


S4B_OUT = "s4b_firm_year_alpha_hybrid.parquet"

# Exact output schema (plan §3.1); asserted before writing.
OUTPUT_COLS = [
    "gvkey", "year", "ind2d", "sector_id", "sector_id_source",
    "a_sector", "a_sector_raw", "a_sector_fallback_flag", "a_sector_source",
    "melast", "kelast", "lelast_gnr_raw", "alpha_gnr_raw", "alpha_no_labor",
    "lelast_klems_sector_a", "alpha_hybrid_sector_a",
    "omega", "in_step2_sample", "rts_outlier_flag",
]


def build_hybrid_columns(
    df: pd.DataFrame, a_sector_raw_by_id: dict, a_bar: float
) -> pd.DataFrame:
    """Attach `a_sector` and construct the hybrid alpha columns (pure core).

    `a_sector` is the KLEMS active-window sector mean of a_iy looked up by
    `sector_id`, falling back to the aggregate `a_bar` where a sector has no
    KLEMS coverage OR `sector_id` is missing. Deterministic; this is the
    unit-tested core (normal / fallback / missing-sector / negative-kelast).

    The identity kelast / (kelast + lelast_klems_sector_a) == a_sector holds
    exactly by construction for every row (negative kelast included).
    """
    out = df.copy()
    out["a_sector_raw"] = out["sector_id"].map(a_sector_raw_by_id)
    out["a_sector_fallback_flag"] = out["a_sector_raw"].isna()
    out["a_sector"] = out["a_sector_raw"].fillna(a_bar)
    out["a_sector_source"] = np.where(
        out["a_sector_fallback_flag"],
        "klems_va_weighted_a_bar",
        "active_window_klems_sector_mean_a_iy",
    )
    out["lelast_gnr_raw"] = out["lelast"]
    out["alpha_gnr_raw"] = out["rts"]
    out["alpha_no_labor"] = out["melast"] + out["kelast"]
    out["lelast_klems_sector_a"] = out["kelast"] * (1.0 - out["a_sector"]) / out["a_sector"]
    out["alpha_hybrid_sector_a"] = (
        out["melast"] + out["kelast"] + out["lelast_klems_sector_a"]
    )
    return out


def build_hybrid_panel(config: dict, logger) -> pd.DataFrame:
    aw = active_window(config)
    a_fallback = float(config["a_i_fallback"])

    # --- 1. GNR firm-year panel; require melast/kelast (NOT lelast) ----------
    path = PATHS.salgado_firm_year_rts
    if not path.exists():
        raise FileNotFoundError(
            f"firm_year_rts.parquet missing at {path}; run the 00_Hubmer_RTS "
            "pipeline (it publishes the firm-level RTS panel)."
        )
    logger.info(f"reading {path.relative_to(PATHS.project_root)}")
    rts_cols = [
        "gvkey", "year", "ind2d", "melast", "kelast", "lelast", "rts",
        "omega", "in_step2_sample", "rts_outlier_flag",
    ]
    df = pd.read_parquet(path, columns=rts_cols)
    keys = ["gvkey", "year", "ind2d"]
    df = df.dropna(subset=keys)
    df = df.dropna(subset=["melast", "kelast"])
    logger.info(f"  firm-year rows with non-missing melast/kelast: {len(df):,}")

    # --- 2. sector_id from S1 merge, fallback to ind2d map -------------------
    s1 = pd.read_parquet(
        PATHS.intermediary / "s1_compustat_firmyear.parquet",
        columns=["gvkey", "year", "ind2d", "sector_id"],
    ).drop_duplicates(keys)
    df = df.merge(s1, on=keys, how="left", validate="many_to_one", indicator=True)
    df["sector_id_source"] = np.where(df["_merge"] == "both", "s1_compustat", "missing")
    df = df.drop(columns="_merge")

    mapped_sector = df["ind2d"].astype(int).map(COMPUSTAT_TO_SECTOR_ID)
    fill_mask = df["sector_id"].isna() & mapped_sector.notna()
    df.loc[fill_mask, "sector_id"] = mapped_sector.loc[fill_mask]
    df.loc[fill_mask, "sector_id_source"] = "ind2d_map"
    logger.info(
        f"  sector_id: filled {int(fill_mask.sum()):,} from ind2d map; "
        f"still missing {int(df['sector_id'].isna().sum()):,}"
    )

    # --- 3/4/5. active-window sector a_iy; a_bar fallback; hybrid alpha ------
    a_bar, _ = pooled_capital_share(aw, a_fallback, logger)
    klems = pd.read_parquet(PATHS.intermediary / "s4_klems_sector_year.parquet")
    klems_active = klems[(klems["year"] >= aw[0]) & (klems["year"] <= aw[1])]
    a_sector_raw_by_id = (
        klems_active.dropna(subset=["a_iy"])
        .groupby("sector_id")["a_iy"]
        .mean()
        .to_dict()
    )
    df = build_hybrid_columns(df, a_sector_raw_by_id, a_bar)
    logger.info(
        f"  sector a_iy for {len(a_sector_raw_by_id):,} sectors; "
        f"a_bar fallback rows {int(df['a_sector_fallback_flag'].sum()):,} "
        f"({100 * df['a_sector_fallback_flag'].mean():.1f}%); a_bar={a_bar:.6f}"
    )

    df = df[OUTPUT_COLS].reset_index(drop=True)
    logger.info(
        f"  alpha_hybrid_sector_a: mean={df['alpha_hybrid_sector_a'].mean():.4f}, "
        f"std={df['alpha_hybrid_sector_a'].std():.4f}, "
        f"n={df['alpha_hybrid_sector_a'].notna().sum():,}"
    )
    return df


def main() -> None:
    config = load_config()
    logger = setup_logger("S4b", config.get("log_level", "INFO"))

    df = build_hybrid_panel(config, logger)

    out_path = ensure_dir(PATHS.intermediary) / S4B_OUT
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info(
        f"wrote {out_path.relative_to(PATHS.project_root)} "
        f"({out_path.stat().st_size / 1e3:.2f} KB; {len(df)} rows)"
    )


if __name__ == "__main__":
    sys.exit(main())
