"""
Stage R4 — Export GNR RTS outputs for the 01_industry_alpha pipeline.

Reads:
  03_outdata/firm_year_rts.parquet            (from 03_postprocess_gnr.py)

Writes:
  03_outdata/naics_rts_gnr.csv                (NAICS, RTS — sector means over
                                               the export window; same format
                                               as the legacy Salgado file)
  03_outdata/rts_diagnostics.json             (counts and RTS moments)
  [publish] 01_industry_alpha/00_indata/04_salgado_data/firm_year_rts.parquet
  [publish] 01_industry_alpha/00_indata/04_salgado_data/naics_rts_gnr.csv

SAFETY / backward compatibility:
  - The original Salgado file 04_salgado_data/naics_rts.csv is the live
    input of the legacy F_alpha fallback path. It is NEVER overwritten
    unless export.overwrite_legacy_naics_rts is true, in which case a
    timestamped backup is created first.
  - Publishing firm_year_rts.parquet to 04_salgado_data is the intended
    integration point (05_build_F_alpha.py auto-detects it). Disable with
    export.publish_firm_year_to_alpha_pipeline: false.

Run:
  python 04_export_rts_for_alpha_pipeline.py [--config config_smoke.yaml]
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from utils import (
    PATHS,
    ensure_dir,
    outdata_dir,
    parse_config_arg,
    setup_logger,
)

FY_IN = "firm_year_rts.parquet"
NAICS_OUT = "naics_rts_gnr.csv"
DIAG_OUT = "rts_diagnostics.json"


def sector_aggregate(fy: pd.DataFrame, window, logger) -> pd.DataFrame:
    """Mean alpha_gross_output by NAICS2, optionally within a year window."""
    df = fy
    if window:
        y0, y1 = int(window[0]), int(window[1])
        df = fy[(fy["year"] >= y0) & (fy["year"] <= y1)]
        logger.info(
            f"  export window [{y0}, {y1}]: {len(df):,} of {len(fy):,} rows"
        )
        if len(df) == 0:
            raise ValueError(
                f"no firm-year RTS rows inside the export window [{y0}, {y1}]"
            )
    agg = (
        df.groupby("ind2d")["alpha_gross_output"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"ind2d": "NAICS", "mean": "RTS", "size": "n_obs"})
    )
    logger.info(
        "  sector means:\n"
        + agg.round({"RTS": 4}).to_string(index=False)
    )
    return agg


def build_diagnostics(fy: pd.DataFrame, agg: pd.DataFrame, window) -> dict:
    rts = fy["alpha_gross_output"].to_numpy()
    pcts = np.percentile(rts, [1, 10, 25, 50, 75, 90, 99])
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_firm_year_rows": int(len(fy)),
        "n_firms": int(fy["gvkey"].nunique()),
        "year_range": [int(fy["year"].min()), int(fy["year"].max())],
        "export_window": list(window) if window else None,
        "rts_moments": {
            "mean": float(rts.mean()),
            "sd": float(rts.std()),
            "median": float(pcts[3]),
            "p1": float(pcts[0]), "p10": float(pcts[1]), "p25": float(pcts[2]),
            "p75": float(pcts[4]), "p90": float(pcts[5]), "p99": float(pcts[6]),
        },
        "n_rts_outlier_flagged": int(fy["rts_outlier_flag"].sum()),
        "counts_by_year": fy.groupby("year").size().astype(int).to_dict(),
        "counts_by_naics2": fy.groupby("ind2d").size().astype(int).to_dict(),
        "sector_means_window": {
            int(r.NAICS): {"RTS": float(r.RTS), "n_obs": int(r.n_obs)}
            for r in agg.itertuples()
        },
    }


def main() -> None:
    config = parse_config_arg("Stage R4 — export RTS for the alpha pipeline")
    logger = setup_logger("R4", config.get("log_level", "INFO"))
    out_dir = outdata_dir(config)
    cfg_exp = config["export"]

    fy_path = out_dir / FY_IN
    if not fy_path.exists():
        raise FileNotFoundError(f"{fy_path} missing; run 03_postprocess_gnr.py first.")
    fy = pd.read_parquet(fy_path)
    logger.info(
        f"loaded {fy_path.relative_to(PATHS.project_root)} ({len(fy):,} rows)"
    )
    required = {"gvkey", "year", "ind2d", "alpha_gross_output", "rts_outlier_flag"}
    missing = required - set(fy.columns)
    if missing:
        raise KeyError(f"firm_year_rts.parquet missing columns: {sorted(missing)}")

    # --- sector aggregation + diagnostics --------------------------------------
    window = cfg_exp.get("window")
    agg = sector_aggregate(fy, window, logger)
    naics_path = ensure_dir(out_dir) / NAICS_OUT
    agg[["NAICS", "RTS"]].to_csv(naics_path, index=False, float_format="%.4f")
    logger.info(f"wrote {naics_path.relative_to(PATHS.project_root)}")

    diag = build_diagnostics(fy, agg, window)
    diag_path = out_dir / DIAG_OUT
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    logger.info(f"wrote {diag_path.relative_to(PATHS.project_root)}")

    # --- publish to the alpha pipeline ------------------------------------------
    salgado_dir = PATHS.alpha_salgado_dir
    if bool(cfg_exp.get("publish_firm_year_to_alpha_pipeline", False)):
        ensure_dir(salgado_dir)
        target = salgado_dir / "firm_year_rts.parquet"
        if target.exists():
            logger.warning(
                f"  overwriting existing {target.relative_to(PATHS.project_root)} "
                "(previous GNR publication)"
            )
        shutil.copy2(fy_path, target)
        shutil.copy2(naics_path, salgado_dir / NAICS_OUT)
        logger.info(
            f"published firm_year_rts.parquet and {NAICS_OUT} to "
            f"{salgado_dir.relative_to(PATHS.project_root)} — "
            "05_build_F_alpha.py will now prefer the firm-year RTS path"
        )
    else:
        logger.info(
            "publish_firm_year_to_alpha_pipeline=false: outputs NOT copied to "
            "the alpha pipeline (alpha pipeline behavior unchanged)"
        )

    # --- optional legacy overwrite (off by default) --------------------------------
    legacy = salgado_dir / "naics_rts.csv"
    if bool(cfg_exp.get("overwrite_legacy_naics_rts", False)):
        if legacy.exists():
            backup = legacy.with_name(
                f"naics_rts_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )
            shutil.copy2(legacy, backup)
            logger.warning(
                f"  backed up legacy Salgado file to "
                f"{backup.relative_to(PATHS.project_root)}"
            )
        agg[["NAICS", "RTS"]].to_csv(legacy, index=False, float_format="%.4f")
        logger.warning(
            f"  OVERWROTE {legacy.relative_to(PATHS.project_root)} with GNR "
            "sector means (overwrite_legacy_naics_rts=true)"
        )
    else:
        logger.info(
            "legacy naics_rts.csv left untouched "
            "(GNR sector means are in naics_rts_gnr.csv)"
        )


if __name__ == "__main__":
    sys.exit(main())
