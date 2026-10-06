"""
Stage R3 — Post-process GNR estimates into a firm-year RTS panel.

Reads:
  02_intermediary/gnr_panel_trimmed.parquet
  02_intermediary/step1_results.json
  02_intermediary/step2_results.json

Writes:
  02_intermediary/firm_year_rts_raw.parquet   (all trimmed rows; Step-2
                                               objects NaN outside Step-2 sample)
  03_outdata/firm_year_rts.parquet            (rows with finite RTS only)

Reconstructs, deterministically from the saved parameters (no re-optimization):
  eps, melast                       on the full trimmed sample
  omega, lomega, eta,
  kelast, lelast, rts               on the Step-2 sample
  alpha_gross_output = rts          (model-language alias, exported field)

Run:
  python 03_postprocess_gnr.py [--config config_smoke.yaml]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from utils import (  # utils puts 03_Empirical/00_GNR_core on sys.path
    PATHS,
    ensure_dir,
    intermediary_dir,
    outdata_dir,
    parse_config_arg,
    setup_logger,
)
from gnr_postprocess import reconstruct_firm_year

TRIM_IN = "gnr_panel_trimmed.parquet"
RAW_OUT = "firm_year_rts_raw.parquet"
FINAL_OUT = "firm_year_rts.parquet"

EXPORT_COLS = [
    "gvkey", "year", "ind2d",
    "eps", "melast", "kelast", "lelast",
    "omega", "lomega", "eta",
    "rts", "alpha_gross_output",
    "in_step2_sample", "rts_outlier_flag",
]


def main() -> None:
    config = parse_config_arg("Stage R3 — post-process GNR firm-year outputs")
    logger = setup_logger("R3", config.get("log_level", "INFO"))
    inter = intermediary_dir(config)

    # --- load inputs -----------------------------------------------------------
    for fname in [TRIM_IN, "step1_results.json", "step2_results.json"]:
        if not (inter / fname).exists():
            raise FileNotFoundError(
                f"{inter / fname} missing; run stages R1/R2 first."
            )
    est = pd.read_parquet(inter / TRIM_IN)
    with open(inter / "step1_results.json") as f:
        s1 = json.load(f)
    with open(inter / "step2_results.json") as f:
        s2 = json.load(f)
    logger.info(f"loaded trimmed panel ({len(est):,} rows) and step 1/2 results")

    est = reconstruct_firm_year(
        est, s1, s2, config["postprocess"]["rts_flag_range"], logger
    )

    # --- write -----------------------------------------------------------------------
    raw_path = ensure_dir(inter) / RAW_OUT
    est[EXPORT_COLS + ["s", "r", "k", "m", "l", "has_w"]].to_parquet(
        raw_path, index=False, compression="snappy"
    )
    logger.info(
        f"wrote {raw_path.relative_to(PATHS.project_root)} ({len(est):,} rows)"
    )

    final = est.loc[np.isfinite(est["rts"]), EXPORT_COLS].reset_index(drop=True)
    if len(final) == 0:
        raise RuntimeError("no rows with finite RTS — estimation failed upstream")
    final_path = ensure_dir(outdata_dir(config)) / FINAL_OUT
    final.to_parquet(final_path, index=False, compression="snappy")
    logger.info(
        f"wrote {final_path.relative_to(PATHS.project_root)} "
        f"({len(final):,} firm-year RTS rows)"
    )


if __name__ == "__main__":
    sys.exit(main())
