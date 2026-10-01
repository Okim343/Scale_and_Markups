"""
Shared utilities for the hybrid EMX/scalability empirical pipeline.

Provides:
  - PATHS         : canonical project directories.
  - load_config() : reads the master config.yaml.
  - setup_logger(): configures a stage-specific logger.
  - active_window(): returns (start, end) for the active window in the config.

All Stage S* scripts import from this module so paths, config, and logging
behave consistently across the pipeline.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# This file lives at:
#   <PROJECT_ROOT>/03_Empirical/01_industry_alpha/02_code/utils.py
# so PROJECT_ROOT is parents[3].
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[3]
EMP_ROOT = _THIS.parents[1]               # 01_industry_alpha
INDATA = EMP_ROOT / "00_indata"
INTERMEDIARY = EMP_ROOT / "01_intermediary"
CODE = EMP_ROOT / "02_code"
OUTDATA = EMP_ROOT / "03_outdata"


@dataclass(frozen=True)
class _Paths:
    project_root: Path = PROJECT_ROOT
    emp_root: Path = EMP_ROOT
    indata: Path = INDATA
    intermediary: Path = INTERMEDIARY
    code: Path = CODE
    outdata: Path = OUTDATA

    # Specific raw input files used across the pipeline
    @property
    def compustat_markup_firmyear(self) -> Path:
        return self.indata / "03_Compustat" / "markup_firm_year.dta"

    @property
    def compustat_markup_cw(self) -> Path:
        return self.indata / "03_Compustat" / "markup_cost_weighted.dta"

    @property
    def compustat_markup_sw(self) -> Path:
        return self.indata / "03_Compustat" / "markup_sales_weighted.dta"

    @property
    def bds_naics2_fz(self) -> Path:
        return self.indata / "01_bds_naics2" / "bds2023_sec_fz.csv"

    @property
    def klems_csv(self) -> Path:
        return self.indata / "klems_labor_share_naics2_with_VA_1997_2023.csv"

    @property
    def salgado_naics_rts(self) -> Path:
        return self.indata / "04_salgado_data" / "naics_rts.csv"

    @property
    def salgado_firm_year_rts(self) -> Path:
        # Optional; present only after firm-level RTS is delivered.
        return self.indata / "04_salgado_data" / "firm_year_rts.parquet"


PATHS = _Paths()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the master config.yaml. Defaults to <code>/config.yaml."""
    if path is None:
        path = PATHS.code / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def active_window(config: dict[str, Any]) -> tuple[int, int]:
    """Return (start_year, end_year) for the currently active window."""
    key = config["active_window"]
    w = config["windows"][key]
    return int(w["start"]), int(w["end"])


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure a stage-level logger writing to stdout with timestamps."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    # Avoid duplicate handlers if the same script is re-imported.
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Make sure a directory exists; return the path for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# KLEMS aggregate capital share
# -----------------------------------------------------------------------------
def pooled_capital_share(
    window: tuple[int, int], a_fallback: float, logger
) -> tuple[float, dict]:
    """VA-weighted aggregate capital share `a` from KLEMS (model.typ Panel A).

    a = Σ_s VA_s · a_s / Σ_s VA_s, with VA_s and a_s the active-window means of
    the KLEMS group's value added and capital share a_iy = 1 - LS_VA.

    Shared by Stage S4b (04b_build_hybrid_alpha.py) and Stage S8
    (08_assemble_bundle.py) so the aggregate `a_bar` used to construct the
    hybrid alpha and the model-facing `a` are computed identically.
    """
    s4 = pd.read_parquet(INTERMEDIARY / "s4_klems_sector_year.parquet")
    s, e = window
    win = s4[(s4["year"] >= s) & (s4["year"] <= e)]
    per_group = win.groupby("sector_id", as_index=False).agg(
        a_i=("a_iy", "mean"), value_added=("value_added", "mean")
    )
    va = per_group["value_added"].to_numpy(dtype=float)
    a_i = per_group["a_i"].to_numpy(dtype=float)
    if va.sum() <= 0:
        logger.warning(f"  KLEMS VA sum non-positive; falling back to a={a_fallback}")
        return float(a_fallback), {"source": "fallback", "n_groups": 0}
    a = float(np.dot(va, a_i) / va.sum())
    logger.info(
        f"  pooled capital share a (KLEMS VA-weighted) = {a:.6f} "
        f"over {len(per_group)} groups"
    )
    return a, {
        "source": "klems_va_weighted",
        "n_groups": int(len(per_group)),
        "formula": "Σ VA_s·a_s / Σ VA_s (active-window means)",
    }


# -----------------------------------------------------------------------------
# Sector identifier convention
# -----------------------------------------------------------------------------
# All stages downstream of S1 key on `sector_id`, a string label that matches
# KLEMS/BDS native granularity. Compustat NAICS2 codes that are split by KLEMS
# (31, 32, 33; 44, 45; 48) are collapsed into the corresponding KLEMS group.
# 44-45 has no KLEMS coverage and uses the a_i_fallback at S8 assembly.
COMPUSTAT_TO_SECTOR_ID: dict[int, str] = {
    11: "11", 21: "21", 22: "22", 23: "23",
    31: "31-33", 32: "31-33", 33: "31-33",
    42: "42",
    44: "44-45", 45: "44-45",
    48: "48-49",
    51: "51", 54: "54", 56: "56",
    61: "61", 71: "71", 72: "72",
}


def map_to_sector_id(ind2d) -> "str | pd.Series":
    """Map Compustat NAICS2 (int or Series) to canonical sector_id."""
    if isinstance(ind2d, pd.Series):
        return ind2d.astype(int).map(COMPUSTAT_TO_SECTOR_ID)
    return COMPUSTAT_TO_SECTOR_ID[int(ind2d)]
