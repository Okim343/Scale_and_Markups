"""Unit tests for the hybrid GNR-KLEMS alpha construction (Stage S4b).

Covers the deterministic core `build_hybrid_columns`: a normal row, an
a_sector fallback row (sector without KLEMS coverage), a missing-sector row
(no sector_id), and a negative-kelast row; plus the by-construction identity
kelast / (kelast + lelast_klems_sector_a) == a_sector, and that S4b and S8
share the identical `pooled_capital_share` helper.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]


def _load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s4b = _load_module("04b_build_hybrid_alpha.py", "s4b_build_hybrid_alpha")

A_BAR = 0.35
A_SECTOR_RAW = {"31-33": 0.40, "51": 0.30}  # "72" deliberately absent (fallback)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # normal, fallback (sector w/o KLEMS), missing-sector, negative-kelast
            "sector_id": ["31-33", "72", None, "51"],
            "melast": [0.50, 0.60, 0.55, 0.70],
            "kelast": [0.20, 0.10, 0.15, -0.10],
            "lelast": [0.25, 0.30, 0.28, 0.40],
            "rts": [0.95, 1.00, 0.98, 1.00],
        }
    )


def _built() -> pd.DataFrame:
    return s4b.build_hybrid_columns(_toy_frame(), A_SECTOR_RAW, A_BAR)


def test_normal_row_formula():
    out = _built()
    row = out.iloc[0]
    assert row["a_sector"] == pytest.approx(0.40)
    assert not bool(row["a_sector_fallback_flag"])
    assert row["a_sector_source"] == "active_window_klems_sector_mean_a_iy"
    # lelast_klems = 0.20*(1-0.40)/0.40 = 0.30 ; alpha = 0.50+0.20+0.30 = 1.00
    assert row["lelast_klems_sector_a"] == pytest.approx(0.30)
    assert row["alpha_hybrid_sector_a"] == pytest.approx(1.00)
    # diagnostic aliases preserve the raw GNR quantities
    assert row["lelast_gnr_raw"] == pytest.approx(0.25)
    assert row["alpha_gnr_raw"] == pytest.approx(0.95)
    assert row["alpha_no_labor"] == pytest.approx(0.70)


def test_fallback_and_missing_sector_use_a_bar():
    out = _built()
    for i in (1, 2):  # sector "72" (no KLEMS) and missing sector_id
        row = out.iloc[i]
        assert bool(row["a_sector_fallback_flag"])
        assert row["a_sector"] == pytest.approx(A_BAR)
        assert row["a_sector_source"] == "klems_va_weighted_a_bar"


def test_negative_kelast_gives_negative_imputed_labor():
    out = _built()
    row = out.iloc[3]
    assert row["a_sector"] == pytest.approx(0.30)
    # -0.10*(1-0.30)/0.30 = -0.23333...
    assert row["lelast_klems_sector_a"] == pytest.approx(-0.10 * 0.70 / 0.30)
    assert row["lelast_klems_sector_a"] < 0.0


def test_identity_kelast_over_kelast_plus_lelast_equals_a_sector():
    out = _built()
    ratio = out["kelast"] / (out["kelast"] + out["lelast_klems_sector_a"])
    np.testing.assert_allclose(ratio.to_numpy(), out["a_sector"].to_numpy())


def test_s4b_and_s8_share_pooled_capital_share_helper():
    import utils

    s8 = _load_module("08_assemble_bundle.py", "s8_assemble_bundle")
    assert s4b.pooled_capital_share is utils.pooled_capital_share
    assert s8.pooled_capital_share is utils.pooled_capital_share
