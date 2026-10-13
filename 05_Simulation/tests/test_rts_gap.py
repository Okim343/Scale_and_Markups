"""Model RTS gap moment ``alpha_rts_gap_top5_bottom50`` (Section 4.3 of the migration prompt).

Checked against a hand-built cross section with a known answer and against the
data function ``rts_gap`` of ``03_Empirical/02_Orbis/04_Moments/01_code/
06_alpha_moments.py`` on the same input (that code file is tracked; no data is
read).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from orbis_model.model.aggregator import RTS_GAP_KEY, alpha_rts_gap, pooled_moments
from orbis_model.model.market_batch import solve_batch

DATA_CODE = (Path(__file__).resolve().parents[2] / "03_Empirical" / "02_Orbis" / "04_Moments"
             / "01_code")


def _data_rts_gap():
    """Import ``rts_gap`` from the numbered data script (its folder holds moments_common)."""
    path = DATA_CODE / "06_alpha_moments.py"
    if not path.exists():
        pytest.skip("data-side 06_alpha_moments.py not found")
    sys.path.insert(0, str(DATA_CODE))
    try:
        spec = importlib.util.spec_from_file_location("orbis_alpha_moments", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DATA_CODE))
    return module.rts_gap


def _data_gap(sales, alpha, mask):
    """Run the data function on the active firms of an (M, H) cross section."""
    m_idx, _ = np.nonzero(mask)
    units = pd.DataFrame({
        "market": m_idx, "log_sales": np.log(sales[mask]), "rts": alpha[mask],
        "melast": 0.0, "lelast": 0.0,
    })
    return _data_rts_gap()(units, ["market"], tops={"top5": 0.95}, bottom=0.5)["top5"]


def _hand_built():
    # Inactive slots carry huge sales and alpha so any leak would show.
    sales = np.full((3, 6), 1e9)
    alpha = np.full((3, 6), 5.0)
    mask = np.zeros((3, 6), dtype=bool)
    # Market 0: mean alpha 1.0, dm = (-0.2, 0.2, -0.1, 0.1); pct = (.25, 1, .5, .75).
    sales[0, :4] = [10.0, 40.0, 20.0, 30.0]
    alpha[0, :4] = [0.8, 1.2, 0.9, 1.1]
    mask[0, :4] = True
    # Market 1: a tie, both firms get pct 0.75, so neither is top nor bottom.
    sales[1, :2] = [5.0, 5.0]
    alpha[1, :2] = [0.7, 1.3]
    mask[1, :2] = True
    # Market 2: mean alpha 1.1, dm = (-0.1, -0.1, 0.2); pct = (1, 1/3, 2/3).
    sales[2, :3] = [3.0, 1.0, 2.0]
    alpha[2, :3] = [1.0, 1.0, 1.3]
    mask[2, :3] = True
    return sales, alpha, mask


def test_hand_built_cross_section_known_answer():
    sales, alpha, mask = _hand_built()
    # Top (pct > 0.95): dm 0.2 (m0) and -0.1 (m2), mean 0.05.
    # Bottom (pct <= 0.5): dm -0.2, -0.1 (m0) and -0.1 (m2), mean -0.4/3.
    expected = 0.05 - (-0.4 / 3.0)
    assert alpha_rts_gap(sales, alpha, mask) == pytest.approx(expected, abs=1e-13)
    # Pooled over firms, not averaged over markets (that would give 0.175).
    assert abs(expected - 0.5 * (0.35 + 0.0)) > 1e-3


def test_hand_built_matches_data_function():
    sales, alpha, mask = _hand_built()
    assert alpha_rts_gap(sales, alpha, mask) == pytest.approx(_data_gap(sales, alpha, mask), abs=1e-14)


def test_random_masked_cross_section_matches_data_function():
    rng = np.random.default_rng(20261013)
    M, H = 60, 40
    n = rng.integers(1, H + 1, size=M)
    mask = np.arange(H)[None, :] < n[:, None]
    # Round sales so ties occur; scramble inactive slots.
    sales = np.where(mask, np.round(rng.lognormal(0.0, 1.0, (M, H)), 1) + 0.1, 1e12)
    alpha = np.where(mask, rng.uniform(0.7, 1.26, (M, H)), -3.0)
    model = alpha_rts_gap(sales, alpha, mask)
    assert np.isfinite(model)
    assert model == pytest.approx(_data_gap(sales, alpha, mask), abs=1e-13)


def test_gap_is_invariant_to_market_order_and_slot_position():
    sales, alpha, mask = _hand_built()
    base = alpha_rts_gap(sales, alpha, mask)
    perm = [2, 0, 1]
    shuffled = alpha_rts_gap(sales[perm][:, ::-1], alpha[perm][:, ::-1], mask[perm][:, ::-1])
    assert shuffled == pytest.approx(base, abs=1e-13)


def test_all_singleton_markets_give_nan():
    sales = np.array([[1.0, 0.0], [2.0, 0.0]])
    alpha = np.array([[0.9, 1.0], [1.1, 1.0]])
    mask = np.array([[True, False], [True, False]])
    assert np.isnan(alpha_rts_gap(sales, alpha, mask))


def test_pooled_moments_reports_gap_only_with_alpha():
    rng = np.random.default_rng(3)
    alpha = rng.uniform(0.7, 1.2, (6, 12))
    v = np.exp(rng.normal(0.0, 0.4, (6, 12)))
    mask = np.ones_like(alpha, dtype=bool)
    sol = solve_batch(alpha, v, eta=2.0, gamma=3.8, w=1.0, R=0.1, X_market=1.0,
                      a_i=0.35, active_mask=mask, backend="vectorized")
    without = pooled_moments(sol, mask, n_firms_cs=100.0)
    assert RTS_GAP_KEY not in without
    with_alpha = pooled_moments(sol, mask, n_firms_cs=100.0, alpha=alpha)
    assert with_alpha[RTS_GAP_KEY] == pytest.approx(alpha_rts_gap(sol.sales, alpha, mask))
    for key, value in without.items():
        assert with_alpha[key] == value
