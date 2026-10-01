from __future__ import annotations

import numpy as np
import pytest

from steady_state.model.normalization import (
    PooledParams,
    euler_R,
    solve_full_pool_reference,
    solve_pooled_ge,
    solve_pooled_ge_anchored,
)
from steady_state.model.pricing import PricingRegime


def _draw():
    alpha = np.ones((3, 5))
    z = np.array([
        [3.0, 2.0, 1.0, 0.5, 0.2],
        [2.8, 1.8, 0.9, 0.4, 0.15],
        [2.5, 1.6, 0.8, 0.3, 0.1],
    ])
    return alpha, z


def _params():
    return PooledParams(xi=8.0, N=4.0, gamma=3.8, eta=2.0, a=0.35, H=5)


def test_euler_R_matches_closed_form():
    assert euler_R(0.96, 0.06) == pytest.approx(1 / 0.96 - 0.94)


def test_full_pool_reference_is_ge_normalized_and_no_operating_cost():
    alpha, z = _draw()
    ref = solve_full_pool_reference(
        alpha, z, _params(), n_firms_cs=100,
        market_solver_kwargs={"backend": "vectorized"},
    )
    assert ref.P == pytest.approx(1.0, abs=2e-4)
    assert ref.L == pytest.approx(1.0, abs=2e-4)
    # EMX entry: no per-period operating cost in the static cross section.
    assert ref.wf_reference == 0.0


def test_exogenous_mask_shared_across_allocations():
    alpha, z = _draw()
    mask = np.ones_like(alpha, dtype=bool)
    market = solve_pooled_ge(
        alpha, z, _params(), n_firms_cs=100, active_mask=mask,
        market_solver_kwargs={"backend": "vectorized"},
    )
    planner = solve_pooled_ge(
        alpha, z, _params(), n_firms_cs=100, regime=PricingRegime.PLANNER,
        active_mask=market.participation.active_mask,
        wf_reference=market.participation.wf_reference,
        market_solver_kwargs={"backend": "vectorized"},
    )
    np.testing.assert_array_equal(planner.participation.active_mask, market.participation.active_mask)
    assert market.participation.wf_reference == 0.0
    assert market.P_agg == pytest.approx(1.0, abs=2e-3)
    assert market.L_agg == pytest.approx(1.0, abs=2e-3)


def test_anchored_ge_solves_price_labor_and_median_output():
    alpha, v_unit = _draw()
    mask = np.ones_like(alpha, dtype=bool)
    params = _params()
    eq = solve_pooled_ge_anchored(
        alpha, v_unit, params, n_firms_cs=100, active_mask=mask,
        market_solver_kwargs={"backend": "vectorized"},
        max_nfev=60,
    )
    assert eq.converged
    assert eq.regime is PricingRegime.MARKET
    assert eq.P_agg == pytest.approx(1.0, abs=2e-4)
    assert eq.L_agg == pytest.approx(1.0, abs=2e-4)
    assert eq.y_anchor == pytest.approx(eq.y_hat, abs=2e-4)
    assert eq.v_min > 0.0
    np.testing.assert_allclose(eq.v, eq.v_min * v_unit)


def test_anchored_ge_rejects_non_market_regime():
    alpha, v_unit = _draw()
    mask = np.ones_like(alpha, dtype=bool)
    with pytest.raises(ValueError, match="only valid for MARKET"):
        solve_pooled_ge_anchored(
            alpha, v_unit, _params(), n_firms_cs=100, active_mask=mask,
            regime=PricingRegime.PLANNER,
        )
