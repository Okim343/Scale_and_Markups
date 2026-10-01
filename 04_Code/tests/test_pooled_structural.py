"""Sector-free pooling, exposure aggregation, and resource accounting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from steady_state.model.aggregator import aggregate_pooled_markets, exposure_weights
from steady_state.model.normalization import PooledParams, solve_pooled_economy
from steady_state.model.pool import derived_z, draw_pool


def _exogenous_mask():
    # EMX-style exogenous entry: first two slots active in each sector.
    mask = np.zeros((2, 5), dtype=bool)
    mask[:, :2] = True
    return mask


def _pooled_equilibrium():
    alpha = np.ones((2, 5))
    z = np.array([
        [3.0, 2.0, 1.0, 0.5, 0.2],
        [2.5, 1.5, 0.8, 0.4, 0.1],
    ])
    # Pin phi_v so this accounting/exposure test is independent of the
    # production DEFAULT_PHI_V primitive (0.45 keeps the toy economy's C > 0).
    params = PooledParams(
        xi=8.0, N=2.0, gamma=3.8, eta=2.0, a=0.35, H=5, phi_v=0.45
    )
    return solve_pooled_economy(
        alpha,
        z,
        params,
        w=1.0,
        R=0.1,
        X_market=1.0,
        n_firms_cs=100.0,
        active_mask=_exogenous_mask(),
        market_solver_kwargs={"backend": "vectorized"},
    )


def test_pooled_forward_solve_and_population_divisor():
    eq = _pooled_equilibrium()
    assert not eq.participation.pool_binds.any()
    assert not eq.participation.empty_market.any()
    np.testing.assert_array_equal(eq.participation.n_active, [2, 2])
    assert eq.aggregates.mean_active_count == 2.0
    assert eq.aggregates.population_markets == 50.0
    assert eq.aggregates.population_scale == 25.0
    assert eq.aggregates.C > 0.0 and eq.aggregates.P > 0.0
    assert eq.aggregates.L > 0.0 and eq.aggregates.K > 0.0
    # EMX entry: no per-period operating cost.
    assert eq.operating_cost_labor == 0.0


def test_exposure_aggregator_uses_normalized_levels_and_weight_hook():
    eq = _pooled_equilibrium()
    solution = eq.participation.solution
    mask = eq.participation.active_mask
    omega = np.array([0.8, 0.2])
    aggregate = aggregate_pooled_markets(
        solution,
        mask,
        eta=2.0,
        operating_cost_labor=eq.operating_cost_labor,
        n_firms_cs=100.0,
        w=1.0,
        R=0.1,
        a=0.35,
        weights=omega,
    )
    Y = solution.Y_j_normalized
    P_j = solution.P_j_normalized
    # The CES composite is now the gross output Q; consumption is net of
    # materials (C = Q - M).
    expected_Q_sample = (np.sum(omega ** 0.5 * Y ** 0.5)) ** 2.0
    expected_scale = aggregate.population_scale
    expected_Q = expected_scale * expected_Q_sample
    expected_P = np.sum(omega * P_j ** -1.0) ** -1.0
    assert aggregate.Q == pytest.approx(expected_Q)
    assert aggregate.P == pytest.approx(expected_P)
    # Materials for the resource constraint are (1-phi_v) of total variable
    # cost on the population scale (matching Q).
    from steady_state.model.pricing import DEFAULT_PHI_V
    expected_M = expected_scale * (1.0 - DEFAULT_PHI_V) * float(solution.cost[mask].sum())
    assert aggregate.M == pytest.approx(expected_M)
    assert aggregate.C == pytest.approx(aggregate.Q - aggregate.M)
    np.testing.assert_allclose(aggregate.exposure_weights, omega)
    np.testing.assert_allclose(exposure_weights(2), [0.5, 0.5])


def test_resource_accounting_uses_active_firms_only():
    eq = _pooled_equilibrium()
    solution = eq.participation.solution
    mask = eq.participation.active_mask
    baseline = aggregate_pooled_markets(
        solution, mask, eta=2.0, operating_cost_labor=0.0,
        n_firms_cs=100.0, w=1.0, R=0.1, a=0.35,
    )
    # Inactive firms must not enter the resource totals, however large their
    # (masked-out) cost entries are.
    corrupted_cost = solution.cost.copy()
    corrupted_cost[~mask] = 1e12
    corrupted = replace(solution, cost=corrupted_cost)
    active_only = aggregate_pooled_markets(
        corrupted, mask, eta=2.0, operating_cost_labor=0.0,
        n_firms_cs=100.0, w=1.0, R=0.1, a=0.35,
    )
    assert active_only.L == baseline.L
    assert active_only.K == baseline.K


def test_pool_draw_is_reproducible_and_common_shape():
    support = np.linspace(0.6, 1.1, 20)
    a = draw_pool(support, xi=8.0, rho_bar=0.5, N=3.0, M=4, H=6, rng=12)
    b = draw_pool(support, xi=8.0, rho_bar=0.5, N=3.0, M=4, H=6, rng=12)
    assert a.alpha.shape == (4, 6)
    assert a.active_mask.shape == (4, 6)
    assert a.active_mask.sum(axis=1).min() >= 1
    np.testing.assert_array_equal(a.alpha, b.alpha)
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.tilde_alpha, b.tilde_alpha)
    np.testing.assert_array_equal(a.active_mask, b.active_mask)


def test_v_min_threads_into_draw_and_scales_v():
    # v_min is a pure multiplicative scale on v; doubling it must double every
    # capability draw and leave alpha / the mask untouched.
    support = np.linspace(0.6, 1.1, 20)
    base = draw_pool(support, xi=8.0, rho_bar=0.5, N=3.0, M=4, H=6, rng=7, v_min=1.0)
    scaled = draw_pool(support, xi=8.0, rho_bar=0.5, N=3.0, M=4, H=6, rng=7, v_min=2.0)
    np.testing.assert_allclose(scaled.v, 2.0 * base.v)
    np.testing.assert_array_equal(scaled.alpha, base.alpha)
    np.testing.assert_array_equal(scaled.active_mask, base.active_mask)


def test_pool_draw_preserves_pinned_crn_streams_after_v_relabeling():
    support = np.array([0.6, 0.75, 0.9, 1.05, 1.2])
    draw = draw_pool(
        support, xi=4.5, rho_bar=0.37, N=2.5, M=3, H=4,
        rng=20260707, v_min=1.0,
    )
    expected_alpha = np.array([
        [0.75, 0.6, 1.2, 0.75],
        [1.2, 1.2, 1.2, 0.9],
        [0.9, 0.9, 0.75, 1.05],
    ])
    expected_tilde = np.array([
        [-0.52440051, -1.28155157, 1.28155157, -0.52440051],
        [1.28155157, 1.28155157, 1.28155157, 0.0],
        [0.0, 0.0, -0.52440051, 0.52440051],
    ])
    expected_mask = np.array([
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=bool)
    expected_u = np.array([
        [0.33025139, 0.82537315, 0.61276711, 0.42379837],
        [0.54177985, 0.65109088, 0.79591626, 0.61536437],
        [0.42668815, 0.19500429, 0.61910358, 0.61142994],
    ])
    u = 1.0 - draw.v ** (-4.5)

    np.testing.assert_array_equal(draw.alpha, expected_alpha)
    np.testing.assert_allclose(draw.tilde_alpha, expected_tilde, rtol=0.0, atol=5e-9)
    np.testing.assert_array_equal(draw.active_mask, expected_mask)
    np.testing.assert_allclose(u, expected_u, rtol=0.0, atol=5e-9)


def test_derived_z_round_trips_phase_one_identity_with_unit_x_ref():
    alpha = np.array([0.7, 0.9, 1.1, 1.3])
    v = np.array([1.4, 1.8, 2.2, 2.7])
    for y_hat in (0.5, 1.0, 3.0):
        z_diag = derived_z(alpha, v, y_hat=y_hat)
        v_round_trip = alpha * z_diag ** (1.0 / alpha) * y_hat ** (1.0 - 1.0 / alpha)
        np.testing.assert_allclose(v_round_trip, v, rtol=1e-14, atol=1e-14)


def test_pooled_params_anchor_fields_and_validation():
    params = PooledParams(
        xi=8.0, N=2.0, gamma=3.8, eta=2.0, v_min=2.0, y_hat=3.0,
        a=0.35, H=5,
    )
    assert params.v_min == 2.0
    assert params.y_hat == 3.0
    assert PooledParams(xi=8.0, N=2.0, gamma=3.8, eta=2.0, a=0.35, H=5).v_min == 1.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            PooledParams(xi=8.0, N=2.0, gamma=3.8, eta=2.0, v_min=bad, a=0.35, H=5)
    with pytest.raises(ValueError):
        PooledParams(xi=8.0, N=2.0, gamma=3.8, eta=2.0, a=0.35, H=5, y_hat=0.0)
