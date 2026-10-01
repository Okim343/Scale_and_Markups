"""Gross-output / materials layer (model.typ §1, roundabout closure).

Pins the equations added by the value-added → gross-output migration:
* Ω^g closed form (@eq:omega_gross) and its phi_v → 1 value-added limit.
* firm profit = revenue − TC = (1 − α/μ)·revenue (@eq:profits).
* aggregate mu_cw_alpha = aggregate revenue / aggregate variable cost (@eq:mu_cw_identity).
* Q = C + M and materials_share = (1 − phi_v)/mu_cw_alpha (@eq:resource, validation).
"""

from __future__ import annotations

import numpy as np
import pytest

from steady_state.model.aggregator import aggregate_pooled_markets, pooled_moments
from steady_state.model.market import _omega, _omega_gross, solve
from steady_state.model.market_batch import solve_batch
from steady_state.model.normalization import PooledParams, solve_pooled_ge
from steady_state.model.pool import draw_pool
from steady_state.model.pricing import DEFAULT_PHI_V


GAMMA = 3.8
ETA = 2.0
W = 1.0
R = 0.10
A_I = 0.35
PHI_V = 0.45
Y_HAT = 7.25


# ---------------------------------------------------------------------------
# Ω^g closed form and limits
# ---------------------------------------------------------------------------


def test_omega_gross_matches_cobb_douglas_closed_form():
    omega_va = _omega(W, R, A_I)
    expected = (omega_va / PHI_V) ** PHI_V * (1.0 / (1.0 - PHI_V)) ** (1.0 - PHI_V)
    assert _omega_gross(W, R, A_I, PHI_V) == pytest.approx(expected)
    # Explicit price argument P should not be hard-coded to 1.
    P = 1.7
    expected_P = (omega_va / PHI_V) ** PHI_V * (P / (1.0 - PHI_V)) ** (1.0 - PHI_V)
    assert _omega_gross(W, R, A_I, PHI_V, P) == pytest.approx(expected_P)


def test_omega_gross_phi_v_to_one_recovers_value_added():
    assert _omega_gross(W, R, A_I, 1.0) == pytest.approx(_omega(W, R, A_I))
    # Approaching 1 from below is continuous.
    near = _omega_gross(W, R, A_I, 1.0 - 1e-6)
    assert near == pytest.approx(_omega(W, R, A_I), rel=1e-3)


# ---------------------------------------------------------------------------
# firm profit and total variable cost (heterogeneous α)
# ---------------------------------------------------------------------------


def test_firm_profit_equals_revenue_minus_tc():
    n = 4
    alpha = np.array([0.7, 0.9, 1.1, 1.3])
    v = np.array([2.0, 1.2, 0.8, 0.4])
    sol = solve(
        n=n, alpha=alpha, v=v, eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, phi_v=PHI_V, y_hat=Y_HAT,
    )
    assert sol.converged
    revenue = sol.p * sol.y
    # TC = α·Ω^g/v·y_hat·(y/y_hat)^(1/α) recomputed from primitives.
    Omega_g = _omega_gross(W, R, A_I, PHI_V)
    tc = alpha * (Omega_g / v) * Y_HAT * (sol.y / Y_HAT) ** (1.0 / alpha)
    np.testing.assert_allclose(sol.cost, tc, rtol=1e-7)
    np.testing.assert_allclose(sol.d, revenue - tc, rtol=1e-8)
    np.testing.assert_allclose(sol.d, (1.0 - alpha / sol.mu) * revenue, rtol=1e-8)


# ---------------------------------------------------------------------------
# aggregate markup, resource constraint, materials share
# ---------------------------------------------------------------------------


def _batch_and_mask():
    alpha = np.array([[0.7, 0.9, 1.1, 1.3], [0.6, 0.85, 1.0, 1.2]])
    v = np.array([[2.0, 1.2, 0.8, 0.4], [1.8, 1.1, 0.7, 0.3]])
    solution = solve_batch(
        alpha, v, eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, phi_v=PHI_V, backend="vectorized",
    )
    return solution, np.ones_like(alpha, dtype=bool)


def test_aggregate_mu_cw_alpha_is_revenue_over_variable_cost():
    solution, mask = _batch_and_mask()
    moments = pooled_moments(solution, mask, n_firms_cs=100, phi_v=PHI_V)
    expected = float(np.sum(solution.sales[mask]) / np.sum(solution.cost[mask]))
    assert moments["mu_cw_alpha"] == pytest.approx(expected)


def test_resource_constraint_q_equals_c_plus_m_and_materials_share():
    solution, mask = _batch_and_mask()
    agg = aggregate_pooled_markets(
        solution, mask, eta=ETA, operating_cost_labor=0.0, n_firms_cs=100.0,
        w=W, R=R, a=A_I, phi_v=PHI_V,
    )
    assert agg.Q == pytest.approx(agg.C + agg.M)
    moments = pooled_moments(solution, mask, n_firms_cs=100, phi_v=PHI_V)
    assert moments["materials_share"] == pytest.approx((1.0 - PHI_V) / moments["mu_cw_alpha"])


def test_phi_v_to_one_zeroes_materials():
    solution, mask = _batch_and_mask()
    agg = aggregate_pooled_markets(
        solution, mask, eta=ETA, operating_cost_labor=0.0, n_firms_cs=100.0,
        w=W, R=R, a=A_I, phi_v=1.0,
    )
    assert agg.M == pytest.approx(0.0)
    assert agg.C == pytest.approx(agg.Q)


# ---------------------------------------------------------------------------
# GE closure with materials netted out of consumption
# ---------------------------------------------------------------------------


def test_ge_consumption_is_net_of_materials():
    support = np.linspace(0.7, 1.2, 40)
    draw = draw_pool(support, xi=8.0, rho_bar=0.5, N=6.0, M=200, H=20, rng=7)
    params = PooledParams(
        xi=8.0, N=6.0, gamma=GAMMA, eta=ETA, a=A_I, H=20, phi_v=PHI_V
    )
    eq = solve_pooled_ge(
        draw.alpha, draw.v, params, n_firms_cs=1000.0,
        active_mask=draw.active_mask,
        market_solver_kwargs={"backend": "vectorized"},
    )
    assert eq.converged
    # Net consumption C = Q − M with positive materials use; chi = w/C is on the
    # net-consumption basis. (Walras C = wL+RK+Π is NOT a tight identity in this
    # pooled+population-scaled implementation — income is population-scaled while
    # the composite is not — so it is left as a diagnostic, not asserted.)
    assert eq.aggregates.M > 0.0
    assert eq.aggregates.C > 0.0
    assert eq.aggregates.Q == pytest.approx(eq.aggregates.C + eq.aggregates.M)
    assert eq.chi == pytest.approx(eq.w / eq.aggregates.C)
