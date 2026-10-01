"""Tests for `steady_state.model.aggregator`.

Covers:
* :func:`concentration_share` — analytical limits and edge cases.
* pooled cost-weighted markup and inverse-markup slope.
"""

from __future__ import annotations

import numpy as np
import pytest

from steady_state.model.aggregator import (
    concentration_share,
    pooled_inverse_markup_regression,
    pooled_moments,
    sector_inverse_markup_regression,
    within_sector_concentration,
)


GAMMA = 3.8
ETA = 2.0
W = 1.0
R = 0.10
A_I = 0.35


# ---------------------------------------------------------------------------
# concentration_share
# ---------------------------------------------------------------------------


def test_concentration_share_uniform_panel():
    """Uniform sales ⇒ top-p% share equals p (asymptotically)."""
    n = 10_000
    sales = np.ones(n)
    assert concentration_share(sales, 0.01) == pytest.approx(0.01, abs=2e-4)
    assert concentration_share(sales, 0.05) == pytest.approx(0.05, abs=2e-4)
    assert concentration_share(sales, 0.5) == pytest.approx(0.5, abs=2e-4)


def test_concentration_share_dominant_firm():
    """One firm holds essentially all sales ⇒ any positive percentile ≈ 1."""
    sales = np.array([1000.0] + [1e-6] * 99)
    assert concentration_share(sales, 0.01) == pytest.approx(1.0, abs=1e-6)


def test_concentration_share_bounds():
    sales = np.array([1.0, 2.0, 3.0, 4.0])
    assert concentration_share(sales, 0.0) == 0.0
    assert concentration_share(sales, 1.0) == 1.0
    # Top 25% = 1 firm of 4 = top firm = 4/(1+2+3+4) = 0.4
    assert concentration_share(sales, 0.25) == pytest.approx(0.4, abs=1e-10)


# ---------------------------------------------------------------------------
# pooled cost-weighted markup and inverse-markup slope
# ---------------------------------------------------------------------------


def test_pooled_cost_weighted_markup_matches_formula():
    from steady_state.model.market_batch import solve_batch
    alpha = np.ones((2, 4))
    z = np.array([[2.0, 1.2, 0.8, 0.4], [1.8, 1.1, 0.7, 0.3]])
    solution = solve_batch(
        alpha, z, eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, backend="vectorized",
    )
    mask = np.ones_like(alpha, dtype=bool)
    moments = pooled_moments(solution, mask, n_firms_cs=100)
    # mu_cw_alpha = aggregate revenue / aggregate variable cost.
    expected = np.sum(solution.sales) / np.sum(solution.cost)
    assert moments["mu_cw_alpha"] == pytest.approx(expected)
    # Under homogeneous α the pure markup (mu_cw) equals the accounting object.
    assert moments["mu_cw"] == pytest.approx(moments["mu_cw_alpha"])


def test_pooled_cost_weighted_markup_heterogeneous_alpha():
    """Under heterogeneous α, mu_cw_alpha = ΣR/ΣVC = Σ λc_i·(μ_i/α_i) ≠ Σ λc_i·μ_i = mu_cw."""
    from steady_state.model.market_batch import solve_batch
    alpha = np.array([[0.7, 0.9, 1.1, 1.3], [0.6, 0.85, 1.0, 1.2]])
    z = np.array([[2.0, 1.2, 0.8, 0.4], [1.8, 1.1, 0.7, 0.3]])
    solution = solve_batch(
        alpha, z, eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, backend="vectorized",
    )
    mask = np.ones_like(alpha, dtype=bool)
    moments = pooled_moments(solution, mask, n_firms_cs=100)
    cost = solution.cost[mask]
    sales = solution.sales[mask]
    mu = solution.mu[mask]
    a = alpha[mask]
    rev_over_vc = np.sum(sales) / np.sum(cost)
    cost_weighted_mu_over_alpha = np.sum(cost * (mu / a)) / np.sum(cost)
    assert moments["mu_cw_alpha"] == pytest.approx(rev_over_vc)
    assert moments["mu_cw_alpha"] == pytest.approx(cost_weighted_mu_over_alpha)
    # The pure cost-weighted markup Σ λc·μ differs here.
    pure_identity = np.sum(cost * mu) / np.sum(cost)
    assert moments["mu_cw"] == pytest.approx(pure_identity)
    assert abs(moments["mu_cw_alpha"] - moments["mu_cw"]) > 1e-6


def test_within_sector_concentration_matches_hand_computation():
    # Two sectors: sector 0 has 4 active firms, sector 1 has 2 (slots 3,4 idle).
    sales = np.array([
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [6.0, 4.0, 0.0, 0.0, 0.0],
    ])
    mask = np.array([
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
    ], dtype=bool)
    conc = within_sector_concentration(sales, mask)
    # Per sector: sector 0 cr4 = (4+3+2+1)/10 = 1.0; sector 1 cr4 = 1.0 (n<4).
    assert conc["cr4"] == pytest.approx(1.0)
    assert conc["cr20"] == pytest.approx(1.0)
    # top1pct: ceil(0.01*n)=1 firm -> sector 0: 4/10=0.4; sector 1: 6/10=0.6.
    # sector-sales weighted: both sectors have sales 10, so this is 0.5.
    assert conc["top1pct"] == pytest.approx(0.5)


def test_sector_inverse_markup_regression_is_within_sector_basis():
    from steady_state.model.market_batch import solve_batch
    rng = np.random.default_rng(0)
    alpha = np.ones((40, 25))
    z = np.exp(rng.normal(0.0, 0.4, (40, 25)))
    mask = np.ones_like(alpha, dtype=bool)
    solution = solve_batch(
        alpha, z, eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=25.0, a_i=A_I, active_mask=mask, backend="vectorized",
    )
    sector = sector_inverse_markup_regression(solution, mask)
    # The sector-level slope must be negative (concentration lowers 1/mu_i, i.e.
    # raises the sector markup) and finite.
    assert sector["slope"] < 0.0
    assert np.isfinite(sector["slope"])
    assert sector["hhi_std"] > 0.0  # finite-sample HHI spread across sectors


def test_inverse_markup_regression_recovers_structural_line():
    from steady_state.model.market_batch import solve_batch
    alpha = np.ones((3, 4))
    z = np.array([[2.0, 1.2, 0.8, 0.4], [1.8, 1.1, 0.7, 0.3], [2.2, 1.0, 0.6, 0.2]])
    solution = solve_batch(
        alpha, z, eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, backend="vectorized",
    )
    result = pooled_inverse_markup_regression(solution, np.ones_like(alpha, dtype=bool))
    assert result["intercept"] == pytest.approx(1.0 - 1.0 / GAMMA)
    assert result["slope"] == pytest.approx(-(1.0 / ETA - 1.0 / GAMMA))
