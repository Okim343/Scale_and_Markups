"""Unit tests for `steady_state.model.market.solve`.

Tests target the three closed-form / qualitative properties listed in
``solver_scaffold.md`` §8 step 2:

1. Monopoly (n=1): markup = η/(η-1) — under nested CES with one firm in the
   market, the residual elasticity collapses to the across-sector η. (The
   scaffold prose has a typo saying γ/(γ-1); the correct formula from
   ``model.typ`` eq (85) at s=1 is η/(η-1).)
2. Symmetric oligopoly with homogeneous α: all firms have equal shares 1/n
   and the closed-form symmetric markup.
3. Heterogeneous α: shares depend on X_market — the level matters. Doubling
   X_market shifts share away from low-α (steep MC) firms toward high-α firms.
"""

from __future__ import annotations

import numpy as np
import pytest

from steady_state.model.market import solve, _omega_gross
from steady_state.model.pricing import DEFAULT_PHI_V


# Standard EMX-ish parameter choices from the baseline calibration target.
GAMMA = 3.8
ETA = 2.0
W = 1.0
R = 0.10
A_I = 0.35
Y_HAT = 11.0


def test_monopoly_markup_matches_eta_over_eta_minus_one():
    sol = solve(
        n=1,
        alpha=np.array([1.0]),
        v=np.array([1.0]),
        eta=ETA, gamma=GAMMA,
        w=W, R=R,
        X_market=1.0,
        a_i=A_I,
    )
    expected = ETA / (ETA - 1.0)
    assert sol.converged, "monopoly should converge"
    assert sol.s[0] == pytest.approx(1.0, abs=1e-10)
    assert sol.mu[0] == pytest.approx(expected, abs=1e-6), (
        f"monopoly markup {sol.mu[0]} != η/(η-1) = {expected}"
    )


def test_symmetric_oligopoly_homogeneous_alpha_equal_shares():
    n = 4
    sol = solve(
        n=n,
        alpha=np.full(n, 1.0),
        v=np.full(n, 1.0),
        eta=ETA, gamma=GAMMA,
        w=W, R=R,
        X_market=1.0,
        a_i=A_I,
    )
    assert sol.converged
    # All shares equal 1/n.
    np.testing.assert_allclose(sol.s, np.full(n, 1.0 / n), atol=1e-8)
    # Closed-form symmetric markup at s = 1/n:
    inv_mu_expected = 1.0 - 1.0 / GAMMA - (1.0 / ETA - 1.0 / GAMMA) * (1.0 / n)
    mu_expected = 1.0 / inv_mu_expected
    np.testing.assert_allclose(sol.mu, np.full(n, mu_expected), atol=1e-6)


def test_market_clears_x_market():
    """Σ p_j y_j should equal X_market at convergence (level closure)."""
    rng = np.random.default_rng(0)
    n = 5
    alpha = rng.uniform(0.95, 1.05, size=n)
    v = np.exp(rng.normal(0.0, 0.3, size=n))
    for X in [0.5, 1.0, 10.0]:
        sol = solve(
            n=n, alpha=alpha, v=v,
            eta=ETA, gamma=GAMMA, w=W, R=R,
            X_market=X, a_i=A_I,
        )
        assert sol.converged, f"failed to converge at X_market={X}"
        revenue = float((sol.p * sol.y).sum())
        assert revenue == pytest.approx(X, rel=1e-6), (
            f"market revenue {revenue} != X_market {X}"
        )
        # CES identity: P_im * Y_im = X_market.
        assert sol.P_im * sol.Y_im == pytest.approx(X, rel=1e-6)


def test_shares_sum_to_one_and_markups_consistent():
    rng = np.random.default_rng(1)
    n = 6
    alpha = rng.uniform(0.9, 1.1, size=n)
    v = np.exp(rng.normal(0.0, 0.5, size=n))
    sol = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, y_hat=Y_HAT,
    )
    assert sol.converged
    assert sol.s.sum() == pytest.approx(1.0, abs=1e-10)
    # Recompute 1/μ from shares and compare to returned μ.
    inv_mu = 1.0 - 1.0 / GAMMA - (1.0 / ETA - 1.0 / GAMMA) * sol.s
    np.testing.assert_allclose(1.0 / sol.mu, inv_mu, atol=1e-9)
    # Recompute MC from primitives and verify p = μ * MC. The solver now uses
    # the gross-output unit cost Ω^g (model.typ @eq:omega_gross).
    Omega = _omega_gross(W, R, A_I, DEFAULT_PHI_V)
    inv_a = 1.0 / alpha
    mc = (
        (Omega / v)
        * np.exp((inv_a - 1.0) * (np.log(sol.y) - np.log(Y_HAT)))
    )
    np.testing.assert_allclose(sol.p, sol.mu * mc, rtol=1e-8)
    # cost holds true total variable cost TC = α·revenue/μ; profit = (1-α/μ)·rev.
    revenue = sol.p * sol.y
    np.testing.assert_allclose(sol.cost, alpha * revenue / sol.mu, rtol=1e-8)
    np.testing.assert_allclose(sol.d, (1.0 - alpha / sol.mu) * revenue, rtol=1e-8)


def test_heterogeneous_alpha_level_matters():
    """With α heterogeneous, equilibrium shares depend on X_market.

    Intuition: MC_j ∝ y_j^{1/α_j - 1}. A firm with α < 1 has rising MC; doubling
    market size raises its MC faster than a firm with α ≈ 1, eroding its share.
    """
    n = 3
    alpha = np.array([0.7, 1.0, 1.2])
    v = np.array([1.0, 1.0, 1.0])

    sol_small = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=0.1, a_i=A_I,
    )
    sol_large = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=100.0, a_i=A_I,
    )
    assert sol_small.converged and sol_large.converged

    # The low-α firm (j=0) should lose share as the market grows; the high-α
    # firm (j=2) should gain share. Equal-α firms in between (j=1) should sit
    # between the two.
    assert sol_large.s[0] < sol_small.s[0] - 1e-6, (
        f"low-α share did not fall with X_market: "
        f"small={sol_small.s[0]:.6f}, large={sol_large.s[0]:.6f}"
    )
    assert sol_large.s[2] > sol_small.s[2] + 1e-6, (
        f"high-α share did not rise with X_market: "
        f"small={sol_small.s[2]:.6f}, large={sol_large.s[2]:.6f}"
    )


def test_homogeneous_alpha_is_scale_invariant_in_shares():
    """With α homogeneous (and equal to 1), shares should not depend on X_market.

    This is the EMX baseline: CRS makes the share-only FP truly scale-free.
    """
    rng = np.random.default_rng(42)
    n = 4
    v = np.exp(rng.normal(0.0, 0.5, size=n))
    alpha = np.ones(n)

    sol_small = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=0.1, a_i=A_I,
    )
    sol_large = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=100.0, a_i=A_I,
    )
    assert sol_small.converged and sol_large.converged
    np.testing.assert_allclose(sol_small.s, sol_large.s, atol=1e-6)
    np.testing.assert_allclose(sol_small.mu, sol_large.mu, atol=1e-6)


def test_alpha_one_nests_emx_and_is_anchor_invariant():
    rng = np.random.default_rng(101)
    n = 5
    alpha = np.ones(n)
    v = np.exp(rng.normal(0.0, 0.4, size=n))

    low_anchor = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, y_hat=0.25,
    )
    high_anchor = solve(
        n=n, alpha=alpha, v=v,
        eta=ETA, gamma=GAMMA, w=W, R=R,
        X_market=1.0, a_i=A_I, y_hat=25.0,
    )
    assert low_anchor.converged and high_anchor.converged
    np.testing.assert_allclose(low_anchor.s, high_anchor.s, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(low_anchor.mu, high_anchor.mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(low_anchor.y, high_anchor.y, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(low_anchor.p, high_anchor.p, rtol=0.0, atol=0.0)

    omega = _omega_gross(W, R, A_I, DEFAULT_PHI_V)
    np.testing.assert_allclose(low_anchor.p, low_anchor.mu * omega / v, rtol=1e-10)
