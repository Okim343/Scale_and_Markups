"""Negative alpha-v copula strength through the pool draw and PooledParams."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import spearmanr

from orbis_model.model.normalization import PooledParams, solve_pooled_ge
from orbis_model.model.pool import draw_pool


SUPPORT = np.linspace(0.7, 1.26, 50)


def _ladder(draw, xi, v_min=1.0):
    """Recover the standardized ladder from v: u = 1 - (v/v_min)^(-xi)."""
    u = 1.0 - (draw.v / v_min) ** (-xi)
    return ndtri(u)


@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.25])
def test_negative_rho_bar_gives_negative_rank_dependence(rho):
    xi = 6.0
    draw = draw_pool(SUPPORT, xi=xi, rho_bar=rho, N=10.0, M=400, H=50, rng=11)
    ladder = _ladder(draw, xi)
    corr = np.corrcoef(draw.tilde_alpha.ravel(), ladder.ravel())[0, 1]
    assert corr == pytest.approx(rho, abs=0.02)
    # The ladder stays standard normal for a negative rho_bar.
    assert ladder.std() == pytest.approx(1.0, abs=0.02)
    assert spearmanr(draw.alpha.ravel(), draw.v.ravel())[0] < 0.0


def test_sign_flip_mirrors_the_ladder():
    # Same CRNs: ladder(-rho) = -rho*tilde + sqrt(1-rho^2)*eps, so
    # ladder(rho) + ladder(-rho) = 2*sqrt(1-rho^2)*eps.
    xi, rho = 5.0, 0.6
    pos = draw_pool(SUPPORT, xi=xi, rho_bar=rho, N=4.0, M=20, H=10, rng=5)
    neg = draw_pool(SUPPORT, xi=xi, rho_bar=-rho, N=4.0, M=20, H=10, rng=5)
    np.testing.assert_array_equal(pos.alpha, neg.alpha)
    np.testing.assert_array_equal(pos.active_mask, neg.active_mask)
    eps_pos = (_ladder(pos, xi) - rho * pos.tilde_alpha) / np.sqrt(1 - rho**2)
    eps_neg = (_ladder(neg, xi) + rho * neg.tilde_alpha) / np.sqrt(1 - rho**2)
    np.testing.assert_allclose(eps_pos, eps_neg, atol=1e-6)


@pytest.mark.parametrize("bad", [-1.0, 1.0, -1.5])
def test_rho_bar_outside_open_interval_is_rejected(bad):
    with pytest.raises(ValueError, match="rho_bar"):
        draw_pool(SUPPORT, xi=5.0, rho_bar=bad, N=3.0, M=2, H=4, rng=0)
    with pytest.raises(ValueError, match="rho_bar"):
        PooledParams(xi=8.0, N=2.0, gamma=3.8, eta=2.0, rho_bar=bad, H=5)


def test_negative_rho_bar_solves_the_pooled_ge():
    params = PooledParams(xi=8.0, N=6.0, gamma=3.8, eta=2.0, rho_bar=-0.5, a=0.35, H=20, phi_v=0.45)
    draw = draw_pool(SUPPORT, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
                     M=50, H=params.H, rng=7)
    eq = solve_pooled_ge(draw.alpha, draw.v, params, n_firms_cs=1000.0,
                         active_mask=draw.active_mask,
                         market_solver_kwargs={"backend": "vectorized"})
    assert eq.converged
    assert eq.P_agg == pytest.approx(1.0, abs=1e-3)
    assert eq.L_agg == pytest.approx(1.0, abs=1e-3)
