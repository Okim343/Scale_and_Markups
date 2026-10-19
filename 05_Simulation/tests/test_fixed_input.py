"""Fixed-input planner (orbis_model.welfare.fixed_input) and the P_mat threading.

* With the default ``P_mat = 1`` the batch solve and the aggregator are
  bit-identical to calls that do not pass it, for both backends.
* The scalar and vectorized backends agree at ``P_mat != 1``.
* Self-consistency: pinning K and M at the free planner's levels returns the
  free planner (``R = R_euler``, ``P_mat = 1``).
* PE_I hits both targets, the three-leg identity closes, and ``C = Q - M - I``
  holds in every regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbis_model.model.aggregator import aggregate_pooled_markets
from orbis_model.model.market_batch import _BATCH_SOLUTION_FIELDS, solve_batch
from orbis_model.model.normalization import PooledParams, euler_R, solve_pooled_ge
from orbis_model.model.pool import draw_pool
from orbis_model.model.pricing import PricingRegime
from orbis_model.welfare.fixed_input import (
    fixed_input_decomposition,
    fixed_input_table,
    solve_fixed_input_chain,
    solve_fixed_input_planner,
)


GAMMA = 3.8
ETA = 2.0
A_I = 0.35
PHI_V = 0.45
BETA = 0.96
DELTA_K = 0.06
N_FIRMS_CS = 1000.0
SOLVER = {"backend": "vectorized", "tol": 1e-10, "max_iter": 2000}


def _draw(M: int = 60, H: int = 20, seed: int = 7):
    support = np.linspace(0.7, 1.2, 40)
    draw = draw_pool(support, xi=8.0, rho_bar=0.5, N=6.0, M=M, H=H, rng=seed)
    params = PooledParams(xi=8.0, N=6.0, gamma=GAMMA, eta=ETA, a=A_I, H=H, phi_v=PHI_V)
    return draw, params


def _batch_kwargs(draw):
    return dict(eta=ETA, gamma=GAMMA, w=1.1, R=0.12, X_market=0.9, a_i=A_I,
                phi_v=PHI_V, active_mask=draw.active_mask)


# ---------------------------------------------------------------------------
# P_mat threading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["scalar", "vectorized"])
def test_default_p_mat_is_bit_identical(backend):
    draw, _ = _draw(M=12)
    base = solve_batch(draw.alpha, draw.v, backend=backend, **_batch_kwargs(draw))
    explicit = solve_batch(draw.alpha, draw.v, backend=backend, P_mat=1.0,
                           **_batch_kwargs(draw))
    for name in _BATCH_SOLUTION_FIELDS:
        np.testing.assert_array_equal(getattr(base, name), getattr(explicit, name), name)
    agg_kwargs = dict(eta=ETA, operating_cost_labor=0.0, n_firms_cs=N_FIRMS_CS,
                      w=1.1, R=0.12, a=A_I, delta_K=DELTA_K, phi_v=PHI_V)
    a0 = aggregate_pooled_markets(base, draw.active_mask, **agg_kwargs)
    a1 = aggregate_pooled_markets(explicit, draw.active_mask, P_mat=1.0, **agg_kwargs)
    for name in ("C", "P", "L", "K", "profits", "Q", "M", "I"):
        assert getattr(a0, name) == getattr(a1, name), name


def test_backends_agree_at_shadow_materials_price():
    draw, _ = _draw(M=12)
    kwargs = dict(**_batch_kwargs(draw), P_mat=1.3, tol=1e-12, max_iter=5000)
    scalar = solve_batch(draw.alpha, draw.v, backend="scalar", **kwargs)
    vector = solve_batch(draw.alpha, draw.v, backend="vectorized", **kwargs)
    for name in ("s", "mu", "sales", "cost", "output", "price",
                 "Y_j_normalized", "P_j_normalized"):
        np.testing.assert_allclose(getattr(scalar, name), getattr(vector, name),
                                   rtol=1e-8, atol=1e-12, err_msg=name)
    # P_mat actually moves the solve, and divides the materials quantity.
    base = solve_batch(draw.alpha, draw.v, backend="vectorized",
                       **{**kwargs, "P_mat": 1.0})
    assert not np.allclose(base.price, vector.price)
    agg_kwargs = dict(eta=ETA, operating_cost_labor=0.0, n_firms_cs=N_FIRMS_CS,
                      w=1.1, R=0.12, a=A_I, delta_K=DELTA_K, phi_v=PHI_V)
    agg = aggregate_pooled_markets(vector, draw.active_mask, P_mat=1.3, **agg_kwargs)
    spend = (1.0 - PHI_V) * vector.cost[draw.active_mask].sum() * agg.population_scale
    assert agg.M == pytest.approx(spend / 1.3, rel=1e-12)
    assert agg.Q == pytest.approx(agg.C + agg.M + agg.I, rel=1e-12)


# ---------------------------------------------------------------------------
# planner solves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain():
    draw, params = _draw()
    out = solve_fixed_input_chain(
        draw.alpha, draw.v, params, beta=BETA, delta_K=DELTA_K,
        n_firms_cs=N_FIRMS_CS, active_mask=draw.active_mask,
        market_solver_kwargs=SOLVER, tol=1e-9, max_nfev=60,
    )
    return draw, params, out


def test_pinning_free_planner_levels_returns_free_planner(chain):
    draw, params, out = chain
    me, pe = out.me, out.pe
    R_euler = euler_R(BETA, DELTA_K)
    common = dict(
        delta_K=DELTA_K, beta=BETA, n_firms_cs=N_FIRMS_CS,
        market_solver_kwargs=SOLVER, regime=PricingRegime.PLANNER,
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask, tol=1e-10, max_nfev=80,
    )
    # Tight free planner to compare against (the chain's PE uses tol 1e-5).
    pe = solve_pooled_ge(
        draw.alpha, draw.v, params, beta=BETA, delta_K=DELTA_K,
        n_firms_cs=N_FIRMS_CS, market_solver_kwargs=SOLVER,
        regime=PricingRegime.PLANNER, wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask, initial=(me.w, me.X_market),
        tol=1e-12, max_nfev=60,
    )
    fixed = solve_fixed_input_planner(
        draw.alpha, draw.v, params, K_target=pe.K_agg, M_target=pe.aggregates.M,
        initial=(me.w, me.X_market, R_euler, 1.0), **common,
    )
    assert fixed.converged
    assert fixed.R == pytest.approx(R_euler, abs=1e-5)
    assert fixed.P_mat == pytest.approx(1.0, abs=1e-5)
    assert fixed.C == pytest.approx(pe.C, rel=1e-6)
    fixed_k = solve_fixed_input_planner(
        draw.alpha, draw.v, params, K_target=pe.K_agg,
        initial=(me.w, me.X_market, R_euler), **common,
    )
    assert fixed_k.converged
    assert fixed_k.P_mat == 1.0
    assert fixed_k.R == pytest.approx(R_euler, abs=1e-5)
    assert fixed_k.C == pytest.approx(pe.C, rel=1e-6)
    assert fixed_k.w == pytest.approx(pe.w, rel=1e-5)


def test_fixed_input_planner_hits_targets_and_identity_closes(chain):
    _, _, out = chain
    me, pe_i, pe_k, pe = out.me, out.pe_i, out.pe_k, out.pe
    assert all(eq.converged for eq in (me, pe_i, pe_k, pe))
    for eq in (pe_i, pe_k):
        assert eq.K_agg / me.K_agg == pytest.approx(1.0, abs=1e-8)
        assert eq.P_agg == pytest.approx(1.0, abs=1e-8)
        assert eq.L_agg == pytest.approx(1.0, abs=1e-8)
    assert pe_i.aggregates.M / me.aggregates.M == pytest.approx(1.0, abs=1e-8)
    # Capital is choked below the free planner's level by a rental above Euler.
    assert pe_k.R > out.R_euler
    assert pe_i.P_mat != pytest.approx(1.0, abs=1e-6)
    for eq in (me, pe_i, pe_k, pe):
        agg = eq.aggregates
        assert agg.C == pytest.approx(agg.Q - agg.M - DELTA_K * agg.K, rel=1e-12)
        assert agg.I == pytest.approx(DELTA_K * agg.K, rel=1e-12)
    # PE_I and PE_K hold K at K_ME, so their investment equals the market's.
    assert pe_i.aggregates.I == pytest.approx(me.aggregates.I, rel=1e-8)
    assert pe_k.aggregates.I == pytest.approx(me.aggregates.I, rel=1e-8)

    decomp = fixed_input_decomposition(me, pe_i, pe_k, pe, chi_me=me.chi, beta=BETA, phi=1.0)
    assert abs(decomp["identity_residual"]) < 1e-10
    assert abs(decomp["Delta_identity_residual"]) < 1e-10
    product = ((1.0 + decomp["lambda_realloc"]) * (1.0 + decomp["lambda_materials"])
               * (1.0 + decomp["lambda_capital"]))
    assert product == pytest.approx(1.0 + decomp["lambda_total"], abs=1e-10)
    assert decomp["Delta_realloc"] == pytest.approx(np.log1p(decomp["lambda_realloc"]))
    assert decomp["lambda_realloc"] > 0.0

    table = fixed_input_table(me, pe_i, pe_k, pe, n_firms_cs=N_FIRMS_CS, R_euler=out.R_euler)
    assert list(table["regime"]) == ["market", "planner_fixed_input",
                                     "planner_fixed_k", "planner"]
    for column in ("C", "Q", "M", "I", "L", "K", "R", "P_mat", "profits",
                   "mu_cw", "mu_cw_alpha", "K_target_gap", "M_target_gap"):
        assert column in table.columns, column
    np.testing.assert_allclose(table["C"], table["Q"] - table["M"] - table["I"], rtol=1e-12)


def test_skipping_regimes_names_the_merged_leg(chain):
    _, _, out = chain
    decomp = fixed_input_decomposition(out.me, out.pe_i, None, out.pe,
                                       chi_me=out.me.chi, beta=BETA, phi=1.0)
    assert "lambda_materials_capital" in decomp
    assert "lambda_materials" not in decomp
    assert abs(decomp["identity_residual"]) < 1e-10
    only = fixed_input_decomposition(out.me, out.pe_i, chi_me=out.me.chi, beta=BETA, phi=1.0)
    assert "lambda_total" not in only and "lambda_realloc" in only


def test_walras_rebates_shadow_materials_rent(chain):
    _, _, out = chain
    me, pe_i = out.me, out.pe_i
    # With P_mat = 1 the rebate is exactly zero and the check is the market one.
    assert me.walras_check()["rent_materials"] == 0.0
    check = pe_i.walras_check(out.R_euler)
    assert check["rent_materials"] == pytest.approx(
        (pe_i.P_mat - 1.0) * pe_i.aggregates.M)
    assert check["rent_capital"] == pytest.approx((pe_i.R - out.R_euler) * pe_i.K_agg)
    # Income minus C + I is exactly the expenditure-aggregation gap (sales - Q)
    # in every regime once the materials rebate is included.
    for eq in (me, pe_i, out.pe_k, out.pe):
        c = eq.walras_check(out.R_euler)
        assert abs(c["closure_residual"]) < 1e-12 * c["income"]
