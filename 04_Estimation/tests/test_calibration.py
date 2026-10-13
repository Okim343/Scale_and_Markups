"""Calibration core: moment keys, the joint trf fit, the transform, the residuals.

Ported from ``04_Code/tests/test_calibration.py``. The Compustat-only tests
(sga/cogs markup switch, hybrid-alpha provenance, reference_scale, the
corr_alpha_log_sales residual) are dropped; the RTS-gap residual and the
negative-rho_bar transform are new.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbis_estimation.inner import _decode, _encode, _rho_to_x, _x_to_rho, calibrate_pooled
from orbis_estimation.objective import (
    MOMENT_KEYS,
    elasticities_from_emx,
    pooled_moments_at_params,
    weighted_relative_residuals,
)
from orbis_model.model.normalization import PooledParams
from orbis_model.primitives import ModelPrimitives

SOLVER = {"backend": "vectorized", "tol": 1e-7}


def _primitives(H=6):
    return ModelPrimitives(
        alpha_support=np.linspace(0.7, 1.05, 20), a=0.34, phi_v=0.45,
        n_firms_cs=120.0, H=H, active_window=(2015, 2019),
    )


def test_pooled_moment_keys_are_emx_concentration_slope_and_rts_gap():
    assert MOMENT_KEYS == (
        "mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
        "alpha_rts_gap_top5_bottom50",
    )


def test_emx_line_recovers_planted_gamma_eta():
    gamma, eta = 4.5, 1.8
    intercept = 1.0 - 1.0 / gamma
    slope = -(1.0 / eta - 1.0 / gamma)
    got_gamma, got_eta = elasticities_from_emx(intercept, slope)
    assert got_gamma == pytest.approx(gamma)
    assert got_eta == pytest.approx(eta)


def test_joint_trf_recovers_planted_gamma_eta():
    prim = _primitives(H=12)
    truth = PooledParams(xi=8.0, N=8.0, gamma=4.2, eta=1.9, a=prim.a, H=prim.H, phi_v=prim.phi_v)
    planted, _, planted_eq = pooled_moments_at_params(
        truth, prim, M=12, rng=19, market_solver_kwargs=SOLVER,
    )
    assert not planted_eq.participation.pool_binds.any()
    assert not planted_eq.participation.empty_market.any()
    targets = {k: planted[k] for k in MOMENT_KEYS}
    result = calibrate_pooled(
        prim, targets, M=12, rng=19, initial=(8.0, 8.0, 3.8, 2.1, 0.35),
        xi_bounds=(7.99, 8.01), N_bounds=(7.99, 8.01),
        moment_keys=("mu_cw", "emx_slope"), max_nfev=25,
        market_solver_kwargs=SOLVER,
    )
    assert result.v_min > 0.0
    assert result.equilibrium.y_anchor == pytest.approx(result.equilibrium.y_hat, abs=1e-4)
    assert result.gamma == pytest.approx(truth.gamma, rel=2e-2)
    assert result.eta == pytest.approx(truth.eta, rel=2e-2)
    assert result.reference.P == pytest.approx(1.0, abs=2e-4)
    assert result.reference.L == pytest.approx(1.0, abs=2e-4)
    # EMX entry: no operating cost; the reference carries wf = 0.
    assert result.reference.wf_reference == 0.0
    assert result.eta_jacobian_column_norm > 0.0


def test_rts_gap_residual_is_weighted_and_corr_is_untargeted():
    prim = _primitives(H=20)
    params = PooledParams(xi=8.0, N=8.0, gamma=4.2, eta=1.9, a=prim.a, H=prim.H, phi_v=prim.phi_v)
    planted, _, _ = pooled_moments_at_params(params, prim, M=12, rng=7, market_solver_kwargs=SOLVER)
    targets = {k: planted[k] for k in MOMENT_KEYS}
    targets["alpha_rts_gap_top5_bottom50"] = planted["alpha_rts_gap_top5_bottom50"] + 0.02
    keys = ("mu_cw", "emx_slope", "alpha_rts_gap_top5_bottom50")
    weights = {"mu_cw": 10.0, "emx_slope": 3.162, "alpha_rts_gap_top5_bottom50": 3.162}
    result = calibrate_pooled(
        prim, targets, M=12, rng=7, initial=(8.0, 8.0, 4.2, 1.9, 0.0),
        xi_bounds=(7.99, 8.01), N_bounds=(7.99, 8.01),
        moment_keys=keys, weights=weights, max_nfev=1, market_solver_kwargs=SOLVER,
    )
    assert result.residuals.shape == (3,)
    gap_model = result.model_moments["alpha_rts_gap_top5_bottom50"]
    gap_target = targets["alpha_rts_gap_top5_bottom50"]
    expected = 3.162 * (gap_model - gap_target) / (1.0 + abs(gap_target))
    assert result.residuals[-1] == pytest.approx(expected)
    # The alpha-sales correlations are reported as model diagnostics only.
    assert "corr_alpha_log_sales" in result.model_moments
    assert "corr_alpha_log_sales" not in result.target_moments


def test_weighted_relative_residuals_use_emx_normalizer():
    model = {"mu_cw": 1.3, "emx_slope": -0.4}
    targets = {"mu_cw": 1.2, "emx_slope": -0.3}
    got = weighted_relative_residuals(model, targets, moment_keys=("mu_cw", "emx_slope"),
                                      weights={"mu_cw": 10.0})
    np.testing.assert_allclose(got, [10.0 * 0.1 / 2.2, -0.1 / 1.3])
    with pytest.raises(KeyError, match="cr4"):
        weighted_relative_residuals(model, targets, moment_keys=("cr4",))


def test_decode_encode_roundtrip_5param():
    # The calibrated vector is (xi, N, gamma, eta, rho_bar).
    values = (6.0, 40.0, 4.5, 1.8, 0.37)
    x = _encode(values)
    assert x.shape == (5,)
    params = _decode(x, a=0.34, H=100)
    got = (params.xi, params.N, params.gamma, params.eta, params.rho_bar)
    np.testing.assert_allclose(got, values, rtol=1e-12)


@pytest.mark.parametrize("rho", [-0.9, -0.25, 0.0, 0.6])
def test_rho_transform_round_trips_negative_values(rho):
    bounds = (-0.95, 0.95)
    assert _x_to_rho(_rho_to_x(rho, bounds), bounds) == pytest.approx(rho, abs=1e-12)
    params = _decode(_encode((6.0, 40.0, 4.5, 1.8, rho), rho_bar_bounds=bounds),
                     a=0.34, H=100, rho_bar_bounds=bounds)
    assert params.rho_bar == pytest.approx(rho, abs=1e-12)


def test_rho_transform_stays_inside_bounds():
    bounds = (-0.95, 0.95)
    for x in (-50.0, -3.0, 0.0, 3.0, 50.0):
        rho = _x_to_rho(x, bounds)
        assert bounds[0] <= rho <= bounds[1]
    assert _x_to_rho(0.0, bounds) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="initial rho_bar outside bounds"):
        _encode((6.0, 40.0, 4.5, 1.8, -0.97), rho_bar_bounds=bounds)


def test_negative_rho_bar_evaluates_through_the_objective():
    prim = _primitives(H=12)
    params = PooledParams(xi=8.0, N=8.0, gamma=4.2, eta=1.9, rho_bar=-0.5,
                          a=prim.a, H=prim.H, phi_v=prim.phi_v)
    model, _, eq = pooled_moments_at_params(params, prim, M=12, rng=5, market_solver_kwargs=SOLVER)
    assert eq.y_anchor == pytest.approx(eq.y_hat, abs=1e-4)
    assert np.isfinite(model["alpha_rts_gap_top5_bottom50"])


def test_v_min_anchoring_solves_median_output():
    prim = _primitives(H=12)
    common = dict(xi=8.0, N=8.0, gamma=4.2, eta=1.9, a=prim.a, H=prim.H, phi_v=prim.phi_v)
    _, _, eq = pooled_moments_at_params(
        PooledParams(**common), prim, M=12, rng=5, market_solver_kwargs=SOLVER,
    )
    assert eq.v_min > 0.0
    assert eq.y_anchor == pytest.approx(eq.y_hat, abs=1e-4)


def test_eta_constraint_is_enforced_by_parameterization():
    with pytest.raises(ValueError, match="1 < eta < gamma"):
        PooledParams(xi=8.0, N=2.0, gamma=2.0, eta=2.1, H=4)
