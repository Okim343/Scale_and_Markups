from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from steady_state.calibration.inner import _decode, _encode, calibrate_pooled
from steady_state.calibration.objective import (
    MOMENT_KEYS,
    elasticities_from_emx,
    pooled_moments_at_params,
)
from steady_state.model.normalization import PooledParams
from steady_state.pooled_inputs import PooledInputs


def _inputs(H=6):
    return PooledInputs(
        targets={k: 1.0 for k in MOMENT_KEYS}, rho_bar=0.35,
        # Deliberately distinct from rho_bar so a test asserting against this
        # field fails loudly if the residual accidentally targets rho_bar
        # instead (see calibration/inner.py residual()).
        alpha_sales_corr_empirical=0.6, alpha_sales_corr_partial_empirical=0.5, a=0.34,
        eta_init=2.0, H=H, alpha_support=np.linspace(0.7, 1.05, 20),
        n_firms_cs=120.0,
    )


def test_pooled_moment_keys_are_emx_concentration_and_slope():
    assert MOMENT_KEYS == (
        "mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
    )


def test_emx_line_recovers_planted_gamma_eta():
    gamma, eta = 4.5, 1.8
    intercept = 1.0 - 1.0 / gamma
    slope = -(1.0 / eta - 1.0 / gamma)
    got_gamma, got_eta = elasticities_from_emx(intercept, slope)
    assert got_gamma == pytest.approx(gamma)
    assert got_eta == pytest.approx(eta)


def test_joint_trf_recovers_planted_gamma_eta():
    inputs = _inputs(H=12)
    truth = PooledParams(xi=8.0, N=8.0, gamma=4.2, eta=1.9, a=inputs.a, H=inputs.H)
    planted, _, planted_eq = pooled_moments_at_params(
        truth, inputs, M=12, rng=19,
        market_solver_kwargs={"backend": "vectorized", "tol": 1e-7},
    )
    assert not planted_eq.participation.pool_binds.any()
    assert not planted_eq.participation.empty_market.any()
    inputs = replace(inputs, targets={k: planted[k] for k in MOMENT_KEYS})
    result = calibrate_pooled(
        inputs, M=12, rng=19, initial=(8.0, 8.0, 3.8, 2.1, 0.35),
        xi_bounds=(7.99, 8.01), N_bounds=(7.99, 8.01),
        moment_keys=("mu_cw", "emx_slope"), max_nfev=25,
        market_solver_kwargs={"backend": "vectorized", "tol": 1e-7},
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


def test_corr_alpha_log_sales_weight_appends_target_residual():
    inputs = _inputs(H=20)
    params = PooledParams(xi=8.0, N=8.0, gamma=4.2, eta=1.9, a=inputs.a, H=inputs.H)
    planted, _, _ = pooled_moments_at_params(
        params, inputs, M=12, rng=7,
        market_solver_kwargs={"backend": "vectorized", "tol": 1e-7},
    )
    inputs = replace(inputs, targets={k: planted[k] for k in MOMENT_KEYS})
    common = dict(
        inputs=inputs, M=12, rng=7, initial=(8.0, 8.0, 4.2, 1.9, 0.35),
        xi_bounds=(7.99, 8.01), N_bounds=(7.99, 8.01),
        moment_keys=("mu_cw", "emx_slope"), max_nfev=1,
        market_solver_kwargs={"backend": "vectorized", "tol": 1e-7},
    )

    without_corr = calibrate_pooled(**common)
    with_corr = calibrate_pooled(
        **common, corr_alpha_log_sales_weight=3.162,
    )

    assert without_corr.residuals.shape == (2,)
    assert with_corr.residuals.shape == (3,)
    assert "corr_alpha_log_sales" in with_corr.model_moments
    target = inputs.alpha_sales_corr_empirical
    assert with_corr.target_moments["corr_alpha_log_sales"] == pytest.approx(target)
    expected = (
        3.162
        * (with_corr.model_moments["corr_alpha_log_sales"] - target)
        / (1.0 + abs(target))
    )
    assert with_corr.residuals[-1] == pytest.approx(expected)


def test_decode_encode_roundtrip_5param():
    # The calibrated vector is (xi, N, gamma, eta, rho_bar).
    values = (6.0, 40.0, 4.5, 1.8, 0.37)
    x = _encode(values)
    assert x.shape == (5,)
    params = _decode(x, a=0.34, H=100)
    got = (params.xi, params.N, params.gamma, params.eta, params.rho_bar)
    np.testing.assert_allclose(got, values, rtol=1e-12)


def test_v_min_anchoring_solves_sales_weighted_output():
    inputs = _inputs(H=12)
    common = dict(xi=8.0, N=8.0, gamma=4.2, eta=1.9, a=inputs.a, H=inputs.H)
    _, _, eq = pooled_moments_at_params(
        PooledParams(**common), inputs, M=12, rng=5,
        market_solver_kwargs={"backend": "vectorized", "tol": 1e-7},
    )
    assert eq.v_min > 0.0
    assert eq.y_anchor == pytest.approx(eq.y_hat, abs=1e-4)


def test_eta_constraint_is_enforced_by_parameterization():
    with pytest.raises(ValueError, match="1 < eta < gamma"):
        PooledParams(xi=8.0, N=2.0, gamma=2.0, eta=2.1, H=4)


def _bundle_with_markups():
    """Minimal in-memory Bundle carrying both markup bases."""
    from pathlib import Path

    import pandas as pd

    from steady_state.io_bundle import Bundle

    agg = {
        "active_window": {"start": 2010, "end": 2019},
        "targets": {
            "mu_cw": 1.4399, "mu_cw_sga": 1.1848,
            "cr4": 0.20, "cr20": 0.48, "top1pct": 0.32, "top5pct": 0.64,
            "emx_slope": -0.51,
        },
        # Model capital share `a` is the top-level KLEMS a_bar (hybrid-consistent);
        # gnr_elasticities.a_gnr is a diagnostic only (not read by Bundle.a).
        "a": 0.3946, "n_firms_cs": 5211.4,
        "alpha_z_copula_rho_bar": 0.22,
        "alpha_sales_corr_empirical": 0.596,
        "alpha_sales_corr_partial_empirical": 0.570,
        "phi_v": 0.3482,
        "gnr_elasticities": {"role": "diagnostic_raw_gnr", "a_gnr": 0.2993},
        # Support shaping is done empirically (S5); the bundle records the
        # variant/λ used so the loader can carry it as provenance.
        "hybrid_alpha": {"support_variant": "rts_shrunk", "shrink_lambda": 0.42},
        "reference_scale": {
            "type": "pooled_naics2_unweighted_median_reference_share",
            "x_ref": 0.001694214148074163,
            "x_ref_role": "model_reference_input_share",
            "x_ref_raw_input_index": 7233.13852002475,
            "x_ref_raw_role": "diagnostic_input_index_level_only",
            "y_ref": 515100.125,
            "y_ref_role": "diagnostic_output_scale_only",
            "raw_y_ref_over_raw_x_ref": 71.21391683208599,
            "x_ref_variable": "median_firm_x / average_annual_sector_total_x",
            "x_ref_raw_variable": "capital_D^(a_bar*phi_v) * emp^((1-a_bar)*phi_v) * cogs_D^(1-phi_v)",
            "y_ref_variable": "sale_D",
        },
    }
    fa = pd.DataFrame({
        "node_id": [0, 1, 2],
        "u": [1.0 / 6.0, 0.5, 5.0 / 6.0],
        "weight": [1.0 / 3.0] * 3,
        "alpha": [0.85, 0.97, 1.15],
    })
    return Bundle(
        f_alpha=fa, alpha_support=fa["alpha"].to_numpy(),
        aggregate_moments=agg, manifest={}, bundle_dir=Path("."),
    )


def test_markup_target_selects_sga_or_cogs():
    bundle = _bundle_with_markups()
    # Default is the SG&A-inclusive target (model.typ primary target).
    sga = bundle.calibration_targets()
    assert sga["mu_cw"] == pytest.approx(1.1848)          # pure markup target
    assert sga["mu_cw_alpha"] == pytest.approx(1.4399)    # accounting diagnostic
    cogs = bundle.calibration_targets(markup_target="cogs")
    assert cogs["mu_cw"] == pytest.approx(1.4399)
    assert cogs["mu_cw_alpha"] == pytest.approx(1.4399)
    # Every other moment is identical across the two bases.
    for k in ("cr4", "cr20", "top1pct", "top5pct", "emx_slope"):
        assert sga[k] == cogs[k]
    with pytest.raises(ValueError, match="sga.*cogs|markup_target"):
        bundle.calibration_targets(markup_target="xsga")


def test_pooled_inputs_default_markup_target_is_sga():
    bundle = _bundle_with_markups()
    inp = PooledInputs.from_bundle(bundle)
    assert inp.markup_target == "sga"
    assert inp.targets["mu_cw"] == pytest.approx(1.1848)
    assert inp.targets["mu_cw_alpha"] == pytest.approx(1.4399)
    inp_cogs = PooledInputs.from_bundle(bundle, markup_target="cogs")
    assert inp_cogs.targets["mu_cw"] == pytest.approx(1.4399)


def test_pooled_inputs_use_shipped_support_verbatim():
    # Support shaping (winsorize -> shrink -> clip) is done in the empirical
    # pipeline (Stage S5); the loader samples bundle.alpha_support verbatim and
    # only records the shrink variant/λ from the bundle as provenance.
    bundle = _bundle_with_markups()
    inp = PooledInputs.from_bundle(bundle)
    np.testing.assert_allclose(inp.alpha_support, bundle.alpha_support)
    assert inp.alpha_support_variant == "rts_shrunk"
    assert inp.alpha_support_shrink_lambda == pytest.approx(0.42)


def test_pooled_inputs_read_raw_corr_and_ignore_reference_scale_metadata():
    bundle = _bundle_with_markups()
    inputs = PooledInputs.from_bundle(bundle)
    assert inputs.alpha_sales_corr_empirical == pytest.approx(0.596)
    assert not hasattr(inputs, "x_ref")
    assert not hasattr(inputs, "y_ref_empirical")
    assert not hasattr(PooledParams(xi=8.0, N=2.0, gamma=3.8, eta=2.0), "y_ref")


def test_bundle_no_longer_requires_reference_scale():
    bundle = _bundle_with_markups()
    no_ref = dict(bundle.aggregate_moments)
    no_ref.pop("reference_scale")
    assert replace(bundle, aggregate_moments=no_ref).reference_scale == {}

    bad_ref = dict(bundle.aggregate_moments)
    bad_ref["reference_scale"] = {**bad_ref["reference_scale"], "x_ref": 0.0}
    assert replace(bundle, aggregate_moments=bad_ref).reference_scale["x_ref"] == 0.0


def test_y_ref_is_not_used_in_model_equations_or_objective():
    root = Path(__file__).resolve().parents[1] / "steady_state"
    disallowed = [
        root / "model" / "market.py",
        root / "model" / "market_batch.py",
        root / "model" / "normalization.py",
        root / "model" / "participation.py",
        root / "model" / "pricing.py",
        root / "model" / "pool.py",
        root / "calibration" / "objective.py",
        root / "calibration" / "inner.py",
    ]
    for path in disallowed:
        text = path.read_text()
        assert "y_ref" not in text
        assert "diagnostic_output_scale_only" not in text
