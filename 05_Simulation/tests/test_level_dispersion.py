"""Reference compatibility, economic transforms, and the two-panel accounting."""

import numpy as np
import pytest

from orbis_model.model.normalization import PooledParams
from orbis_model.model.pool import PoolDraw, draw_pool
from orbis_model.counterfactuals.level_dispersion.experiment import (
    DRAW_POLICY, arrangements, compact_draw, compare_reference, decompose, homogenize_v,
    reference_draw, solve_economy,
)


def params():
    return PooledParams(xi=6, N=4, gamma=6, eta=2, v_min=2, H=12,
                        rho_bar=-.3, phi_v=.6)


def test_reference_draw_is_the_truncated_sorting_draw():
    p = params()
    support = np.array([.7, .9, 1.2])
    draw = reference_draw(support, p, M=8, seed=19, cap=1.05)
    sweep = draw_pool(support, xi=p.xi, rho_bar=p.rho_bar, N=p.N, M=8, H=p.H,
                      rng=19, v_min=p.v_min, alpha_max=1.05)
    for key in ("alpha", "v", "tilde_alpha", "active_mask"):
        np.testing.assert_array_equal(getattr(draw, key), getattr(sweep, key))
    # Truncation: the node above the cap is dropped, not stacked at the cap.
    assert set(np.unique(draw.alpha)) == {.7, .9}
    assert DRAW_POLICY == "draw_pool_truncated_v1"


def test_compaction_preserves_active_firms_and_sector_harmonic_means():
    p = params()
    draw = reference_draw(np.linspace(.7, 1.1, 8), p, M=12, seed=3, cap=1.05)
    compact, smaller_params = compact_draw(draw, p)
    assert smaller_params.H <= p.H
    for key in ("alpha", "v", "tilde_alpha"):
        np.testing.assert_array_equal(getattr(draw, key)[draw.active_mask],
                                      getattr(compact, key)[compact.active_mask])
    cases = arrangements(compact)
    for key, result in cases.items():
        np.testing.assert_array_equal(result.active_mask, compact.active_mask)
        for i, mask in enumerate(result.active_mask):
            for variable in ("alpha", "v"):
                np.testing.assert_allclose(np.mean(1 / getattr(result, variable)[i, mask]),
                                           np.mean(1 / getattr(compact, variable)[i, mask]), rtol=1e-12)
        if key in ("hom_v", "hom_both"):
            assert all(np.ptp(result.v[i, mask]) == 0 for i, mask in enumerate(result.active_mask))


def test_capability_transform_preserves_singletons_and_inactive_values():
    a = np.array([[.8, .9, 1.0], [.9, 1.0, 1.1]])
    v = np.array([[2., 9., 10.], [2., 4., 20.]])
    mask = np.array([[True, False, False], [True, True, False]])
    d = PoolDraw(a, v, a.copy(), mask)
    new = homogenize_v(d)
    assert new.v[0, 0] == 2
    np.testing.assert_array_equal(new.v[~mask], v[~mask])
    assert new.v[1, 0] == pytest.approx(8 / 3)
    np.testing.assert_array_equal(new.alpha, a)


def test_percentage_point_interaction_recovers_planted_values():
    rows = {"hom_both": {"Y_pct": 20, "welfare_pct": 10},
            "hom_alpha": {"Y_pct": 24, "welfare_pct": 15},
            "hom_v": {"Y_pct": 30, "welfare_pct": 18},
            "baseline": {"Y_pct": 80, "welfare_pct": 40}}
    split = decompose(rows)
    assert split["Y_pct"]["interaction"] == 46
    assert split["welfare_pct"]["interaction"] == 17
    assert split["welfare_pct"]["identity_residual"] == 0


def test_reference_gate_is_symmetric_and_rejects_a_changed_economy():
    row = {f"{k}_{r}": 2. for r in ("ME", "PE")
           for k in ("Q", "K", "M", "C", "L", "R", "w", "X_market", "W", "mu_cw")}
    row.update(lambda_total=.2, n_active=100, chi=.4)
    close = {**row, "Q_PE": 2.0000001}
    assert compare_reference(row, close)["passed"]
    assert compare_reference(row, close)["checks"]["Q_PE"]["absolute_error"] == compare_reference(close, row)["checks"]["Q_PE"]["absolute_error"]
    bad = {**row, "C_PE": 2.01}
    assert not compare_reference(row, bad)["passed"]
    assert not compare_reference(bad, row)["passed"]


def test_small_four_economy_solves_keep_welfare_and_consumption_consistent():
    p = params()
    draw = reference_draw(np.linspace(.75, .95, 8), p, M=8, seed=7, cap=1.05)
    draw, p = compact_draw(draw, p)
    rows = {}
    for label, d in arrangements(draw).items():
        rows[label] = solve_economy(
            label, d, p, chi=.4, beta=.96, phi=1., delta_K=.06,
            n_firms_cs=100, wf_reference=0, initial=(1, 1), outer_tol=1e-8,
            market_kwargs={"backend": "vectorized", "parallel_n_jobs": 1,
                           "tol": 1e-10, "max_iter": 1500, "damping": .4})
        assert rows[label]["consumption_welfare_gap"] < 1e-6
        assert rows[label]["n_active"] == int(draw.active_mask.sum())
        assert rows[label]["lambda_total"] >= -1e-6
        replay = solve_economy(
            label, d, p, chi=.4, beta=.96, phi=1., delta_K=.06,
            n_firms_cs=100, wf_reference=0, initial=None,
            reference_prices=rows[label],
            market_kwargs={"backend": "vectorized", "parallel_n_jobs": 1,
                           "tol": 1e-10, "max_iter": 1500, "damping": .4})
        assert compare_reference(replay, rows[label], rtol=1e-10, atol=1e-10)["passed"]
    assert abs(decompose(rows)["welfare_pct"]["identity_residual"]) < 1e-10
