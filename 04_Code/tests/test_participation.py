"""EMX-style exogenous Poisson entry and masked-kernel economic invariants."""

from __future__ import annotations

import numpy as np
import pytest

from steady_state.model.market_batch import solve_batch
from steady_state.model.participation import allocate_exogenous
from steady_state.model.pool import allocate_poisson


COMMON = dict(eta=2.0, gamma=3.8, w=1.0, R=0.1, X_market=1.0, a=0.35)


def _draw() -> tuple[np.ndarray, np.ndarray]:
    alpha = np.ones((2, 5))
    v = np.array([
        [3.0, 2.0, 1.0, 0.5, 0.2],
        [2.5, 1.5, 0.8, 0.4, 0.1],
    ])
    return alpha, v


def _solve(alpha, v, mask):
    return solve_batch(
        alpha, v, active_mask=mask, a_i=COMMON["a"], backend="vectorized",
        **{k: v for k, v in COMMON.items() if k != "a"},
    )


def test_allocate_poisson_shape_min_one_and_mean():
    mask, counts = allocate_poisson(N=30.0, M=4000, H=120, rng=0)
    assert mask.shape == (4000, 120)
    assert counts.min() >= 1
    # mask activates exactly the first counts[m] slots per sector
    np.testing.assert_array_equal(mask.sum(axis=1), counts)
    assert mask[:, 0].all()  # at least one firm everywhere
    # realized Poisson mean is close to N
    assert abs(counts.mean() - 30.0) < 1.0


def test_allocate_poisson_caps_and_warns_when_pool_binds():
    with pytest.warns(RuntimeWarning, match="pool binds"):
        mask, counts = allocate_poisson(N=50.0, M=200, H=40, rng=1)
    assert counts.max() <= 40
    assert mask.sum(axis=1).max() <= 40


def test_allocate_exogenous_prices_all_firms_no_cutoff():
    alpha, v = _draw()
    mask = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=bool)
    result = allocate_exogenous(alpha, v, mask, **COMMON)
    # no deletion: the active mask is returned verbatim
    np.testing.assert_array_equal(result.active_mask, mask)
    np.testing.assert_array_equal(result.n_active, mask.sum(axis=1))
    # no operating cost in the static cross section
    assert result.wf_reference == 0.0
    # every allocated firm produces with positive sales; inactive slots are zero
    sales = result.solution.sales
    assert np.all(sales[mask] > 0.0)
    assert np.all(sales[~mask] == 0.0)


def test_allocate_exogenous_rejects_empty_sector():
    alpha, v = _draw()
    mask = np.zeros_like(alpha, dtype=bool)
    mask[0, 0] = True  # sector 1 left empty
    with pytest.raises(ValueError, match="at least one active firm"):
        allocate_exogenous(alpha, v, mask, **COMMON)


def test_variable_profit_decreases_when_active_set_expands():
    alpha, v = _draw()
    small = np.zeros_like(alpha, dtype=bool)
    small[:, :2] = True
    large = np.ones_like(alpha, dtype=bool)
    small_solution = _solve(alpha, v, small)
    large_solution = _solve(alpha, v, large)
    d_small = small_solution.sales - small_solution.cost
    d_large = large_solution.sales - large_solution.cost
    assert np.all(d_large[small] < d_small[small])
