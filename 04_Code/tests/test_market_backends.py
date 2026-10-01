"""Backend equivalence and determinism for the batched market solver.

The scalar backend is the correctness oracle (see ``speed_up_code.md`` §4).
The vectorized backend must reproduce it up to solver tolerance under the same
inputs. We therefore compare economic outputs with ``np.allclose`` (cross-
backend float reductions differ in order) and require *exact* equality only for
same-backend determinism.

Per ``speed_up_code.md`` §2.3 / §4.4, ``iterations`` and ``used_newton`` are
*not* asserted equal across backends: a market the scalar solver clears with
its in-loop Newton step is reached in the vectorized path via the scalar
fallback, which can label the path differently while landing on the same
(unique) equilibrium. Economic equality and the ``converged`` flag are the
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from steady_state.model.market import solve
from steady_state.model.market_batch import solve_batch


GAMMA = 3.8
ETA = 2.0
W = 1.0
R = 0.10
A_I = 0.35
X_MARKET = 1.0
Y_HAT = 9.0

ECON_FIELDS = (
    "s", "mu", "sales", "cost", "output", "price", "Y_im", "P_im",
    "Y_j_normalized", "P_j_normalized",
)
ALL_FIELDS = ECON_FIELDS + ("residual", "converged", "used_newton", "iterations")


def test_variety_normalization_is_return_path_only():
    """The 1/n normalization changes aggregate levels, not the micro solve."""
    n = 7
    alpha, v = _draw_markets(1, n, seed=314)
    # damping=0.2 matches the production config (config.yaml solver.market):
    # large nested-CES markets need gentler damping. At the looser default 0.5
    # this stiff n=7 draw needs ~1900 iterations to converge.
    common = dict(
        n=n, alpha=alpha[0], v=v[0], eta=ETA, gamma=GAMMA,
        w=W, R=R, X_market=X_MARKET, a_i=A_I, max_iter=1000, damping=0.2,
    )
    off = solve(**common, love_of_variety="off")
    on = solve(**common, love_of_variety="on")
    assert off.converged and on.converged

    for field in ("s", "mu", "d", "y", "p"):
        np.testing.assert_array_equal(getattr(off, field), getattr(on, field))
    assert off.Y_im == on.Y_im
    assert off.P_im == on.P_im

    factor = n ** (GAMMA / (GAMMA - 1.0))
    assert off.Y_j_normalized == pytest.approx(on.Y_j_normalized / factor)
    assert off.P_j_normalized == pytest.approx(on.P_j_normalized * factor)
    assert off.P_j_normalized * off.Y_j_normalized == pytest.approx(X_MARKET)
    assert on.P_j_normalized * on.Y_j_normalized == pytest.approx(X_MARKET)


def _draw_markets(M: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random but reproducible (alpha, v) stacks of shape (M, n)."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.5, 1.0, size=(M, n))
    v = np.exp(rng.normal(0.0, 0.4, size=(M, n)))
    return alpha, v


def _assert_econ_allclose(a, b, *, atol=1e-6, rtol=1e-6):
    for f in ECON_FIELDS:
        np.testing.assert_allclose(
            getattr(a, f), getattr(b, f), atol=atol, rtol=rtol,
            err_msg=f"backend mismatch in field {f!r}",
        )


@pytest.mark.parametrize("n", [4, 50, 200])
def test_batch_backends_match(n: int):
    """Scalar vs vectorized agree on all economic outputs for several n."""
    M = 30
    alpha, v = _draw_markets(M, n, seed=100 + n)
    common = dict(
        eta=ETA, gamma=GAMMA, w=W, R=R, X_market=X_MARKET, a_i=A_I,
        y_hat=Y_HAT, tol=1e-8, max_iter=200, damping=0.5, newton_fallback_after=50,
    )
    scal = solve_batch(alpha, v, backend="scalar", **common)
    vec = solve_batch(alpha, v, backend="vectorized", **common)

    _assert_econ_allclose(scal, vec)
    np.testing.assert_array_equal(scal.converged, vec.converged)
    # Shares are a valid simplex in both.
    np.testing.assert_allclose(vec.s.sum(axis=1), np.ones(M), atol=1e-8)


def test_batch_vectorized_deterministic():
    """Same backend + same inputs → bit-identical output."""
    alpha, v = _draw_markets(20, 8, seed=7)
    common = dict(
        eta=ETA, gamma=GAMMA, w=W, R=R, X_market=X_MARKET, a_i=A_I, y_hat=Y_HAT
    )
    a = solve_batch(alpha, v, backend="vectorized", **common)
    b = solve_batch(alpha, v, backend="vectorized", **common)
    for f in ECON_FIELDS:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    np.testing.assert_array_equal(a.converged, b.converged)


@pytest.mark.parametrize("backend", ["scalar", "vectorized"])
def test_batch_chunk_parallelism_is_bit_exact(backend: str):
    alpha, v = _draw_markets(11, 9, seed=202)
    mask = np.array([
        [1, 1, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 1],
    ], dtype=bool)
    common = dict(
        eta=ETA, gamma=GAMMA, w=W, R=R, X_market=X_MARKET, a_i=A_I,
        tol=1e-8, max_iter=200, damping=0.5, newton_fallback_after=50,
        active_mask=mask, backend=backend,
    )
    serial = solve_batch(
        alpha, v,
        parallel_n_jobs=1,
        parallel_min_markets=10_000,
        parallel_min_markets_per_job=10_000,
        **common,
    )
    chunked = solve_batch(
        alpha, v,
        parallel_n_jobs=3,
        parallel_min_markets=1,
        parallel_min_markets_per_job=2,
        **common,
    )
    for field in ALL_FIELDS:
        assert np.array_equal(getattr(serial, field), getattr(chunked, field)), field


def test_all_true_mask_is_bit_identical():
    alpha, v = _draw_markets(8, 6, seed=72)
    common = dict(eta=ETA, gamma=GAMMA, w=W, R=R, X_market=X_MARKET, a_i=A_I)
    unmasked = solve_batch(alpha, v, backend="vectorized", **common)
    masked = solve_batch(
        alpha, v, active_mask=np.ones_like(alpha, dtype=bool),
        backend="vectorized", **common,
    )
    for field in ECON_FIELDS:
        np.testing.assert_array_equal(getattr(unmasked, field), getattr(masked, field))
    np.testing.assert_array_equal(unmasked.converged, masked.converged)


def test_partial_mask_scalar_vectorized_parity():
    alpha, v = _draw_markets(5, 7, seed=91)
    mask = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 0],
    ], dtype=bool)
    common = dict(
        eta=ETA, gamma=GAMMA, w=W, R=R, X_market=X_MARKET, a_i=A_I,
        active_mask=mask,
    )
    scalar = solve_batch(alpha, v, backend="scalar", **common)
    vectorized = solve_batch(alpha, v, backend="vectorized", **common)

    _assert_econ_allclose(scalar, vectorized)
    np.testing.assert_array_equal(scalar.converged, vectorized.converged)
    np.testing.assert_array_equal(vectorized.s[~mask], 0.0)
    np.testing.assert_array_equal(vectorized.sales[~mask], 0.0)
    np.testing.assert_allclose(vectorized.s.sum(axis=1), 1.0, atol=1e-8)


def test_unknown_backend_raises():
    alpha, v = _draw_markets(3, 4, seed=1)
    with pytest.raises(ValueError, match="unknown backend"):
        solve_batch(
            alpha, v, backend="turbo",
            eta=ETA, gamma=GAMMA, w=W, R=R, X_market=X_MARKET, a_i=A_I,
        )
