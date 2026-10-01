"""Batched market interface for the sector Monte Carlo.

This module isolates the per-market solve behind a single entry point,
:func:`solve_batch`, that takes a stack of ``M`` markets (arrays with a leading
market axis) and returns a :class:`BatchSolution` with the same leading axis.

Rationale (see ``speed_up_code.md`` §1): the scalar :func:`model.market.solve`
remains the permanent correctness oracle. By routing the sector Monte Carlo
through ``solve_batch`` we can later swap in a NumPy-vectorized kernel without
touching the surrounding normalization, aggregation, or calibration logic.

v1 ships only ``backend="scalar"``, which loops over markets calling
``market.solve`` exactly as ``simulate_sector`` used to — no behavior change.
``backend="vectorized"`` is a documented follow-up.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures.process as _cf_process
from dataclasses import dataclass
import multiprocessing
import os
import threading
from typing import Any

import numpy as np

from . import market as _market
from . import pricing
from .market import _omega_gross, _variety_normalization_factor
from .pricing import DEFAULT_PHI_V, PricingRegime


# --- Lightweight profiler (opt-in via SS_PROFILE=1) -----------------------
# Localizes where a single calibration objective eval spends its time:
# the vectorized fixed-point loop vs the serial scalar+Newton fallback, plus
# how many markets fall back and whether solve_batch dispatched serially or to
# the process pool. Counters live at module scope, so they only aggregate
# meaningfully when work runs IN THE MAIN PROCESS (run with parallel_n_jobs=1
# for a complete picture; worker-side counters are not collected back).
import time as _time  # noqa: E402

_PROFILE = os.environ.get("SS_PROFILE", "") not in ("", "0")
_PROFILE_STATS: dict[str, float] = {}


def profile_reset() -> None:
    _PROFILE_STATS.clear()
    _PROFILE_STATS.update({
        "solve_batch_calls": 0,
        "serial_dispatches": 0,
        "parallel_dispatches": 0,
        "t_solve_batch": 0.0,
        "vectorized_calls": 0,
        "markets_total": 0,
        "t_vectorized_loop": 0.0,
        "t_fallback_loop": 0.0,
        "n_fallback_markets": 0,
    })


def profile_report() -> dict[str, float]:
    return dict(_PROFILE_STATS)


profile_reset()


@dataclass(frozen=True)
class BatchSolution:
    """Solved equilibria for ``M`` markets, each with ``n`` firms.

    Per-firm arrays have shape ``(M, n)``; per-market arrays have shape
    ``(M,)``. Mirrors :class:`model.market.MarketSolution` with a leading
    market axis.
    """

    # Per-firm, shape (M, n).
    s: np.ndarray          # within-market sales share (rows sum to 1)
    mu: np.ndarray         # markup
    sales: np.ndarray      # p * y
    cost: np.ndarray       # total variable cost TC = alpha * sales / mu
    output: np.ndarray     # y
    price: np.ndarray      # p

    # Per-market, shape (M,).
    Y_im: np.ndarray
    P_im: np.ndarray
    Y_j_normalized: np.ndarray
    P_j_normalized: np.ndarray
    residual: np.ndarray
    converged: np.ndarray      # bool
    used_newton: np.ndarray    # bool
    iterations: np.ndarray     # int


_BATCH_SOLUTION_FIELDS = tuple(BatchSolution.__dataclass_fields__)
_EXECUTOR_CACHE: dict[int, ProcessPoolExecutor] = {}
_EXECUTOR_LOCK = threading.Lock()
_ORIGINAL_CHECK_SYSTEM_LIMITS = _cf_process._check_system_limits


def _solve_scalar(
    alpha: np.ndarray,
    v: np.ndarray,
    *,
    eta: float,
    gamma: float,
    w: float,
    R: float,
    X_market: float,
    a_i: float,
    phi_v: float = DEFAULT_PHI_V,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    love_of_variety: bool | str = False,
    active_mask: np.ndarray | None = None,
    **solver_kwargs: Any,
) -> BatchSolution:
    """Scalar oracle backend: loop over markets calling :func:`market.solve`.

    ``solver_kwargs`` (e.g. ``tol``, ``max_iter``, ``damping``,
    ``newton_fallback_after``) are forwarded unchanged; any omitted key falls
    back to the ``market.solve`` default, so an empty ``solver_kwargs``
    reproduces the previous ``simulate_sector`` behavior bit-for-bit.
    """
    M, n = alpha.shape
    if active_mask is None:
        active_mask = np.ones((M, n), dtype=bool)
    n_active = active_mask.sum(axis=1)
    variety_factor = np.array([
        _variety_normalization_factor(int(k), gamma, love_of_variety)
        for k in n_active
    ])

    s = np.empty((M, n))
    mu = np.empty((M, n))
    sales = np.empty((M, n))
    cost = np.empty((M, n))
    output = np.empty((M, n))
    price = np.empty((M, n))

    Y_im = np.empty(M)
    P_im = np.empty(M)
    residual = np.empty(M)
    converged = np.empty(M, dtype=bool)
    used_newton = np.empty(M, dtype=bool)
    iterations = np.empty(M, dtype=np.int32)

    for m in range(M):
        mask_m = active_mask[m]
        sol = _market.solve(
            n=int(n_active[m]),
            alpha=alpha[m, mask_m],
            v=v[m, mask_m],
            eta=eta, gamma=gamma,
            w=w, R=R,
            X_market=X_market, a_i=a_i, phi_v=phi_v, y_hat=y_hat,
            regime=regime, mu_bar=mu_bar,
            love_of_variety=love_of_variety,
            **solver_kwargs,
        )
        s[m] = 0.0
        mu[m] = 0.0
        sales[m] = 0.0
        cost[m] = 0.0
        output[m] = 0.0
        price[m] = 0.0
        s[m, mask_m] = sol.s
        mu[m, mask_m] = sol.mu
        sales[m, mask_m] = sol.p * sol.y
        cost[m, mask_m] = sol.cost
        output[m, mask_m] = sol.y
        price[m, mask_m] = sol.p
        Y_im[m] = sol.Y_im
        P_im[m] = sol.P_im
        residual[m] = sol.residual
        converged[m] = sol.converged
        used_newton[m] = sol.used_newton
        iterations[m] = sol.iterations

    return BatchSolution(
        s=s, mu=mu, sales=sales, cost=cost, output=output, price=price,
        Y_im=Y_im, P_im=P_im, residual=residual,
        Y_j_normalized=Y_im / variety_factor,
        P_j_normalized=P_im * variety_factor,
        converged=converged, used_newton=used_newton, iterations=iterations,
    )


# ---------------------------------------------------------------------------
# vectorized backend
# ---------------------------------------------------------------------------


def _prices_from_state_batch(
    s: np.ndarray,        # (M, n)
    Y_im: np.ndarray,     # (M,)
    alpha: np.ndarray,    # (M, n)
    v: np.ndarray,        # (M, n)
    eta: float,
    gamma: float,
    Omega: float,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized mirror of ``market._prices_from_state`` → (y, p), shapes (M, n).

    Same flooring and same markup rule (from clipped shares) as the scalar
    oracle, so the per-element arithmetic is identical.
    """
    if active_mask is None:
        active_mask = np.ones_like(s, dtype=bool)
    s_safe = np.where(active_mask, np.clip(s, 1e-12, 1.0), 1e-12)
    q = s_safe ** (gamma / (gamma - 1.0))
    y = q * Y_im[:, None]
    mu = pricing.markups_from_shares(s_safe, eta, gamma, regime, mu_bar=mu_bar)
    inv_a = 1.0 / alpha
    log_y_hat = np.log(y_hat)
    mc = (Omega / v) * np.exp((inv_a - 1.0) * (np.log(y) - log_y_hat))
    p = mu * mc
    y = np.where(active_mask, y, 0.0)
    p = np.where(active_mask, p, 0.0)
    return y, p


def _update_state_batch(
    s: np.ndarray,
    Y_im: np.ndarray,
    X_market: float,
    alpha: np.ndarray,
    v: np.ndarray,
    eta: float,
    gamma: float,
    Omega: float,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One fixed-point evaluation for a stack of markets → (s_new, Y_im_new)."""
    if active_mask is None:
        active_mask = np.ones_like(s, dtype=bool)
    _y, p = _prices_from_state_batch(
        s, Y_im, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar, active_mask
    )
    p_for_power = np.where(active_mask, p, 1.0)
    p_pow = np.where(active_mask, p_for_power ** (1.0 - gamma), 0.0)
    denom = p_pow.sum(axis=1)
    P_im = denom ** (1.0 / (1.0 - gamma))
    s_new = p_pow / denom[:, None]
    Y_im_new = X_market / P_im
    return s_new, Y_im_new


def _solve_vectorized(
    alpha: np.ndarray,
    v: np.ndarray,
    *,
    eta: float,
    gamma: float,
    w: float,
    R: float,
    X_market: float,
    a_i: float,
    phi_v: float = DEFAULT_PHI_V,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
    damping: float = 0.5,
    newton_fallback_after: int = 50,
    love_of_variety: bool | str = False,
    active_mask: np.ndarray | None = None,
) -> BatchSolution:
    """NumPy-vectorized damped fixed point, with scalar fallback on stragglers.

    Advances the damped FP on all markets at once, mirroring the scalar
    ``market.solve`` trajectory exactly (same init, update, damping, and
    ``res < tol`` test). A market is frozen the iteration its residual drops
    below ``tol``. Any market that does not converge by ``max_iter`` is
    re-solved with the scalar oracle ``market.solve`` (which has the Newton
    fallback) from the *same* fresh initial guess, so the result matches the
    oracle. ``newton_fallback_after`` is accepted only to pass straight through
    to that scalar re-solve.
    """
    M, n = alpha.shape
    if active_mask is None:
        active_mask = np.ones((M, n), dtype=bool)
    n_active = active_mask.sum(axis=1)
    if y_hat <= 0.0:
        raise ValueError(f"y_hat must be positive; got {y_hat}")
    Omega = _omega_gross(w, R, a_i, phi_v)
    variety_factor = np.array([
        _variety_normalization_factor(int(k), gamma, love_of_variety)
        for k in n_active
    ])

    s = active_mask / n_active[:, None]
    Y_im = np.full(M, float(X_market))
    converged = np.zeros(M, dtype=bool)
    used_newton = np.zeros(M, dtype=bool)
    iterations = np.zeros(M, dtype=np.int32)
    residual = np.full(M, np.inf)
    active = np.ones(M, dtype=bool)

    if _PROFILE:
        _PROFILE_STATS["vectorized_calls"] += 1
        _PROFILE_STATS["markets_total"] += M
        _t_fp = _time.perf_counter()

    for it in range(1, max_iter + 1):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        s_a = s[idx]
        Y_a = Y_im[idx]
        s_new, Y_new = _update_state_batch(
            s_a, Y_a, X_market, alpha[idx], v[idx], eta, gamma, Omega,
            y_hat, regime, mu_bar, active_mask[idx]
        )
        res = np.max(np.abs(s_new - s_a), axis=1) + np.abs(
            np.log(Y_new) - np.log(Y_a)
        )

        conv = res < tol
        ci = idx[conv]
        if ci.size:
            # Accept the undamped new state, exactly like the scalar solver.
            s[ci] = s_new[conv]
            Y_im[ci] = Y_new[conv]
            converged[ci] = True
            iterations[ci] = it
            residual[ci] = res[conv]
            active[ci] = False

        nconv = ~conv
        ni = idx[nconv]
        if ni.size:
            s_d = damping * s_new[nconv] + (1.0 - damping) * s_a[nconv]
            s_d = np.clip(s_d, 1e-12, 1.0 - 1e-12)
            s_d = np.where(active_mask[ni], s_d, 0.0)
            s_d = s_d / s_d.sum(axis=1, keepdims=True)
            log_Y = damping * np.log(Y_new[nconv]) + (1.0 - damping) * np.log(Y_a[nconv])
            s[ni] = s_d
            Y_im[ni] = np.exp(log_Y)
            residual[ni] = res[nconv]
            iterations[ni] = it

    if _PROFILE:
        _PROFILE_STATS["t_vectorized_loop"] += _time.perf_counter() - _t_fp

    # Final outputs for converged markets (recompute at the frozen state so the
    # returned vectors match the reported s, Y_im — same as the scalar solver).
    out_s = s.copy()
    out_mu = np.where(
        active_mask,
        pricing.markups_from_shares(s, eta, gamma, regime, mu_bar=mu_bar),
        0.0,
    )
    out_y, out_p = _prices_from_state_batch(
        s, Y_im, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar, active_mask
    )
    p_for_power = np.where(active_mask, out_p, 1.0)
    p_pow = np.where(active_mask, p_for_power ** (1.0 - gamma), 0.0)
    out_P_im = p_pow.sum(axis=1) ** (1.0 / (1.0 - gamma))
    out_Y_im = Y_im.copy()
    out_sales = out_p * out_y
    # Total variable cost TC = α·MC·y = α·revenue/μ (non-CRS gross-output cost).
    out_cost = alpha * np.divide(
        out_sales, out_mu, out=np.zeros_like(out_sales), where=active_mask
    )

    # Scalar fallback for any market that did not converge by max_iter. Solve
    # fresh (no warm start) so the result is identical to the oracle path.
    fallback_idx = np.nonzero(~converged)[0]
    if _PROFILE:
        _PROFILE_STATS["n_fallback_markets"] += int(fallback_idx.size)
        _t_fb = _time.perf_counter()
    for m in fallback_idx:
        mask_m = active_mask[m]
        sol = _market.solve(
            n=int(n_active[m]), alpha=alpha[m, mask_m], v=v[m, mask_m],
            eta=eta, gamma=gamma, w=w, R=R,
            X_market=X_market, a_i=a_i, phi_v=phi_v, y_hat=y_hat,
            regime=regime, mu_bar=mu_bar,
            love_of_variety=love_of_variety,
            tol=tol, max_iter=max_iter,
            damping=damping, newton_fallback_after=newton_fallback_after,
        )
        out_s[m] = 0.0
        out_mu[m] = 0.0
        out_sales[m] = 0.0
        out_cost[m] = 0.0
        out_y[m] = 0.0
        out_p[m] = 0.0
        out_s[m, mask_m] = sol.s
        out_mu[m, mask_m] = sol.mu
        out_sales[m, mask_m] = sol.p * sol.y
        out_cost[m, mask_m] = sol.cost
        out_y[m, mask_m] = sol.y
        out_p[m, mask_m] = sol.p
        out_Y_im[m] = sol.Y_im
        out_P_im[m] = sol.P_im
        residual[m] = sol.residual
        converged[m] = sol.converged
        used_newton[m] = sol.used_newton
        iterations[m] = sol.iterations

    if _PROFILE and fallback_idx.size:
        _PROFILE_STATS["t_fallback_loop"] += _time.perf_counter() - _t_fb

    return BatchSolution(
        s=out_s, mu=out_mu, sales=out_sales, cost=out_cost,
        output=out_y, price=out_p,
        Y_im=out_Y_im, P_im=out_P_im, residual=residual,
        Y_j_normalized=out_Y_im / variety_factor,
        P_j_normalized=out_P_im * variety_factor,
        converged=converged, used_newton=used_newton, iterations=iterations,
    )


def _shutdown_executors() -> None:
    with _EXECUTOR_LOCK:
        executors = tuple(_EXECUTOR_CACHE.values())
        _EXECUTOR_CACHE.clear()
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_executors)


def _safe_check_system_limits() -> None:
    try:
        _ORIGINAL_CHECK_SYSTEM_LIMITS()
    except PermissionError:
        # Some managed runtimes deny the sysconf semaphore probe even though
        # process pools themselves still work. Skip that probe in that case.
        return None


_cf_process._check_system_limits = _safe_check_system_limits


def _normalize_parallel_n_jobs(n_jobs: int | None) -> int:
    if n_jobs in (None, 0):
        return 1
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs < -1:
        return max(1, (os.cpu_count() or 1) + 1 + int(n_jobs))
    return max(1, int(n_jobs))


def _market_chunk_bounds(
    M: int,
    *,
    n_jobs: int | None,
    min_markets: int,
    min_markets_per_job: int,
) -> tuple[tuple[int, int], ...]:
    """Return contiguous row chunks; empty tuple means "stay serial"."""
    if M < max(1, int(min_markets)):
        return ()
    jobs = min(_normalize_parallel_n_jobs(n_jobs), M)
    if jobs <= 1:
        return ()
    min_chunk = max(1, int(min_markets_per_job))
    jobs = min(jobs, M // min_chunk)
    if jobs <= 1:
        return ()
    bounds = np.linspace(0, M, num=jobs + 1, dtype=np.int64)
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:])
        if stop > start
    )


def _concat_batch_solutions(chunks: list[BatchSolution]) -> BatchSolution:
    data = {
        field_name: np.concatenate(
            [getattr(chunk, field_name) for chunk in chunks],
            axis=0,
        )
        for field_name in _BATCH_SOLUTION_FIELDS
    }
    return BatchSolution(**data)


def _solve_batch_chunk(
    alpha: np.ndarray,
    v: np.ndarray,
    *,
    eta: float,
    gamma: float,
    w: float,
    R: float,
    X_market: float,
    a_i: float,
    phi_v: float,
    y_hat: float,
    regime: PricingRegime,
    mu_bar: float | None,
    love_of_variety: bool | str,
    active_mask: np.ndarray,
    backend: str,
    solver_kwargs: dict[str, Any],
) -> BatchSolution:
    if backend == "scalar":
        return _solve_scalar(
            alpha, v,
            eta=eta, gamma=gamma, w=w, R=R, X_market=X_market, a_i=a_i,
            phi_v=phi_v, y_hat=y_hat,
            regime=regime, mu_bar=mu_bar, love_of_variety=love_of_variety,
            active_mask=active_mask,
            **solver_kwargs,
        )
    return _solve_vectorized(
        alpha, v,
        eta=eta, gamma=gamma, w=w, R=R, X_market=X_market, a_i=a_i,
        phi_v=phi_v, y_hat=y_hat,
        regime=regime, mu_bar=mu_bar, love_of_variety=love_of_variety,
        active_mask=active_mask,
        **solver_kwargs,
    )


def _executor_for_workers(max_workers: int) -> ProcessPoolExecutor:
    with _EXECUTOR_LOCK:
        executor = _EXECUTOR_CACHE.get(max_workers)
        if executor is None:
            executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            _EXECUTOR_CACHE[max_workers] = executor
        return executor


def _solve_batch_chunk_from_payload(payload: dict[str, Any]) -> BatchSolution:
    return _solve_batch_chunk(**payload)


def solve_batch(
    alpha: np.ndarray,
    v: np.ndarray,
    *,
    eta: float,
    gamma: float,
    w: float,
    R: float,
    X_market: float,
    a_i: float,
    phi_v: float = DEFAULT_PHI_V,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    love_of_variety: bool | str = False,
    active_mask: np.ndarray | None = None,
    backend: str = "scalar",
    parallel_n_jobs: int | None = 1,
    parallel_min_markets: int = 0,
    parallel_min_markets_per_job: int = 1,
    **solver_kwargs: Any,
) -> BatchSolution:
    """Solve ``M`` markets sharing ``(w, R, X_market, a_i, eta, gamma)``.

    Parameters
    ----------
    alpha, v
        Firm primitives, shape ``(M, n)``. Row ``m`` is one market.
    eta, gamma, w, R, X_market, a_i
        Scalars common to all ``M`` markets (the sector solves every market at
        the same expenditure and factor prices).
    y_hat
        Common output anchor. Production calibration normalizes this to one,
        but it is threaded for identity tests and diagnostics.
    backend
        ``"scalar"`` (default) loops over markets calling the oracle
        :func:`market.solve`. ``"vectorized"`` advances a NumPy-vectorized
        damped fixed point over all markets at once (bit-exact on real data,
        ~2.4x faster), falling back to the scalar oracle on stragglers.
    parallel_n_jobs, parallel_min_markets, parallel_min_markets_per_job
        Optional process-level row chunking across markets. When enabled,
        ``solve_batch`` splits the leading ``M`` axis into contiguous chunks,
        solves each chunk independently, and concatenates results back in the
        original row order. Small jobs stay serial.
    love_of_variety
        Post-solve variety toggle forwarded to both backends. It changes only
        ``Y_j_normalized`` and ``P_j_normalized``.
    solver_kwargs
        Forwarded to :func:`market.solve` (``tol``, ``max_iter``, ``damping``,
        ``newton_fallback_after``).
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if alpha.ndim != 2 or v.shape != alpha.shape:
        raise ValueError(
            f"alpha and v must be 2-D with matching shape (M, n); "
            f"got {alpha.shape}, {v.shape}"
        )
    if active_mask is None:
        active_mask = np.ones(alpha.shape, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape != alpha.shape:
            raise ValueError(
                f"active_mask must have shape {alpha.shape}; got {active_mask.shape}"
            )
    if np.any(active_mask.sum(axis=1) == 0):
        raise ValueError("each market must have at least one active firm")
    if backend not in {"scalar", "vectorized"}:
        raise ValueError(f"unknown backend {backend!r}; expected 'scalar' or 'vectorized'")

    _t_sb = _time.perf_counter() if _PROFILE else 0.0
    chunk_bounds = _market_chunk_bounds(
        alpha.shape[0],
        n_jobs=parallel_n_jobs,
        min_markets=parallel_min_markets,
        min_markets_per_job=parallel_min_markets_per_job,
    )
    if not chunk_bounds:
        result = _solve_batch_chunk(
            alpha, v,
            eta=eta, gamma=gamma, w=w, R=R, X_market=X_market, a_i=a_i,
            phi_v=phi_v, y_hat=y_hat,
            regime=regime, mu_bar=mu_bar, love_of_variety=love_of_variety,
            active_mask=active_mask, backend=backend,
            solver_kwargs=dict(solver_kwargs),
        )
        if _PROFILE:
            _PROFILE_STATS["solve_batch_calls"] += 1
            _PROFILE_STATS["serial_dispatches"] += 1
            _PROFILE_STATS["t_solve_batch"] += _time.perf_counter() - _t_sb
        return result
    payloads = [
        {
            "alpha": alpha[start:stop],
            "v": v[start:stop],
            "eta": eta,
            "gamma": gamma,
            "w": w,
            "R": R,
            "X_market": X_market,
            "a_i": a_i,
            "phi_v": phi_v,
            "y_hat": y_hat,
            "regime": regime,
            "mu_bar": mu_bar,
            "love_of_variety": love_of_variety,
            "active_mask": active_mask[start:stop],
            "backend": backend,
            "solver_kwargs": dict(solver_kwargs),
        }
        for start, stop in chunk_bounds
    ]
    executor = _executor_for_workers(len(chunk_bounds))
    chunked = list(executor.map(_solve_batch_chunk_from_payload, payloads))
    result = _concat_batch_solutions(chunked)
    if _PROFILE:
        _PROFILE_STATS["solve_batch_calls"] += 1
        _PROFILE_STATS["parallel_dispatches"] += 1
        _PROFILE_STATS["t_solve_batch"] += _time.perf_counter() - _t_sb
    return result
