"""
Core GNR math, translated from the MATLAB research code in
03_Empirical/00_Hubmer_RTS/00_original_code (est_gnr.m, nls_obj.m,
nls_const_filter.m, step2_obj.m, gensobol.m).

Conventions (all inputs in logs):
  Step 1 polynomial (10 terms, MATLAB column order preserved):
      P1 = [1, k, m, l, k^2, m^2, l^2, k*m, k*l, m*l]
  Step 2 integration-constant polynomial (5 terms):
      P2 = [k, l, k^2, l^2, k*l]
  Step 2 instruments:
      IV = [k, k^2, ll, ll^2, k*ll]      (ll = lagged log labor)

Deliberate divergences from the MATLAB script (documented in README.md):
  1. Output elasticities of capital/labor are computed via the analytic
     derivative of the integrated share polynomial directly, instead of
     the MATLAB trick of multiplying selected P1/P2 columns and dividing
     by k (or l). Algebraically identical, but well-defined when log
     capital or log labor is near zero.
  2. The Step 1 positivity constraint (P1 @ gamma > 0) is enforced through
     a large finite penalty rather than MATLAB's NaN objective, so scipy
     line searches backtrack instead of failing.
  3. Wages are optional: the lagged-wage non-missing filter that the MATLAB
     script applies before Step 2 is only imposed when require_wages=true.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

P1SIZE = 10  # Stage 1 elasticity polynomial terms
P2SIZE = 5   # Stage 2 constant-of-integration polynomial terms

# Integration of the share polynomial with respect to log materials m:
# term-by-term factor so that int_melast = (P1 @ (gamma * INTVEC)) * m.
# MATLAB: intvec = [1, 1, 1/2, 1, 1, 1/3, 1, 1/2, 1, 1/2]
INTVEC = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0 / 3.0, 1.0, 0.5, 1.0, 0.5])

# -----------------------------------------------------------------------------
# Polynomial builders
# -----------------------------------------------------------------------------
def build_p1(k: np.ndarray, m: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Step 1 design matrix [1, k, m, l, k^2, m^2, l^2, k*m, k*l, m*l]."""
    one = np.ones_like(k)
    return np.column_stack(
        [one, k, m, l, k**2, m**2, l**2, k * m, k * l, m * l]
    )


def build_p2(k: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Step 2 integration-constant design matrix [k, l, k^2, l^2, k*l]."""
    return np.column_stack([k, l, k**2, l**2, k * l])


def build_iv(k: np.ndarray, ll: np.ndarray) -> np.ndarray:
    """Step 2 instruments [k, k^2, ll, ll^2, k*ll] (capital for itself,
    lagged labor for labor)."""
    return np.column_stack([k, k**2, ll, ll**2, k * ll])


# -----------------------------------------------------------------------------
# Sobol starting values (gensobol.m analog)
# -----------------------------------------------------------------------------
def gensobol(
    n_obs: int, n_params: int, lb: float, ub: float, seed: int
) -> np.ndarray:
    """Scrambled Sobol sequence scaled to [lb, ub]^n_params.

    MATLAB used sobolset(...,'Skip',1000,'Leap',100) with MatousekAffineOwen
    scrambling; scipy's scrambled Sobol with a fixed seed is the equivalent
    quasi-random pool (exact point values differ across platforms anyway).
    """
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    pts = sampler.random(n_obs)
    return pts * (ub - lb) + lb


def step1_sobol_pool(cfg_s1: dict) -> np.ndarray:
    """Step 1 start pool with the MATLAB per-block bounds:
    intercept in [-2,2], 6 mid terms in [-0.5,0.5], 3 tail terms in [-0.1,0.1].
    """
    n = int(cfg_s1["n_sobol"])
    seed = int(cfg_s1["seed"])
    b = cfg_s1["sobol_bounds"]
    block1 = gensobol(n, 1, *map(float, b["intercept"]), seed=seed)
    block2 = gensobol(n, P1SIZE - 4, *map(float, b["mid"]), seed=seed + 1)
    block3 = gensobol(n, 3, *map(float, b["tail"]), seed=seed + 2)
    return np.hstack([block1, block2, block3])


# -----------------------------------------------------------------------------
# Step 1: NLS share regression (nls_obj.m, 'leastsquares_sum')
# -----------------------------------------------------------------------------
def nls_const_filter(paramset: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Keep parameter vectors satisfying min(P1 @ gamma) > 0 (nls_const_filter.m)."""
    c = (p1 @ paramset.T).min(axis=0)
    return paramset[c > 0, :]


_LOG_FLOOR = 1.0e-6  # below this, log() is replaced by its tangent extension


def step1_objective(
    gamma: np.ndarray, s: np.ndarray, p1: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean squared residual of log(P1 @ gamma) - s, with analytic gradient.

    Feasibility handling (divergence #2 from MATLAB, which returned NaN):
    for observations with P1 @ gamma below a small floor, log(x) is replaced
    by its first-order tangent extension at the floor,
        log(floor) + (x - floor) / floor,
    making the objective C^1 everywhere. The extension is steeply increasing
    in feasibility violations, so gradient-based line searches are pulled
    back into the feasible region instead of aborting against a flat penalty
    wall (a hard 1e10 wall made L-BFGS-B fail with ABNORMAL line-search
    terminations on the full sample). Final solutions are still screened
    with a strict P1 @ gamma > 0 check in the solver wrapper.
    """
    x = p1 @ gamma
    below = x < _LOG_FLOOR
    x_safe = np.maximum(x, _LOG_FLOOR)
    logx = np.log(x_safe)
    if below.any():
        logx = logx + np.where(below, (x - _LOG_FLOOR) / _LOG_FLOOR, 0.0)
    eps = logx - s
    obj = float(np.mean(eps**2))
    dlog = np.where(below, 1.0 / _LOG_FLOOR, 1.0 / x_safe)
    grad = 2.0 * (p1 * (eps * dlog)[:, None]).mean(axis=0)
    return obj, grad


def step1_eps(gamma: np.ndarray, s: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Step 1 residual eps = log(P1 @ gamma) - s (NaN where infeasible)."""
    ppval = p1 @ gamma
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(np.where(ppval > 0, ppval, np.nan)) - s
    return out


# -----------------------------------------------------------------------------
# Step 1 derived objects (est_gnr.m post-Step-1 block)
# -----------------------------------------------------------------------------
def melast(p1: np.ndarray, gamma_scaled: np.ndarray) -> np.ndarray:
    """Materials output elasticity: P1 @ (soln / bigeps)."""
    return p1 @ gamma_scaled


def int_melast(p1: np.ndarray, gamma_scaled: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Integral of the materials elasticity over log m:
    (P1 @ (gamma * INTVEC)) * m."""
    return (p1 @ (gamma_scaled * INTVEC)) * m


def kelast(
    k: np.ndarray,
    m: np.ndarray,
    l: np.ndarray,
    p2: np.ndarray,
    gamma_scaled: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Capital output elasticity.

    d(int_melast)/dk - dC/dk where C(k,l) = P2 @ a is the constant of
    integration. Analytic form (divergence #1; algebraically identical to
    MATLAB's k1sel/k2sel column trick):
      d(int_melast)/dk = m * (g1 + 2*g4*k + g7*m/2 + g8*l)
      dC/dk            = a0 + 2*a2*k + a4*l
    with gamma indexed over [1, k, m, l, k^2, m^2, l^2, k*m, k*l, m*l]
    and a over [k, l, k^2, l^2, k*l].
    """
    g = gamma_scaled
    d_int_dk = m * (g[1] + 2.0 * g[4] * k + 0.5 * g[7] * m + g[8] * l)
    d_c_dk = a[0] + 2.0 * a[2] * k + a[4] * l
    return d_int_dk - d_c_dk


def lelast(
    k: np.ndarray,
    m: np.ndarray,
    l: np.ndarray,
    p2: np.ndarray,
    gamma_scaled: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Labor output elasticity (analytic analog of MATLAB l1sel/l2sel):
      d(int_melast)/dl = m * (g3 + 2*g6*l + g8*k + g9*m/2)
      dC/dl            = a1 + 2*a3*l + a4*k
    """
    g = gamma_scaled
    d_int_dl = m * (g[3] + 2.0 * g[6] * l + g[8] * k + 0.5 * g[9] * m)
    d_c_dl = a[1] + 2.0 * a[3] * l + a[4] * k
    return d_int_dl - d_c_dl


# -----------------------------------------------------------------------------
# Step 2: nested GMM (step2_obj.m, algorithm == "nested")
# -----------------------------------------------------------------------------
def markov_design(
    lomega: np.ndarray, indmat: np.ndarray, h2size: int
) -> np.ndarray:
    """Design matrix of the productivity Markov process:
    [industry dummies, 1, lomega, lomega^2, ...] up to degree h2size - 1."""
    if h2size not in (2, 3, 4):
        raise ValueError(f"h2size must be 2, 3, or 4; got {h2size}")
    ones = np.ones_like(lomega)
    cols = [indmat, ones[:, None]]
    for p in range(1, h2size):
        cols.append((lomega**p)[:, None])
    return np.hstack(cols)


def step2_objective_nested(
    a: np.ndarray,
    rt: np.ndarray,
    lrt: np.ndarray,
    p2: np.ndarray,
    lp2: np.ndarray,
    iv: np.ndarray,
    indmat: np.ndarray,
    h2size: int,
    w: np.ndarray,
    return_full: bool = False,
):
    """Nested Step 2 GMM objective.

    Given the 5 integration-constant parameters `a`:
      omega  = rt  + P2  @ a
      lomega = lrt + LP2 @ a
      delta  = OLS of omega on [indmat, 1, lomega, lomega^2, lomega^3]
      eta    = omega - fitted
      g      = mean(eta * IV);  obj = N * g' W g / 2

    Returns obj, or (obj, delta, eta) when return_full is True.
    """
    n = rt.shape[0]
    omega = rt + p2 @ a
    lomega = lrt + lp2 @ a
    o2 = markov_design(lomega, indmat, h2size)
    # lstsq (not normal equations) for numerical stability with dummies.
    delta, *_ = np.linalg.lstsq(o2, omega, rcond=None)
    eta = omega - o2 @ delta
    g = (eta[:, None] * iv).mean(axis=0)
    obj = float(n * g @ w @ g / 2.0)
    if return_full:
        return obj, delta, eta
    return obj


# -----------------------------------------------------------------------------
# Parallel multistart (MultiStart 'UseParallel' analog)
# -----------------------------------------------------------------------------
# The local solves from independent starting points share read-only data, so
# they are embarrassingly parallel. Data is shipped to each worker process
# once via the pool initializer (module-level _WORKER), not per task.
_WORKER: dict = {}


def _init_worker(payload: tuple) -> None:
    _WORKER["payload"] = payload


def _step1_solve(x0: np.ndarray) -> tuple[float, np.ndarray, bool]:
    """One Step 1 local solve. Payload: (s, p1, bounds, maxiter).

    Tolerances are much tighter than scipy defaults: near-infeasible
    observations give the objective a huge local Lipschitz constant
    (slope ~ 1/_LOG_FLOOR), so with default ftol the relative-reduction
    test fires after one iteration and the solve stalls. maxls=100 lets
    the line search survive the steep barrier region.
    """
    s, p1, bounds, maxiter = _WORKER["payload"]
    res = minimize(
        step1_objective, x0, args=(s, p1), jac=True,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 100},
    )
    feasible = bool((p1 @ res.x).min() > 0)
    return (float(res.fun) if feasible else np.inf, res.x, feasible)


def _step2_solve(x0: np.ndarray) -> tuple[float, np.ndarray]:
    """One Step 2 nested local solve.
    Payload: (rt, lrt, p2, lp2, iv, indmat, h2size, w, maxiter)."""
    rt, lrt, p2, lp2, iv, indmat, h2size, w, maxiter = _WORKER["payload"]
    res = minimize(
        step2_objective_nested, x0,
        args=(rt, lrt, p2, lp2, iv, indmat, h2size, w),
        method="BFGS", options={"maxiter": maxiter},
    )
    return float(res.fun), res.x


def parallel_multistart(solver, starts: np.ndarray, payload: tuple, n_workers: int):
    """Run `solver` (one of _step1_solve / _step2_solve) from every starting
    point, on `n_workers` processes (serial when n_workers <= 1). Returns the
    list of per-start results in start order.

    Worker BLAS pools are pinned to a single thread (the env vars below are
    inherited by the spawned children before they import numpy). Without
    this, n_workers x BLAS-threads oversubscription makes the parallel run
    SLOWER than serial — measured 15x slower on an M4 Pro.
    """
    if n_workers <= 1:
        _init_worker(payload)
        return [solver(x0) for x0 in starts]
    import os

    blas_vars = [
        "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    ]
    saved = {v: os.environ.get(v) for v in blas_vars}
    for v in blas_vars:
        os.environ[v] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(payload,)
        ) as pool:
            return list(pool.map(solver, list(starts)))
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
