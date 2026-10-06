"""
GNR Step 1 (NLS share regression) and Step 2 (nested GMM), shared by every data source.

Moved out of 01_Compustat/00_Hubmer_RTS/01_code/02_estimate_gnr.py (stage R2), logic
unchanged; see that file's docstring for the estimator description. Inputs are the
trimmed panel from gnr_panel.trim_panel and the `step1` / `step2` config blocks.

Step 2's Markov process carries one dummy per industry after the first. With a single
`ind2d` (within-cell estimation, as Hubmer et al.) there are zero dummy columns and the
process is a constant plus the polynomial in lagged omega.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import gnr_model as gm


# -----------------------------------------------------------------------------
# Step 1
# -----------------------------------------------------------------------------
def build_step1_starts(p1: np.ndarray, s: np.ndarray, cfg_s1: dict, logger) -> np.ndarray:
    """Starting set per est_gnr.m: OLS guess, gamma0, 2*gamma0, Sobol points,
    all filtered for feasibility (P1 @ gamma > 0)."""
    n_params = p1.shape[1]

    gamma0 = np.full(n_params, 0.1)
    gamma0[0] = 1.0

    # OLS of exp(s) on P1, intercept rescaled so the guess is feasible
    # (est_gnr.m gamma_ols block).
    gamma_ols, *_ = np.linalg.lstsq(p1, np.exp(s), rcond=None)
    fitted_noint = p1 @ gamma_ols - gamma_ols[0]
    gamma_ols[0] = 0.1 - fitted_noint.min()

    sobol = gm.step1_sobol_pool(cfg_s1)
    n_starts = int(cfg_s1["n_starts"])

    # Jittered copies of the OLS guess. The raw Sobol/gamma0 starts mostly
    # sit in very poor basins on the full sample (objective values of 20+,
    # i.e. share predictions off by e^4); multiplicative jitter around the
    # OLS guess populates the relevant basin so the multistart actually
    # explores where the optimum lives, while the MATLAB-style starts are
    # retained for global coverage.
    n_jitter = int(cfg_s1.get("n_ols_jitter", 20))
    jitter_scale = float(cfg_s1.get("ols_jitter_scale", 0.25))
    rng = np.random.default_rng(int(cfg_s1["seed"]) + 7)
    jittered = gamma_ols * (
        1.0 + jitter_scale * rng.standard_normal((n_jitter, len(gamma_ols)))
    )

    candidates = np.vstack(
        [gamma_ols, gamma0, 2.0 * gamma0, jittered, sobol[:n_starts]]
    )

    starts = gm.nls_const_filter(candidates, p1)
    logger.info(
        f"  step 1 starting guesses: {len(candidates)} candidates, "
        f"{len(starts)} satisfy P1 @ gamma > 0"
    )
    if len(starts) == 0:
        raise RuntimeError(
            "no feasible Step 1 starting point — check the trimmed panel "
            "(this should not happen with the rescaled OLS guess)"
        )
    return starts


def run_step1(est: pd.DataFrame, cfg_s1: dict, logger) -> dict:
    p1 = gm.build_p1(est["k"].to_numpy(), est["m"].to_numpy(), est["l"].to_numpy())
    s = est["s"].to_numpy()
    n_params = p1.shape[1]

    gamma0 = np.full(n_params, 0.1)
    gamma0[0] = 1.0
    bounds = list(zip(-100.0 * gamma0, 100.0 * gamma0))

    starts = build_step1_starts(p1, s, cfg_s1, logger)
    n_workers = int(cfg_s1.get("n_workers", 1))

    t0 = time.time()
    payload = (s, p1, bounds, int(cfg_s1["maxiter_local"]))
    results = gm.parallel_multistart(gm._step1_solve, starts, payload, n_workers)
    n_feas = sum(1 for r in results if r[2])
    logger.info(
        f"  step 1 multistart: {len(results)} local solves "
        f"({n_feas} feasible solutions) in {time.time() - t0:.1f}s "
        f"[{n_workers} worker(s)]"
    )
    if n_feas == 0:
        raise RuntimeError("step 1: no feasible local solution found")

    best_fval, best_x, _ = min(results, key=lambda r: r[0])
    logger.info(f"  best multistart fval: {best_fval:.6g}")

    # Nelder-Mead polish (MATLAB fminsearch verification step)
    def obj_only(x):
        return gm.step1_objective(x, s, p1)[0]

    polish = minimize(
        obj_only,
        best_x,
        method="Nelder-Mead",
        options={
            "maxfev": int(cfg_s1["maxfev_polish"]),
            "maxiter": int(cfg_s1["maxfev_polish"]),
            "xatol": 1e-8,
            "fatol": 1e-10,
        },
    )
    if polish.fun <= best_fval and (p1 @ polish.x).min() > 0:
        soln, fval = polish.x, float(polish.fun)
    else:
        soln, fval = best_x, float(best_fval)
    logger.info(f"  polished fval: {fval:.6g}")

    eps = gm.step1_eps(soln, s, p1)
    if np.isnan(eps).any():
        raise RuntimeError("step 1: residuals contain NaN at the solution")
    bigeps = float(np.mean(np.exp(eps)))
    gamma_scaled = soln / bigeps
    melast = gm.melast(p1, gamma_scaled)
    logger.info(
        f"  mean(eps) = {eps.mean():+.5f}; bigeps = mean(exp(eps)) = {bigeps:.5f}"
    )
    logger.info(
        f"  materials elasticity: mean = {melast.mean():.4f}, "
        f"sd = {melast.std():.4f}, share <= 0: {(melast <= 0).mean():.2%}"
    )

    return {
        "gamma_raw": soln.tolist(),
        "gamma_scaled": gamma_scaled.tolist(),
        "bigeps": bigeps,
        "fval": fval,
        "fval_multistart_best": float(best_fval),
        "n_starts": len(starts),
        "n_feasible_solutions": n_feas,
        "n_obs": int(len(est)),
        "eps_mean": float(eps.mean()),
        "eps_sd": float(eps.std()),
        "melast_mean": float(melast.mean()),
        "melast_sd": float(melast.std()),
        "seed": int(cfg_s1["seed"]),
    }


# -----------------------------------------------------------------------------
# Step 2
# -----------------------------------------------------------------------------
def build_step2_sample(
    est: pd.DataFrame, soln: np.ndarray, require_wages: bool, logger
) -> tuple[pd.DataFrame, dict]:
    """Step 2 selection: lag-polynomial feasibility (+ optional wage filter)."""
    lp1 = gm.build_p1(est["lk"].to_numpy(), est["lm"].to_numpy(), est["ll"].to_numpy())
    lag_feasible = (lp1 @ soln) > 0
    counts = {"n_trimmed": int(len(est)), "n_lag_infeasible": int((~lag_feasible).sum())}
    sel = lag_feasible
    if require_wages:
        has_lw = est["lw"].notna().to_numpy()
        counts["n_missing_lw"] = int((~has_lw).sum())
        sel = sel & has_lw
    counts["n_step2"] = int(sel.sum())
    logger.info(
        f"  step 2 sample: {counts['n_step2']:,} of {counts['n_trimmed']:,} "
        f"trimmed rows (lag-infeasible: {counts['n_lag_infeasible']:,}"
        + (f", missing lw: {counts['n_missing_lw']:,}" if require_wages else "")
        + ")"
    )
    if counts["n_step2"] == 0:
        raise RuntimeError("step 2 sample is empty")
    return est.loc[sel].reset_index(drop=True), counts


def step2_arrays(sub: pd.DataFrame, s1: dict, logger):
    """Construct rt, lrt, P2, LP2, IV and industry dummies for Step 2."""
    soln = np.asarray(s1["gamma_raw"])
    gamma_scaled = np.asarray(s1["gamma_scaled"])

    k, m, l = (sub[c].to_numpy() for c in ["k", "m", "l"])
    lk, lm, ll = (sub[c].to_numpy() for c in ["lk", "lm", "ll"])

    p1 = gm.build_p1(k, m, l)
    lp1 = gm.build_p1(lk, lm, ll)

    eps = gm.step1_eps(soln, sub["s"].to_numpy(), p1)
    leps = gm.step1_eps(soln, sub["ls"].to_numpy(), lp1)
    if np.isnan(eps).any() or np.isnan(leps).any():
        raise RuntimeError("step 2: NaN residuals after lag-feasibility selection")

    rt = sub["r"].to_numpy() - eps - gm.int_melast(p1, gamma_scaled, m)
    lrt = sub["lr"].to_numpy() - leps - gm.int_melast(lp1, gamma_scaled, lm)

    p2 = gm.build_p2(k, l)
    lp2 = gm.build_p2(lk, ll)
    iv = gm.build_iv(k, ll)

    # Industry dummies, first category dropped (est_gnr.m: dumind(:,2:end)).
    ind_categories = sorted(sub["ind2d"].unique().tolist())
    dummies = pd.get_dummies(
        pd.Categorical(sub["ind2d"], categories=ind_categories), dtype=float
    ).to_numpy()[:, 1:]
    logger.info(
        f"  industry dummies: {len(ind_categories)} categories "
        f"({ind_categories}), first dropped -> {dummies.shape[1]} columns"
    )
    return rt, lrt, p2, lp2, iv, dummies, ind_categories


def run_step2(sub: pd.DataFrame, s1: dict, cfg_s2: dict, logger) -> dict:
    rt, lrt, p2, lp2, iv, dummies, ind_categories = step2_arrays(sub, s1, logger)
    h2size = int(cfg_s2["h2size"])
    w_mat = np.eye(iv.shape[1])

    def obj(a):
        return gm.step2_objective_nested(
            a, rt, lrt, p2, lp2, iv, dummies, h2size, w_mat
        )

    # Starting points: OLS linguess, alpha0, first n_starts Sobol (5-dim).
    x_design = np.column_stack([np.ones(len(rt)), -p2])
    linguess_full, *_ = np.linalg.lstsq(x_design, rt, rcond=None)
    linguess = linguess_full[1:]
    alpha0 = np.full(gm.P2SIZE, 1.0e-4)
    lo, hi = map(float, cfg_s2["sobol_bounds"])
    sobol = gm.gensobol(
        int(cfg_s2["n_sobol"]), gm.P2SIZE, lo, hi, seed=int(cfg_s2["seed"])
    )
    start_blocks = [linguess[None, :], alpha0[None, :]]
    economic_starts = cfg_s2.get("economic_starts", [])
    if economic_starts:
        econ = np.asarray(economic_starts, dtype=float)
        if econ.ndim != 2 or econ.shape[1] != gm.P2SIZE:
            raise ValueError(
                "step2.economic_starts must be a list of 5-parameter vectors"
            )
        start_blocks.append(econ)
    start_blocks.append(sobol[: int(cfg_s2["n_starts"])])

    n_wide = int(cfg_s2.get("n_wide_starts", 0))
    if n_wide:
        wlo, whi = map(float, cfg_s2.get("wide_sobol_bounds", [-3.0, 3.0]))
        wide = gm.gensobol(
            max(n_wide, 1), gm.P2SIZE, wlo, whi, seed=int(cfg_s2["seed"]) + 101
        )
        start_blocks.append(wide[:n_wide])

    starts = np.vstack(start_blocks)
    n_workers = int(cfg_s2.get("n_workers", 1))
    logger.info(
        f"  step 2 multistart: {len(starts)} starting points "
        f"(nested, W = I; economic={len(economic_starts)}, wide={n_wide})"
    )

    t0 = time.time()
    payload = (rt, lrt, p2, lp2, iv, dummies, h2size, w_mat, int(cfg_s2["maxiter_local"]))
    raw = gm.parallel_multistart(gm._step2_solve, starts, payload, n_workers)
    results = [(f, x) for f, x in raw if np.isfinite(f)]
    logger.info(
        f"  step 2 multistart: {len(results)} finite local solutions "
        f"in {time.time() - t0:.1f}s [{n_workers} worker(s)]"
    )
    if not results:
        raise RuntimeError("step 2: no finite local solution found")
    best_fval, best_x = min(results, key=lambda r: r[0])
    fvals = np.asarray([r[0] for r in results], dtype=float)
    fval_q = np.quantile(fvals, [0.0, 0.1, 0.5, 0.9, 1.0])
    logger.info(
        "  step 2 local objective distribution "
        f"min/p10/p50/p90/max = {np.round(fval_q, 6).tolist()}"
    )
    near_best = int((fvals <= best_fval + max(1e-8, abs(best_fval) * 1e-4)).sum())
    logger.info(f"  step 2 near-best local solutions: {near_best} / {len(results)}")
    logger.info(f"  best multistart fval: {best_fval:.6g}")

    polish = minimize(
        obj, best_x, method="Nelder-Mead",
        options={
            "maxfev": int(cfg_s2["maxfev_polish"]),
            "maxiter": int(cfg_s2.get("maxiter_polish", cfg_s2["maxfev_polish"])),
            "xatol": 1e-8,
            "fatol": 1e-10,
        },
    )
    if np.isfinite(polish.fun) and polish.fun <= best_fval:
        a_hat, fval = polish.x, float(polish.fun)
    else:
        a_hat, fval = best_x, float(best_fval)
    logger.info(f"  polished fval: {fval:.6g}")

    # Recover the concentrated-out Markov coefficients at the solution.
    _, delta, eta = gm.step2_objective_nested(
        a_hat, rt, lrt, p2, lp2, iv, dummies, h2size, w_mat, return_full=True
    )
    omega = rt + p2 @ a_hat
    lomega = lrt + lp2 @ a_hat
    logger.info(
        f"  sd(omega) = {omega.std():.4f}, sd(eta) = {eta.std():.4f}, "
        f"corr(omega, lomega) = {np.corrcoef(omega, lomega)[0, 1]:.4f}"
    )

    return {
        "alpha": a_hat.tolist(),
        "delta": delta.tolist(),
        "delta_layout": {
            "ind_dummy_categories": ind_categories,
            "first_category_dropped": True,
            "n_dummy_cols": int(dummies.shape[1]),
            "markov_terms": [f"lomega^{p}" for p in range(0, h2size)],
            "order": "[dummies..., const, lomega, lomega^2, ...]",
        },
        "h2size": h2size,
        "fval": fval,
        "fval_multistart_best": float(best_fval),
        "local_solution_diagnostics": {
            "n_finite": int(len(results)),
            "fval_min": float(fval_q[0]),
            "fval_p10": float(fval_q[1]),
            "fval_p50": float(fval_q[2]),
            "fval_p90": float(fval_q[3]),
            "fval_max": float(fval_q[4]),
            "near_best_count": near_best,
        },
        "n_starts": int(len(starts)),
        "n_obs": int(len(sub)),
        "omega_sd": float(omega.std()),
        "eta_sd": float(eta.std()),
        "seed": int(cfg_s2["seed"]),
    }
