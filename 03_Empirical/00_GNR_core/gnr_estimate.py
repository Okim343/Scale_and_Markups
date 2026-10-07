"""
GNR Step 1 (NLS share regression) and Step 2 (nested GMM), shared by every data source.

Moved out of 01_Compustat/00_Hubmer_RTS/01_code/02_estimate_gnr.py (stage R2), logic
unchanged; see that file's docstring for the estimator description. Inputs are the
trimmed panel from gnr_panel.trim_panel and the `step1` / `step2` config blocks.

Search mode (`search` key in the step1 / step2 config blocks):
  extended (default): the Compustat pipeline's multistart. Step 1 adds 20 jittered copies
      of the OLS guess; Step 2 adds 2 economic starts and 50 wide Sobol starts in [-3, 3]
      and solves each start with scipy BFGS (maxiter from config). Polish results are kept
      only if they improve the objective.
  matlab (Step 1 only): est_gnr.m's start set, {OLS guess, gamma0, 2*gamma0, first n_starts
      Sobol points}, and an fminsearch-like Nelder-Mead polish (TolX = TolFun = 1e-4,
      MATLAB defaults) whose result is accepted.
  exact (Step 2 only): est_gnr.m's moments and start set {OLS guess, alpha0, first n_starts
      Sobol points in sobol_bounds}, with k and l (current and lagged) centered and scaled
      before building P2 / IV; each start solves the moment equations g = 0 by
      Levenberg-Marquardt with the exact Jacobian (gnr_model.step2_moments_jacobian).
      Standardizing is a linear reparametrization: it leaves the set of exact roots
      unchanged; the solution is mapped back to the original P2 coefficients. Every endpoint
      is reported, with how many are exact roots and how many distinct roots there are; one
      root is picked by the rule in step2.root_selection (see README.md). est_gnr.m's own
      fminunc search stalls at non-stationary points on Orbis cells, so it is not offered.

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

STEP1_SEARCH_MODES = ("extended", "matlab")
STEP2_SEARCH_MODES = ("extended", "exact")


def search_mode(cfg: dict, modes: tuple = STEP1_SEARCH_MODES) -> str:
    """The `search` key of a step1 / step2 config block (default: extended)."""
    mode = cfg.get("search", "extended")
    if mode not in modes:
        raise ValueError(f"search must be one of {modes}, got {mode!r}")
    return mode


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

    if search_mode(cfg_s1) == "matlab":
        # est_gnr.m: paramset = [gamma_ols; gamma0; gamma0.*2; sobolvec1(1:100,:)]
        candidates = np.vstack([gamma_ols, gamma0, 2.0 * gamma0, sobol[:n_starts]])
    else:
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

    matlab_search = search_mode(cfg_s1) == "matlab"
    polish = minimize(
        obj_only,
        best_x,
        method="Nelder-Mead",
        options={
            "maxfev": int(cfg_s1["maxfev_polish"]),
            "maxiter": int(cfg_s1["maxfev_polish"]),
            # fminsearch defaults TolX = TolFun = 1e-4 under search: matlab
            "xatol": 1e-4 if matlab_search else 1e-8,
            "fatol": 1e-4 if matlab_search else 1e-10,
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

    out = {
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
    if matlab_search:  # extended output keeps its original keys
        out["search"] = "matlab"
    return out


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
    if search_mode(cfg_s2, STEP2_SEARCH_MODES) == "exact":
        return run_step2_exact(sub, s1, cfg_s2, logger)
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


ROOT_J_TOL = 1e-6        # efficient J below this counts as an exact root
ROOT_CLUSTER_TOL = 1e-5  # relative distance in original coefficients for "same root"
ROOT_SELECTION_RULES = ("most_starts", "most_starts_grouped")


def root_elasticities(sub: pd.DataFrame, s1: dict, a: np.ndarray) -> dict:
    """Step-2-sample moments of the elasticities implied by P2 coefficients `a`, computed as
    gnr_postprocess.reconstruct_firm_year does (melast from step 1, gm.kelast / gm.lelast)."""
    gamma_scaled = np.asarray(s1["gamma_scaled"])
    k, m, l = (sub[c].to_numpy() for c in ["k", "m", "l"])
    p2 = gm.build_p2(k, l)
    mel = gm.melast(gm.build_p1(k, m, l), gamma_scaled)
    kel = gm.kelast(k, m, l, p2, gamma_scaled, a)
    lel = gm.lelast(k, m, l, p2, gamma_scaled, a)
    out = {}
    for name, x in (("melast", mel), ("lelast", lel), ("kelast", kel), ("rts", mel + kel + lel)):
        q = np.percentile(x, [10, 50, 90])
        out[name] = {"mean": float(x.mean()), "p10": float(q[0]), "p50": float(q[1]),
                     "p90": float(q[2]), "share_neg": float((x < 0).mean())}
    return out


def group_roots(elast: list, tol: dict) -> list:
    """Single-linkage groups of roots whose mean RTS, lelast and kelast all differ by less
    than tol["rts"], tol["lelast"], tol["kelast"]. Returns lists of root indices."""
    n = len(elast)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if all(abs(elast[i][v]["mean"] - elast[j][v]["mean"]) < float(tol[v])
                   for v in ("rts", "lelast", "kelast")):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def select_root(counts: list, jvals: list, elast: list | None = None, tol: dict | None = None):
    """Selection rule over distinct roots (coefficient clusters).

    Without `elast` (rule most_starts): the root reached from most starts, ties broken by the
    smaller J. With `elast` and `tol` (rule most_starts_grouped): roots are first grouped by
    group_roots; the group with most starts in total wins (ties: smaller J of its
    representative), and its representative is its member reached from most starts (ties:
    smaller J). Returns (chosen root index, groups, chosen group index, representatives)."""
    def rank(i):
        return (counts[i], -jvals[i])

    if elast is None:
        groups = [[i] for i in range(len(counts))]
    else:
        groups = group_roots(elast, tol)
    reps = [max(g, key=rank) for g in groups]
    chosen_g = max(range(len(groups)),
                   key=lambda gi: (sum(counts[i] for i in groups[gi]), -jvals[reps[gi]]))
    return reps[chosen_g], groups, chosen_g, reps


def _standardized_step2(sub: pd.DataFrame):
    """P2, LP2 and IV built from k, l and lags centered and scaled by the current-period
    means and SDs, plus the exact linear maps between original (a) and standardized (b)
    P2 coefficients: P2 a = const + P2s (B a), P2s b = const + P2 (A b)."""
    k, l, lk, ll = (sub[c].to_numpy() for c in ["k", "l", "lk", "ll"])
    mk, sk, ml, sl = k.mean(), k.std(), l.mean(), l.std()
    zk, zl, zlk, zll = (k - mk) / sk, (l - ml) / sl, (lk - mk) / sk, (ll - ml) / sl
    p2s, lp2s = gm.build_p2(zk, zl), gm.build_p2(zlk, zll)
    ivs = gm.build_iv(zk, zll)
    p2 = gm.build_p2(k, l)
    rows = slice(0, min(len(k), 5000))
    ones = np.ones((rows.stop, 1))
    b_of_a = np.linalg.lstsq(np.hstack([ones, p2s[rows]]), p2[rows], rcond=None)[0][1:]
    a_of_b = np.linalg.lstsq(np.hstack([ones, p2[rows]]), p2s[rows], rcond=None)[0][1:]
    if not np.allclose(a_of_b @ b_of_a, np.eye(p2.shape[1]), atol=1e-8):
        raise RuntimeError("standardization map is not invertible")
    return p2s, lp2s, ivs, b_of_a, a_of_b


def run_step2_exact(sub: pd.DataFrame, s1: dict, cfg_s2: dict, logger) -> dict:
    """Step 2 by direct root solving (see module docstring, search: exact)."""
    rt, lrt, p2, lp2, iv, dummies, ind_categories = step2_arrays(sub, s1, logger)
    h2size = int(cfg_s2["h2size"])
    w_mat = np.eye(iv.shape[1])
    n = len(rt)
    p2s, lp2s, ivs, b_of_a, a_of_b = _standardized_step2(sub)

    # est_gnr.m's Step 2 start set (linguess, alpha0, Sobol), in standardized coefficients.
    x_design = np.column_stack([np.ones(n), -p2])
    linguess = np.linalg.lstsq(x_design, rt, rcond=None)[0][1:]
    lo, hi = map(float, cfg_s2["sobol_bounds"])
    sobol = gm.gensobol(int(cfg_s2["n_sobol"]), gm.P2SIZE, lo, hi, seed=int(cfg_s2["seed"]))
    starts_a = np.vstack([linguess, np.full(gm.P2SIZE, 1.0e-4), sobol[: int(cfg_s2["n_starts"])]])
    starts_b = starts_a @ b_of_a.T
    n_workers = int(cfg_s2.get("n_workers", 1))
    logger.info(f"  step 2 (search: exact): {len(starts_b)} Levenberg-Marquardt root solves, "
                "standardized k / l, exact Jacobian")

    t0 = time.time()
    payload = (rt, lrt, p2s, lp2s, ivs, dummies, h2size)
    raw = gm.parallel_multistart(gm._step2_solve_exact, starts_b, payload, n_workers)
    endpoints = []
    for cost, b, status in raw:
        if not np.isfinite(cost):
            endpoints.append({"finite": False, "status": status})
            continue
        a = a_of_b @ b
        g, _, eta, _ = gm.step2_moments_jacobian(a, rt, lrt, p2, lp2, iv, dummies, h2size)
        hmat = eta[:, None] * iv
        j_eff = float(n * g @ np.linalg.solve(np.cov(hmat.T), g))
        endpoints.append({"finite": True, "status": status, "a": a, "J": j_eff,
                          "fval_orig": float(n * g @ g / 2.0), "cost_std": cost})
    fin = [e for e in endpoints if e["finite"]]
    roots = [e for e in fin if e["J"] < ROOT_J_TOL]
    clusters = []
    for e in sorted(roots, key=lambda e: e["J"]):
        for c in clusters:
            if np.max(np.abs(e["a"] - c["a"]) / (1.0 + np.abs(c["a"]))) < ROOT_CLUSTER_TOL:
                c["count"] += 1
                break
        else:
            clusters.append({"a": e["a"], "count": 1, "J": e["J"]})
    logger.info(
        f"  step 2 root solves: {len(fin)} finite of {len(endpoints)}, {len(roots)} exact roots "
        f"(J < {ROOT_J_TOL:g}), {len(clusters)} distinct root(s) in {time.time() - t0:.1f}s "
        f"[{n_workers} worker(s)]"
    )
    if not fin:
        raise RuntimeError("step 2: no finite root-solve endpoint")
    sel_cfg = cfg_s2.get("root_selection")
    rule = sel_cfg["rule"] if sel_cfg else "most_starts"
    if rule not in ROOT_SELECTION_RULES:
        raise ValueError(f"step2.root_selection.rule must be one of {ROOT_SELECTION_RULES}, got {rule!r}")
    grouped = None
    if clusters:
        counts = [c["count"] for c in clusters]
        jv = [c["J"] for c in clusters]
        if rule == "most_starts_grouped":
            tol = sel_cfg["group_tol"]
            elast = [root_elasticities(sub, s1, c["a"]) for c in clusters]
            chosen, groups, chosen_g, reps = select_root(counts, jv, elast, tol)
            grouped = {"selection_rule": rule, "group_tol": {v: float(tol[v]) for v in tol},
                       "root_elasticities": elast, "chosen_root": int(chosen),
                       "n_groups": len(groups), "chosen_group": int(chosen_g),
                       "root_groups": [{"members": [int(i) for i in g],
                                        "starts": int(sum(counts[i] for i in g)),
                                        "representative": int(r), "elasticities": elast[r]}
                                       for g, r in zip(groups, reps)]}
            for gi, g in enumerate(grouped["root_groups"]):
                e = g["elasticities"]
                logger.info(f"  root group {gi}{' (chosen)' if gi == chosen_g else ''}: roots "
                            f"{g['members']}, {g['starts']} starts; mean m {e['melast']['mean']:.3f}, "
                            f"l {e['lelast']['mean']:.3f}, k {e['kelast']['mean']:.4f}, "
                            f"RTS {e['rts']['mean']:.3f}")
            if len(groups) > 1:
                logger.warning(f"  {len(groups)} economically distinct root groups; using group "
                               f"{chosen_g} ({grouped['root_groups'][chosen_g]['starts']} of "
                               f"{len(roots)} root solves), root {chosen}")
        else:
            chosen = select_root(counts, jv)[0]
        best = clusters[chosen]
        a_hat = best["a"]
        if len(clusters) > 1 and grouped is None:
            logger.warning(f"  {len(clusters)} distinct exact roots; using the one reached most often "
                           f"({best['count']} of {len(roots)} root solves)")
    else:
        best_e = min(fin, key=lambda e: e["J"])
        a_hat = best_e["a"]
        logger.warning(f"  no exact root reached; using the endpoint with the smallest J ({best_e['J']:.4g})")

    fval, delta, eta = gm.step2_objective_nested(
        a_hat, rt, lrt, p2, lp2, iv, dummies, h2size, w_mat, return_full=True
    )
    omega = rt + p2 @ a_hat
    hmat = eta[:, None] * iv
    g_hat = hmat.mean(axis=0)
    j_hat = float(n * g_hat @ np.linalg.solve(np.cov(hmat.T), g_hat))
    logger.info(f"  J at the estimate: {j_hat:.3g}")
    logger.info(f"  fval (original W = I objective): {fval:.6g}; "
                f"sd(omega) = {omega.std():.4f}, sd(eta) = {eta.std():.4f}")
    jvals = np.asarray([e["J"] for e in fin])
    return {
        "alpha": np.asarray(a_hat).tolist(),
        "delta": delta.tolist(),
        "delta_layout": {
            "ind_dummy_categories": ind_categories,
            "first_category_dropped": True,
            "n_dummy_cols": int(dummies.shape[1]),
            "markov_terms": [f"lomega^{p}" for p in range(0, h2size)],
            "order": "[dummies..., const, lomega, lomega^2, ...]",
        },
        "h2size": h2size,
        "fval": float(fval),
        "fval_multistart_best": float(min(e["fval_orig"] for e in fin)),
        "local_solution_diagnostics": {
            "n_finite": int(len(fin)),
            "n_converged": int(len(roots)),
            "n_exact_roots": int(len(roots)),
            "n_distinct_roots": int(len(clusters)),
            "root_counts": [int(c["count"]) for c in clusters],
            "root_alphas": [np.asarray(c["a"]).tolist() for c in clusters],
            "J_min": float(jvals.min()), "J_p50": float(np.median(jvals)),
            "J_estimate": j_hat, "root_found": bool(clusters),
            "near_best_count": int(clusters[0]["count"]) if clusters else 0,
            **(grouped or {}),
        },
        "n_starts": int(len(starts_b)),
        "n_obs": int(len(sub)),
        "omega_sd": float(omega.std()),
        "eta_sd": float(eta.std()),
        "seed": int(cfg_s2["seed"]),
        "search": "exact",
    }
