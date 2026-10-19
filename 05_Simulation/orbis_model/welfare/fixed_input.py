"""Fixed-input planner: reallocate a fixed bundle of capital, labor and materials.

Ported from ``04_Code/counterfactuals/fixed_input_welfare`` (the fixed-capital
planner and its decomposition). Every allocation imposes ``P = 1`` and
``L = 1``. The 04_Code planner additionally pins aggregate capital at the
market level by freeing the rental rate ``R``; materials stay free there, so
part of its "reallocation" gain is a scale effect through intermediates. This
module also pins aggregate materials by freeing a shadow materials price
``P_mat``, which replaces ``P = 1`` inside the gross-output unit cost
``Omega^g(w, R, P_mat)`` exactly as ``R`` does for capital. The forward pass is
:func:`orbis_model.model.normalization.solve_pooled_economy`, unchanged apart
from the ``P_mat`` argument (default 1).

Consumption is net of materials and replacement investment in every regime,
``C = Q - M - delta_K K`` (:func:`orbis_model.model.aggregator.aggregate_pooled_markets`).
The fixed-input regimes hold ``K`` at the market level, so their investment
equals the market's; the free planner's larger ``K`` carries its own
investment cost.

The regime chain is

    ME -> PE_I (K, L, M fixed) -> PE_K (K, L fixed, M free) -> PE (L fixed)

with legs ``lambda_realloc`` (ME -> PE_I), ``lambda_materials`` (PE_I -> PE_K)
and ``lambda_capital`` (PE_K -> PE), and the multiplicative identity
``(1+l_r)(1+l_m)(1+l_c) = 1 + lambda_total``. The additive versions are
``Delta = ln(1 + lambda)``, which sum exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..model.aggregator import pooled_moments
from ..model.normalization import (
    PooledEquilibrium,
    PooledParams,
    euler_R,
    solve_pooled_economy,
    solve_pooled_ge,
)
from ..model.pricing import PricingRegime
from .welfare_metrics import consumption_equivalent_lambda, steady_state_welfare


# Regime labels in chain order, and the leg names between consecutive regimes.
REGIMES: tuple[str, ...] = (
    "market", "planner_fixed_input", "planner_fixed_k", "planner",
)
LEGS: dict[tuple[str, str], str] = {
    ("market", "planner_fixed_input"): "realloc",
    ("planner_fixed_input", "planner_fixed_k"): "materials",
    ("planner_fixed_k", "planner"): "capital",
}


def solve_fixed_input_planner(
    alpha: np.ndarray,
    v: np.ndarray,
    params: PooledParams,
    *,
    K_target: float,
    M_target: float | None = None,
    delta_K: float,
    beta: float = 0.96,
    n_firms_cs: float = 1_000.0,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    regime: PricingRegime = PricingRegime.PLANNER,
    mu_bar: float | None = None,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    active_mask: np.ndarray,
    initial: tuple[float, ...] | None = None,
    tol: float = 1e-5,
    max_nfev: int = 40,
    stats: dict | None = None,
) -> PooledEquilibrium:
    """Solve pooled GE with ``P = 1``, ``L = 1``, ``K = K_target`` (and ``M = M_target``).

    With ``M_target`` None this is the 04_Code fixed-capital planner: unknowns
    ``(log w, log X_market, log R)``, residuals ``log P``, ``log L`` and
    ``log(K / K_target)``. With ``M_target`` set, ``log P_mat`` is a fourth
    unknown and ``log(M / M_target)`` a fourth residual, where ``M`` is the
    materials quantity ``(1 - phi_v) TC / P_mat``.

    ``R`` is a free unknown, so it no longer implies ``delta_K``; ``delta_K`` is
    passed explicitly to the forward pass for the investment term. ``beta`` only
    sets the default initial ``R`` (the Euler rate). ``initial`` is
    ``(w, X_market, R)`` or ``(w, X_market, R, P_mat)``; a 3-tuple is padded with
    ``P_mat = 1`` when ``M_target`` is set. When ``stats`` is a dict it is filled
    with ``nfev``, ``njev`` and ``n_forward`` (forward passes, Jacobian
    evaluations included).
    """
    if wf_reference is None:
        wf_reference = 0.0
    fix_materials = M_target is not None
    n_unknowns = 4 if fix_materials else 3
    if initial is None:
        initial = (1.0, 1.0, euler_R(beta, delta_K))
    initial = tuple(float(x) for x in initial)
    if fix_materials and len(initial) == 3:
        initial = initial + (1.0,)
    if len(initial) != n_unknowns:
        raise ValueError(f"initial must have {n_unknowns} entries; got {len(initial)}")
    selected_mask = np.asarray(active_mask, dtype=bool)
    n_forward = 0

    def evaluate(x: np.ndarray) -> PooledEquilibrium:
        nonlocal n_forward
        n_forward += 1
        prices = np.exp(x)
        w, X, R = prices[:3]
        P_mat = prices[3] if fix_materials else 1.0
        return solve_pooled_economy(
            alpha, v, params, w=w, R=R, X_market=X,
            n_firms_cs=n_firms_cs, delta_K=delta_K, exposure=exposure,
            K_switch=K_switch, love_of_variety=love_of_variety,
            regime=regime, mu_bar=mu_bar,
            market_solver_kwargs=market_solver_kwargs,
            wf_reference=wf_reference, active_mask=selected_mask, P_mat=P_mat,
        )

    def residual_of(candidate: PooledEquilibrium) -> np.ndarray:
        targets = [candidate.P_agg, candidate.L_agg, candidate.K_agg / K_target]
        if fix_materials:
            targets.append(candidate.aggregates.M / M_target)
        return np.log(targets)

    sol = least_squares(
        lambda value: residual_of(evaluate(value)), np.log(initial),
        method="trf", ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_nfev, diff_step=1e-3, x_scale=np.ones(n_unknowns),
    )
    eq = evaluate(sol.x)
    resid = float(np.max(np.abs(residual_of(eq))))
    if stats is not None:
        stats.update({"nfev": int(sol.nfev), "njev": int(sol.njev or 0),
                      "n_forward": int(n_forward), "status": int(sol.status)})
    return PooledEquilibrium(
        **{**eq.__dict__, "chi": eq.w / max(eq.C, 1e-30),
           "converged": bool(sol.success and eq.converged and resid <= 10 * tol),
           "residual": resid, "n_func_evals": int(sol.nfev)}
    )


@dataclass(frozen=True)
class FixedInputChain:
    """The four regimes on common draws and ME's active mask (PE_K, PE optional).

    ``wall_s`` and ``stats`` are keyed by regime label; ``stats`` holds the
    solver counts (``nfev``, ``njev``, ``n_forward`` for the fixed-input legs).
    """

    me: PooledEquilibrium
    pe_i: PooledEquilibrium
    pe_k: PooledEquilibrium | None
    pe: PooledEquilibrium | None
    R_euler: float
    wall_s: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def solve_fixed_input_chain(
    alpha: np.ndarray,
    v: np.ndarray,
    params: PooledParams,
    *,
    beta: float,
    delta_K: float,
    n_firms_cs: float,
    active_mask: np.ndarray,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    initial: tuple[float, float] | None = None,
    tol: float = 1e-5,
    max_nfev: int = 40,
    fixed_k: bool = True,
    full_planner: bool = True,
) -> FixedInputChain:
    """Solve ME, PE_K, PE_I and PE with the warm starts of the 04_Code runner.

    ME is solved exactly as the market leg of
    :func:`orbis_model.welfare.allocations.solve_market_and_planner`, and PE
    exactly as its planner leg (warm start ``(me.w, me.X_market)``), so the
    chain's ``lambda_total`` matches ``02_welfare.py``. All planners run on ME's
    active mask and ``wf_reference``. PE_K starts from ``(me.w, me.X_market,
    R_euler)``; PE_I from ``(pe_k.w, pe_k.X_market, pe_k.R, 1)``, or from
    ``(me.w, me.X_market, R_euler, 1)`` when PE_K is skipped. ``tol`` and
    ``max_nfev`` apply to the fixed-input legs only.
    """
    R_euler = euler_R(beta, delta_K)
    common = dict(
        n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
        love_of_variety=love_of_variety,
        market_solver_kwargs=market_solver_kwargs,
    )
    wall: dict = {}
    stats: dict = {}

    t0 = time.perf_counter()
    me = solve_pooled_ge(
        alpha, v, params, beta=beta, delta_K=delta_K,
        regime=PricingRegime.MARKET, wf_reference=wf_reference,
        active_mask=active_mask, initial=initial, **common,
    )
    wall["market"] = time.perf_counter() - t0
    stats["market"] = {"nfev": me.n_func_evals}
    planner_common = dict(
        delta_K=delta_K, beta=beta, regime=PricingRegime.PLANNER,
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask, tol=tol, max_nfev=max_nfev,
        **common,
    )
    K_target = float(me.K_agg)
    M_target = float(me.aggregates.M)

    pe_k = None
    if fixed_k:
        t0 = time.perf_counter()
        stats["planner_fixed_k"] = {}
        pe_k = solve_fixed_input_planner(
            alpha, v, params, K_target=K_target,
            initial=(me.w, me.X_market, R_euler),
            stats=stats["planner_fixed_k"], **planner_common,
        )
        wall["planner_fixed_k"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats["planner_fixed_input"] = {}
    start = ((pe_k.w, pe_k.X_market, pe_k.R, 1.0) if pe_k is not None
             else (me.w, me.X_market, R_euler, 1.0))
    pe_i = solve_fixed_input_planner(
        alpha, v, params, K_target=K_target, M_target=M_target,
        initial=start, stats=stats["planner_fixed_input"], **planner_common,
    )
    wall["planner_fixed_input"] = time.perf_counter() - t0

    pe = None
    if full_planner:
        t0 = time.perf_counter()
        pe = solve_pooled_ge(
            alpha, v, params, beta=beta, delta_K=delta_K,
            regime=PricingRegime.PLANNER,
            wf_reference=me.participation.wf_reference,
            active_mask=me.participation.active_mask,
            initial=(me.w, me.X_market), **common,
        )
        wall["planner"] = time.perf_counter() - t0
        stats["planner"] = {"nfev": pe.n_func_evals}
    return FixedInputChain(me=me, pe_i=pe_i, pe_k=pe_k, pe=pe,
                           R_euler=R_euler, wall_s=wall, stats=stats)


def _chain(
    me: PooledEquilibrium,
    pe_i: PooledEquilibrium | None,
    pe_k: PooledEquilibrium | None,
    pe: PooledEquilibrium | None,
) -> list[tuple[str, PooledEquilibrium]]:
    regimes = zip(REGIMES, (me, pe_i, pe_k, pe))
    return [(label, eq) for label, eq in regimes if eq is not None]


def fixed_input_table(
    me: PooledEquilibrium,
    pe_i: PooledEquilibrium | None = None,
    pe_k: PooledEquilibrium | None = None,
    pe: PooledEquilibrium | None = None,
    *,
    n_firms_cs: float,
    R_euler: float | None = None,
) -> pd.DataFrame:
    """Regime-level aggregates for ME, PE_I, PE_K and PE (any of the last three optional).

    Same schema as :func:`orbis_model.welfare.welfare_metrics.welfare_decomposition_table`
    plus ``Q, M, I, R, P_mat``, the gaps of ``K`` and ``M`` to the market levels
    (the targets of the fixed-input regimes) and the Walras diagnostics.
    """
    K_target = float(me.K_agg)
    M_target = float(me.aggregates.M)
    rows = []
    for label, eq in _chain(me, pe_i, pe_k, pe):
        m = pooled_moments(
            eq.participation.solution, eq.participation.active_mask,
            n_firms_cs=n_firms_cs,
        )
        walras = eq.walras_check(R_euler)
        rows.append({
            "regime": label, "C": eq.C, "Q": eq.aggregates.Q,
            "M": eq.aggregates.M, "I": eq.aggregates.I, "L": eq.L_agg,
            "K": eq.K_agg, "R": eq.R, "P_mat": eq.P_mat, "w": eq.w,
            "X_market": eq.X_market, "profits": eq.Pi_agg,
            "mu_cw": m["mu_cw"],              # pure cost-weighted markup
            "mu_cw_alpha": m["mu_cw_alpha"],  # accounting Σsales/Σcost = μ/α
            "K_target_gap": eq.K_agg / K_target - 1.0,
            "M_target_gap": eq.aggregates.M / M_target - 1.0,
            "rent_capital": walras.get("rent_capital", np.nan),
            "rent_materials": walras["rent_materials"],
            "walras_rel_gap": walras["rel_gap"],
            "mean_active_count": eq.aggregates.mean_active_count,
            "wf_reference": eq.participation.wf_reference,
            "converged": bool(eq.converged), "residual": eq.residual,
            "n_func_evals": eq.n_func_evals,
        })
    return pd.DataFrame(rows)


def fixed_input_decomposition(
    me: PooledEquilibrium,
    pe_i: PooledEquilibrium | None = None,
    pe_k: PooledEquilibrium | None = None,
    pe: PooledEquilibrium | None = None,
    *,
    chi_me: float,
    beta: float,
    phi: float,
) -> dict:
    """Consumption-equivalent welfare along ME -> PE_I -> PE_K -> PE.

    ``chi_me`` (the MARKET labor-disutility weight) is held fixed across regimes,
    as in :func:`orbis_model.welfare.welfare_metrics.two_channel_decomposition`.
    Each leg between consecutive supplied regimes is reported as ``lambda_<leg>``
    and ``Delta_<leg> = ln(1 + lambda_<leg>)``; a leg that skips a regime is
    named by joining the skipped leg names (e.g. ``lambda_materials_capital`` for
    PE_I -> PE without PE_K). With PE supplied, ``lambda_total`` is ME -> PE and
    ``identity_residual = prod(1 + lambda_leg) - (1 + lambda_total)``;
    ``Delta_identity_residual`` is the additive analogue.
    """
    chain = _chain(me, pe_i, pe_k, pe)
    order = {label: i for i, label in enumerate(REGIMES)}
    W = {label: steady_state_welfare(eq.C, eq.L_agg, chi_me, beta, phi)
         for label, eq in chain}
    out: dict = {f"W_{label}": float(W[label]) for label, _ in chain}

    lambdas = []
    for (a, _), (b, _) in zip(chain[:-1], chain[1:]):
        steps = REGIMES[order[a]:order[b] + 1]
        name = "_".join(LEGS[pair] for pair in zip(steps[:-1], steps[1:]))
        lam = consumption_equivalent_lambda(W[a], W[b], beta)
        out[f"lambda_{name}"] = float(lam)
        out[f"Delta_{name}"] = float(np.log1p(lam))
        lambdas.append(lam)

    if pe is not None:
        total = consumption_equivalent_lambda(W["market"], W["planner"], beta)
        out["lambda_total"] = float(total)
        out["Delta_total"] = float(np.log1p(total))
        out["identity_residual"] = float(
            np.prod([1.0 + lam for lam in lambdas]) - (1.0 + total)
        )
        out["Delta_identity_residual"] = float(
            sum(np.log1p(lam) for lam in lambdas) - np.log1p(total)
        )

    K_target = float(me.K_agg)
    M_target = float(me.aggregates.M)
    out["K_target"] = K_target
    out["M_target"] = M_target
    for label, eq in chain:
        out[f"R_{label}"] = float(eq.R)
        out[f"P_mat_{label}"] = float(eq.P_mat)
        out[f"K_target_gap_{label}"] = float(eq.K_agg / K_target - 1.0)
        out[f"M_target_gap_{label}"] = float(eq.aggregates.M / M_target - 1.0)
    return out
