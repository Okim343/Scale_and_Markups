"""Steady-state welfare and fixed-participation two-channel decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.aggregator import pooled_moments
from ..model.normalization import PooledEquilibrium, PooledParams, solve_pooled_ge
from ..model.pricing import PricingRegime


def steady_state_welfare(C: float, L: float, chi: float, beta: float, phi: float) -> float:
    exponent = 1.0 + 1.0 / phi
    return float((np.log(C) - chi * L ** exponent / exponent) / (1.0 - beta))


def consumption_equivalent_lambda(W_me: float, W_pe: float, beta: float) -> float:
    return float(np.expm1((1.0 - beta) * (W_pe - W_me)))


def aggregate_markup_cw(eq: PooledEquilibrium, *, n_firms_cs: float) -> float:
    """Cost-weighted PURE aggregate markup ``Σ TC·μ / Σ TC`` of an equilibrium.

    This is ``pooled_moments[...]["mu_cw"]`` — the pure price/MC wedge aggregate,
    NOT the accounting object ``mu_cw_alpha`` (= Σsales/Σcost = μ/α). It is the
    correct common markup for the UNIFORM counterfactual, whose
    ``markups_from_shares`` sets every firm's pure markup μ ≡ μ̄. (Pre gross-output
    migration the accounting object and the pure markup coincided because cost was
    MC·y; they diverge once cost holds true total variable cost α·p·y/μ.)
    """
    return pooled_moments(
        eq.participation.solution, eq.participation.active_mask,
        n_firms_cs=n_firms_cs,
    )["mu_cw"]


def welfare_decomposition_table(
    me: PooledEquilibrium,
    pe: PooledEquilibrium,
    uniform: PooledEquilibrium,
    *,
    n_firms_cs: float,
) -> pd.DataFrame:
    rows = []
    for label, eq in (("market", me), ("uniform", uniform), ("planner", pe)):
        m = pooled_moments(
            eq.participation.solution, eq.participation.active_mask,
            n_firms_cs=n_firms_cs,
        )
        rows.append({
            "regime": label, "C": eq.C, "L": eq.L_agg, "K": eq.K_agg,
            "profits": eq.Pi_agg,
            "mu_cw": m["mu_cw"],              # pure cost-weighted markup
            "mu_cw_alpha": m["mu_cw_alpha"],  # accounting Σsales/Σcost = μ/α
            "mean_active_count": eq.aggregates.mean_active_count,
            "wf_reference": eq.participation.wf_reference,
        })
    return pd.DataFrame(rows)


def two_channel_decomposition(
    me: PooledEquilibrium,
    pe: PooledEquilibrium,
    *,
    params: PooledParams,
    chi_me: float,
    beta: float,
    phi: float,
    delta_K: float = 0.06,
    n_firms_cs: float = 1_000.0,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
    tol: float = 1e-5,
    max_nfev: int = 30,
) -> dict:
    """Insert a uniform-markup allocation on the same draws and active mask."""
    mu_bar = aggregate_markup_cw(me, n_firms_cs=n_firms_cs)
    uniform = solve_pooled_ge(
        me.alpha, me.v, params, beta=beta, delta_K=delta_K,
        n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
        love_of_variety=love_of_variety, regime=PricingRegime.UNIFORM,
        mu_bar=mu_bar, market_solver_kwargs=market_solver_kwargs,
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask,
        initial=(me.w, me.X_market), tol=tol, max_nfev=max_nfev,
    )
    W_me = steady_state_welfare(me.C, me.L_agg, chi_me, beta, phi)
    W_u = steady_state_welfare(uniform.C, uniform.L_agg, chi_me, beta, phi)
    W_pe = steady_state_welfare(pe.C, pe.L_agg, chi_me, beta, phi)
    total = consumption_equivalent_lambda(W_me, W_pe, beta)
    dispersion = consumption_equivalent_lambda(W_me, W_u, beta)
    level = consumption_equivalent_lambda(W_u, W_pe, beta)
    identity_residual = (1.0 + level) * (1.0 + dispersion) - (1.0 + total)
    return {
        "W_market": W_me, "W_uniform": W_u, "W_planner": W_pe,
        "lambda_total": total, "lambda_level": level,
        "lambda_dispersion": dispersion, "identity_residual": identity_residual,
        "mu_bar": mu_bar, "uniform": uniform,
    }
