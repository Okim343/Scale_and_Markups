"""Market and planner allocations on identical draws and participation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.normalization import PooledEquilibrium, PooledParams, solve_pooled_ge
from ..model.pricing import PricingRegime


@dataclass(frozen=True)
class AllocationPair:
    me: PooledEquilibrium
    pe: PooledEquilibrium


def solve_market_and_planner(
    alpha: np.ndarray,
    v: np.ndarray,
    params: PooledParams,
    *,
    beta: float = 0.96,
    delta_K: float = 0.06,
    n_firms_cs: float = 1_000.0,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    active_mask: np.ndarray,
    initial: tuple[float, float] | None = None,
    tol: float = 1e-5,
    max_nfev: int = 30,
) -> AllocationPair:
    """Compute ``W_Planner(A_Market)-W_Market(A_Market)`` inputs.

    ``active_mask`` is the exogenous Poisson entry mask shared by both regimes.
    """
    me = solve_pooled_ge(
        alpha, v, params, beta=beta, delta_K=delta_K,
        n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
        love_of_variety=love_of_variety, regime=PricingRegime.MARKET,
        market_solver_kwargs=market_solver_kwargs, wf_reference=wf_reference,
        active_mask=active_mask, initial=initial,
        tol=tol, max_nfev=max_nfev,
    )
    pe = solve_pooled_ge(
        alpha, v, params, beta=beta, delta_K=delta_K,
        n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
        love_of_variety=love_of_variety, regime=PricingRegime.PLANNER,
        market_solver_kwargs=market_solver_kwargs,
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask,
        initial=(me.w, me.X_market), tol=tol, max_nfev=max_nfev,
    )
    return AllocationPair(me=me, pe=pe)
