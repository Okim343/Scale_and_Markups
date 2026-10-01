"""EMX-style exogenous entry on a masked firm pool.

Market structure is no longer the outcome of a Berry/Bresnahan participation
game. Following Edmond–Midrigan–Xu (2023, §VI), the per-sector firm count is
**exogenous**: each sector draws ``n_m = max(1, Poisson(N))`` producing firms
(see :func:`model.pool.allocate_poisson`), ``N`` is a directly calibrated
parameter, and **all** allocated firms produce. There is no per-period operating
cost and no within-sector profitability cutoff in the static cross-section — the
sunk entry cost lives only in the dynamic free-entry block and is backed out
after calibration (``calibration/entry_cost.py``).

This module therefore just prices a given exogenous active mask: one
:func:`solve_batch` over the Poisson-allocated firms. The :class:`ParticipationResult`
container is kept (with vestigial diagnostic fields) so downstream consumers are
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market_batch import BatchSolution, solve_batch
from .pricing import DEFAULT_PHI_V, PricingRegime


@dataclass(frozen=True)
class ParticipationResult:
    active_mask: np.ndarray
    solution: BatchSolution
    n_active: np.ndarray
    wf_reference: float
    Ed_full_pool: float
    pool_binds: np.ndarray
    empty_market: np.ndarray
    converged: np.ndarray


def allocate_exogenous(
    alpha: np.ndarray,
    v: np.ndarray,
    active_mask: np.ndarray,
    *,
    eta: float,
    gamma: float,
    w: float,
    R: float,
    X_market: float,
    a: float,
    phi_v: float = DEFAULT_PHI_V,
    y_hat: float = 1.0,
    love_of_variety: bool | str = False,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    market_solver_kwargs: dict | None = None,
) -> ParticipationResult:
    """Price an exogenous Poisson active mask — one Cournot solve, no selection.

    Every firm in ``active_mask`` produces; nothing is deleted. ``wf_reference``
    is fixed at ``0.0`` (no per-period operating cost) and the Berry-game
    diagnostic fields (``Ed_full_pool``, ``pool_binds``, ``empty_market``) are
    retained for container compatibility but carry no economic content here.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if alpha.ndim != 2 or v.shape != alpha.shape:
        raise ValueError("alpha and v must have matching shape (M, H)")
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != alpha.shape:
        raise ValueError("active_mask must match the (M, H) draw")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("every sector must have at least one active firm")

    solution = solve_batch(
        alpha, v, eta=eta, gamma=gamma, w=w, R=R, X_market=X_market, a_i=a,
        phi_v=phi_v, y_hat=y_hat, active_mask=mask,
        love_of_variety=love_of_variety,
        regime=regime, mu_bar=mu_bar, **dict(market_solver_kwargs or {}),
    )
    M = mask.shape[0]
    return ParticipationResult(
        active_mask=mask.copy(),
        solution=solution,
        n_active=mask.sum(axis=1),
        wf_reference=0.0,
        Ed_full_pool=float("nan"),
        pool_binds=np.zeros(M, dtype=bool),
        empty_market=np.zeros(M, dtype=bool),
        converged=solution.converged.copy(),
    )
