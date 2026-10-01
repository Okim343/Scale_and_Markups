"""Fixed-capital-envelope planner solve.

PLANNER pricing (:math:`\\mu \\equiv 1`) with aggregate capital ``K`` pinned to a
target level (the MARKET equilibrium's realized ``K``) by promoting the rental
rate ``R`` to a *third* solved unknown, instead of fixing it at the Euler rate
``euler_R(beta, delta_K)``. This isolates a "reallocation-only" welfare loss
from the much larger "scale" effect that comes from letting capital re-optimize
freely at the fixed Euler rate.

The forward pass (:func:`steady_state.model.normalization.solve_pooled_economy`)
is imported and used unchanged: ``R`` already feeds every firm's marginal cost
via ``_omega_gross(w, R, a_i, phi_v)`` inside the Cournot/planner fixed point,
so making ``R`` a free unknown genuinely re-prices every firm — it is a new GE
object, not a relabeling. Nothing under ``steady_state/`` is modified.

This function is modeled directly on
:func:`steady_state.model.normalization.solve_pooled_ge_anchored`: a 3-unknown
``least_squares`` wrapping the unchanged forward pass, where ``R`` replaces the
anchored ``v_min`` slot and the third residual drives ``K_agg`` to ``K_target``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from steady_state.model.normalization import (
    PooledEquilibrium,
    PooledParams,
    euler_R,
    solve_pooled_economy,
)
from steady_state.model.pricing import PricingRegime


def solve_fixed_capital_planner(
    alpha: np.ndarray,
    v: np.ndarray,
    params: PooledParams,
    *,
    K_target: float,
    beta: float = 0.96,
    delta_K: float = 0.06,
    n_firms_cs: float = 1_000.0,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    regime: PricingRegime = PricingRegime.PLANNER,
    mu_bar: float | None = None,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    active_mask: np.ndarray,
    initial: tuple[float, float, float] | None = None,
    tol: float = 1e-5,
    max_nfev: int = 40,
) -> PooledEquilibrium:
    """Solve pooled GE with ``P=1``, ``L=1``, and ``K=K_target``.

    Unknowns are ``(log w, log X_market, log R)``; the third residual drives the
    realized aggregate capital ``K_agg`` to ``K_target``. ``R`` therefore rises
    above the Euler rate to choke capital demand down to the pinned level (the
    target ``K`` is smaller than a freely-optimizing planner's ``K``).

    Kept generic over ``regime``/``mu_bar`` (costs nothing; only ``PLANNER`` is
    used today, but this permits a future "fixed-K UNIFORM" robustness check
    without touching this function again). No new :class:`PricingRegime` member
    is added — this is a PLANNER-priced equilibrium with a non-Euler ``R``,
    distinguished purely by the free-text ``regime`` label used downstream and
    by which output files it lands in.
    """
    if wf_reference is None:
        wf_reference = 0.0
    if initial is None:
        initial = (1.0, 1.0, euler_R(beta, delta_K))
    selected_mask = np.asarray(active_mask, dtype=bool)

    def evaluate(x: np.ndarray) -> PooledEquilibrium:
        w, X, R = np.exp(x)
        return solve_pooled_economy(
            alpha, v, params, w=w, R=R, X_market=X,
            n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
            love_of_variety=love_of_variety, regime=regime, mu_bar=mu_bar,
            market_solver_kwargs=market_solver_kwargs,
            wf_reference=wf_reference, active_mask=selected_mask,
        )

    x0 = np.log(initial)

    def residual(value: np.ndarray) -> np.ndarray:
        candidate = evaluate(value)
        return np.log([candidate.P_agg, candidate.L_agg,
                       candidate.K_agg / K_target])

    sol = least_squares(
        residual, x0, method="trf", ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_nfev, diff_step=1e-3, x_scale=np.ones(3),
    )
    eq = evaluate(sol.x)
    resid = float(np.max(np.abs(residual(sol.x))))
    return PooledEquilibrium(
        **{**eq.__dict__, "chi": eq.w / max(eq.C, 1e-30),
           "converged": bool(sol.success and eq.converged and resid <= 10 * tol),
           "residual": resid, "n_func_evals": int(sol.nfev)}
    )
