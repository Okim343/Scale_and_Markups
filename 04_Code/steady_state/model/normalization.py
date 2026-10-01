"""Pooled-market normalization, participation, and steady-state closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .aggregator import PooledAggregate, aggregate_pooled_markets
from .participation import ParticipationResult, allocate_exogenous
from .pricing import DEFAULT_PHI_V, PricingRegime


@dataclass(frozen=True)
class PooledParams:
    """The common calibrated vector plus externally assigned pooled inputs.

    EMX-style entry: ``xi`` is the Pareto capability tail index and ``N`` the
    Poisson mean firms per sector. They replace the old ``(sigma_z, phi_f)``.

    ``v_min`` is the anchoring scale solved by the baseline market condition;
    ``y_hat`` is the frozen output anchor, normalized to one in production runs.
    ``phi_v`` (the gross-output value-added weight) and ``a`` are externally
    assigned primitives, **not** part of the calibrated vector.
    """

    xi: float
    N: float
    gamma: float
    eta: float
    v_min: float = 1.0
    y_hat: float = 1.0
    rho_bar: float = 0.0
    a: float = 1.0 / 3.0
    H: int = 900
    phi_v: float = DEFAULT_PHI_V

    def __post_init__(self) -> None:
        if not self.xi > 1.0:
            raise ValueError("Pareto tail index xi must exceed 1")
        if not self.v_min > 0.0:
            raise ValueError("v_min must be positive")
        if not self.y_hat > 0.0:
            raise ValueError("y_hat must be positive")
        if self.N < 1.0:
            raise ValueError("Poisson mean N must be at least 1")
        if not 1.0 < self.eta < self.gamma:
            raise ValueError("elasticities must satisfy 1 < eta < gamma")
        if not 0.0 <= self.rho_bar < 1.0:
            raise ValueError("rho_bar must lie in [0, 1)")
        if not 0.0 < self.a < 1.0 or self.H <= 0:
            raise ValueError("a and H are invalid")
        if self.N > self.H:
            raise ValueError("Poisson mean N must not exceed the pool size H")
        if not 0.0 < self.phi_v <= 1.0:
            raise ValueError("value-added weight phi_v must lie in (0, 1]")



@dataclass(frozen=True)
class FullPoolReference:
    w: float
    R: float
    X_market: float
    wf_reference: float
    Ed_full_pool: float
    P: float
    L: float
    converged: bool
    residual: float


@dataclass(frozen=True)
class PooledEquilibrium:
    participation: ParticipationResult
    aggregates: PooledAggregate
    operating_cost_labor: float
    alpha: np.ndarray | None = None
    v: np.ndarray | None = None
    tilde_alpha: np.ndarray | None = None
    w: float = 1.0
    R: float = 0.1
    X_market: float = 1.0
    C: float = 1.0
    chi: float = 1.0
    regime: PricingRegime = PricingRegime.MARKET
    converged: bool = True
    residual: float = 0.0
    n_func_evals: int = 1
    v_min: float = 1.0
    y_hat: float = 1.0
    y_sw: float = 1.0        # sales-weighted mean active output (diagnostic)
    y_anchor: float = 1.0    # anchoring statistic: median active output

    @property
    def P_agg(self) -> float:
        return self.aggregates.P

    @property
    def L_agg(self) -> float:
        return self.aggregates.L

    @property
    def K_agg(self) -> float:
        return self.aggregates.K

    @property
    def Pi_agg(self) -> float:
        return self.aggregates.profits

    def walras_check(self) -> dict[str, float]:
        income = self.w * self.L_agg + self.R * self.K_agg + self.Pi_agg
        gap = abs(self.C - income)
        return {"C": self.C, "income": income, "abs_gap": gap,
                "rel_gap": gap / max(abs(self.C), 1e-30)}



def euler_R(beta: float, delta_K: float) -> float:
    return 1.0 / beta - (1.0 - delta_K)


def solve_pooled_economy(
    alpha: np.ndarray,
    v: np.ndarray,
    params: PooledParams,
    *,
    w: float,
    R: float,
    X_market: float,
    n_firms_cs: float,
    active_mask: np.ndarray,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
) -> PooledEquilibrium:
    """One pooled forward pass pricing the exogenous Poisson active mask.

    ``active_mask`` is the EMX-style exogenous entry mask (see
    :func:`model.pool.allocate_poisson`); every firm in it produces. ``K_switch``
    is accepted for signature compatibility but unused (there is no selection).
    """
    alpha = np.asarray(alpha, dtype=float)
    v = np.asarray(v, dtype=float)
    if alpha.ndim != 2 or v.shape != alpha.shape or alpha.shape[1] != params.H:
        raise ValueError("alpha and v must have shape (M, params.H)")
    if active_mask is None:
        raise ValueError("active_mask (exogenous Poisson entry) is required")
    participation = allocate_exogenous(
        alpha, v, np.asarray(active_mask, dtype=bool),
        eta=params.eta, gamma=params.gamma, w=w, R=R, X_market=X_market,
        a=params.a, phi_v=params.phi_v, y_hat=params.y_hat,
        love_of_variety=love_of_variety,
        regime=regime, mu_bar=mu_bar, market_solver_kwargs=market_solver_kwargs,
    )
    operating_cost_labor = participation.wf_reference / w
    aggregates = aggregate_pooled_markets(
        participation.solution, participation.active_mask, eta=params.eta,
        operating_cost_labor=operating_cost_labor, n_firms_cs=n_firms_cs,
        w=w, R=R, a=params.a, phi_v=params.phi_v, weights=exposure,
    )
    active = participation.active_mask
    sales_active = participation.solution.sales[active]
    output_active = participation.solution.output[active]
    y_sw = float(np.sum(sales_active * output_active) / np.sum(sales_active))
    # Anchoring statistic: median active output. A right-skewed size
    # distribution puts the sales-weighted mean (y_sw) deep in the tail, so
    # ~all firms sit below it in the region where alpha is a size penalty
    # (MC ~ (y/y_hat)^(1/alpha-1)); anchoring at the median instead splits
    # firms ~evenly around y_hat. See 02_Drafts/md_files/scale_issue_experiments.md.
    y_anchor = float(np.median(output_active)) if output_active.size else 1.0
    return PooledEquilibrium(
        participation=participation, aggregates=aggregates,
        operating_cost_labor=operating_cost_labor, alpha=alpha, v=v,
        w=w, R=R, X_market=X_market, C=aggregates.C,
        chi=w / max(aggregates.C, 1e-30), regime=regime,
        converged=bool(np.all(participation.converged)),
        v_min=params.v_min, y_hat=params.y_hat, y_sw=y_sw, y_anchor=y_anchor,
    )


def solve_full_pool_reference(
    alpha: np.ndarray,
    v: np.ndarray,
    params: PooledParams,
    *,
    beta: float = 0.96,
    delta_K: float = 0.06,
    n_firms_cs: float = 1_000.0,
    exposure: np.ndarray | None = None,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
    tol: float = 1e-6,
    max_nfev: int = 40,
) -> FullPoolReference:
    """GE-normalize the full-pool allocation (no operating cost; ``wf=0``).

    Retained for diagnostics. The main calibration/simulation path builds its
    reference directly from the GE solve on the exogenous Poisson mask; under
    EMX entry there is no per-period operating cost, so ``wf_reference`` is 0.
    """
    from .market_batch import solve_batch

    alpha = np.asarray(alpha, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = np.ones_like(alpha, dtype=bool)
    R = euler_R(beta, delta_K)

    def evaluate(x: np.ndarray):
        w, X = np.exp(x)
        solution = solve_batch(
            alpha, v, eta=params.eta, gamma=params.gamma, w=w, R=R,
            X_market=X, a_i=params.a, phi_v=params.phi_v, active_mask=mask,
            y_hat=params.y_hat, love_of_variety=love_of_variety,
            **dict(market_solver_kwargs or {}),
        )
        Ed = float(np.mean(solution.sales - solution.cost))
        wf = 0.0
        agg = aggregate_pooled_markets(
            solution, mask, eta=params.eta,
            operating_cost_labor=wf / w, n_firms_cs=n_firms_cs,
            w=w, R=R, a=params.a, phi_v=params.phi_v, weights=exposure,
        )
        return solution, Ed, wf, agg

    def residual(x: np.ndarray) -> np.ndarray:
        _, _, _, agg = evaluate(x)
        return np.log([agg.P, agg.L])

    sol = least_squares(
        residual, np.zeros(2), method="trf", ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_nfev, diff_step=1e-3,
    )
    solution, Ed, wf, agg = evaluate(sol.x)
    resid = float(np.max(np.abs(np.log([agg.P, agg.L]))))
    return FullPoolReference(
        w=float(np.exp(sol.x[0])), R=R, X_market=float(np.exp(sol.x[1])),
        wf_reference=wf, Ed_full_pool=Ed, P=agg.P, L=agg.L,
        converged=bool(sol.success and np.all(solution.converged) and resid <= 10 * tol),
        residual=resid,
    )


def solve_pooled_ge(
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
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    active_mask: np.ndarray,
    initial: tuple[float, float] | None = None,
    tol: float = 1e-5,
    max_nfev: int = 30,
) -> PooledEquilibrium:
    """Solve pooled ``P=1,L=1`` GE on the exogenous Poisson active mask.

    ``active_mask`` is mandatory (EMX-style exogenous entry); there is no
    selection. ``wf_reference`` defaults to ``0.0`` (no per-period operating
    cost). ``K_switch`` is accepted but unused.
    """
    R = euler_R(beta, delta_K)
    if wf_reference is None:
        wf_reference = 0.0
    if initial is None:
        initial = (1.0, 1.0)

    selected_mask = np.asarray(active_mask, dtype=bool)

    def evaluate(x: np.ndarray, mask: np.ndarray) -> PooledEquilibrium:
        w, X = np.exp(x)
        return solve_pooled_economy(
            alpha, v, params, w=w, R=R, X_market=X,
            n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
            love_of_variety=love_of_variety, regime=regime, mu_bar=mu_bar,
            market_solver_kwargs=market_solver_kwargs,
            wf_reference=wf_reference, active_mask=mask,
        )

    x = np.log(initial)

    def residual(value: np.ndarray) -> np.ndarray:
        candidate = evaluate(value, selected_mask)
        return np.log([candidate.P_agg, candidate.L_agg])

    sol = least_squares(
        residual, x, method="trf", ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_nfev, diff_step=1e-3,
    )
    total_nfev = int(sol.nfev)
    x = sol.x

    eq = evaluate(x, selected_mask)
    resid = float(np.max(np.abs(np.log([eq.P_agg, eq.L_agg]))))
    return PooledEquilibrium(
        **{**eq.__dict__, "chi": eq.w / max(eq.C, 1e-30),
           "converged": bool(sol.success and eq.converged and resid <= 10 * tol),
           "residual": resid, "n_func_evals": total_nfev}
    )


def solve_pooled_ge_anchored(
    alpha: np.ndarray,
    v_unit: np.ndarray,
    params: PooledParams,
    *,
    beta: float = 0.96,
    delta_K: float = 0.06,
    n_firms_cs: float = 1_000.0,
    exposure: np.ndarray | None = None,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    regime: PricingRegime = PricingRegime.MARKET,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    active_mask: np.ndarray,
    initial: tuple[float, float, float] | None = None,
    tol: float = 1e-5,
    max_nfev: int = 30,
) -> PooledEquilibrium:
    """Baseline market GE with anchoring condition solving ``v_min``.

    Unknowns are ``(log w, log X_market, log v_min)``; the third residual drives
    the anchoring statistic to ``y_hat``. The anchor is the **median** active
    output (``eq.y_anchor``), not the sales-weighted mean (``eq.y_sw``): the
    median splits firms ~evenly around ``y_hat`` so alpha is not a size penalty
    for ~all of them (see :func:`solve_pooled_economy` and
    ``02_Drafts/md_files/scale_issue_experiments.md``). The plain
    :func:`solve_pooled_ge` remains the frozen-technology entry point for planner,
    uniform-markup, simulation, and welfare paths.
    """
    if regime is not PricingRegime.MARKET:
        raise ValueError("anchored GE solve is only valid for MARKET pricing")
    R = euler_R(beta, delta_K)
    if wf_reference is None:
        wf_reference = 0.0
    if initial is None:
        initial = (1.0, 1.0, params.v_min)
    selected_mask = np.asarray(active_mask, dtype=bool)

    def evaluate(x: np.ndarray) -> PooledEquilibrium:
        w, X, v_min = np.exp(x)
        scaled_params = PooledParams(
            xi=params.xi, N=params.N, gamma=params.gamma, eta=params.eta,
            v_min=float(v_min), y_hat=params.y_hat, rho_bar=params.rho_bar,
            a=params.a, H=params.H, phi_v=params.phi_v,
        )
        return solve_pooled_economy(
            alpha, v_min * v_unit, scaled_params, w=w, R=R, X_market=X,
            n_firms_cs=n_firms_cs, exposure=exposure, K_switch=K_switch,
            love_of_variety=love_of_variety, regime=PricingRegime.MARKET,
            market_solver_kwargs=market_solver_kwargs,
            wf_reference=wf_reference, active_mask=selected_mask,
        )

    x0 = np.log(initial)

    def residual(value: np.ndarray) -> np.ndarray:
        candidate = evaluate(value)
        return np.log([candidate.P_agg, candidate.L_agg,
                       candidate.y_anchor / params.y_hat])

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
