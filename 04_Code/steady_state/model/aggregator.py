"""Pooled exposure aggregation and pooled cross-sectional moments.

Pure functions on a solved pooled cross section (a :class:`BatchSolution`
plus its active mask):

* :func:`pooled_moments` — cost-weighted aggregate markup, the four
  concentration moments (CR4, CR20, top-1%, top-5%) via the
  percentile-equivalence rule, the mean active count, and the pooled
  inverse-markup/share slope.
* :func:`aggregate_pooled_markets` — the top-level exposure CES over markets
  ``j`` (``model.typ`` eq:agg_ces / eq:agg_price) plus the resource totals,
  including the per-active-firm operating-cost overhead ``|A_j|·f``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market_batch import BatchSolution
from .pricing import DEFAULT_PHI_V


@dataclass(frozen=True)
class PooledAggregate:
    """Exposure and resource aggregates for one pooled market simulation.

    ``Q`` is the top-level gross composite (CES over markets) on which demand
    loads; ``M`` is aggregate materials use; ``C = Q - M`` is net consumption.
    """

    C: float
    P: float
    L: float
    K: float
    profits: float
    mean_active_count: float
    population_markets: float
    population_scale: float
    exposure_weights: np.ndarray
    Q: float = 0.0
    M: float = 0.0


POOLED_MOMENT_KEYS: tuple[str, ...] = (
    "mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
)


def pooled_inverse_markup_regression(
    solution: BatchSolution, active_mask: np.ndarray
) -> dict[str, float]:
    """OLS of inverse markup on within-market share across active firms.

    Firm-level diagnostic only (share spread, etc.). The identifying
    ``emx_slope`` moment is the *sector-level* regression
    (:func:`sector_inverse_markup_regression`).
    """
    mask = np.asarray(active_mask, dtype=bool)
    x = np.asarray(solution.s, dtype=float)[mask]
    y = 1.0 / np.asarray(solution.mu, dtype=float)[mask]
    X = np.column_stack((np.ones(x.size), x))
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(x.size - 2, 1)
    sigma2 = float(resid @ resid / dof)
    covariance = sigma2 * np.linalg.pinv(X.T @ X)
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "slope_se": float(np.sqrt(max(covariance[1, 1], 0.0))),
        "n": int(x.size),
        "rank": int(rank),
        "share_std": float(np.std(x)),
        "share_min": float(np.min(x)),
        "share_max": float(np.max(x)),
    }


def sector_inverse_markup_regression(
    solution: BatchSolution, active_mask: np.ndarray
) -> dict[str, float]:
    r"""OLS of the sector inverse markup on sector HHI across markets (EMX eq. 83).

    For each market (sector) the sales-weighted inverse markup
    ``1/μ_i = Σ_j s_j·(1/μ_j)`` and ``HHI_i = Σ_j s_j²`` are formed over its
    active firms (``s_j`` is the within-market sales share, summing to one). The
    slope of ``1/μ_i ~ HHI_i`` recovers ``−(1/η − 1/γ)``; this is the
    steady-state cross-sector analogue of EMX's Δ-over-time identifying
    regression and what identifies the ``γ–η`` gap.
    """
    mask = np.asarray(active_mask, dtype=bool)
    s = np.where(mask, np.asarray(solution.s, dtype=float), 0.0)
    mu = np.asarray(solution.mu, dtype=float)
    inv_mu = np.divide(1.0, mu, out=np.zeros_like(mu), where=mask)
    inv_mu_sector = (s * inv_mu).sum(axis=1)   # sales-weighted, per market
    hhi = (s * s).sum(axis=1)
    X = np.column_stack((np.ones(hhi.size), hhi))
    coef, _, rank, _ = np.linalg.lstsq(X, inv_mu_sector, rcond=None)
    resid = inv_mu_sector - X @ coef
    dof = max(hhi.size - 2, 1)
    sigma2 = float(resid @ resid / dof)
    covariance = sigma2 * np.linalg.pinv(X.T @ X)
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "slope_se": float(np.sqrt(max(covariance[1, 1], 0.0))),
        "hhi_mean": float(hhi.mean()),
        "hhi_std": float(hhi.std()),
        "rank": int(rank),
    }


def within_sector_concentration(
    sales: np.ndarray, active_mask: np.ndarray
) -> dict[str, float]:
    """Within-sector CR4/CR20/top1pct/top5pct, sales-weighted across sectors.

    Mirrors the empirical construction
    (``03_Empirical/.../02_pooled_targets.py``): per market, sort active-firm
    sales descending, take ``cum_share`` at ranks 4, 20, ``ceil(0.01·n)`` and
    ``ceil(0.05·n)``; then average across markets weighted by market sales,
    matching the EMX sector-size weighting. Inactive slots carry zero sales, so
    ``cum_share`` at a rank past ``n_m`` equals one — reproducing the
    ``min(k-1, n-1)`` clamp exactly.
    """
    sales = np.asarray(sales, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)
    M, H = sales.shape
    n = mask.sum(axis=1)
    s = np.where(mask, sales, 0.0)
    total = s.sum(axis=1)
    desc = -np.sort(-s, axis=1)
    csum = np.cumsum(desc, axis=1)
    rows = np.arange(M)
    # Zero-padding makes csum at a rank past n_m equal the sector total, so a
    # fixed rank index reproduces the empirical min(k-1, n-1) clamp; index is
    # only capped at H-1 to stay in-bounds when the pool itself is tiny.
    idx4 = min(3, H - 1)
    idx20 = min(19, H - 1)
    k1 = np.minimum(np.maximum(1, np.ceil(0.01 * n).astype(np.int64)), H)
    k5 = np.minimum(np.maximum(1, np.ceil(0.05 * n).astype(np.int64)), H)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr4_m = csum[:, idx4] / total
        cr20_m = csum[:, idx20] / total
        top1_m = csum[rows, k1 - 1] / total
        top5_m = csum[rows, k5 - 1] / total
    w = total.astype(float)
    wsum = float(w.sum())

    def wavg(v: np.ndarray) -> float:
        return float(np.dot(w, np.nan_to_num(v)) / wsum)

    return {
        "cr4": wavg(cr4_m), "cr20": wavg(cr20_m),
        "top1pct": wavg(top1_m), "top5pct": wavg(top5_m),
    }


def pooled_moments(
    solution: BatchSolution,
    active_mask: np.ndarray,
    *,
    n_firms_cs: float | None = None,
    phi_v: float = DEFAULT_PHI_V,
) -> dict[str, float]:
    """Calibration moments from one simulated cross section of sectors.

    Concentration moments are computed *within sector* then sector-sales-weighted
    across sectors (EMX §VI basis). ``mu_cw`` is the pure firm-markup
    aggregate ``Σ TC_i·μ_i / Σ TC_i`` used for calibration against the
    DLEU/DLW-style empirical markup target. ``mu_cw_alpha`` is the pooled
    accounting markup, equal to aggregate revenue / aggregate variable cost
    (``Σ λc_i·μ_i/α_i`` with ``λc`` the share of variable cost; model.typ
    @eq:mu_cw_identity). Because ``solution.cost`` holds true total variable cost
    ``TC = α·p·y/μ``, this is simply ``Σ sales / Σ cost``. ``emx_slope`` is the
    sector-level ``1/μ_i ~ HHI_i`` slope. ``materials_share`` is a validation
    diagnostic ``(1-phi_v)/mu_cw_alpha`` (data ≈ 0.506), not a calibration target.
    ``mean_active_count`` is reported as a diagnostic (it is exogenous = ~N under
    EMX entry). ``n_firms_cs`` is accepted for signature compatibility but unused.
    """
    mask = np.asarray(active_mask, dtype=bool)
    sales_all = np.asarray(solution.sales, dtype=float)
    sales = sales_all[mask]
    cost = np.asarray(solution.cost, dtype=float)[mask]
    mu = np.asarray(solution.mu, dtype=float)[mask]
    conc = within_sector_concentration(sales_all, mask)
    sector = sector_inverse_markup_regression(solution, mask)
    firm = pooled_inverse_markup_regression(solution, mask)
    mu_cw_alpha = float(np.sum(sales) / np.sum(cost))
    mu_cw = float(np.sum(cost * mu) / np.sum(cost))
    return {
        "mu_cw": mu_cw,
        "mu_cw_alpha": mu_cw_alpha,
        **conc,
        "mean_active_count": float(mask.sum(axis=1).mean()),
        "emx_slope": sector["slope"],
        "emx_intercept": sector["intercept"],
        "share_std": firm["share_std"],
        "share_range": firm["share_max"] - firm["share_min"],
        "materials_share": float((1.0 - phi_v) / mu_cw_alpha),
    }


def exposure_weights(Q: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Return normalized market exposure weights, uniform by default."""
    if Q <= 0:
        raise ValueError("Q must be positive")
    if weights is None:
        return np.full(Q, 1.0 / Q)
    omega = np.asarray(weights, dtype=np.float64)
    if omega.shape != (Q,) or np.any(omega < 0.0) or omega.sum() <= 0.0:
        raise ValueError(f"weights must be a non-negative vector of shape ({Q},)")
    return omega / omega.sum()


def aggregate_pooled_markets(
    solution: BatchSolution,
    active_mask: np.ndarray,
    *,
    eta: float,
    operating_cost_labor: float,
    n_firms_cs: float,
    w: float,
    R: float,
    a: float,
    phi_v: float = DEFAULT_PHI_V,
    weights: np.ndarray | None = None,
) -> PooledAggregate:
    """Aggregate active firms and normalized market bundles without sectors.

    Gross-output closure (model.typ §1.5): the CES over markets produces a
    sample gross composite that is scaled to the same population units as costs.
    Only the ``phi_v`` fraction of total variable cost is a primary-factor
    payment (``wL + RK = phi_v·TC``), the ``1-phi_v`` fraction buys materials
    ``M = (1-phi_v)·TC`` (at ``P=1``); net consumption is ``C = Q - M``.
    As ``phi_v → 1`` materials vanish and ``C → Q``.
    """
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != solution.s.shape:
        raise ValueError(f"active_mask must have shape {solution.s.shape}")
    if eta <= 1.0:
        raise ValueError("eta must exceed one")
    if operating_cost_labor < 0.0 or n_firms_cs <= 0.0 or w <= 0.0 or R <= 0.0:
        raise ValueError("costs, population, and factor prices must be valid")
    if not 0.0 < a < 1.0:
        raise ValueError("a must lie strictly between zero and one")
    if not 0.0 < phi_v <= 1.0:
        raise ValueError("phi_v must lie in (0, 1]")

    n_markets = mask.shape[0]
    omega = exposure_weights(n_markets, weights)
    Y = np.asarray(solution.Y_j_normalized, dtype=np.float64)
    P_j = np.asarray(solution.P_j_normalized, dtype=np.float64)
    # Top-level gross composite Q (demand loads on Q, not on consumption).
    Q_sample = float(
        np.sum(omega ** (1.0 / eta) * Y ** ((eta - 1.0) / eta))
        ** (eta / (eta - 1.0))
    )
    P = float(np.sum(omega * P_j ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))

    n_active = mask.sum(axis=1)
    n_bar = float(n_active.mean())
    if n_bar <= 0.0:
        raise ValueError("pooled mean active count must be positive")
    population_markets = float(n_firms_cs / n_bar)
    scale = population_markets / n_markets
    Q = scale * Q_sample

    cost_sample = float(solution.cost[mask].sum())  # Σ TC over simulated markets
    variable_cost = cost_sample * scale             # population-scaled Σ TC
    overhead_labor = float(n_active.sum()) * operating_cost_labor * scale
    # Only phi_v of variable cost is a primary-factor payment (wL + RK = φv·TC);
    # the 1-φv fraction buys materials of the composite (P=1). Factor demands
    # keep the original population scaling.
    primary_factor_cost = phi_v * variable_cost
    L = (1.0 - a) * primary_factor_cost / w + overhead_labor
    K = a * primary_factor_cost / R
    # The resource constraint C = Q - M uses the same population scale as factor
    # payments and profits. As phi_v → 1, M → 0 and C → Q.
    M = (1.0 - phi_v) * variable_cost
    C = Q - M
    variable_profit = float((solution.sales[mask] - solution.cost[mask]).sum()) * scale
    profits = variable_profit - w * overhead_labor
    return PooledAggregate(
        C=C,
        P=P,
        L=float(L),
        K=float(K),
        profits=float(profits),
        mean_active_count=n_bar,
        population_markets=population_markets,
        population_scale=float(scale),
        exposure_weights=omega,
        Q=Q,
        M=float(M),
    )


# ---------------------------------------------------------------------------
# concentration moments — percentile equivalence
# ---------------------------------------------------------------------------


def concentration_share(sales: np.ndarray, percentile: float) -> float:
    """Cumulative sales share of the top ``percentile`` fraction of firms.

    Sort sales descending; return the cumulative share of the first
    ``ceil(percentile * len(sales))`` firms. Equivalent to a quantile-based
    Lorenz-curve evaluation at $1 - p$.
    """
    if percentile <= 0:
        return 0.0
    if percentile >= 1:
        return 1.0
    n = sales.size
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(percentile * n)))
    # Partial selection is O(n) — avoids a full sort.
    idx = np.argpartition(sales, n - k)[n - k:]
    return float(sales[idx].sum() / sales.sum())
