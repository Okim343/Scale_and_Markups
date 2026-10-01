"""Inner Cournot fixed point — one product market.

Solves the joint $(\\mathbf{s}, Y_{im})$ fixed point described in
``solver_scaffold.md`` §3.1. Implements the gross-output system from
``model.typ`` (eqs. @eq:firm_demand–@eq:profits):

.. math::

    y_j         = (p_j/P_{im})^{-\\gamma}\\, Y_{im} \\\\
    1/\\mu_j   = 1 - 1/\\gamma - (1/\\eta - 1/\\gamma)\\, s_j \\\\
    p_j         = \\mu_j\\, MC_j(y_j) \\\\
    MC_j(y)    = \\Omega^g(w,R,P)/v_j \\cdot (y_j/\\hat y)^{1/\\alpha_j - 1}

where the gross-output unit cost $\\Omega^g$ adds the materials layer to the
value-added cost $\\Omega$ (see :func:`_omega_gross`). The capability primitive
``v`` is inverse marginal cost at the common output anchor ``y_hat``. Total
variable cost is $TC_j = \\alpha_j\\, MC_j\\, y_j$ and profit is
$(1 - \\alpha_j/\\mu_j)\\, p_j y_j$.

with the level closure $\\sum_j p_j y_j = X_{market}$ (equivalently
$P_{im} Y_{im} = X_{market}$ under CES). Heterogeneous $\\alpha_j$ means the
system is **not** scale-free: levels enter shares through the curvature term
$y^{1/\\alpha_j-1}$, so $Y_{im}$ has to be solved jointly with the share vector.

Algorithm: damped fixed-point iteration on $(\\mathbf{s}, Y_{im})$ with a
2-block Newton fallback after a configurable number of stalled iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import pricing
from .pricing import DEFAULT_PHI_V, PricingRegime


@dataclass(frozen=True)
class MarketSolution:
    """Solved single-market equilibrium.

    All arrays have length ``n``. ``Y_im`` and ``P_im`` are scalars.
    """

    y: np.ndarray              # firm output
    p: np.ndarray              # firm price
    s: np.ndarray              # within-market sales share (sums to 1)
    mu: np.ndarray             # markup
    d: np.ndarray              # profits = (1 - alpha/mu) * p * y
    cost: np.ndarray           # total variable cost TC = alpha * p*y/mu
    Y_im: float                # market output index (CES)
    P_im: float                # market price index (CES)
    Y_j_normalized: float      # variety-normalized market output index
    P_j_normalized: float      # variety-normalized market price index
    converged: bool
    iterations: int
    used_newton: bool
    residual: float            # max(|delta s|) + |delta log Y_im| at exit


# ---------------------------------------------------------------------------
# core helpers
# ---------------------------------------------------------------------------


def _omega(w: float, R: float, a_i: float) -> float:
    """Value-added unit cost index Ω(w, R) — model.typ @eq:omega."""
    return (R / a_i) ** a_i * (w / (1.0 - a_i)) ** (1.0 - a_i)


def _omega_gross(
    w: float, R: float, a_i: float, phi_v: float, P: float = 1.0
) -> float:
    """Gross-output unit cost Ω^g(w, R, P) — model.typ @eq:omega_gross.

    Cost-minimizing the constant-returns Cobb-Douglas bundle of the value-added
    composite (weight ``phi_v``) and materials (weight ``1 - phi_v``, priced at
    ``P``) gives ``Ω^g = (Ω/phi_v)^phi_v · (P/(1-phi_v))^(1-phi_v)``. As
    ``phi_v → 1`` this collapses to the value-added cost ``Ω`` (materials drop
    out), recovering the value-added-only case.
    """
    omega_va = _omega(w, R, a_i)
    # phi_v == 1 is the value-added-only limit: the materials weight 1-phi_v is
    # zero and the materials factor (P/(1-phi_v))^(1-phi_v) → 1, so Ω^g = Ω.
    if phi_v >= 1.0:
        return omega_va
    return (omega_va / phi_v) ** phi_v * (P / (1.0 - phi_v)) ** (1.0 - phi_v)


def _variety_normalization_factor(
    n: int, gamma: float, love_of_variety: bool | str
) -> float:
    """Return the post-solve price rescaling for the variety toggle."""
    if isinstance(love_of_variety, str):
        value = love_of_variety.strip().lower()
        if value not in {"on", "off"}:
            raise ValueError("love_of_variety must be a bool or 'on'/'off'")
        enabled = value == "on"
    elif isinstance(love_of_variety, (bool, np.bool_)):
        enabled = bool(love_of_variety)
    else:
        raise ValueError("love_of_variety must be a bool or 'on'/'off'")
    return 1.0 if enabled else float(n ** (gamma / (gamma - 1.0)))


def _markups_from_shares(
    s: np.ndarray, eta: float, gamma: float, mu_max: float = 1e6
) -> np.ndarray:
    """μ_j from the EMX inverse-markup rule (decentralized MARKET regime).

    Thin wrapper around :func:`pricing.markups_from_shares` at the MARKET
    regime, kept for backward-compatible imports and unchanged arithmetic.
    Clips 1/μ from below at 1/mu_max to keep things sane near the boundary
    s_max = η(γ-1)/(γ-η) where 1/μ → 0. For (γ, η) = (3.8, 2.0) this boundary
    is s ≈ 3.11, well outside the simplex, but heterogeneous draws can drive
    a single firm's share close to 1, so we keep the safety net.
    """
    return pricing.markups_from_shares(
        s, eta, gamma, PricingRegime.MARKET, mu_max=mu_max
    )


def _prices_from_state(
    s: np.ndarray,
    Y_im: float,
    alpha: np.ndarray,
    v: np.ndarray,
    eta: float,
    gamma: float,
    Omega: float,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given (s, Y_im), evaluate (y, MC, p) at one shot.

    Uses the CES identity y_j / Y_im = (p_j/P_im)^{-γ} = s_j^{γ/(γ-1)} so that
    y_j = s_j^{γ/(γ-1)} Y_im — no need to know P_im at this step.

    ``regime`` selects the pricing rule via :func:`pricing.markups_from_shares`
    (MARKET = EMX markup, PLANNER = p = MC, UNIFORM = common μ̄). Everything
    else is regime-agnostic.
    """
    # Floor on shares: avoids log(0) and the y_j^{1/α-1} blow-up when α < 1.
    s_safe = np.clip(s, 1e-12, 1.0)
    q = s_safe ** (gamma / (gamma - 1.0))  # y_j / Y_im
    y = q * Y_im
    mu = pricing.markups_from_shares(s_safe, eta, gamma, regime, mu_bar=mu_bar)
    inv_a = 1.0 / alpha
    log_y_hat = np.log(y_hat)
    mc = (Omega / v) * np.exp((inv_a - 1.0) * (np.log(y) - log_y_hat))
    p = mu * mc
    return y, mc, p


def _update_state(
    s: np.ndarray,
    Y_im: float,
    X_market: float,
    alpha: np.ndarray,
    v: np.ndarray,
    eta: float,
    gamma: float,
    Omega: float,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """One fixed-point evaluation: state → (new_s, new_Y_im, y, p, P_im).

    Steps (scaffold §3.1):
      1. y_j = s_j^{γ/(γ-1)} Y_im
      2. μ_j from s_j; MC_j from (α_j, v_j, y_j); p_j = μ_j MC_j
      3. P_im from CES aggregation of prices
      4. new s_j = (p_j / P_im)^{1-γ}
      5. new Y_im = X_market / P_im (level closure; equivalent to Σp_jy_j=X_market)
    """
    y, _mc, p = _prices_from_state(
        s, Y_im, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar
    )
    # P_im = (Σ p^{1-γ})^{1/(1-γ)}
    p_pow = p ** (1.0 - gamma)
    P_im = p_pow.sum() ** (1.0 / (1.0 - gamma))
    s_new = p_pow / p_pow.sum()
    Y_im_new = X_market / P_im
    return s_new, Y_im_new, y, p, P_im


def _residual(
    s: np.ndarray, Y_im: float, s_new: np.ndarray, Y_im_new: float
) -> float:
    """L∞-style residual: max(|Δs|) + |Δlog Y_im|."""
    return float(np.max(np.abs(s_new - s)) + abs(np.log(Y_im_new) - np.log(Y_im)))


# ---------------------------------------------------------------------------
# Newton fallback
# ---------------------------------------------------------------------------


def _g_vector(
    theta: np.ndarray,
    n: int,
    X_market: float,
    alpha: np.ndarray,
    v: np.ndarray,
    eta: float,
    gamma: float,
    Omega: float,
    y_hat: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
) -> np.ndarray:
    """Residual map g(θ) = step(θ) − θ in reduced coords.

    θ = (s_1, ..., s_{n-1}, log Y_im) — n unknowns, since s_n = 1 - Σ_{j<n} s_j
    and we work in log Y_im to avoid sign issues.
    """
    s_head = theta[:-1]
    s_n = 1.0 - s_head.sum()
    s = np.concatenate([s_head, [s_n]])
    Y_im = float(np.exp(theta[-1]))
    s_new, Y_im_new, _, _, _ = _update_state(
        s, Y_im, X_market, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar
    )
    theta_new = np.concatenate([s_new[:-1], [np.log(Y_im_new)]])
    return theta_new - theta


def _newton_step(
    theta: np.ndarray,
    n: int,
    X_market: float,
    alpha: np.ndarray,
    v: np.ndarray,
    eta: float,
    gamma: float,
    Omega: float,
    y_hat: float = 1.0,
    fd_step: float = 1e-6,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
) -> np.ndarray:
    """One Newton step on θ; finite-difference Jacobian of g."""
    g0 = _g_vector(theta, n, X_market, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar)
    m = theta.size
    J = np.empty((m, m))
    for k in range(m):
        bump = np.zeros_like(theta)
        bump[k] = fd_step
        g_plus = _g_vector(
            theta + bump, n, X_market, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar
        )
        J[:, k] = (g_plus - g0) / fd_step
    # Newton: solve J Δθ = -g, then θ_new = θ + Δθ.
    try:
        dtheta = np.linalg.solve(J, -g0)
    except np.linalg.LinAlgError:
        dtheta, *_ = np.linalg.lstsq(J, -g0, rcond=None)
    return theta + dtheta


def _theta_to_state(theta: np.ndarray) -> tuple[np.ndarray, float]:
    s_head = theta[:-1]
    s = np.concatenate([s_head, [1.0 - s_head.sum()]])
    return s, float(np.exp(theta[-1]))


def _state_to_theta(s: np.ndarray, Y_im: float) -> np.ndarray:
    return np.concatenate([s[:-1], [np.log(Y_im)]])


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def solve(
    n: int,
    alpha: np.ndarray,
    v: np.ndarray,
    eta: float,
    gamma: float,
    w: float,
    R: float,
    X_market: float,
    a_i: float,
    *,
    phi_v: float = DEFAULT_PHI_V,
    y_hat: float = 1.0,
    P: float = 1.0,
    regime: PricingRegime = PricingRegime.MARKET,
    mu_bar: float | None = None,
    love_of_variety: bool | str = False,
    tol: float = 1e-8,
    max_iter: int = 200,
    damping: float = 0.5,
    newton_fallback_after: int = 50,
    s0: np.ndarray | None = None,
    Y_im0: float | None = None,
) -> MarketSolution:
    """Solve one product market.

    Parameters
    ----------
    n
        Number of firms.
    alpha, v
        Length-``n`` arrays of firm primitives.
    eta, gamma
        Demand elasticities (γ > η > 1).
    w, R
        Wage and rental rate from the outer normalization.
    X_market
        Market expenditure $X_{market} = P_{im} Y_{im}$, from the middle
        (sector) loop. The level closure of the joint FP.
    a_i
        Sector capital share in the value-added composite.
    phi_v
        Value-added weight in the gross-output bundle (externally assigned, not
        calibrated). Enters only through the gross-output unit cost ``Ω^g``.
    y_hat
        Common output anchor. Production calibration normalizes this to one,
        but it is threaded for identity tests and diagnostics.
    P
        Aggregate composite (materials) price. Normalized to ``1`` in the
        steady state, but kept explicit so ``Ω^g`` is not hard-coded.
    regime
        Pricing rule (:class:`pricing.PricingRegime`). ``MARKET`` (default) is
        the decentralized EMX markup; ``PLANNER`` sets p = MC (μ ≡ 1);
        ``UNIFORM`` applies a common markup ``mu_bar``.
    mu_bar
        Common markup for the ``UNIFORM`` regime; ignored otherwise.
    love_of_variety
        ``False`` or ``"off"`` (default) reports the normalized market bundle
        and price index. ``True`` or ``"on"`` restores love of variety. This
        affects only ``Y_j_normalized`` and ``P_j_normalized``; the inner
        equilibrium and legacy ``Y_im``/``P_im`` fields are unchanged.
    tol, max_iter, damping, newton_fallback_after
        Solver controls (see ``config.yaml``).
    s0, Y_im0
        Optional warm-start. Defaults: uniform shares and $Y_{im} = X_{market}$.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if alpha.shape != (n,) or v.shape != (n,):
        raise ValueError(
            f"alpha and v must have shape ({n},); got {alpha.shape}, {v.shape}"
        )
    # γ > η required for the cross-firm/between-sector wedge spread.
    # η > 0 required for the CES aggregator. η < 1 is admissible
    # (across-sector complementarity; EMX 2023 Table 5 uses η = 0.99 at
    # M = 1.35). At η < 1, the wedge formula 1/μ = (1-1/γ) - (1/η-1/γ)·s
    # only gives μ > 0 for s < s_crit = η(γ-1)/(γ-η); the inner FP must
    # avoid this boundary, which it does whenever n_i is even modestly
    # large (max share is then well below s_crit).
    if eta <= 0.0 or gamma <= eta:
        raise ValueError(f"need γ > η > 0; got γ={gamma}, η={eta}")
    if X_market <= 0:
        raise ValueError(f"X_market must be positive; got {X_market}")
    if y_hat <= 0:
        raise ValueError(f"y_hat must be positive; got {y_hat}")

    variety_factor = _variety_normalization_factor(n, gamma, love_of_variety)

    # Ω^g is the gross-output unit cost; it enters the inner Cournot FP exactly
    # like the old value-added Ω (a positive constant given w, R, P).
    Omega = _omega_gross(w, R, a_i, phi_v, P)

    # --- initial guess ------------------------------------------------------
    if s0 is None:
        s = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        s = np.asarray(s0, dtype=np.float64).copy()
        s = s / s.sum()
    if Y_im0 is None:
        # X_market = P_im * Y_im. We don't know P_im yet — start as if P_im = 1.
        Y_im = float(X_market)
    else:
        Y_im = float(Y_im0)

    # --- damped FP loop -----------------------------------------------------
    converged = False
    used_newton = False
    last_residual = np.inf
    stall_count = 0
    best_residual = np.inf
    it = 0

    y = p = None  # populated by _update_state below
    P_im = np.nan

    for it in range(1, max_iter + 1):
        s_new, Y_im_new, y, p, P_im = _update_state(
            s, Y_im, X_market, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar
        )
        res = _residual(s, Y_im, s_new, Y_im_new)
        if res < tol:
            # Take the new state as the converged one.
            s, Y_im = s_new, Y_im_new
            # Recompute (y, p, P_im) at the converged state so the returned
            # vectors are consistent with the reported s, Y_im.
            y, _, p = _prices_from_state(
                s, Y_im, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar
            )
            p_pow = p ** (1.0 - gamma)
            P_im = p_pow.sum() ** (1.0 / (1.0 - gamma))
            last_residual = res
            converged = True
            break

        # Damped update.
        s = damping * s_new + (1.0 - damping) * s
        s = np.clip(s, 1e-12, 1.0 - 1e-12)
        s = s / s.sum()
        log_Y = damping * np.log(Y_im_new) + (1.0 - damping) * np.log(Y_im)
        Y_im = float(np.exp(log_Y))
        last_residual = res

        # Stall detection.
        if res >= best_residual * 0.99:
            stall_count += 1
        else:
            stall_count = 0
            best_residual = res

        # Newton fallback on persistent stall.
        if (
            not used_newton
            and stall_count >= newton_fallback_after
            and it < max_iter
        ):
            used_newton = True
            theta = _state_to_theta(s, Y_im)
            try:
                theta = _newton_step(
                    theta, n, X_market, alpha, v, eta, gamma, Omega,
                    y_hat=y_hat, regime=regime, mu_bar=mu_bar,
                )
                s_try, Y_im_try = _theta_to_state(theta)
                # Reject Newton step if it leaves the simplex.
                if (s_try > 0).all() and (s_try < 1).all() and Y_im_try > 0:
                    s, Y_im = s_try / s_try.sum(), Y_im_try
                    stall_count = 0
                    best_residual = np.inf
            except Exception:
                # Don't let Newton fallback abort the solve; we'll just keep
                # iterating with the damped scheme and exit non-converged.
                pass

    if not converged:
        # Final recompute so returned (y, p, P_im) match returned (s, Y_im).
        y, _, p = _prices_from_state(
            s, Y_im, alpha, v, eta, gamma, Omega, y_hat, regime, mu_bar
        )
        p_pow = p ** (1.0 - gamma)
        P_im = p_pow.sum() ** (1.0 / (1.0 - gamma))

    mu = pricing.markups_from_shares(s, eta, gamma, regime, mu_bar=mu_bar)
    revenue = p * y
    # p = μ·MC ⇒ MC·y = revenue/μ; total variable cost TC = α·MC·y (non-CRS),
    # so cost holds TC and profit = revenue − TC = (1 − α/μ)·revenue.
    cost = alpha * revenue / mu
    profits = revenue - cost

    return MarketSolution(
        y=y,
        p=p,
        s=s,
        mu=mu,
        d=profits,
        cost=cost,
        Y_im=float(Y_im),
        P_im=float(P_im),
        Y_j_normalized=float(Y_im / variety_factor),
        P_j_normalized=float(P_im * variety_factor),
        converged=converged,
        iterations=it,
        used_newton=used_newton,
        residual=float(last_residual),
    )
