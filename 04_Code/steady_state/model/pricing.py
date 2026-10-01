"""Pricing-regime switch — the single lever that distinguishes the market and
planner allocations.

The pricing rule (markup vs. marginal-cost) is the *only* economic difference
between the decentralized steady state and the planner allocation. Everything
else in the nested fixed-point stack — Monte-Carlo draws, the sector $X_{im}$
fixed point, the outer 2-D GE root finder — is agnostic to whether $\\mu$ enters
pricing (see ``04_Code/implement_planner.md`` §"Key architectural insight").

This module isolates that lever behind :func:`markups_from_shares`, which the
inner market solvers (:mod:`model.market`, :mod:`model.market_batch`) call in
place of the old private ``_markups_from_shares`` helper. Three regimes:

* ``MARKET``  — μ from the EMX inverse-markup rule (model.typ eq. @eq:emx_markup),
  the decentralized equilibrium. This is the default everywhere, so the existing
  ``simulate``/``calibrate`` paths are byte-identical.
* ``PLANNER`` — μ ≡ 1 (p = MC), the efficient within-market allocation
  (model.typ eqs. @eq:planner_labor_foc, @eq:planner_capital_foc).
* ``UNIFORM`` — μ ≡ μ̄ (a scalar input), used by the two-channel welfare
  decomposition to isolate the aggregate-markup *level* from cross-firm
  *dispersion* (model.typ §"Planner's Allocation and Welfare").
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np


# Value-added weight phi_v (Typst's phi.alt_v) in the gross-output bundle. This
# is an externally-assigned primitive, NOT part of the calibrated vector. Under
# the hybrid GNR-KLEMS alpha the value-added weight is
# phi_v = 1 - mean(melast)/mean(alpha_hybrid_sector_a) (denominator is the
# model-facing hybrid alpha, not the raw GNR rts). When the empirical pipeline
# exports `phi_v` in aggregate_moments.yaml the bundle value overrides this
# default (see io_bundle.Bundle.phi_v); this constant is only the legacy-bundle
# fallback. The value below (~0.348) is a preview of the hybrid phi_v computed
# on alpha_hybrid_sector_a over the full step-2 non-outlier sample (mean melast
# 0.534 / mean alpha_hybrid 0.819); the rerun's exported phi_v supersedes it.
DEFAULT_PHI_V: float = 0.3482


class PricingRegime(StrEnum):
    """Which pricing rule the inner market solver applies."""

    MARKET = "market"     # μ from EMX inverse-markup rule (eq. @eq:emx_markup)
    PLANNER = "planner"   # μ ≡ 1 (p = MC), eq. @eq:planner_*_foc
    UNIFORM = "uniform"   # μ ≡ μ̄ (scalar mu_bar), for two-channel decomposition


def _emx_markups(
    s: np.ndarray, eta: float, gamma: float, mu_max: float
) -> np.ndarray:
    """μ_j from the EMX inverse-markup rule (the decentralized pricing object).

    Clips 1/μ from below at 1/mu_max to keep things sane near the boundary
    s_max = η(γ-1)/(γ-η) where 1/μ → 0.

    Kept arithmetically identical to the historical ``market._markups_from_shares``
    so the MARKET regime is byte-for-byte unchanged.
    """
    inv_mu = 1.0 - 1.0 / gamma - (1.0 / eta - 1.0 / gamma) * s
    inv_mu = np.maximum(inv_mu, 1.0 / mu_max)
    return 1.0 / inv_mu


def markups_from_shares(
    s: np.ndarray,
    eta: float,
    gamma: float,
    regime: PricingRegime = PricingRegime.MARKET,
    *,
    mu_bar: float | None = None,
    mu_max: float = 1e6,
) -> np.ndarray:
    """Markup vector for the given pricing ``regime``.

    Parameters
    ----------
    s
        Within-market sales shares, any shape. The return matches ``s``.
    eta, gamma
        Demand elasticities (used only by the MARKET regime).
    regime
        :class:`PricingRegime`. Defaults to ``MARKET`` so existing call sites
        that don't pass a regime reproduce the decentralized equilibrium.
    mu_bar
        Required for ``UNIFORM``: the common markup applied to every firm
        (typically the cost-weighted aggregate markup μ̄^cw of the market
        equilibrium). Ignored by the other regimes.
    mu_max
        Upper clip on μ for the MARKET regime (boundary safety net).
    """
    if regime is PricingRegime.PLANNER:
        return np.ones_like(s)
    if regime is PricingRegime.UNIFORM:
        if mu_bar is None:
            raise ValueError("UNIFORM regime requires a scalar mu_bar")
        return np.full_like(s, float(mu_bar))
    if regime is PricingRegime.MARKET:
        return _emx_markups(s, eta, gamma, mu_max)
    raise ValueError(f"unknown pricing regime {regime!r}")
