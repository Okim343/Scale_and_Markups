"""Common-random-number draws for the pooled market kernel.

Two separable stages generate productivity (see ``concentration_moment_diagnosis.md``
Appendix A.3):

* **Stage 1 — ladder position.** Combine the firm's α-rank score
  ``tilde_alpha = Φ⁻¹(rank(α))`` with independent luck ``ε`` into a standardized
  ladder ``ρ̄·tilde_alpha + √(1−ρ̄²)·ε ~ N(0,1)``. This is the *only* place α and v
  are linked, and the link strength is exactly ``rho_bar`` (the α–v rank-copula
  strength), independent of the marginal shape.
* **Stage 2 — Pareto marginal.** Map the ladder percentile ``u = Φ(ladder)`` through
  a Pareto(ξ) quantile ``v = v_min·(1−u)^(−1/ξ)``. The tail index ``ξ`` is the
  tail of inverse marginal cost at the common anchor.

Entry is EMX-style and exogenous: each sector draws ``n_m = max(1, Poisson(N))``
producing firms (:func:`allocate_poisson`). All allocated firms produce; there is
no per-period operating cost and no profitability cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import ndtr, ndtri


@dataclass(frozen=True)
class PoolDraw:
    alpha: np.ndarray
    v: np.ndarray
    tilde_alpha: np.ndarray
    active_mask: np.ndarray


def derived_z(alpha: np.ndarray, v: np.ndarray, y_hat: float = 1.0) -> np.ndarray:
    """Diagnostic implied legacy productivity under the anchored relabeling."""
    alpha = np.asarray(alpha, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if y_hat <= 0.0:
        raise ValueError("y_hat must be positive")
    return (v / alpha) ** alpha * y_hat ** (1.0 - alpha)


def allocate_poisson(
    N: float,
    M: int,
    H: int,
    rng: int | np.random.SeedSequence | np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r"""EMX-style exogenous entry: ``n_m = max(1, Poisson(N))`` firms per sector.

    Returns ``(active_mask, counts)`` where ``active_mask`` is an ``(M, H)`` bool
    array activating the first ``n_m`` pool slots in sector ``m`` (mirrors EMX's
    ``ns = max(1, poissrnd(N, S, 1))``, :file:`objective.m:84`). Because the pool
    columns are i.i.d. draws, activating the first ``n_m`` is equivalent to a
    random ``n_m``-firm sample. The count is capped at ``H``; a warning fires if
    the pool binds so ``H`` can be raised.
    """
    if N <= 0.0:
        raise ValueError("Poisson mean N must be positive")
    if M <= 0 or H <= 0:
        raise ValueError("M and H must be positive")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    counts = np.maximum(1, gen.poisson(float(N), size=M)).astype(np.int64)
    if np.any(counts > H):
        warnings.warn(
            "Poisson firm count exceeds the pool size H in at least one sector "
            "(G1: pool binds) — raise H",
            RuntimeWarning,
            stacklevel=2,
        )
        counts = np.minimum(counts, H)
    active_mask = np.arange(H)[None, :] < counts[:, None]
    return active_mask, counts


def draw_pool(
    alpha_support: np.ndarray,
    *,
    xi: float,
    rho_bar: float,
    N: float,
    M: int,
    H: int,
    rng: int | np.random.SeedSequence,
    v_min: float = 1.0,
) -> PoolDraw:
    """Draw a common ``(M,H)`` pool with the α–v copula and a Pareto-ξ v marginal.

    ``xi`` is the Pareto tail index (>1 for a finite mean); ``v_min`` is the
    lower bound for inverse marginal cost at the anchor. It preserves the α–v
    *rank* dependence while scaling capability without redrawing. ``N`` is the
    Poisson entry mean.
    """
    support = np.asarray(alpha_support, dtype=np.float64)
    if support.ndim != 1 or support.size == 0:
        raise ValueError("alpha_support must be a non-empty vector")
    if M <= 0 or H <= 0:
        raise ValueError("M and H must be positive")
    if not xi > 1.0:
        raise ValueError(f"Pareto tail index xi must exceed 1; got {xi}")
    if not (0.0 <= rho_bar < 1.0):
        raise ValueError(f"rho_bar must lie in [0, 1); got {rho_bar}")
    if v_min <= 0.0:
        raise ValueError("v_min must be positive")

    gen = np.random.default_rng(rng)
    node = gen.integers(0, support.size, size=(M, H))
    alpha = support[node]
    ranks = (np.argsort(np.argsort(support, kind="stable"), kind="stable") + 0.5) / support.size
    tilde_alpha = ndtri(ranks[node])
    eps = gen.standard_normal((M, H))

    # Stage 1 — standardized ladder; corr(tilde_alpha, ladder) = rho_bar exactly.
    ladder = rho_bar * tilde_alpha + np.sqrt(1.0 - rho_bar * rho_bar) * eps
    # bridge: ladder position -> percentile in (0, 1)
    u = ndtr(ladder)
    # Stage 2 — Pareto(xi) marginal (fat tail dedicated to concentration).
    v = v_min * (1.0 - u) ** (-1.0 / xi)

    # Exogenous EMX entry: reuse the same generator so the mask is part of the CRN.
    active_mask, _ = allocate_poisson(N, M, H, rng=gen)
    return PoolDraw(alpha=alpha, v=v, tilde_alpha=tilde_alpha, active_mask=active_mask)
