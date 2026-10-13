"""Common-random-number draws for the pooled market kernel.

Two separable stages generate productivity (see ``concentration_moment_diagnosis.md``
Appendix A.3):

* **Stage 1: ladder position.** Combine the firm's α-rank score
  ``tilde_alpha = Φ⁻¹(rank(α))`` with independent luck ``ε`` into a standardized
  ladder ``ρ̄·tilde_alpha + √(1−ρ̄²)·ε ~ N(0,1)``. This is the *only* place α and v
  are linked, and the link strength is exactly ``rho_bar`` (the α–v rank-copula
  strength), independent of the marginal shape. ``rho_bar`` may be negative
  (reverse sorting: high-α firms sit low on the capability ladder); the ladder
  stays standard normal for any ``rho_bar`` in (-1, 1), because the variance
  ``ρ̄² + (1−ρ̄²) = 1`` does not depend on the sign.
* **Stage 2: Pareto marginal.** Map the ladder percentile ``u = Φ(ladder)`` through
  a Pareto(ξ) quantile ``v = v_min·(1−u)^(−1/ξ)``. The tail index ``ξ`` is the
  tail of inverse marginal cost at the common anchor.

Entry is EMX-style and exogenous: each sector draws ``n_m = max(1, Poisson(N))``
producing firms (:func:`allocate_poisson`). All allocated firms produce; there is
no per-period operating cost and no profitability cutoff.

**Alpha cap.** A firm with ``alpha >= gamma/(gamma-1)`` (the smallest EMX markup)
earns negative variable profit at small share, and the inner Cournot fixed point
is locally unstable for it: such firms either sit at the share floor or take over
their market, and the anchored GE has no root (see
``02_Drafts/md_files/ORBIS_migration/ORBIS_ANCHORED_GE_DIAGNOSIS.md``).
:func:`draw_pool` therefore caps the drawn alpha at
``alpha_cap(gamma, factor) = factor * gamma/(gamma-1)``, recomputed at every
``gamma``. The cap changes alpha values only: the node draw, the copula rank
``tilde_alpha`` (from the raw support) and the v draw are the same with or
without it, so common random numbers are preserved across ``gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import ndtr, ndtri


DEFAULT_ALPHA_CAP_FACTOR = 0.99


def alpha_cap(gamma: float, factor: float | None = DEFAULT_ALPHA_CAP_FACTOR) -> float | None:
    """Upper bound ``factor * gamma/(gamma-1)`` on alpha; ``None`` when ``factor`` is ``None``.

    ``factor`` must lie in (0, 1]. Keep a margin below one: at the bound itself the
    fixed-point multiplier of a small firm tends to one and inner solves stall
    (factor 0.999 or 1.0 hit ``max_iter`` in the diagnosis; 0.99 does not).
    """
    if factor is None:
        return None
    factor = float(factor)
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"alpha cap factor must lie in (0, 1]; got {factor}")
    if not gamma > 1.0:
        raise ValueError(f"gamma must exceed one; got {gamma}")
    return factor * gamma / (gamma - 1.0)


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
            "(G1: pool binds); raise H",
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
    alpha_max: float | None = None,
) -> PoolDraw:
    """Draw a common ``(M,H)`` pool with the α–v copula and a Pareto-ξ v marginal.

    ``xi`` is the Pareto tail index (>1 for a finite mean); ``v_min`` is the
    lower bound for inverse marginal cost at the anchor. It preserves the α–v
    *rank* dependence while scaling capability without redrawing. ``N`` is the
    Poisson entry mean. ``rho_bar`` lies in (-1, 1): a negative value gives
    negative α–v rank dependence (Salgado et al. calibrate rho_{z,eta} = -0.25).
    ``alpha_max`` (from :func:`alpha_cap`) caps the drawn alpha from above; the
    copula rank ``tilde_alpha`` is still computed on the raw support, so capped
    firms keep their raw ladder position.
    """
    support = np.asarray(alpha_support, dtype=np.float64)
    if support.ndim != 1 or support.size == 0:
        raise ValueError("alpha_support must be a non-empty vector")
    if M <= 0 or H <= 0:
        raise ValueError("M and H must be positive")
    if not xi > 1.0:
        raise ValueError(f"Pareto tail index xi must exceed 1; got {xi}")
    if not (-1.0 < rho_bar < 1.0):
        raise ValueError(f"rho_bar must lie in (-1, 1); got {rho_bar}")
    if v_min <= 0.0:
        raise ValueError("v_min must be positive")
    if alpha_max is not None and not alpha_max > 0.0:
        raise ValueError(f"alpha_max must be positive; got {alpha_max}")

    gen = np.random.default_rng(rng)
    node = gen.integers(0, support.size, size=(M, H))
    alpha = support[node]
    if alpha_max is not None:
        alpha = np.minimum(alpha, float(alpha_max))
    ranks = (np.argsort(np.argsort(support, kind="stable"), kind="stable") + 0.5) / support.size
    tilde_alpha = ndtri(ranks[node])
    eps = gen.standard_normal((M, H))

    # Stage 1: standardized ladder; corr(tilde_alpha, ladder) = rho_bar exactly,
    # for either sign of rho_bar.
    ladder = rho_bar * tilde_alpha + np.sqrt(1.0 - rho_bar * rho_bar) * eps
    # bridge: ladder position -> percentile in (0, 1)
    u = ndtr(ladder)
    # Stage 2: Pareto(xi) marginal (fat tail dedicated to concentration).
    v = v_min * (1.0 - u) ** (-1.0 / xi)

    # Exogenous EMX entry: reuse the same generator so the mask is part of the CRN.
    active_mask, _ = allocate_poisson(N, M, H, rng=gen)
    return PoolDraw(alpha=alpha, v=v, tilde_alpha=tilde_alpha, active_mask=active_mask)
