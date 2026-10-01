"""Curvature-preserving collapse of the alpha distribution in a PoolDraw.

Two variants of a "homogeneous economy" that hold each sector's mean marginal-
cost curvature ``kappa_bar_j = mean(1/alpha_j) - 1`` fixed while eliminating
within-sector alpha *dispersion* entirely:

- :func:`homogenize_alpha_primitive` — per-sector harmonic mean, matching the
  invariant :func:`~counterfactuals.scalability_sorting.permute.permute_alpha`
  already preserves (the active-alpha multiset per sector, hence its mean
  curvature). This is the headline variant: it isolates within-sector alpha
  dispersion as a single, comparable dimension against the shuffle economy.
- :func:`homogenize_alpha_cost_weighted` — an economy-wide cost-weighted
  harmonic mean using baseline true-variable-cost weights. This answers a
  different question ("hold the aggregate MC elasticity fixed") and is a
  labeled robustness check, not the headline.

``steady_state/`` is untouched; this module only imports ``PoolDraw``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from steady_state.model.pool import PoolDraw


def sector_mean_inv_alpha(draw: PoolDraw) -> np.ndarray:
    """Per-sector ``mean(1/alpha)`` over active firms only (an ``(M,)`` vector).

    Diagnostic curvature statistic: ``mean(1/alpha) - 1`` is the sector's mean
    MC curvature ``kappa_bar_j``.
    """
    alpha = np.asarray(draw.alpha, dtype=np.float64)
    mask = np.asarray(draw.active_mask, dtype=bool)
    M = alpha.shape[0]
    out = np.full(M, np.nan, dtype=np.float64)
    for m in range(M):
        active = np.nonzero(mask[m])[0]
        if active.size == 0:
            continue
        out[m] = float(np.mean(1.0 / alpha[m, active]))
    return out


def homogenize_alpha_primitive(base_draw: PoolDraw) -> PoolDraw:
    """Collapse alpha within each sector to that sector's harmonic mean.

    ``alpha_hom_j = ((1/n_j) * sum_{i in active_j} 1/alpha_ji)^-1``. This
    conserves ``mean(1/alpha)`` per sector exactly (hence kappa_bar_j), so the
    resulting economy differs from the calibrated shuffle economies in
    exactly one dimension: within-sector alpha variance collapses from the
    shuffle's full dispersion to zero.

    Inactive slots are left at the base draw's alpha (never solver-consumed).
    ``tilde_alpha`` is set to NaN economy-wide: alpha's rank score is
    meaningless once alpha is constant within a sector, and it only feeds a
    diagnostic panel column, never the solver.
    """
    alpha_base = np.asarray(base_draw.alpha, dtype=np.float64)
    mask = np.asarray(base_draw.active_mask, dtype=bool)
    M = alpha_base.shape[0]

    alpha = alpha_base.copy()
    for m in range(M):
        active = np.nonzero(mask[m])[0]
        # Singleton sectors: skip the 1/mean(1/x) round-trip so the result is
        # bit-identical to the input, not merely numerically close.
        if active.size <= 1:
            continue
        alpha_hom_m = 1.0 / np.mean(1.0 / alpha_base[m, active])
        alpha[m, active] = alpha_hom_m

    tilde_alpha = np.full_like(alpha, np.nan)
    new_draw = dataclasses.replace(base_draw, alpha=alpha, tilde_alpha=tilde_alpha)
    _assert_homogenized(new_draw, base_draw, variant="primitive")
    return new_draw


def homogenize_alpha_cost_weighted(base_draw: PoolDraw, me_base) -> PoolDraw:
    """Collapse alpha economy-wide to a cost-weighted harmonic mean.

    ``alpha_hom = (sum_i c_i/alpha_i)^-1`` where ``c_i = cost_i / sum(cost)``
    are baseline true-variable-cost shares (``me_base.participation.solution
    .cost``, ``TC = alpha*revenue/mu``) over every active firm economy-wide.
    Unlike :func:`homogenize_alpha_primitive`, this is a single scalar applied
    to every active firm regardless of sector -- it holds the aggregate MC
    elasticity fixed, not each sector's mean curvature. Robustness variant
    only; see module docstring.
    """
    alpha_base = np.asarray(base_draw.alpha, dtype=np.float64)
    mask = np.asarray(base_draw.active_mask, dtype=bool)
    cost = np.asarray(me_base.participation.solution.cost, dtype=np.float64)

    cost_active = cost[mask]
    alpha_active = alpha_base[mask]
    weights = cost_active / np.sum(cost_active)
    alpha_hom = 1.0 / np.sum(weights / alpha_active)

    alpha = alpha_base.copy()
    alpha[mask] = alpha_hom

    tilde_alpha = np.full_like(alpha, np.nan)
    new_draw = dataclasses.replace(base_draw, alpha=alpha, tilde_alpha=tilde_alpha)
    _assert_homogenized(new_draw, base_draw, variant="cost_weighted")
    return new_draw


def _assert_homogenized(
    new_draw: PoolDraw,
    base_draw: PoolDraw,
    *,
    variant: str,
) -> None:
    """Runtime guards (house style, cf. ``permute._assert_marginals_preserved``).

    Checked for every variant: ``v``/``active_mask`` are bit-identical, and
    every active sector has zero within-sector alpha variance. Checked only
    for ``variant="primitive"`` (properties specific to the per-sector
    harmonic-mean construction, not the economy-wide cost-weighted one):
    per-sector ``mean(1/alpha)`` is preserved exactly, and singleton sectors
    (``n_active == 1``) are left unchanged.
    """
    v_new = np.asarray(new_draw.v, dtype=np.float64)
    v_base = np.asarray(base_draw.v, dtype=np.float64)
    if not np.array_equal(v_new, v_base):
        raise AssertionError(f"homogenize_alpha_{variant} changed v")

    mask_new = np.asarray(new_draw.active_mask, dtype=bool)
    mask_base = np.asarray(base_draw.active_mask, dtype=bool)
    if not np.array_equal(mask_new, mask_base):
        raise AssertionError(f"homogenize_alpha_{variant} changed active_mask")

    alpha_new = np.asarray(new_draw.alpha, dtype=np.float64)
    alpha_base = np.asarray(base_draw.alpha, dtype=np.float64)
    M = alpha_new.shape[0]

    for m in range(M):
        active = np.nonzero(mask_new[m])[0]
        if active.size == 0:
            continue
        if not np.isclose(np.var(alpha_new[m, active]), 0.0, atol=1e-12):
            raise AssertionError(
                f"homogenize_alpha_{variant} left nonzero alpha variance in "
                f"sector {m}"
            )
        if variant == "primitive":
            mean_inv_new = float(np.mean(1.0 / alpha_new[m, active]))
            mean_inv_base = float(np.mean(1.0 / alpha_base[m, active]))
            if not np.isclose(mean_inv_new, mean_inv_base, rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"homogenize_alpha_primitive changed mean(1/alpha) in "
                    f"sector {m}: {mean_inv_new} != {mean_inv_base}"
                )
            if active.size == 1 and not np.allclose(
                alpha_new[m, active], alpha_base[m, active], rtol=1e-9, atol=1e-12
            ):
                raise AssertionError(
                    f"homogenize_alpha_primitive changed alpha in singleton "
                    f"sector {m}"
                )
