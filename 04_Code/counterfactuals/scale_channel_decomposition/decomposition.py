"""Pure ``Delta = log(1+lambda)`` algebra for the scale-channel decomposition.

No solves, no I/O -- just the bookkeeping that turns per-arrangement
consumption-equivalent lambdas into the additive 3x3
``{Delta_K, Delta_total, Delta_scale} x {common-alpha, heterogeneity, sorting}``
table. The governing identities are all exact on the log scale, arrangement by
arrangement, because ``log(1+lambda) = (1-beta)*Delta_W`` is linear in welfare:

* multiplicative Lens-A split ``(1+lambda_K)(1+lambda_scale) = (1+lambda_total)``
  => ``Delta_scale = Delta_total - Delta_K`` termwise;
* three-channel split of any ``Delta``-series into
  ``common_alpha = Delta_hom``, ``heterogeneity = Delta_shuffle - Delta_hom``,
  ``sorting = Delta_base - Delta_shuffle`` (mirrors
  :mod:`counterfactuals.homogeneous_rts`, whose channels are named
  markup_common / dispersion / sorting).

Sibling of :func:`steady_state.welfare.welfare_metrics.consumption_equivalent_lambda`
math; nothing under ``steady_state/`` is imported or modified here.
"""

from __future__ import annotations

import numpy as np


def three_channel_split(
    delta_base: float,
    delta_shuffle_mean: float,
    delta_hom: float,
) -> dict:
    """Additive ``{common_alpha, heterogeneity, sorting}`` split of a Delta-series.

    ``common_alpha`` is the homogeneous-alpha leg, ``heterogeneity`` the
    within-sector-dispersion residual (shuffle - hom), ``sorting`` the
    alpha<->v assortativity residual (base - shuffle). The three sum to
    ``delta_base`` by construction. Applied to the ``Delta_total`` series; the
    ``Delta_scale`` split follows by termwise subtraction of the (cached)
    ``Delta_K`` split -- never recomputed from lambda.
    """
    common_alpha = delta_hom
    heterogeneity = delta_shuffle_mean - delta_hom
    sorting = delta_base - delta_shuffle_mean
    return {
        "common_alpha": float(common_alpha),
        "heterogeneity": float(heterogeneity),
        "sorting": float(sorting),
    }


def per_draw_scale_delta(
    delta_total_draws,
    delta_K_draws,
) -> tuple[float, float]:
    """Mean +/- SE of the per-draw scale delta ``Delta_scale_i = Delta_total_i - Delta_K_i``.

    SPEC sec. 6-iii: the shuffle draws share common random numbers across the
    total and reallocation legs, so difference *per matched draw* first and then
    average -- never difference the averaged Delta's (that discards the CRN
    pairing and inflates the SE). The two inputs must align 1:1 by shuffle draw
    (the caller guarantees perm_seed alignment). Both are already on the
    ``Delta = log(1+lambda)`` scale. Returns ``(mean, SE)``.
    """
    total = np.asarray(delta_total_draws, dtype=np.float64)
    reall = np.asarray(delta_K_draws, dtype=np.float64)
    if total.shape != reall.shape:
        raise ValueError(
            "delta_total_draws and delta_K_draws must align 1:1 "
            f"(got shapes {total.shape} and {reall.shape})"
        )
    scale = total - reall
    mean = float(np.mean(scale)) if scale.size else float("nan")
    se = (float(np.std(scale, ddof=1) / np.sqrt(scale.size))
          if scale.size > 1 else float("nan"))
    return mean, se


def identity_residual(
    lambda_K: float,
    lambda_scale: float,
    lambda_total: float,
) -> float:
    """Multiplicative Lens-A residual ``(1+lambda_K)(1+lambda_scale) - (1+lambda_total)``.

    Expected ~0 to machine precision when all three lambdas are built from a
    consistent ``(W_market, W_planner_fixed_k, W_planner)`` welfare triple:
    with ``W_market`` and ``W_planner_fixed_k`` read from the cached
    reallocation run and ``W_planner`` the new full-GE solve, the exponentials
    telescope exactly.
    """
    return float(
        (1.0 + lambda_K) * (1.0 + lambda_scale) - (1.0 + lambda_total)
    )
