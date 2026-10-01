"""Pure ``Delta = log(1+lambda)`` algebra for the Lens-B decomposition.

No solves, no I/O -- just the bookkeeping that turns per-arrangement
consumption-equivalent lambdas into the additive 3x3
``{Delta_dispersion, Delta_level} x {common-alpha, heterogeneity, sorting}``
table, sharing the (cached) ``Delta_total`` row. The governing identities are
all exact on the log scale, arrangement by arrangement, because
``log(1+lambda) = (1-beta)*Delta_W`` is linear in welfare:

* multiplicative Lens-B split
  ``(1+lambda_dispersion)(1+lambda_level) = (1+lambda_total)``
  => ``Delta_level = Delta_total - Delta_dispersion`` termwise;
* three-channel split of any ``Delta``-series into
  ``common_alpha = Delta_hom``, ``heterogeneity = Delta_shuffle - Delta_hom``,
  ``sorting = Delta_base - Delta_shuffle`` (mirrors
  :mod:`counterfactuals.homogeneous_rts` and
  :mod:`counterfactuals.scale_channel_decomposition`).

This module is intentionally a self-contained copy of the scale-channel
Delta-algebra (with ``scale`` -> ``level``) so the Lens-B package carries no
run-time dependency on the scale-channel *code*; the only cross-package reuse is
the cached *data* (parquet). Sibling of
:func:`steady_state.welfare.welfare_metrics.consumption_equivalent_lambda` math;
nothing under ``steady_state/`` is imported or modified here.
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
    ``delta_base`` by construction. Applied to the ``Delta_dispersion`` and
    (cached) ``Delta_total`` series; the ``Delta_level`` split follows by
    termwise subtraction ``Delta_total - Delta_dispersion``.
    """
    common_alpha = delta_hom
    heterogeneity = delta_shuffle_mean - delta_hom
    sorting = delta_base - delta_shuffle_mean
    return {
        "common_alpha": float(common_alpha),
        "heterogeneity": float(heterogeneity),
        "sorting": float(sorting),
    }


def per_draw_level_delta(
    delta_total_draws,
    delta_dispersion_draws,
) -> tuple[float, float]:
    """Mean +/- SE of the per-draw level delta ``Delta_level_i = Delta_total_i - Delta_disp_i``.

    The shuffle draws share common random numbers across the total and
    dispersion legs (both anchored on the same cached ``W_market``), so
    difference *per matched draw* first and then average -- never difference the
    averaged Delta's (that discards the CRN pairing and inflates the SE). The two
    inputs must align 1:1 by shuffle draw (the caller guarantees perm_seed
    alignment). Both are already on the ``Delta = log(1+lambda)`` scale. Returns
    ``(mean, SE)``.
    """
    total = np.asarray(delta_total_draws, dtype=np.float64)
    disp = np.asarray(delta_dispersion_draws, dtype=np.float64)
    if total.shape != disp.shape:
        raise ValueError(
            "delta_total_draws and delta_dispersion_draws must align 1:1 "
            f"(got shapes {total.shape} and {disp.shape})"
        )
    level = total - disp
    mean = float(np.mean(level)) if level.size else float("nan")
    se = (float(np.std(level, ddof=1) / np.sqrt(level.size))
          if level.size > 1 else float("nan"))
    return mean, se


def identity_residual(
    lambda_dispersion: float,
    lambda_level: float,
    lambda_total: float,
) -> float:
    """Multiplicative Lens-B residual ``(1+lambda_disp)(1+lambda_level) - (1+lambda_total)``.

    Expected ~0 to machine precision when all three lambdas are built from a
    consistent ``(W_market, W_uniform, W_planner)`` welfare triple: with
    ``W_market`` and ``W_planner`` read from the cached scale-channel run and
    ``W_uniform`` the new UNIFORM-regime solve (all on the same frozen
    ``chi_baseline``), the exponentials telescope exactly.
    """
    return float(
        (1.0 + lambda_dispersion) * (1.0 + lambda_level) - (1.0 + lambda_total)
    )
