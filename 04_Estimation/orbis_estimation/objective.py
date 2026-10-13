"""Pooled calibration moments evaluated through the anchored GE solve.

Ported from ``04_Code/steady_state/calibration/objective.py``. The model now
reports the RTS gap ``alpha_rts_gap_top5_bottom50`` next to the six pooled
moments, and the residual helper is shared with ``inner.calibrate_pooled``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from orbis_model.model.aggregator import POOLED_MOMENT_KEYS, pooled_moments
from orbis_model.model.normalization import (
    FullPoolReference,
    PooledEquilibrium,
    PooledParams,
    solve_pooled_ge_anchored,
)
from orbis_model.model.pool import draw_pool
from orbis_model.primitives import ModelPrimitives


MOMENT_KEYS = POOLED_MOMENT_KEYS


def pooled_moments_at_params(
    params: PooledParams,
    primitives: ModelPrimitives,
    *,
    M: int,
    rng: int | np.random.SeedSequence,
    beta: float = 0.96,
    delta_K: float = 0.06,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
) -> tuple[dict[str, float], FullPoolReference, PooledEquilibrium]:
    """Draw CRNs (incl. the exogenous Poisson mask) and solve GE on that mask."""
    draw = draw_pool(
        primitives.alpha_support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
        M=M, H=params.H, rng=rng, v_min=1.0,
    )
    equilibrium = solve_pooled_ge_anchored(
        draw.alpha, draw.v, params, beta=beta, delta_K=delta_K,
        n_firms_cs=primitives.n_firms_cs, K_switch=K_switch,
        love_of_variety=love_of_variety,
        market_solver_kwargs=market_solver_kwargs,
        wf_reference=0.0, active_mask=draw.active_mask,
    )
    sol = equilibrium.participation.solution
    mask = equilibrium.participation.active_mask
    reference = FullPoolReference(
        w=equilibrium.w, R=equilibrium.R, X_market=equilibrium.X_market,
        wf_reference=0.0,
        Ed_full_pool=float(np.mean((sol.sales - sol.cost)[mask])),
        P=equilibrium.P_agg, L=equilibrium.L_agg,
        converged=bool(equilibrium.converged), residual=equilibrium.residual,
    )
    moments = pooled_moments(
        sol, mask, n_firms_cs=primitives.n_firms_cs, phi_v=primitives.phi_v,
        alpha=equilibrium.alpha,
    )
    moments["reference_P"] = reference.P
    moments["reference_L"] = reference.L
    return moments, reference, equilibrium


def weighted_relative_residuals(
    model: Mapping[str, float],
    targets: Mapping[str, float],
    *,
    moment_keys: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Weighted target-relative residuals, one per moment key.

    EMX (2022) normalizer: residuals are scaled by (1 + |target|), not
    |target| (objective.m:183, ``./(1 + moment_data)``). This keeps every
    normalizer at O(1-2) so small-magnitude moments (cr4/cr20, slope, RTS gap)
    are NOT amplified into the objective, letting the weights set priority and
    the aggregate markup dominate (EMX's intended markup-vs-concentration
    resolution). We use 1 + |target| (EMX use raw 1 + data because all their
    targets are positive; our emx_slope target is negative, so the abs() keeps
    the normalizer positive and sign-safe). A key missing from ``weights``
    gets weight one.
    """
    residuals = []
    for key in moment_keys:
        if key not in targets:
            raise KeyError(f"pooled target {key!r} is missing")
        target = float(targets[key])
        value = (float(model[key]) - target) / (1.0 + abs(target))
        residuals.append(value * float((weights or {}).get(key, 1.0)))
    return np.asarray(residuals)


def pooled_relative_residuals(
    params: PooledParams,
    primitives: ModelPrimitives,
    targets: Mapping[str, float],
    *,
    M: int,
    rng,
    moment_keys: Sequence[str] = MOMENT_KEYS,
    weights: Mapping[str, float] | None = None,
    **kwargs,
) -> np.ndarray:
    model, _, _ = pooled_moments_at_params(params, primitives, M=M, rng=rng, **kwargs)
    return weighted_relative_residuals(model, targets, moment_keys=moment_keys, weights=weights)


def elasticities_from_emx(intercept: float, slope: float) -> tuple[float, float]:
    """Recover ``(gamma, eta)`` from the inverse-markup line."""
    gamma = 1.0 / (1.0 - float(intercept))
    eta = 1.0 / (1.0 / gamma - float(slope))
    if not 1.0 < eta < gamma:
        raise ValueError("the planted line does not imply 1 < eta < gamma")
    return gamma, eta
