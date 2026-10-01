"""Pooled calibration moments evaluated at a frozen full-pool reference."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..model.aggregator import POOLED_MOMENT_KEYS, pooled_moments
from ..model.normalization import (
    FullPoolReference,
    PooledEquilibrium,
    PooledParams,
    solve_pooled_ge_anchored,
)
from ..model.pool import draw_pool
from ..pooled_inputs import PooledInputs


MOMENT_KEYS = POOLED_MOMENT_KEYS
SECTOR_MOMENT_KEYS = MOMENT_KEYS  # temporary import compatibility; values are pooled


def pooled_moments_at_params(
    params: PooledParams,
    inputs: PooledInputs,
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
    support = (
        inputs.alpha_support
        if inputs.alpha_support is not None
        else np.linspace(0.6, 1.20, 500)
    )
    draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
        M=M, H=params.H, rng=rng, v_min=1.0,
    )
    equilibrium = solve_pooled_ge_anchored(
        draw.alpha, draw.v, params, beta=beta, delta_K=delta_K,
        n_firms_cs=inputs.n_firms_cs, K_switch=K_switch,
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
        sol, mask, n_firms_cs=inputs.n_firms_cs, phi_v=inputs.phi_v
    )
    moments["reference_P"] = reference.P
    moments["reference_L"] = reference.L
    return moments, reference, equilibrium


def pooled_relative_residuals(
    params: PooledParams,
    inputs: PooledInputs,
    *,
    M: int,
    rng,
    moment_keys: Sequence[str] = MOMENT_KEYS,
    weights: Mapping[str, float] | None = None,
    **kwargs,
) -> np.ndarray:
    model, _, _ = pooled_moments_at_params(params, inputs, M=M, rng=rng, **kwargs)
    residuals = []
    for key in moment_keys:
        if key not in inputs.targets:
            raise KeyError(f"pooled target {key!r} is missing")
        target = float(inputs.targets[key])
        # EMX (2022) normalizer: 1 + |target| (objective.m:183). Must match the
        # live residual in calibration.inner.calibrate_pooled.
        value = (float(model[key]) - target) / (1.0 + abs(target))
        residuals.append(value * float((weights or {}).get(key, 1.0)))
    return np.asarray(residuals)


def elasticities_from_emx(intercept: float, slope: float) -> tuple[float, float]:
    """Recover ``(gamma, eta)`` from the inverse-markup line."""
    gamma = 1.0 / (1.0 - float(intercept))
    eta = 1.0 / (1.0 / gamma - float(slope))
    if not 1.0 < eta < gamma:
        raise ValueError("the planted line does not imply 1 < eta < gamma")
    return gamma, eta
