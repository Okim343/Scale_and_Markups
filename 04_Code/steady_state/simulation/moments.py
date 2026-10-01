"""Pooled targeted and validation moments."""

from __future__ import annotations

import pandas as pd

from ..model.aggregator import pooled_inverse_markup_regression, pooled_moments
from ..pooled_inputs import PooledInputs
from .cross_section import SimulationResult


def pooled_moments_table(sim: SimulationResult, inputs: PooledInputs) -> pd.DataFrame:
    model = economy_moments(sim, inputs)
    rows = []
    for key, target in inputs.targets.items():
        rows.append({"moment": key, "model": model.get(key), "target": target})
    return pd.DataFrame(rows)


def economy_moments(sim: SimulationResult, inputs: PooledInputs) -> dict[str, float]:
    eq = sim.equilibrium
    moments = pooled_moments(
        eq.participation.solution, eq.participation.active_mask,
        n_firms_cs=inputs.n_firms_cs, phi_v=inputs.phi_v,
    )
    regression = pooled_inverse_markup_regression(
        eq.participation.solution, eq.participation.active_mask,
    )
    return {
        **moments,
        "emx_intercept": regression["intercept"],
        "emx_slope_se": regression["slope_se"],
        "share_std": regression["share_std"],
        "share_range": regression["share_max"] - regression["share_min"],
        "w": eq.w, "R": eq.R, "C": eq.C, "chi": eq.chi,
        "P_agg": eq.P_agg, "L_agg": eq.L_agg, "K_agg": eq.K_agg,
        "Pi_agg": eq.Pi_agg, "wf_reference": eq.participation.wf_reference,
        "pool_binds": int(eq.participation.pool_binds.sum()),
        "empty_markets": int(eq.participation.empty_market.sum()),
        "converged": bool(eq.converged), "normalization_residual": eq.residual,
    }
