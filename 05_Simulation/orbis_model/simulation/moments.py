"""Pooled model moments of a simulated cross section.

Ported from ``04_Code/steady_state/simulation/moments.py``. The economy moments
now include the RTS gap ``alpha_rts_gap_top5_bottom50`` (computed by
:func:`orbis_model.model.aggregator.alpha_rts_gap` on the simulated firms), and
the moments table compares to targets only when the caller passes them (the
targets live in ``04_Estimation``).
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from ..model.aggregator import POOLED_MOMENT_KEYS, pooled_inverse_markup_regression, pooled_moments
from ..primitives import ModelPrimitives
from .cross_section import SimulationResult


def pooled_moments_table(
    sim: SimulationResult,
    primitives: ModelPrimitives,
    targets: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """One row per pooled moment; adds a ``target`` column when targets are given."""
    model = economy_moments(sim, primitives)
    keys = tuple(targets) if targets is not None else (*POOLED_MOMENT_KEYS, "mu_cw_alpha")
    rows = []
    for key in keys:
        row = {"moment": key, "model": model.get(key)}
        if targets is not None:
            row["target"] = float(targets[key])
        rows.append(row)
    return pd.DataFrame(rows)


def economy_moments(sim: SimulationResult, primitives: ModelPrimitives) -> dict[str, float]:
    eq = sim.equilibrium
    moments = pooled_moments(
        eq.participation.solution, eq.participation.active_mask,
        n_firms_cs=primitives.n_firms_cs, phi_v=primitives.phi_v, alpha=eq.alpha,
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
