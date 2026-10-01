"""Regime-level table and welfare split for the fixed-capital counterfactual.

Sibling of :func:`steady_state.welfare.welfare_metrics.welfare_decomposition_table`
/ :func:`steady_state.welfare.welfare_metrics.two_channel_decomposition`, which
are hardcoded to the market/uniform/planner 3-way comparison and do not
generalize to a 4th regime. These functions reuse
:func:`steady_state.model.aggregator.pooled_moments` and the welfare-metric
helpers unchanged; nothing under ``steady_state/`` is modified.

The decomposition splits the existing ``lambda_total`` (MARKET -> PLANNER) into:

- ``lambda_reallocation`` (MARKET -> FIXED_K_PLANNER): reallocation at a pinned
  aggregate capital stock -- the object structurally comparable in spirit to a
  Hsieh-Klenow-style static misallocation number / the reference paper's static
  Table 6.
- ``lambda_scale`` (FIXED_K_PLANNER -> PLANNER, only when the full planner is
  supplied): the incremental gain from additionally letting capital expand at
  the fixed Euler rate -- the ~2.14x K effect that inflated the headline number.

with the multiplicative identity check
``(1+lambda_reallocation)(1+lambda_scale) = (1+lambda_total)`` mirroring
:func:`two_channel_decomposition`.
"""

from __future__ import annotations

import pandas as pd

from steady_state.model.aggregator import pooled_moments
from steady_state.model.normalization import PooledEquilibrium
from steady_state.welfare.welfare_metrics import (
    aggregate_markup_cw,
    consumption_equivalent_lambda,
    steady_state_welfare,
)


def fixed_capital_table(
    me: PooledEquilibrium,
    fp: PooledEquilibrium,
    pe: PooledEquilibrium | None = None,
    *,
    n_firms_cs: float,
) -> pd.DataFrame:
    """Regime-level aggregates for MARKET, FIXED_K_PLANNER, (optional) PLANNER.

    Same schema as
    :func:`steady_state.welfare.welfare_metrics.welfare_decomposition_table`
    plus an ``R`` column (the whole point of this counterfactual is that ``R``
    varies across regimes here).
    """
    regimes = [("market", me), ("planner_fixed_k", fp)]
    if pe is not None:
        regimes.append(("planner", pe))
    rows = []
    for label, eq in regimes:
        m = pooled_moments(
            eq.participation.solution, eq.participation.active_mask,
            n_firms_cs=n_firms_cs,
        )
        rows.append({
            "regime": label, "C": eq.C, "L": eq.L_agg, "K": eq.K_agg,
            "R": eq.R, "profits": eq.Pi_agg,
            "mu_cw": m["mu_cw"],              # pure cost-weighted markup
            "mu_cw_alpha": m["mu_cw_alpha"],  # accounting Sigma sales / Sigma cost
            "mean_active_count": eq.aggregates.mean_active_count,
            "wf_reference": eq.participation.wf_reference,
        })
    return pd.DataFrame(rows)


def fixed_capital_decomposition(
    me: PooledEquilibrium,
    fp: PooledEquilibrium,
    pe: PooledEquilibrium | None = None,
    *,
    chi_me: float,
    beta: float,
    phi: float,
    n_firms_cs: float,
) -> dict:
    """Consumption-equivalent welfare split for the fixed-capital counterfactual.

    ``lambda_reallocation`` is MARKET -> FIXED_K_PLANNER. When the full planner
    ``pe`` is supplied, also returns ``lambda_scale`` (FIXED_K_PLANNER ->
    PLANNER), ``lambda_total`` (MARKET -> PLANNER), and the multiplicative
    ``identity_residual``. ``chi_me`` (the MARKET labor-disutility weight) is
    held fixed across regimes exactly as in
    :func:`two_channel_decomposition`.
    """
    K_target = float(me.K_agg)
    W_me = steady_state_welfare(me.C, me.L_agg, chi_me, beta, phi)
    W_fp = steady_state_welfare(fp.C, fp.L_agg, chi_me, beta, phi)
    lambda_reallocation = consumption_equivalent_lambda(W_me, W_fp, beta)

    out: dict = {
        "W_market": float(W_me),
        "W_planner_fixed_k": float(W_fp),
        "lambda_reallocation": float(lambda_reallocation),
        "K_target": K_target,
        "K_target_gap": float(fp.K_agg / K_target - 1.0),
        "R_market": float(me.R),
        "R_planner_fixed_k": float(fp.R),
        "mu_cw_planner_fixed_k": float(aggregate_markup_cw(fp, n_firms_cs=n_firms_cs)),
    }

    if pe is not None:
        W_pe = steady_state_welfare(pe.C, pe.L_agg, chi_me, beta, phi)
        lambda_scale = consumption_equivalent_lambda(W_fp, W_pe, beta)
        lambda_total = consumption_equivalent_lambda(W_me, W_pe, beta)
        identity_residual = (
            (1.0 + lambda_reallocation) * (1.0 + lambda_scale)
            - (1.0 + lambda_total)
        )
        out.update({
            "W_planner": float(W_pe),
            "lambda_scale": float(lambda_scale),
            "lambda_total": float(lambda_total),
            "identity_residual": float(identity_residual),
        })

    return out
