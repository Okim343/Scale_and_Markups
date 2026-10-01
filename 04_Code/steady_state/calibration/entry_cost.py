r"""Post-calibration sunk entry cost (welfare-only, EMX free-entry backout).

Under EMX-style exogenous entry the per-sector firm count ``N`` is calibrated
directly and there is **no** per-period operating cost in the static cross
section. The sunk entry cost ``F`` is therefore not used to select firms; it is
backed out *after* calibration from the dynamic free-entry condition, exactly as
in the EMX replication (``start.m:297``):

.. math::

    F \cdot W = \beta \cdot E[\pi] \big/ \big(1 - \beta(1 - \varphi)\big)
    \;\;\Longleftrightarrow\;\;
    F = \frac{\beta\, E[\pi]}{W\,(1/\beta - 1 + \varphi)}

where ``E[π]`` is the mean per-firm variable (operating) profit on the solved
cross section, ``W`` the wage, ``β`` the discount factor and ``φ = varphi`` the
exogenous exit rate. ``F`` is reported for the welfare/dynamic layer only and is
**never** fed back into the static cross-section markups or shares.
"""

from __future__ import annotations

import numpy as np

from ..model.normalization import PooledEquilibrium


def compute_sunk_entry_cost(
    eq: PooledEquilibrium,
    *,
    beta: float = 0.96,
    varphi: float = 0.04,
) -> float:
    """Back out ``F`` from the EMX free-entry condition. Welfare-only.

    ``varphi`` is the exogenous exit rate (EMX assign ``0.04``). Returns ``F`` in
    the same labor-unit/wage convention as EMX (``F = β·E[π]/(W·(1/β−1+φ))``).
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    if not 0.0 <= varphi < 1.0:
        raise ValueError("varphi must lie in [0, 1)")
    sol = eq.participation.solution
    mask = np.asarray(eq.participation.active_mask, dtype=bool)
    variable_profit = (np.asarray(sol.sales, dtype=float) - np.asarray(sol.cost, dtype=float))[mask]
    E_pi = float(variable_profit.mean())
    discount = 1.0 / beta - 1.0 + varphi
    return float(beta * E_pi / (eq.w * discount))
