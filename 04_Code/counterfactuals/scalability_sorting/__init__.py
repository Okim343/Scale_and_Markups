"""Scalability-sorting counterfactual (non-invasive).

Perturbs the assignment of scalability ``alpha`` to latent capability ``v`` in a
calibrated :class:`~steady_state.model.pool.PoolDraw` *before* the market
equilibrium is solved, then runs the existing fixed-capital-envelope pipeline
(``counterfactuals.fixed_input_welfare``) on each perturbed economy. Nothing
under ``steady_state/`` is modified and ``fixed_capital_planner`` /
``decomposition`` are imported unchanged.

See ``IMPLEMENTATION_PLAN.md`` for the design.
"""

from .permute import active_corr_av, permute_alpha

__all__ = ["permute_alpha", "active_corr_av"]
