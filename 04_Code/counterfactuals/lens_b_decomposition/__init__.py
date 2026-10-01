"""Lens-B (dispersion x level) per-arrangement welfare decomposition.

Standalone sibling of :mod:`counterfactuals.scale_channel_decomposition`. Runs
AFTER the scale-channel run and reuses its cached full-planner / market welfare
levels, so the only NEW solves are the UNIFORM legs. See ``run_lens_b.py``.
"""
