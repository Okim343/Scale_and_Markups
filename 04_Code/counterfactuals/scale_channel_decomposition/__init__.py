"""Scale-channel decomposition counterfactual.

Companion to :mod:`counterfactuals.scalability_sorting` and
:mod:`counterfactuals.homogeneous_rts`. Those two decompose only the
*reallocation* leg ``lambda_K`` (MARKET -> FIXED-K PLANNER) by alpha-arrangement.
This package runs the **full free-capital PLANNER** leg -- deliberately skipped
there (``pe=None``) -- under each same arrangement to obtain ``lambda_total``,
then backs out the *scale* leg ``lambda_scale`` and assembles the additive 3x3
``{Delta_K, Delta_total, Delta_scale} x {common-alpha, heterogeneity, sorting}``
table on the ``Delta = log(1+lambda)`` scale, where ``Delta_scale = Delta_total
- Delta_K`` holds exactly arrangement by arrangement.
"""
