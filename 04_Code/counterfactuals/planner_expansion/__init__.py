"""Planner-expansion incidence (descriptive, non-invasive).

Reads the stored MARKET and PLANNER firm panels of the baseline welfare run and
asks *where* the planner expands input demand relative to the market: by alpha
quintile, by nu quintile, on an alpha-by-nu double sort, by within-sector size
rank, and in a sector-fixed-effects cross-firm regression of
``log(TC_planner / TC_market)`` on ``log(mu_market)``, ``alpha`` and ``log(v)``.

Within a regime every primary and intermediate input is a common fraction of a
firm's total cost (``k = a*phi_v*TC/R``, ``l = (1-a)*phi_v*TC/W``,
``m = (1-phi_v)*TC/P`` with scalar ``a``, ``phi_v``), so a firm's share of
aggregate inputs equals its share of aggregate ``TC``. Nothing is re-solved and
nothing under ``steady_state/`` is imported.
"""
