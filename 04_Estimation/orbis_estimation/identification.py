"""Identification map for the single pooled calibration (EMX-style entry).

Ported from ``04_Code/steady_state/calibration/identification.py`` (a static
map; nothing imports it at run time). Updated to the five calibrated
parameters and the Orbis moment set: rho_bar is matched to the RTS gap.
"""

PARAMETERS = ("xi", "N", "gamma", "eta", "rho_bar")
MOMENTS = (
    "mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
    "alpha_rts_gap_top5_bottom50",
)
PRIMARY_ID_MAP = {
    "xi": ("cr4", "cr20", "top1pct", "top5pct"),
    "N": ("cr4", "cr20"),
    "gamma": ("mu_cw",),
    "eta": ("emx_slope",),
    "rho_bar": ("alpha_rts_gap_top5_bottom50",),
}
