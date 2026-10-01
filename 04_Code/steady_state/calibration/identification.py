"""Identification map for the single pooled calibration (EMX-style entry)."""

PARAMETERS = ("xi", "N", "gamma", "eta")
MOMENTS = (
    "mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
)
PRIMARY_ID_MAP = {
    "xi": ("cr4", "cr20", "top1pct", "top5pct"),
    "N": ("cr4", "cr20"),
    "gamma": ("mu_cw",),
    "eta": ("emx_slope",),
}
