"""Calibration targets from the Orbis Spain bundle.

Replaces the target half of ``04_Code/steady_state/io_bundle.py``. Reads
``<bundle_dir>/aggregate_moments.yaml`` and returns the targets under their
Orbis names, which are also the model moment names:

* ``targets.mu_cw``: (V + kexp)-weighted DLEU markup (V = mate + staf, no SG&A),
  the target of the model's PURE cost-weighted markup ``sum TC mu / sum TC``.
  There is no markup-target switch and no SG&A variant.
* ``targets.cr4, cr20, top1pct, top5pct``: within-NAICS4 concentration.
* ``targets.emx_slope``: NAICS2 long-difference inverse-markup on HHI slope.
* ``alpha_rts_gap_top5_bottom50`` (top level): RTS gap, top 5% minus bottom 50%
  of the within-NAICS4 revenue distribution; the rho_bar target.

Definitions: ``02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md``.
"""

from __future__ import annotations

from pathlib import Path

from orbis_model.model.aggregator import RTS_GAP_KEY
from orbis_model.primitives import read_aggregate_moments


TARGET_KEYS = ("mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope")


def load_targets(bundle_dir: Path | str) -> dict[str, float]:
    """The seven calibration targets keyed by model moment name."""
    agg = read_aggregate_moments(bundle_dir)
    if not isinstance(agg.get("targets"), dict):
        raise ValueError("aggregate_moments.yaml: missing 'targets' mapping")
    missing = [k for k in TARGET_KEYS if k not in agg["targets"]]
    if missing:
        raise ValueError(f"aggregate_moments.yaml: targets missing {missing}")
    if RTS_GAP_KEY not in agg:
        raise ValueError(f"aggregate_moments.yaml: missing {RTS_GAP_KEY!r}")
    targets = {k: float(agg["targets"][k]) for k in TARGET_KEYS}
    targets[RTS_GAP_KEY] = float(agg[RTS_GAP_KEY])
    return targets
