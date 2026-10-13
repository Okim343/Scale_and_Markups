"""Model primitives from the Orbis Spain calibration bundle.

Replaces the model-input half of ``04_Code/steady_state/io_bundle.py`` and
``pooled_inputs.py``. Reads from ``<bundle_dir>`` (built by
``03_Empirical/02_Orbis/04_Moments/01_code/07_assemble_bundle.py``):

* ``F_alpha.parquet``: a single pooled scalability distribution with columns
  ``node_id, u, weight, alpha`` (equal-mass nodes, no ``sector_id``). The
  ``alpha`` column is the model-facing support and is sampled verbatim: raw GNR
  returns to scale, no clip or shrink.
* ``aggregate_moments.yaml``: the model inputs ``active_window``, ``a`` (capital
  share of value added, in (0, 1)), ``phi_v`` (gross-output value-added weight,
  in (0, 1]) and ``n_firms_cs`` (population firm count); optional
  ``mean_firms_per_sector`` (a diagnostic analogue of ``N``, never an initial
  value).

Targets are not read here; they belong to ``04_Estimation`` (``targets.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


F_ALPHA_REQUIRED_COLS = ("node_id", "u", "weight", "alpha")
REQUIRED_INPUT_KEYS = ("active_window", "a", "phi_v", "n_firms_cs")


def read_aggregate_moments(bundle_dir: Path | str) -> Mapping[str, Any]:
    """Parse ``aggregate_moments.yaml``; shared by the primitive and target readers."""
    path = Path(bundle_dir) / "aggregate_moments.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Bundle artifact missing: {path}")
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} is not a mapping")
    return data


def read_f_alpha(bundle_dir: Path | str) -> pd.DataFrame:
    """Read and validate the pooled ``F_alpha.parquet`` node table."""
    path = Path(bundle_dir) / "F_alpha.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Bundle artifact missing: {path}")
    f_alpha = pd.read_parquet(path)
    missing = [c for c in F_ALPHA_REQUIRED_COLS if c not in f_alpha.columns]
    if missing:
        raise ValueError(f"F_alpha.parquet missing columns: {missing}")
    if "sector_id" in f_alpha.columns:
        raise ValueError(
            "F_alpha.parquet carries a 'sector_id' column; the model expects a "
            "SINGLE pooled distribution (no sector dimension)."
        )
    if len(f_alpha) < 1:
        raise ValueError("F_alpha.parquet is empty")
    alpha = f_alpha["alpha"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("F_alpha.parquet: alpha must be finite and positive")
    return f_alpha.sort_values("node_id").reset_index(drop=True)


@dataclass(frozen=True)
class ModelPrimitives:
    """Externally assigned model inputs: the alpha support and pooled scalars.

    ``H`` is the per-market slot pool from the simulation config (an assigned
    parameter, not a bundle input); it is carried here because every solve
    needs it next to the bundle scalars.
    """

    alpha_support: np.ndarray
    a: float
    phi_v: float
    n_firms_cs: float
    H: int
    active_window: tuple[int, int]
    mean_firms_per_sector: float | None = None
    bundle_dir: Path | None = None

    @property
    def window_tag(self) -> str:
        """``<start>_<end>`` label for output file names (e.g. ``2015_2019``)."""
        start, end = self.active_window
        return f"{start}_{end}"


def load_primitives(bundle_dir: Path | str, *, H: int) -> ModelPrimitives:
    """Read and validate the model inputs of the bundle.

    Raises ``FileNotFoundError`` if an artifact is missing and ``ValueError``
    if a required key is missing or out of range.
    """
    bundle_dir = Path(bundle_dir).resolve()
    f_alpha = read_f_alpha(bundle_dir)
    agg = read_aggregate_moments(bundle_dir)
    missing = [k for k in REQUIRED_INPUT_KEYS if k not in agg]
    if missing:
        raise ValueError(f"aggregate_moments.yaml: missing model inputs {missing}")
    window = agg["active_window"]
    if not isinstance(window, Mapping) or not {"start", "end"} <= set(window):
        raise ValueError("aggregate_moments.yaml: active_window needs start and end")
    a = float(agg["a"])
    if not 0.0 < a < 1.0:
        raise ValueError(f"aggregate_moments.yaml: 'a' must lie in (0, 1); got {a}")
    phi_v = float(agg["phi_v"])
    if not 0.0 < phi_v <= 1.0:
        raise ValueError(f"aggregate_moments.yaml: 'phi_v' must lie in (0, 1]; got {phi_v}")
    n_firms_cs = float(agg["n_firms_cs"])
    if not n_firms_cs > 0.0:
        raise ValueError(f"aggregate_moments.yaml: 'n_firms_cs' must be positive; got {n_firms_cs}")
    mean_firms = agg.get("mean_firms_per_sector")
    if int(H) <= 0:
        raise ValueError("H must be positive")
    return ModelPrimitives(
        alpha_support=f_alpha["alpha"].to_numpy(dtype=np.float64),
        a=a,
        phi_v=phi_v,
        n_firms_cs=n_firms_cs,
        H=int(H),
        active_window=(int(window["start"]), int(window["end"])),
        mean_firms_per_sector=float(mean_firms) if mean_firms is not None else None,
        bundle_dir=bundle_dir,
    )
