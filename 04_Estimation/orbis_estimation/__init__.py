"""Calibration of the Orbis Spain model (ported from ``04_Code/steady_state/calibration``).

The model lives in ``05_Simulation/orbis_model``; the dependency runs one way,
estimation to simulation. Importing this package puts ``05_Simulation`` on
``sys.path`` (once) so ``orbis_model`` resolves from a plain checkout.
See ``04_Estimation/README.md``.
"""

from pathlib import Path
import sys

SIMULATION_DIR = Path(__file__).resolve().parents[2] / "05_Simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from .entry_cost import compute_sunk_entry_cost  # noqa: E402
from .inner import PooledCalibrationResult, calibrate_pooled  # noqa: E402

__all__ = ["PooledCalibrationResult", "calibrate_pooled", "compute_sunk_entry_cost"]
