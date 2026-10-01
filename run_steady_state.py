"""Project-root launcher for the steady_state CLI.

The package lives under ``04_Code/steady_state/``. This shim prepends
``04_Code/`` to ``sys.path`` so the package is importable from the project
root, then forwards to :mod:`steady_state.__main__`.

Usage from the project root:

    python run_steady_state.py calibrate --M 1000 --outer-maxiter 5
    python run_steady_state.py simulate --calibration 04_Code/out_results/calibration_2010_2019.yaml
    python run_steady_state.py validate --calibration 04_Code/out_results/calibration_2010_2019.yaml

For a more permanent setup, add a ``pyproject.toml`` and ``pip install -e .``
to make ``python -m steady_state ...`` work from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_PARENT = Path(__file__).resolve().parent / "04_Code"
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from steady_state.__main__ import main  # noqa: E402  (sys.path mutated above)


if __name__ == "__main__":
    raise SystemExit(main())
