"""Compact saved-fit diagnostics for pooled calibration results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .inner import PooledCalibrationResult


def calibration_fit_table(result: PooledCalibrationResult) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "moment": key,
            "model": result.model_moments[key],
            "target": result.target_moments[key],
            "relative_residual": residual,
        }
        for key, residual in zip(result.moment_keys, result.residuals, strict=True)
    ])


def write_calibration_fit_plots(
    _inputs, result: PooledCalibrationResult, out_dir: Path | str
) -> Path:
    """Write the pooled fit table; plotting is intentionally kept data-first."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    calibration_fit_table(result).to_csv(path / "pooled_fit.csv", index=False)
    return path
