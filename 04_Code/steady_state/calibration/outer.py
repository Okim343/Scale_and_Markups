"""Pooled calibration entry point and inverse-markup slope producer."""

from __future__ import annotations

from ..model.aggregator import pooled_inverse_markup_regression
from .inner import PooledCalibrationResult, calibrate_pooled


def _pooled_emx_slope_from_panels(solution, active_mask) -> float:
    """Retained slope-moment helper, now operating on pooled active firms."""
    return pooled_inverse_markup_regression(solution, active_mask)["slope"]


__all__ = ["PooledCalibrationResult", "calibrate_pooled", "_pooled_emx_slope_from_panels"]
