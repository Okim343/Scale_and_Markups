"""Anchor-scale diagnostics for the pooled calibration."""

from __future__ import annotations

import numpy as np

from ..model.normalization import PooledEquilibrium
from ..model.pool import derived_z


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok] - float(np.mean(x[ok]))
    y = y[ok] - float(np.mean(y[ok]))
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 0.0 else float("nan")


def _partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if ok.sum() < controls.shape[1] + 3:
        return float("nan")
    X = np.column_stack([np.ones(ok.sum()), controls[ok]])
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    return _corr(rx, ry)


def anchor_diagnostics(eq: PooledEquilibrium) -> dict[str, float]:
    """Return calibration diagnostics at the active-firm level."""
    mask = eq.participation.active_mask
    alpha = np.asarray(eq.alpha, dtype=float)[mask]
    v = np.asarray(eq.v, dtype=float)[mask]
    sales = np.asarray(eq.participation.solution.sales, dtype=float)[mask]
    output = np.asarray(eq.participation.solution.output, dtype=float)[mask]
    log_sales = np.log(np.maximum(sales, 1e-300))
    log_y_over_yhat = np.log(np.maximum(output / eq.y_hat, 1e-300))
    log_v = np.log(np.maximum(v, 1e-300))
    z_diag = derived_z(alpha, v, y_hat=eq.y_hat)
    log_z_diag = np.log(np.maximum(z_diag, 1e-300))
    sales_weight = sales / max(float(np.sum(sales)), 1e-300)
    centered = log_y_over_yhat - float(np.dot(sales_weight, log_y_over_yhat))
    return {
        "corr_alpha_log_sales": _corr(alpha, log_sales),
        "corr_alpha_log_sales_partial_z": _partial_corr(alpha, log_sales, log_z_diag),
        "corr_alpha_log_sales_partial_v": _partial_corr(alpha, log_sales, log_v),
        "median_log_y_over_yhat": float(np.median(log_y_over_yhat)),
        "mean_log_y_over_yhat": float(np.mean(log_y_over_yhat)),
        "sales_weighted_sd_log_y_over_yhat": float(np.sqrt(np.dot(sales_weight, centered * centered))),
        "y_sw": float(eq.y_sw),
        "y_anchor": float(eq.y_anchor),
    }
