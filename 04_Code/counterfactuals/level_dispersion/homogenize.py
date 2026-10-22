"""Collapse capability v within each sector, the TFP twin of homogeneous_rts."""
from __future__ import annotations

import dataclasses

import numpy as np

from steady_state.model.pool import PoolDraw


def homogenize_v_primitive(base_draw: PoolDraw) -> PoolDraw:
    """Replace every active v with its sector's harmonic mean.

    ``v_hom_j = ((1/n_j) sum_i 1/v_ji)^-1`` keeps each sector's mean anchor
    marginal cost ``mean(Omega/v)`` exactly, as the alpha transform keeps
    ``mean(1/alpha)``. Alpha, tilde_alpha and the mask are untouched; singleton
    sectors and inactive slots keep their base v bit for bit.
    """
    v_base = np.asarray(base_draw.v, dtype=np.float64)
    mask = np.asarray(base_draw.active_mask, dtype=bool)
    v = v_base.copy()
    for m in range(v.shape[0]):
        active = np.nonzero(mask[m])[0]
        if active.size > 1:
            v[m, active] = 1.0 / np.mean(1.0 / v_base[m, active])
    new_draw = dataclasses.replace(base_draw, v=v)
    _assert_v_homogenized(new_draw, base_draw)
    return new_draw


def _assert_v_homogenized(new_draw: PoolDraw, base_draw: PoolDraw) -> None:
    for field in ("alpha", "active_mask"):
        if not np.array_equal(getattr(new_draw, field), getattr(base_draw, field), equal_nan=False):
            raise AssertionError(f"homogenize_v_primitive changed {field}")
    mask = np.asarray(base_draw.active_mask, dtype=bool)
    if not np.array_equal(new_draw.v[~mask], base_draw.v[~mask]):
        raise AssertionError("homogenize_v_primitive changed inactive v")
    for m in range(mask.shape[0]):
        active = np.nonzero(mask[m])[0]
        if active.size == 0:
            continue
        v_new, v_old = new_draw.v[m, active], base_draw.v[m, active]
        if active.size > 1 and np.ptp(v_new) != 0.0:
            raise AssertionError(f"nonzero v dispersion left in sector {m}")
        if not np.isclose(np.mean(1.0 / v_new), np.mean(1.0 / v_old), rtol=1e-12, atol=0):
            raise AssertionError(f"mean anchor marginal cost changed in sector {m}")
