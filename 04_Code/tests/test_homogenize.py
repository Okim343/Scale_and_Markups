"""Pure-function checks for the homogeneous-scalability alpha transforms."""

from __future__ import annotations

import numpy as np
import pytest

from counterfactuals.homogeneous_rts.homogenize import (
    homogenize_alpha_primitive,
    sector_mean_inv_alpha,
)
from steady_state.model.pool import draw_pool


def _small_draw():
    support = np.linspace(0.6, 1.2, 25)
    return draw_pool(
        support, xi=8.0, rho_bar=0.5, N=4.0, M=6, H=8, rng=1234, v_min=1.0,
    )


def test_homogenize_primitive_preserves_mask_and_v():
    base = _small_draw()
    hom = homogenize_alpha_primitive(base)
    np.testing.assert_array_equal(hom.active_mask, base.active_mask)
    np.testing.assert_array_equal(hom.v, base.v)


def test_homogenize_primitive_zeroes_within_sector_variance():
    base = _small_draw()
    hom = homogenize_alpha_primitive(base)
    mask = np.asarray(hom.active_mask, dtype=bool)
    for m in range(hom.alpha.shape[0]):
        active = np.nonzero(mask[m])[0]
        assert np.isclose(np.var(hom.alpha[m, active]), 0.0, atol=1e-12)


def test_homogenize_primitive_preserves_sector_mean_inv_alpha():
    base = _small_draw()
    hom = homogenize_alpha_primitive(base)
    np.testing.assert_allclose(
        sector_mean_inv_alpha(hom), sector_mean_inv_alpha(base), rtol=1e-9,
    )


def test_homogenize_primitive_leaves_singleton_sectors_unchanged():
    support = np.linspace(0.6, 1.2, 25)
    base = draw_pool(support, xi=8.0, rho_bar=0.5, N=1.0, M=30, H=4, rng=7,
                     v_min=1.0)
    hom = homogenize_alpha_primitive(base)
    mask = np.asarray(base.active_mask, dtype=bool)
    singleton = mask.sum(axis=1) == 1
    if not singleton.any():
        pytest.skip("no singleton sectors drawn at this seed")
    np.testing.assert_array_equal(hom.alpha[singleton], base.alpha[singleton])
