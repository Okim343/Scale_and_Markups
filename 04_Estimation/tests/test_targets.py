"""The target reader (orbis_estimation.targets) and the Orbis bundle contract."""

from __future__ import annotations

import copy

import pytest

from conftest import SYNTHETIC_MOMENTS, write_bundle
from orbis_estimation.targets import TARGET_KEYS, load_targets


def test_targets_keep_orbis_names(tmp_path):
    targets = load_targets(write_bundle(tmp_path / "b"))
    assert set(targets) == {*TARGET_KEYS, "alpha_rts_gap_top5_bottom50"}
    # The data mu_cw is the target of the model's pure mu_cw: no remap, no sga.
    assert targets["mu_cw"] == pytest.approx(1.18)
    assert "mu_cw_alpha" not in targets and "mu_cw_sga" not in targets
    assert targets["alpha_rts_gap_top5_bottom50"] == pytest.approx(0.05)


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_missing_target_raises(tmp_path, key):
    moments = copy.deepcopy(SYNTHETIC_MOMENTS)
    del moments["targets"][key]
    with pytest.raises(ValueError, match=key):
        load_targets(write_bundle(tmp_path / "b", moments=moments))


def test_missing_rts_gap_raises(tmp_path):
    moments = copy.deepcopy(SYNTHETIC_MOMENTS)
    del moments["alpha_rts_gap_top5_bottom50"]
    with pytest.raises(ValueError, match="alpha_rts_gap_top5_bottom50"):
        load_targets(write_bundle(tmp_path / "b", moments=moments))


def test_missing_targets_block_raises(tmp_path):
    moments = copy.deepcopy(SYNTHETIC_MOMENTS)
    del moments["targets"]
    with pytest.raises(ValueError, match="targets"):
        load_targets(write_bundle(tmp_path / "b", moments=moments))


def test_dropped_compustat_keys_are_not_required(tmp_path):
    moments = copy.deepcopy(SYNTHETIC_MOMENTS)
    del moments["validation"]
    for key in ("alpha_z_copula_rho_bar", "alpha_sales_corr_empirical",
                "alpha_sales_corr_partial_empirical", "hybrid_alpha", "reference_scale"):
        assert key not in moments
    assert len(load_targets(write_bundle(tmp_path / "b", moments=moments))) == 7


def test_real_orbis_targets(real_bundle):
    targets = load_targets(real_bundle)
    assert targets["mu_cw"] == pytest.approx(1.18155, abs=1e-5)
    assert targets["cr4"] == pytest.approx(0.32850, abs=1e-5)
    assert targets["emx_slope"] == pytest.approx(-0.33443, abs=1e-5)
    assert targets["alpha_rts_gap_top5_bottom50"] == pytest.approx(0.04597, abs=1e-5)
