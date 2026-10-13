"""The model-input reader (orbis_model.primitives) and the Orbis bundle contract."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from conftest import SYNTHETIC_MOMENTS, write_bundle
from orbis_model.primitives import load_primitives


def _moments(**changes):
    moments = copy.deepcopy(SYNTHETIC_MOMENTS)
    for key, value in changes.items():
        if value is None:
            moments.pop(key)
        else:
            moments[key] = value
    return moments


def test_load_primitives_reads_model_inputs(synthetic_bundle):
    prim = load_primitives(synthetic_bundle, H=7)
    assert prim.a == pytest.approx(0.35)
    assert prim.phi_v == pytest.approx(0.45)
    assert prim.n_firms_cs == pytest.approx(100.0)
    assert prim.mean_firms_per_sector == pytest.approx(3.0)
    assert prim.active_window == (2015, 2019)
    assert prim.window_tag == "2015_2019"
    assert prim.H == 7
    # The support is the F_alpha alpha column verbatim (no clip, no shrink).
    np.testing.assert_array_equal(prim.alpha_support, np.linspace(0.75, 1.05, 10))


def test_support_is_sorted_by_node_id_and_not_transformed(tmp_path):
    f_alpha = pd.DataFrame({
        "node_id": [3, 1, 2], "u": [5 / 6, 1 / 6, 0.5], "weight": 1 / 3,
        "alpha": np.array([1.26, 0.71, 0.97], dtype=np.float32),
    })
    prim = load_primitives(write_bundle(tmp_path / "b", f_alpha=f_alpha), H=3)
    np.testing.assert_array_equal(prim.alpha_support, np.array([0.71, 0.97, 1.26], dtype=np.float32))
    assert prim.alpha_support.dtype == np.float64


@pytest.mark.parametrize("key", ["active_window", "a", "phi_v", "n_firms_cs"])
def test_missing_required_input_raises(tmp_path, key):
    bundle = write_bundle(tmp_path / "b", moments=_moments(**{key: None}))
    with pytest.raises(ValueError, match=key):
        load_primitives(bundle, H=5)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.3])
def test_bad_capital_share_raises(tmp_path, bad):
    bundle = write_bundle(tmp_path / "b", moments=_moments(a=bad))
    with pytest.raises(ValueError, match="'a' must lie in"):
        load_primitives(bundle, H=5)


@pytest.mark.parametrize("bad", [0.0, 1.2])
def test_bad_phi_v_raises(tmp_path, bad):
    bundle = write_bundle(tmp_path / "b", moments=_moments(phi_v=bad))
    with pytest.raises(ValueError, match="phi_v"):
        load_primitives(bundle, H=5)


def test_phi_v_equal_to_one_is_allowed(tmp_path):
    bundle = write_bundle(tmp_path / "b", moments=_moments(phi_v=1.0))
    assert load_primitives(bundle, H=5).phi_v == 1.0


def test_sector_id_column_raises(tmp_path):
    f_alpha = pd.DataFrame({"node_id": [0, 1], "u": [0.25, 0.75], "weight": 0.5,
                            "alpha": [0.9, 1.1], "sector_id": [1, 2]})
    bundle = write_bundle(tmp_path / "b", f_alpha=f_alpha)
    with pytest.raises(ValueError, match="sector_id"):
        load_primitives(bundle, H=5)


@pytest.mark.parametrize("column", ["node_id", "u", "weight", "alpha"])
def test_missing_f_alpha_column_raises(tmp_path, column):
    f_alpha = pd.DataFrame({"node_id": [0, 1], "u": [0.25, 0.75], "weight": 0.5,
                            "alpha": [0.9, 1.1]}).drop(columns=column)
    bundle = write_bundle(tmp_path / "b", f_alpha=f_alpha)
    with pytest.raises(ValueError, match="missing columns"):
        load_primitives(bundle, H=5)


def test_missing_file_raises(tmp_path):
    bundle = write_bundle(tmp_path / "b")
    (bundle / "F_alpha.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        load_primitives(bundle, H=5)


def test_dropped_compustat_keys_are_not_required(tmp_path):
    # The Orbis bundle has none of these; the reader must not ask for them.
    moments = _moments(validation=None, mean_firms_per_sector=None)
    for key in ("alpha_z_copula_rho_bar", "alpha_sales_corr_empirical",
                "alpha_sales_corr_partial_empirical", "hybrid_alpha", "reference_scale"):
        assert key not in moments
    assert "mu_cw_sga" not in moments["targets"]
    prim = load_primitives(write_bundle(tmp_path / "b", moments=moments), H=5)
    assert prim.mean_firms_per_sector is None


def test_primitives_ignore_targets(tmp_path):
    # Targets are the estimation's business: a bundle without them still loads here.
    bundle = write_bundle(tmp_path / "b", moments=_moments(targets=None, alpha_rts_gap_top5_bottom50=None))
    assert load_primitives(bundle, H=5).a == pytest.approx(0.35)


def test_real_orbis_bundle_loads(real_bundle):
    prim = load_primitives(real_bundle, H=2500)
    assert prim.active_window == (2015, 2019)
    assert prim.alpha_support.size == 500
    assert 0.70 < prim.alpha_support.min() < prim.alpha_support.max() < 1.27
    assert prim.a == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert 0.0 < prim.phi_v < 1.0
    assert prim.mean_firms_per_sector > 2500  # exceeds H: diagnostic only
