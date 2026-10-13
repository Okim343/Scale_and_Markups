"""Params-file round trip and the bundle sha256 check."""

from __future__ import annotations

import pytest
import yaml

from conftest import SYNTHETIC_MOMENTS, write_bundle
from orbis_model.model.normalization import PooledParams
from orbis_model.params import (
    SCHEMA,
    build_provenance,
    bundle_hashes,
    load_params_file,
    write_params_file,
)

PARAMS = PooledParams(xi=8.0, N=3.0, gamma=3.8, eta=2.0, v_min=1.7, y_hat=1.0,
                      rho_bar=-0.3, a=0.35, H=5, phi_v=0.45)
REFERENCE = {"w": 1.2, "R": 0.1, "X_market": 0.8, "wf_reference": 0.0,
             "Ed_full_pool": 0.01, "P": 1.0, "L": 1.0}


def _write(tmp_path, bundle):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n")
    prov = build_provenance(bundle_dir=bundle, configs={"simulation": cfg},
                            code_dirs=(), script="test")
    return write_params_file(tmp_path / "params.yaml", PARAMS, reference=REFERENCE,
                             provenance=prov, fit={"cost": 0.5})


def test_round_trip(tmp_path, synthetic_bundle):
    path = _write(tmp_path, synthetic_bundle)
    params, wf, initial, data = load_params_file(path, bundle_dir=synthetic_bundle)
    assert params == PARAMS
    assert wf == 0.0
    assert initial == (1.2, 0.8)
    assert data["schema"] == SCHEMA
    assert data["fit"]["cost"] == 0.5
    assert data["provenance"]["bundle_sha256"] == bundle_hashes(synthetic_bundle)
    assert set(data["provenance"]) >= {"bundle_manifest_sha256", "configs", "commit", "code_dirty"}


def test_changed_bundle_is_refused(tmp_path, synthetic_bundle):
    path = _write(tmp_path, synthetic_bundle)
    other = write_bundle(tmp_path / "other", moments={**SYNTHETIC_MOMENTS, "n_firms_cs": 101.0})
    with pytest.raises(ValueError, match="different bundle.*aggregate_moments.yaml"):
        load_params_file(path, bundle_dir=other)


def test_changed_f_alpha_is_refused(tmp_path, synthetic_bundle):
    import pandas as pd

    path = _write(tmp_path, synthetic_bundle)
    f_alpha = pd.DataFrame({"node_id": [0, 1], "u": [0.25, 0.75], "weight": 0.5, "alpha": [0.9, 1.1]})
    other = write_bundle(tmp_path / "other", f_alpha=f_alpha)
    with pytest.raises(ValueError, match="F_alpha.parquet"):
        load_params_file(path, bundle_dir=other)


def test_missing_provenance_is_refused(tmp_path, synthetic_bundle):
    path = _write(tmp_path, synthetic_bundle)
    data = yaml.safe_load(path.read_text())
    del data["provenance"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="no provenance.bundle_sha256"):
        load_params_file(path, bundle_dir=synthetic_bundle)


def test_wrong_schema_is_refused(tmp_path, synthetic_bundle):
    path = _write(tmp_path, synthetic_bundle)
    data = yaml.safe_load(path.read_text())
    data["schema"] = "pooled-emx-v2"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unsupported params schema"):
        load_params_file(path, bundle_dir=synthetic_bundle)


def test_provenance_requires_bundle_manifest(tmp_path, synthetic_bundle):
    (synthetic_bundle / "manifest.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        _write(tmp_path, synthetic_bundle)
