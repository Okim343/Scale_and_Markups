"""The 01_calibrate / 02_validate scripts end to end, config plumbing, and the handoff.

Ported from the calibrate/validate parts of ``04_Code/tests/test_simulation_cli.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from conftest import EST_DIR, WEIGHTS, write_configs
from orbis_estimation.config import load_config
from orbis_model.params import bundle_hashes, load_params_file


def _script(path: Path):
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibrate():
    return _script(EST_DIR / "01_calibrate.py")


def _validate():
    return _script(EST_DIR / "02_validate.py")


def test_configured_bounds_weights_and_initial_values_reach_the_optimizer(tmp_path, monkeypatch):
    est_cfg, _, _ = write_configs(tmp_path, rho_bar_bounds=(-0.8, 0.7), rho_bar_init=-0.2, max_nfev=7)
    module = _calibrate()
    captured = {}

    class Stop(Exception):
        pass

    def fake_calibrate(primitives, targets, **kwargs):
        captured.update(kwargs, targets=targets)
        raise Stop

    monkeypatch.setattr(module, "calibrate_pooled", fake_calibrate)
    with pytest.raises(Stop):
        module.main(["--config", str(est_cfg)])
    assert captured["rho_bar_bounds"] == (-0.8, 0.7)
    assert captured["N_bounds"] == (2.0, 4.0)
    assert captured["xi_bounds"] == (1.05, 30.0)
    assert captured["eta_bounds"] == (1.01, 10.0)
    assert captured["gamma_gap_bounds"] == (0.01, 30.0)
    assert captured["initial"] == (8.0, 3.0, 3.8, 2.0, -0.2)
    assert captured["max_nfev"] == 7
    assert captured["weights"] == WEIGHTS
    assert captured["moment_keys"] == tuple(WEIGHTS)
    assert captured["M"] == 2 and captured["rng"] == 3


def test_cli_overrides_reach_the_optimizer_without_editing_config(tmp_path, monkeypatch):
    est_cfg, _, _ = write_configs(tmp_path)
    module = _calibrate()
    captured = {}

    def fake_calibrate(primitives, targets, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "calibrate_pooled", fake_calibrate)
    with pytest.raises(RuntimeError):
        module.main(["--config", str(est_cfg), "--M", "9", "--max-nfev", "2", "--seed", "11"])
    assert (captured["M"], captured["max_nfev"], captured["rng"]) == (9, 2, 11)


def test_configured_rho_bar_bounds_bind_in_the_transform(tmp_path):
    from orbis_estimation.inner import _decode

    est_cfg, _, _ = write_configs(tmp_path, rho_bar_bounds=(-0.8, 0.7))
    bounds = load_config(est_cfg).bounds["rho_bar"]
    for x, edge in ((-60.0, -0.8), (60.0, 0.7)):
        assert _decode([2.0, 1.0, 0.0, 0.5, x], a=0.35, H=5, rho_bar_bounds=bounds).rho_bar == pytest.approx(edge)


def test_shipped_config_defaults():
    cfg = load_config()
    assert cfg.bounds["rho_bar"] == (-0.95, 0.95)
    assert cfg.bounds["N"] == (5.0, 1000.0)
    assert cfg.initial_values["rho_bar"] == 0.0
    assert cfg.initial_values["N"] == 149.586
    assert cfg.weights == WEIGHTS
    assert cfg.simulation.parameters.H == 2500
    assert cfg.bundle_dir == cfg.simulation.bundle_dir


def test_mismatched_bundle_dirs_are_refused(tmp_path):
    est_cfg, _, _ = write_configs(tmp_path)
    raw = yaml.safe_load(est_cfg.read_text())
    raw["paths"]["bundle_dir"] = str(tmp_path / "elsewhere")
    est_cfg.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="same bundle"):
        load_config(est_cfg)


def test_calibrate_writes_params_with_provenance_and_validate_reads_it(tmp_path):
    est_cfg, sim_cfg, out = write_configs(tmp_path, rho_bar_init=-0.3)
    assert _calibrate().main(["--config", str(est_cfg)]) == 0
    params_path = out / "calibrated_params.yaml"
    for name in ("calibrated_params.yaml", "calibration_manifest.yaml",
                 "calibration_fit_2015_2019.csv", "calibration_checkpoint.jsonl"):
        assert (out / name).exists(), name
    data = yaml.safe_load(params_path.read_text())
    bundle = tmp_path / "bundle"
    assert data["provenance"]["bundle_sha256"] == bundle_hashes(bundle)
    assert set(data["fit"]["moments"]) == set(WEIGHTS)
    assert "corr_alpha_log_sales" in data["fit"]["diagnostics"]
    assert data["fit"]["diagnostics"]["mean_firms_per_sector_data"] == 3.0
    manifest = yaml.safe_load((out / "calibration_manifest.yaml").read_text())
    assert manifest["outputs"]["calibrated_params.yaml"]["sha256"]
    assert {"estimation", "simulation"} == set(manifest["configs"])
    params, wf, initial, _ = load_params_file(params_path, bundle_dir=bundle)
    assert params.eta < params.gamma and wf == 0.0 and initial is not None
    assert -0.95 < params.rho_bar < 0.95
    assert _validate().main(["--config", str(est_cfg), "--params", str(params_path)]) == 0


def test_validate_evaluates_initial_values_and_writes_params(tmp_path):
    est_cfg, _, _ = write_configs(tmp_path, rho_bar_init=-0.3)
    target = tmp_path / "warm" / "warm_params.yaml"
    assert _validate().main(["--config", str(est_cfg), "--write-params", str(target)]) == 0
    params, _, initial, data = load_params_file(target, bundle_dir=tmp_path / "bundle")
    assert (params.xi, params.N, params.gamma, params.eta, params.rho_bar) == (8.0, 3.0, 3.8, 2.0, -0.3)
    assert params.v_min > 0.0 and initial is not None
    assert "not a fit" in data["fit"]["note"]
    assert data["fit"]["cost"] >= 0.0


def test_estimation_code_does_not_import_04_code():
    for path in [*EST_DIR.glob("*.py"), *(EST_DIR / "orbis_estimation").glob("*.py")]:
        text = path.read_text()
        assert "import steady_state" not in text and "from steady_state" not in text, path
        for line in text.splitlines():
            if "04_Code" in line:
                assert "import" not in line and "sys.path" not in line, (path, line)
