"""Zipf-calibration plumbing: the dedicated config and 05_planner_expansion --sorting-dir."""

from __future__ import annotations

import importlib.util
import json

import pytest
import yaml

from conftest import SIM_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SIM_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zipf_config_differs_from_default_only_in_paths():
    from orbis_model.config import load_config

    default = yaml.safe_load((SIM_DIR / "config_simulation.yaml").read_text())
    zipf = yaml.safe_load((SIM_DIR / "config_simulation_zipf.yaml").read_text())
    assert {k: v for k, v in zipf.items() if k != "paths"} == \
        {k: v for k, v in default.items() if k != "paths"}
    assert zipf["paths"]["bundle_dir"] == default["paths"]["bundle_dir"]
    cfg = load_config(SIM_DIR / "config_simulation_zipf.yaml")
    assert cfg.params_file.name == "calibrated_params_zipf.yaml"
    assert cfg.out_results_dir == SIM_DIR / "out_results" / "zipf"
    assert cfg.out_figs_dir == SIM_DIR / "out_figs" / "zipf"
    # The default config keeps the current calibration and its output folders.
    base = load_config()
    assert base.params_file.name == "calibrated_params.yaml"
    assert base.out_results_dir == SIM_DIR / "out_results"


def _write_refs(tmp_path, *, M=5, seed=3, sha="abc", corr=-0.1234):
    welfare = tmp_path / "welfare"
    sorting = tmp_path / "sorting"
    welfare.mkdir()
    sorting.mkdir()
    (welfare / "welfare_manifest_2015_2019.yaml").write_text(yaml.safe_dump(
        {"M": M, "seed": seed, "params_file": {"path": "p.yaml", "sha256": sha}}))
    (sorting / "run_manifest.json").write_text(json.dumps(
        {"fingerprint": "f", "economics": {"M": 5, "seed": 3, "params_sha256": "abc"}}))
    (sorting / "checkpoint_baseline.json").write_text(json.dumps({"corr_av": corr}))
    return welfare, sorting


def test_planner_expansion_reads_corr_target_from_matching_sorting_run(tmp_path):
    module = _script("05_planner_expansion.py")
    welfare, sorting = _write_refs(tmp_path)
    assert module.sorting_corr_target(sorting, welfare, "2015_2019") == -0.1234


@pytest.mark.parametrize("change", [{"M": 6}, {"seed": 4}, {"sha": "other"}])
def test_planner_expansion_refuses_a_sorting_run_on_other_draws(tmp_path, change):
    module = _script("05_planner_expansion.py")
    welfare, sorting = _write_refs(tmp_path, **change)
    with pytest.raises(SystemExit, match="sorting reference"):
        module.sorting_corr_target(sorting, welfare, "2015_2019")


def test_planner_expansion_rejects_both_corr_sources(tmp_path):
    module = _script("05_planner_expansion.py")
    welfare, sorting = _write_refs(tmp_path)
    with pytest.raises(SystemExit, match="not both"):
        module.main(["--welfare-dir", str(welfare), "--sorting-dir", str(sorting),
                     "--corr-target", "0.1", "--out-dir", str(tmp_path / "out")])


def test_figures_default_folders_follow_the_config():
    from orbis_model.figures.simulation_figures import _default_results_dir, default_output_dir

    zipf = SIM_DIR / "config_simulation_zipf.yaml"
    assert _default_results_dir(zipf) == SIM_DIR / "out_results" / "zipf"
    assert default_output_dir(SIM_DIR / "out_results" / "zipf_M1000", zipf) == \
        SIM_DIR / "out_figs" / "zipf" / "welfare" / "zipf_M1000"
    assert _default_results_dir() == SIM_DIR / "out_results"
