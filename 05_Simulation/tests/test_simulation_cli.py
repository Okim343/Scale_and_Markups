"""Simulation and welfare plumbing, and the 01_simulate / 02_welfare scripts end to end.

Ported from ``04_Code/tests/test_simulation_cli.py`` (simulation and welfare
parts; the calibrate/validate parts live in ``04_Estimation/tests``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import SIM_DIR, write_bundle, write_sim_config
from orbis_model.model.normalization import PooledParams
from orbis_model.params import build_provenance, write_params_file
from orbis_model.primitives import ModelPrimitives
from orbis_model.simulation.cross_section import SIM_PANEL_COLUMNS, run_simulation
from orbis_model.simulation.moments import economy_moments
from orbis_model.welfare import solve_market_and_planner, two_channel_decomposition


def _primitives(H=5):
    return ModelPrimitives(
        alpha_support=np.linspace(0.75, 1.0, 12), a=0.35, phi_v=0.45,
        n_firms_cs=100.0, H=H, active_window=(2015, 2019),
    )


def _params(H=5):
    return PooledParams(xi=8.0, N=3.0, gamma=3.8, eta=2.0, a=0.35, H=H, phi_v=0.45)


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SIM_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_simulation_panel_schema_and_moments(tmp_path):
    sim = run_simulation(
        _primitives(), _params(), M=3, master_seed=2,
        market_solver_kwargs={"backend": "vectorized"}, out_dir=tmp_path,
    )
    assert list(sim.panel.columns) == list(SIM_PANEL_COLUMNS)
    assert len(sim.panel) == 15
    assert {"active", "n_active"}.issubset(sim.panel.columns)
    moments = economy_moments(sim, _primitives())
    assert "emx_slope" in moments and "mean_active_count" in moments
    assert "alpha_rts_gap_top5_bottom50" in moments
    assert (tmp_path / "sim_panel.parquet").exists()


def test_welfare_masks_reference_and_multiplicative_identity():
    sim = run_simulation(
        _primitives(), _params(), M=3, master_seed=4,
        market_solver_kwargs={"backend": "vectorized"},
    )
    pair = solve_market_and_planner(
        sim.equilibrium.alpha, sim.equilibrium.v, _params(), n_firms_cs=100,
        wf_reference=sim.equilibrium.participation.wf_reference,
        active_mask=sim.equilibrium.participation.active_mask,
        market_solver_kwargs={"backend": "vectorized"},
    )
    result = two_channel_decomposition(
        pair.me, pair.pe, params=_params(), chi_me=pair.me.chi,
        beta=0.96, phi=1.0, n_firms_cs=100,
        market_solver_kwargs={"backend": "vectorized"},
    )
    np.testing.assert_array_equal(pair.me.participation.active_mask, pair.pe.participation.active_mask)
    np.testing.assert_array_equal(pair.me.participation.active_mask, result["uniform"].participation.active_mask)
    refs = {pair.me.participation.wf_reference, pair.pe.participation.wf_reference,
            result["uniform"].participation.wf_reference}
    assert len(refs) == 1
    assert result["identity_residual"] == pytest.approx(0.0, abs=1e-12)


def test_welfare_paths_use_frozen_v_without_anchored_solver(monkeypatch):
    import orbis_model.model.normalization as normalization

    def fail_anchored(*args, **kwargs):
        raise AssertionError("anchored solver must not run outside calibration")

    monkeypatch.setattr(normalization, "solve_pooled_ge_anchored", fail_anchored)
    sim = run_simulation(
        _primitives(), _params(), M=3, master_seed=12,
        market_solver_kwargs={"backend": "vectorized"},
    )
    pair = solve_market_and_planner(
        sim.equilibrium.alpha, sim.equilibrium.v, _params(), n_firms_cs=100,
        wf_reference=sim.equilibrium.participation.wf_reference,
        active_mask=sim.equilibrium.participation.active_mask,
        market_solver_kwargs={"backend": "vectorized"},
    )
    result = two_channel_decomposition(
        pair.me, pair.pe, params=_params(), chi_me=pair.me.chi,
        beta=0.96, phi=1.0, n_firms_cs=100,
        market_solver_kwargs={"backend": "vectorized"},
    )
    np.testing.assert_array_equal(pair.me.v, sim.equilibrium.v)
    np.testing.assert_array_equal(pair.pe.v, sim.equilibrium.v)
    np.testing.assert_array_equal(result["uniform"].v, sim.equilibrium.v)


def _setup_run(tmp_path, *, rho_bar=0.3):
    bundle = write_bundle(tmp_path / "bundle")
    params_file = tmp_path / "params.yaml"
    out = tmp_path / "out"
    cfg = write_sim_config(tmp_path / "config_simulation.yaml", bundle_dir=bundle,
                           params_file=params_file, out_dir=out)
    prov = build_provenance(bundle_dir=bundle, configs={"simulation": cfg},
                            code_dirs=(), script="test")
    params = PooledParams(xi=8.0, N=3.0, gamma=3.8, eta=2.0, rho_bar=rho_bar,
                          a=0.35, H=5, phi_v=0.45)
    write_params_file(params_file, params, reference={"w": 1.0, "X_market": 1.0, "wf_reference": 0.0},
                      provenance=prov)
    return bundle, cfg, out


@pytest.mark.parametrize("rho_bar", [0.3, -0.4])
def test_simulate_and_welfare_scripts_end_to_end(tmp_path, rho_bar):
    _, cfg, out = _setup_run(tmp_path, rho_bar=rho_bar)
    assert _script("01_simulate.py").main(["--config", str(cfg)]) == 0
    assert _script("02_welfare.py").main(["--config", str(cfg)]) == 0
    for name in ("sim_panel_2015_2019.parquet", "sim_moments_2015_2019.parquet",
                 "sim_economy_moments_2015_2019.yaml", "simulate_manifest_2015_2019.yaml",
                 "sim_panel_market_2015_2019.parquet", "sim_panel_planner_2015_2019.parquet",
                 "welfare_decomposition_2015_2019.parquet", "welfare_2015_2019.yaml",
                 "welfare_manifest_2015_2019.yaml"):
        assert (out / name).exists(), name
    moments = pd.read_parquet(out / "sim_moments_2015_2019.parquet")
    assert "alpha_rts_gap_top5_bottom50" in set(moments["moment"])
    assert "target" not in moments.columns
    welfare = yaml.safe_load((out / "welfare_2015_2019.yaml").read_text())
    assert welfare["identity_residual"] == pytest.approx(0.0, abs=1e-12)
    manifest = yaml.safe_load((out / "simulate_manifest_2015_2019.yaml").read_text())
    assert manifest["M"] == 3 and "bundle_sha256" in manifest


def test_scripts_refuse_params_from_another_bundle(tmp_path):
    bundle, cfg, out = _setup_run(tmp_path)
    agg = yaml.safe_load((bundle / "aggregate_moments.yaml").read_text())
    agg["n_firms_cs"] = 250.0
    (bundle / "aggregate_moments.yaml").write_text(yaml.safe_dump(agg))
    for script in ("01_simulate.py", "02_welfare.py"):
        with pytest.raises(ValueError, match="different bundle"):
            _script(script).main(["--config", str(cfg)])
    assert not Path(out).exists()


def test_default_config_points_at_orbis_bundle_and_estimation_output():
    from orbis_model.config import load_config

    cfg = load_config()
    assert cfg.bundle_dir.as_posix().endswith("03_Empirical/02_Orbis/04_Moments/03_outdata")
    assert cfg.params_file.as_posix().endswith("04_Estimation/out_results/calibrated_params.yaml")
    assert cfg.parameters.H == 2500


def test_figures_find_window_tagged_welfare_outputs(tmp_path):
    from orbis_model.figures.simulation_figures import generate_simulation_figures

    _, cfg, out = _setup_run(tmp_path)
    assert _script("02_welfare.py").main(["--config", str(cfg)]) == 0
    # welfare_manifest_<window>.yaml sits next to welfare_<window>.yaml and must not match.
    figs = generate_simulation_figures(out, tmp_path / "figs")
    assert (figs / "markup_distribution.png").exists()
    assert (figs / "welfare_lambda_contribution_grid.png").exists()
