from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from steady_state.__main__ import _load_calibration_yaml, main
from steady_state.model.normalization import PooledParams
from steady_state.pooled_inputs import PooledInputs
from steady_state.simulation.cross_section import SIM_PANEL_COLUMNS, run_simulation
from steady_state.simulation.moments import economy_moments
from steady_state.welfare import solve_market_and_planner, two_channel_decomposition


def _inputs(H=5):
    return PooledInputs(
        targets={"mu_cw": 1.18, "mu_cw_alpha": 1.4, "cr4": 0.45, "cr20": 0.75,
                 "top1pct": 0.30, "top5pct": 0.55, "emx_slope": -0.25},
        rho_bar=0.3, a=0.35, eta_init=2.0, H=H,
        alpha_support=np.linspace(0.75, 1.0, 12), n_firms_cs=100,
    )


def _params(H=5):
    return PooledParams(xi=8.0, N=3.0, gamma=3.8, eta=2.0, a=0.35, H=H)


def test_run_simulation_panel_schema_and_moments(tmp_path):
    sim = run_simulation(
        _inputs(), _params(), M=3, master_seed=2,
        market_solver_kwargs={"backend": "vectorized"}, out_dir=tmp_path,
    )
    assert list(sim.panel.columns) == list(SIM_PANEL_COLUMNS)
    assert len(sim.panel) == 15
    assert {"active", "n_active"}.issubset(sim.panel.columns)
    moments = economy_moments(sim, _inputs())
    assert "emx_slope" in moments and "mean_active_count" in moments
    assert (tmp_path / "sim_panel.parquet").exists()


def test_welfare_masks_reference_and_multiplicative_identity():
    sim = run_simulation(
        _inputs(), _params(), M=3, master_seed=4,
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
    import steady_state.model.normalization as normalization

    def fail_anchored(*args, **kwargs):
        raise AssertionError("anchored solver must not run outside calibration")

    monkeypatch.setattr(normalization, "solve_pooled_ge_anchored", fail_anchored)
    sim = run_simulation(
        _inputs(), _params(), M=3, master_seed=12,
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


def _write_bundle_and_config(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"; bundle.mkdir()
    # Single pooled F(alpha): no sector_id, equal-mass nodes.
    nodes = np.linspace(0.75, 1.05, 10)
    pd.DataFrame({
        "node_id": np.arange(1, 11),
        "u": (np.arange(10) + 0.5) / 10, "weight": 0.1,
        "alpha_raw": nodes, "alpha": nodes,
    }).to_parquet(bundle / "F_alpha.parquet", index=False)
    with open(bundle / "aggregate_moments.yaml", "w") as f:
        yaml.safe_dump({"active_window": {"start": 2010, "end": 2019},
                        "alpha_z_copula_rho_bar": 0.3,
                        "alpha_sales_corr_empirical": 0.596,
                        "alpha_sales_corr_partial_empirical": 0.570,
                        "a": 0.35, "n_firms_cs": 100.0,
                        "mean_firms_per_sector": 3.0,
                        "phi_v": 0.45,
                        "gnr_elasticities": {"a_gnr": 0.30},
                        "reference_scale": {
                            "type": "pooled_naics2_unweighted_median",
                            "x_ref": 100.0,
                            "x_ref_role": "model_reference_input_scale",
                            "y_ref": 1000.0,
                            "y_ref_role": "diagnostic_output_scale_only",
                            "y_ref_over_x_ref": 10.0,
                            "x_ref_variable": "test_input_composite",
                            "y_ref_variable": "sale_D",
                        },
                        "targets": {"mu_cw": 1.4, "mu_cw_sga": 1.18,
                                    "cr4": 0.45, "cr20": 0.75,
                                    "top1pct": 0.30, "top5pct": 0.55,
                                    "emx_slope": -0.25},
                        "validation": {}}, f)
    with open(bundle / "manifest.yaml", "w") as f: yaml.safe_dump({}, f)
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    with open(config, "w") as f:
        yaml.safe_dump({
            "paths": {"bundle_dir": str(bundle), "out_results_dir": str(out), "out_figs_dir": str(out)},
            "window": {"primary": [2010, 2019], "active": "primary"},
            "monte_carlo": {"M": 2, "M_final": 2, "master_seed": 3},
            "parameters": {"beta": 0.96, "delta_K": 0.06, "phi": 1.0, "H": 5},
            "participation": {"K_switch": 3, "love_of_variety": "off"},
            "exposure": {"uniform": True},
            "solver": {"market": {"tol": 1e-7, "max_iter": 100, "damping": 0.5,
                                      "newton_fallback_after": 50, "backend": "vectorized"}},
            "calibration": {"eta": {"init": 2.0, "lo": 1.01},
                            "xi": {"lo": 1.05, "hi": 30.0},
                            "N": {"lo": 2.0, "hi": 4.0},
                            "inner": {"max_nfev": 1}},
            "initial_values": {"gamma": 3.8, "xi_default": 8.0, "N": 3.0},
        }, f)
    return config


def test_load_bundle_requires_top_level_a(tmp_path):
    """Model capital share is the top-level KLEMS a_bar; a bundle carrying only
    the diagnostic gnr_elasticities.a_gnr (no top-level `a`) must fail loudly."""
    from steady_state.io_bundle import load_bundle

    _write_bundle_and_config(tmp_path)
    bundle_dir = tmp_path / "bundle"
    # Well-formed bundle: Bundle.a is the top-level a_bar, not a_gnr (0.30).
    assert load_bundle(bundle_dir).a == pytest.approx(0.35)

    agg_path = bundle_dir / "aggregate_moments.yaml"
    agg = yaml.safe_load(agg_path.read_text())
    del agg["a"]  # leaves gnr_elasticities.a_gnr behind -> an a_gnr-only bundle
    with open(agg_path, "w") as f:
        yaml.safe_dump(agg, f)
    with pytest.raises(ValueError, match="top-level scalars missing"):
        load_bundle(bundle_dir)


def test_load_bundle_requires_raw_alpha_sales_corr(tmp_path):
    from steady_state.io_bundle import load_bundle

    _write_bundle_and_config(tmp_path)
    bundle_dir = tmp_path / "bundle"
    agg_path = bundle_dir / "aggregate_moments.yaml"
    agg = yaml.safe_load(agg_path.read_text())
    del agg["alpha_sales_corr_empirical"]
    with open(agg_path, "w") as f:
        yaml.safe_dump(agg, f)
    with pytest.raises(ValueError, match="alpha_sales_corr_empirical"):
        load_bundle(bundle_dir)


def test_pooled_cli_commands_run_end_to_end(tmp_path):
    cfg = _write_bundle_and_config(tmp_path)
    assert main(["--config", str(cfg), "calibrate"]) == 0
    assert main(["--config", str(cfg), "simulate"]) == 0
    assert main(["--config", str(cfg), "validate"]) == 0
    assert main(["--config", str(cfg), "welfare"]) == 0
    out = tmp_path / "out"
    assert (out / "calibration_pooled.yaml").exists()
    assert (out / "welfare.yaml").exists()
    params, wf, initial = _load_calibration_yaml(out / "calibration_pooled.yaml", a=0.35, H=5)
    assert params.eta < params.gamma and wf is not None
    assert initial is not None
