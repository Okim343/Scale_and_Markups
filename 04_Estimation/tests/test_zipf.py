"""The market-free (Zipf) calibration: tail index, transforms, config, fit plumbing, CLI.

Synthetic bundle only (conftest); the real bundle is never read here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import EST_DIR, write_configs
from orbis_estimation.zipf import (
    ZIPF_TARGET_KEYS, calibrate_zipf, decode, encode, load_zipf_config, load_zipf_targets,
    model_kwargs, moment_elasticities, structural_transform, tail_index, validation_moments,
    zipf_moments_at_params,
)
from orbis_model.model.normalization import PooledParams
from orbis_model.params import load_params_file
from orbis_model.primitives import load_primitives

DATA_SCRIPT = EST_DIR.parent / "03_Empirical" / "02_Orbis" / "04_Moments" / "01_code" / "12_zipf_targets.py"
ZIPF_WEIGHTS = {"mu_cw": 10.0, "zeta_top5": 3.162, "zeta_top2": 3.162, "alpha_rts_gap_top5_bottom50": 3.162}


def _script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_zipf_configs(tmp_path: Path, *, max_nfev: int = 1, M: int = 40) -> tuple[Path, Path]:
    """Synthetic bundle, simulation config, zipf_targets.yaml and a zipf estimation config."""
    _, sim_cfg, out = write_configs(tmp_path)
    bundle = tmp_path / "bundle"
    zt = bundle / "zipf_targets.yaml"
    zt.write_text(yaml.safe_dump({
        "schema": "orbis-zipf-targets-v1",
        "targets": {"zeta_top5": 1.108, "zeta_top2": 1.145},
        "untargeted": {"zeta_top1": 1.168},
    }))
    cfg = tmp_path / "config_estimation_zipf.yaml"
    cfg.write_text(yaml.safe_dump({
        "paths": {"bundle_dir": str(bundle), "zipf_targets": str(zt), "simulation_config": str(sim_cfg),
                  "out_results_dir": str(out / "zipf")},
        "monte_carlo": {"M": M, "seed": 3},
        "fixed": {"eta": 1.2},
        "weights": ZIPF_WEIGHTS,
        "bounds": {"xi": {"lo": 1.05, "hi": 60.0}, "N": {"lo": 2.0, "hi": 4.0},
                   "gamma_gap": {"lo": 0.01, "hi": 30.0}, "rho_bar": {"lo": -0.95, "hi": 0.95}},
        "initial_values": {"xi": 8.0, "N": 3.0, "gamma": 3.8, "rho_bar": -0.2},
        "inner": {"ftol": 1e-4, "xtol": 1e-4, "max_nfev": max_nfev, "fd_step": 1e-2},
        "entry": {"varphi": 0.04},
    }, sort_keys=False))
    return cfg, out / "zipf"


# ---------------------------------------------------------------- tail index

def test_tail_index_recovers_a_pareto_exponent():
    rng = np.random.default_rng(0)
    x = rng.pareto(1.1, size=400_000) + 1.0
    assert tail_index(x, 0.05) == pytest.approx(1.1, abs=0.03)
    assert tail_index(x, 0.01) == pytest.approx(1.1, abs=0.05)


def test_tail_index_is_scale_invariant_and_uses_the_top_k():
    rng = np.random.default_rng(1)
    x = rng.lognormal(0.0, 2.0, size=10_000)
    assert tail_index(1e6 * x, 0.05) == pytest.approx(tail_index(x, 0.05), rel=1e-10)
    top = np.sort(x)[::-1][:500]
    assert tail_index(np.concatenate([top, np.full(9_500, 1e-9)]), 0.05) == pytest.approx(tail_index(x, 0.05))


def test_model_and_data_tail_index_are_the_same_function(monkeypatch):
    monkeypatch.syspath_prepend(str(DATA_SCRIPT.parent))  # moments_common
    data = _script(DATA_SCRIPT, "zipf_targets_step12")
    rng = np.random.default_rng(2)
    x = rng.lognormal(0.0, 1.5, size=5_000)
    for frac in (0.05, 0.02, 0.01, 0.001):  # 0.001 hits the k = 20 floor
        assert tail_index(x, frac) == data.tail_index(x, frac)


# ---------------------------------------------------------------- transforms

def test_encode_decode_round_trip_keeps_eta_fixed():
    bounds = (-0.95, 0.95)
    x = encode((14.0, 450.0, 7.0, -0.285), eta=1.2, rho_bar_bounds=bounds)
    p = decode(x, eta=1.2, a=0.33, H=2500, phi_v=0.5, rho_bar_bounds=bounds)
    assert (p.xi, p.N, p.gamma, p.eta, p.rho_bar) == pytest.approx((14.0, 450.0, 7.0, 1.2, -0.285))
    assert x[2] == pytest.approx(np.log(7.0 - 1.2))
    with pytest.raises(ValueError, match="eta"):
        encode((14.0, 450.0, 1.1, 0.0), eta=1.2)


def test_moment_elasticities_undo_residual_scaling_and_transform():
    bounds = (-0.95, 0.95)
    params = PooledParams(xi=14.0, N=450.0, gamma=7.0, eta=1.2, rho_bar=-0.3)
    keys = ZIPF_TARGET_KEYS
    targets = {"mu_cw": 1.18, "zeta_top5": 1.108, "zeta_top2": 1.145, "alpha_rts_gap_top5_bottom50": 0.046}
    model = {"mu_cw": 1.17, "zeta_top5": 1.12, "zeta_top2": 1.13, "alpha_rts_gap_top5_bottom50": 0.05}
    E = np.arange(16, dtype=float).reshape(4, 4) / 7.0 - 1.0  # planted elasticities
    T = np.diag(structural_transform(params, bounds))
    log_scale = np.array([params.xi, params.N, params.gamma, 1.0])
    dlogm_dx = E / log_scale[None, :] * T[None, :]
    scale = np.array([(1.0 + abs(targets[k])) / (ZIPF_WEIGHTS[k] * model[k]) for k in keys])
    jac = dlogm_dx / scale[:, None]
    got = moment_elasticities(jac, params, model, targets, moment_keys=keys, weights=ZIPF_WEIGHTS,
                              rho_bar_bounds=bounds)
    np.testing.assert_allclose(got, E, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- config and targets

def test_config_and_targets_load(tmp_path):
    cfg_path, out = write_zipf_configs(tmp_path)
    cfg = load_zipf_config(cfg_path)
    assert cfg.eta == 1.2 and cfg.moment_keys == ZIPF_TARGET_KEYS
    assert cfg.bounds["N"] == (2.0, 4.0) and cfg.out_results_dir == out.resolve()
    targets = load_zipf_targets(cfg.bundle_dir, cfg.zipf_targets)
    assert targets["zeta_top5"] == 1.108 and targets["zeta_top2"] == 1.145 and targets["zeta_top1"] == 1.168
    assert targets["mu_cw"] == 1.18 and targets["alpha_rts_gap_top5_bottom50"] == 0.05


def test_config_rejects_N_bound_at_or_above_H(tmp_path):
    cfg_path, _ = write_zipf_configs(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["bounds"]["N"]["hi"] = 5.0  # synthetic H = 5
    cfg_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="below H"):
        load_zipf_config(cfg_path)


def test_zipf_moments_require_every_model_setting(tmp_path):
    cfg_path, _ = write_zipf_configs(tmp_path)
    cfg = load_zipf_config(cfg_path)
    prim = load_primitives(cfg.bundle_dir, H=cfg.simulation.parameters.H)
    params = PooledParams(xi=8.0, N=3.0, gamma=3.8, eta=1.2, a=prim.a, H=prim.H, phi_v=prim.phi_v)
    with pytest.raises(TypeError):
        zipf_moments_at_params(params, prim, M=4, rng=3)
    kw = model_kwargs(cfg.simulation)
    assert set(kw) == {"beta", "delta_K", "K_switch", "love_of_variety", "market_solver_kwargs", "alpha_cap_factor"}
    assert kw["market_solver_kwargs"]["max_iter"] == 100  # the synthetic simulation config's solver block


# ---------------------------------------------------------------- fit and validation

def test_calibrate_zipf_checkpoints_every_evaluation_and_keeps_eta(tmp_path):
    cfg_path, out = write_zipf_configs(tmp_path, max_nfev=2)
    cfg = load_zipf_config(cfg_path)
    prim = load_primitives(cfg.bundle_dir, H=cfg.simulation.parameters.H)
    targets = load_zipf_targets(cfg.bundle_dir, cfg.zipf_targets)
    ckpt = out / "ckpt.jsonl"
    res = calibrate_zipf(
        prim, targets, M=cfg.M, rng=cfg.seed, eta=cfg.eta,
        initial=[cfg.initial_values[k] for k in ("xi", "N", "gamma", "rho_bar")],
        bounds=cfg.bounds, weights=cfg.weights, max_nfev=2, checkpoint_path=ckpt,
        **model_kwargs(cfg.simulation),
    )
    lines = [json.loads(line) for line in ckpt.read_text().splitlines()]
    assert len(lines) == res.n_evals >= 5  # f(x0) plus a 4-column forward-difference Jacobian
    assert [r["eval"] for r in lines] == list(range(1, len(lines) + 1))
    assert all("error" not in r and "zeta_top5" in r["moments"] and r["eta"] == 1.2 for r in lines)
    assert res.params.eta == 1.2 and res.params.gamma > 1.2
    assert res.jacobian_x.shape == (4, 4) and res.elasticities.shape == (4, 4)
    assert res.moment_keys == ZIPF_TARGET_KEYS and np.isfinite(res.cost)
    v = validation_moments(res.equilibrium, res.params, alpha_support=prim.alpha_support,
                           alpha_cap_factor=cfg.simulation.parameters.alpha_cap_factor)
    assert v["mu_floor"] == pytest.approx(res.params.gamma / (res.params.gamma - 1.0))
    assert v["alpha_top_node"] < v["alpha_cap"]
    assert 0.0 <= v["share_top1pct_firms_top_node"] <= 1.0
    assert v["mu_cw_true"] == pytest.approx(res.model_moments["mu_cw"])


def test_failed_evaluation_is_recorded_and_penalized(tmp_path, monkeypatch):
    import orbis_estimation.zipf as zipf

    cfg_path, out = write_zipf_configs(tmp_path)
    cfg = load_zipf_config(cfg_path)
    prim = load_primitives(cfg.bundle_dir, H=cfg.simulation.parameters.H)
    targets = load_zipf_targets(cfg.bundle_dir, cfg.zipf_targets)
    real = zipf.zipf_moments_at_params
    calls = [0]

    def flaky(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:  # the first Jacobian column
            raise RuntimeError("GE did not converge")
        return real(*args, **kwargs)

    monkeypatch.setattr(zipf, "zipf_moments_at_params", flaky)
    ckpt = out / "ckpt.jsonl"
    calibrate_zipf(
        prim, targets, M=cfg.M, rng=cfg.seed, eta=cfg.eta, initial=[8.0, 3.0, 3.8, -0.2],
        bounds=cfg.bounds, weights=cfg.weights, max_nfev=1, checkpoint_path=ckpt,
        **model_kwargs(cfg.simulation),
    )
    lines = [json.loads(line) for line in ckpt.read_text().splitlines()]
    assert "GE did not converge" in lines[1]["error"] and "error" not in lines[0]


def test_cli_writes_a_params_file_that_simulation_can_load(tmp_path):
    cfg_path, out = write_zipf_configs(tmp_path, max_nfev=1)
    module = _script(EST_DIR / "01_calibrate_zipf.py", "script_01_calibrate_zipf")
    assert module.main(["--config", str(cfg_path)]) == 0
    params_path = out / "calibrated_params_zipf.yaml"
    cfg = load_zipf_config(cfg_path)
    params, _, _, data = load_params_file(params_path, bundle_dir=cfg.bundle_dir)
    assert params.eta == 1.2 and data["calibration"]["variant"] == "zipf"
    assert set(data["fit"]["moments"]) == set(ZIPF_TARGET_KEYS)
    assert data["model"]["alpha_cap_factor"] == cfg.simulation.parameters.alpha_cap_factor
    for name in ("calibration_manifest_zipf.yaml", "calibration_identification_zipf.yaml",
                 "calibration_checkpoint_zipf.jsonl"):
        assert (out / name).exists()
    assert list(out.glob("calibration_fit_zipf_*.csv"))

    validate = _script(EST_DIR / "02_validate_zipf.py", "script_02_validate_zipf")
    zt = yaml.safe_load(cfg.zipf_targets.read_text())
    zt["detail"] = {"normalized_ind2d": {t: {"hill": 1.0} for t in ("top5", "top2", "top1")}}
    conc = {"cr4": 0.3, "cr20": 0.5, "top1pct": 0.4, "top5pct": 0.6, "median_firms_per_market": 3.0}
    pct = {lab: {"p50": 1.1, "p90": 1.4, "p99": 2.0} for lab in ("cost_weighted", "sales_weighted", "unweighted")}
    zt["validation"] = {"concentration": {"naics4": conc, "ind2d": conc},
                        "markups": {"percentiles": pct, "mu_sw": 1.2, "gap_sw_cw": 0.08,
                                    "var_sw_log": 0.07, "var_cw_log": 0.05}}
    cfg.zipf_targets.write_text(yaml.safe_dump(zt))
    prefix = out / "val"
    assert validate.main(["--config", str(cfg_path), "--params", f"zipf={params_path}",
                          "--seed", "5", "--out", str(prefix)]) == 0
    table = prefix.with_suffix(".csv").read_text()
    assert "share_top1pct_firms_top_node" in table and "zeta_top1" in table
