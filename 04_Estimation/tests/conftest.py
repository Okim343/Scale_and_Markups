"""Shared helpers: synthetic bundle and configs (never real data), real-bundle path.

Importing ``orbis_estimation`` puts ``05_Simulation`` on ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import orbis_estimation

# Importing orbis_estimation puts 05_Simulation on sys.path (orbis_model).
assert orbis_estimation.SIMULATION_DIR.is_dir()

EST_DIR = Path(__file__).resolve().parents[1]
REAL_BUNDLE = EST_DIR.parent / "03_Empirical" / "02_Orbis" / "04_Moments" / "03_outdata"

SYNTHETIC_MOMENTS = {
    "active_window": {"key": "primary", "start": 2015, "end": 2019},
    "targets": {"mu_cw": 1.18, "cr4": 0.45, "cr20": 0.75, "top1pct": 0.30,
                "top5pct": 0.55, "emx_slope": -0.25},
    "validation": {},
    "a": 0.35,
    "phi_v": 0.45,
    "n_firms_cs": 100.0,
    "mean_firms_per_sector": 3.0,
    "alpha_rts_gap_top5_bottom50": 0.05,
}

WEIGHTS = {"mu_cw": 10.0, "cr4": 1.0, "cr20": 1.0, "top1pct": 0.0, "top5pct": 0.0,
           "emx_slope": 3.162, "alpha_rts_gap_top5_bottom50": 3.162}


def write_bundle(bundle_dir: Path, *, moments: dict | None = None) -> Path:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    nodes = np.linspace(0.75, 1.05, 10)
    pd.DataFrame({
        "node_id": np.arange(1, 11), "u": (np.arange(10) + 0.5) / 10,
        "weight": 0.1, "alpha": nodes,
    }).to_parquet(bundle_dir / "F_alpha.parquet", index=False)
    with open(bundle_dir / "aggregate_moments.yaml", "w") as stream:
        yaml.safe_dump(SYNTHETIC_MOMENTS if moments is None else moments, stream)
    with open(bundle_dir / "manifest.yaml", "w") as stream:
        yaml.safe_dump({"schema": "pooled-bundle-v1", "synthetic": True}, stream)
    return bundle_dir


def write_configs(tmp_path: Path, *, rho_bar_bounds=(-0.95, 0.95), rho_bar_init=0.0,
                  max_nfev: int = 1) -> tuple[Path, Path, Path]:
    """Synthetic bundle plus matching estimation and simulation configs.

    Returns ``(estimation_config, simulation_config, out_dir)``.
    """
    bundle = write_bundle(tmp_path / "bundle")
    out = tmp_path / "out"
    sim_cfg = tmp_path / "config_simulation.yaml"
    with open(sim_cfg, "w") as stream:
        yaml.safe_dump({
            "paths": {"bundle_dir": str(bundle), "params_file": str(out / "calibrated_params.yaml"),
                      "out_results_dir": str(tmp_path / "sim_out"), "out_figs_dir": str(tmp_path / "figs")},
            "monte_carlo": {"M": 2, "master_seed": 3},
            "welfare": {"M": 2},
            "parameters": {"beta": 0.96, "delta_K": 0.06, "phi": 1.0, "H": 5},
            "participation": {"K_switch": 3, "love_of_variety": "off"},
            "solver": {"market": {"tol": 1e-7, "max_iter": 100, "damping": 0.5,
                                  "newton_fallback_after": 50, "backend": "vectorized"}},
        }, stream, sort_keys=False)
    est_cfg = tmp_path / "config_estimation.yaml"
    with open(est_cfg, "w") as stream:
        yaml.safe_dump({
            "paths": {"bundle_dir": str(bundle), "simulation_config": str(sim_cfg),
                      "out_results_dir": str(out)},
            "monte_carlo": {"M": 2, "seed": 3},
            "weights": WEIGHTS,
            "bounds": {"xi": {"lo": 1.05, "hi": 30.0}, "N": {"lo": 2.0, "hi": 4.0},
                       "eta": {"lo": 1.01, "hi": 10.0}, "gamma_gap": {"lo": 0.01, "hi": 30.0},
                       "rho_bar": {"lo": rho_bar_bounds[0], "hi": rho_bar_bounds[1]}},
            "initial_values": {"xi": 8.0, "N": 3.0, "gamma": 3.8, "eta": 2.0, "rho_bar": rho_bar_init},
            "inner": {"ftol": 1e-4, "xtol": 1e-4, "max_nfev": max_nfev},
            "entry": {"varphi": 0.04},
        }, stream, sort_keys=False)
    return est_cfg, sim_cfg, out


@pytest.fixture
def real_bundle() -> Path:
    if not (REAL_BUNDLE / "aggregate_moments.yaml").exists():
        pytest.skip("Orbis bundle not present (gitignored, WRDS-licensed)")
    return REAL_BUNDLE
