"""Shared helpers: a small synthetic bundle (never real data) and the real-bundle path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SIM_DIR = Path(__file__).resolve().parents[1]
REAL_BUNDLE = SIM_DIR.parent / "03_Empirical" / "02_Orbis" / "04_Moments" / "03_outdata"

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


def write_bundle(bundle_dir: Path, *, moments: dict | None = None, f_alpha: pd.DataFrame | None = None) -> Path:
    """Write a synthetic three-file bundle; ``moments`` replaces aggregate_moments.yaml."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if f_alpha is None:
        nodes = np.linspace(0.75, 1.05, 10)
        f_alpha = pd.DataFrame({
            "node_id": np.arange(1, 11), "u": (np.arange(10) + 0.5) / 10,
            "weight": 0.1, "alpha": nodes,
        })
    f_alpha.to_parquet(bundle_dir / "F_alpha.parquet", index=False)
    with open(bundle_dir / "aggregate_moments.yaml", "w") as stream:
        yaml.safe_dump(SYNTHETIC_MOMENTS if moments is None else moments, stream)
    with open(bundle_dir / "manifest.yaml", "w") as stream:
        yaml.safe_dump({"schema": "pooled-bundle-v1", "synthetic": True}, stream)
    return bundle_dir


def write_sim_config(path: Path, *, bundle_dir: Path, params_file: Path, out_dir: Path, H: int = 5) -> Path:
    with open(path, "w") as stream:
        yaml.safe_dump({
            "paths": {"bundle_dir": str(bundle_dir), "params_file": str(params_file),
                      "out_results_dir": str(out_dir), "out_figs_dir": str(out_dir / "figs")},
            "monte_carlo": {"M": 3, "master_seed": 3},
            "welfare": {"M": 3},
            "parameters": {"beta": 0.96, "delta_K": 0.06, "phi": 1.0, "H": H},
            "participation": {"K_switch": 3, "love_of_variety": "off"},
            "solver": {"market": {"tol": 1e-7, "max_iter": 100, "damping": 0.5,
                                  "newton_fallback_after": 50, "backend": "vectorized"}},
        }, stream)
    return path


@pytest.fixture
def synthetic_bundle(tmp_path) -> Path:
    return write_bundle(tmp_path / "bundle")


@pytest.fixture
def real_bundle() -> Path:
    if not (REAL_BUNDLE / "aggregate_moments.yaml").exists():
        pytest.skip("Orbis bundle not present (gitignored, WRDS-licensed)")
    return REAL_BUNDLE
