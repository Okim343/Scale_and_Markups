"""Typed configuration for the Orbis calibration (``config_estimation.yaml``).

Holds the calibration-only keys of ``04_Code/steady_state/config.py`` (bounds,
initial values, weights, inner solver settings, Monte Carlo size and seed).
The model settings (assigned parameters, participation, market solver) are read
from the simulation config named in ``paths.simulation_config``, so the
calibration and the simulation solve the same model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from orbis_model.config import SimulationConfig
from orbis_model.config import load_config as load_simulation_config
from orbis_model.model.normalization import PooledParams
from orbis_model.primitives import ModelPrimitives


EST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = EST_DIR / "config_estimation.yaml"
PARAMETER_NAMES = ("xi", "N", "gamma", "eta", "rho_bar")
BOUND_NAMES = ("xi", "N", "eta", "gamma_gap", "rho_bar")


@dataclass(frozen=True)
class EstimationConfig:
    path: Path
    bundle_dir: Path
    out_results_dir: Path
    simulation: SimulationConfig
    M: int
    seed: int
    weights: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    initial_values: dict[str, float]
    ftol: float
    xtol: float
    max_nfev: int
    varphi: float
    log_level: str
    raw: Mapping[str, Any]

    @property
    def moment_keys(self) -> tuple[str, ...]:
        """Targeted moments: the keys of ``weights`` (a zero weight keeps a moment reported)."""
        return tuple(self.weights)


def _resolve(path_str: str, anchor: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (anchor / path).resolve()


def _bounds(raw: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    block = raw.get("bounds") or {}
    missing = [k for k in BOUND_NAMES if k not in block]
    if missing:
        raise ValueError(f"config_estimation.yaml: bounds missing {missing}")
    bounds = {k: (float(block[k]["lo"]), float(block[k]["hi"])) for k in BOUND_NAMES}
    for name, (lo, hi) in bounds.items():
        if not lo < hi:
            raise ValueError(f"bounds.{name} must satisfy lo < hi")
    if not 1.0 < bounds["xi"][0]:
        raise ValueError("bounds.xi.lo must exceed one")
    if not 1.0 <= bounds["N"][0]:
        raise ValueError("bounds.N.lo must be at least one")
    if not 1.0 < bounds["eta"][0]:
        raise ValueError("bounds.eta.lo must be strictly above one")
    if not 0.0 < bounds["gamma_gap"][0]:
        raise ValueError("bounds.gamma_gap.lo must be positive")
    if not -1.0 < bounds["rho_bar"][0] < bounds["rho_bar"][1] < 1.0:
        raise ValueError("bounds.rho_bar must lie inside (-1, 1)")
    return bounds


def load_config(path: Path | str | None = None) -> EstimationConfig:
    cfg_path = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as stream:
        raw = yaml.safe_load(stream)
    paths = raw["paths"]
    bundle_dir = _resolve(paths["bundle_dir"], cfg_path.parent)
    simulation = load_simulation_config(_resolve(paths["simulation_config"], cfg_path.parent))
    if simulation.bundle_dir != bundle_dir:
        raise ValueError(
            f"estimation bundle_dir {bundle_dir} differs from the simulation config's "
            f"{simulation.bundle_dir}; both must point at the same bundle"
        )
    mc = raw["monte_carlo"]
    weights = {str(k): float(v) for k, v in (raw.get("weights") or {}).items()}
    if not weights:
        raise ValueError("config_estimation.yaml: weights must list the targeted moments")
    initial = raw.get("initial_values") or {}
    missing = [k for k in PARAMETER_NAMES if k not in initial]
    if missing:
        raise ValueError(f"config_estimation.yaml: initial_values missing {missing}")
    inner = raw.get("inner") or {}
    return EstimationConfig(
        path=cfg_path,
        bundle_dir=bundle_dir,
        out_results_dir=_resolve(paths["out_results_dir"], cfg_path.parent),
        simulation=simulation,
        M=int(mc["M"]),
        seed=int(mc["seed"]),
        weights=weights,
        bounds=_bounds(raw),
        initial_values={k: float(initial[k]) for k in PARAMETER_NAMES},
        ftol=float(inner.get("ftol", 1e-4)),
        xtol=float(inner.get("xtol", 1e-4)),
        max_nfev=int(inner.get("max_nfev", 40)),
        varphi=float((raw.get("entry") or {}).get("varphi", 0.04)),
        log_level=str(raw.get("log_level", "INFO")),
        raw=raw,
    )


def initial_params(cfg: EstimationConfig, primitives: ModelPrimitives) -> PooledParams:
    """Warm start from ``initial_values`` (ported from 04_Code ``_initial_params``).

    No fallback to bundle values: ``mean_firms_per_sector`` is never an initial
    N, and there is no bundle rho_bar warm start.
    """
    iv = cfg.initial_values
    eta = iv["eta"]
    gamma = max(iv["gamma"], eta + 0.1)
    return PooledParams(
        xi=iv["xi"], N=iv["N"], gamma=gamma, eta=eta, rho_bar=iv["rho_bar"],
        a=primitives.a, H=primitives.H, phi_v=primitives.phi_v,
    )
