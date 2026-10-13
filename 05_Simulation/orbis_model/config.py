"""Typed configuration for the Orbis simulation pipeline (``config_simulation.yaml``).

Ported from ``04_Code/steady_state/config.py``, keeping only the simulation keys:
paths (bundle, params file, outputs), Monte Carlo size and seed, assigned
parameters, participation, solver settings and the welfare run size. The
calibration block lives in ``04_Estimation/config_estimation.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


SIM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SIM_DIR / "config_simulation.yaml"


@dataclass(frozen=True)
class ParallelSolverConfig:
    n_jobs: int
    min_markets: int
    min_markets_per_job: int

    def to_kwargs(self) -> dict:
        return {
            "parallel_n_jobs": self.n_jobs,
            "parallel_min_markets": self.min_markets,
            "parallel_min_markets_per_job": self.min_markets_per_job,
        }


@dataclass(frozen=True)
class MarketSolverConfig:
    tol: float
    max_iter: int
    damping: float
    newton_fallback_after: int
    backend: str
    parallel: ParallelSolverConfig

    def to_kwargs(self) -> dict:
        return {
            "tol": self.tol,
            "max_iter": self.max_iter,
            "damping": self.damping,
            "newton_fallback_after": self.newton_fallback_after,
            "backend": self.backend,
            **self.parallel.to_kwargs(),
        }


@dataclass(frozen=True)
class MonteCarloConfig:
    M: int
    master_seed: int


@dataclass(frozen=True)
class ParameterBlock:
    """Assigned primitives only; the calibrated vector comes from the params file."""
    beta: float
    delta_K: float
    phi: float
    H: int


@dataclass(frozen=True)
class ParticipationConfig:
    K_switch: int
    love_of_variety: bool


@dataclass(frozen=True)
class SimulationConfig:
    path: Path
    bundle_dir: Path
    params_file: Path
    out_results_dir: Path
    out_figs_dir: Path
    monte_carlo: MonteCarloConfig
    welfare_M: int
    market: MarketSolverConfig
    parameters: ParameterBlock
    participation: ParticipationConfig
    log_level: str
    raw: Mapping[str, Any]


def _resolve(path_str: str, anchor: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (anchor / path).resolve()


def _toggle(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"on", "true", "yes", "1"}:
        return True
    if text in {"off", "false", "no", "0"}:
        return False
    raise ValueError(f"invalid on/off toggle {value!r}")


def load_config(path: Path | str | None = None) -> SimulationConfig:
    cfg_path = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as stream:
        raw = yaml.safe_load(stream)
    paths = raw["paths"]
    mc = raw["monte_carlo"]
    solver = raw["solver"]
    market = solver["market"]
    parallel = solver.get("parallel", {}) or {}
    backend = str(market.get("backend", "vectorized"))
    if backend not in {"scalar", "vectorized"}:
        raise ValueError("solver.market.backend must be 'scalar' or 'vectorized'")
    parallel_n_jobs = int(parallel.get("n_jobs", 1))
    parallel_min_markets = int(parallel.get("min_markets", 0))
    parallel_min_markets_per_job = int(parallel.get("min_markets_per_job", 1))
    if parallel_n_jobs == 0:
        raise ValueError("solver.parallel.n_jobs cannot be zero")
    if parallel_min_markets < 0:
        raise ValueError("solver.parallel.min_markets must be non-negative")
    if parallel_min_markets_per_job <= 0:
        raise ValueError("solver.parallel.min_markets_per_job must be positive")
    assigned = raw.get("parameters", {}) or {}
    participation = raw.get("participation", {}) or {}
    welfare = raw.get("welfare", {}) or {}
    return SimulationConfig(
        path=cfg_path,
        bundle_dir=_resolve(paths["bundle_dir"], cfg_path.parent),
        params_file=_resolve(paths["params_file"], cfg_path.parent),
        out_results_dir=_resolve(paths["out_results_dir"], cfg_path.parent),
        out_figs_dir=_resolve(paths["out_figs_dir"], cfg_path.parent),
        monte_carlo=MonteCarloConfig(int(mc["M"]), int(mc["master_seed"])),
        welfare_M=int(welfare.get("M", mc["M"])),
        market=MarketSolverConfig(
            float(market["tol"]), int(market["max_iter"]), float(market["damping"]),
            int(market["newton_fallback_after"]), backend,
            ParallelSolverConfig(
                n_jobs=parallel_n_jobs,
                min_markets=parallel_min_markets,
                min_markets_per_job=parallel_min_markets_per_job,
            ),
        ),
        parameters=ParameterBlock(
            beta=float(assigned.get("beta", 0.96)),
            delta_K=float(assigned.get("delta_K", 0.06)),
            phi=float(assigned.get("phi", 1.0)), H=int(assigned.get("H", 1000)),
        ),
        participation=ParticipationConfig(
            K_switch=int(participation.get("K_switch", 3)),
            love_of_variety=_toggle(participation.get("love_of_variety", "off")),
        ),
        log_level=str(raw.get("log_level", "INFO")),
        raw=raw,
    )


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Stdout logger in the style of the Orbis moments pipeline (04_Moments)."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger
