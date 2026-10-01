"""Typed configuration for the pooled steady-state pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PKG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PKG_DIR / "config.yaml"


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
    M_final: int
    master_seed: int


@dataclass(frozen=True)
class ParameterBlock:
    """Assigned primitives only; eta is deliberately absent."""
    beta: float
    delta_K: float
    phi: float
    H: int


@dataclass(frozen=True)
class ParticipationConfig:
    K_switch: int
    love_of_variety: bool


@dataclass(frozen=True)
class CalibrationConfig:
    eta_init: float
    eta_lo: float
    xi_lo: float
    xi_hi: float
    N_lo: float
    N_hi: float
    max_nfev: int
    markup_target: str = "sga"  # empirical base for mu_cw: "sga" | "cogs"


@dataclass(frozen=True)
class Config:
    bundle_dir: Path
    out_results_dir: Path
    out_figs_dir: Path
    active_window: tuple[int, int]
    monte_carlo: MonteCarloConfig
    market: MarketSolverConfig
    parameters: ParameterBlock
    participation: ParticipationConfig
    calibration: CalibrationConfig
    exposure_uniform: bool
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


def load_config(path: Path | str | None = None) -> Config:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as stream:
        raw = yaml.safe_load(stream)
    paths = raw["paths"]
    window = raw["window"][raw["window"]["active"]]
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
    calibration = raw.get("calibration", {}) or {}
    eta = calibration.get("eta", {}) or {}
    tail = calibration.get("xi", {}) or {}
    firm_count = calibration.get("N", {}) or {}
    eta_lo = float(eta.get("lo", 1.01))
    if eta_lo <= 1.0:
        raise ValueError("calibration.eta.lo must be strictly above one")
    xi_lo = float(tail.get("lo", 1.05))
    xi_hi = float(tail.get("hi", 30.0))
    if not 1.0 < xi_lo < xi_hi:
        raise ValueError("calibration.xi bounds must satisfy 1 < lo < hi")
    N_lo = float(firm_count.get("lo", 5.0))
    N_hi = float(firm_count.get("hi", 600.0))
    if not 1.0 <= N_lo < N_hi:
        raise ValueError("calibration.N bounds must satisfy 1 <= lo < hi")
    markup_target = str(calibration.get("markup_target", "sga")).lower()
    if markup_target not in {"sga", "cogs"}:
        raise ValueError("calibration.markup_target must be 'sga' or 'cogs'")
    # NOTE: the RTS shrink + clip now live in the empirical pipeline (Stage S5);
    # any legacy `alpha_support` config block is ignored (the bundle ships the
    # final model-facing support).
    return Config(
        bundle_dir=_resolve(paths["bundle_dir"], cfg_path.parent),
        out_results_dir=_resolve(paths["out_results_dir"], cfg_path.parent),
        out_figs_dir=_resolve(paths["out_figs_dir"], cfg_path.parent),
        active_window=(int(window[0]), int(window[1])),
        monte_carlo=MonteCarloConfig(int(mc["M"]), int(mc["M_final"]), int(mc["master_seed"])),
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
        calibration=CalibrationConfig(
            eta_init=float(eta.get("init", 2.0)), eta_lo=eta_lo,
            xi_lo=xi_lo, xi_hi=xi_hi, N_lo=N_lo, N_hi=N_hi,
            max_nfev=int((calibration.get("inner", {}) or {}).get("max_nfev", 40)),
            markup_target=markup_target,
        ),
        exposure_uniform=bool((raw.get("exposure", {}) or {}).get("uniform", True)),
        raw=raw,
    )
