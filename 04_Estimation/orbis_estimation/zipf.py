"""Market-free calibration: ``(xi, N, gamma, rho_bar)`` with ``eta`` fixed, Zipf tail targets.

A parallel calibration next to :mod:`.inner` (which it leaves untouched). Design,
diagnostics and acceptance criteria:
``02_Drafts/ARCHIVE/md_files/ORBIS_migration/ORBIS_ZIPF_CALIBRATION.md`` (Section 8).

* ``eta`` is fixed (1.2 in ``config_estimation_zipf.yaml``); ``gamma`` targets
  ``mu_cw``; ``(xi, N)`` target the rank-1/2 OLS Zipf index of pooled active-firm
  sales at the top 5% and top 2% (:func:`tail_index`); ``rho_bar`` targets the
  RTS gap. Four parameters, four moments: exactly identified.
* The model moment is computed on pooled sales of all active firms. Markets are
  ex ante identical, so no market normalization is needed (the index is scale
  invariant); the data side normalizes by the NAICS2 mean
  (``04_Moments/01_code/12_zipf_targets.py``).
* Evaluations go through :func:`.objective.pooled_moments_at_params` (anchored
  GE, ``v_min`` solved inside). The model settings must come from the
  simulation config (:func:`model_kwargs`): every solver argument is a required
  keyword here, because the defaults of ``pooled_moments_at_params`` give wrong
  moments (CR20 0.77 instead of 0.51).
* Coordinates: ``log xi, log N, log(gamma - eta), logit rho_bar`` (as
  :mod:`.inner` without the ``eta`` coordinate).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import least_squares

from orbis_model.config import SimulationConfig
from orbis_model.config import load_config as load_simulation_config
from orbis_model.model.aggregator import RTS_GAP_KEY
from orbis_model.model.normalization import FullPoolReference, PooledEquilibrium, PooledParams
from orbis_model.model.pool import alpha_cap
from orbis_model.primitives import ModelPrimitives
from .anchor_diagnostics import anchor_diagnostics
from .inner import DEFAULT_RHO_BAR_BOUNDS, _checkpoint_moments, _rho_to_x, _x_to_rho
from .objective import pooled_moments_at_params, weighted_relative_residuals
from .targets import load_targets


EST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = EST_DIR / "config_estimation_zipf.yaml"

ZIPF_CUTOFFS: dict[str, float] = {"zeta_top5": 0.05, "zeta_top2": 0.02, "zeta_top1": 0.01}
ZIPF_TARGET_KEYS: tuple[str, ...] = ("mu_cw", "zeta_top5", "zeta_top2", RTS_GAP_KEY)
PARAMETER_NAMES: tuple[str, ...] = ("xi", "N", "gamma", "rho_bar")
BOUND_NAMES: tuple[str, ...] = ("xi", "N", "gamma_gap", "rho_bar")
MODEL_KWARG_NAMES: tuple[str, ...] = (
    "beta", "delta_K", "K_switch", "love_of_variety", "market_solver_kwargs", "alpha_cap_factor",
)


# --------------------------------------------------------------------------- moments

def tail_index(x: np.ndarray, frac: float) -> float:
    """Gabaix-Ibragimov rank-1/2 OLS tail index on the top ``frac`` of ``x``.

    The md file's reference implementation, identical to the data side
    (``12_zipf_targets.py``): ``k = max(int(frac * n), 20)``.
    """
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(int(frac * len(x)), 20)
    xs = x[:k]
    rk = np.arange(1, k + 1) - 0.5
    return float(-np.polyfit(np.log(xs), np.log(rk), 1)[0])


def hill_index(x: np.ndarray, frac: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(int(frac * len(x)), 20)
    xs = x[:k]
    return float(1.0 / np.mean(np.log(xs[:-1] / xs[-1])))


def active_sales(eq: PooledEquilibrium) -> np.ndarray:
    sol, mask = eq.participation.solution, eq.participation.active_mask
    return np.asarray(sol.sales, dtype=float)[np.asarray(mask, dtype=bool)]


def zipf_moments(eq: PooledEquilibrium) -> dict[str, float]:
    """``zeta_top5``, ``zeta_top2`` (targets), ``zeta_top1`` (untargeted) and Hill variants."""
    sales = active_sales(eq)
    out: dict[str, float] = {"n_active_firms": float(sales.size)}
    for key, frac in ZIPF_CUTOFFS.items():
        out[key] = tail_index(sales, frac)
        out[key.replace("zeta_", "zeta_hill_")] = hill_index(sales, frac)
    return out


def model_kwargs(sim: SimulationConfig) -> dict[str, Any]:
    """The model settings of the simulation config, as ``pooled_moments_at_params`` keywords."""
    return {
        "beta": sim.parameters.beta,
        "delta_K": sim.parameters.delta_K,
        "K_switch": sim.participation.K_switch,
        "love_of_variety": sim.participation.love_of_variety,
        "market_solver_kwargs": sim.market.to_kwargs(),
        "alpha_cap_factor": sim.parameters.alpha_cap_factor,
    }


def zipf_moments_at_params(
    params: PooledParams,
    primitives: ModelPrimitives,
    *,
    M: int,
    rng,
    beta: float,
    delta_K: float,
    K_switch: int,
    love_of_variety: bool | str,
    market_solver_kwargs: dict,
    alpha_cap_factor: float | None,
) -> tuple[dict[str, float], FullPoolReference, PooledEquilibrium]:
    """:func:`pooled_moments_at_params` plus the Zipf moments; every model setting required."""
    if market_solver_kwargs is None:
        raise ValueError("market_solver_kwargs must come from the simulation config (model_kwargs)")
    model, reference, eq = pooled_moments_at_params(
        params, primitives, M=M, rng=rng, beta=beta, delta_K=delta_K, K_switch=K_switch,
        love_of_variety=love_of_variety, market_solver_kwargs=market_solver_kwargs,
        alpha_cap_factor=alpha_cap_factor,
    )
    model = dict(model)
    model.update(zipf_moments(eq))
    return model, reference, eq


# --------------------------------------------------------------------------- targets

def read_zipf_targets(path: Path | str) -> dict[str, Any]:
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if data.get("schema") != "orbis-zipf-targets-v1":
        raise ValueError(f"{path}: not a zipf_targets.yaml (schema {data.get('schema')!r})")
    return data


def load_zipf_targets(bundle_dir: Path | str, zipf_targets_path: Path | str) -> dict[str, float]:
    """Bundle targets (``mu_cw``, RTS gap, and the untargeted rest) plus the Zipf indices."""
    targets = load_targets(bundle_dir)
    zipf = read_zipf_targets(zipf_targets_path)
    for key in ("zeta_top5", "zeta_top2"):
        targets[key] = float(zipf["targets"][key])
    targets["zeta_top1"] = float(zipf["untargeted"]["zeta_top1"])
    return targets


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class ZipfConfig:
    path: Path
    bundle_dir: Path
    zipf_targets: Path
    out_results_dir: Path
    simulation: SimulationConfig
    M: int
    seed: int
    eta: float
    weights: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    initial_values: dict[str, float]
    ftol: float
    xtol: float
    max_nfev: int
    fd_step: float
    varphi: float
    log_level: str
    raw: Mapping[str, Any]

    @property
    def moment_keys(self) -> tuple[str, ...]:
        return tuple(self.weights)


def _resolve(path_str: str, anchor: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (anchor / path).resolve()


def load_zipf_config(path: Path | str | None = None) -> ZipfConfig:
    cfg_path = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as stream:
        raw = yaml.safe_load(stream)
    paths = raw["paths"]
    bundle_dir = _resolve(paths["bundle_dir"], cfg_path.parent)
    simulation = load_simulation_config(_resolve(paths["simulation_config"], cfg_path.parent))
    if simulation.bundle_dir != bundle_dir:
        raise ValueError(f"bundle_dir {bundle_dir} differs from the simulation config's {simulation.bundle_dir}")
    eta = float(raw["fixed"]["eta"])
    if not eta > 1.0:
        raise ValueError("fixed.eta must exceed one")
    weights = {str(k): float(v) for k, v in (raw.get("weights") or {}).items()}
    if not weights:
        raise ValueError("weights must list the targeted moments")
    block = raw.get("bounds") or {}
    missing = [k for k in BOUND_NAMES if k not in block]
    if missing:
        raise ValueError(f"bounds missing {missing}")
    bounds = {k: (float(block[k]["lo"]), float(block[k]["hi"])) for k in BOUND_NAMES}
    for name, (lo, hi) in bounds.items():
        if not lo < hi:
            raise ValueError(f"bounds.{name} must satisfy lo < hi")
    if not (bounds["xi"][0] > 1.0 and bounds["N"][0] >= 1.0 and bounds["gamma_gap"][0] > 0.0):
        raise ValueError("bounds: need xi.lo > 1, N.lo >= 1, gamma_gap.lo > 0")
    if not -1.0 < bounds["rho_bar"][0] < bounds["rho_bar"][1] < 1.0:
        raise ValueError("bounds.rho_bar must lie inside (-1, 1)")
    if bounds["N"][1] >= simulation.parameters.H:
        raise ValueError(f"bounds.N.hi {bounds['N'][1]} must stay below H = {simulation.parameters.H}")
    initial = raw.get("initial_values") or {}
    missing = [k for k in PARAMETER_NAMES if k not in initial]
    if missing:
        raise ValueError(f"initial_values missing {missing}")
    inner = raw.get("inner") or {}
    mc = raw["monte_carlo"]
    return ZipfConfig(
        path=cfg_path, bundle_dir=bundle_dir,
        zipf_targets=_resolve(paths["zipf_targets"], cfg_path.parent),
        out_results_dir=_resolve(paths["out_results_dir"], cfg_path.parent),
        simulation=simulation, M=int(mc["M"]), seed=int(mc["seed"]), eta=eta,
        weights=weights, bounds=bounds,
        initial_values={k: float(initial[k]) for k in PARAMETER_NAMES},
        ftol=float(inner.get("ftol", 1e-8)), xtol=float(inner.get("xtol", 1e-8)),
        max_nfev=int(inner.get("max_nfev", 25)), fd_step=float(inner.get("fd_step", 1e-2)),
        varphi=float((raw.get("entry") or {}).get("varphi", 0.04)),
        log_level=str(raw.get("log_level", "INFO")), raw=raw,
    )


# --------------------------------------------------------------------------- transforms

def encode(
    values: Sequence[float], *, eta: float,
    rho_bar_bounds: tuple[float, float] = DEFAULT_RHO_BAR_BOUNDS,
) -> np.ndarray:
    """``(xi, N, gamma, rho_bar)`` to ``(log xi, log N, log(gamma - eta), logit rho_bar)``."""
    xi, N, gamma, rho_bar = map(float, values)
    if not gamma > eta:
        raise ValueError("initial gamma must exceed the fixed eta")
    if not xi > 1.0:
        raise ValueError("initial Pareto tail index xi must exceed 1")
    if N < 1.0:
        raise ValueError("initial Poisson mean N must be at least 1")
    lo, hi = rho_bar_bounds
    if not lo <= rho_bar < hi:
        raise ValueError("initial rho_bar outside bounds")
    return np.array([np.log(xi), np.log(N), np.log(gamma - eta), _rho_to_x(rho_bar, rho_bar_bounds)])


def decode(
    x: np.ndarray, *, eta: float, a: float, H: int, phi_v: float,
    rho_bar_bounds: tuple[float, float] = DEFAULT_RHO_BAR_BOUNDS,
) -> PooledParams:
    return PooledParams(
        xi=float(np.exp(x[0])), N=float(np.exp(x[1])), gamma=float(eta + np.exp(x[2])),
        eta=float(eta), rho_bar=_x_to_rho(float(x[3]), rho_bar_bounds), a=a, H=H, phi_v=phi_v,
    )


def structural_transform(params: PooledParams, rho_bar_bounds: tuple[float, float]) -> np.ndarray:
    """``d(xi, N, gamma, rho_bar) / d x`` (diagonal) at ``params``."""
    lo, hi = rho_bar_bounds
    scaled = (params.rho_bar - lo) / (hi - lo)
    return np.diag([params.xi, params.N, params.gamma - params.eta, (hi - lo) * scaled * (1.0 - scaled)])


def moment_elasticities(
    jac_x: np.ndarray, params: PooledParams, model: Mapping[str, float], targets: Mapping[str, float],
    *, moment_keys: Sequence[str], weights: Mapping[str, float],
    rho_bar_bounds: tuple[float, float] = DEFAULT_RHO_BAR_BOUNDS,
) -> np.ndarray:
    """``d log m / d log p`` for ``(xi, N, gamma)`` and ``d log m / d rho_bar`` from the residual Jacobian.

    Undoes the residual scaling ``w (m - t) / (1 + |t|)`` row by row and the
    coordinate transform column by column.
    """
    scale = np.array([(1.0 + abs(float(targets[k]))) / (float(weights.get(k, 1.0)) * float(model[k]))
                      for k in moment_keys])
    dlogm_dx = jac_x * scale[:, None]
    T = np.diag(structural_transform(params, rho_bar_bounds))
    dlogm_dp = dlogm_dx / T[None, :]
    log_scale = np.array([params.xi, params.N, params.gamma, 1.0])
    return dlogm_dp * log_scale[None, :]


# --------------------------------------------------------------------------- calibration

@dataclass(frozen=True)
class ZipfCalibrationResult:
    params: PooledParams
    cost: float
    residuals: np.ndarray
    moment_keys: tuple[str, ...]
    model_moments: Mapping[str, float]
    target_moments: Mapping[str, float]
    reference: FullPoolReference
    equilibrium: PooledEquilibrium
    success: bool
    status: int
    nfev: int
    n_evals: int
    message: str
    jacobian_x: np.ndarray
    jacobian_condition: float
    elasticities: np.ndarray
    elasticity_condition: float
    zeta_block_condition: float
    at_bound: tuple[str, ...]


def calibrate_zipf(
    primitives: ModelPrimitives,
    targets: Mapping[str, float],
    *,
    M: int,
    rng,
    eta: float,
    initial: Sequence[float],
    bounds: Mapping[str, tuple[float, float]],
    weights: Mapping[str, float],
    beta: float,
    delta_K: float,
    K_switch: int,
    love_of_variety: bool | str,
    market_solver_kwargs: dict,
    alpha_cap_factor: float | None,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    max_nfev: int = 25,
    fd_step: float = 1e-2,
    checkpoint_path: str | Path | None = None,
    logger=None,
) -> ZipfCalibrationResult:
    """Fit ``(xi, N, gamma, rho_bar)`` at fixed ``eta`` by ``trf`` least squares.

    ``initial`` is ``(xi, N, gamma, rho_bar)``; ``bounds`` has ``xi``, ``N``,
    ``gamma_gap`` (``gamma - eta``) and ``rho_bar``. Residuals are
    :func:`.objective.weighted_relative_residuals` over the keys of ``weights``.
    The Jacobian is a forward difference with absolute step ``fd_step`` in the
    transformed coordinates (as :func:`.inner.calibrate_pooled`). Every
    evaluation (fit and difference steps) appends one JSON line to
    ``checkpoint_path``. An evaluation that raises is recorded and returns a
    large residual, so ``trf`` shrinks its step instead of the job dying.
    """
    moment_keys = tuple(weights)
    rho_bounds = tuple(bounds["rho_bar"])
    kw = dict(beta=beta, delta_K=delta_K, K_switch=K_switch, love_of_variety=love_of_variety,
              market_solver_kwargs=market_solver_kwargs, alpha_cap_factor=alpha_cap_factor)
    lower = np.array([np.log(bounds["xi"][0]), np.log(bounds["N"][0]), np.log(bounds["gamma_gap"][0]), -np.inf])
    upper = np.array([np.log(bounds["xi"][1]), np.log(bounds["N"][1]), np.log(bounds["gamma_gap"][1]), np.inf])
    x0 = np.clip(encode(initial, eta=eta, rho_bar_bounds=rho_bounds), lower + 1e-9, upper - 1e-9)

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    if ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("")  # truncate any stale checkpoint from a prior run
    count = [0]
    penalty = np.array([10.0 * float(weights[k]) for k in moment_keys])

    def to_params(x: np.ndarray) -> PooledParams:
        return decode(x, eta=eta, a=primitives.a, H=primitives.H, phi_v=primitives.phi_v,
                      rho_bar_bounds=rho_bounds)

    def residual(x: np.ndarray) -> np.ndarray:
        count[0] += 1
        params = to_params(x)
        t0 = time.time()
        record: dict[str, Any] = {"eval": count[0], "xi": params.xi, "N": params.N,
                                  "gamma": params.gamma, "eta": params.eta, "rho_bar": params.rho_bar}
        try:
            model, _, eq = zipf_moments_at_params(params, primitives, M=M, rng=rng, **kw)
            arr = weighted_relative_residuals(model, targets, moment_keys=moment_keys, weights=weights)
            sol = eq.participation.solution
            record.update({
                "cost": 0.5 * float(arr @ arr),
                "residuals": dict(zip(moment_keys, map(float, arr))),
                "ge_converged": bool(eq.converged), "ge_residual": float(eq.residual),
                "markets_converged_share": float(np.mean(np.asarray(sol.converged))),
                "v_min": float(eq.v_min), "y_sw": float(eq.y_sw), "y_anchor": float(eq.y_anchor),
                "moments": _checkpoint_moments(model),
            })
            try:
                record.update(anchor_diagnostics(eq))
            except Exception:  # diagnostics only
                pass
        except Exception as exc:
            arr = penalty.copy()
            record.update({"cost": 0.5 * float(arr @ arr), "error": f"{type(exc).__name__}: {exc}"})
            if logger is not None:
                logger.error(f"eval {count[0]} failed at {params}: {record['error']}")
        record["secs"] = round(time.time() - t0, 2)
        if ckpt is not None:
            with ckpt.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        if logger is not None:
            logger.info(f"eval {count[0]:3d} xi={params.xi:.5g} N={params.N:.5g} gamma={params.gamma:.5g} "
                        f"rho_bar={params.rho_bar:.4g} cost={record['cost']:.4g} ({record['secs']:.0f} s)")
        last[0] = (np.array(x, dtype=float), arr)
        return arr

    last: list[tuple[np.ndarray, np.ndarray] | None] = [None]

    def jacobian(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if last[0] is not None and np.array_equal(last[0][0], x):
            f0 = last[0][1]  # trf evaluates fun(x) before jac(x); reuse it
        else:
            f0 = residual(x)
        jac = np.empty((f0.size, x.size))
        for j in range(x.size):
            h = fd_step if x[j] + fd_step <= upper[j] else -fd_step
            xj = x.copy()
            xj[j] += h
            jac[:, j] = (residual(xj) - f0) / h
        return jac

    sol = least_squares(
        residual, x0=x0, jac=jacobian, bounds=(lower, upper), method="trf",
        ftol=ftol, xtol=xtol, gtol=1e-10, max_nfev=max_nfev, verbose=2,
    )
    params = to_params(sol.x)
    model, reference, equilibrium = zipf_moments_at_params(params, primitives, M=M, rng=rng, **kw)
    params = PooledParams(
        xi=params.xi, N=params.N, gamma=params.gamma, eta=params.eta, v_min=equilibrium.v_min,
        y_hat=equilibrium.y_hat, rho_bar=params.rho_bar, a=params.a, H=params.H, phi_v=params.phi_v,
    )
    equilibrium = PooledEquilibrium(**{**equilibrium.__dict__, "v_min": params.v_min})
    model = dict(model)
    try:
        diag = anchor_diagnostics(equilibrium)
        for key in ("corr_alpha_log_sales", "corr_alpha_log_sales_partial_z", "corr_alpha_log_sales_partial_v"):
            model[key] = float(diag[key])
    except Exception:
        pass
    residuals = weighted_relative_residuals(model, targets, moment_keys=moment_keys, weights=weights)
    jac = np.asarray(sol.jac, dtype=float)
    elas = moment_elasticities(jac, params, model, targets, moment_keys=moment_keys,
                               weights=weights, rho_bar_bounds=rho_bounds)
    zeta_rows = [moment_keys.index(k) for k in ("zeta_top5", "zeta_top2") if k in moment_keys]
    zeta_block = elas[np.ix_(zeta_rows, [0, 1])] if len(zeta_rows) == 2 else np.full((2, 2), np.nan)
    names = ("xi", "N", "gamma_gap", "rho_bar")
    at_bound = tuple(n for n, xv, lo, hi in zip(names, sol.x, lower, upper)
                     if np.isfinite(lo) and (xv - lo < 1e-6 or hi - xv < 1e-6))
    return ZipfCalibrationResult(
        params=params, cost=0.5 * float(residuals @ residuals), residuals=residuals,
        moment_keys=moment_keys, model_moments=model, target_moments=dict(targets),
        reference=reference, equilibrium=equilibrium, success=bool(sol.success), status=int(sol.status),
        nfev=int(sol.nfev), n_evals=count[0], message=str(sol.message), jacobian_x=jac,
        jacobian_condition=float(np.linalg.cond(jac)), elasticities=elas,
        elasticity_condition=float(np.linalg.cond(elas)),
        zeta_block_condition=float(np.linalg.cond(zeta_block)) if np.all(np.isfinite(zeta_block)) else float("nan"),
        at_bound=at_bound,
    )


# --------------------------------------------------------------------------- validation

def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: Sequence[float]) -> dict[str, float]:
    o = np.argsort(x, kind="stable")
    x, w = np.asarray(x, dtype=float)[o], np.asarray(w, dtype=float)[o]
    c = np.cumsum(w) / w.sum()
    return {f"p{int(round(q * 100))}": float(x[min(np.searchsorted(c, q), len(x) - 1)]) for q in qs}


def validation_moments(
    eq: PooledEquilibrium, params: PooledParams, *, alpha_support: np.ndarray,
    alpha_cap_factor: float | None, quantiles: Sequence[float] = (0.5, 0.9, 0.99),
) -> dict[str, float]:
    """Untargeted outputs of Section 8.7 on the active firms of one solved economy.

    Markups: true ``mu`` and the common-elasticity measured markup
    ``mu * abar / alpha`` (``abar`` the cost-weighted mean alpha, as the data's
    one-elasticity-per-cell ``markup_wi``). Pooled top shares; the share of the
    top 1% of firms (by sales) in the top alpha node of the truncated support
    and in the top 1% of its nodes; ``mu_cw`` split into the floor
    ``gamma/(gamma-1)`` and the share-driven rest.
    """
    sol, mask = eq.participation.solution, np.asarray(eq.participation.active_mask, dtype=bool)
    S, C, MU, A = (np.asarray(v, dtype=float)[mask] for v in (sol.sales, sol.cost, sol.mu, eq.alpha))
    abar = float(np.average(A, weights=C))
    out: dict[str, float] = {"abar_cw": abar}
    for tag, mu in (("true", MU), ("meas", MU * abar / A)):
        s, c = S / S.sum(), C / C.sum()
        lm = np.log(mu)
        out[f"mu_cw_{tag}"] = float(c @ mu)
        out[f"mu_sw_{tag}"] = float(s @ mu)
        out[f"gap_sw_cw_{tag}"] = float(s @ mu - c @ mu)
        out[f"var_sw_log_{tag}"] = float(s @ (lm - s @ lm) ** 2)
        out[f"var_cw_log_{tag}"] = float(c @ (lm - c @ lm) ** 2)
        for lab, w in (("uw", np.ones_like(mu)), ("cw", C), ("sw", S)):
            for q, v in _weighted_quantiles(mu, w, quantiles).items():
                out[f"{q}_{lab}_{tag}"] = v
    r = np.sort(S)[::-1]
    for q in (0.001, 0.01, 0.05):
        out[f"pooled_top{q * 100:g}pct"] = float(r[:max(1, int(q * r.size))].sum() / r.sum())
    support = np.sort(np.asarray(alpha_support, dtype=float))
    cap = alpha_cap(params.gamma, alpha_cap_factor)
    kept = support[support < cap] if cap is not None else support
    top = S >= np.quantile(S, 0.99)
    out["alpha_cap"] = float(cap) if cap is not None else float("nan")
    out["alpha_top_node"] = float(kept[-1])
    out["share_top1pct_firms_top_node"] = float(np.mean(A[top] >= kept[-1]))
    out["share_top1pct_firms_top1pct_nodes"] = float(np.mean(A[top] >= kept[int(np.floor(0.99 * kept.size))]))
    out["share_all_firms_top_node"] = float(np.mean(A >= kept[-1]))
    out["mean_alpha_top1pct_firms"] = float(A[top].mean())
    out["mu_sw_top1pct_firms"] = float(np.average(MU[top], weights=S[top]))
    floor = params.gamma / (params.gamma - 1.0)
    out["mu_floor"] = float(floor)
    out["mu_cw_share_part"] = float(out["mu_cw_true"] - floor)
    out["mu_cw_share_part_frac"] = float((out["mu_cw_true"] - floor) / (out["mu_cw_true"] - 1.0))
    out["markets_converged_share"] = float(np.mean(np.asarray(sol.converged)))
    return out
