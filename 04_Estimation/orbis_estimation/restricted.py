"""Restricted-heterogeneity recalibrations of the market-free (Zipf) model.

Every restricted economy is the canonical model on modified inputs, so the
moment function is :func:`.zipf.zipf_moments_at_params` unchanged:

* common capability (``hom_v``): ``xi = inf``. The Pareto quantile in
  :func:`orbis_model.model.pool.draw_pool` then returns ``v = v_min`` exactly,
  and ``v_min`` is still solved by the anchor.
* common scalability (``hom_alpha``): a one-node ``alpha_support`` at the
  harmonic mean of ``F^alpha`` truncated at ``alpha_cap(gamma)``
  (:func:`common_alpha`), recomputed at each evaluation's ``gamma``.

``rho_bar`` has no role once either primitive is common and is pinned at 0.
Every economy minimizes the baseline objective (same weights and bounds)
over its own free parameters (:data:`ECONOMIES`), restricted to the targets it
can reach: ``hom_alpha`` fits ``(xi, N, gamma)`` to all four (its RTS gap is
zero by construction); ``hom_v`` and ``hom_both`` cannot reach the Zipf indices
or the RTS gap, so they fit ``gamma`` to ``mu_cw`` with ``N`` at the baseline
value. The target subset is set in the driver's config.

Evaluations are checkpointed with the exact ``x`` (hex floats) and residuals,
and a restart replays them: ``trf`` is deterministic, so it fast-forwards to
where the previous process stopped. An evaluation that runs past the watchdog
limit is recorded as a timeout and the process exits with
:data:`EXIT_TIMEOUT`; on replay that point returns the penalty residual, so
``trf`` shrinks its step instead of stalling there again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from orbis_model.model.aggregator import RTS_GAP_KEY
from orbis_model.model.normalization import FullPoolReference, PooledEquilibrium, PooledParams
from orbis_model.model.pool import alpha_cap
from orbis_model.primitives import ModelPrimitives
from .anchor_diagnostics import anchor_diagnostics
from .inner import _checkpoint_moments, _rho_to_x, _x_to_rho
from .objective import weighted_relative_residuals
from .zipf import zipf_moments_at_params


ALL_PARAMETERS: tuple[str, ...] = ("xi", "N", "gamma", "rho_bar")
EXIT_TIMEOUT = 75   # EX_TEMPFAIL: the submit script restarts the process on this code


@dataclass(frozen=True)
class Economy:
    """Which primitive is common, and which parameters are calibrated."""

    label: str
    free: tuple[str, ...]
    common_alpha: bool
    common_v: bool


ECONOMIES: dict[str, Economy] = {
    "baseline": Economy("baseline", ("xi", "N", "gamma", "rho_bar"), False, False),
    "hom_alpha": Economy("hom_alpha", ("xi", "N", "gamma"), True, False),
    "hom_v": Economy("hom_v", ("gamma",), False, True),
    "hom_both": Economy("hom_both", ("gamma",), True, True),
}


# --------------------------------------------------------------------------- inputs

def common_alpha(support: np.ndarray, gamma: float, factor: float | None) -> float:
    """Harmonic mean of the ``F^alpha`` nodes below ``alpha_cap(gamma, factor)``.

    The support nodes have equal mass, so this is ``1 / E[1/alpha]`` under the
    truncated draw of :func:`orbis_model.model.pool.draw_pool`. It is strictly
    below the cap.
    """
    nodes = np.sort(np.asarray(support, dtype=np.float64))
    cap = alpha_cap(gamma, factor)
    if cap is not None:
        nodes = nodes[nodes < cap]
    if nodes.size == 0:
        raise ValueError(f"alpha cap {cap} lies below the whole alpha support")
    return float(1.0 / np.mean(1.0 / nodes))


def economy_inputs(
    economy: Economy, primitives: ModelPrimitives, params: PooledParams, *, alpha_cap_factor: float | None,
) -> tuple[ModelPrimitives, PooledParams]:
    """The canonical inputs that implement ``economy`` at ``params``.

    Common v sets ``xi = inf``; common alpha replaces the support by one node
    (:func:`common_alpha`); either pins ``rho_bar`` at 0. The baseline returns
    its inputs unchanged.
    """
    if economy.common_alpha or economy.common_v:
        params = dataclasses.replace(params, rho_bar=0.0, xi=np.inf if economy.common_v else params.xi)
    if economy.common_alpha:
        node = common_alpha(primitives.alpha_support, params.gamma, alpha_cap_factor)
        primitives = dataclasses.replace(primitives, alpha_support=np.array([node]))
    return primitives, params


def economy_moments_at_params(
    economy: Economy, params: PooledParams, primitives: ModelPrimitives, *, M: int, rng,
    alpha_cap_factor: float | None, **kw,
) -> tuple[dict[str, float], FullPoolReference, PooledEquilibrium, ModelPrimitives, PooledParams]:
    """:func:`.zipf.zipf_moments_at_params` on :func:`economy_inputs`; also returns those inputs."""
    prim, par = economy_inputs(economy, primitives, params, alpha_cap_factor=alpha_cap_factor)
    model, reference, eq = zipf_moments_at_params(
        par, prim, M=M, rng=rng, alpha_cap_factor=alpha_cap_factor, **kw)
    if economy.common_alpha and not np.isfinite(model.get(RTS_GAP_KEY, 0.0)):
        # alpha minus its market mean is zero for every firm, so the gap is zero for any
        # rank split; the canonical moment is NaN only when equal sales leave the top 5%
        # empty (common v as well).
        model = {**model, RTS_GAP_KEY: 0.0, "rts_gap_zero_by_construction": 1.0}
    return model, reference, eq, prim, par


def structure_checks(economy: Economy, eq: PooledEquilibrium, params: PooledParams,
                     primitives: ModelPrimitives, *, alpha_cap_factor: float | None) -> dict[str, Any]:
    """The restriction holds in the solved economy: one value where common, every alpha below the cap."""
    mask = np.asarray(eq.participation.active_mask, dtype=bool)
    alpha = np.asarray(eq.alpha, dtype=float)[mask]
    v = np.asarray(eq.v, dtype=float)[mask]
    cap = alpha_cap(params.gamma, alpha_cap_factor)
    out: dict[str, Any] = {
        "alpha_unique": int(np.unique(alpha).size), "v_unique": int(np.unique(v).size),
        "alpha_below_cap": bool(cap is None or np.all(alpha < cap)),
    }
    ok = out["alpha_below_cap"]
    if economy.common_alpha:
        out["alpha_common"] = float(primitives.alpha_support[0])
        ok &= out["alpha_unique"] == 1 and bool(np.all(alpha == primitives.alpha_support[0]))
    if economy.common_v:
        ok &= out["v_unique"] == 1 and bool(np.all(v == eq.v_min))
    out["passed"] = bool(ok)
    return out


# --------------------------------------------------------------------------- coordinates

def encode(values: Mapping[str, float], free: Sequence[str], *, eta: float,
           rho_bar_bounds: tuple[float, float]) -> np.ndarray:
    """Free parameters to ``log xi, log N, log(gamma - eta), logit rho_bar`` (as :func:`.zipf.encode`)."""
    out = []
    for name in free:
        value = float(values[name])
        if name == "gamma":
            if not value > eta:
                raise ValueError("initial gamma must exceed the fixed eta")
            out.append(np.log(value - eta))
        elif name == "rho_bar":
            out.append(_rho_to_x(value, rho_bar_bounds))
        else:
            out.append(np.log(value))
    return np.array(out)


def decode(x: np.ndarray, free: Sequence[str], fixed: Mapping[str, float], *, eta: float,
           rho_bar_bounds: tuple[float, float], primitives: ModelPrimitives) -> PooledParams:
    values = dict(fixed)
    for name, xv in zip(free, np.asarray(x, dtype=float), strict=True):
        if name == "gamma":
            values[name] = float(eta + np.exp(xv))
        elif name == "rho_bar":
            values[name] = _x_to_rho(float(xv), rho_bar_bounds)
        else:
            values[name] = float(np.exp(xv))
    return PooledParams(
        xi=values["xi"], N=values["N"], gamma=values["gamma"], eta=float(eta),
        rho_bar=values["rho_bar"], a=primitives.a, H=primitives.H, phi_v=primitives.phi_v,
    )


def coordinate_bounds(free: Sequence[str], bounds: Mapping[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    key = {"xi": "xi", "N": "N", "gamma": "gamma_gap", "rho_bar": None}
    lo, hi = [], []
    for name in free:
        if key[name] is None:
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            lo.append(np.log(bounds[key[name]][0]))
            hi.append(np.log(bounds[key[name]][1]))
    return np.array(lo), np.array(hi)


# --------------------------------------------------------------------------- resumable evaluations

def _hex(x: np.ndarray) -> list[str]:
    return [float(v).hex() for v in np.asarray(x, dtype=float)]


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class Watchdog:
    """Exit the process with :data:`EXIT_TIMEOUT` if one evaluation runs past its limit.

    The limit is ``max(min_s, factor * median)`` of the completed evaluations
    once three have finished, and ``first_s`` before that. On expiry the
    ``on_timeout`` callback records the point, the market-solve worker
    processes are killed, and the process exits; the caller's restart replays
    the checkpoint.
    """

    def __init__(self, *, factor: float, min_s: float, first_s: float, on_timeout, logger=None):
        self.factor, self.min_s, self.first_s = float(factor), float(min_s), float(first_s)
        self.on_timeout, self.logger = on_timeout, logger
        self.durations: list[float] = []
        self._timer: threading.Timer | None = None

    def limit(self) -> float:
        if len(self.durations) < 3:
            return self.first_s
        return max(self.min_s, self.factor * float(np.median(self.durations)))

    def start(self, context: Mapping[str, Any]) -> None:
        limit = self.limit()

        def fire():
            if self.logger is not None:
                self.logger.error(f"evaluation exceeded {limit:.0f} s; recording a timeout and exiting")
            self.on_timeout(dict(context), limit)
            for child in multiprocessing.active_children():
                child.kill()
            os._exit(EXIT_TIMEOUT)

        self._timer = threading.Timer(limit, fire)
        self._timer.daemon = True
        self._timer.start()

    def stop(self, seconds: float | None) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if seconds is not None:
            self.durations.append(float(seconds))


class Checkpoint:
    """Append-only JSON-lines record of every evaluation, replayed on restart.

    The first line is a header with the run fingerprint; a restart with another
    fingerprint refuses to reuse the file.
    """

    def __init__(self, path: Path, run_fingerprint: str, header: Mapping[str, Any]):
        self.path = Path(path)
        self.cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.path.exists() and self.path.stat().st_size > 0:
            lines = self.path.read_text().splitlines()
            head = json.loads(lines[0])
            if head.get("fingerprint") != run_fingerprint:
                raise RuntimeError(f"{self.path} belongs to another run (fingerprint differs); "
                                   "choose another --out-dir or remove it")
            for line in lines[1:]:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:   # a line cut by a kill
                    continue
                if "x_hex" in record:
                    self.cache[tuple(record["x_hex"])] = record
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"fingerprint": run_fingerprint, **header}, default=str) + "\n")

    def get(self, x: np.ndarray) -> dict[str, Any] | None:
        return self.cache.get(tuple(_hex(x)))

    def append(self, record: Mapping[str, Any]) -> None:
        with self.lock:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, allow_nan=False, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if "x_hex" in record:
                self.cache[tuple(record["x_hex"])] = dict(record)


@dataclass
class Evaluator:
    """Residuals of one economy at reduced coordinates ``x``, with checkpoint replay."""

    economy: Economy
    primitives: ModelPrimitives
    targets: Mapping[str, float]
    weights: Mapping[str, float]
    fixed: Mapping[str, float]
    eta: float
    rho_bar_bounds: tuple[float, float]
    M: int
    rng: int
    alpha_cap_factor: float | None
    model_kw: Mapping[str, Any]
    checkpoint: Checkpoint
    watchdog: Watchdog | None = None
    logger: Any = None

    def __post_init__(self) -> None:
        self.moment_keys = tuple(self.weights)
        self.penalty = np.array([10.0 * float(self.weights[k]) for k in self.moment_keys])
        self.count = 0
        self.replayed = 0

    def params(self, x: np.ndarray) -> PooledParams:
        return decode(x, self.economy.free, self.fixed, eta=self.eta,
                      rho_bar_bounds=self.rho_bar_bounds, primitives=self.primitives)

    def solve(self, params: PooledParams):
        return economy_moments_at_params(
            self.economy, params, self.primitives, M=self.M, rng=self.rng,
            alpha_cap_factor=self.alpha_cap_factor, **self.model_kw)

    def residual(self, x: np.ndarray, *, tag: str = "fit") -> np.ndarray:
        self.count += 1
        cached = self.checkpoint.get(x)
        if cached is not None:
            self.replayed += 1
            arr = (np.array([float.fromhex(h) for h in cached["residuals_hex"]])
                   if "residuals_hex" in cached else self.penalty.copy())
            if self.logger is not None:
                self.logger.info(f"eval {self.count:3d} replayed ({cached.get('status', 'ok')}) "
                                 f"cost={0.5 * float(arr @ arr):.4g}")
            return arr
        params = self.params(x)
        _, effective = economy_inputs(self.economy, self.primitives, params,
                                      alpha_cap_factor=self.alpha_cap_factor)
        record: dict[str, Any] = {"eval": self.count, "tag": tag, "x_hex": _hex(x),
                                  **{k: float(getattr(effective, k)) for k in ALL_PARAMETERS}}
        t0 = time.time()
        if self.watchdog is not None:
            self.watchdog.start(record)
        try:
            model, _, eq, prim, _ = self.solve(params)
            arr = weighted_relative_residuals(model, self.targets, moment_keys=self.moment_keys,
                                              weights=self.weights)
            if not np.all(np.isfinite(arr)):
                bad = [k for k, r in zip(self.moment_keys, arr) if not np.isfinite(r)]
                raise FloatingPointError(f"non-finite moments {bad}")
            sol = eq.participation.solution
            record.update({
                "status": "ok", "cost": 0.5 * float(arr @ arr),
                "residuals": dict(zip(self.moment_keys, map(float, arr))),
                "residuals_hex": _hex(arr),
                "ge_converged": bool(eq.converged), "ge_residual": float(eq.residual),
                "markets_converged_share": float(np.mean(np.asarray(sol.converged))),
                "v_min": float(eq.v_min), "y_anchor": float(eq.y_anchor),
                "moments": _checkpoint_moments(model),
            })
            if self.economy.common_alpha:
                record["alpha_common"] = float(prim.alpha_support[0])
            try:
                record.update(anchor_diagnostics(eq))
            except Exception:  # diagnostics only
                pass
        except Exception as exc:
            arr = self.penalty.copy()
            record.update({"status": "error", "cost": 0.5 * float(arr @ arr),
                           "error": f"{type(exc).__name__}: {exc}"})
            if self.logger is not None:
                self.logger.error(f"eval {self.count} failed at {params}: {record['error']}")
        finally:
            if self.watchdog is not None:
                self.watchdog.stop(None)
        record["secs"] = round(time.time() - t0, 2)
        if self.watchdog is not None and record["status"] == "ok":
            self.watchdog.durations.append(record["secs"])
        self.checkpoint.append(_json_safe(record))
        if self.logger is not None:
            shown = " ".join(f"{k}={record[k]:.5g}" for k in self.economy.free)
            self.logger.info(f"eval {self.count:3d} [{tag}] {shown} cost={record['cost']:.4g} "
                             f"({record['secs']:.0f} s)")
        return arr

    def record_timeout(self, context: Mapping[str, Any], limit: float) -> None:
        self.checkpoint.append(_json_safe({**context, "status": "timeout", "limit_s": round(limit, 1)}))


def _json_safe(record: Mapping[str, Any]) -> dict[str, Any]:
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(u) for k, u in v.items()}
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    return clean(dict(record))


# --------------------------------------------------------------------------- calibration

@dataclass(frozen=True)
class RestrictedResult:
    economy: Economy
    params: PooledParams              # the economy's params (xi = inf, rho_bar = 0 where restricted), v_min solved
    primitives: ModelPrimitives       # the economy's primitives (one-node support under common alpha)
    cost: float
    residuals: np.ndarray
    moment_keys: tuple[str, ...]
    model_moments: Mapping[str, float]
    reference: FullPoolReference
    equilibrium: PooledEquilibrium
    success: bool
    status: int
    nfev: int
    n_evals: int
    n_replayed: int
    message: str
    jacobian_x: np.ndarray
    jacobian_condition: float
    at_bound: tuple[str, ...]
    structure: Mapping[str, Any]


def calibrate_restricted(
    evaluator: Evaluator, *, initial: Mapping[str, float], bounds: Mapping[str, tuple[float, float]],
    ftol: float, xtol: float, max_nfev: int, fd_step: float,
) -> RestrictedResult:
    """``trf`` least squares over the economy's free parameters (forward-difference Jacobian).

    Mirrors :func:`.zipf.calibrate_zipf` (same residuals, coordinates, step and
    tolerances) on a reduced vector. The final solve repeats
    :func:`.zipf.calibrate_zipf`'s: the anchored GE from ``v_min = 1``.
    """
    ev = evaluator
    free = ev.economy.free
    lower, upper = coordinate_bounds(free, bounds)
    x0 = np.clip(encode(initial, free, eta=ev.eta, rho_bar_bounds=ev.rho_bar_bounds), lower + 1e-9, upper - 1e-9)
    last: list[tuple[np.ndarray, np.ndarray] | None] = [None]

    def fun(x):
        arr = ev.residual(x)
        last[0] = (np.array(x, dtype=float), arr)
        return arr

    def jac(x):
        x = np.asarray(x, dtype=float)
        f0 = last[0][1] if last[0] is not None and np.array_equal(last[0][0], x) else fun(x)
        out = np.empty((f0.size, x.size))
        for j in range(x.size):
            h = fd_step if x[j] + fd_step <= upper[j] else -fd_step
            xj = x.copy()
            xj[j] += h
            out[:, j] = (ev.residual(xj, tag="jac") - f0) / h
        return out

    sol = least_squares(fun, x0=x0, jac=jac, bounds=(lower, upper), method="trf",
                        ftol=ftol, xtol=xtol, gtol=1e-10, max_nfev=max_nfev, verbose=2)
    if ev.watchdog is not None:   # the final solve is not checkpointed; give it the first-eval limit
        ev.watchdog.durations.clear()
    params = ev.params(sol.x)
    if ev.watchdog is not None:
        ev.watchdog.start({"tag": "final", "x_hex": _hex(sol.x)})
    try:
        model, reference, eq, prim, par = ev.solve(params)
    finally:
        if ev.watchdog is not None:
            ev.watchdog.stop(None)
    par = dataclasses.replace(par, v_min=float(eq.v_min), y_hat=float(eq.y_hat))
    eq = PooledEquilibrium(**{**eq.__dict__, "v_min": par.v_min})
    model = dict(model)
    try:
        diag = anchor_diagnostics(eq)
        for key in ("corr_alpha_log_sales", "corr_alpha_log_sales_partial_z", "corr_alpha_log_sales_partial_v"):
            model[key] = float(diag[key])
    except Exception:
        pass
    residuals = weighted_relative_residuals(model, ev.targets, moment_keys=ev.moment_keys, weights=ev.weights)
    jacobian = np.asarray(sol.jac, dtype=float)
    at_bound = tuple(n for n, xv, lo, hi in zip(free, sol.x, lower, upper)
                     if np.isfinite(lo) and (xv - lo < 1e-6 or hi - xv < 1e-6))
    return RestrictedResult(
        economy=ev.economy, params=par, primitives=prim, cost=0.5 * float(residuals @ residuals),
        residuals=residuals, moment_keys=ev.moment_keys, model_moments=model, reference=reference,
        equilibrium=eq, success=bool(sol.success), status=int(sol.status), nfev=int(sol.nfev),
        n_evals=ev.count, n_replayed=ev.replayed, message=str(sol.message), jacobian_x=jacobian,
        jacobian_condition=float(np.linalg.cond(jacobian)), at_bound=at_bound,
        structure=structure_checks(ev.economy, eq, par, prim, alpha_cap_factor=ev.alpha_cap_factor),
    )


def scan_points(grid: Mapping[str, Sequence[float]], free: Sequence[str],
                fixed_values: Mapping[str, float]) -> list[dict[str, float]]:
    """Cartesian grid over the named axes; free parameters without an axis take ``fixed_values``."""
    axes = [name for name in free if name in grid]
    unknown = set(grid) - set(free)
    if unknown:
        raise ValueError(f"scan axes {sorted(unknown)} are not free parameters of this economy")
    mesh = np.meshgrid(*[np.asarray(grid[a], dtype=float) for a in axes], indexing="ij")
    points = []
    for idx in np.ndindex(mesh[0].shape):
        point = {name: float(fixed_values[name]) for name in free}
        point.update({a: float(m[idx]) for a, m in zip(axes, mesh)})
        points.append(point)
    return points
