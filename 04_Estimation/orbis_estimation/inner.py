"""Single pooled ``trf`` calibration of ``(xi, N, gamma, eta, rho_bar)``.

Ported from ``04_Code/steady_state/calibration/inner.py``. Changes: the targets
and model primitives come as separate objects (``targets.py`` and
``orbis_model.primitives``); the targeted moments and weights are passed by the
caller (``config_estimation.yaml``); the ``corr_alpha_log_sales`` residual is
gone (the RTS gap moment is the alpha-sorting target and enters through
``moment_keys``); ``rho_bar`` may be negative (its logistic transform maps
onto any bounds inside (-1, 1)); the initial vector is required (no bundle
warm start).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from orbis_model.model.normalization import FullPoolReference, PooledEquilibrium, PooledParams
from orbis_model.model.pricing import DEFAULT_PHI_V
from orbis_model.primitives import ModelPrimitives
from .anchor_diagnostics import anchor_diagnostics
from .objective import MOMENT_KEYS, pooled_moments_at_params, weighted_relative_residuals


DEFAULT_RHO_BAR_BOUNDS = (-0.95, 0.95)


@dataclass(frozen=True)
class PooledCalibrationResult:
    params: PooledParams
    cost: float
    residuals: np.ndarray
    moment_keys: tuple[str, ...]
    model_moments: Mapping[str, float]
    target_moments: Mapping[str, float]
    reference: FullPoolReference
    equilibrium: PooledEquilibrium
    success: bool
    nfev: int
    message: str
    jacobian: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    jacobian_rank: int
    jacobian_condition: float
    eta_jacobian_column_norm: float

    @property
    def xi(self) -> float:
        return self.params.xi

    @property
    def N(self) -> float:
        return self.params.N

    @property
    def gamma(self) -> float:
        return self.params.gamma

    @property
    def eta(self) -> float:
        return self.params.eta

    @property
    def v_min(self) -> float:
        return self.params.v_min

    @property
    def rho_bar(self) -> float:
        return self.params.rho_bar


def _rho_to_x(rho: float, bounds: tuple[float, float]) -> float:
    """Logit of ``rho`` rescaled from ``[lo, hi]`` to (0, 1); valid for negative bounds."""
    lo, hi = map(float, bounds)
    scaled = (float(rho) - lo) / (hi - lo)
    scaled = np.clip(scaled, 1e-12, 1.0 - 1e-12)
    return float(np.log(scaled / (1.0 - scaled)))


def _x_to_rho(x: float, bounds: tuple[float, float]) -> float:
    """Inverse of :func:`_rho_to_x`: logistic of ``x`` rescaled into ``[lo, hi]``."""
    lo, hi = map(float, bounds)
    scaled = 1.0 / (1.0 + np.exp(-float(x)))
    return float(lo + (hi - lo) * scaled)


def _decode(
    x: np.ndarray, *, a: float, H: int, phi_v: float = DEFAULT_PHI_V,
    rho_bar_bounds: tuple[float, float] = DEFAULT_RHO_BAR_BOUNDS,
) -> PooledParams:
    xi = float(np.exp(x[0]))
    N = float(np.exp(x[1]))
    eta = float(1.0 + np.exp(x[2]))
    gamma = float(eta + np.exp(x[3]))
    rho_bar = _x_to_rho(float(x[4]), rho_bar_bounds)
    return PooledParams(
        xi=xi, N=N, gamma=gamma, eta=eta, rho_bar=rho_bar, a=a, H=H,
        phi_v=phi_v,
    )


def _anchor_checkpoint_diag(eq: PooledEquilibrium) -> dict[str, float]:
    try:
        return anchor_diagnostics(eq)
    except Exception:
        return {
            "corr_alpha_log_sales": float("nan"),
            "corr_alpha_log_sales_partial_z": float("nan"),
            "corr_alpha_log_sales_partial_v": float("nan"),
            "median_log_y_over_yhat": float("nan"),
            "sales_weighted_sd_log_y_over_yhat": float("nan"),
            "y_sw": float("nan"),
            "y_anchor": float("nan"),
        }


def _encode(
    values: Sequence[float],
    *,
    rho_bar_bounds: tuple[float, float] = DEFAULT_RHO_BAR_BOUNDS,
) -> np.ndarray:
    xi, N, gamma, eta, rho_bar = map(float, values)
    if not 1.0 < eta < gamma:
        raise ValueError("initial elasticities must satisfy 1 < eta < gamma")
    if not xi > 1.0:
        raise ValueError("initial Pareto tail index xi must exceed 1")
    if N < 1.0:
        raise ValueError("initial Poisson mean N must be at least 1")
    lo, hi = rho_bar_bounds
    if not lo <= rho_bar < hi:
        raise ValueError("initial rho_bar outside bounds")
    return np.array([
        np.log(xi),
        np.log(N),
        np.log(eta - 1.0),
        np.log(gamma - eta),
        _rho_to_x(rho_bar, rho_bar_bounds),
    ])


def calibrate_pooled(
    primitives: ModelPrimitives,
    targets: Mapping[str, float],
    *,
    M: int,
    rng=0,
    initial: Sequence[float],
    xi_bounds: tuple[float, float] = (1.05, 30.0),
    N_bounds: tuple[float, float] = (5.0, 1000.0),
    eta_bounds: tuple[float, float] = (1.01, 10.0),
    gamma_gap_bounds: tuple[float, float] = (0.01, 30.0),
    rho_bar_bounds: tuple[float, float] = DEFAULT_RHO_BAR_BOUNDS,
    moment_keys: Sequence[str] = MOMENT_KEYS,
    weights: Mapping[str, float] | None = None,
    beta: float = 0.96,
    delta_K: float = 0.06,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    max_nfev: int = 40,
    checkpoint_path: str | Path | None = None,
) -> PooledCalibrationResult:
    """Fit the anchored five-parameter vector jointly.

    ``initial`` is ``(xi, N, gamma, eta, rho_bar)``. Residuals are
    :func:`objective.weighted_relative_residuals` over ``moment_keys``.
    """
    x0 = _encode(initial, rho_bar_bounds=rho_bar_bounds)
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    if ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("")  # truncate any stale checkpoint from a prior run
    _eval_count = [0]
    lower = np.array([
        np.log(xi_bounds[0]),
        np.log(N_bounds[0]),
        np.log(eta_bounds[0] - 1.0),
        np.log(gamma_gap_bounds[0]),
        -np.inf,
    ])
    upper = np.array([
        np.log(xi_bounds[1]),
        np.log(N_bounds[1]),
        np.log(eta_bounds[1] - 1.0),
        np.log(gamma_gap_bounds[1]),
        np.inf,
    ])
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)

    def residual(x: np.ndarray) -> np.ndarray:
        params = _decode(
            x, a=primitives.a, H=primitives.H, phi_v=primitives.phi_v,
            rho_bar_bounds=rho_bar_bounds,
        )
        model, _, eq = pooled_moments_at_params(
            params, primitives, M=M, rng=rng, beta=beta, delta_K=delta_K,
            K_switch=K_switch, love_of_variety=love_of_variety,
            market_solver_kwargs=market_solver_kwargs,
        )
        # EMX (2022) normalizer 1 + |target|; see weighted_relative_residuals.
        arr = weighted_relative_residuals(
            model, targets, moment_keys=moment_keys, weights=weights,
        )
        if ckpt is not None:
            _eval_count[0] += 1
            record = {
                "eval": _eval_count[0],
                "v_min": eq.v_min,
                "y_sw": eq.y_sw,
                "y_anchor": eq.y_anchor,
                "xi": params.xi, "N": params.N,
                "gamma": params.gamma, "eta": params.eta,
                "rho_bar": params.rho_bar,
                "cost": 0.5 * float(arr @ arr),
            }
            record.update(_anchor_checkpoint_diag(eq))
            with ckpt.open("a") as fh:  # append + flush-on-close so a kill keeps it
                fh.write(json.dumps(record) + "\n")
        return arr

    # verbose=2 streams the per-iteration cost/step to stdout so a long
    # Marvin2 run is observable via `tail -f` (paired with unbuffered output in
    # cluster/submit_calibrate.sh). SciPy otherwise runs the fit silently.
    sol = least_squares(
        residual, x0=x0, bounds=(lower, upper), method="trf",
        ftol=ftol, xtol=xtol, gtol=1e-8, max_nfev=max_nfev,
        diff_step=1e-3, verbose=2,
    )
    params = _decode(
        sol.x, a=primitives.a, H=primitives.H, phi_v=primitives.phi_v,
        rho_bar_bounds=rho_bar_bounds,
    )
    model, reference, equilibrium = pooled_moments_at_params(
        params, primitives, M=M, rng=rng, beta=beta, delta_K=delta_K,
        K_switch=K_switch, love_of_variety=love_of_variety,
        market_solver_kwargs=market_solver_kwargs,
    )
    params = PooledParams(
        xi=params.xi, N=params.N, gamma=params.gamma, eta=params.eta,
        v_min=equilibrium.v_min, y_hat=equilibrium.y_hat, rho_bar=params.rho_bar,
        a=params.a, H=params.H, phi_v=params.phi_v,
    )
    equilibrium = PooledEquilibrium(**{**equilibrium.__dict__, "v_min": params.v_min})
    # Untargeted alpha-sales correlations (cheap; computed on the final
    # equilibrium only). Diagnostics, not residuals: no data counterpart.
    diag = _anchor_checkpoint_diag(equilibrium)
    model = dict(model)
    model["corr_alpha_log_sales"] = float(diag["corr_alpha_log_sales"])
    model["corr_alpha_log_sales_partial_z"] = float(diag["corr_alpha_log_sales_partial_z"])
    model["corr_alpha_log_sales_partial_v"] = float(diag["corr_alpha_log_sales_partial_v"])
    target_moments = dict(targets)
    jac = np.asarray(sol.jac, dtype=float)
    rank = int(np.linalg.matrix_rank(jac))
    condition = float(np.linalg.cond(jac)) if rank else float("inf")
    dof = max(jac.shape[0] - jac.shape[1], 1)
    sigma2 = float(2.0 * sol.cost / dof)
    covariance_x = sigma2 * np.linalg.pinv(jac.T @ jac)
    # Delta method from transformed coordinates to (xi, N, gamma, eta, rho_bar).
    eta_gap = params.eta - 1.0
    gamma_gap = params.gamma - params.eta
    rho_lo, rho_hi = rho_bar_bounds
    rho_scaled = (params.rho_bar - rho_lo) / (rho_hi - rho_lo)
    drho_dx = (rho_hi - rho_lo) * rho_scaled * (1.0 - rho_scaled)
    transform = np.array([
        [params.xi, 0.0, 0.0, 0.0, 0.0],
        [0.0, params.N, 0.0, 0.0, 0.0],
        [0.0, 0.0, eta_gap, gamma_gap, 0.0],
        [0.0, 0.0, eta_gap, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, drho_dx],
    ])
    covariance = transform @ covariance_x @ transform.T
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    jac_structural = jac @ np.linalg.pinv(transform)
    return PooledCalibrationResult(
        params=params, cost=float(sol.cost), residuals=np.asarray(sol.fun),
        moment_keys=tuple(moment_keys), model_moments=model,
        target_moments=target_moments, reference=reference,
        equilibrium=equilibrium, success=bool(sol.success), nfev=int(sol.nfev),
        message=str(sol.message), jacobian=jac_structural, covariance=covariance,
        standard_errors=standard_errors, jacobian_rank=rank,
        jacobian_condition=condition,
        eta_jacobian_column_norm=float(np.linalg.norm(jac_structural[:, 3])),
    )
