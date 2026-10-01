"""Pooled-market simulation and rectangular firm panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.normalization import PooledEquilibrium, PooledParams, solve_pooled_ge
from ..model.pool import draw_pool
from ..pooled_inputs import PooledInputs


SIM_PANEL_COLUMNS: tuple[str, ...] = (
    "market_id", "firm_id", "alpha", "v", "tilde_alpha", "active", "n_active",
    "share", "markup", "sales", "cost", "output", "price",
)


@dataclass(frozen=True)
class SimulationResult:
    equilibrium: PooledEquilibrium
    panel: pd.DataFrame


def build_firm_panel(
    equilibrium: PooledEquilibrium,
    *,
    tilde_alpha: np.ndarray | None = None,
    regime: str | None = None,
) -> pd.DataFrame:
    solution = equilibrium.participation.solution
    mask = equilibrium.participation.active_mask
    M, H = mask.shape
    tilde = np.zeros_like(equilibrium.alpha) if tilde_alpha is None else tilde_alpha
    frame = pd.DataFrame({
        "market_id": np.repeat(np.arange(M), H),
        "firm_id": np.tile(np.arange(H), M),
        "alpha": equilibrium.alpha.ravel(), "v": equilibrium.v.ravel(),
        "tilde_alpha": np.asarray(tilde).ravel(), "active": mask.ravel(),
        "n_active": np.repeat(equilibrium.participation.n_active, H),
        "share": solution.s.ravel(), "markup": solution.mu.ravel(),
        "sales": solution.sales.ravel(), "cost": solution.cost.ravel(),
        "output": solution.output.ravel(), "price": solution.price.ravel(),
    })
    if regime is not None:
        frame["regime"] = regime
    return frame


def run_simulation(
    inputs: PooledInputs,
    params: PooledParams,
    *,
    M: int,
    master_seed: int = 20260525,
    beta: float = 0.96,
    delta_K: float = 0.06,
    K_switch: int = 3,
    love_of_variety: bool | str = False,
    market_solver_kwargs: dict | None = None,
    wf_reference: float | None = None,
    initial: tuple[float, float] | None = None,
    out_dir: Path | str | None = None,
    panel_filename: str = "sim_panel.parquet",
    tol: float = 1e-5,
    max_nfev: int = 30,
) -> SimulationResult:
    support = inputs.alpha_support if inputs.alpha_support is not None else np.linspace(0.6, 1.20, 500)
    draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
        M=M, H=params.H, rng=master_seed, v_min=params.v_min,
    )
    eq = solve_pooled_ge(
        draw.alpha, draw.v, params, beta=beta, delta_K=delta_K,
        n_firms_cs=inputs.n_firms_cs, K_switch=K_switch,
        love_of_variety=love_of_variety,
        market_solver_kwargs=market_solver_kwargs, wf_reference=wf_reference,
        active_mask=draw.active_mask, initial=initial,
        tol=tol, max_nfev=max_nfev,
    )
    eq = PooledEquilibrium(**{**eq.__dict__, "tilde_alpha": draw.tilde_alpha})
    panel = build_firm_panel(eq, tilde_alpha=draw.tilde_alpha)
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path / panel_filename, index=False)
    return SimulationResult(eq, panel)
