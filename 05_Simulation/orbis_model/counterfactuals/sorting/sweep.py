"""Solve one alpha arrangement under both welfare lenses, and the sweep plan.

Ported from ``04_Code/counterfactuals/sorting_curve/run_sorting_curve.py``
(``solve_row``, ``add_lenses``, ``validate_baseline``) onto the Orbis welfare
chain. Each arrangement solves five allocations on its own draw and the
baseline's exogenous active set:

    ME, PE_I (K, L, M fixed), PE_K (K, L fixed), PE (L fixed)   via solve_fixed_input_chain
    U (uniform markup mu_bar = the arrangement's ME mu_cw)      via two_channel_decomposition

Welfare uses ``chi`` frozen at the BASELINE market value; ``v_min`` and
``y_hat`` stay at their calibrated values (the frozen anchor). The legs are

    Lens A: realloc (ME -> PE_I), materials (PE_I -> PE_K), capital (PE_K -> PE)
    Lens B: disp (ME -> U), level (U -> PE)
    total (ME -> PE)

with ``Delta = (1 - beta) (W_b - W_a)`` and ``lambda = exp(Delta) - 1``, so the
Delta legs of each lens sum to ``Delta_total`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from ...model.aggregator import pooled_moments
from ...model.normalization import PooledEquilibrium, PooledParams
from ...model.pool import PoolDraw
from ...welfare.fixed_input import solve_fixed_input_chain
from ...welfare.welfare_metrics import aggregate_markup_cw, steady_state_welfare, two_channel_decomposition
from .arrangements import active_corr_av, cost_weighted_alpha, within_sector_rank_corr

REGIMES = ("ME", "PE_I", "PE_K", "PE", "U")
LEGS = ("realloc", "materials", "capital", "total", "disp", "level")
PAIRS = {
    "realloc": ("ME", "PE_I"), "materials": ("PE_I", "PE_K"), "capital": ("PE_K", "PE"),
    "total": ("ME", "PE"), "disp": ("ME", "U"), "level": ("U", "PE"),
}
LENS_A = ("realloc", "materials", "capital")
LENS_B = ("disp", "level")

# Copula grid over rho_bar (the calibrated value is added as the baseline).
COPULA_GRID = (-0.9, -0.6, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)

CONSUMPTION_GAP_TOL = 1e-4   # Delta vs log C ratio; L = 1 holds to solver tolerance
IDENTITY_TOL = 1e-12         # exact log identities of both lenses
LABOR_TOL = 1e-4
TARGET_TOL = 1e-4            # fixed K and M of PE_I / PE_K (converged needs residual <= 10 tol)
VALIDATION_RTOL = 1e-6


@dataclass(frozen=True)
class Arrangement:
    """One point of the sweep. ``mode`` is the permutation / homogenization variant."""

    label: str
    kind: str                 # copula | permutation | homogeneous
    mode: str                 # identity | copula | shuffle | reverse | sorted | sector | cost_weighted
    rho_bar: float | None = None
    replicate: int = 0


def build_plan(
    rho_calibrated: float,
    *,
    grid: tuple[float, ...] = COPULA_GRID,
    n_shuffle: int = 2,
    homogeneous: bool = True,
) -> list[Arrangement]:
    """Baseline first, then the copula grid, the permutation points and homogeneous alpha."""
    plan = [Arrangement("baseline", "copula", "identity", float(rho_calibrated))]
    plan += [Arrangement(f"rho_{r:+.2f}", "copula", "copula", float(r))
             for r in grid if not math.isclose(r, rho_calibrated, abs_tol=1e-9)]
    plan.append(Arrangement("reverse", "permutation", "reverse"))
    plan += [Arrangement(f"shuffle_{k}", "permutation", "shuffle", replicate=k)
             for k in range(n_shuffle)]
    plan.append(Arrangement("sorted", "permutation", "sorted"))
    if homogeneous:
        plan += [Arrangement("hom_sector", "homogeneous", "sector"),
                 Arrangement("hom_cost", "homogeneous", "cost_weighted")]
    labels = [a.label for a in plan]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate arrangement labels: {labels}")
    return plan


def shuffle_seed(master_seed: int, replicate: int) -> list[int]:
    """SeedSequence entropy of a shuffle replicate (stored with each row)."""
    return [int(master_seed), 1, int(replicate)]


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def market_structure(eq: PooledEquilibrium, *, phi_v: float, alpha_max: float | None) -> dict:
    """Concentration and cap diagnostics of the market allocation."""
    sol = eq.participation.solution
    mask = np.asarray(eq.participation.active_mask, dtype=bool)
    m = pooled_moments(sol, mask, phi_v=phi_v, alpha=eq.alpha)
    share = np.where(mask, sol.s, np.nan)
    top = np.nanmax(share, axis=1)
    sales = np.where(mask, sol.sales, 0.0)
    # Zero under the truncated draw (no alpha reaches the cap); nonzero only for clipped legacy draws.
    capped = (mask & (np.asarray(eq.alpha) >= alpha_max)) if alpha_max is not None else np.zeros_like(mask)
    out = {
        "mu_cw_alpha_ME": m["mu_cw_alpha"],
        "cr4_ME": m["cr4"], "cr20_ME": m["cr20"],
        "top1pct_ME": m["top1pct"], "top5pct_ME": m["top5pct"],
        "emx_slope_ME": m["emx_slope"],
        "rts_gap_ME": m.get("alpha_rts_gap_top5_bottom50", float("nan")),
        "top_share_median_ME": float(np.median(top)),
        "top_share_p90_ME": float(np.quantile(top, 0.9)),
        "cap_firm_share_ME": float(capped.sum() / mask.sum()),
        "cap_sales_share_ME": float(sales[capped].sum() / sales.sum()),
    }
    return {k: finite_or_none(v) for k, v in out.items()}


def _regime_values(key: str, eq: PooledEquilibrium, *, chi: float, beta: float, phi: float,
                   n_firms_cs: float) -> dict:
    agg = eq.aggregates
    return {
        f"converged_{key}": bool(eq.converged), f"residual_{key}": float(eq.residual),
        f"L_{key}": float(eq.L_agg), f"C_{key}": float(eq.C),
        f"W_{key}": steady_state_welfare(eq.C, eq.L_agg, chi, beta, phi),
        f"Q_{key}": float(agg.Q), f"M_{key}": float(agg.M), f"I_{key}": float(agg.I),
        f"K_{key}": float(eq.K_agg), f"R_{key}": float(eq.R), f"P_mat_{key}": float(eq.P_mat),
        f"w_{key}": float(eq.w), f"X_market_{key}": float(eq.X_market),
        f"profits_{key}": float(eq.Pi_agg),
        f"mu_cw_{key}": float(aggregate_markup_cw(eq, n_firms_cs=n_firms_cs)),
    }


def add_lenses(row: dict, beta: float) -> dict:
    """Add Delta, lambda and consumption gaps for every leg; assert both lens identities."""
    for leg, (a, b) in PAIRS.items():
        delta = (1.0 - beta) * (row[f"W_{b}"] - row[f"W_{a}"])
        row[f"Delta_{leg}"] = float(delta)
        row[f"lambda_{leg}"] = float(np.expm1(delta))
        gap = float(delta - np.log(row[f"C_{b}"] / row[f"C_{a}"]))
        row[f"consumption_gap_{leg}"] = gap
        if abs(gap) > CONSUMPTION_GAP_TOL:
            raise AssertionError(f"{row.get('label')}: consumption/welfare gap {gap:.3e} on leg {leg}")
    row["identity_A"] = sum(row[f"Delta_{leg}"] for leg in LENS_A) - row["Delta_total"]
    row["identity_B"] = sum(row[f"Delta_{leg}"] for leg in LENS_B) - row["Delta_total"]
    if max(abs(row["identity_A"]), abs(row["identity_B"])) > IDENTITY_TOL:
        raise AssertionError(f"{row.get('label')}: lens identity failed "
                             f"(A {row['identity_A']:.3e}, B {row['identity_B']:.3e})")
    return row


def solve_arrangement(
    label: str,
    draw: PoolDraw,
    params: PooledParams,
    *,
    chi: float | None,
    beta: float,
    phi: float,
    delta_K: float,
    n_firms_cs: float,
    K_switch: int,
    love_of_variety,
    market_solver_kwargs: dict,
    wf_reference: float | None,
    initial: tuple[float, float] | None,
    tol: float,
    max_nfev: int,
    alpha_max: float | None,
    logger=None,
) -> tuple[dict, PooledEquilibrium]:
    """Solve ME, PE_I, PE_K, PE and U; return ``(row, me)`` with every check enforced.

    ``chi = None`` (the baseline) freezes ``chi`` at this arrangement's own ME value.
    """
    say = logger.info if logger is not None else (lambda msg: None)
    t0 = time.perf_counter()
    say(f"[{label}] corr(alpha, v) = {active_corr_av(draw):+.6f}; solving ME, PE_K, PE_I, PE")
    common = dict(
        beta=beta, delta_K=delta_K, n_firms_cs=n_firms_cs, active_mask=draw.active_mask,
        K_switch=K_switch, love_of_variety=love_of_variety,
        market_solver_kwargs=market_solver_kwargs, wf_reference=wf_reference,
        tol=tol, max_nfev=max_nfev,
    )
    chain = solve_fixed_input_chain(draw.alpha, draw.v, params, initial=initial, **common)
    if not chain.me.converged and initial is not None:
        say(f"[{label}] ME did not converge from the warm start; retrying from (1, 1)")
        chain = solve_fixed_input_chain(draw.alpha, draw.v, params, initial=None, **common)
    me = chain.me
    if not me.converged:
        raise RuntimeError(f"{label}: market did not converge")
    chi_used = float(me.chi) if chi is None else float(chi)
    say(f"[{label}] solving U")
    t_u = time.perf_counter()
    decomp = two_channel_decomposition(
        me, chain.pe, params=params, chi_me=chi_used, beta=beta, phi=phi, delta_K=delta_K,
        n_firms_cs=n_firms_cs, K_switch=K_switch, love_of_variety=love_of_variety,
        market_solver_kwargs=market_solver_kwargs,
    )
    wall_u = time.perf_counter() - t_u
    regimes = {"ME": me, "PE_I": chain.pe_i, "PE_K": chain.pe_k, "PE": chain.pe, "U": decomp["uniform"]}

    row: dict = {
        "label": label,
        "corr_av": finite_or_none(active_corr_av(draw)),
        "rank_corr_within": finite_or_none(within_sector_rank_corr(draw)),
        "chi": chi_used, "chi_own_ME": float(me.chi),
        "mu_bar": float(decomp["mu_bar"]), "v_min": float(params.v_min), "y_hat": float(params.y_hat),
        "n_active": int(np.asarray(draw.active_mask).sum()),
        "alpha_hom_cost_weighted": cost_weighted_alpha(draw.alpha, me.participation.solution.cost,
                                                        me.participation.active_mask),
    }
    for key, eq in regimes.items():
        if not eq.converged:
            raise RuntimeError(f"{label}: {key} did not converge (residual {eq.residual:.3e})")
        if not np.array_equal(eq.participation.active_mask, draw.active_mask):
            raise AssertionError(f"{label}: {key} active_mask drift")
        if eq.participation.wf_reference != me.participation.wf_reference:
            raise AssertionError(f"{label}: {key} wf_reference drift")
        if not np.isclose(eq.L_agg, 1.0, atol=LABOR_TOL, rtol=0.0):
            raise AssertionError(f"{label}: {key} L = {eq.L_agg} is not 1 within {LABOR_TOL}")
        row.update(_regime_values(key, eq, chi=chi_used, beta=beta, phi=phi, n_firms_cs=n_firms_cs))

    row["K_gap_PE_I"] = row["K_PE_I"] / row["K_ME"] - 1.0
    row["M_gap_PE_I"] = row["M_PE_I"] / row["M_ME"] - 1.0
    row["K_gap_PE_K"] = row["K_PE_K"] / row["K_ME"] - 1.0
    if max(abs(row["K_gap_PE_I"]), abs(row["M_gap_PE_I"]), abs(row["K_gap_PE_K"])) > TARGET_TOL:
        raise AssertionError(f"{label}: fixed-input targets missed")
    if not np.isclose(row["mu_cw_U"], row["mu_bar"], rtol=1e-7, atol=0.0):
        raise AssertionError(f"{label}: uniform markup {row['mu_cw_U']} != mu_bar {row['mu_bar']}")
    row["K_PE_over_K_ME"] = row["K_PE"] / row["K_ME"]
    row.update(market_structure(me, phi_v=params.phi_v, alpha_max=alpha_max))
    row.update({f"wall_s_{key}": round(float(sec), 2) for key, sec in
                (("ME", chain.wall_s["market"]), ("PE_K", chain.wall_s["planner_fixed_k"]),
                 ("PE_I", chain.wall_s["planner_fixed_input"]), ("PE", chain.wall_s["planner"]),
                 ("U", wall_u))})
    row["wall_s_total"] = round(time.perf_counter() - t0, 2)
    return add_lenses(row, beta), me


# ---------------------------------------------------------------------------
# Baseline validation gate
# ---------------------------------------------------------------------------

FIXED_INPUT_KEYS = {
    "W_ME": "W_market", "W_PE_I": "W_planner_fixed_input", "W_PE_K": "W_planner_fixed_k",
    "W_PE": "W_planner", "lambda_realloc": "lambda_realloc", "lambda_materials": "lambda_materials",
    "lambda_capital": "lambda_capital", "lambda_total": "lambda_total",
    "K_ME": "K_target", "M_ME": "M_target", "R_PE_I": "R_planner_fixed_input",
    "R_PE_K": "R_planner_fixed_k", "P_mat_PE_I": "P_mat_planner_fixed_input",
}
FIXED_INPUT_REGIMES = {"ME": "market", "PE_I": "planner_fixed_input",
                       "PE_K": "planner_fixed_k", "PE": "planner"}
WELFARE_KEYS = {
    "W_ME": "W_market", "W_U": "W_uniform", "W_PE": "W_planner",
    "lambda_total": "lambda_total", "lambda_level": "lambda_level",
    "lambda_disp": "lambda_dispersion", "mu_bar": "mu_bar",
}
WELFARE_REGIMES = {"ME": "market", "U": "uniform", "PE": "planner"}


@dataclass(frozen=True)
class ReferencePaths:
    """The 02_welfare and 03_fixed_input_welfare outputs the baseline must reproduce."""

    welfare_yaml: Path
    welfare_table: Path
    welfare_manifest: Path
    fixed_input_yaml: Path
    fixed_input_table: Path
    fixed_input_manifest: Path

    @classmethod
    def from_dirs(cls, welfare_dir: Path, fixed_input_dir: Path, tag: str) -> "ReferencePaths":
        welfare_dir, fixed_input_dir = Path(welfare_dir), Path(fixed_input_dir)
        return cls(
            welfare_yaml=welfare_dir / f"welfare_{tag}.yaml",
            welfare_table=welfare_dir / f"welfare_decomposition_{tag}.parquet",
            welfare_manifest=welfare_dir / f"welfare_manifest_{tag}.yaml",
            fixed_input_yaml=fixed_input_dir / f"fixed_input_welfare_{tag}.yaml",
            fixed_input_table=fixed_input_dir / f"fixed_input_regimes_{tag}.parquet",
            fixed_input_manifest=fixed_input_dir / f"fixed_input_manifest_{tag}.yaml",
        )


def check_reference_manifests(refs: ReferencePaths, *, M: int, seed: int, params_sha256: str) -> None:
    """Fail before any solve unless both references exist and match M, seed and the params file."""
    for path in (refs.welfare_yaml, refs.welfare_table, refs.welfare_manifest,
                 refs.fixed_input_yaml, refs.fixed_input_table, refs.fixed_input_manifest):
        if not path.exists():
            raise FileNotFoundError(
                f"baseline reference {path} is missing; run 02_welfare.py and "
                f"03_fixed_input_welfare.py at M={M}, seed={seed} first")
    for name, path in (("02_welfare", refs.welfare_manifest), ("03_fixed_input", refs.fixed_input_manifest)):
        manifest = yaml.safe_load(path.read_text())
        got = (int(manifest["M"]), int(manifest["seed"]), manifest["params_file"]["sha256"])
        if got != (M, seed, params_sha256):
            raise RuntimeError(
                f"{name} reference {path} has (M, seed, params sha256) = {got}; the sweep needs "
                f"({M}, {seed}, {params_sha256}). Rerun it on the current params file.")


def validate_baseline(row: dict, refs: ReferencePaths, out_dir: Path) -> dict:
    """Compare the baseline row with both references at relative tolerance 1e-6.

    Writes ``validation.json`` and raises on failure, before any new arrangement runs.
    """
    fixed = yaml.safe_load(refs.fixed_input_yaml.read_text())
    welfare = yaml.safe_load(refs.welfare_yaml.read_text())
    fixed_c = pd.read_parquet(refs.fixed_input_table).set_index("regime")["C"]
    welfare_c = pd.read_parquet(refs.welfare_table).set_index("regime")["C"]
    pairs = {f"fixed_input:{ref}->{key}": (row[key], fixed[ref]) for key, ref in FIXED_INPUT_KEYS.items()}
    pairs.update({f"fixed_input:C[{ref}]->C_{key}": (row[f"C_{key}"], fixed_c[ref])
                  for key, ref in FIXED_INPUT_REGIMES.items()})
    pairs.update({f"welfare:{ref}->{key}": (row[key], welfare[ref]) for key, ref in WELFARE_KEYS.items()})
    pairs.update({f"welfare:C[{ref}]->C_{key}": (row[f"C_{key}"], welfare_c[ref])
                  for key, ref in WELFARE_REGIMES.items()})
    diffs = {name: abs(float(a) - float(b)) / max(abs(float(b)), 1e-30) for name, (a, b) in pairs.items()}
    payload = {
        "relative_tolerance": VALIDATION_RTOL,
        "references": {k: str(v) for k, v in refs.__dict__.items()},
        "relative_differences": diffs,
        "max_relative_difference": max(diffs.values()),
        "passed": all(d <= VALIDATION_RTOL for d in diffs.values()),
    }
    atomic_json(Path(out_dir) / "validation.json", payload)
    if not payload["passed"]:
        worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
        raise RuntimeError(f"baseline validation failed (worst {worst}); stopped before new "
                           f"arrangements. See {Path(out_dir) / 'validation.json'}")
    return payload


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
