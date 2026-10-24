"""Four economies and an additive decomposition on percentage-change scales.

The draw is the one of 02/03/04: ``draw_pool`` with alpha truncated below the
cap, at the sorting reference's M, seed and cap. A fresh baseline and
common-alpha solve must pass the saved reference gates before either new
economy is calculated.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
from ...model.pool import PoolDraw, draw_pool
from ...model.pricing import PricingRegime
from ...model.normalization import solve_pooled_economy, solve_pooled_ge
from ...welfare.welfare_metrics import aggregate_markup_cw, steady_state_welfare
from ..sorting.arrangements import homogenize_alpha_sector


LABELS = ("baseline", "hom_alpha", "hom_v", "hom_both")
TITLES = {
    "hom_both": "Both capability and scalability common",
    "hom_alpha": "Capability heterogeneity only",
    "hom_v": "Scalability heterogeneity only",
    "baseline": "Calibrated Spain",
}
DRAW_POLICY = "draw_pool_truncated_v1"


def reference_draw(support, params, *, M, seed, cap):
    """The sorting sweep's draw: ``draw_pool`` with alpha truncated below ``cap``."""
    return draw_pool(support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
                     M=M, H=params.H, rng=seed, v_min=params.v_min, alpha_max=cap)


def compact_draw(draw, params):
    """Drop inactive padding without changing any active primitive or market."""
    counts = draw.active_mask.sum(axis=1)
    if not np.array_equal(draw.active_mask, np.arange(draw.alpha.shape[1])[None, :] < counts[:, None]):
        raise ValueError("reference mask must have contiguous active slots")
    width = max(int(counts.max()), int(np.ceil(params.N)))
    result = PoolDraw(*(getattr(draw, key)[:, :width].copy()
                        for key in ("alpha", "v", "tilde_alpha", "active_mask")))
    assert result.active_mask.sum() == draw.active_mask.sum()
    return result, dataclasses.replace(params, H=width)


def homogenize_v(draw):
    """Keep sector mean anchor marginal cost: mean(1/v), not mean(v)."""
    v = draw.v.copy()
    for m, mask in enumerate(draw.active_mask):
        if mask.sum() > 1:
            v[m, mask] = 1 / np.mean(1 / draw.v[m, mask])
        if not np.isclose(np.mean(1 / v[m, mask]), np.mean(1 / draw.v[m, mask]), rtol=1e-12, atol=0):
            raise AssertionError("capability collapse changed mean anchor marginal cost")
    return dataclasses.replace(draw, v=v)


def arrangements(draw):
    common_alpha = homogenize_alpha_sector(draw)
    return {
        "baseline": draw,
        "hom_alpha": common_alpha,
        "hom_v": homogenize_v(draw),
        "hom_both": homogenize_v(common_alpha),
    }


def solve_economy(label, draw, params, *, chi, beta, phi, delta_K, n_firms_cs,
                  market_kwargs, initial, wf_reference, outer_tol=1e-7, logger=None,
                  reference_prices=None):
    """Solve only ME and PE, with fixed participation, anchor and baseline chi."""
    started = time.perf_counter()
    row = {"label": label, "chi": float(chi), "n_active": int(draw.active_mask.sum())}
    common = dict(beta=beta, delta_K=delta_K, n_firms_cs=n_firms_cs,
                  market_solver_kwargs=market_kwargs, active_mask=draw.active_mask,
                  wf_reference=wf_reference, tol=outer_tol, max_nfev=80,
                  K_switch=1, love_of_variety=False)
    for key, regime in (("ME", PricingRegime.MARKET), ("PE", PricingRegime.PLANNER)):
        if logger:
            logger.info("[%s] solving %s", label, key)
        t0 = time.perf_counter()
        if reference_prices is None:
            eq = solve_pooled_ge(draw.alpha, draw.v, params, regime=regime,
                                 initial=initial, **common)
        else:
            eq = solve_pooled_economy(
                draw.alpha, draw.v, params, regime=regime,
                w=reference_prices[f"w_{key}"], R=reference_prices[f"R_{key}"],
                X_market=reference_prices[f"X_market_{key}"],
                n_firms_cs=n_firms_cs, delta_K=delta_K,
                active_mask=draw.active_mask, market_solver_kwargs=market_kwargs,
                wf_reference=wf_reference, K_switch=1, love_of_variety=False)
        if not eq.converged or not np.isfinite(eq.C) or eq.C <= 0:
            raise RuntimeError(f"{label}/{key}: failed GE or nonpositive consumption")
        if not np.array_equal(eq.participation.active_mask, draw.active_mask):
            raise AssertionError(f"{label}/{key}: participation changed")
        if not np.isclose(eq.L_agg, 1, rtol=0, atol=1e-5):
            raise AssertionError(f"{label}/{key}: labor is not one")
        agg = eq.aggregates
        if not np.isclose(eq.C, agg.Q - agg.M - delta_K * eq.K_agg, rtol=1e-12):
            raise AssertionError(f"{label}/{key}: resource constraint failed")
        values = {"Q": agg.Q, "K": eq.K_agg, "M": agg.M, "C": eq.C,
                  "L": eq.L_agg, "R": eq.R, "w": eq.w, "X_market": eq.X_market,
                  "W": steady_state_welfare(eq.C, eq.L_agg, chi, beta, phi),
                  "mu_cw": aggregate_markup_cw(eq, n_firms_cs=n_firms_cs),
                  "residual": eq.residual}
        row.update({f"{k}_{key}": float(v) for k, v in values.items()})
        row[f"wall_s_{key}"] = time.perf_counter() - t0
        initial = (eq.w, eq.X_market)
        del eq
    row["lambda_total"] = float(np.expm1((1 - beta) * (row["W_PE"] - row["W_ME"])))
    row["lambda_consumption"] = row["C_PE"] / row["C_ME"] - 1
    row["consumption_welfare_gap"] = abs(row["lambda_total"] - row["lambda_consumption"])
    if row["consumption_welfare_gap"] > 1e-5:
        raise AssertionError(f"{label}: consumption and welfare disagree")
    row["Y_pct"] = 100 * (row["Q_PE"] / row["Q_ME"] - 1)
    row["welfare_pct"] = 100 * row["lambda_total"]
    row["wall_s_total"] = time.perf_counter() - started
    return row


def compare_reference(row, reference, *, rtol=2e-6, atol=1e-6):
    """Bidirectional comparison of all common aggregates, prices and welfare."""
    checks = {}
    keys = [f"{k}_{reg}" for reg in ("ME", "PE")
            for k in ("Q", "K", "M", "C", "L", "R", "w", "X_market", "W", "mu_cw")]
    keys += ["lambda_total", "n_active", "chi"]
    for key in keys:
        a, b = float(row[key]), float(reference[key])
        # Symmetric scale prevents an arbitrary choice of comparison direction.
        error, limit = abs(a - b), atol + rtol * max(abs(a), abs(b))
        checks[key] = {"fresh": a, "saved": b, "absolute_error": error,
                       "tolerance": limit, "passed": bool(error <= limit)}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def decompose(rows):
    """Finite differences of percentage changes, with an exact additive total."""
    result = {}
    for metric in ("Y_pct", "welfare_pct"):
        b, a, v, h = (float(rows[k][metric]) for k in LABELS)
        components = {"common": h, "capability": a - h, "scalability": v - h,
                      "interaction": b - a - v + h, "total": b}
        residual = sum(components[k] for k in ("common", "capability", "scalability", "interaction")) - b
        if abs(residual) > 1e-10:
            raise AssertionError("percentage-point decomposition does not add up")
        result[metric] = {**components, "identity_residual": residual}
    return result


def write_results(out, rows, validation, provenance):
    """Persist the two panels, their units, and the full audit trail."""
    import pandas as pd
    import yaml

    split = decompose(rows)
    pd.DataFrame([rows[k] for k in LABELS]).to_parquet(out / "level_dispersion_economies.parquet", index=False)
    payload = {"provenance": provenance, "validation": validation,
               "economies": rows, "decomposition": split}
    (out / "level_dispersion.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    lines = ["# Capability and scalability: Orbis Spain", "",
             "Panel A: steady-state comparisons (% change), each planner relative to its own market.", "",
             "| Economy | Y | Welfare lambda |", "|---|---:|---:|"]
    for key in ("hom_both", "hom_alpha", "hom_v", "baseline"):
        r = rows[key]
        lines.append(f"| {TITLES[key]} | {r['Y_pct']:.4f} | {r['welfare_pct']:.4f} |")
    lines += ["", "Panel B: contributions to the calibrated gains (percentage points).", "",
              "| Component | Y | Welfare lambda |", "|---|---:|---:|"]
    for key, title in (("common", "Common-technology benchmark"), ("capability", "Capability heterogeneity alone"),
                       ("scalability", "Scalability heterogeneity alone"), ("interaction", "Joint interaction"),
                       ("total", "Calibrated total")):
        lines.append(f"| {title} | {split['Y_pct'][key]:.4f} | {split['welfare_pct'][key]:.4f} |")
    lines += ["", "The interaction is a joint-heterogeneity residual, not the reassignment sorting effect.",
              "Labor and firm sets are fixed; chi and the anchor are frozen at the reference baseline.",
              "Both-common means homogeneous within each sector; sector means may differ.",
              "All source numbers and checks are in level_dispersion.yaml and validation.json.", ""]
    (out / "results.md").write_text("\n".join(lines))
