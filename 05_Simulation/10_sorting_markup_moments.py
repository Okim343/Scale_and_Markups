"""Step 10: firm-markup moments of the market allocation across the sorting arrangements.

The sorting sweep (``04_sorting_sweep.py``) stores aggregates only. This step
re-solves ONLY the market equilibrium of every arrangement, on exactly the
draws, seed, active mask, warm start and chi of the sweep, and records how
markups are spread and where they land:

    var_sw_log_mu, var_cw_log_mu     sales- and cost-weighted Var(ln mu)
    corr_cw_logmu_alpha, corr_sw_...  weighted corr(ln mu, alpha) over active firms
    harberger_*                      0.5 * sum_i c_i eps(alpha_i) (ln mu_i - ln mu_bar)^2,
                                     split into spread (mean eps x Var) and placement
                                     (cov_c(eps, squared deviation)), with
                                     eps(alpha) = gamma alpha / (alpha + gamma (1 - alpha))

Each re-solved market is checked against the sweep row (mu_cw_ME and C_ME to
relative tolerance 1e-6) before its moments are written, so the moments belong
to the economies of Table 3 and Figure 4.

Reads   <paths.params_file>, <paths.bundle_dir>, <sweep-dir>/sorting_economies.parquet
Writes  <out-dir>/markup_moments_<label>.json   one per arrangement (atomic, resumable)
        <out-dir>/sorting_markup_moments.parquet

    python 10_sorting_markup_moments.py --config FILE --M N --sweep-dir D [--out-dir D] [--labels a,b]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from orbis_model.config import load_config, setup_logger
from orbis_model.counterfactuals.sorting import (
    COPULA_GRID,
    assert_copula_crn,
    build_plan,
    homogenize_alpha_cost_weighted,
    homogenize_alpha_sector,
    permute_alpha,
)
from orbis_model.counterfactuals.sorting.sweep import atomic_json, shuffle_seed
from orbis_model.model.normalization import solve_pooled_ge
from orbis_model.model.pool import alpha_cap, draw_pool
from orbis_model.model.pricing import PricingRegime
from orbis_model.params import load_simulation_inputs
from orbis_model.welfare.welfare_metrics import aggregate_markup_cw

VALIDATION_RTOL = 1e-6


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None)
    p.add_argument("--params", default=None, help="override paths.params_file")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sweep-dir", required=True, help="04_sorting_sweep.py output to reproduce")
    p.add_argument("--out-dir", default=None, help="default: <sweep-dir>/markup_moments")
    p.add_argument("--labels", default=None, help="comma-separated subset of the plan")
    return p


def weighted_corr(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    mx, my = w @ x, w @ y
    cov = w @ ((x - mx) * (y - my))
    return float(cov / np.sqrt((w @ (x - mx) ** 2) * (w @ (y - my) ** 2)))


def markup_moments(me, alpha: np.ndarray, gamma: float) -> dict:
    """Spread and placement of markups over active firms of the market allocation."""
    sol = me.participation.solution
    mask = np.asarray(me.participation.active_mask, dtype=bool)
    S, C, MU = (np.asarray(x, dtype=float)[mask] for x in (sol.sales, sol.cost, sol.mu))
    A = np.asarray(alpha, dtype=float)[mask]
    s, c, lm = S / S.sum(), C / C.sum(), np.log(MU)
    dev2 = (lm - c @ lm) ** 2
    var_cw = float(c @ dev2)
    eps = gamma * A / (A + gamma * (1.0 - A))
    harberger = 0.5 * float(c @ (eps * dev2))
    spread = 0.5 * float(c @ eps) * var_cw
    return {
        "var_sw_log_mu": float(s @ (lm - s @ lm) ** 2),
        "var_cw_log_mu": var_cw,
        "corr_cw_logmu_alpha": weighted_corr(c, lm, A),
        "corr_sw_logmu_alpha": weighted_corr(s, lm, A),
        "mean_cw_eps": float(c @ eps),
        "harberger_total": harberger,
        "harberger_spread": spread,
        "harberger_placement": harberger - spread,
        "n_active": int(mask.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logger("sorting_markups", cfg.log_level)
    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, logger)
    M = int(args.M)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    sweep_dir = Path(args.sweep_dir).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else sweep_dir / "markup_moments"
    out.mkdir(parents=True, exist_ok=True)
    sweep = pd.read_parquet(sweep_dir / "sorting_economies.parquet").set_index("label")
    if int(sweep["M"].iloc[0]) != M or int(sweep["draw_seed"].iloc[0]) != seed:
        raise RuntimeError(f"sweep was solved at M={sweep['M'].iloc[0]}, seed={sweep['draw_seed'].iloc[0]}")

    P = cfg.parameters
    cap = alpha_cap(params.gamma, P.alpha_cap_factor)

    def make_draw(rho: float):
        return draw_pool(primitives.alpha_support, xi=params.xi, rho_bar=rho, N=params.N,
                         M=M, H=params.H, rng=seed, v_min=params.v_min, alpha_max=cap)

    # every copula point the sweep solved, including ones added beyond COPULA_GRID
    grid = tuple(sorted(set(COPULA_GRID) | {float(s[4:]) for s in sweep.index if s.startswith("rho_")}))
    plan = build_plan(params.rho_bar, grid=grid, n_shuffle=2, homogeneous=True)
    plan = [a for a in plan if a.label in sweep.index]
    if args.labels:
        keep = {s.strip() for s in args.labels.split(",")}
        plan = [a for a in plan if a.label in keep]

    common = dict(
        beta=P.beta, delta_K=P.delta_K, n_firms_cs=primitives.n_firms_cs,
        K_switch=cfg.participation.K_switch, love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        regime=PricingRegime.MARKET,
    )
    base = make_draw(params.rho_bar)
    rows = []
    for arr in plan:
        path = out / f"markup_moments_{arr.label}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            logger.info(f"[{arr.label}] loaded")
            continue
        if arr.mode == "identity":
            draw = base
        elif arr.mode == "copula":
            draw = make_draw(arr.rho_bar)
            assert_copula_crn(draw, base)
        elif arr.mode == "shuffle":
            draw = permute_alpha(base, "shuffle", np.random.default_rng(shuffle_seed(seed, arr.replicate)))
        elif arr.mode in ("reverse", "sorted"):
            draw = permute_alpha(base, arr.mode)
        elif arr.mode == "sector":
            draw = homogenize_alpha_sector(base)
        elif arr.mode == "cost_weighted":
            draw = homogenize_alpha_cost_weighted(base, float(sweep.loc["baseline", "alpha_hom_cost_weighted"]))
        else:
            raise ValueError(f"unknown mode {arr.mode!r}")
        t0 = time.perf_counter()
        me = solve_pooled_ge(draw.alpha, draw.v, params, active_mask=draw.active_mask,
                             initial=initial, **common)
        if not me.converged and initial is not None:
            me = solve_pooled_ge(draw.alpha, draw.v, params, active_mask=draw.active_mask,
                                 initial=None, **common)
        if not me.converged:
            raise RuntimeError(f"{arr.label}: market did not converge")
        ref = sweep.loc[arr.label]
        mu_cw = float(aggregate_markup_cw(me, n_firms_cs=primitives.n_firms_cs))
        gaps = {"mu_cw_ME": abs(mu_cw / ref["mu_cw_ME"] - 1.0), "C_ME": abs(float(me.C) / ref["C_ME"] - 1.0)}
        if max(gaps.values()) > VALIDATION_RTOL:
            raise RuntimeError(f"{arr.label}: re-solved market differs from the sweep {gaps}")
        row = {"label": arr.label, "kind": arr.kind, "corr_av": float(ref["corr_av"]),
               "mu_cw_ME": mu_cw, "Delta_disp": float(ref["Delta_disp"]),
               "Delta_realloc": float(ref["Delta_realloc"]), "Delta_total": float(ref["Delta_total"]),
               "validation_max_rel_gap": max(gaps.values()),
               **markup_moments(me, draw.alpha, params.gamma),
               "wall_s": round(time.perf_counter() - t0, 1), "M": M, "seed": seed}
        # a common alpha (hom_cost_weighted) leaves the correlations undefined
        row = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()}
        atomic_json(path, row)
        rows.append(row)
        logger.info(f"[{arr.label}] corr={row['corr_av']} var_cw={row['var_cw_log_mu']:.5f} "
                    f"corr_cw(lnmu,alpha)={row['corr_cw_logmu_alpha']} wall={row['wall_s']}s")
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "sorting_markup_moments.parquet", index=False)
    logger.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
