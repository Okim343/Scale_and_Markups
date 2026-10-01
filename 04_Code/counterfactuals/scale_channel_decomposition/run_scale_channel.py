"""CLI entry point for the scale-channel decomposition counterfactual.

Copy-adapt of
:func:`counterfactuals.scalability_sorting.run_scalability_sorting.main`. It
rebuilds the *same* calibrated base :class:`~steady_state.model.pool.PoolDraw`
as the prior ``scalability_sorting`` run (from that run's stored ``M``/``seed``),
loops over the same 14 alpha-arrangements (baseline / shuffle x10 / reverse /
hom-primitive / hom-cost-weighted), and for each one re-solves the MARKET leg
(for the warm point + drift guard) and solves the **full free-capital PLANNER**
leg -- the one the sorting/homogeneous drivers deliberately skip (``pe=None``).

The reallocation leg ``lambda_K`` and its ``W_market`` / ``W_planner_fixed_k``
welfare levels are **read from the cached parquets** (design (a),
``reuse_cached_lambda_k``); combined with the new full-planner welfare
``W_planner`` this yields ``lambda_total`` and, by the multiplicative Lens-A
identity, ``lambda_scale``. Because ``lambda_total`` is anchored on the *cached*
``W_market``, the log identity ``Delta_scale = Delta_total - Delta_K`` closes to
machine precision and the two-run CRN-drift surface is isolated into
``consistency_check.yaml`` (same design as ``homogeneous_rts``).

Everything under ``steady_state/`` and the ``fixed_input_welfare`` /
``scalability_sorting`` / ``homogeneous_rts`` modules are imported unchanged;
this package only adds the ``Delta``-algebra helpers in ``decomposition.py``.

Run from ``04_Code/``, after ``scalability_sorting`` and ``homogeneous_rts``
runs have produced their outputs::

    python -m counterfactuals.scale_channel_decomposition.run_scale_channel \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml

Smoke run (no matching low-M cache exists; falls back to an in-process fixed-K
baseline -- see ``--M-smoke``)::

    python -m counterfactuals.scale_channel_decomposition.run_scale_channel \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml --M-smoke 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from steady_state.__main__ import (
    _params_for_command,
    _pooled_inputs,
    _write_yaml,
    load_config,
)
from steady_state.model.normalization import euler_R, solve_pooled_ge
from steady_state.model.pool import draw_pool
from steady_state.model.pricing import PricingRegime
from steady_state.welfare.welfare_metrics import (
    aggregate_markup_cw,
    consumption_equivalent_lambda,
    steady_state_welfare,
)

from counterfactuals.fixed_input_welfare.decomposition import (
    fixed_capital_decomposition,
)
from counterfactuals.fixed_input_welfare.fixed_capital_planner import (
    solve_fixed_capital_planner,
)
from counterfactuals.homogeneous_rts.homogenize import (
    homogenize_alpha_cost_weighted,
    homogenize_alpha_primitive,
)
from counterfactuals.scalability_sorting.permute import (
    active_corr_av,
    permute_alpha,
)

from .decomposition import (
    identity_residual,
    per_draw_scale_delta,
    three_channel_split,
)

_HOM_VARIANT_LABEL = {"primitive": "hom_primitive",
                      "cost_weighted": "hom_cost_weighted"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.scale_channel_decomposition.run_scale_channel",
        description="Scale-channel decomposition: run the full free-capital "
                    "planner under each alpha-arrangement to put the scale leg "
                    "of the Lens-A welfare gain on the arrangement axis, "
                    "companion to scalability_sorting / homogeneous_rts.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--calibration",
        default="out_results/calib_pooled/calibration_pooled.yaml",
        help="calibration YAML (must match the scalability_sorting run's).",
    )
    parser.add_argument(
        "--sorting-dir", default=None,
        help="directory holding sorting_objects.yaml / sorting_economies.parquet "
             "(default out_results/counterfactuals/scalability_sorting).",
    )
    parser.add_argument(
        "--homogeneous-dir", default=None,
        help="directory holding homogeneous_economies.parquet "
             "(default out_results/counterfactuals/homogeneous_rts).",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--M-smoke", type=int, default=None, dest="M_smoke",
        help="override M for a smoke run. When it differs from the cached "
             "sorting run's M, the reallocation (fixed-K) leg cannot be read "
             "from cache, so it is solved IN-PROCESS for every arrangement "
             "(a self-contained single-process run at low M).",
    )
    parser.add_argument(
        "--lens-b", action="store_true", dest="lens_b",
        help="INERT stub this release: reserved for the Lens-B level-channel "
             "per-arrangement decomposition; no UNIFORM-regime solves are run.",
    )
    return parser


# --- solver legs (copy-adapt of the sorting / homogeneous templates) ---------

def _solve_market(draw, params, cfg, inputs, wf, initial):
    return solve_pooled_ge(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        regime=PricingRegime.MARKET, active_mask=draw.active_mask, initial=initial,
    )


def _solve_fixed_k(draw, params, cfg, inputs, me):
    """Fixed-K PLANNER leg -- used only by the in-process smoke fallback.

    Production reads ``lambda_K`` / ``W_planner_fixed_k`` from cache instead.
    """
    return solve_fixed_capital_planner(
        draw.alpha, draw.v, params, K_target=me.K_agg,
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        n_firms_cs=inputs.n_firms_cs, K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        regime=PricingRegime.PLANNER, market_solver_kwargs=cfg.market.to_kwargs(),
        wf_reference=me.participation.wf_reference,
        active_mask=me.participation.active_mask,
        initial=(me.w, me.X_market,
                 euler_R(cfg.parameters.beta, cfg.parameters.delta_K)),
    )


def _solve_full_planner(draw, params, cfg, inputs, me):
    """The single new leg: full free-capital PLANNER on this arrangement.

    Correctness-critical (SPEC sec. 4 flag): the planner MUST be seeded with
    THIS arrangement's own frozen participation --
    ``wf_reference=me.participation.wf_reference`` and
    ``active_mask=me.participation.active_mask`` -- and warm-started from its own
    market solve ``initial=(me.w, me.X_market)``. Copied verbatim from
    ``welfare/allocations.py:49-57`` /
    ``fixed_input_welfare/run_fixed_capital_planner.py:128-138``. A naive
    ``solve_pooled_ge(regime=PLANNER)`` with a fresh entry mask would silently
    break CRN.
    """
    return solve_pooled_ge(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(),
        regime=PricingRegime.PLANNER,
        wf_reference=me.participation.wf_reference,   # FROZEN from this arrangement
        active_mask=me.participation.active_mask,      # FROZEN -- never a fresh mask
        initial=(me.w, me.X_market),                   # warm from own market solve
    )


# --- provenance + cache readers ----------------------------------------------

def _load_provenance(sorting_objects_yaml: Path) -> dict:
    """Inherit ``M``/``seed``/``shuffle_seed``/... verbatim from the sorting run.

    There are deliberately NO ``--M``/``--seed`` CLI flags: the base draw must
    be bit-identical to the cached reallocation run so lambda_K can be reused.
    """
    with open(sorting_objects_yaml) as stream:
        objects = yaml.safe_load(stream)
    return {
        "M": int(objects["M"]),
        "seed": int(objects["seed"]),
        "shuffle_seed": int(objects["shuffle_seed"]),
        "n_shuffle": int(objects["n_shuffle"]),
        "chi_baseline": float(objects["chi_baseline"]),
        "y_hat": float(objects["y_hat"]),
        "reanchored": bool(objects.get("reanchored_yhat_baseline_median", False)),
        "modes": list(objects.get("modes", [])),
    }


def _cache_row(*, W_market, W_planner_fixed_k, lambda_K, K_target_gap,
               corr_av, C, K, R, y_anchor, mu_cw_market) -> dict:
    """Uniform per-arrangement reallocation-leg record (cache-read or in-process)."""
    return {
        "W_market": float(W_market),
        "W_planner_fixed_k": float(W_planner_fixed_k),
        "lambda_K": float(lambda_K),
        "K_target_gap": float(K_target_gap),
        "corr_av": float(corr_av),
        "C": float(C), "K": float(K), "R": float(R),
        "y_anchor": float(y_anchor),
        "mu_cw_market": float(mu_cw_market),
    }


def _read_cached_lambda_k(sorting_dir: Path, homogeneous_dir: Path) -> dict:
    """Read the cached reallocation leg, keyed by ``(label, perm_seed)``.

    ``sorting_economies.parquet`` -> baseline (perm_seed None), shuffle x n
    (keyed by int perm_seed), reverse (perm_seed None).
    ``homogeneous_economies.parquet`` -> hom_primitive / hom_cost_weighted
    (perm_seed None); these carry only welfare levels + lambda_K, so the
    sorting-only diagnostics (corr_av, C, K, R, y_anchor, mu_cw_market) are NaN.
    """
    econ = pd.read_parquet(sorting_dir / "sorting_economies.parquet")
    hom = pd.read_parquet(homogeneous_dir / "homogeneous_economies.parquet")

    cache: dict = {}
    for _, r in econ.iterrows():
        mode = str(r["mode"])
        if mode == "shuffle":
            key = ("shuffle", int(r["perm_seed"]))
        elif mode in ("baseline", "reverse"):
            key = (mode, None)
        else:
            continue
        cache[key] = _cache_row(
            W_market=r["W_market"], W_planner_fixed_k=r["W_planner_fixed_k"],
            lambda_K=r["lambda_K"], K_target_gap=r["K_target_gap"],
            corr_av=r["corr_av"], C=r["C"], K=r["K"], R=r["R"],
            y_anchor=r["y_anchor"], mu_cw_market=r["mu_cw_market"],
        )
    for _, r in hom.iterrows():
        label = _HOM_VARIANT_LABEL.get(str(r["variant"]))
        if label is None:
            continue
        cache[(label, None)] = _cache_row(
            W_market=r["W_market"], W_planner_fixed_k=r["W_planner_fixed_k"],
            lambda_K=r["lambda_K"], K_target_gap=float("nan"),
            corr_av=float("nan"), C=float("nan"), K=float("nan"),
            R=float("nan"), y_anchor=float("nan"), mu_cw_market=float("nan"),
        )
    return cache


# --- arrangement family (reproduce the cached shuffle-seed sequence) ---------

def _arrangement_family(base_draw, me_base, shuffle_seed, n_shuffle):
    """Yield ``(label, perm_seed, draw)`` for the 14 arrangements.

    CRITICAL: the shuffle perm_seed sequence and the per-shuffle RNG are
    reproduced EXACTLY as in ``run_scalability_sorting.py:223-227``
    (``perm_seed = shuffle_seed + i``; RNG seeded with ``[shuffle_seed, i]``)
    so the shuffle rows align 1:1 with ``sorting_economies.parquet`` for the
    per-draw ``Delta_scale`` differencing (SPEC sec. 6-iii).
    """
    yield "baseline", None, permute_alpha(base_draw, "identity")
    for i in range(int(n_shuffle)):
        perm_seed = int(shuffle_seed) + i
        rng = np.random.default_rng([int(shuffle_seed), i])
        yield "shuffle", perm_seed, permute_alpha(base_draw, "shuffle", rng)
    yield "reverse", None, permute_alpha(base_draw, "reverse")
    yield "hom_primitive", None, homogenize_alpha_primitive(base_draw)
    yield ("hom_cost_weighted", None,
           homogenize_alpha_cost_weighted(base_draw, me_base))


# --- consistency guard (copy-adapt homogeneous_rts:183-215) ------------------

def _rel_diff(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _consistency_guard(me_base, base_draw, cached_base_row, cached_perm_seeds,
                       regenerated_perm_seeds, chi_baseline, prov, out_dir,
                       *, in_process: bool) -> dict:
    """Two-run drift guard (design (a)): rel-diff the recomputed baseline MARKET
    economy against the cached sorting-run baseline row, and assert the shuffle
    perm_seeds align 1:1. Writes ``consistency_check.yaml`` BEFORE raising.

    In the in-process smoke mode there is no second run, so ``cached_base_row``
    is the baseline's own fresh fixed-K record and ``chi`` is checked against
    itself -- the rel-diffs are trivially ~0 and the guard documents the mode.
    """
    checks = {
        "C": (float(me_base.C), cached_base_row["C"]),
        "K": (float(me_base.K_agg), cached_base_row["K"]),
        "R": (float(me_base.R), cached_base_row["R"]),
        "y_anchor": (float(me_base.y_anchor), cached_base_row["y_anchor"]),
        "corr_av": (active_corr_av(base_draw), cached_base_row["corr_av"]),
        "chi_baseline": (chi_baseline,
                         chi_baseline if in_process else prov["chi_baseline"]),
    }
    consistency = {
        key: {"recomputed": actual, "stored": expected,
              "rel_diff": _rel_diff(actual, expected)}
        for key, (actual, expected) in checks.items()
    }
    cached_seeds = sorted(int(s) for s in cached_perm_seeds)
    regen_seeds = sorted(int(s) for s in regenerated_perm_seeds)
    seeds_aligned = set(cached_seeds) == set(regen_seeds)

    payload = {
        "mode": "in_process" if in_process else "reuse_cached_lambda_k",
        "checks": consistency,
        "perm_seeds_cached": cached_seeds,
        "perm_seeds_regenerated": regen_seeds,
        "perm_seeds_aligned": bool(seeds_aligned),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(out_dir / "consistency_check.yaml", payload)

    bad = {k: v for k, v in consistency.items() if v["rel_diff"] > 1e-6}
    if bad or not seeds_aligned:
        raise AssertionError(
            "scale-channel consistency guard FAILED: "
            f"rel-diff>1e-6 on {list(bad)} and/or perm_seed misalignment "
            f"(aligned={seeds_aligned}). The scale-channel run and the cached "
            "reallocation runs are decoupled -- rerun scalability_sorting / "
            "homogeneous_rts with matching --calibration, or check for "
            "calibration/config drift. See consistency_check.yaml."
        )
    print(f"consistency guard passed (mode={payload['mode']}, "
          f"max rel_diff={max(v['rel_diff'] for v in consistency.values()):.2e}, "
          f"perm_seeds_aligned={seeds_aligned})")
    return payload


# --- per-arrangement row ------------------------------------------------------

def _economy_row(label, perm_seed, draw, me, pe, cached_row, *, chi_baseline,
                 beta, phi, n_firms_cs) -> dict:
    """One row of ``scale_channel_economies.parquet``.

    ``lambda_total`` / ``lambda_scale`` are anchored on the CACHED reallocation
    welfare levels (``W_market``, ``W_planner_fixed_k``) so that, with
    ``W_planner`` the new full-GE solve, the exponentials telescope and the log
    identity ``Delta_scale = Delta_total - Delta_K`` closes to machine
    precision. CRN drift is surfaced separately as ``drift_W_market_rel``.
    """
    W_market_cached = cached_row["W_market"]
    W_fp_cached = cached_row["W_planner_fixed_k"]
    lambda_K = cached_row["lambda_K"]

    # new full free-capital planner welfare, chi frozen at the baseline value
    W_planner = steady_state_welfare(pe.C, pe.L_agg, chi_baseline, beta, phi)
    lambda_scale = consumption_equivalent_lambda(W_fp_cached, W_planner, beta)
    lambda_total = consumption_equivalent_lambda(W_market_cached, W_planner, beta)

    # re-solved market welfare (guard/diagnostic only; NOT used for lambda_total)
    W_market_resolved = steady_state_welfare(me.C, me.L_agg, chi_baseline, beta, phi)

    delta_K = float(np.log1p(lambda_K))
    delta_total = float(np.log1p(lambda_total))
    delta_scale = delta_total - delta_K  # SPEC canonical definition

    K_market = float(me.K_agg)
    K_planner = float(pe.K_agg)
    return {
        "label": label,
        "perm_seed": (float(perm_seed) if perm_seed is not None else float("nan")),
        "corr_av": active_corr_av(draw),
        "W_market_cached": float(W_market_cached),
        "W_market_resolved": float(W_market_resolved),
        "drift_W_market_rel": _rel_diff(W_market_resolved, W_market_cached),
        "W_planner": float(W_planner),
        "W_planner_fixed_k_cached": float(W_fp_cached),
        "lambda_K_cached": float(lambda_K),
        "lambda_total": float(lambda_total),
        "lambda_scale": float(lambda_scale),
        "delta_K": delta_K,
        "delta_total": delta_total,
        "delta_scale": float(delta_scale),
        # termwise-vs-direct closure of Delta_scale (acceptance check)
        "delta_scale_termwise_gap": float(delta_scale - np.log1p(lambda_scale)),
        "K_market": K_market,
        "K_planner": K_planner,
        "K_ratio": (K_planner / K_market if K_market else float("nan")),
        "K_target_gap_cached": cached_row["K_target_gap"],
        "mu_cw_market": cached_row["mu_cw_market"],
        "mu_cw_planner": float(aggregate_markup_cw(pe, n_firms_cs=n_firms_cs)),
        "identity_residual": identity_residual(lambda_K, lambda_scale, lambda_total),
        "market_converged": bool(me.converged),
        "planner_converged": bool(pe.converged),
    }


# --- 3x3 assembly -------------------------------------------------------------

def _log1p_mean_se(values: np.ndarray) -> tuple[float, float]:
    """Mean +/- SE on the ``log1p`` scale (matches homogeneous_rts)."""
    x = np.log1p(np.asarray(values, dtype=np.float64))
    mean = float(np.mean(x)) if x.size else float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
    return mean, se


def _channel_block(split: dict, delta_base: float, *, se: float | None) -> dict:
    """One 3x3 row: values, percentage shares of delta_base, SEs on the
    shuffle-derived (heterogeneity, sorting) terms."""
    def share(x: float) -> float:
        return float(100.0 * x / delta_base) if abs(delta_base) > 1e-30 else float("nan")
    block = {
        "common_alpha": {"value": split["common_alpha"],
                         "share_pct": share(split["common_alpha"])},
        "heterogeneity": {"value": split["heterogeneity"],
                          "share_pct": share(split["heterogeneity"])},
        "sorting": {"value": split["sorting"],
                    "share_pct": share(split["sorting"])},
        "delta_base": float(delta_base),
    }
    if se is not None:
        block["heterogeneity"]["se"] = float(se)
        block["sorting"]["se"] = float(se)
    return block


def _assemble_3x3(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)

    def one(label: str):
        sub = df[df["label"] == label]
        return sub.iloc[0] if not sub.empty else None

    base = one("baseline")
    rev = one("reverse")
    hom_p = one("hom_primitive")
    hom_cw = one("hom_cost_weighted")
    shuf = df[df["label"] == "shuffle"].sort_values("perm_seed")

    lam_tot_shuf = shuf["lambda_total"].to_numpy()
    lam_K_shuf = shuf["lambda_K_cached"].to_numpy()

    # --- Delta_total row (the only genuinely new estimand) ---
    dtot_base = float(base["delta_total"])
    dtot_shuf_mean, dtot_shuf_se = _log1p_mean_se(lam_tot_shuf)
    dtot_hom = float(hom_p["delta_total"])
    total_split = three_channel_split(dtot_base, dtot_shuf_mean, dtot_hom)

    # --- Delta_K row (cached reallocation split; reproduces homogeneous_rts) ---
    dK_base = float(base["delta_K"])
    dK_shuf_mean, dK_shuf_se = _log1p_mean_se(lam_K_shuf)
    dK_hom = float(hom_p["delta_K"])
    K_split = three_channel_split(dK_base, dK_shuf_mean, dK_hom)

    # --- Delta_scale row: termwise (Delta_total - Delta_K), per-draw shuffle SE ---
    scale_split = {ch: total_split[ch] - K_split[ch] for ch in total_split}
    _, dscale_shuf_se = per_draw_scale_delta(
        np.log1p(lam_tot_shuf), np.log1p(lam_K_shuf)
    )
    dscale_base = dtot_base - dK_base

    # termwise-vs-per-draw closure of the shuffle-derived scale terms
    dscale_shuf_mean = float(np.mean(np.log1p(lam_tot_shuf) - np.log1p(lam_K_shuf)))
    dscale_hom = dtot_hom - dK_hom
    scale_split_perdraw = three_channel_split(dscale_base, dscale_shuf_mean, dscale_hom)
    termwise_gap = max(abs(scale_split[ch] - scale_split_perdraw[ch])
                       for ch in scale_split)

    three_by_three = {
        "delta_K": _channel_block(K_split, dK_base, se=dK_shuf_se),
        "delta_total": _channel_block(total_split, dtot_base, se=dtot_shuf_se),
        "delta_scale": _channel_block(scale_split, dscale_base, se=dscale_shuf_se),
        "delta_scale_termwise_gap": float(termwise_gap),
    }

    def arrangement(row, *, with_se=False):
        if row is None:
            return None
        out = {
            "lambda_K": float(row["lambda_K_cached"]),
            "lambda_total": float(row["lambda_total"]),
            "lambda_scale": float(row["lambda_scale"]),
            "K_ratio": float(row["K_ratio"]),
        }
        return out

    per_arrangement = {
        "baseline": arrangement(base),
        "reverse": arrangement(rev),
        "hom_primitive": arrangement(hom_p),
        "hom_cost_weighted": arrangement(hom_cw),
        "shuffle_mean": {
            "lambda_K": float(np.mean(lam_K_shuf)),
            "lambda_K_se": (float(np.std(lam_K_shuf, ddof=1) / np.sqrt(lam_K_shuf.size))
                            if lam_K_shuf.size > 1 else float("nan")),
            "lambda_total": float(np.mean(lam_tot_shuf)),
            "lambda_total_se": (float(np.std(lam_tot_shuf, ddof=1) / np.sqrt(lam_tot_shuf.size))
                                if lam_tot_shuf.size > 1 else float("nan")),
            "lambda_scale_mean_perdraw": dscale_shuf_mean,
            "lambda_scale_se_perdraw": dscale_shuf_se,
            "K_ratio": float(shuf["K_ratio"].mean()),
            "n_shuffle": int(shuf.shape[0]),
        },
    }

    # robustness: primitive vs cost-weighted common-alpha scale term (SPEC 6-ii)
    common_alpha_scale_primitive = scale_split["common_alpha"]  # uses hom_primitive
    common_alpha_scale_cost_weighted = (
        float(hom_cw["delta_total"] - hom_cw["delta_K"]) if hom_cw is not None
        else float("nan")
    )
    robustness = {
        "common_alpha_scale_primitive": float(common_alpha_scale_primitive),
        "common_alpha_scale_cost_weighted": float(common_alpha_scale_cost_weighted),
        "note": "SPEC sec. 6-ii: if the scale common-alpha term is stable across "
                "the primitive (per-sector harmonic-mean) and cost-weighted "
                "(aggregate-elasticity-preserving) hom constructions, the "
                "curvature-level confound is not driving it.",
    }

    return {
        "three_by_three": three_by_three,
        "per_arrangement": per_arrangement,
        "robustness": robustness,
        "diagnostics": {
            "identity_residual_max": float(df["identity_residual"].abs().max()),
            "delta_scale_termwise_gap_max": float(df["delta_scale_termwise_gap"].abs().max()),
            "drift_W_market_rel_max": float(df["drift_W_market_rel"].max()),
            "mu_cw_planner_max_abs_dev_from_1": float((df["mu_cw_planner"] - 1.0).abs().max()),
            "K_target_gap_cached_max_abs": float(df["K_target_gap_cached"].abs().max(skipna=True)),
            "all_converged": bool(df["market_converged"].all() and df["planner_converged"].all()),
        },
    }


# --- orchestration ------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    if args.calibration is None:
        print("warning: no --calibration given; using config initial params.")
    beta, phi = cfg.parameters.beta, cfg.parameters.phi

    if args.lens_b:
        print("note: --lens-b is an INERT stub this release; no UNIFORM-regime "
              "solves are run. Ignoring.")

    sorting_dir = (Path(args.sorting_dir) if args.sorting_dir
                   else cfg.out_results_dir / "counterfactuals" / "scalability_sorting")
    homogeneous_dir = (Path(args.homogeneous_dir) if args.homogeneous_dir
                       else cfg.out_results_dir / "counterfactuals" / "homogeneous_rts")
    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.out_results_dir / "counterfactuals" / "scale_channel_decomposition")

    prov = _load_provenance(sorting_dir / "sorting_objects.yaml")
    if prov["reanchored"]:
        raise ValueError(
            "cached sorting run was y_hat-reanchored "
            "(reanchored_yhat_baseline_median=true); this driver assumes the "
            "frozen calibrated y_hat. Rerun scalability_sorting without "
            "--reanchor-yhat-baseline-median."
        )

    M = int(args.M_smoke) if args.M_smoke is not None else prov["M"]
    seed = prov["seed"]
    use_cache = (M == prov["M"])
    if not use_cache:
        print(f"smoke: M={M} != cached M={prov['M']}; falling back to IN-PROCESS "
              f"fixed-K reallocation leg for every arrangement (no cache read).")

    # invariant 3: frozen y_hat (calibrated params field, never re-anchored)
    assert float(params.y_hat) == prov["y_hat"], (
        f"params.y_hat={float(params.y_hat)} != cached y_hat={prov['y_hat']}; "
        "calibration drift would break CRN comparability."
    )

    support = (inputs.alpha_support if inputs.alpha_support is not None
               else np.linspace(0.6, 1.20, 500))
    base_draw = draw_pool(
        support, xi=params.xi, rho_bar=params.rho_bar, N=params.N, M=M,
        H=params.H, rng=seed, v_min=params.v_min,
    )

    # baseline MARKET (cold start from calibration warm point) -> freeze chi
    me_base = _solve_market(base_draw, params, cfg, inputs, wf, initial)
    chi_baseline = float(me_base.chi)          # FREEZE once (invariant 1)
    warm = (me_base.w, me_base.X_market)
    print(f"baseline chi={chi_baseline:.6g}  K_base={me_base.K_agg:.6g}  "
          f"y_hat={float(params.y_hat):.6g}  corr_av={active_corr_av(base_draw):+.4f}  "
          f"M={M}")

    regenerated_perm_seeds = [int(prov["shuffle_seed"]) + i
                              for i in range(prov["n_shuffle"])]

    if use_cache:
        cached = _read_cached_lambda_k(sorting_dir, homogeneous_dir)
        cached_base_row = cached[("baseline", None)]
        cached_perm_seeds = [s for (lab, s) in cached if lab == "shuffle"]
    else:
        cached = None  # reallocation leg solved in-loop
        fp_base = _solve_fixed_k(base_draw, params, cfg, inputs, me_base)
        base_fk = fixed_capital_decomposition(
            me_base, fp_base, pe=None, chi_me=chi_baseline, beta=beta, phi=phi,
            n_firms_cs=inputs.n_firms_cs,
        )
        cached_base_row = _cache_row(
            W_market=base_fk["W_market"],
            W_planner_fixed_k=base_fk["W_planner_fixed_k"],
            lambda_K=base_fk["lambda_reallocation"],
            K_target_gap=base_fk["K_target_gap"],
            corr_av=active_corr_av(base_draw), C=me_base.C, K=me_base.K_agg,
            R=me_base.R, y_anchor=me_base.y_anchor,
            mu_cw_market=aggregate_markup_cw(me_base, n_firms_cs=inputs.n_firms_cs),
        )
        cached_perm_seeds = list(regenerated_perm_seeds)

    # consistency guard BEFORE the loop (raises on drift / perm_seed mismatch)
    _consistency_guard(
        me_base, base_draw, cached_base_row, cached_perm_seeds,
        regenerated_perm_seeds, chi_baseline, prov, out_dir,
        in_process=not use_cache,
    )

    rows: list[dict] = []
    for label, perm_seed, draw in _arrangement_family(
        base_draw, me_base, prov["shuffle_seed"], prov["n_shuffle"]
    ):
        if label == "baseline":
            me = me_base
        else:
            me = _solve_market(draw, params, cfg, inputs, wf, warm)
            if not me.converged:  # reverse is the poorest warm start; retry cold
                me = _solve_market(draw, params, cfg, inputs, wf, initial)

        pe = _solve_full_planner(draw, params, cfg, inputs, me)  # NEW leg

        # invariant 5: the planner reused THIS arrangement's own frozen mask
        assert (pe.participation.active_mask is me.participation.active_mask
                or np.array_equal(pe.participation.active_mask,
                                  me.participation.active_mask)), (
            f"active_mask not preserved market->planner for arrangement {label!r}"
        )

        if use_cache:
            cached_row = cached[(label, perm_seed)]
        else:
            fp = _solve_fixed_k(draw, params, cfg, inputs, me)
            fk = fixed_capital_decomposition(
                me, fp, pe=None, chi_me=chi_baseline, beta=beta, phi=phi,
                n_firms_cs=inputs.n_firms_cs,
            )
            cached_row = _cache_row(
                W_market=fk["W_market"], W_planner_fixed_k=fk["W_planner_fixed_k"],
                lambda_K=fk["lambda_reallocation"], K_target_gap=fk["K_target_gap"],
                corr_av=active_corr_av(draw), C=me.C, K=me.K_agg, R=me.R,
                y_anchor=me.y_anchor,
                mu_cw_market=aggregate_markup_cw(me, n_firms_cs=inputs.n_firms_cs),
            )

        row = _economy_row(label, perm_seed, draw, me, pe, cached_row,
                           chi_baseline=chi_baseline, beta=beta, phi=phi,
                           n_firms_cs=inputs.n_firms_cs)
        rows.append(row)
        print(f"[{label:>17}] corr_av={row['corr_av']:+.4f}  "
              f"lam_K={row['lambda_K_cached']:+.6g}  "
              f"lam_tot={row['lambda_total']:+.6g}  "
              f"lam_scale={row['lambda_scale']:+.6g}  "
              f"K_ratio={row['K_ratio']:.4g}  "
              f"mu_pe={row['mu_cw_planner']:.6g}  "
              f"id_res={row['identity_residual']:+.1e}  "
              f"drift={row['drift_W_market_rel']:.1e}")

    econ = pd.DataFrame(rows)
    assembled = _assemble_3x3(rows)

    decomposition = {
        "provenance": {
            "M": M, "seed": seed, "shuffle_seed": prov["shuffle_seed"],
            "n_shuffle": prov["n_shuffle"], "chi_baseline": chi_baseline,
            "y_hat": float(params.y_hat), "beta": float(beta), "phi": float(phi),
            "design": ("reuse_cached_lambda_k" if use_cache
                       else "in_process_fixed_k_smoke"),
            "source_parquets": {
                "sorting": str(sorting_dir / "sorting_economies.parquet"),
                "homogeneous": str(homogeneous_dir / "homogeneous_economies.parquet"),
            },
            "note": ("full free-capital planner run per arrangement; "
                     "lambda_K reused from cache (design (a))"
                     if use_cache else
                     "smoke run: reallocation leg solved in-process at reduced M "
                     "(cached parquets are a different M)"),
        },
        **assembled,
        "caveats": [
            "SPEC sec. 6-i: the Delta_scale SORTING term is expected ~0 -- the "
            "scale channel is capital deepening driven by aggregate curvature, "
            "which the marginal-preserving shuffle holds fixed. Reported, not "
            "asserted; it is the headline positive result.",
            "SPEC sec. 6-ii: homogeneous-alpha confound -- see robustness block.",
            "SPEC sec. 6-iii: Delta_scale shuffle SE is from per-draw CRN "
            "differencing (per_draw_scale_delta), not a ratio of averaged lambdas.",
            "SPEC sec. 6-v: absolute lambda levels ride the uncalibrated mu-tail; "
            "the relative 3x3 decomposition is the clean object.",
            "SPEC sec. 6-vi: two-run drift (design (a)) -- see consistency_check.yaml.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    econ.to_parquet(out_dir / "scale_channel_economies.parquet", index=False)
    _write_yaml(out_dir / "scale_channel_decomposition.yaml", decomposition)

    tbt = assembled["three_by_three"]
    diag = assembled["diagnostics"]
    print("\n--- 3x3 (Delta = log(1+lambda)) ---")
    for row_name in ("delta_K", "delta_total", "delta_scale"):
        blk = tbt[row_name]
        print(f"{row_name:>12}: common_alpha={blk['common_alpha']['value']:+.6g} "
              f"({blk['common_alpha']['share_pct']:.1f}%)  "
              f"heterogeneity={blk['heterogeneity']['value']:+.6g} "
              f"({blk['heterogeneity']['share_pct']:.1f}%)  "
              f"sorting={blk['sorting']['value']:+.6g} "
              f"({blk['sorting']['share_pct']:.1f}%)")
    print(f"\nDelta_scale SORTING term (headline, reported) = "
          f"{tbt['delta_scale']['sorting']['value']:+.6g} "
          f"(se {tbt['delta_scale']['sorting'].get('se', float('nan')):.3g})")
    print(f"identity_residual_max={diag['identity_residual_max']:.2e}  "
          f"delta_scale_termwise_gap_max={diag['delta_scale_termwise_gap_max']:.2e}  "
          f"mu_pe_dev_max={diag['mu_cw_planner_max_abs_dev_from_1']:.2e}  "
          f"drift_max={diag['drift_W_market_rel_max']:.2e}")
    print(f"\nwrote outputs to {out_dir}")
    if not diag["all_converged"]:
        print("WARNING: at least one market/planner leg did not fully converge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
