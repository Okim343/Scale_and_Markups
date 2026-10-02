"""Checkpointed sorting sweep. Run as a module from 04_Code."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from steady_state.__main__ import _params_for_command, _pooled_inputs, load_config
from steady_state.model.pool import draw_pool
from steady_state.welfare.welfare_metrics import aggregate_markup_cw, steady_state_welfare
from counterfactuals.scalability_sorting.permute import active_corr_av, partial_permute
from counterfactuals.scalability_sorting.run_scalability_sorting import _solve_market, _solve_fixed_k
from counterfactuals.scale_channel_decomposition.run_scale_channel import _solve_full_planner
from counterfactuals.lens_b_decomposition.run_lens_b import _solve_uniform

LEGS = ("realloc", "scale", "disp", "level", "total")
PAIRS = (("ME", "PE_I"), ("PE_I", "PE"), ("ME", "U"), ("U", "PE"), ("ME", "PE"))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def add_lenses(row, beta):
    for leg, (a, b) in zip(LEGS, PAIRS):
        delta = (1 - beta) * (row[f"W_{b}"] - row[f"W_{a}"])
        row[f"delta_{leg}"] = float(delta)
        row[f"lambda_{leg}"] = float(np.expm1(delta))
        # L=1 makes the consumption and welfare definitions identical.
        row[f"consumption_gap_{leg}"] = float(delta - np.log(row[f"C_{b}"] / row[f"C_{a}"]))
        if abs(row[f"consumption_gap_{leg}"]) > 1e-4:
            raise AssertionError(f"consumption/welfare mismatch: {leg}")
    row["identity_A"] = row["delta_realloc"] + row["delta_scale"] - row["delta_total"]
    row["identity_B"] = row["delta_disp"] + row["delta_level"] - row["delta_total"]
    if max(abs(row["identity_A"]), abs(row["identity_B"])) > 1e-12:
        raise AssertionError("lens identity failed")
    return row


def solve_row(label, draw, params, cfg, inputs, wf, initial, chi):
    print(f"[{label}] MARKET corr={active_corr_av(draw):+.6f}", flush=True)
    me = _solve_market(draw, params, cfg, inputs, wf, initial)
    if not me.converged:
        me = _solve_market(draw, params, cfg, inputs, wf, (1., 1.))
    if not me.converged:
        raise RuntimeError(f"{label}: market did not converge")
    print(f"[{label}] FIXED-K", flush=True)
    fp = _solve_fixed_k(draw, params, cfg, inputs, me)
    print(f"[{label}] PLANNER", flush=True)
    pe = _solve_full_planner(draw, params, cfg, inputs, me)
    mu = float(aggregate_markup_cw(me, n_firms_cs=inputs.n_firms_cs))
    print(f"[{label}] UNIFORM", flush=True)
    u = _solve_uniform(draw, params, cfg, inputs, mu_bar=mu,
                       wf=me.participation.wf_reference, initial=(me.w, me.X_market))
    if not u.converged:
        u = _solve_uniform(draw, params, cfg, inputs, mu_bar=mu,
                           wf=me.participation.wf_reference, initial=(1., 1.))
    row = {"label": label, "corr_av": active_corr_av(draw), "source": "solved",
           "chi_own_ME": float(me.chi), "K_ME": float(me.K_agg), "K_PE": float(pe.K_agg),
           "mu_cw": mu, "y_hat": float(params.y_hat), "market_R": float(me.R),
           "market_y_anchor": float(me.y_anchor), "fixed_K_gap": float(fp.K_agg / me.K_agg - 1)}
    for key, eq in (("ME", me), ("PE_I", fp), ("U", u), ("PE", pe)):
        if not eq.converged:
            raise RuntimeError(f"{label}: {key} did not converge")
        if not np.array_equal(eq.participation.active_mask, me.participation.active_mask):
            raise AssertionError(f"{label}: {key} active_mask drift")
        if not np.array_equal(eq.participation.wf_reference, me.participation.wf_reference):
            raise AssertionError(f"{label}: {key} wf_reference drift")
        if not np.isclose(eq.L_agg, 1., atol=1e-4, rtol=0):
            raise AssertionError(f"{label}: {key} L != 1 within solver tolerance")
        row[f"converged_{key}"] = True
        row[f"L_{key}"] = float(eq.L_agg)
        row[f"C_{key}"] = float(eq.C)
        row[f"W_{key}"] = steady_state_welfare(eq.C, eq.L_agg, chi, cfg.parameters.beta, cfg.parameters.phi)
    row["mu_cw_U"] = float(aggregate_markup_cw(u, n_firms_cs=inputs.n_firms_cs))
    if abs(row["fixed_K_gap"]) > 1e-4 or not np.isclose(row["mu_cw_U"], mu, rtol=1e-7):
        raise AssertionError(f"{label}: fixed K or uniform markup target failed")
    return add_lenses(row, cfg.parameters.beta)


def cached_rows(root, prov):
    """Preserve each saved shuffle draw; reconstruct C from saved W at L=1."""
    sc = pd.read_parquet(root / "scale_channel_decomposition/scale_channel_economies.parquet")
    lb = pd.read_parquet(root / "lens_b_decomposition/lens_b_economies.parquet")
    so = pd.read_parquet(root / "scalability_sorting/sorting_economies.parquet")
    rows = []
    for _, s in sc[sc.label.isin(["baseline", "reverse", "shuffle"])].iterrows():
        sel = lb.label.eq(s.label)
        sel_so = so["mode"].eq(s.label)
        if s.label == "shuffle":
            sel &= lb.perm_seed.eq(s.perm_seed)
            sel_so &= so.perm_seed.eq(s.perm_seed)
        b, o = lb[sel].iloc[0], so[sel_so].iloc[0]
        label = s.label if s.label != "shuffle" else f"shuffle_{int(s.perm_seed)}"
        row = {"label": label, "kind": "cached", "source": "saved_endpoint",
               "base_mode": s.label, "q": None, "perm_seed": None if pd.isna(s.perm_seed) else str(int(s.perm_seed)),
               "rho_bar": None, "corr_av": float(s.corr_av), "K_ME": float(s.K_market),
               "K_PE": float(s.K_planner), "mu_cw": float(s.mu_cw_market), "mu_cw_U": float(b.mu_cw_uniform),
               "y_hat": prov["y_hat"], "chi_own_ME": float(o.chi_own),
               "market_R": float(o.R), "market_y_anchor": float(o.y_anchor), "fixed_K_gap": float(o.K_target_gap)}
        for key, w in (("ME", s.W_market_cached), ("PE_I", s.W_planner_fixed_k_cached),
                       ("U", b.W_uniform), ("PE", s.W_planner)):
            row[f"L_{key}"] = 1.0
            row[f"W_{key}"] = float(w)
            row[f"C_{key}"] = float(np.exp((1-prov["beta"])*w + prov["chi_baseline"]/(1+1/prov["phi"])))
        for key, flag in (("ME", o.market_converged), ("PE_I", o.fixed_k_converged),
                          ("PE", s.planner_converged), ("U", b.uniform_converged)):
            row[f"converged_{key}"] = bool(flag)
        if not all(row[f"converged_{k}"] for k in ("ME", "PE_I", "PE", "U")):
            raise AssertionError("unconverged cached endpoint")
        if not np.isclose(row["C_ME"], o.C, rtol=1e-6):
            raise AssertionError("cached C reconstruction failed")
        row["C_ME"] = float(o.C)
        row["L_ME"] = float(o.L)
        rows.append(add_lenses(row, prov["beta"]))
    return rows


def validate_baseline(row, reference, chi, out):
    keys = [f"{prefix}_{key}" for prefix in ("W", "C") for key in ("ME", "PE_I", "U", "PE")]
    keys += [f"{prefix}_{key}" for prefix in ("lambda", "delta") for key in LEGS]
    keys += ["corr_av", "K_ME", "K_PE", "mu_cw", "market_R", "market_y_anchor"]
    differences = {k: abs(row[k]-reference[k])/max(abs(reference[k]), 1e-30) for k in keys}
    differences["chi_baseline"] = abs(row["chi_own_ME"]-chi)/abs(chi)
    payload = {"relative_tolerance": 1e-6, "relative_differences": differences,
               "passed": all(v <= 1e-6 for v in differences.values())}
    atomic_json(out / "validation.json", payload)
    if not payload["passed"]:
        raise RuntimeError("q=0 baseline validation failed; stopped before new points. See validation.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--calibration", default="out_results/calib_pooled/calibration_pooled.yaml")
    p.add_argument("--out-dir", default="out_results/counterfactuals/sorting_curve")
    p.add_argument("--fig-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--M-smoke", type=int)
    p.add_argument("--no-rho-check", action="store_true")
    p.add_argument("--validation-only", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    cfg = dataclasses.replace(cfg, market=dataclasses.replace(cfg.market,
           parallel=dataclasses.replace(cfg.market.parallel, n_jobs=args.workers)))
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    root = cfg.out_results_dir / "counterfactuals"
    prov = yaml.safe_load((root / "scale_channel_decomposition/scale_channel_decomposition.yaml").read_text())["provenance"]
    if params.y_hat != prov["y_hat"]:
        raise AssertionError("y_hat drift")
    M = args.M_smoke or prov["M"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    support = inputs.alpha_support
    if support is None:
        raise ValueError("calibrated alpha support is required")
    # Every calibrated draw parameter comes from params, never inputs.
    def make_draw(rho):
        return draw_pool(support, xi=params.xi, rho_bar=rho, N=params.N,
                         M=M, H=params.H, rng=prov["seed"], v_min=params.v_min)
    base = make_draw(params.rho_bar)
    chi = prov["chi_baseline"]
    provenance = {**prov, "M": M, "rho_bar": float(params.rho_bar), "workers": args.workers,
                  "calibrated_params": dataclasses.asdict(params), "smoke": args.M_smoke is not None,
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "config_sha256": hashlib.sha256(Path(args.config or 'steady_state/config.yaml').read_bytes()).hexdigest(),
                  "calibration_sha256": hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest(),
                  "support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
                  "design": "active-within-sector partial permutation; frozen chi and y_hat",
                  "rho_crosscheck": not args.no_rho_check,
                  "rho_caveat": "alpha, tilde_alpha, eps, active_mask share CRN; v changes and its marginal is identical in distribution only, not bitwise",
                  "source_parquets": {"scale": str(root / "scale_channel_decomposition/scale_channel_economies.parquet"),
                    "lens_b": str(root / "lens_b_decomposition/lens_b_economies.parquet"),
                    "sorting": str(root / "scalability_sorting/sorting_economies.parquet")}}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, default=str).encode()).hexdigest()
    manifest = out / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("checkpoint provenance differs; choose a new output directory")
    atomic_json(manifest, {"fingerprint": fingerprint, "provenance": provenance})
    cache = [] if args.M_smoke else cached_rows(root, prov)
    plan = [("baseline", "identity", 0., 0)]
    plan += [(f"identity_q{q:.2f}_d{d}", "identity", q, d)
             for q in (.2, .4, .6, .8) for d in (range(2) if q == .4 else range(1))]
    plan += [(f"reverse_q{q:.2f}_d0", "reverse", q, 0) for q in (.33, .67)]
    if args.M_smoke:
        plan += [("reverse", "reverse", 0., 0), ("shuffle_smoke", "identity", 1., 0)]
    if not args.no_rho_check:
        plan += [(f"rho_{rho:.2f}", "rho", rho, 0) for rho in (.45, .70)]
    rows = [r for r in cache if r["label"] != "baseline"]
    for idx, (label, mode, q, d) in enumerate(plan):
        if args.validation_only and idx > 0:
            break
        checkpoint = out / f"checkpoint_{label}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text())
        else:
            perm_seed = [int(prov["shuffle_seed"]), idx, d]
            draw = make_draw(q) if mode == "rho" else partial_permute(base, mode, q, np.random.default_rng(perm_seed))
            if mode == "rho":
                for field in ("alpha", "tilde_alpha", "active_mask"):
                    if not np.array_equal(getattr(base, field), getattr(draw, field)):
                        raise AssertionError(f"rho CRN drift: {field}")
            row = solve_row(label, draw, params, cfg, inputs, wf, initial, chi)
            row.update(kind="rho" if mode == "rho" else "partial", base_mode=mode,
                       q=None if mode == "rho" else q, perm_seed=json.dumps(perm_seed),
                       rho_bar=q if mode == "rho" else float(params.rho_bar))
            if label == "baseline" and not args.M_smoke:
                validate_baseline(row, next(r for r in cache if r["label"] == "baseline"), chi, out)
            atomic_json(checkpoint, row)
        if label == "baseline" and not args.M_smoke:
            validate_baseline(row, next(r for r in cache if r["label"] == "baseline"), chi, out)
        rows.append(row)
        frame = pd.DataFrame(rows)
        tmp = out / "sorting_curve_economies.tmp.parquet"
        frame.to_parquet(tmp, index=False)
        tmp.replace(out / "sorting_curve_economies.parquet")
        from .figure import write_artifacts
        fig_dir = Path(args.fig_dir) if args.fig_dir else (out / "figures" if args.M_smoke else cfg.out_figs_dir / "counterfactuals")
        write_artifacts(frame, provenance, out, fig_dir, complete=idx == len(plan)-1)
        print(f"[{label}] CHECKPOINT lambda_total={row['lambda_total']:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
