"""Level and dispersion of TFP (v) and RTS (alpha) in the welfare cost of markups.

Run as a module from 04_Code. Every economy is the calibrated sorting-curve draw
with v, alpha or both collapsed to their sector harmonic means, solved through
``sorting_curve.run_sorting_curve.solve_row`` (ME, PE_I, U, PE, both lenses).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from steady_state.__main__ import _params_for_command, _pooled_inputs, load_config
from steady_state.model.pool import draw_pool
from counterfactuals.homogeneous_rts.homogenize import homogenize_alpha_primitive
from counterfactuals.scalability_sorting.permute import active_corr_av
from counterfactuals.sorting_curve import run_sorting_curve as runner
from counterfactuals.sorting_curve.run_remaining_parallel import compact_inputs
from .homogenize import homogenize_v_primitive

ECONOMIES = {
    "baseline": lambda d: d,
    "hom_alpha": homogenize_alpha_primitive,
    "hom_v": homogenize_v_primitive,
    "hom_both": lambda d: homogenize_v_primitive(homogenize_alpha_primitive(d)),
}
GATE_KEYS = [f"W_{k}" for k in ("ME", "PE_I", "U", "PE")] + [f"delta_{leg}" for leg in runner.LEGS]


def cached_hom_alpha(root):
    """Saved homogeneous-alpha (primitive) W levels from the scale and lens-B runs."""
    sc = pd.read_parquet(root / "scale_channel_decomposition/scale_channel_economies.parquet")
    lb = pd.read_parquet(root / "lens_b_decomposition/lens_b_economies.parquet")
    s, b = sc[sc.label.eq("hom_primitive")].iloc[0], lb[lb.label.eq("hom_primitive")].iloc[0]
    row = {"W_ME": s.W_market_cached, "W_PE_I": s.W_planner_fixed_k_cached,
           "W_U": b.W_uniform, "W_PE": s.W_planner}
    return {k: float(v) for k, v in row.items()}


def gate(label, row, reference, beta, out):
    reference = dict(reference)
    for leg, (a, b) in zip(runner.LEGS, runner.PAIRS):
        reference.setdefault(f"delta_{leg}", (1 - beta) * (reference[f"W_{b}"] - reference[f"W_{a}"]))
    # W levels are compared relatively; Delta legs absolutely, since a near-zero
    # leg (hom_alpha disp ~1.6e-4) turns solver noise into a large relative gap.
    gaps = {k: abs(row[k] - reference[k]) / (max(abs(reference[k]), 1e-30) if k.startswith("W_") else 1.0)
            for k in GATE_KEYS}
    payload = {"label": label, "tolerance": 1e-6, "W_relative_delta_absolute_differences": gaps,
               "passed": max(gaps.values()) <= 1e-6}
    runner.atomic_json(out / f"validation_{label}.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"{label} does not reproduce the saved solve; see validation_{label}.json")


def decompose(rows):
    d = {lab: {leg: rows[lab][f"delta_{leg}"] for leg in runner.LEGS} for lab in rows}
    out = {}
    for leg in runner.LEGS:
        base, a, v, both = (d[k][leg] for k in ("baseline", "hom_alpha", "hom_v", "hom_both"))
        terms = {
            "tfp_level": v, "tfp_dispersion": base - v,
            "rts_level": a, "rts_dispersion": base - a,
            "joint_common": both, "joint_tfp_dispersion": a - both,
            "joint_rts_dispersion": v - both, "joint_interaction": base - a - v + both,
        }
        out[leg] = {"delta_base": base, **{k: {"value": float(x), "share_pct": float(100 * x / base)}
                                           for k, x in terms.items()}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--calibration", default="out_results/calib_pooled/calibration_pooled.yaml")
    p.add_argument("--out-dir", default="out_results/counterfactuals/level_dispersion")
    p.add_argument("--sorting-dir", default="out_results/counterfactuals/sorting_curve")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--M-smoke", type=int)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    cfg = dataclasses.replace(cfg, market=dataclasses.replace(cfg.market,
           parallel=dataclasses.replace(cfg.market.parallel, n_jobs=args.workers)))
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    prov = json.loads((Path(args.sorting_dir) / "run_manifest.json").read_text())["provenance"]
    checks = {"support_sha256": hashlib.sha256(inputs.alpha_support.tobytes()).hexdigest(),
              "calibration_sha256": hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()}
    for key, value in checks.items():
        if value != prov[key]:
            raise RuntimeError(f"{key} differs from the sorting-curve run")
    if params.y_hat != prov["y_hat"]:
        raise AssertionError("y_hat drift")
    M = args.M_smoke or prov["M"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = draw_pool(inputs.alpha_support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
                     M=M, H=params.H, rng=prov["seed"], v_min=params.v_min)
    # Trim inactive padding once; the transforms only touch active firms.
    compact, solver_params = compact_inputs(base, params)
    root = cfg.out_results_dir / "counterfactuals"
    references = {} if args.M_smoke else {
        "baseline": json.loads((Path(args.sorting_dir) / "checkpoint_baseline.json").read_text()),
        "hom_alpha": cached_hom_alpha(root)}
    rows = {}
    for label, transform in ECONOMIES.items():
        checkpoint = out / f"checkpoint_{label}.json"
        if checkpoint.exists():
            rows[label] = json.loads(checkpoint.read_text())
            continue
        draw = transform(compact)
        row = runner.solve_row(label, draw, solver_params, cfg, inputs, wf, initial, prov["chi_baseline"])
        row.update(draw_pool_H=params.H, numerical_solver_H=solver_params.H, solver_workers=args.workers,
                   corr_av=active_corr_av(draw))
        if label in references:
            gate(label, row, references[label], cfg.parameters.beta, out)
        runner.atomic_json(checkpoint, row)
        rows[label] = row
        print(f"[{label}] CHECKPOINT lambda_total={row['lambda_total']:.8f}", flush=True)
    pd.DataFrame(list(rows.values())).to_parquet(out / "level_dispersion_economies.parquet", index=False)
    summary = {"provenance": {**{k: prov[k] for k in ("seed", "chi_baseline", "y_hat", "beta", "phi")},
                              "M": M, "smoke": args.M_smoke is not None, **checks,
                              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "design": "sector harmonic means of v and alpha over active firms; frozen chi and y_hat"},
               "decomposition": decompose(rows)}
    (out / "level_dispersion.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
