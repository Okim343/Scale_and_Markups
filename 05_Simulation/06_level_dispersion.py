"""Step 06: joint capability/scalability experiment consistent with Tables 2/3.

Freshly solves all four market/planner pairs. Baseline and common-scalability
results must reproduce the saved sorting references before new economies run.
The draw policy is explicit: draw_pool_truncated_v1 is the truncated draw of
02/03/04, at the M, seed and alpha cap of the sorting reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from orbis_model.config import load_config, setup_logger
from orbis_model.counterfactuals.level_dispersion.experiment import (
    DRAW_POLICY, LABELS, arrangements, compact_draw, compare_reference,
    reference_draw, solve_economy, write_results,
)
from orbis_model.counterfactuals.sorting.sweep import atomic_json
from orbis_model.params import bundle_hashes, file_sha256, load_simulation_inputs


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config")
    p.add_argument("--params")
    p.add_argument("--sorting-dir")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--smoke-M", type=int, help="small functional run; no production reference gate")
    p.add_argument("--reference-draw-policy", required=True, choices=[DRAW_POLICY])
    return p


def verify_inputs(economics, *, params_sha, bundle_sha, params, cfg):
    if params_sha != economics["params_sha256"]:
        raise ValueError("calibration file differs from the saved sorting reference")
    if bundle_sha != economics["bundle_sha256"]:
        raise ValueError("bundle differs from the saved sorting reference")
    for key in ("beta", "delta_K", "phi", "H", "alpha_cap_factor"):
        if getattr(cfg.parameters, key) != economics["parameters"][key]:
            raise ValueError(f"config {key} differs from the reference")
    if params.H != economics["parameters"]["H"]:
        raise ValueError("params-file pool width differs from the reference")
    if cfg.participation.love_of_variety or cfg.participation.K_switch != economics["participation"]["K_switch"]:
        raise ValueError("participation settings differ from the reference")


def code_hash():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve()] + sorted((root / "orbis_model").rglob("*.py"))
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logger("level_dispersion", cfg.log_level)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.smoke_M is not None and args.smoke_M < 1:
        raise ValueError("smoke-M must be positive")
    if args.smoke_M and not args.out_dir:
        raise ValueError("smoke runs require a separate explicit --out-dir")
    started = time.perf_counter()
    refs = Path(args.sorting_dir).resolve() if args.sorting_dir else cfg.out_results_dir / "sorting"
    out = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir / "level_dispersion"
    if out == refs or refs in out.parents:
        raise ValueError("experiment outputs must be separate from the sorting references")
    economics = json.loads((refs / "run_manifest.json").read_text())["economics"]
    reference_rows = {"baseline": json.loads((refs / "checkpoint_baseline.json").read_text()),
                      "hom_alpha": json.loads((refs / "checkpoint_hom_sector.json").read_text())}
    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    params_sha = file_sha256(params_path)
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, log)
    if file_sha256(params_path) != params_sha:
        raise RuntimeError("calibration file changed while it was being read")
    bundle_sha = bundle_hashes(cfg.bundle_dir)
    verify_inputs(economics, params_sha=params_sha, bundle_sha=bundle_sha,
                  params=params, cfg=cfg)
    M = args.smoke_M or int(economics["M"])
    source_hashes = {p.name: file_sha256(p) for p in
                     (refs / "run_manifest.json", refs / "checkpoint_baseline.json",
                      refs / "checkpoint_hom_sector.json")}
    provenance = {"M": M, "seed": economics["seed"], "draw_policy": DRAW_POLICY,
                  "params_sha256": params_sha, "bundle_sha256": bundle_sha,
                  "source_hashes": source_hashes, "code_sha256": code_hash(),
                  "workers": args.workers, "smoke": bool(args.smoke_M),
                  "reference_economics": economics, "outer_tol": 1e-7}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("output contains another experiment; choose a new --out-dir")
    atomic_json(manifest_path, {"fingerprint": fingerprint, "provenance": provenance})
    (out / "frozen_calibrated_params.yaml").write_bytes(params_path.read_bytes())
    if file_sha256(out / "frozen_calibrated_params.yaml") != params_sha:
        raise RuntimeError("calibration file changed while freezing inputs")
    log.info("Rebuilding the sorting reference draw (%s) at M=%s, H=%s", DRAW_POLICY, M, params.H)
    full = reference_draw(primitives.alpha_support, params, M=M, seed=economics["seed"],
                          cap=economics["alpha_cap"])
    draw, solver_params = compact_draw(full, params)
    del full
    if not args.smoke_M and int(draw.active_mask.sum()) != reference_rows["baseline"]["n_active"]:
        raise RuntimeError("reconstructed firm count differs from the saved reference")
    mask = draw.active_mask
    draw_checks = {"n_markets": M, "n_active": int(mask.sum()), "compact_width": solver_params.H,
                   "corr_av": float(np.corrcoef(draw.alpha[mask], draw.v[mask])[0, 1]),
                   "cap_firm_share": float(np.mean(draw.alpha[mask] >= economics["alpha_cap"]))}
    if not args.smoke_M:
        ref = reference_rows["baseline"]
        if not np.isclose(draw_checks["corr_av"], ref["corr_av"], rtol=0, atol=1e-12):
            raise RuntimeError("reconstructed alpha-v pairing differs from the saved reference")
    market_kwargs = {**cfg.market.to_kwargs(), **economics["solver_market"],
                     "parallel_n_jobs": args.workers}
    P = cfg.parameters
    common = dict(chi=reference_rows["baseline"]["chi"], beta=P.beta, phi=P.phi,
                  delta_K=P.delta_K, n_firms_cs=primitives.n_firms_cs,
                  market_kwargs=market_kwargs, initial=initial, wf_reference=wf, logger=log)
    rows, validation = {}, {"draw": draw_checks, "reference_gates": {}, "smoke": bool(args.smoke_M)}
    for label, arrangement in arrangements(draw).items():
        cp = out / f"checkpoint_{label}.json"
        if cp.exists():
            row = json.loads(cp.read_text())
            log.info("[%s] resumed the matching experiment checkpoint", label)
        else:
            row = solve_economy(label, arrangement, solver_params, **common)
            atomic_json(cp, row)
        if label in reference_rows and not args.smoke_M:
            gate = compare_reference(row, reference_rows[label])
            # Saved solves stopped at outer tolerance 1e-5. Independently
            # reproduce their allocation at their recorded prices to distinguish
            # a tiny normalization improvement from a change in model or draws.
            at_saved_prices = solve_economy(
                f"reference_prices_{label}", arrangement, solver_params,
                reference_prices=reference_rows[label], **common)
            price_gate = compare_reference(at_saved_prices, reference_rows[label],
                                           rtol=1e-8, atol=1e-8)
            atomic_json(out / f"reference_prices_{label}.json", at_saved_prices)
            gate["saved_price_reproduction"] = price_gate
            gate["passed"] = gate["passed"] and price_gate["passed"]
            validation["reference_gates"][label] = gate
            atomic_json(out / "validation.json", validation)
            if not gate["passed"]:
                failed = {k: v for k, v in gate["checks"].items() if not v["passed"]}
                raise RuntimeError(f"{label} reference gate failed; no new economies run: {failed}")
        rows[label] = row
        log.info("[%s] Y=%+.6f%% welfare=%+.6f%% (%0.1fs)", label,
                 row["Y_pct"], row["welfare_pct"], row["wall_s_total"])
    validation["passed"] = bool(not args.smoke_M and len(validation["reference_gates"]) == 2
                                 and all(g["passed"] for g in validation["reference_gates"].values()))
    validation["max_consumption_welfare_gap"] = max(r["consumption_welfare_gap"] for r in rows.values())
    validation["runtime_s"] = time.perf_counter() - started
    atomic_json(out / "validation.json", validation)
    write_results(out, rows, validation, provenance)
    log.info("Wrote %s; production validation=%s; elapsed %.1fs", out,
             validation["passed"], validation["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
