"""Step 07: both welfare lenses for the four economies of 06 (heterogeneity anatomy).

Composes existing pieces without changing them. The four economies are the
ones of ``06_level_dispersion.py``: one ``draw_pool_truncated_v1`` draw at the
sorting reference's M, seed and alpha cap, collapsed by
``level_dispersion.experiment.arrangements`` (sector harmonic means of alpha
and v). Each economy is solved by ``sorting.sweep.solve_arrangement``, the
code behind the lens-along-the-sweep figure, which solves ME, PE_I, PE_K, PE
and U and asserts both lens identities (``add_lenses``):

    Lens B: dispersion (ME -> U) + level (U -> PE)
    Lens A: misallocation (ME -> PE_I) + materials (PE_I -> PE_K) + capital (PE_K -> PE)

chi is frozen at the sorting baseline's market value, labor is one, the anchor
and firm sets are those of the draw; the fixed-input tolerance is the sorting
sweep's.

Gates (production only):
    1. each economy's lambda_total reproduces its 06 checkpoint;
    2. the baseline reproduces 02_welfare.py and 03_fixed_input_welfare.py
       (``sorting.sweep.validate_baseline``, relative tolerance 1e-6);
    3. lens identities and convergence (enforced inside ``solve_arrangement``).

Each economy is checkpointed as it finishes, so economies can run as separate
processes (``--labels``); outputs are assembled once all four checkpoints exist.

Reads   <paths.params_file>, <paths.bundle_dir>, <level-dispersion-dir>/{run_manifest,checkpoint_*}.json,
        <paths.out_results_dir>/{welfare_*,fixed_input/fixed_input_*}
Writes  <out-dir>/checkpoint_<label>.json, run_manifest.json, validation.json,
        heterogeneity_anatomy_economies.parquet, anatomy_regimes.csv, anatomy_legs.csv,
        heterogeneity_anatomy.yaml, results.md
(default <out-dir> = <paths.out_results_dir>/heterogeneity_anatomy)

    python 07_heterogeneity_anatomy.py --reference-draw-policy draw_pool_truncated_v1
        [--config FILE] [--params FILE] [--labels a,b] [--workers N] [--assemble-only]
        [--smoke-M N --out-dir D]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from orbis_model.config import load_config, setup_logger
from orbis_model.counterfactuals.level_dispersion.experiment import (
    DRAW_POLICY, LABELS, arrangements, compact_draw, compare_reference, reference_draw,
)
from orbis_model.counterfactuals.sorting.sweep import (
    LENS_A, LENS_B, REGIMES, ReferencePaths, atomic_json, check_reference_manifests,
    solve_arrangement, validate_baseline,
)
from orbis_model.model.pool import alpha_cap
from orbis_model.params import bundle_hashes, file_sha256, load_simulation_inputs

COLUMNS = ("hom_both", "hom_alpha", "hom_v", "baseline")
COLUMN_TITLES = {"hom_both": "Homogeneous", "hom_alpha": "ν only",
                 "hom_v": "α only", "baseline": "Calibrated"}
PANEL_ROWS = (("disp", "Dispersion"), ("level", "Level"), ("realloc", "Misallocation"),
              ("materials", "Materials"), ("capital", "Capital"), ("total", "Total"))
LEGS = ("realloc", "materials", "capital", "total", "disp", "level")
DELTA_RTOL = 1e-6


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config")
    p.add_argument("--params")
    p.add_argument("--level-dispersion-dir", help="06 outputs (default: <out_results_dir>/level_dispersion)")
    p.add_argument("--out-dir")
    p.add_argument("--labels", help=f"comma-separated subset of {','.join(LABELS)} (default: all)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--assemble-only", action="store_true", help="solve nothing; assemble checkpoints")
    p.add_argument("--smoke-M", type=int, help="small functional run; no production reference gate")
    p.add_argument("--reference-draw-policy", required=True, choices=[DRAW_POLICY])
    return p


def verify_inputs(economics, *, params_sha, bundle_sha, params, cfg):
    """The checks of 06's ``verify_inputs`` against the sorting reference economics."""
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
    if not math.isclose(alpha_cap(params.gamma, cfg.parameters.alpha_cap_factor),
                        economics["alpha_cap"], rel_tol=1e-14):
        raise ValueError("alpha cap differs from the reference")


def atomic_json_shared(path, value):
    """``atomic_json`` with a per-process temporary, for files concurrent economy jobs share."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def code_hash():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve()] + sorted((root / "orbis_model").rglob("*.py"))
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def arrangement_checks(arr, base):
    """Which primitive is common within sectors, and that the other is the drawn one bitwise."""
    mask = base.active_mask
    multi = mask.sum(axis=1) > 1

    def within_sector_spread(x):
        lo = np.where(mask, x, np.inf).min(axis=1)
        hi = np.where(mask, x, -np.inf).max(axis=1)
        return float(np.max((hi - lo)[multi])) if multi.any() else 0.0

    def harmonic_gap(key):
        n = mask.sum(axis=1)
        a = np.where(mask, 1 / getattr(arr, key), 0).sum(axis=1) / n
        b = np.where(mask, 1 / getattr(base, key), 0).sum(axis=1) / n
        return float(np.max(np.abs(a / b - 1)))

    return {
        "n_active": int(mask.sum()),
        "active_mask_equal": bool(np.array_equal(arr.active_mask, mask)),
        "corr_av": float(np.corrcoef(arr.alpha[mask], arr.v[mask])[0, 1]),
        "alpha_within_sector_spread": within_sector_spread(arr.alpha),
        "v_within_sector_spread": within_sector_spread(arr.v),
        "alpha_equals_draw": bool(np.array_equal(arr.alpha[mask], base.alpha[mask])),
        "v_equals_draw": bool(np.array_equal(arr.v[mask], base.v[mask])),
        "max_rel_gap_sector_mean_inv_alpha": harmonic_gap("alpha"),
        "max_rel_gap_sector_mean_inv_v": harmonic_gap("v"),
    }


def expected_structure(label, checks):
    """hom_alpha collapses alpha only, hom_v v only, hom_both both, baseline neither."""
    common_alpha = label in ("hom_alpha", "hom_both")
    common_v = label in ("hom_v", "hom_both")
    ok = checks["active_mask_equal"]
    ok &= (checks["alpha_within_sector_spread"] == 0) if common_alpha else checks["alpha_equals_draw"]
    ok &= (checks["v_within_sector_spread"] == 0) if common_v else checks["v_equals_draw"]
    ok &= checks["max_rel_gap_sector_mean_inv_alpha"] <= 1e-12
    ok &= checks["max_rel_gap_sector_mean_inv_v"] <= 1e-12
    return bool(ok)


def table2_gate(row, ld_row):
    """Gate 1: lambda_total against the 06 checkpoint (06's comparison tolerances)."""
    full = compare_reference(row, ld_row)
    lam = full["checks"]["lambda_total"]
    return {"passed": lam["passed"], "lambda_total": lam,
            "all_common_aggregates_within_tolerance": full["passed"],
            "aggregates": full["checks"]}


def canonical_delta_gate(row, refs):
    """Gate 2, Delta form: the baseline legs against the 02/03 YAML outputs."""
    import yaml

    fixed = yaml.safe_load(refs.fixed_input_yaml.read_text())
    welfare = yaml.safe_load(refs.welfare_yaml.read_text())
    saved = {"realloc": fixed["Delta_realloc"], "materials": fixed["Delta_materials"],
             "capital": fixed["Delta_capital"], "total": fixed["Delta_total"],
             "disp": math.log1p(welfare["lambda_dispersion"]),
             "level": math.log1p(welfare["lambda_level"])}
    checks = {}
    for leg, b in saved.items():
        a = row[f"Delta_{leg}"]
        rel = abs(a - b) / abs(b)
        checks[leg] = {"fresh": a, "saved": float(b), "relative_error": rel,
                       "passed": bool(rel <= DELTA_RTOL)}
    return {"relative_tolerance": DELTA_RTOL, "passed": all(c["passed"] for c in checks.values()),
            "checks": checks}


def gate_row(label, row, *, ld_rows, refs, out, smoke):
    """Attach gates 1 and 2 to a solved row (gate 3 is enforced while solving)."""
    gates = {}
    if not smoke:
        gates["table2"] = table2_gate(row, ld_rows[label])
        if label == "baseline":
            payload = validate_baseline_safely(row, refs, out / "baseline_reference_gate")
            gates["canonical_welfare_fixed_input"] = payload
            gates["canonical_deltas"] = canonical_delta_gate(row, refs)
    gates["passed"] = all(g["passed"] for g in gates.values() if isinstance(g, dict))
    return gates


def validate_baseline_safely(row, refs, out):
    """``validate_baseline`` raises on failure; keep its payload either way."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        return validate_baseline(row, refs, out)
    except RuntimeError:
        return json.loads((out / "validation.json").read_text())


def assemble(out, rows, validation, provenance):
    """Persist the economies, the legs, the audit trail and the ready-to-paste Panel C."""
    import pandas as pd
    import yaml

    ordered = [rows[k] for k in COLUMNS]
    flat = [{k: v for k, v in r.items() if k not in ("arrangement", "gates")} for r in ordered]
    pd.DataFrame(flat).to_parquet(out / "heterogeneity_anatomy_economies.parquet", index=False)
    regimes = [{"economy": r["label"], "regime": reg,
                **{k: r[f"{k}_{reg}"] for k in ("C", "K", "M", "Q", "mu_cw", "W", "L", "converged", "residual")}}
               for r in ordered for reg in REGIMES]
    pd.DataFrame(regimes).to_csv(out / "anatomy_regimes.csv", index=False)
    legs = [{"economy": r["label"],
             **{f"{p}_{leg}": r[f"{p}_{leg}"] for leg in LEGS for p in ("Delta", "lambda")},
             **{f"share_{leg}": r[f"Delta_{leg}"] / r["Delta_total"] for leg in LEGS},
             "identity_A": r["identity_A"], "identity_B": r["identity_B"],
             **{f"converged_{reg}": r[f"converged_{reg}"] for reg in REGIMES},
             **{f"consumption_gap_{leg}": r[f"consumption_gap_{leg}"] for leg in LEGS}}
            for r in ordered]
    pd.DataFrame(legs).to_csv(out / "anatomy_legs.csv", index=False)
    panel = {k: {leg: {"Delta": rows[k][f"Delta_{leg}"],
                       "share": rows[k][f"Delta_{leg}"] / rows[k]["Delta_total"]}
                 for leg, _ in PANEL_ROWS} for k in COLUMNS}
    payload = {"provenance": provenance, "validation": validation,
               "panel_c": panel, "economies": rows}
    (out / "heterogeneity_anatomy.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    head = "| Channel | " + " | ".join(COLUMN_TITLES[k] for k in COLUMNS) + " |"
    lines = ["# Heterogeneity anatomy: Orbis Spain", "",
             "Panel C: channels of the welfare cost, Delta = ln(1 + lambda); "
             "share of each economy's total in parentheses.", "",
             head, "|---|" + "---:|" * len(COLUMNS)]
    lens_headers = {"disp": "*Lens B*", "realloc": "*Lens A*"}
    for leg, title in PANEL_ROWS:
        if leg in lens_headers:
            lines.append(f"| {lens_headers[leg]} |" + " |" * len(COLUMNS))
        cells = []
        for k in COLUMNS:
            c = panel[k][leg]
            cells.append(f"{c['Delta']:.4f}" if leg == "total"
                         else f"{c['Delta']:.4f} ({100 * c['share']:.1f}%)")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines +=["", "| lambda_total (%) | " + " | ".join(f"{100 * rows[k]['lambda_total']:.4f}"
                                                       for k in COLUMNS) + " |",
              "", "Lens B: dispersion ME -> U (U at the economy's own cost-weighted markup), level U -> PE.",
              "Lens A: misallocation ME -> PE_I (K, L, M fixed), materials PE_I -> PE_K (K, L fixed), "
              "capital PE_K -> PE.",
              "Each lens sums exactly to the total. One common draw; labor fixed at one; chi, "
              "the anchor and firm sets as in Table 2.",
              "Source numbers and checks: heterogeneity_anatomy.yaml, anatomy_legs.csv, validation.json.", ""]
    (out / "results.md").write_text("\n".join(lines))


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logger("heterogeneity_anatomy", cfg.log_level)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.smoke_M is not None and args.smoke_M < 1:
        raise ValueError("smoke-M must be positive")
    if args.smoke_M and not args.out_dir:
        raise ValueError("smoke runs require a separate explicit --out-dir")
    labels = LABELS if not args.labels else tuple(s.strip() for s in args.labels.split(",") if s.strip())
    if set(labels) - set(LABELS):
        raise ValueError(f"unknown labels {sorted(set(labels) - set(LABELS))}")
    smoke = bool(args.smoke_M)
    started = time.perf_counter()

    ld_dir = (Path(args.level_dispersion_dir).resolve() if args.level_dispersion_dir
              else cfg.out_results_dir / "level_dispersion")
    out = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir / "heterogeneity_anatomy"
    if out == ld_dir or ld_dir in out.parents:
        raise ValueError("outputs must be separate from the 06 references")
    ld_prov = json.loads((ld_dir / "run_manifest.json").read_text())["provenance"]
    economics = ld_prov["reference_economics"]
    ld_rows = {k: json.loads((ld_dir / f"checkpoint_{k}.json").read_text()) for k in LABELS}
    ld_draw = json.loads((ld_dir / "validation.json").read_text())["draw"]

    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    params_sha = file_sha256(params_path)
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, log)
    if file_sha256(params_path) != params_sha:
        raise RuntimeError("calibration file changed while it was being read")
    bundle_sha = bundle_hashes(cfg.bundle_dir)
    verify_inputs(economics, params_sha=params_sha, bundle_sha=bundle_sha, params=params, cfg=cfg)
    if params_sha != ld_prov["params_sha256"] or bundle_sha != ld_prov["bundle_sha256"]:
        raise ValueError("params or bundle differ from the 06 run")
    if ld_prov["draw_policy"] != DRAW_POLICY or ld_prov["smoke"]:
        raise ValueError("06 reference is not a production draw_pool_truncated_v1 run")
    M = args.smoke_M or int(economics["M"])
    seed = int(economics["seed"])
    chi = float(ld_rows["baseline"]["chi"])
    if any(r["chi"] != chi for r in ld_rows.values()):
        raise ValueError("06 economies do not share one chi")
    refs = ReferencePaths.from_dirs(cfg.out_results_dir, cfg.out_results_dir / "fixed_input",
                                    primitives.window_tag)
    if not smoke:
        check_reference_manifests(refs, M=M, seed=seed, params_sha256=params_sha)
    tol = float(economics["fixed_input"]["tol"])
    max_nfev = int(economics["fixed_input"]["max_nfev"])

    source_hashes = {f"level_dispersion/{p.name}": file_sha256(p) for p in
                     [ld_dir / "run_manifest.json", ld_dir / "validation.json"]
                     + [ld_dir / f"checkpoint_{k}.json" for k in LABELS]}
    source_hashes.update({Path(v).name: file_sha256(v) for v in refs.__dict__.values()
                          if Path(v).exists()})
    provenance = {"M": M, "seed": seed, "draw_policy": DRAW_POLICY, "alpha_cap": economics["alpha_cap"],
                  "params_file": str(params_path), "params_sha256": params_sha,
                  "bundle_sha256": bundle_sha, "code_sha256": code_hash(),
                  "source_hashes": source_hashes, "chi": chi, "smoke": smoke,
                  "fixed_input": {"tol": tol, "max_nfev": max_nfev},
                  "solver_market": economics["solver_market"],
                  "solver": "orbis_model.counterfactuals.sorting.sweep.solve_arrangement"}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("output contains another experiment; choose a new --out-dir")
    atomic_json_shared(manifest_path, {"fingerprint": fingerprint, "provenance": provenance})

    todo = [k for k in labels if not (out / f"checkpoint_{k}.json").exists()]
    if todo and not args.assemble_only:
        log.info("Rebuilding the reference draw (%s) at M=%s, H=%s", DRAW_POLICY, M, params.H)
        full = reference_draw(primitives.alpha_support, params, M=M, seed=seed, cap=economics["alpha_cap"])
        draw, solver_params = compact_draw(full, params)
        del full
        if not smoke:
            corr = float(np.corrcoef(draw.alpha[draw.active_mask], draw.v[draw.active_mask])[0, 1])
            if int(draw.active_mask.sum()) != ld_draw["n_active"] or solver_params.H != ld_draw["compact_width"]:
                raise RuntimeError("reconstructed firm set differs from the 06 draw")
            if not np.isclose(corr, ld_draw["corr_av"], rtol=0, atol=1e-12):
                raise RuntimeError("reconstructed alpha-v pairing differs from the 06 draw")
        market_kwargs = {**cfg.market.to_kwargs(), **economics["solver_market"],
                         "parallel_n_jobs": args.workers}
        P = cfg.parameters
        common = dict(chi=chi, beta=P.beta, phi=P.phi, delta_K=P.delta_K,
                      n_firms_cs=primitives.n_firms_cs, K_switch=cfg.participation.K_switch,
                      love_of_variety=cfg.participation.love_of_variety,
                      market_solver_kwargs=market_kwargs, wf_reference=wf, initial=initial,
                      tol=tol, max_nfev=max_nfev, alpha_max=economics["alpha_cap"], logger=log)
        arrs = arrangements(draw)
        for label in todo:
            checks = arrangement_checks(arrs[label], draw)
            if not expected_structure(label, checks):
                raise AssertionError(f"{label}: arrangement does not have the 06 structure: {checks}")
            row, _ = solve_arrangement(label, arrs[label], solver_params, **common)
            if label == "baseline" and not np.isclose(row["chi_own_ME"], chi, rtol=1e-6, atol=0) and not smoke:
                raise RuntimeError("baseline market chi differs from the frozen chi")
            row.update(arrangement=checks, M=M, seed=seed, workers=args.workers, smoke=smoke)
            row["gates"] = gate_row(label, row, ld_rows=ld_rows, refs=refs, out=out, smoke=smoke)
            atomic_json(out / f"checkpoint_{label}.json", row)
            log.info("[%s] lambda_total=%.6f disp=%.6f level=%.6f realloc=%.6f materials=%.6f "
                     "capital=%.6f gates=%s (%.1fs)", label, row["lambda_total"], row["Delta_disp"],
                     row["Delta_level"], row["Delta_realloc"], row["Delta_materials"],
                     row["Delta_capital"], row["gates"]["passed"], row["wall_s_total"])
            if not row["gates"]["passed"]:
                raise RuntimeError(f"{label}: validation gate failed; see checkpoint_{label}.json")

    missing = [k for k in LABELS if not (out / f"checkpoint_{k}.json").exists()]
    if missing:
        log.info("Waiting for %s; outputs are assembled once all four checkpoints exist", missing)
        return 0
    rows = {k: json.loads((out / f"checkpoint_{k}.json").read_text()) for k in LABELS}
    if any(r["smoke"] != smoke or r["M"] != M for r in rows.values()):
        raise RuntimeError("checkpoints come from different runs")
    validation = {
        "smoke": smoke,
        "table2_lambda": {k: rows[k]["gates"].get("table2", {}).get("lambda_total") for k in LABELS},
        "baseline_canonical": {k: v for k, v in rows["baseline"]["gates"].items()
                               if k.startswith("canonical")},
        "identity_A": {k: rows[k]["identity_A"] for k in LABELS},
        "identity_B": {k: rows[k]["identity_B"] for k in LABELS},
        "converged": {k: all(rows[k][f"converged_{reg}"] for reg in REGIMES) for k in LABELS},
        "gates_passed": {k: rows[k]["gates"]["passed"] for k in LABELS},
        "wall_s": {k: rows[k]["wall_s_total"] for k in LABELS},
        "workers": {k: rows[k]["workers"] for k in LABELS},
    }
    validation["passed"] = bool(not smoke and all(validation["gates_passed"].values())
                                and all(validation["converged"].values()))
    atomic_json_shared(out / "validation.json", validation)
    assemble(out, rows, validation, provenance)
    log.info("Wrote %s; production validation=%s; elapsed %.1fs", out, validation["passed"],
             time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
