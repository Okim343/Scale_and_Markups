"""Step 08: both welfare lenses for the four economies of Table 2, each fit to the same data.

The successor of ``07_heterogeneity_anatomy.py`` for refit economies. 07 collapses
heterogeneity in one draw at the baseline parameters; here every restricted
economy has its own calibration to the baseline targets
(``04_Estimation/03_calibrate_restricted.py``) and the baseline is the canonical
calibration. Per economy:

* the inputs are rebuilt from the params file's ``restriction`` block: common v
  is ``xi = inf`` (``v = v_min``); common alpha is a one-node support at the
  harmonic mean of the ``F^alpha`` nodes below the economy's own cap
  (recomputed here and checked against the file);
* ``draw_pool`` at the welfare M and the master seed with the economy's own
  alpha cap, compacted (``level_dispersion.experiment.reference_draw`` and
  ``compact_draw``);
* ME, PE_I, PE_K, PE and U by ``sorting.sweep.solve_arrangement`` at the
  economy's own market chi. Labor is one in every regime, so
  ``Delta = ln(C_b / C_a)`` and chi only enters through ``|L - 1|``
  (the consumption-gap check bounds it).

Gates (production only, besides the identities, convergence and consumption
gaps enforced inside ``solve_arrangement``):
    1. structure: the common primitive takes one value, every alpha is below the cap;
    2. the baseline reproduces 02_welfare.py and 03_fixed_input_welfare.py
       (``validate_baseline``, relative tolerance 1e-6) and 07's baseline legs
       (relative tolerance 1e-12).

Each economy is checkpointed as it finishes, so economies can run as separate
processes (``--labels``); outputs are assembled once all four checkpoints exist.

Reads   <params files>, <paths.bundle_dir>, <paths.out_results_dir>/{welfare_*,fixed_input/*,
        heterogeneity_anatomy/checkpoint_baseline.json}
Writes  <out-dir>/checkpoint_<label>.json, run_manifest.json, validation.json,
        recalibrated_anatomy_economies.parquet, recalibrated_legs.csv,
        recalibrated_anatomy.yaml, results.md
(default <out-dir> = <paths.out_results_dir>/recalibrated_anatomy)

    python 08_recalibrated_anatomy.py --params hom_alpha=FILE --params hom_v=FILE --params hom_both=FILE
        [--params baseline=FILE] [--config FILE] [--labels a,b] [--workers N] [--assemble-only]
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
import yaml

from orbis_model.config import load_config, setup_logger
from orbis_model.counterfactuals.level_dispersion.experiment import compact_draw, reference_draw
from orbis_model.counterfactuals.sorting.sweep import (
    LEGS, REGIMES, ReferencePaths, atomic_json, check_reference_manifests, solve_arrangement,
    validate_baseline,
)
from orbis_model.model.pool import alpha_cap
from orbis_model.params import bundle_hashes, file_sha256, load_simulation_inputs

COLUMNS = ("hom_both", "hom_alpha", "hom_v", "baseline")
COLUMN_TITLES = {"hom_both": "Homogeneous", "hom_alpha": "ν only", "hom_v": "α only", "baseline": "Calibrated"}
RESTRICTION = {"hom_both": (True, True), "hom_alpha": (True, False), "hom_v": (False, True),
               "baseline": (False, False)}            # (common alpha, common v)
PANEL_ROWS = (("disp", "Dispersion"), ("level", "Level"), ("realloc", "Misallocation"),
              ("materials", "Materials"), ("capital", "Capital"), ("total", "Total"))
ANATOMY_RTOL = 1e-12
DELTA_RTOL = 1e-6


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config")
    p.add_argument("--params", action="append", default=[], metavar="LABEL=FILE",
                   help="one per restricted economy; baseline defaults to paths.params_file")
    p.add_argument("--anatomy-dir", help="07 outputs (default: <out_results_dir>/heterogeneity_anatomy)")
    p.add_argument("--out-dir")
    p.add_argument("--labels", help=f"comma-separated subset of {','.join(COLUMNS)} (default: all)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--assemble-only", action="store_true", help="solve nothing; assemble checkpoints")
    p.add_argument("--smoke-M", type=int, help="small functional run; no production reference gate")
    return p


def common_alpha(support, gamma, factor):
    """Harmonic mean of the ``F^alpha`` nodes below the cap (as ``orbis_estimation.restricted``)."""
    nodes = np.sort(np.asarray(support, dtype=np.float64))
    cap = alpha_cap(gamma, factor)
    if cap is not None:
        nodes = nodes[nodes < cap]
    return float(1.0 / np.mean(1.0 / nodes))


def economy_support(label, params, data, primitives, factor):
    """The alpha support the economy draws from, after checking its file implements ``label``."""
    common_a, common_v = RESTRICTION[label]
    restriction = data.get("restriction")
    if label == "baseline":
        if restriction is not None:
            raise ValueError("the baseline params file carries a restriction block")
        return primitives.alpha_support, None
    if restriction is None or restriction.get("economy") != label:
        raise ValueError(f"{label}: params file is not a {label} calibration (restriction {restriction})")
    if (bool(restriction["common_alpha"]), bool(restriction["common_v"])) != (common_a, common_v):
        raise ValueError(f"{label}: restriction block {restriction} does not match the economy")
    if (common_a or common_v) and params.rho_bar != 0.0:
        raise ValueError(f"{label}: rho_bar must be pinned at 0")
    if common_v and not math.isinf(params.xi):
        raise ValueError(f"{label}: common v needs xi = inf")
    if not common_a:
        return primitives.alpha_support, None
    node = common_alpha(primitives.alpha_support, params.gamma, factor)
    if node != float(restriction["alpha_common"]):
        raise ValueError(f"{label}: common alpha {node!r} differs from the file's {restriction['alpha_common']!r}")
    return np.array([node]), node


def structure_checks(label, draw, cap, node):
    common_a, common_v = RESTRICTION[label]
    mask = draw.active_mask
    a, v = draw.alpha[mask], draw.v[mask]
    out = {"n_active": int(mask.sum()), "alpha_unique": int(np.unique(a).size), "v_unique": int(np.unique(v).size),
           "alpha_below_cap": bool(np.all(a < cap)), "alpha_common": node}
    ok = out["alpha_below_cap"]
    ok &= (out["alpha_unique"] == 1 and bool(np.all(a == node))) if common_a else out["alpha_unique"] > 1
    ok &= out["v_unique"] == 1 if common_v else out["v_unique"] > 1
    out["passed"] = bool(ok)
    return out


def markup_stats(me, gamma):
    """Markup dispersion and concentration of the market allocation (active firms, pooled)."""
    sol, mask = me.participation.solution, np.asarray(me.participation.active_mask, dtype=bool)
    S, C, MU = (np.asarray(x, dtype=float)[mask] for x in (sol.sales, sol.cost, sol.mu))
    s, c, lm = S / S.sum(), C / C.sum(), np.log(MU)
    r = np.sort(S)[::-1]
    return {"var_sw_log_mu_ME": float(s @ (lm - s @ lm) ** 2), "var_cw_log_mu_ME": float(c @ (lm - c @ lm) ** 2),
            "mu_sw_ME": float(s @ MU), "mu_floor": float(gamma / (gamma - 1.0)),
            "pooled_top1pct_ME": float(r[:max(1, int(0.01 * r.size))].sum() / r.sum())}


def anatomy_gate(row, path):
    """Gate 2b: the baseline legs and regime aggregates equal 07's baseline checkpoint."""
    saved = json.loads(Path(path).read_text())
    keys = ([f"Delta_{leg}" for leg in LEGS] + [f"lambda_{leg}" for leg in LEGS]
            + [f"{k}_{reg}" for reg in REGIMES for k in ("C", "K", "M", "Q", "W", "mu_cw")])
    checks = {}
    for key in keys:
        a, b = float(row[key]), float(saved[key])
        rel = abs(a - b) / max(abs(b), 1e-300)
        checks[key] = {"fresh": a, "saved": b, "relative_error": rel, "passed": bool(rel <= ANATOMY_RTOL)}
    return {"relative_tolerance": ANATOMY_RTOL, "source": str(path),
            "max_relative_error": max(c["relative_error"] for c in checks.values()),
            "passed": all(c["passed"] for c in checks.values()), "checks": checks}


def canonical_delta_gate(row, refs):
    """Gate 2c, Delta form: the baseline legs against the 02/03 YAML outputs (as 07)."""
    fixed = yaml.safe_load(refs.fixed_input_yaml.read_text())
    welfare = yaml.safe_load(refs.welfare_yaml.read_text())
    saved = {"realloc": fixed["Delta_realloc"], "materials": fixed["Delta_materials"],
             "capital": fixed["Delta_capital"], "total": fixed["Delta_total"],
             "disp": math.log1p(welfare["lambda_dispersion"]), "level": math.log1p(welfare["lambda_level"])}
    checks = {}
    for leg, b in saved.items():
        a = row[f"Delta_{leg}"]
        rel = abs(a - b) / abs(b)
        checks[leg] = {"fresh": a, "saved": float(b), "relative_error": rel, "passed": bool(rel <= DELTA_RTOL)}
    return {"relative_tolerance": DELTA_RTOL, "passed": all(c["passed"] for c in checks.values()), "checks": checks}


def validate_baseline_safely(row, refs, out):
    """``validate_baseline`` raises on failure; keep its payload either way."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        return validate_baseline(row, refs, out)
    except RuntimeError:
        return json.loads((out / "validation.json").read_text())


def decompose(rows):
    """Finite differences of percentage changes across the refit economies, with an exact total."""
    result = {}
    for metric in ("Y_pct", "K_pct", "welfare_pct"):
        b, a, v, h = (float(rows[k][metric]) for k in ("baseline", "hom_alpha", "hom_v", "hom_both"))
        comp = {"common": h, "capability": a - h, "scalability": v - h, "interaction": b - a - v + h, "total": b}
        residual = comp["common"] + comp["capability"] + comp["scalability"] + comp["interaction"] - b
        if abs(residual) > 1e-10:
            raise AssertionError("percentage-point decomposition does not add up")
        result[metric] = {**comp, "identity_residual": residual}
    return result


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


def _f(x, fmt="{:.4f}"):
    return "inf" if isinstance(x, float) and math.isinf(x) else ("" if x is None else fmt.format(x))


def assemble(out, rows, validation, provenance):
    """Persist the economies, the legs, the audit trail and the ready-to-paste panels."""
    import pandas as pd

    ordered = [rows[k] for k in COLUMNS]
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in ordered]
    pd.DataFrame(flat).to_parquet(out / "recalibrated_anatomy_economies.parquet", index=False)
    legs = [{"economy": r["label"],
             **{f"{p}_{leg}": r[f"{p}_{leg}"] for leg in LEGS for p in ("Delta", "lambda")},
             **{f"share_{leg}": r[f"Delta_{leg}"] / r["Delta_total"] for leg in LEGS},
             "identity_A": r["identity_A"], "identity_B": r["identity_B"]} for r in ordered]
    pd.DataFrame(legs).to_csv(out / "recalibrated_legs.csv", index=False)
    split = decompose(rows)
    panel_c = {k: {leg: {"Delta": rows[k][f"Delta_{leg}"], "share": rows[k][f"Delta_{leg}"] / rows[k]["Delta_total"]}
                   for leg, _ in PANEL_ROWS} for k in COLUMNS}
    payload = {"provenance": provenance, "validation": validation, "decomposition": split,
               "panel_c": panel_c, "economies": rows}
    (out / "recalibrated_anatomy.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    head = "| | " + " | ".join(COLUMN_TITLES[k] for k in COLUMNS) + " |"
    rule = "|---|" + "---:|" * len(COLUMNS)
    lines = ["# Heterogeneity anatomy, refit economies: Orbis Spain", "",
             "Panel A: each economy refit to the baseline targets; planner vs its own market (% change).", "",
             head, rule]
    for key, title, fmt in (("xi", "ξ", "{:.3f}"), ("N", "N", "{:.1f}"), ("gamma", "γ", "{:.4f}"),
                            ("alpha_common", "common α", "{:.4f}"), ("mu_cw_ME", "μ_cw", "{:.4f}"),
                            ("top_share_median_ME", "median top share", "{:.4f}"),
                            ("var_sw_log_mu_ME", "Var_λ(log μ)", "{:.5f}"), ("Y_pct", "Y", "{:.2f}"),
                            ("K_pct", "K", "{:.2f}"), ("welfare_pct", "Welfare λ", "{:.2f}")):
        cells = ["∞" if key == "xi" and rows[k].get(key) is None else _f(rows[k].get(key), fmt) for k in COLUMNS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "Panel B: contributions to the calibrated gains (pp; share of the total).", "",
              "| Component | Y | K | Welfare λ |", "|---|---:|---:|---:|"]
    for key, title in (("common", "Common level (both homogeneous)"), ("capability", "Capability heterogeneity"),
                       ("scalability", "Scalability heterogeneity"), ("interaction", "Joint interaction"),
                       ("total", "Total (calibrated)")):
        cells = [f"{split[m][key]:.2f} ({100 * split[m][key] / split[m]['total']:.1f}%)"
                 for m in ("Y_pct", "K_pct", "welfare_pct")]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "Panel C: channels of the welfare cost, Delta = ln(1 + λ); share of each economy's total.", "",
              "| Channel | " + " | ".join(COLUMN_TITLES[k] for k in COLUMNS) + " |", rule]
    for leg, title in PANEL_ROWS:
        if leg in ("disp", "realloc"):
            lines.append(f"| *Lens {'B' if leg == 'disp' else 'A'}* |" + " |" * len(COLUMNS))
        cells = [f"{panel_c[k][leg]['Delta']:.4f}" if leg == "total"
                 else f"{panel_c[k][leg]['Delta']:.4f} ({100 * panel_c[k][leg]['share']:.1f}%)" for k in COLUMNS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "Panel D: fit (model value; * not targeted by that economy, which cannot reach it). "
              "The fit cost is the weighted objective on the economy's own targets.", "",
              "| Moment | Target | " + " | ".join(COLUMN_TITLES[k] for k in COLUMNS) + " |",
              "|---|---:|" + "---:|" * len(COLUMNS)]
    for moment in rows["baseline"]["fit_moments"]:
        target = rows["baseline"]["fit_moments"][moment]["target"]
        lines.append(f"| {moment} | {target:.4f} | "
                     + " | ".join(f"{rows[k]['fit_moments'][moment]['model']:.4f}"
                                  + ("" if rows[k]["fit_moments"][moment]["targeted"] else "*") for k in COLUMNS)
                     + " |")
    lines.append("| fit cost | | " + " | ".join(f"{rows[k]['fit_cost']:.3g}" for k in COLUMNS) + " |")
    lines += ["", "Lens B: dispersion ME -> U (U at the economy's own cost-weighted markup), level U -> PE.",
              "Lens A: misallocation ME -> PE_I (K, L, M fixed), materials PE_I -> PE_K (K, L fixed), "
              "capital PE_K -> PE.",
              "Each lens sums exactly to the total. Labor is one; each economy uses its own draw, anchor and chi.",
              "Source numbers and checks: recalibrated_anatomy.yaml, recalibrated_legs.csv, validation.json.", ""]
    (out / "results.md").write_text("\n".join(lines))


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logger("recalibrated_anatomy", cfg.log_level)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.smoke_M and not args.out_dir:
        raise ValueError("smoke runs require a separate explicit --out-dir")
    labels = COLUMNS if not args.labels else tuple(s.strip() for s in args.labels.split(",") if s.strip())
    if set(labels) - set(COLUMNS):
        raise ValueError(f"unknown labels {sorted(set(labels) - set(COLUMNS))}")
    smoke = bool(args.smoke_M)
    started = time.perf_counter()

    paths = {"baseline": cfg.params_file}
    for item in args.params:
        label, _, path = item.partition("=")
        if label not in COLUMNS or not path:
            raise ValueError(f"--params expects LABEL=FILE with LABEL in {COLUMNS}; got {item!r}")
        paths[label] = Path(path)
    paths = {k: Path(v).resolve() for k, v in paths.items()}
    missing = [k for k in COLUMNS if k not in paths]
    if missing:
        raise ValueError(f"no params file for {missing}")
    shas = {k: file_sha256(p) for k, p in paths.items()}

    out = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir / "recalibrated_anatomy"
    anatomy_dir = (Path(args.anatomy_dir).resolve() if args.anatomy_dir
                   else cfg.out_results_dir / "heterogeneity_anatomy")
    P = cfg.parameters
    M = args.smoke_M or cfg.welfare_M
    seed = int(cfg.monte_carlo.master_seed)
    fixed_cfg = cfg.raw.get("fixed_input", {}) or {}
    tol, max_nfev = float(fixed_cfg.get("tol", 1e-5)), int(fixed_cfg.get("max_nfev", 40))
    bundle_sha = bundle_hashes(cfg.bundle_dir)
    provenance = {"M": M, "seed": seed, "smoke": smoke, "alpha_cap_factor": P.alpha_cap_factor,
                  "params_files": {k: {"path": str(paths[k]), "sha256": shas[k]} for k in COLUMNS},
                  "bundle_sha256": bundle_sha, "code_sha256": code_hash(),
                  "fixed_input": {"tol": tol, "max_nfev": max_nfev}, "solver_market": cfg.market.to_kwargs(),
                  "chi": "own market chi per economy",
                  "solver": "orbis_model.counterfactuals.sorting.sweep.solve_arrangement"}
    provenance["solver_market"].pop("parallel_n_jobs", None)
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, default=str).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("output contains another experiment; choose a new --out-dir")
    atomic_json_shared(manifest_path, {"fingerprint": fingerprint, "provenance": provenance})

    todo = [k for k in labels if not (out / f"checkpoint_{k}.json").exists()]
    if todo and not args.assemble_only:
        market_kwargs = {**cfg.market.to_kwargs(), "parallel_n_jobs": args.workers}
        for label in todo:
            primitives, params, wf, initial = load_simulation_inputs(cfg, paths[label], log)
            if file_sha256(paths[label]) != shas[label]:
                raise RuntimeError(f"{paths[label]} changed while it was being read")
            data = yaml.safe_load(paths[label].read_text())
            refs = ReferencePaths.from_dirs(cfg.out_results_dir, cfg.out_results_dir / "fixed_input",
                                            primitives.window_tag)
            if label == "baseline" and not smoke:
                check_reference_manifests(refs, M=M, seed=seed, params_sha256=shas[label])
            cap = alpha_cap(params.gamma, P.alpha_cap_factor)
            support, node = economy_support(label, params, data, primitives, P.alpha_cap_factor)
            log.info("[%s] drawing at M=%s (cap %.6f, %s support nodes)", label, M, cap, support.size)
            draw, solver_params = compact_draw(reference_draw(support, params, M=M, seed=seed, cap=cap), params)
            checks = structure_checks(label, draw, cap, node)
            if not checks["passed"]:
                raise AssertionError(f"{label}: draw does not implement the economy: {checks}")
            row, me = solve_arrangement(
                label, draw, solver_params, chi=None, beta=P.beta, phi=P.phi, delta_K=P.delta_K,
                n_firms_cs=primitives.n_firms_cs, K_switch=cfg.participation.K_switch,
                love_of_variety=cfg.participation.love_of_variety, market_solver_kwargs=market_kwargs,
                wf_reference=wf, initial=initial, tol=tol, max_nfev=max_nfev, alpha_max=cap, logger=log)
            row.update(markup_stats(me, params.gamma))
            del me
            row.update(
                xi=float(params.xi), N=float(params.N), gamma=float(params.gamma), rho_bar=float(params.rho_bar),
                alpha_common=node, alpha_cap=float(cap), structure=checks, M=M, seed=seed,
                workers=args.workers, smoke=smoke,
                Y_pct=100 * (row["Q_PE"] / row["Q_ME"] - 1), K_pct=100 * (row["K_PE"] / row["K_ME"] - 1),
                welfare_pct=100 * row["lambda_total"],
                fit_moments={
                    **{k: {"model": float(v["model"]), "target": float(v["target"]), "targeted": True}
                       for k, v in data["fit"]["moments"].items()},
                    **{k: {"model": float(v["model"]), "target": float(v["target"]), "targeted": False}
                       for k, v in (data["fit"].get("missed_baseline_targets") or {}).items()}},
                fit_cost=float(data["fit"]["cost"]), fit_success=bool(data["fit"]["success"]),
                fit_at_bound=list(data["fit"].get("at_bound", [])),
            )
            if math.isinf(row["xi"]):
                row["xi"] = None   # JSON has no inf; null means common v (xi = inf)
            gates = {"structure": checks}
            if label == "baseline" and not smoke:
                gates["canonical_welfare_fixed_input"] = validate_baseline_safely(row, refs, out / "baseline_reference_gate")
                gates["canonical_deltas"] = canonical_delta_gate(row, refs)
                gates["anatomy_07"] = anatomy_gate(row, anatomy_dir / "checkpoint_baseline.json")
            gates["passed"] = all(g["passed"] for g in gates.values())
            row["gates"] = gates
            atomic_json(out / f"checkpoint_{label}.json", row)
            log.info("[%s] lambda_total=%.6f disp=%.6f level=%.6f realloc=%.6f materials=%.6f capital=%.6f "
                     "mu_cw=%.4f gates=%s (%.1fs)", label, row["lambda_total"], row["Delta_disp"], row["Delta_level"],
                     row["Delta_realloc"], row["Delta_materials"], row["Delta_capital"], row["mu_cw_ME"],
                     gates["passed"], row["wall_s_total"])
            if not gates["passed"]:
                raise RuntimeError(f"{label}: validation gate failed; see checkpoint_{label}.json")

    missing = [k for k in COLUMNS if not (out / f"checkpoint_{k}.json").exists()]
    if missing:
        log.info("Waiting for %s; outputs are assembled once all four checkpoints exist", missing)
        return 0
    rows = {k: json.loads((out / f"checkpoint_{k}.json").read_text()) for k in COLUMNS}
    if any(r["smoke"] != smoke or r["M"] != M for r in rows.values()):
        raise RuntimeError("checkpoints come from different runs")
    validation = {
        "smoke": smoke,
        "baseline": {k: v for k, v in rows["baseline"]["gates"].items() if k not in ("structure", "passed")},
        "structure": {k: rows[k]["gates"]["structure"] for k in COLUMNS},
        "identity_A": {k: rows[k]["identity_A"] for k in COLUMNS},
        "identity_B": {k: rows[k]["identity_B"] for k in COLUMNS},
        "converged": {k: all(rows[k][f"converged_{reg}"] for reg in REGIMES) for k in COLUMNS},
        "gates_passed": {k: rows[k]["gates"]["passed"] for k in COLUMNS},
        "fit": {k: {"cost": rows[k]["fit_cost"], "success": rows[k]["fit_success"],
                    "at_bound": rows[k]["fit_at_bound"]} for k in COLUMNS},
        "wall_s": {k: rows[k]["wall_s_total"] for k in COLUMNS},
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
