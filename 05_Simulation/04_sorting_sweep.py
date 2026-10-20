"""Step 04: welfare legs across alpha<->v sorting arrangements (checkpointed, resumable).

Ported from ``04_Code/counterfactuals/sorting_curve/run_sorting_curve.py``. On
the draws, seed and exogenous Poisson mask of ``02_welfare.py`` it solves, for
every arrangement, ME, PE_I, PE_K, PE (``solve_fixed_input_chain``) and the
uniform-markup U (``two_channel_decomposition``), and reports Lens A (realloc,
materials, capital) and Lens B (disp, level). Arrangements, in plan order:

    baseline             the calibrated draw (rho_bar from the params file)
    rho_<r>              copula re-draws at rho_bar = r, same seed (alpha, tilde_alpha,
                         eps, active_mask common; asserted bitwise)
    reverse, sorted      within-sector extreme pairings (bitwise marginals)
    shuffle_<k>          within-sector random permutations (bitwise marginals)
    hom_sector, hom_cost homogeneous alpha: sector harmonic mean; one cost-weighted alpha

chi is frozen at the baseline market value; v_min and y_hat at the calibration.

Baseline gate: before any other arrangement, the baseline must reproduce the
``02_welfare.py`` and ``03_fixed_input_welfare.py`` outputs at the same M, seed
and params file to relative tolerance 1e-6 (``validation.json``). Their
manifests are checked before anything is solved.

Reads   <paths.bundle_dir>, <paths.params_file>,
        <welfare-ref-dir>/{welfare,welfare_decomposition,welfare_manifest}_<window>.*
        <fixed-input-ref-dir>/{fixed_input_welfare,fixed_input_regimes,fixed_input_manifest}_<window>.*
Writes  <out-dir>/checkpoint_<label>.json   one per solved arrangement (atomic)
        <out-dir>/run_manifest.json          fingerprint checked on resume
        <out-dir>/validation.json            baseline gate
        <out-dir>/sorting_economies.parquet  one row per arrangement
        <out-dir>/sorting.yaml               provenance and summary
        <out-dir>/results.md                 readout citing file keys
        <fig-dir>/sorting_curve.{pdf,png}
(defaults: <out-dir> = <paths.out_results_dir>/sorting, <fig-dir> = <paths.out_figs_dir>/sorting)

    python 04_sorting_sweep.py [--config FILE] [--params FILE] [--M N] [--seed S]
        [--out-dir D] [--fig-dir D] [--welfare-ref-dir D] [--fixed-input-ref-dir D]
        [--grid r1,r2,...] [--n-shuffle K] [--no-homogeneous] [--labels a,b] [--validation-only]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from orbis_model.config import load_config, setup_logger
from orbis_model.counterfactuals.sorting import (
    COPULA_GRID,
    ReferencePaths,
    assert_copula_crn,
    build_plan,
    check_reference_manifests,
    homogenize_alpha_cost_weighted,
    homogenize_alpha_sector,
    permute_alpha,
    solve_arrangement,
    validate_baseline,
)
from orbis_model.counterfactuals.sorting.readout import write_artifacts
from orbis_model.counterfactuals.sorting.sweep import atomic_json, shuffle_seed
from orbis_model.model.pool import alpha_cap, draw_pool
from orbis_model.params import bundle_hashes, file_sha256, load_simulation_inputs

SIM_DIR = Path(__file__).resolve().parent
PKG = SIM_DIR / "orbis_model"
# Code whose edits change solved numbers; a resume refuses if any of it changed.
FINGERPRINT_CODE = (
    sorted((PKG / "model").glob("*.py")) + sorted((PKG / "welfare").glob("*.py"))
    + [PKG / "counterfactuals" / "sorting" / "arrangements.py",
       PKG / "counterfactuals" / "sorting" / "sweep.py", Path(__file__).resolve()]
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="default: config_simulation.yaml")
    p.add_argument("--params", default=None, help="override paths.params_file")
    p.add_argument("--M", type=int, default=None, help="override welfare.M")
    p.add_argument("--seed", type=int, default=None, help="override monte_carlo.master_seed")
    p.add_argument("--out-dir", default=None, help="default: <paths.out_results_dir>/sorting")
    p.add_argument("--fig-dir", default=None, help="default: <paths.out_figs_dir>/sorting")
    p.add_argument("--welfare-ref-dir", default=None,
                   help="02_welfare.py outputs (default: <paths.out_results_dir>)")
    p.add_argument("--fixed-input-ref-dir", default=None,
                   help="03_fixed_input_welfare.py outputs (default: <paths.out_results_dir>/fixed_input)")
    p.add_argument("--grid", default=None,
                   help="comma-separated copula rho_bar values (default: %s)"
                        % ",".join(f"{r:g}" for r in COPULA_GRID))
    p.add_argument("--n-shuffle", type=int, default=2, help="shuffle replicates")
    p.add_argument("--no-homogeneous", action="store_true", help="skip hom_sector and hom_cost")
    p.add_argument("--labels", default=None,
                   help="comma-separated subset of the plan to run (baseline always runs first)")
    p.add_argument("--validation-only", action="store_true", help="stop after the baseline gate")
    return p


def _sha_files(paths) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).name.encode())
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logger("sorting", cfg.log_level)
    t_start = time.time()
    params_path = Path(args.params).resolve() if args.params else cfg.params_file
    primitives, params, wf, initial = load_simulation_inputs(cfg, params_path, logger)
    M = int(args.M or cfg.welfare_M)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    tag = primitives.window_tag
    out = Path(args.out_dir).resolve() if args.out_dir else cfg.out_results_dir / "sorting"
    fig_dir = Path(args.fig_dir).resolve() if args.fig_dir else cfg.out_figs_dir / "sorting"
    refs = ReferencePaths.from_dirs(
        Path(args.welfare_ref_dir).resolve() if args.welfare_ref_dir else cfg.out_results_dir,
        Path(args.fixed_input_ref_dir).resolve() if args.fixed_input_ref_dir
        else cfg.out_results_dir / "fixed_input",
        tag,
    )
    params_sha = file_sha256(params_path)
    check_reference_manifests(refs, M=M, seed=seed, params_sha256=params_sha)

    P = cfg.parameters
    fixed_cfg = cfg.raw.get("fixed_input", {}) or {}
    tol = float(fixed_cfg.get("tol", 1e-5))
    max_nfev = int(fixed_cfg.get("max_nfev", 40))
    cap = alpha_cap(params.gamma, P.alpha_cap_factor)

    def make_draw(rho: float):
        # Every calibrated draw parameter comes from params; only rho_bar varies.
        return draw_pool(primitives.alpha_support, xi=params.xi, rho_bar=rho, N=params.N,
                         M=M, H=params.H, rng=seed, v_min=params.v_min, alpha_max=cap)

    grid = (tuple(float(x) for x in args.grid.split(",") if x.strip()) if args.grid is not None
            else COPULA_GRID)
    plan = build_plan(params.rho_bar, grid=grid, n_shuffle=args.n_shuffle,
                      homogeneous=not args.no_homogeneous)
    if args.labels:
        keep = {s.strip() for s in args.labels.split(",")} | {"baseline"}
        unknown = keep - {a.label for a in plan}
        if unknown:
            raise ValueError(f"unknown labels {sorted(unknown)}; plan has {[a.label for a in plan]}")
        plan = [a for a in plan if a.label in keep]

    raw = cfg.raw
    economics = {
        "M": M, "seed": seed, "window": tag,
        "params_sha256": params_sha, "bundle_sha256": bundle_hashes(cfg.bundle_dir),
        "alpha_cap": cap, "alpha_cap_factor": P.alpha_cap_factor,
        "parameters": raw.get("parameters"), "participation": raw.get("participation"),
        "solver_market": (raw.get("solver") or {}).get("market"), "fixed_input": {"tol": tol, "max_nfev": max_nfev},
        "code_sha256": _sha_files(FINGERPRINT_CODE),
    }
    fingerprint = hashlib.sha256(json.dumps(economics, sort_keys=True, default=str).encode()).hexdigest()
    provenance = {
        **economics, "fingerprint": fingerprint,
        "rho_bar_calibrated": float(params.rho_bar),
        "calibrated_params": dataclasses.asdict(params),
        "params_file": str(params_path), "config": str(cfg.path),
        "smoke": M < cfg.welfare_M,
        "plan": [dataclasses.asdict(a) for a in plan],
        "references": {k: str(v) for k, v in refs.__dict__.items()},
        "design": ("copula re-draws over rho_bar on common alpha/tilde_alpha/eps/active_mask; "
                   "within-sector permutations with bitwise marginals; homogeneous alpha; "
                   "chi frozen at the baseline market value; v_min and y_hat at the calibration"),
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError(f"{out} holds checkpoints from different inputs or code; "
                           "choose a new --out-dir")
    atomic_json(manifest, {"fingerprint": fingerprint, "economics": economics})

    base = make_draw(params.rho_bar)
    common = dict(
        beta=P.beta, phi=P.phi, delta_K=P.delta_K, n_firms_cs=primitives.n_firms_cs,
        K_switch=cfg.participation.K_switch, love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf, initial=initial,
        tol=tol, max_nfev=max_nfev, alpha_max=cap, logger=logger,
    )
    rows: list[dict] = []
    baseline_row: dict | None = None
    for idx, arr in enumerate(plan):
        checkpoint = out / f"checkpoint_{arr.label}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text())
            logger.info(f"[{arr.label}] loaded checkpoint")
            if arr.label == "baseline":
                validate_baseline(row, refs, out)
        else:
            perm_seed = None
            if arr.mode == "identity":
                draw = base
            elif arr.mode == "copula":
                draw = make_draw(arr.rho_bar)
                assert_copula_crn(draw, base)
            elif arr.mode == "shuffle":
                perm_seed = shuffle_seed(seed, arr.replicate)
                draw = permute_alpha(base, "shuffle", np.random.default_rng(perm_seed))
            elif arr.mode in ("reverse", "sorted"):
                draw = permute_alpha(base, arr.mode)
            elif arr.mode == "sector":
                draw = homogenize_alpha_sector(base)
            elif arr.mode == "cost_weighted":
                draw = homogenize_alpha_cost_weighted(base, baseline_row["alpha_hom_cost_weighted"])
            else:
                raise ValueError(f"unknown arrangement mode {arr.mode!r}")
            chi = None if arr.label == "baseline" else baseline_row["chi"]
            row, _ = solve_arrangement(arr.label, draw, params, chi=chi, **common)
            row.update(kind=arr.kind, mode=arr.mode,
                       rho_bar_draw=float(arr.rho_bar if arr.kind == "copula" else params.rho_bar),
                       replicate=arr.replicate, perm_seed=None if perm_seed is None else json.dumps(perm_seed),
                       draw_seed=seed, M=M, source="solved")
            if arr.label == "baseline":
                validate_baseline(row, refs, out)
            atomic_json(checkpoint, row)
        if arr.label == "baseline":
            baseline_row = row
        rows.append(row)
        frame = pd.DataFrame(rows)
        tmp = out / "sorting_economies.tmp.parquet"
        frame.to_parquet(tmp, index=False)
        tmp.replace(out / "sorting_economies.parquet")
        last = idx == len(plan) - 1 or args.validation_only
        write_artifacts(frame, provenance, out, fig_dir, complete=idx == len(plan) - 1)
        logger.info(f"[{arr.label}] corr={row['corr_av']} lambda_total={row['lambda_total']:.6f} "
                    f"disp={row['lambda_disp']:.6f} realloc={row['lambda_realloc']:.6f} "
                    f"wall={row.get('wall_s_total')}s")
        if last:
            break
    logger.info(f"wrote {out} ({time.time() - t_start:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
