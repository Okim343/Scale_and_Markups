"""Step 02 (market-free variant): untargeted validation and side-by-side comparison.

Evaluates one or more params files through the anchored GE solve the calibration
uses (``v_min`` re-solved; truncated alpha draws; model settings from the simulation
config) and tabulates, next to the data, the Section 8.7 outputs of
``ORBIS_ZIPF_CALIBRATION.md``: targeted moments, zeta at the top 1% and Hill
variants, CR4/CR20 and within-market top shares against NAICS2 and NAICS4 data,
true and common-elasticity measured markup percentiles, weighted variances of log
markups, the alpha-node composition of the top 1% of firms and the ``mu_cw`` split
into floor and share-driven parts. Also the stability check: the same params file
at a second seed (``--seed``) or a larger ``--M``.

Reads   <paths.bundle_dir>, <paths.zipf_targets>, the params files
Writes  <out>.csv and <out>.md  (default <out_results_dir>/validation_zipf)

    python 02_validate_zipf.py [--config config_estimation_zipf.yaml]
                               [--params LABEL=FILE ...] [--M N] [--seed S] [--out PREFIX]

Default ``--params``: zipf=<out_results_dir>/calibrated_params_zipf.yaml and
current=out_results/calibrated_params.yaml (the market-defined calibration,
re-evaluated under truncated draws).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from orbis_estimation.anchor_diagnostics import anchor_diagnostics
from orbis_estimation.zipf import (
    load_zipf_config, load_zipf_targets, model_kwargs, read_zipf_targets, validation_moments,
    zipf_moments_at_params,
)
from orbis_model.config import setup_logger
from orbis_model.params import check_alpha_cap_factor, load_params_file
from orbis_model.primitives import load_primitives

EST_DIR = Path(__file__).resolve().parent

TARGET_ROWS = ("mu_cw", "zeta_top5", "zeta_top2", "alpha_rts_gap_top5_bottom50")
TAIL_ROWS = ("zeta_top1", "zeta_hill_top5", "zeta_hill_top2", "zeta_hill_top1",
             "pooled_top0.1pct", "pooled_top1pct", "pooled_top5pct")
CONC_ROWS = ("cr4", "cr20", "top1pct", "top5pct", "emx_slope", "mean_active_count")
MARKUP_ROWS = tuple(f"{q}_{w}_{t}" for t in ("true", "meas") for w in ("cw", "sw", "uw") for q in ("p50", "p90", "p99")) + \
    tuple(f"{s}_{t}" for t in ("true", "meas") for s in ("mu_sw", "gap_sw_cw", "var_sw_log", "var_cw_log"))
ALPHA_ROWS = ("alpha_cap", "alpha_top_node", "share_top1pct_firms_top_node", "share_top1pct_firms_top1pct_nodes",
              "share_all_firms_top_node", "mean_alpha_top1pct_firms", "mu_sw_top1pct_firms",
              "mu_floor", "mu_cw_share_part", "mu_cw_share_part_frac", "corr_alpha_log_sales")
PARAM_ROWS = ("xi", "N", "gamma", "eta", "rho_bar", "v_min")
STATUS_ROWS = ("ge_converged", "ge_residual", "markets_converged_share", "secs")


def data_column(targets: dict, zipf: dict) -> tuple[dict, dict]:
    """(NAICS4 or market-free data, NAICS2 data) keyed by table row."""
    val = zipf["validation"]
    conc4, conc2 = val["concentration"]["naics4"], val["concentration"]["ind2d"]
    mk = val["markups"]
    main = {k: targets[k] for k in TARGET_ROWS}
    main["zeta_top1"] = targets["zeta_top1"]
    for tag in ("top5", "top2", "top1"):
        main[f"zeta_hill_{tag}"] = zipf["detail"]["normalized_ind2d"][tag]["hill"]
    main.update({k: conc4[k] for k in ("cr4", "cr20", "top1pct", "top5pct")})
    main["emx_slope"] = targets["emx_slope"]
    main["mean_active_count"] = conc4["median_firms_per_market"]
    # The data markup_wi uses one elasticity per NAICS2 cell-year: compare with the measured rows.
    for w, lab in (("cw", "cost_weighted"), ("sw", "sales_weighted"), ("uw", "unweighted")):
        for q in ("p50", "p90", "p99"):
            main[f"{q}_{w}_meas"] = mk["percentiles"][lab][q]
    for s in ("mu_sw", "gap_sw_cw", "var_sw_log", "var_cw_log"):
        main[f"{s}_meas"] = mk[s]
    naics2 = {k: conc2[k] for k in ("cr4", "cr20", "top1pct", "top5pct")}
    naics2["mean_active_count"] = conc2["median_firms_per_market"]
    return main, naics2


def evaluate(label, path, cfg, primitives, *, M, seed, logger) -> dict:
    sim = cfg.simulation
    params, _, _, data = load_params_file(path, bundle_dir=cfg.bundle_dir)
    check_alpha_cap_factor(data, sim.parameters.alpha_cap_factor, source=str(path))
    t0 = time.time()
    model, _, eq = zipf_moments_at_params(params, primitives, M=M, rng=seed, **model_kwargs(sim))
    out = dict(model)
    out.update(validation_moments(eq, params, alpha_support=primitives.alpha_support,
                                  alpha_cap_factor=sim.parameters.alpha_cap_factor))
    try:
        out["corr_alpha_log_sales"] = anchor_diagnostics(eq)["corr_alpha_log_sales"]
    except Exception:
        pass
    out.update(xi=params.xi, N=params.N, gamma=params.gamma, eta=params.eta, rho_bar=params.rho_bar,
               v_min=float(eq.v_min), ge_converged=float(bool(eq.converged)), ge_residual=float(eq.residual),
               secs=time.time() - t0)
    logger.info(f"{label}: {path} evaluated in {out['secs']:.0f} s (GE converged {bool(eq.converged)})")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_estimation_zipf.yaml")
    parser.add_argument("--params", action="append", default=None, help="LABEL=FILE, repeatable")
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="output prefix (no extension)")
    args = parser.parse_args(argv)
    cfg = load_zipf_config(args.config)
    logger = setup_logger("validate_zipf", cfg.log_level)
    primitives = load_primitives(cfg.bundle_dir, H=cfg.simulation.parameters.H)
    targets = load_zipf_targets(cfg.bundle_dir, cfg.zipf_targets)
    zipf = read_zipf_targets(cfg.zipf_targets)
    M = int(args.M or cfg.M)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    specs = args.params or [f"zipf={cfg.out_results_dir / 'calibrated_params_zipf.yaml'}",
                            f"current={EST_DIR / 'out_results' / 'calibrated_params.yaml'}"]
    columns = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        columns[label] = evaluate(label, Path(path).resolve(), cfg, primitives, M=M, seed=seed, logger=logger)
    data_main, data_naics2 = data_column(targets, zipf)
    rows = []
    for block, keys in (("target", TARGET_ROWS), ("tail", TAIL_ROWS), ("concentration", CONC_ROWS),
                        ("markups", MARKUP_ROWS), ("alpha", ALPHA_ROWS), ("params", PARAM_ROWS),
                        ("status", STATUS_ROWS)):
        for k in keys:
            row = {"block": block, "moment": k, "data": data_main.get(k), "data_naics2": data_naics2.get(k)}
            row.update({lab: col.get(k) for lab, col in columns.items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    prefix = Path(args.out).resolve() if args.out else cfg.out_results_dir / "validation_zipf"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(prefix.with_suffix(".csv"), index=False)
    header = (f"# Zipf calibration validation\n\nM = {M}, seed = {seed}, truncated alpha draws. "
              f"`data` is the market-free or NAICS4 data value (markup rows: the data markup_wi, "
              f"one elasticity per NAICS2 cell-year, to compare with the `_meas` model rows); "
              f"`data_naics2` the NAICS2 value.\n\n")
    try:
        body = table.to_markdown(index=False, floatfmt=".4g")
    except ImportError:  # tabulate missing (workstation venv): plain-text table instead
        body = "```\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n```"
    prefix.with_suffix(".md").write_text(header + body + "\n")
    logger.info("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    logger.info(f"wrote {prefix}.csv and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
