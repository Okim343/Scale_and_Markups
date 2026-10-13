"""Step 07: assemble the Orbis Spain calibration bundle and check it with the model loader.

Ports the bundle contract of Compustat 08_assemble_bundle.py, reduced to the keys the model
reads (D11). Writes into <output.outdata_dir>:
  F_alpha.parquet         node_id, u, weight, alpha of step 01 (no sector_id)
  aggregate_moments.yaml  active_window, targets (mu_cw, cr4, cr20, top1pct, top5pct, emx_slope),
                          validation {}, a, phi_v, n_firms_cs, mean_firms_per_sector,
                          alpha_rts_gap_top5_bottom50
  manifest.yaml           provenance (commit, dirty flag, config sha256, input fingerprints and
                          upstream commits, output sha256), definitions, markets, diagnostics
Then load_bundle(outdata) from 04_Code (D12). The current loader still requires
alpha_z_copula_rho_bar and alpha_sales_corr_empirical, which this bundle drops on purpose
(04_Code is to be rewritten): that one error is logged; any other loader error fails the step.

Reads   <output.intermediary_dir>/{F_alpha.parquet, F_alpha_manifest.yaml, pooled_targets.json,
        aggregate_markup.json, emx_slope_moment.yaml, alpha_moments.json}

    python 07_assemble_bundle.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import datetime
import json
import sys

import pandas as pd
import yaml

from moments_common import (PROJECT_ROOT, dirs, file_sha256, fingerprint, git_state, load_config,
                            resolve, setup_logger)

DEFINITIONS = {
    "mu_cw": "mean over the active window of the yearly (V_D + kexp)-weighted markup_wi = theta_wi_c "
             "rev_D / V_D (DLEU, 03_Markups); V = mate + staf basis, no SG&A; use markup_target cogs",
    "cr4": "within-(year, naics4) sales share of the 4 largest firms, window mean per market, "
           "window-mean-sales-weighted across markets (Compustat 02_pooled_targets.py)",
    "cr20": "as cr4 with the 20 largest firms",
    "top1pct": "as cr4 with the largest max(1, ceil(0.01 n)) firms",
    "top5pct": "as cr4 with the largest max(1, ceil(0.05 n)) firms",
    "emx_slope": "OLS slope of the 2015-2019 change in the NAICS2 cell inverse V-weighted markup "
                 "(S1 trims) on the change in cell HHI, 14 cells, HC1 (Compustat 08b)",
    "a": "a = 1/3 from EMX (p.alpha = 1/3, capital share in value added); not estimated",
    "phi_v": "1 - mean(melast) / mean(rts) over GNR step-2 firm-years in the active window",
    "n_firms_cs": "sum over naics4 markets of the window-mean firm count in the markup sample; "
                  "literal, no rescaling",
    "mean_firms_per_sector": "window-mean-sales-weighted mean of the naics4 window-mean firm counts",
    "alpha_rts_gap_top5_bottom50": "mean within-naics4-demeaned firm-mean rts of the top 5% minus the "
                                   "bottom 50% of the within-naics4 distribution of firm-mean log "
                                   "revenue (Hubmer, Chan, Ozkan, Salgado and Hong, Table II); the "
                                   "rho_bar target",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    logger = setup_logger("bundle", config.get("log_level", "INFO"))
    inter, outd = dirs(config)
    y0, y1 = map(int, config["active_window"])

    fa_manifest = yaml.safe_load((inter / "F_alpha_manifest.yaml").read_text())
    if list(map(int, fa_manifest["settings"]["window"])) != [y0, y1]:
        raise RuntimeError(f"F_alpha was built on {fa_manifest['settings']['window']}, not {[y0, y1]}")
    conc = json.loads((inter / "pooled_targets.json").read_text())
    agg = json.loads((inter / "aggregate_markup.json").read_text())
    emx = yaml.safe_load((inter / "emx_slope_moment.yaml").read_text())
    am = json.loads((inter / "alpha_moments.json").read_text())
    for name, w in [("pooled_targets", conc["bundle"]["window"]),
                    ("aggregate_markup", agg["windows"]["active"]["window"]),
                    ("emx_slope", emx["window"]), ("alpha_moments", am["active"]["window"])]:
        if list(map(int, w)) != [y0, y1]:
            raise RuntimeError(f"{name} window {w} differs from the active window {[y0, y1]}")
    pc, a6 = conc["bundle"], am["active"]

    f_alpha = pd.read_parquet(inter / "F_alpha.parquet", columns=["node_id", "u", "weight", "alpha"])
    f_alpha.to_parquet(outd / "F_alpha.parquet", index=False)

    moments = {
        "active_window": {"key": "primary", "start": y0, "end": y1},
        "targets": {"mu_cw": agg["mu_cw"], "cr4": pc["cr4"], "cr20": pc["cr20"],
                    "top1pct": pc["top1pct"], "top5pct": pc["top5pct"], "emx_slope": emx["value"]},
        "validation": {},
        "a": float(config["a"]),
        "phi_v": a6["phi_v"],
        "n_firms_cs": pc["n_firms_cs"],
        "mean_firms_per_sector": pc["mean_firms_per_sector"],
        "alpha_rts_gap_top5_bottom50": a6["alpha_rts_gap_top5_bottom50"],
    }
    (outd / "aggregate_moments.yaml").write_text(yaml.safe_dump(moments, sort_keys=False, width=100))

    # --- Manifest ---------------------------------------------------------------------------
    commit, branch, dirty = git_state()
    mk_dir, gnr_dir = resolve(config["input"]["markups_dir"]), resolve(config["gnr"]["outdata_dir"])
    mk_manifest = yaml.safe_load((mk_dir / "manifest.yaml").read_text())
    gnr_manifest = yaml.safe_load((gnr_dir / "manifest.yaml").read_text())
    markups = pd.read_parquet(mk_dir / "firm_year_markups.parquet", columns=["year"])
    if len(markups) != mk_manifest["rows"] or {int(k): int(v) for k, v in markups.groupby(
            "year").size().items()} != mk_manifest["rows_by_year"]:
        raise RuntimeError("firm_year_markups.parquet row counts differ from its manifest")
    clean = fingerprint(resolve(config["input"]["clean_panel"]))
    if clean["sha256"] != mk_manifest["input_sha256"]:
        logger.warning("clean panel differs from the one the markups were built on")
    outputs = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
               for p in [outd / "F_alpha.parquet", outd / "aggregate_moments.yaml"]}
    n4 = conc["by_market"][config["markets"]["concentration"]]["active"]
    manifest = {
        "schema": "pooled-bundle-v1",
        "description": "Orbis Spain calibration bundle (04_Moments). Decisions D1-D13: "
                       "02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit, "branch": branch, "code_dirty": dirty,
        "pipeline_version": config["pipeline_version"],
        "config": str(config_path.resolve().relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "markup_target": config["bundle"]["markup_target_note"],
        "inputs": {
            "markups": {**fingerprint(mk_dir / "firm_year_markups.parquet"),
                        "upstream_commit": mk_manifest["commit"],
                        "upstream_code_dirty": mk_manifest["code_dirty"],
                        "upstream_config_sha256": mk_manifest["config_sha256"]},
            "gnr": {**fingerprint(gnr_dir / "firm_year_rts.parquet"),
                    "upstream_commit": gnr_manifest["commit"],
                    "upstream_code_dirty": gnr_manifest["code_dirty"],
                    "upstream_config_sha256": gnr_manifest["config_sha256"]},
            "clean_panel": {**clean, "role": "naics4 merge onto the GNR firm-years only"},
            "F_alpha": {**fingerprint(inter / "F_alpha.parquet"), "built_commit": fa_manifest["commit"],
                        "settings": fa_manifest["settings"]},
        },
        "outputs": outputs,
        "definitions": DEFINITIONS,
        "markets": {"concentration": config["markets"]["concentration"], "emx": emx["market"],
                    "n_markets": n4["n_markets"], "n_markets_emx": emx["n"]},
        "sample_sizes": {"markup_firm_years": int(len(markups)),
                         "markup_firm_years_window": int(markups["year"].between(y0, y1).sum()),
                         "gnr_firm_years_window": a6["firm_years"], "gnr_firms_window": a6["firms"],
                         "F_alpha_firm_units": fa_manifest["sample"]["firms"]},
        "diagnostics": {
            "a_gnr": a6["a_gnr"], "gnr_elasticity_means": a6["gnr_means"],
            "rts_gap_bootstrap": a6["rts_gap_bootstrap"],
            "emx_slope_se_hc1": emx["se"],
            "median_firms_per_market": pc["median_firms_per_market"],
            "naics4_multi_code_firms": am["naics4_multi_code_firms"],
        },
        "report": "moments_report.md",
    }
    (outd / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    logger.info("aggregate_moments.yaml:\n" + yaml.safe_dump(moments, sort_keys=False))

    # --- Model loader check (D12) -------------------------------------------------------------
    sys.path.insert(0, str(PROJECT_ROOT / "04_Code"))
    from steady_state.io_bundle import REQUIRED_SCALAR_KEYS, load_bundle
    dropped = [k for k in REQUIRED_SCALAR_KEYS if k not in moments]
    try:
        bundle = load_bundle(outd)
    except ValueError as e:
        # The loader's last check is the required scalars: every earlier check has passed.
        if str(e) != f"aggregate_moments.yaml: top-level scalars missing {dropped}" or not dropped:
            raise
        logger.warning(f"load_bundle: every check passes except the scalars dropped on purpose "
                       f"{dropped}; the current 04_Code loader rejects this bundle until its rewrite")
    else:
        logger.info("load_bundle OK; calibration_targets: "
                    f"{bundle.calibration_targets(markup_target=config['bundle']['markup_target_note'])}")


if __name__ == "__main__":
    main()
