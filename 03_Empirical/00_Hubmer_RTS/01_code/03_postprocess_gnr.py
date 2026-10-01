"""
Stage R3 — Post-process GNR estimates into a firm-year RTS panel.

Reads:
  02_intermediary/gnr_panel_trimmed.parquet
  02_intermediary/step1_results.json
  02_intermediary/step2_results.json

Writes:
  02_intermediary/firm_year_rts_raw.parquet   (all trimmed rows; Step-2
                                               objects NaN outside Step-2 sample)
  03_outdata/firm_year_rts.parquet            (rows with finite RTS only)

Reconstructs, deterministically from the saved parameters (no re-optimization):
  eps, melast                       on the full trimmed sample
  omega, lomega, eta,
  kelast, lelast, rts               on the Step-2 sample
  alpha_gross_output = rts          (model-language alias, exported field)

Run:
  python 03_postprocess_gnr.py [--config config_smoke.yaml]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import gnr_model as gm
from utils import (
    PATHS,
    ensure_dir,
    intermediary_dir,
    outdata_dir,
    parse_config_arg,
    setup_logger,
)

TRIM_IN = "gnr_panel_trimmed.parquet"
RAW_OUT = "firm_year_rts_raw.parquet"
FINAL_OUT = "firm_year_rts.parquet"

EXPORT_COLS = [
    "gvkey", "year", "ind2d",
    "eps", "melast", "kelast", "lelast",
    "omega", "lomega", "eta",
    "rts", "alpha_gross_output",
    "in_step2_sample", "rts_outlier_flag",
]


def main() -> None:
    config = parse_config_arg("Stage R3 — post-process GNR firm-year outputs")
    logger = setup_logger("R3", config.get("log_level", "INFO"))
    inter = intermediary_dir(config)

    # --- load inputs -----------------------------------------------------------
    for fname in [TRIM_IN, "step1_results.json", "step2_results.json"]:
        if not (inter / fname).exists():
            raise FileNotFoundError(
                f"{inter / fname} missing; run stages R1/R2 first."
            )
    est = pd.read_parquet(inter / TRIM_IN)
    with open(inter / "step1_results.json") as f:
        s1 = json.load(f)
    with open(inter / "step2_results.json") as f:
        s2 = json.load(f)
    logger.info(f"loaded trimmed panel ({len(est):,} rows) and step 1/2 results")

    soln = np.asarray(s1["gamma_raw"])
    gamma_scaled = np.asarray(s1["gamma_scaled"])
    a_hat = np.asarray(s2["alpha"])
    delta = np.asarray(s2["delta"])
    h2size = int(s2["h2size"])
    require_wages = bool(s2.get("require_wages", False))

    # --- Step 1 objects on the full trimmed sample ------------------------------
    k, m, l = (est[c].to_numpy() for c in ["k", "m", "l"])
    p1 = gm.build_p1(k, m, l)
    est["eps"] = gm.step1_eps(soln, est["s"].to_numpy(), p1)
    est["melast"] = gm.melast(p1, gamma_scaled)

    # --- Step 2 selection (same rule as estimation; validated against R2) -------
    lp1 = gm.build_p1(est["lk"].to_numpy(), est["lm"].to_numpy(), est["ll"].to_numpy())
    sel = (lp1 @ soln) > 0
    if require_wages:
        sel &= est["lw"].notna().to_numpy()
    n_sel = int(sel.sum())
    n_expected = int(s2["sample_counts"]["n_step2"])
    if n_sel != n_expected:
        raise RuntimeError(
            f"Step 2 sample reconstruction mismatch: {n_sel} vs "
            f"{n_expected} at estimation time — panel or params changed "
            "since R2 ran; re-run the pipeline in order."
        )
    logger.info(f"  step 2 sample reconstructed: {n_sel:,} rows (matches R2)")

    # --- Step 2 objects ----------------------------------------------------------
    sub = est.loc[sel]
    ks, ms, ls_ = (sub[c].to_numpy() for c in ["k", "m", "l"])
    lks, lms, lls = (sub[c].to_numpy() for c in ["lk", "lm", "ll"])
    p1_s = gm.build_p1(ks, ms, ls_)
    lp1_s = gm.build_p1(lks, lms, lls)
    p2_s = gm.build_p2(ks, ls_)
    lp2_s = gm.build_p2(lks, lls)

    eps_s = gm.step1_eps(soln, sub["s"].to_numpy(), p1_s)
    leps_s = gm.step1_eps(soln, sub["ls"].to_numpy(), lp1_s)
    rt = sub["r"].to_numpy() - eps_s - gm.int_melast(p1_s, gamma_scaled, ms)
    lrt = sub["lr"].to_numpy() - leps_s - gm.int_melast(lp1_s, gamma_scaled, lms)

    omega = rt + p2_s @ a_hat
    lomega = lrt + lp2_s @ a_hat

    # eta = omega - Markov fit; dummy columns must match the estimation layout.
    ind_categories = s2["delta_layout"]["ind_dummy_categories"]
    sample_inds = sorted(sub["ind2d"].unique().tolist())
    if sample_inds != ind_categories:
        raise RuntimeError(
            f"industry set mismatch between R2 ({ind_categories}) and the "
            f"reconstructed sample ({sample_inds})"
        )
    dummies = pd.get_dummies(
        pd.Categorical(sub["ind2d"], categories=ind_categories), dtype=float
    ).to_numpy()[:, 1:]
    o2 = gm.markov_design(lomega, dummies, h2size)
    if o2.shape[1] != len(delta):
        raise RuntimeError(
            f"Markov design has {o2.shape[1]} columns but delta has "
            f"{len(delta)} coefficients"
        )
    eta = omega - o2 @ delta

    kel = gm.kelast(ks, ms, ls_, p2_s, gamma_scaled, a_hat)
    lel = gm.lelast(ks, ms, ls_, p2_s, gamma_scaled, a_hat)
    rts = sub["melast"].to_numpy() + kel + lel

    for col in ["omega", "lomega", "eta", "kelast", "lelast", "rts"]:
        est[col] = np.nan
    est.loc[sel, "omega"] = omega
    est.loc[sel, "lomega"] = lomega
    est.loc[sel, "eta"] = eta
    est.loc[sel, "kelast"] = kel
    est.loc[sel, "lelast"] = lel
    est.loc[sel, "rts"] = rts
    est["in_step2_sample"] = sel
    est["alpha_gross_output"] = est["rts"]

    lo_flag, hi_flag = map(float, config["postprocess"]["rts_flag_range"])
    est["rts_outlier_flag"] = est["rts"].notna() & (
        (est["rts"] < lo_flag) | (est["rts"] > hi_flag)
    )

    # --- diagnostics ---------------------------------------------------------------
    logger.info(
        "  mean elasticities (k, l, m): "
        f"{np.nanmean(kel):.4f}, {np.nanmean(lel):.4f}, "
        f"{sub['melast'].mean():.4f}"
    )
    q = np.nanpercentile(rts, [1, 10, 50, 90, 99])
    logger.info(
        f"  RTS: mean = {np.nanmean(rts):.4f}, sd = {np.nanstd(rts):.4f}; "
        f"p1/p10/p50/p90/p99 = {np.round(q, 3).tolist()}"
    )
    logger.info(
        f"  share of negative elasticities: k {(kel < 0).mean():.2%}, "
        f"l {(lel < 0).mean():.2%}, m {(sub['melast'] < 0).mean():.2%}"
    )
    n_flag = int(est["rts_outlier_flag"].sum())
    if n_flag:
        logger.warning(
            f"  {n_flag:,} rows have RTS outside [{lo_flag}, {hi_flag}] "
            "(flagged, not dropped)"
        )

    # --- write -----------------------------------------------------------------------
    raw_path = ensure_dir(inter) / RAW_OUT
    est[EXPORT_COLS + ["s", "r", "k", "m", "l", "has_w"]].to_parquet(
        raw_path, index=False, compression="snappy"
    )
    logger.info(
        f"wrote {raw_path.relative_to(PATHS.project_root)} ({len(est):,} rows)"
    )

    final = est.loc[np.isfinite(est["rts"]), EXPORT_COLS].reset_index(drop=True)
    if len(final) == 0:
        raise RuntimeError("no rows with finite RTS — estimation failed upstream")
    final_path = ensure_dir(outdata_dir(config)) / FINAL_OUT
    final.to_parquet(final_path, index=False, compression="snappy")
    logger.info(
        f"wrote {final_path.relative_to(PATHS.project_root)} "
        f"({len(final):,} firm-year RTS rows)"
    )


if __name__ == "__main__":
    sys.exit(main())
