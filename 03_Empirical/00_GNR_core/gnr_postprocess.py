"""
Reconstruct firm-year GNR objects from saved Step 1 / Step 2 parameters, shared by
every data source.

Moved out of 01_Compustat/00_Hubmer_RTS/01_code/03_postprocess_gnr.py (stage R3), logic
unchanged. Deterministic, no re-optimization:
  eps, melast                       on the full trimmed sample
  omega, lomega, eta,
  kelast, lelast, rts               on the Step-2 sample (NaN elsewhere)
  alpha_gross_output = rts, in_step2_sample, rts_outlier_flag
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gnr_model as gm


def reconstruct_firm_year(
    est: pd.DataFrame, s1: dict, s2: dict, rts_flag_range, logger
) -> pd.DataFrame:
    """Add the firm-year GNR columns to the trimmed panel `est` (modified in place)."""
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

    lo_flag, hi_flag = map(float, rts_flag_range)
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
    return est
