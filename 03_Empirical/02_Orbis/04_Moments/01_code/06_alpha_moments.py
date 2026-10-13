"""Step 06: alpha moments on the GNR step-2 firm-years (RTS gap, phi_v, GNR diagnostics).

Sample: GNR firm-years with alpha_moments.sample (in_step2_sample) and a finite rts, in the
window; alpha = rts (raw GNR returns to scale, as F(alpha)); log sales = r = log rev_D.
  (a) RTS gap, the rho_bar target (D6; Hubmer, Chan, Ozkan, Salgado and Hong, Table II): firm
      means of rts and r by (naics4, bvdid), no winsorization; within each naics4 market,
      pct = rank(mean r) / n and dm = rts - market mean;
      gap = mean(dm | pct > 0.95) - mean(dm | pct <= 0.50) over all firms of all markets.
      Firm-level bootstrap SE (resample bvdid). Variants: firm-year within ind2d-year
      (Salgado literal) and naics4-year, firm means within ind2d; top 1% and 10%; the
      20-bin within-market revenue profile of demeaned rts, melast and lelast.
  (b) phi_v = 1 - mean(melast) / mean(rts) over firm-years (D9, Compustat 08:158-231); GNR
      elasticity means and a_gnr = kelast / (kelast + lelast) (diagnostic, D8).
naics4 comes from the clean panel by (bvdid, year); it is constant within firm in the data
(checked; the modal code would be used otherwise).

Reads   <gnr.outdata_dir>/firm_year_rts.parquet, <input.clean_panel>
Writes  <output.intermediary_dir>/alpha_moments.json          moments by window, variants, profile
        <output.intermediary_dir>/alpha_moments_by_cell.csv   per ind2d cell, active window

    python 06_alpha_moments.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import time

import numpy as np
import pandas as pd

from moments_common import dirs, load_config, resolve, setup_logger, write_json

GAP_TOPS = {"top1": 0.99, "top5": 0.95, "top10": 0.90}


def gap_units(fy, id_col, market, collapse, cols=("rts", "log_sales", "melast", "lelast")):
    """Units ranked within `group`: firm means by (market, firm), or firm-years by (market, year)."""
    if collapse == "firm_mean":
        return fy.groupby([market, id_col], as_index=False)[list(cols)].mean(), [market]
    return fy[[id_col, "year", market, *cols]], [market, "year"]


def rts_gap(units, group, tops=GAP_TOPS, bottom=0.5, bins=None):
    """Mean within-group-demeaned rts of the top shares minus the bottom half of the within-group
    revenue distribution; optionally the binned profile of demeaned rts, melast and lelast."""
    g = units.groupby(group)
    pct = g["log_sales"].rank(pct=True).to_numpy()
    dm = {c: (units[c] - g[c].transform("mean")).to_numpy() for c in ["rts", "melast", "lelast"]}
    low = dm["rts"][pct <= bottom].mean()
    out = {k: float(dm["rts"][pct > t].mean() - low) for k, t in tops.items()}
    if bins:
        b = np.clip(np.ceil(pct * bins), 1, bins).astype(int)
        prof = pd.DataFrame({"bin": b, **{f"dm_{c}": v for c, v in dm.items()}}).groupby("bin").mean()
        prof["n"] = np.bincount(b)[1:]
        return out, prof.reset_index()
    return out


def bootstrap_gap(units, market, top, bottom, reps, seed):
    """Firm-level bootstrap of the firm-mean gap: resample firms with replacement (multiplicity
    weights), recompute within-market ranks and means on each resample."""
    u = units.sort_values([market, "log_sales"], kind="stable")
    mkt = pd.factorize(u[market])[0]
    x = u["rts"].to_numpy()
    n, n_mkt = len(u), mkt.max() + 1
    start = np.r_[0, np.flatnonzero(np.diff(mkt)) + 1]          # first row of each market
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        cum = np.cumsum(c)
        tot = np.bincount(mkt, weights=c, minlength=n_mkt)
        before = cum - c - (cum - c)[start][mkt]                 # resampled firms ranked below
        pct = np.divide(before + (c + 1) / 2, tot[mkt], out=np.zeros(n), where=tot[mkt] > 0)
        mean = np.divide(np.bincount(mkt, weights=c * x, minlength=n_mkt), tot,
                         out=np.zeros(n_mkt), where=tot > 0)
        dm = x - mean[mkt]
        hi, lo = c * (pct > top), c * (pct <= bottom)
        out[r] = (hi @ dm) / hi.sum() - (lo @ dm) / lo.sum()
    return out


def load_sample(config, logger):
    """GNR step-2 firm-years (all years) with naics4 merged from the clean panel."""
    g, am = config["gnr"], config["alpha_moments"]
    id_col = g["id_col"]
    fy = pd.read_parquet(resolve(g["outdata_dir"]) / "firm_year_rts.parquet",
                         columns=[id_col, "year", "ind2d", am["sample"], am["alpha_col"], "r", "melast",
                                  "kelast", "lelast"])
    fy = fy[fy[am["sample"]] & np.isfinite(fy[am["alpha_col"]])]
    fy = fy.drop(columns=am["sample"]).rename(columns={"r": "log_sales"})
    cp = pd.read_parquet(resolve(config["input"]["clean_panel"]), columns=[id_col, "year", "naics4"])
    cp = cp[cp[id_col].isin(fy[id_col].unique())]
    n_codes = cp.groupby(id_col)["naics4"].nunique()
    n_multi = int((n_codes > 1).sum())
    modal = cp.groupby([id_col, "naics4"]).size().reset_index().sort_values(
        [id_col, 0], ascending=[True, False]).drop_duplicates(id_col).set_index(id_col)["naics4"]
    fy["naics4"] = fy[id_col].map(modal)
    if fy["naics4"].isna().any():
        raise RuntimeError("GNR firm-years without a naics4 in the clean panel")
    logger.info(f"GNR sample: {len(fy):,} step-2 firm-years, {fy[id_col].nunique():,} firms; naics4 "
                f"constant within firm except {n_multi} firms (modal code used)")
    return fy, n_multi


def window_moments(fy, config, window, logger, profile=False):
    am = config["alpha_moments"]
    gp, id_col, a = am["rts_gap"], config["gnr"]["id_col"], am["alpha_col"]
    w = fy[fy["year"].between(*window)]
    variants = {}
    for collapse in ["firm_mean", "firm_year"]:
        for m in ["ind2d", gp["market"]]:
            units, group = gap_units(w, id_col, m, collapse)
            variants["_".join([collapse, "within", *group])] = rts_gap(units, group, bottom=gp["bottom"])
    target = f"{gp['collapse']}_within_{gp['market']}"
    means = w[["melast", "kelast", "lelast", a]].mean()
    res = {"window": [int(x) for x in window], "firm_years": len(w), "firms": int(w[id_col].nunique()),
           "rts_gap_variants": variants,
           "alpha_rts_gap_top5_bottom50": variants[target]["top5"],
           "phi_v": float(1.0 - means["melast"] / means[a]),
           "gnr_means": {k: float(v) for k, v in means.items()},
           "a_gnr": float(means["kelast"] / (means["kelast"] + means["lelast"]))}
    logger.info(f"{window}: gap {res['alpha_rts_gap_top5_bottom50']:.4f}; phi_v {res['phi_v']:.4f}; "
                f"a_gnr {res['a_gnr']:.4f}")
    if not profile:
        return res, None
    # Profiles and the per-cell table on the active window.
    prof = {}
    for name, (m, collapse) in {"target": (gp["market"], gp["collapse"]),
                                "firm_year_within_ind2d_year": ("ind2d", "firm_year")}.items():
        units, group = gap_units(w, id_col, m, collapse)
        _, prof[name] = rts_gap(units, group, bottom=gp["bottom"], bins=int(gp["profile_bins"]))
    units, group = gap_units(w, id_col, gp["market"], gp["collapse"])
    units["ind2d"] = units[gp["market"]].map(w.groupby(gp["market"])["ind2d"].first())
    gcell = units.groupby("ind2d").apply(
        lambda u: rts_gap(u, group, tops={"top5": gp["top"]}, bottom=gp["bottom"])["top5"],
        include_groups=False).rename("rts_gap_top5_bottom50")
    cells = (units.groupby("ind2d").size().rename("n_firms").to_frame().join(gcell)
             .join(w.groupby("ind2d")[["rts", "melast", "kelast", "lelast"]].mean().add_prefix("mean_"))
             .reset_index())
    res["profiles"] = {k: v.to_dict(orient="list") for k, v in prof.items()}
    return res, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("alpha_moments", config.get("log_level", "INFO"))
    t0 = time.time()
    inter, _ = dirs(config)
    gp = config["alpha_moments"]["rts_gap"]
    if config["alpha_moments"]["alpha_col"] != "rts":
        raise ValueError("alpha_moments.alpha_col: the RTS gap is defined on rts")

    fy, n_multi = load_sample(config, logger)
    active, cells = window_moments(fy, config, config["active_window"], logger, profile=True)
    robust, _ = window_moments(fy, config, config["robustness_window"], logger)

    w = fy[fy["year"].between(*config["active_window"])]
    units, _ = gap_units(w, config["gnr"]["id_col"], gp["market"], gp["collapse"])
    if units[config["gnr"]["id_col"]].duplicated().any():
        raise RuntimeError("a firm appears in two markets: the firm bootstrap needs one unit per firm")
    boot = bootstrap_gap(units, gp["market"], gp["top"], gp["bottom"], int(gp["bootstrap_reps"]),
                         int(gp["seed"]))
    check = rts_gap(units, [gp["market"]], tops={"t": gp["top"]}, bottom=gp["bottom"])["t"]
    active["rts_gap_bootstrap"] = {"se": float(boot.std(ddof=1)), "mean": float(boot.mean()),
                                   "reps": int(gp["bootstrap_reps"]), "seed": int(gp["seed"]),
                                   "unit": "firm (bvdid)", "point": check}
    logger.info(f"RTS gap bootstrap: point {check:.4f}, se {boot.std(ddof=1):.4f} "
                f"({gp['bootstrap_reps']} reps)")

    res = {"alpha_col": config["alpha_moments"]["alpha_col"], "naics4_multi_code_firms": n_multi,
           "active": active, "robustness": robust}
    write_json(inter / "alpha_moments.json", res)
    cells.to_csv(inter / "alpha_moments_by_cell.csv", index=False)
    logger.info("by cell:\n" + cells.round(3).to_string(index=False))
    logger.info(f"wrote alpha_moments.json and alpha_moments_by_cell.csv ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
