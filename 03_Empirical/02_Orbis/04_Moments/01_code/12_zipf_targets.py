"""Step 12: Zipf tail index of firm sales, the market-free (xi, N) targets.

Separate from the bundle (steps 01-07): writes its own 03_outdata/zipf_targets.yaml, so
aggregate_moments.yaml and the bundle manifest hash are unchanged. Design and diagnostics:
02_Drafts/ARCHIVE/md_files/ORBIS_migration/ORBIS_ZIPF_CALIBRATION.md (Sections 4 and 8.1).

Sample: firm-years of the DLEU markup panel in the active window with finite rev_D, V_D and
markup_wi, rev_D > 0 and V_D > 0 (the sample of the md file's market_free_moments.py).
Unit: the firm mean of rev_D over the window, divided by the mean of firm means in the firm's
NAICS2 cell (ind2d, the first ind2d of the firm). Statistic: Gabaix-Ibragimov rank-1/2 OLS on
the top `frac` of units, k = max(int(frac * n), 20); the model counterpart is
04_Estimation/orbis_estimation/zipf.py::tail_index (same formula). Also reported: the Hill
estimate, economy-wide raw sales, NAICS4-normalized sales (diagnostic), per-year values on
firm-year sales normalized by the ind2d-year mean, k and the approximate sampling SE
zeta * sqrt(2 / k) (Gabaix-Ibragimov).

The `validation` block holds untargeted data counterparts for 04_Estimation/02_validate_zipf.py:
concentration within NAICS2 and NAICS4 (copied from step 03's pooled_targets.json) and the
markup_wi distribution on the same firm-years (percentiles pooled over firm-years; mu_cw,
mu_sw and the weighted variances of log markups as window means of yearly values, the
md file's market_free_moments.py definitions; cost = V_D + kexp).

Reads   <input.markups_dir>/firm_year_markups.parquet, <output.intermediary_dir>/pooled_targets.json
Writes  <output.outdata_dir>/zipf_targets.yaml

    python 12_zipf_targets.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import datetime
import json
import time

import numpy as np
import pandas as pd
import yaml

from moments_common import PROJECT_ROOT, dirs, file_sha256, fingerprint, git_state, load_config, resolve, setup_logger

CUTOFFS = {"top5": 0.05, "top2": 0.02, "top1": 0.01}
TARGETED = ("top5", "top2")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
CONCENTRATION_KEYS = ("cr4", "cr20", "top1pct", "top5pct", "median_firms_per_market",
                      "mean_firms_per_sector", "n_markets")


def tail_index(x, frac):
    """Rank-1/2 OLS tail index on the top `frac` of x (the md file's reference implementation)."""
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(int(frac * len(x)), 20)
    xs = x[:k]
    rk = np.arange(1, k + 1) - 0.5
    return float(-np.polyfit(np.log(xs), np.log(rk), 1)[0])


def hill_index(x, frac):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(int(frac * len(x)), 20)
    xs = x[:k]
    return float(1.0 / np.mean(np.log(xs[:-1] / xs[-1])))


def tail_block(x):
    """OLS, Hill, k and approximate SE at every cutoff."""
    out = {}
    for tag, frac in CUTOFFS.items():
        z = tail_index(x, frac)
        k = max(int(frac * len(x)), 20)
        out[tag] = {"ols": z, "hill": hill_index(x, frac), "k": int(k), "se_approx": float(z * np.sqrt(2.0 / k))}
    return out


def load_sample(config, window):
    path = resolve(config["input"]["markups_dir"]) / "firm_year_markups.parquet"
    cols = ["bvdid", "year", "ind2d", "naics4", "rev_D", "V_D", "kexp", "markup_wi"]
    d = pd.read_parquet(path, columns=cols)
    d = d[d["year"].between(*window)].dropna(subset=["rev_D", "V_D", "markup_wi"])
    d = d[(d["rev_D"] > 0) & (d["V_D"] > 0)]
    return d, path


def weighted_quantiles(x, w, qs):
    o = np.argsort(x, kind="stable")
    x, w = np.asarray(x, dtype=float)[o], np.asarray(w, dtype=float)[o]
    c = np.cumsum(w) / w.sum()
    return {f"p{int(round(q * 100))}": float(x[min(np.searchsorted(c, q), len(x) - 1)]) for q in qs}


def markup_validation(d):
    """Data markup distribution: pooled firm-year percentiles and yearly aggregates (window mean)."""
    cost = (d["V_D"] + d["kexp"].fillna(0.0)).to_numpy()
    mu, sales = d["markup_wi"].to_numpy(), d["rev_D"].to_numpy()
    pct = {lab: weighted_quantiles(mu, w, QUANTILES)
           for lab, w in (("unweighted", np.ones(len(d))), ("cost_weighted", cost), ("sales_weighted", sales))}
    rows = []
    for _, g in d.assign(cost=cost).groupby("year"):
        s, c = g["rev_D"] / g["rev_D"].sum(), g["cost"] / g["cost"].sum()
        lm = np.log(g["markup_wi"])
        rows.append({"mu_cw": float(c @ g["markup_wi"]), "mu_sw": float(s @ g["markup_wi"]),
                     "var_sw_log": float(s @ (lm - s @ lm) ** 2), "var_cw_log": float(c @ (lm - c @ lm) ** 2)})
    agg = pd.DataFrame(rows).mean()
    return {"percentiles": pct, **{k: float(v) for k, v in agg.items()},
            "gap_sw_cw": float(agg["mu_sw"] - agg["mu_cw"])}


def concentration_validation(config, window):
    path = resolve(config["output"]["intermediary_dir"]) / "pooled_targets.json"
    pt = json.loads(path.read_text())
    out = {}
    for market in ("ind2d", "naics4"):
        block = pt["by_market"][market]["active"]
        if [int(y) for y in block["window"]] != window:
            raise ValueError(f"pooled_targets.json {market} window {block['window']} != {window}")
        out[market] = {k: float(block[k]) for k in CONCENTRATION_KEYS}
    return out, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, cfg_path = load_config(args.config)
    logger = setup_logger("zipf_targets", config.get("log_level", "INFO"))
    t0 = time.time()
    _, outd = dirs(config)
    window = [int(y) for y in config["active_window"]]

    d, path = load_sample(config, window)
    fm = d.groupby("bvdid").agg(rev=("rev_D", "mean"), ind2d=("ind2d", "first"), naics4=("naics4", "first"))
    rel2 = (fm["rev"] / fm.groupby("ind2d")["rev"].transform("mean")).to_numpy()
    rel4 = (fm["rev"] / fm.groupby("naics4")["rev"].transform("mean")).to_numpy()
    logger.info(f"{len(d):,} firm-years, {len(fm):,} firms, {fm['ind2d'].nunique()} ind2d cells ({window})")

    main_block = tail_block(rel2)
    by_year = {}
    for year, g in d.groupby("year"):
        rel = (g["rev_D"] / g.groupby("ind2d")["rev_D"].transform("mean")).to_numpy()
        by_year[int(year)] = {tag: tail_index(rel, frac) for tag, frac in CUTOFFS.items()}
        by_year[int(year)]["n_firms"] = int(len(rel))

    concentration, pt_path = concentration_validation(config, window)
    commit, branch, dirty = git_state()
    out = {
        "schema": "orbis-zipf-targets-v1",
        "description": "Zipf tail index of firm sales (rank-1/2 OLS); targets of the market-free "
                       "(xi, N) calibration. See ORBIS_ZIPF_CALIBRATION.md.",
        "active_window": window,
        "basis": "firm mean of rev_D over the window / ind2d mean of firm means",
        "estimator": "Gabaix-Ibragimov rank-1/2 OLS, k = max(int(frac * n), 20)",
        "targets": {f"zeta_{tag}": main_block[tag]["ols"] for tag in TARGETED},
        "untargeted": {"zeta_top1": main_block["top1"]["ols"]},
        "n_firms": int(len(fm)),
        "n_firm_years": int(len(d)),
        "detail": {
            "normalized_ind2d": main_block,
            "raw_economy_wide": tail_block(fm["rev"].to_numpy()),
            "normalized_naics4": tail_block(rel4),
            "by_year_normalized_ind2d_year": by_year,
        },
        "validation": {
            "concentration": concentration,
            "markups": markup_validation(d),
        },
        "provenance": {
            "script": "03_Empirical/02_Orbis/04_Moments/01_code/12_zipf_targets.py",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": commit, "branch": branch, "code_dirty": dirty,
            "config": str(cfg_path.resolve().relative_to(PROJECT_ROOT)),
            "config_sha256": file_sha256(cfg_path),
            "input": fingerprint(path),
            "pooled_targets": fingerprint(pt_path),
        },
    }
    target = outd / "zipf_targets.yaml"
    target.write_text(yaml.safe_dump(out, sort_keys=False, width=100))

    for tag in CUTOFFS:
        b = main_block[tag]
        logger.info(f"{tag}: OLS {b['ols']:.4f} (se ~{b['se_approx']:.4f}), Hill {b['hill']:.4f}, k={b['k']:,}")
    logger.info("by year (top 5%): " + ", ".join(f"{y} {v['top5']:.3f}" for y, v in by_year.items()))
    logger.info(f"wrote {target} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
