"""Step 05: EMX slope, Delta(inverse cost-weighted markup) on Delta(HHI) across markets.

Ports Compustat 08b_emx_slope_moment.py. Per (year, market) from step 02: inv_mu = 1 / mu_cw_iy
(V-weighted, S1-trimmed markup_wi) and hhi = sum s^2.
  Primary (D5): markets.emx (ind2d) long difference from active_window start to end, one obs
    per market, OLS with a constant, HC1 SE, unweighted (08b:76-82, 128-131).
  Robustness, for every market in markets.emx_robustness: long differences over emx.ld_windows,
    the same with two-year averaged endpoints, and stacked LDs of emx.stacked_gaps years over
    robustness_window with and without market FE (08b:85-113). Finer markets add sales-weighted
    versions (WLS weight = market sales in the start year of each difference, as Autor et al.)
    and a sample without markets that have n < markets.low_firm_threshold in any year.
  Decomposition (ORBIS_SPAIN_MOMENTS.md Section 8), LD over active_window: at ind2d,
    1 / mu = (1 / theta) (V / rev) on the untrimmed sample, so Delta(1/mu) = mean(V/rev)
    Delta(1/theta) + mean(1/theta) Delta(V/rev) (endpoint means); each part regressed on
    Delta(HHI). At naics4: the within-cell slope (cell FE), the between-cell slope (sales-
    weighted cell means of the naics4 changes, start-year sales) and the within-cell share of
    Var(Delta HHI). The raw slopes of Delta(1/theta) and Delta(V/rev) are reported as well.

Reads   <output.intermediary_dir>/market_year.parquet
        <input.markups_dir>/firm_year_markups.parquet   (thetas and V/rev for the decomposition)
Writes  <output.intermediary_dir>/emx_slope_moment.yaml       primary value and decomposition
        <output.intermediary_dir>/emx_slope_robustness.csv    every (market, spec) row

    python 05_emx_slope.py [--config config_moments_orbis_es.yaml]
"""

import argparse

import pandas as pd
import statsmodels.api as sm
import yaml

from moments_common import dirs, load_config, resolve, setup_logger


def ols(d, y="dy", x="dx", fe=None, w=None):
    """OLS (or WLS) of y on x with a constant and optional FE dummies; HC1 SE (08b:_ols)."""
    rhs = sm.add_constant(d[[x]].astype(float))
    if fe is not None:
        rhs = pd.concat([rhs, pd.get_dummies(d[fe], drop_first=True, prefix="fe").astype(float)], axis=1)
    model = sm.WLS(d[y].astype(float), rhs, weights=d[w]) if w else sm.OLS(d[y].astype(float), rhs)
    m = model.fit(cov_type="HC1")
    return {"b_hat": float(m.params[x]), "se_hc1": float(m.bse[x]), "t_stat": float(m.tvalues[x]),
            "n_obs": int(m.nobs), "r_squared": float(m.rsquared)}


def wide(panel):
    """Market x year tables of inv_mu, hhi, tot_sale and n."""
    p = panel.assign(inv_mu=1.0 / panel["mu_cw_iy"])
    return {c: p.pivot(index="market", columns="year", values=c) for c in ["inv_mu", "hhi", "tot_sale", "n"]}


def diff(W, start, end):
    """Delta from the mean over `start` years to the mean over `end` years, per market;
    markets missing any endpoint year drop out. Weight = mean sales over the `start` years."""
    yrs = list(start) + list(end)
    if not set(yrs) <= set(W["hhi"].columns):
        return pd.DataFrame(columns=["market", "dy", "dx", "w"])
    def d(c):
        return W[c][list(end)].mean(axis=1, skipna=False) - W[c][list(start)].mean(axis=1, skipna=False)
    out = pd.DataFrame({"dy": d("inv_mu"), "dx": d("hhi"),
                        "w": W["tot_sale"][list(start)].mean(axis=1, skipna=False)})
    return out.dropna().rename_axis("market").reset_index()


def long_diff(W, y0, y1, avg=False):
    return diff(W, [y0, y0 + 1] if avg else [y0], [y1 - 1, y1] if avg else [y1])


def stacked(W, y0, y1, gap):
    return pd.concat([diff(W, [s], [s + gap]).assign(y_start=s) for s in range(y0, y1 - gap + 1)],
                     ignore_index=True)


def robustness(panel, ld_windows, stack_window, gaps, fine, low_firm):
    """All (spec, weighting, sample) rows for one market definition."""
    W_all = wide(panel)
    ok = (W_all["n"] >= low_firm).all(axis=1)
    specs = []
    for y0, y1 in ld_windows:
        for avg in (False, True):
            specs.append(("ld_2yr_avg" if avg else "ld", (y0, y1), lambda W, a=avg, y0=y0, y1=y1:
                          long_diff(W, y0, y1, a), [None]))
    for gap in gaps:
        specs.append((f"stacked_{gap}yr", tuple(stack_window), lambda W, g=gap:
                      stacked(W, *stack_window, g), [None, "market"]))
    rows = []
    for sample in (["all", f"n>={low_firm}"] if fine else ["all"]):
        for spec, (y0, y1), build, fes in specs:
            W = W_all if sample == "all" else {c: t.loc[ok] for c, t in W_all.items()}
            d = build(W)
            if len(d) < 3:
                continue
            for fe in fes:
                for wt in (["none", "sales"] if fine else ["none"]):
                    r = ols(d, fe=fe, w="w" if wt == "sales" else None)
                    rows.append({"spec": spec, "years": f"{y0}-{y1}", "fe": fe or "none",
                                 "weight": wt, "sample": sample, **r})
    return rows


def decomposition(cfg_window, panel4, cells, n4_to_cell, logger):
    """Section 8 of the plan: theta vs cost-share channels at ind2d; within vs between at naics4."""
    y0, y1 = cfg_window
    # ind2d: 1/mu = (1/theta)(V/rev) on the untrimmed markup sample.
    c = cells.set_index(["ind2d", "year"])
    a = (1.0 / c["theta"]).unstack()
    b = (c["V"] / c["rev"]).unstack()
    hhi = c["hhi"].unstack()
    d = pd.DataFrame({"dx": hhi[y1] - hhi[y0],
                      "theta_part": 0.5 * (b[y0] + b[y1]) * (a[y1] - a[y0]),
                      "cost_share_part": 0.5 * (a[y0] + a[y1]) * (b[y1] - b[y0]),
                      "d_inv_theta": a[y1] - a[y0], "d_V_over_rev": b[y1] - b[y0]})
    d["total_untrimmed"] = d["theta_part"] + d["cost_share_part"]
    out = {"window": [int(y0), int(y1)], "ind2d": {
        k: ols(d, y=k) for k in ["total_untrimmed", "theta_part", "cost_share_part",
                                 "d_inv_theta", "d_V_over_rev"]}}

    # naics4: within (cell FE) and between (sales-weighted cell means) slopes.
    ld = long_diff(wide(panel4), y0, y1)
    ld["cell"] = ld["market"].map(n4_to_cell)
    within = ols(ld, fe="cell")
    cm = ld.assign(wdy=ld["dy"] * ld["w"], wdx=ld["dx"] * ld["w"]).groupby("cell")[["wdy", "wdx", "w"]].sum()
    between = ols(pd.DataFrame({"dy": cm["wdy"] / cm["w"], "dx": cm["wdx"] / cm["w"]}))
    dev = ld["dx"] - ld.groupby("cell")["dx"].transform("mean")
    share_within = float((dev ** 2).sum() / ((ld["dx"] - ld["dx"].mean()) ** 2).sum())
    out["naics4"] = {"pooled": ols(ld), "within_cell_fe": within, "between_cell_means": between,
                     "share_var_dhhi_within_cells": share_within,
                     "sd_dhhi": float(ld["dx"].std()),
                     "sd_dhhi_ind2d": float(d["dx"].std())}
    logger.info("decomposition, ind2d LD: " + ", ".join(
        f"{k} {v['b_hat']:+.3f} ({v['se_hc1']:.3f})" for k, v in out["ind2d"].items()))
    logger.info(f"decomposition, naics4 LD: pooled {out['naics4']['pooled']['b_hat']:+.3f}, within "
                f"{within['b_hat']:+.3f} ({within['se_hc1']:.3f}), between {between['b_hat']:+.3f} "
                f"({between['se_hc1']:.3f}); share of Var(dHHI) within cells {share_within:.3f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("emx_slope", config.get("log_level", "INFO"))
    inter, _ = dirs(config)
    mk, emx = config["markets"], config["emx"]
    if emx["primary"] != "long_difference":
        raise ValueError("emx.primary: only long_difference is implemented")
    y0, y1 = map(int, config["active_window"])
    my = pd.read_parquet(inter / "market_year.parquet")

    rows = []
    for m in mk["emx_robustness"]:
        r = robustness(my[my["market_def"] == m], [tuple(w) for w in emx["ld_windows"]],
                       config["robustness_window"], emx["stacked_gaps"], m != mk["emx"],
                       int(mk["low_firm_threshold"]))
        rows += [{"market": m, **x} for x in r]
    rob = pd.DataFrame(rows)
    rob.to_csv(inter / "emx_slope_robustness.csv", index=False)
    logger.info("robustness, unweighted, all markets:\n" + rob[
        (rob["weight"] == "none") & (rob["sample"] == "all")].drop(columns=["weight", "sample"])
        .round(4).to_string(index=False))

    primary = ols(long_diff(wide(my[my["market_def"] == mk["emx"]]), y0, y1))

    # Decomposition inputs from the untrimmed markup sample.
    fy = pd.read_parquet(resolve(config["input"]["markups_dir"]) / "firm_year_markups.parquet",
                         columns=["year", "ind2d", "naics4", "theta_wi_c", "V_D", "rev_D"])
    g = fy.groupby(["ind2d", "year"])
    if (g["theta_wi_c"].nunique() > 1).any():
        raise RuntimeError("theta_wi_c varies within an (ind2d, year) cell")
    n4_cells = fy.groupby("naics4")["ind2d"].unique()
    if (n4_cells.str.len() > 1).any():
        raise RuntimeError("naics4 does not nest in ind2d")
    cells = g.agg(theta=("theta_wi_c", "first"), V=("V_D", "sum"), rev=("rev_D", "sum")).reset_index()
    p2 = my[my["market_def"] == "ind2d"].assign(ind2d=lambda x: x["market"].astype(int))
    cells = cells.merge(p2[["ind2d", "year", "hhi"]], on=["ind2d", "year"], validate="one_to_one")
    dec = decomposition((y0, y1), my[my["market_def"] == "naics4"], cells,
                        n4_cells.str[0].astype(int), logger)

    moment = {
        "name": "emx_oligopoly_slope",
        "description": "Slope of Delta(market V-weighted inverse markup) on Delta(market HHI) across "
                       "NAICS2 cells, long difference; EMX (2023) eq. 83: b = -(1/eta - 1/gamma).",
        "value": primary["b_hat"], "se": primary["se_hc1"], "t_stat": primary["t_stat"],
        "n": primary["n_obs"], "r_squared": primary["r_squared"],
        "spec": "long_difference, OLS with constant, HC1, unweighted",
        "market": mk["emx"], "window": [y0, y1],
        "decomposition": dec,
        "robustness_table": "emx_slope_robustness.csv",
    }
    (inter / "emx_slope_moment.yaml").write_text(yaml.safe_dump(moment, sort_keys=False, width=100))
    logger.info(f"primary: {mk['emx']} LD {y0}-{y1}: b {primary['b_hat']:+.4f} "
                f"(se {primary['se_hc1']:.4f}, n {primary['n_obs']})")
    logger.info("wrote emx_slope_moment.yaml and emx_slope_robustness.csv")


if __name__ == "__main__":
    main()
