"""Step 03: output elasticities of V by NAICS2 cell and year (DLEU 02_Produce_Thetas_st.do).

Baseline (theta_WI1_ct in the DLEU code): for each cell and window centre t, on the rows with
year in t-2..t+2 (truncated at the data edges),

    ivregress gmm r (c = L.c) k L.i L.k2 L.k

with r = ln(rev_D), c = ln(V_D), k = ln(tfas_D), L.k2 = (L.k)^2, i = ln(K - 0.9 L.K). The model
is just-identified, so the GMM point estimate is 2SLS: b = (Z'X)^-1 Z'y. Standard errors are
Stata's default for ivregress gmm (robust, no small-sample correction). No guardrails, as DLEU.
theta_c of window t is assigned to year t; years in theta.impute_years (2015) take the first
window's estimate and are flagged theta_imputed
(02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MARKUPS.md, D2).

Alternative (paper Section VI.A): theta_cs = median of costshare1 = V / (V + kexp) by cell x year.

Reads   <output.intermediary_dir>/markup_sample.parquet
Writes  <output.intermediary_dir>/thetas.parquet and thetas.csv (cell x year)

    python 03_estimate_thetas.py [--config ...] [--check CELL YEAR]

--check compares the numpy estimator with statsmodels IV2SLS and an explicit two-stage OLS on
one cell-window and exits.
"""

import argparse

import numpy as np
import pandas as pd

from markups_common import load_config, resolve, setup_logger

EXOG = ["k", "i_lag", "k_lag2", "k_lag"]   # included exogenous regressors, in DLEU's order


def design(d):
    """(y, X, Z) for the WI-IV regression; X = [c, exog, 1], Z = [L.c, exog, 1]."""
    one = np.ones(len(d))
    ex = d[EXOG].to_numpy()
    y = d["r"].to_numpy()
    X = np.column_stack([d["c"].to_numpy(), ex, one])
    Z = np.column_stack([d["c_lag"].to_numpy(), ex, one])
    return y, X, Z


def iv_gmm(y, X, Z):
    """Just-identified IV: coefficients, robust (HC0) standard errors, first-stage robust t on L.c."""
    ZX = Z.T @ X
    b = np.linalg.solve(ZX, Z.T @ y)
    u = y - X @ b
    S = (Z * (u ** 2)[:, None]).T @ Z
    ZXi = np.linalg.inv(ZX)
    V = ZXi @ S @ ZXi.T
    # First stage: c on Z; robust t-statistic of the excluded instrument L.c.
    ZZi = np.linalg.inv(Z.T @ Z)
    g = ZZi @ (Z.T @ X[:, 0])
    e = X[:, 0] - Z @ g
    Vg = ZZi @ ((Z * (e ** 2)[:, None]).T @ Z) @ ZZi
    return b, np.sqrt(np.diag(V)), g[0] / np.sqrt(Vg[0, 0])


def estimation_rows(df):
    """Rows with every variable of the regression non-missing (Stata's listwise deletion)."""
    df = df.assign(k_lag2=df["k_lag"] ** 2)
    need = ["r", "c", "c_lag", *EXOG]
    return df[np.isfinite(df[need]).all(axis=1)]


def window_rows(est, cell, t, h, years):
    lo, hi = max(t - h, years[0]), min(t + h, years[1])
    return est[(est["ind2d"] == cell) & est["year"].between(lo, hi)], lo, hi


def check(est, cell, t, h, years, logger):
    """numpy 2SLS vs statsmodels IV2SLS vs explicit two-stage OLS on one cell-window."""
    from statsmodels.sandbox.regression.gmm import IV2SLS
    d, lo, hi = window_rows(est, cell, t, h, years)
    y, X, Z = design(d)
    b, se, ft = iv_gmm(y, X, Z)
    sm = IV2SLS(y, X, Z).fit()
    chat = Z @ np.linalg.lstsq(Z, X[:, 0], rcond=None)[0]
    b2 = np.linalg.lstsq(np.column_stack([chat, X[:, 1:]]), y, rcond=None)[0]
    names = ["c", *EXOG, "_cons"]
    tab = pd.DataFrame({"numpy": b, "statsmodels": sm.params, "two_stage_ols": b2,
                        "se_robust": se}, index=names)
    logger.info(f"check cell {cell}, window {lo}-{hi}, n = {len(d):,}, first-stage t = {ft:.1f}\n"
                + tab.to_string(float_format=lambda x: f"{x:.10f}"))
    diff = max(np.abs(b - sm.params).max(), np.abs(b - b2).max())
    logger.info(f"max abs difference across implementations: {diff:.2e}")
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--check", nargs=2, type=int, metavar=("CELL", "YEAR"))
    parser.add_argument("--export-check", default=None,
                        help="with --check: write the cell-window estimation rows (r c c_lag k "
                             "i_lag k_lag year) to this path, for the Stata cross-check")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("thetas", config.get("log_level", "INFO"))
    th, smp = config["theta"], config["sample"]
    inter = resolve(config["output"]["intermediary_dir"])
    h = int(th["window_half_width"])
    years = (int(smp["first_year"]), int(smp["markup_years"][1]))

    df = pd.read_parquet(inter / "markup_sample.parquet")
    est = estimation_rows(df)
    logger.info(f"estimation rows (all regressors non-missing): {len(est):,} of {len(df):,}")

    if args.check:
        d = check(est, args.check[0], args.check[1], h, years, logger)
        if args.export_check:
            d[["year", "r", "c", "c_lag", "k", "i_lag", "k_lag"]].to_csv(args.export_check, index=False)
            logger.info(f"wrote {len(d):,} rows to {args.export_check}")
        return

    c0, c1 = map(int, th["centres"])
    rows = []
    for cell in smp["cells"]:
        for t in range(c0, c1 + 1):
            d, lo, hi = window_rows(est, cell, t, h, years)
            if len(d) < th["min_obs"]:
                raise RuntimeError(f"cell {cell} window {lo}-{hi}: only {len(d)} rows")
            b, se, ft = iv_gmm(*design(d))
            rows.append({"ind2d": cell, "year": t, "theta_wi_c": b[0], "theta_wi_k": b[1],
                         "se_wi_c": se[0], "se_wi_k": se[1], "first_stage_t": ft,
                         "n_iv": len(d), "window_lo": lo, "window_hi": hi,
                         "theta_imputed": False})
    thetas = pd.DataFrame(rows)

    # Edge years: the first window's estimate, flagged.
    first = thetas[thetas["year"] == c0]
    imp = [first.assign(year=int(y), theta_imputed=True) for y in th["impute_years"]]
    thetas = pd.concat([*imp, thetas], ignore_index=True)

    # Cost-share elasticity (paper VI.A): median costshare1 by cell x year.
    if th["cost_share"] != "median":
        raise ValueError("only theta.cost_share = median (the paper) is implemented")
    m0, m1 = map(int, smp["markup_years"])
    cs = (df[df["year"].between(m0, m1)].groupby(["ind2d", "year"])["cs1"]
          .agg(theta_cs="median", n_cs="size").reset_index())
    thetas = thetas.merge(cs, on=["ind2d", "year"], how="outer").sort_values(["ind2d", "year"])
    if thetas[["theta_wi_c", "theta_cs"]].isna().any().any():
        raise RuntimeError("a cell-year lacks theta_wi_c or theta_cs")

    thetas.to_parquet(inter / "thetas.parquet", index=False)
    thetas.to_csv(inter / "thetas.csv", index=False)
    piv = thetas.pivot(index="ind2d", columns="year", values="theta_wi_c")
    logger.info("theta_wi_c (cell x year):\n" + piv.round(3).to_string())
    logger.info("theta_cs (cell x year):\n"
                + thetas.pivot(index="ind2d", columns="year", values="theta_cs").round(3).to_string())
    logger.info(f"min first-stage t: {thetas['first_stage_t'].min():.1f}; "
                f"theta_wi_c range [{thetas['theta_wi_c'].min():.3f}, {thetas['theta_wi_c'].max():.3f}]")


if __name__ == "__main__":
    main()
