"""Benchmark the Orbis Spain labour share against Karabarbounis and Neiman (2014, QJE).

Reads the clean panel (never writes it), the extra cost items (01_extract_extra_costs.py),
the national-accounts benchmark (05_fetch_kn_benchmarks.py) and the pipeline's Eurostat
deflators. Writes 03_outdata/kn_results.json.

Sample: Spanish unconsolidated accounts 2015-2023, all non-financial industries (NAICS 52
finance and 92 public administration dropped), nature-of-expense accounts with reported
depreciation, other external costs >= 0. Labour share = staff costs / gross operating value
added, va = opre - mate - (oope - staf - depr), as in KN's W N / Q.

  1. Aggregate labour share vs national accounts S11 (D1 / B1G), and Orbis coverage.
  2. Aggregation: VA-weighted aggregate vs unweighted firm mean (Olley-Pakes covariance),
     and KN's within/between-industry decomposition of the trend (their eq. 2, A64 industries).
  3. KN income accounting (eqs. 11-13) by size decile: s_L + s_K + s_Pi = 1, with
     s_K = R K / Y, R = 1/beta - 1 + delta (KN: beta = 0.91, delta = 0.10), K = book capital.
  4. KN eq. (19) across NACE sections: trend in labour share / (1 - s_L) on trend in the log
     relative price of investment (GFCF deflator / HICP); slope = sigma - 1.

    python3 01_code/06_kn_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ORBIS = HERE.parent
CLEAN = ORBIS / "01_Cleaning" / "01_intermediary" / "orbis_es_clean_panel.parquet"
DEFL = ORBIS / "01_Cleaning" / "00_indata" / "deflators" / "deflators_es.csv"
EXTRA = HERE / "00_indata" / "orbis_es_extra_costs.parquet"
NA = HERE / "00_indata" / "benchmarks" / "na_s11_es.csv"
OUT = HERE / "03_outdata"
KEYS = ["bvdid", "conscode", "filing_type", "closdate"]
YEARS = list(range(2015, 2024))
BETA, DELTA = 0.91, 0.10            # KN section IV.B
R_KN = 1 / BETA - 1 + DELTA         # rental rate per euro of capital, about 0.20
R_LOW = 0.04 + 0.08                 # lower alternative: 4% real rate, 8% depreciation
TOL = 0.01
DROP_SEC = {"K": "financial activities", "L": "real estate (outlier: -407 on the KN scale)",
            "O": "public administration"}


def trend(y, x=None):
    """OLS slope of y on year, per decade (KN report trends per 10 years)."""
    x = np.asarray(YEARS if x is None else x, dtype=float)
    return float(np.polyfit(x, np.asarray(y, dtype=float), 1)[0] * 10)


def load():
    cols = KEYS + ["year", "naics2_raw", "naics4", "nace_section", "a64", "inc_year",
                   "rev", "opre", "mate", "staf", "tfas", "empl"]
    d = pd.read_parquet(CLEAN, columns=cols)
    d = d[d["year"].between(2015, 2023) & ~d["naics2_raw"].isin([52, 92]) & (d["opre"] > 0)]
    x = pd.read_parquet(EXTRA)
    for c in ["conscode", "filing_type"]:
        d[c] = d[c].astype(str)
        x[c] = x[c].astype(str)
    d["closdate"] = pd.to_datetime(d["closdate"])
    x["closdate"] = pd.to_datetime(x["closdate"])
    n0 = len(d)
    d = d.merge(x, on=KEYS, how="left", validate="one_to_one")
    d = d[d[["depr", "oope", "oppl", "cost"]].notna().all(axis=1)]
    d = d[(d["cost"] - d["mate"]).abs() / d["opre"] < TOL]
    d["oth"] = d["oope"] - d["staf"] - d["depr"]
    d = d[d["oth"] >= 0].copy()
    d["va"] = d["opre"] - d["mate"] - d["oth"]
    d["op"] = d["va"] - d["staf"] - d["depr"]
    d["k"] = d["tfas"].clip(lower=0).fillna(0.0)
    return d, n0


def aggregate(d, na):
    g = d.groupby("year")[["staf", "va"]].sum()
    na_p = na[na["direct"] == "PAID"].set_index("year")
    na_r = na[na["direct"] == "RECV"].set_index("year")
    s_na = na_p["D1"] / na_p["B1G"]
    s_na_fc = na_p["D1"] / (na_p["B1G"] - na_p["D29"] + na_r["D39"])   # factor-cost GVA
    out = {"years": YEARS,
           "orbis": (g["staf"] / g["va"]).loc[YEARS].round(5).tolist(),
           "na": s_na.loc[YEARS].round(5).tolist(),
           "na_fc": s_na_fc.loc[YEARS].round(5).tolist(),
           "na_years_long": [int(y) for y in na_p.index if y <= 2024],
           "na_long": s_na.loc[[y for y in na_p.index if y <= 2024]].round(5).tolist(),
           "cov_va": (g["va"] / (na_p["B1G"] * 1e6)).loc[YEARS].round(4).tolist(),
           "cov_l": (g["staf"] / (na_p["D1"] * 1e6)).loc[YEARS].round(4).tolist()}
    # Balanced panel: firms with a kept account in every year, so coverage changes cannot
    # move the aggregate.
    nyr = d.groupby("bvdid")["year"].transform("nunique")
    b = d[nyr == len(YEARS)].groupby("year")[["staf", "va"]].sum()
    out["orbis_bal"] = (b["staf"] / b["va"]).loc[YEARS].round(5).tolist()
    out["bal_firms"] = int(d.loc[nyr == len(YEARS), "bvdid"].nunique())
    out["bal_va_share"] = round(float(b["va"].sum() / g["va"].sum()), 4)
    out["trend_orbis_bal"] = round(100 * trend(out["orbis_bal"]), 2)
    out["trend_orbis"] = round(100 * trend(out["orbis"]), 2)
    out["trend_na"] = round(100 * trend(out["na"]), 2)
    pre = [y for y in YEARS if y != 2020]
    out["trend_orbis_ex2020"] = round(100 * trend([v for y, v in zip(YEARS, out["orbis"]) if y != 2020], pre), 2)
    out["trend_na_ex2020"] = round(100 * trend([v for y, v in zip(YEARS, out["na"]) if y != 2020], pre), 2)
    return out


def aggregation(d):
    f = d[d["va"] > 0].copy()
    f["s"] = f["staf"] / f["va"]
    lo, hi = f["s"].quantile([0.01, 0.99])
    f = f[f["s"].between(lo, hi)]
    rows = []
    for y, g in f.groupby("year"):
        w = g["va"] / g["va"].sum()
        rows.append({"year": y, "mean": g["s"].mean(), "wmean": float((w * g["s"]).sum()),
                     "median": g["s"].median()})
    t = pd.DataFrame(rows).set_index("year").loc[YEARS]
    out = {k: t[k].round(5).tolist() for k in ["mean", "wmean", "median"]}
    out["cov"] = (t["wmean"] - t["mean"]).round(5).tolist()
    # KN eq. (2): within / between A64 industries, linear trends per decade
    a = d.groupby(["a64", "year"])[["staf", "va"]].sum().reset_index()
    a = a[a["va"] > 0]
    tot = a.groupby("year")["va"].transform("sum")
    a["w"] = a["va"] / tot
    a["s"] = a["staf"] / a["va"]
    full = a.groupby("a64")["year"].transform("nunique") == len(YEARS)
    a = a[full]
    within = between = 0.0
    for _, g in a.groupby("a64"):
        g = g.set_index("year").loc[YEARS]
        within += g["w"].mean() * trend(g["s"])
        between += g["s"].mean() * trend(g["w"])
    agg = d.groupby("year")[["staf", "va"]].sum()
    out["kn_decomp"] = {"total": round(100 * trend((agg["staf"] / agg["va"]).loc[YEARS]), 2),
                        "within": round(100 * within, 2), "between": round(100 * between, 2),
                        "n_ind": int(a["a64"].nunique())}
    return out


def accounting(d):
    d = d.copy()
    d["fy"] = d["naics4"].astype(str) + "_" + d["year"].astype(str)
    d["size_dec"] = np.minimum(np.ceil(d.groupby("fy")["rev"].rank(pct=True) * 10), 10).astype(int)
    d["age"] = d["year"] - d["inc_year"].astype(float)
    d["abin"] = pd.cut(d["age"], [-1, 2, 5, 10, 20, 150], labels=["0-2", "3-5", "6-10", "11-20", "21+"]).astype(str)

    def split(g, R):
        L, Y, K = g["staf"].sum(), g["va"].sum(), g["k"].sum()
        sL, sK = L / Y, R * K / Y
        return {"sL": sL, "sK": sK, "sPi": 1 - sL - sK, "mu": 1 / (sL + sK), "sL_cost": L / (L + R * K),
                "dep": g["depr"].sum() / Y, "op": g["op"].sum() / Y}

    out = {}
    for key, col, rows in [("size", "size_dec", list(range(1, 11))),
                           ("age", "abin", ["0-2", "3-5", "6-10", "11-20", "21+"])]:
        res = {}
        for R, tag in [(R_KN, "kn"), (R_LOW, "low")]:
            t = pd.DataFrame({r: split(d[d[col] == r], R) for r in rows}).T
            res[tag] = {c: t[c].round(5).tolist() for c in t.columns}
        res["rows"] = rows
        res["n"] = d.groupby(col).size().reindex(rows).tolist()
        out[key] = res
    tot = split(d, R_KN)
    out["total_kn"] = {k: round(v, 5) for k, v in tot.items()}
    out["total_low"] = {k: round(v, 5) for k, v in split(d, R_LOW).items()}
    out["R_kn"], out["R_low"] = round(R_KN, 4), R_LOW
    return out


def kn_regression(d):
    defl = pd.read_csv(DEFL)
    hicp = defl[defl["series"] == "hicp"].set_index("year")["index_raw"]
    g = defl[defl["series"] == "gfcf_defl"].pivot_table(index="year", columns="code", values="index_raw")
    rel = np.log(g.div(hicp, axis=0)).loc[YEARS]
    s = d.groupby(["nace_section", "year"])[["staf", "va"]].sum().reset_index()
    s["s"] = s["staf"] / s["va"]
    pts = []
    for sec, t in s.groupby("nace_section"):
        if sec in DROP_SEC or sec not in rel.columns or t["year"].nunique() < len(YEARS):
            continue
        t = t.set_index("year").loc[YEARS]
        sbar = t["s"].mean()
        lhs = sbar / (1 - sbar) * trend(np.log(t["s"])) * 100     # KN eq. (19) LHS, per decade x 100
        xi = trend(rel[sec]) * 100                                 # trend in log relative price
        pts.append({"sec": sec, "x": round(xi, 3), "y": round(lhs, 3), "sL": round(sbar, 4),
                    "va": float(t["va"].sum())})
    p = pd.DataFrame(pts)
    w = p["va"] / p["va"].sum()

    def wls(weights):
        X = np.column_stack([np.ones(len(p)), p["x"]])
        W = np.diag(weights)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ p["y"])
        e = p["y"] - X @ beta
        # heteroskedasticity-robust (HC1) standard errors
        bread = np.linalg.inv(X.T @ W @ X)
        meat = X.T @ W @ np.diag(e ** 2) @ W @ X
        V = bread @ meat @ bread * len(p) / (len(p) - 2)
        return float(beta[0]), float(beta[1]), float(np.sqrt(V[1, 1]))

    a_w, b_w, se_w = wls(w.values)
    a_o, b_o, se_o = wls(np.ones(len(p)))
    tot = trend(np.log(g["TOTAL"].div(hicp).loc[YEARS])) * 100
    return {"points": p.drop(columns="va").to_dict("records"),
            "wls": {"const": round(a_w, 3), "slope": round(b_w, 3), "se": round(se_w, 3)},
            "ols": {"const": round(a_o, 3), "slope": round(b_o, 3), "se": round(se_o, 3)},
            "kn_slope": 0.28, "trend_rel_price_total": round(tot, 2), "dropped": DROP_SEC}


def main():
    d, n0 = load()
    na = pd.read_csv(NA)
    out = {"N": int(len(d)), "firms": int(d["bvdid"].nunique()), "n_base": int(n0),
           "agg": aggregate(d, na), "aggr": aggregation(d), "acct": accounting(d),
           "reg": kn_regression(d)}
    (OUT / "kn_results.json").write_text(json.dumps(out))
    A = out["agg"]
    print(f"N {out['N']:,} of {n0:,} firm-years, {out['firms']:,} firms")
    print("labour share orbis:", [round(v, 3) for v in A["orbis"]])
    print("labour share NA   :", [round(v, 3) for v in A["na"]])
    print("NA factor cost    :", [round(v, 3) for v in A["na_fc"]])
    print("coverage VA / D1  :", A["cov_va"], A["cov_l"])
    print("balanced:", [round(v, 3) for v in A["orbis_bal"]], "trend", A["trend_orbis_bal"], "firms", A["bal_firms"], "va share", A["bal_va_share"])
    print("trends pp/decade orbis, na, ex2020:", A["trend_orbis"], A["trend_na"], A["trend_orbis_ex2020"], A["trend_na_ex2020"])
    G = out["aggr"]
    print("mean / wmean / median:", [round(v, 3) for v in G["mean"]], [round(v, 3) for v in G["wmean"]], [round(v, 3) for v in G["median"]])
    print("KN decomposition:", G["kn_decomp"])
    C = out["acct"]
    print("total (KN R):", C["total_kn"]); print("total (low R):", C["total_low"])
    for k in ["size", "age"]:
        print(k, pd.DataFrame(C[k]["kn"], index=C[k]["rows"]).round(3).to_string())
        print(k, "low R mu:", [round(v, 3) for v in C[k]["low"]["mu"]])
    print("regression:", out["reg"]["wls"], out["reg"]["ols"], "rel price total trend", out["reg"]["trend_rel_price_total"])
    print(pd.DataFrame(out["reg"]["points"]).to_string())


if __name__ == "__main__":
    main()
