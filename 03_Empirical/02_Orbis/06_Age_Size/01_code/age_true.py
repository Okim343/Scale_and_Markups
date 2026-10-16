"""Age/size vs capital intensity and outsourcing in the Orbis Spain clean panel."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = "/Users/emtruzzi/Documents/Coding/Ideas/Scale_and_Markups_static_v2/03_Empirical/02_Orbis"
cols = ["bvdid", "year", "sector_id", "naics4", "excluded_sector", "rev", "mate", "staf", "tfas", "fias", "empl"]
d = pd.read_parquet(f"{ROOT}/01_Cleaning/01_intermediary/orbis_es_clean_panel.parquet", columns=cols)
d = d[(d.excluded_sector == 0) & d.year.between(2015, 2023) & (d.tfas > 0)].copy()

# True age from incorporation year (carried into the clean panel).
inc = pd.read_parquet(f"{ROOT}/01_Cleaning/01_intermediary/orbis_es_clean_panel.parquet",
                      columns=["bvdid", "inc_year"]).drop_duplicates("bvdid").set_index("bvdid").inc_year
d = d.join(inc, on="bvdid")
print(f"inc_year coverage: {d.inc_year.notna().mean():.3f}")
d["age"] = d.year - d.inc_year
d = d[d.age.between(0, 150)].copy()
d["age_bin"] = pd.cut(d.age, [-1, 2, 5, 10, 20, 150],
                      labels=["0-2", "3-5", "6-10", "11-20", "21+"]).astype(str)
# Outcomes
d["m_share"] = d.mate / d.rev                     # purchased inputs / sales: outsourcing
d["va"] = d.rev - d.mate
d["l_share"] = d.staf / d.rev
d["k_rev"] = np.log(d.tfas / d.rev)
d["k_wage"] = np.log(d.tfas / d.staf)             # K per unit of labor cost
d["k_emp"] = np.log(d.tfas / d.empl)
d["ln_k"] = np.log(d.tfas)
d["ln_rev"] = np.log(d.rev)
d = d[d.m_share.between(0, 1)]                    # m_share > 1 rows are accounting noise
d["fy"] = d.naics4.astype(str) + "_" + d.year.astype(str)
d["size_dec"] = np.minimum(np.ceil(d.groupby("fy").ln_rev.rank(pct=True) * 10), 10).astype(int)

# GNR elasticities
g = pd.read_parquet(f"{ROOT}/02_GNR/01_intermediary/firm_year_rts.parquet",
                    columns=["bvdid", "year", "melast", "kelast", "lelast", "rts", "rts_outlier_flag"])
g = g[~g.rts_outlier_flag.astype(bool)].drop(columns="rts_outlier_flag")
d = d.merge(g, on=["bvdid", "year"], how="left")

print(f"N = {len(d):,} firm-years, {d.bvdid.nunique():,} firms; GNR matched {d.melast.notna().mean():.2f}")
print(d.age_bin.value_counts().sort_index().to_string(), "\n")

Y = ["m_share", "k_rev", "k_wage", "k_emp", "ln_k", "ln_rev", "melast", "kelast"]
ORDER = ["0-2", "3-5", "6-10", "11-20", "21+"]
med = lambda df, by: df.groupby(by)[Y].median().round(3)
print("== Medians by age bin ==\n", med(d, "age_bin").loc[ORDER].to_string(), "\n")
print("== Medians by within-NAICS4-year revenue decile ==\n", med(d, "size_dec").to_string(), "\n")
d["size3"] = pd.cut(d.size_dec, [0, 3, 7, 10], labels=["small (D1-3)", "mid (D4-7)", "large (D8-10)"])
for y in ["m_share", "k_rev", "k_wage"]:
    print(f"== Median {y}: age bin x size ==\n",
          d.pivot_table(index="age_bin", columns="size3", values=y, aggfunc="median", observed=True)
          .loc[ORDER].round(3).to_string(), "\n")


def fe_reg(df, y, x_cols):
    """OLS of y on x_cols, absorbing NAICS4 x year FE by demeaning; firm-clustered SE."""
    s = df[[y, "fy", "bvdid"] + x_cols].dropna()
    Z = s[[y] + x_cols].astype(float)
    Z = Z - Z.groupby(s.fy).transform("mean")
    res = sm.OLS(Z[y], Z[x_cols]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.bvdid)[0]})
    return res.params, res.bse, len(s)


age_d = pd.get_dummies(d.age_bin)[["0-2", "3-5", "6-10", "11-20"]].astype(float)   # base: 21+
size_d = pd.get_dummies(d.size_dec, prefix="D").drop(columns="D_1").astype(float)  # base: D1
d = pd.concat([d, age_d, size_d], axis=1)
X = list(age_d.columns) + list(size_d.columns)
rows = {}
for y in Y:
    b, se, n = fe_reg(d, y, X)
    rows[y] = {k: f"{b[k]:+.3f} ({se[k]:.3f})" for k in ["0-2", "3-5", "6-10", "11-20", "D_5", "D_10"]} | {"N": f"{n:,}"}
print("== Joint regression: y on age bins (base: 21+) + size deciles (base: D1), NAICS4 x year FE ==")
print(pd.DataFrame(rows).T.to_string())

# Within-firm: how do shares move as a firm ages/grows? (firm FE, year FE via demeaning by year after firm)
w = d[["bvdid", "year", "m_share", "k_rev", "ln_rev", "age", "inc_year"]].copy()
w = w[w.inc_year >= 2010]
for c in ["m_share", "k_rev", "ln_rev", "age"]:
    w[c + "_w"] = w[c] - w.groupby("bvdid")[c].transform("mean")
print("\n== Within-firm (born 2010+), firm FE: slope on age, and on ln_rev ==")
for y in ["m_share", "k_rev"]:
    for x in ["age", "ln_rev"]:
        r = sm.OLS(w[y + "_w"], w[[x + "_w"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(w.bvdid)[0]})
        print(f"  {y} on {x}: {r.params.iloc[0]:+.4f} ({r.bse.iloc[0]:.4f})")
