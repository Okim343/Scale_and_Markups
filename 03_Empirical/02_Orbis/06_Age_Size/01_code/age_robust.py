import numpy as np, pandas as pd, statsmodels.api as sm
ROOT = "/Users/emtruzzi/Documents/Coding/Ideas/Scale_and_Markups_static_v2/03_Empirical/02_Orbis"
d = pd.read_parquet(f"{ROOT}/01_Cleaning/01_intermediary/orbis_es_clean_panel.parquet",
    columns=["bvdid","year","naics4","excluded_sector","rev","mate","tfas","fias","inc_year"])
d = d[(d.excluded_sector==0)&d.year.between(2015,2023)&(d.tfas>0)&(d.fias>0)].copy()
d["age"] = d.year - d.inc_year; d = d[d.age.between(0,150)]
d["m_share"] = d.mate/d.rev; d = d[d.m_share.between(0,1)]
d["k_rev"] = np.log(d.tfas/d.rev); d["kf_rev"] = np.log(d.fias/d.rev); d["ln_rev"]=np.log(d.rev)
d["fy"] = d.naics4.astype(str)+"_"+d.year.astype(str)
d["size_dec"] = np.minimum(np.ceil(d.groupby("fy").ln_rev.rank(pct=True)*10),10).astype(int)
d["a"] = np.minimum(d.age, 30)
# age profile net of size decile x NAICS4-year: residualize on fy x size cells, then mean by age
for y in ["m_share","k_rev","kf_rev"]:
    d[y+"_r"] = d[y] - d.groupby(["fy","size_dec"])[y].transform("mean")
prof = d.groupby("a")[["m_share_r","k_rev_r","kf_rev_r"]].mean()
prof = prof - prof.iloc[-1]   # relative to age 30+
print("== Age profile net of NAICS4-year x size decile, relative to age 30+ ==")
print(prof.loc[[0,1,2,3,5,7,10,15,20,25,30]].round(3).to_string())
