import json, numpy as np, pandas as pd
ROOT = "/Users/emtruzzi/Documents/Coding/Ideas/Scale_and_Markups_static_v2/03_Empirical/02_Orbis"
d = pd.read_parquet(f"{ROOT}/01_Cleaning/01_intermediary/orbis_es_clean_panel.parquet",
    columns=["bvdid","year","naics4","sector_id","excluded_sector","rev","mate","staf","tfas","fias","empl","inc_year"])
d = d[(d.excluded_sector==0)&d.year.between(2015,2023)&(d.tfas>0)].copy()
d["age"] = d.year - d.inc_year; d = d[d.age.between(0,150)]
d["m_share"] = d.mate/d.rev; d = d[d.m_share.between(0,1)]
d["k_rev"] = np.log(d.tfas/d.rev); d["k_emp"] = np.log(d.tfas/d.empl); d["ln_rev"] = np.log(d.rev)
d["fy"] = d.naics4.astype(str)+"_"+d.year.astype(str)
d["size_dec"] = np.minimum(np.ceil(d.groupby("fy").ln_rev.rank(pct=True)*10),10).astype(int)
d["a"] = np.minimum(d.age, 30)
d["abin"] = pd.cut(d.age,[-1,2,5,10,20,150],labels=["0-2","3-5","6-10","11-20","21+"]).astype(str)
Y = ["m_share","k_rev","k_emp"]
out = {"N": int(len(d)), "firms": int(d.bvdid.nunique())}
# raw (only NAICS4-year demeaned) and conditional (NAICS4-year x size) age profiles
for y in Y:
    d[y+"_fy"] = d[y] - d.groupby("fy")[y].transform("mean")
    d[y+"_fys"] = d[y] - d.groupby(["fy","size_dec"])[y].transform("mean")
    d[y+"_fya"] = d[y] - d.groupby(["fy","abin"])[y].transform("mean")
ga = d.groupby("a")
prof = {}
for y in Y:
    raw = ga[y+"_fy"].mean(); cond = ga[y+"_fys"].mean()
    prof[y] = {"raw": (raw-raw.iloc[-1]).round(4).tolist(), "cond": (cond-cond.iloc[-1]).round(4).tolist()}
out["age"] = {"ages": list(range(31)), "n": ga.size().tolist(), **prof,
              "mean_ln_rev": (ga.ln_rev.mean()).round(3).tolist(),
              "size_dec_mean": ga.size_dec.mean().round(2).tolist()}
gs = d.groupby("size_dec")
out["size"] = {"dec": list(range(1,11)), **{y: {"raw": (gs[y+"_fy"].mean()-gs[y+"_fy"].mean().iloc[0]).round(4).tolist(),
        "cond": (gs[y+"_fya"].mean()-gs[y+"_fya"].mean().iloc[0]).round(4).tolist()} for y in Y},
        "median_m_share": gs.m_share.median().round(3).tolist(), "median_age": gs.age.median().tolist()}
# heatmap: age bin x size decile of NAICS4-year demeaned values
hm = {}
for y in ["m_share","k_rev"]:
    t = d.pivot_table(index="abin", columns="size_dec", values=y+"_fy", aggfunc="mean")
    hm[y] = t.loc[["0-2","3-5","6-10","11-20","21+"]].round(3).values.tolist()
hm["n"] = d.pivot_table(index="abin", columns="size_dec", values="year", aggfunc="size").loc[["0-2","3-5","6-10","11-20","21+"]].values.tolist()
out["heat"] = {"rows": ["0-2","3-5","6-10","11-20","21+"], **hm}
# by sector: young (0-2) minus old (21+), within NAICS4-year x size cell
sec = []
for s, g in d.groupby("sector_id"):
    row = {"sector": s, "n": int(len(g))}
    for y in ["m_share","k_rev"]:
        m = g.groupby("abin")[y+"_fys"].mean()
        row[y] = round(float(m.get("0-2",np.nan)-m.get("21+",np.nan)),4)
        ms = g.groupby("size_dec")[y+"_fya"].mean()
        row[y+"_size"] = round(float(ms.get(10,np.nan)-ms.get(1,np.nan)),4)
    sec.append(row)
out["sectors"] = sec
from pathlib import Path
json.dump(out, open(Path(__file__).resolve().parents[1] / "03_outdata" / "viz_data.json", "w"))
print(json.dumps({k: out[k] for k in ["N","firms"]}), json.dumps(sec)[:800])
