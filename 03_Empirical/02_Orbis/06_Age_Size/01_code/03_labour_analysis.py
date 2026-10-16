"""Cost split and labour measures by firm age and size (Orbis Spain, 2015-2023).

Reads 01_intermediary/panel_{baseline,zero_fill}.parquet; writes 03_outdata/labour_results.json
and CSV tables. Comparisons are within 4-digit NAICS x year, as in age_true.py:
  by age  : also within revenue decile (size held fixed)
  by size : also within age group (age held fixed)

    python3 01_code/03_labour_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
IN = HERE / "01_intermediary"
OUT = HERE / "03_outdata"
SHARES = ["s_mate", "s_oth", "s_staf", "s_depr", "s_op"]
LAB = ["ln_w", "ln_vapw", "lshare"]
ABINS = ["0-2", "3-5", "6-10", "11-20", "21+"]


def prepare(path):
    d = pd.read_parquet(path)
    d["age"] = d["year"] - d["inc_year"].astype(float)
    d = d[d["age"].between(0, 150) & (d["mate"] <= d["opre"])].copy()
    d["fy"] = d["naics4"].astype(str) + "_" + d["year"].astype(str)
    d["size_dec"] = np.minimum(np.ceil(d.groupby("fy")["rev"].rank(pct=True) * 10), 10).astype(int)
    d["abin"] = pd.cut(d["age"], [-1, 2, 5, 10, 20, 150], labels=ABINS).astype(str)
    d["a"] = np.minimum(d["age"], 30).astype(int)
    for c, num in [("s_mate", "mate"), ("s_oth", "oth"), ("s_staf", "staf"), ("s_depr", "depr"), ("s_op", "op")]:
        d[c] = d[num] / d["opre"]
    # Trim the 1% tails of the operating-profit share (losses of up to several times sales);
    # trimming one share keeps the five shares summing to one.
    lo, hi = d["s_op"].quantile([0.01, 0.99])
    d = d[d["s_op"].between(lo, hi)].copy()
    ok_l = d["empl"] > 0
    d["ln_w"] = np.where(ok_l, np.log(d["staf"] * d["defl"] / d["empl"]), np.nan)
    d["ln_vapw"] = np.where(ok_l & (d["va"] > 0), np.log(d["va"] * d["defl"] / d["empl"]), np.nan)
    ls = d["staf"] / d["va"]
    d["lshare"] = np.where((d["va"] > 0) & (ls <= 1.5), ls, np.nan)   # labour share of operating value added
    return d


def adjusted(d, cols, cell):
    """Within-cell deviation plus the grand mean: levels net of industry-year x cell mix."""
    g = d.groupby(cell)
    return pd.DataFrame({c: d[c] - g[c].transform("mean") + d[c].mean() for c in cols})


def gap(d, cols, cell):
    g = d.groupby(cell)
    return pd.DataFrame({c: d[c] - g[c].transform("mean") for c in cols})


def results(d):
    out = {"N": int(len(d)), "firms": int(d["bvdid"].nunique())}
    # Composition of a euro of sales
    adj_a = adjusted(d, SHARES, ["fy", "size_dec"]).groupby(d["abin"]).mean().loc[ABINS]
    adj_s = adjusted(d, SHARES, ["fy", "abin"]).groupby(d["size_dec"]).mean()
    out["comp"] = {"age_rows": ABINS, "size_rows": list(range(1, 11)),
                   "age": adj_a.round(5).to_dict("list"), "size": adj_s.round(5).to_dict("list"),
                   "n_age": d.groupby("abin").size().loc[ABINS].tolist(),
                   "n_size": d.groupby("size_dec").size().tolist()}
    # Labour profiles: gap to age 30+ / to D1, conditional and raw
    prof = {"ages": list(range(31)), "dec": list(range(1, 11))}
    for y in LAB:
        s = d[[y, "fy", "size_dec", "abin", "a"]].dropna()
        ca = gap(s, [y], ["fy", "size_dec"])[y].groupby(s["a"]).mean()
        ra = gap(s, [y], ["fy"])[y].groupby(s["a"]).mean()
        cs = gap(s, [y], ["fy", "abin"])[y].groupby(s["size_dec"]).mean()
        rs = gap(s, [y], ["fy"])[y].groupby(s["size_dec"]).mean()
        prof[y] = {"age_cond": (ca - ca.iloc[-1]).round(5).tolist(), "age_raw": (ra - ra.iloc[-1]).round(5).tolist(),
                   "size_cond": (cs - cs.iloc[0]).round(5).tolist(), "size_raw": (rs - rs.iloc[0]).round(5).tolist(),
                   "n": int(len(s))}
    out["lab"] = prof
    out["medians"] = {"w_eur": float(np.exp(d["ln_w"].median())), "vapw_eur": float(np.exp(d["ln_vapw"].median())),
                      "lshare": float(d["lshare"].median())}
    return out


def main():
    res = {}
    for name in ["baseline", "zero_fill"]:
        d = prepare(IN / f"panel_{name}.parquet")
        res[name] = results(d)
        print(f"{name}: N={res[name]['N']:,}")
    b, z = res["baseline"], res["zero_fill"]
    diff_comp = max(abs(np.array(b["comp"][k][c]) - np.array(z["comp"][k][c])).max()
                    for k in ["age", "size"] for c in SHARES)
    diff_lab = {y: max(abs(np.array(b["lab"][y][k]) - np.array(z["lab"][y][k])).max()
                       for k in ["age_cond", "size_cond"]) for y in LAB}
    robust = {"N_zero_fill": z["N"], "max_abs_diff_comp_pp": round(100 * diff_comp, 3),
              "max_abs_diff_lab": {k: round(v, 4) for k, v in diff_lab.items()}}
    out = dict(b, robust=robust, zero_fill={"comp": z["comp"], "lab": z["lab"]})
    (OUT / "labour_results.json").write_text(json.dumps(out))
    pd.DataFrame(b["comp"]["age"], index=ABINS).to_csv(OUT / "composition_by_age.csv")
    pd.DataFrame(b["comp"]["size"], index=b["comp"]["size_rows"]).to_csv(OUT / "composition_by_size.csv")
    print(json.dumps(robust, indent=1))
    print("composition by age (baseline):\n", pd.DataFrame(b["comp"]["age"], index=ABINS).round(3).to_string())
    print("composition by size (baseline):\n", pd.DataFrame(b["comp"]["size"], index=b["comp"]["size_rows"]).round(3).to_string())
    for y in LAB:
        L = b["lab"][y]
        print(f"{y}: age0 cond {L['age_cond'][0]:+.3f} raw {L['age_raw'][0]:+.3f} | D10 cond {L['size_cond'][9]:+.3f} raw {L['size_raw'][9]:+.3f} | n {L['n']:,}")
    print("medians", b["medians"])


if __name__ == "__main__":
    main()
