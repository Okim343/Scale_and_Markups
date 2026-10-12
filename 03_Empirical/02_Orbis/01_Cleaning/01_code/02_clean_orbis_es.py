"""Clean Spanish Orbis into a GNR-ready firm-year panel (Hubmer et al. 2025, App. C.3).

Reads (gitignored, WRDS-licensed):
    00_indata/orbis_es_financials.parquet, 00_indata/orbis_es_firms.parquet
    00_indata/deflators/deflators_es.csv   (from 01b_fetch_deflators_es.py)

Writes to 01_intermediary/ (gitignored):
    orbis_es_clean_panel.parquet   all years 2014-2024, all sectors, both capital flags
    gnr_ready_{tfas,fias}.parquet  GNR inputs: 2014-2023, retained sectors, capital > 0
    gnr_ready_{tfas,fias}.dta      same, in Stata format
    gnr_ready_tfas_with2024.*      only with --include-2024 (robustness)
and 01_intermediary/cleaning_report.md (row waterfall, cell sizes, summary stats).

Steps (decisions documented in
02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_CLEANING.md):
  1. bvdid starts with "ES"             5. revenue, mate, staf > 0; flags tfas > 0, fias > 0
  2. nr_months == "12"                  6. core NACE / NAICS present; NAICS2 cells
  3. one account per (bvdid, year)      7. deflate (HICP baseline; sector robustness)
  4. revenue = turn, else opre          8. consecutive-year indicators
                                        9. sector exclusions; window 2014-2023

    python 03_Empirical/02_Orbis/01_Cleaning/01_code/02_clean_orbis_es.py [--include-2024]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "00_indata"
OUT_DIR = ROOT / "01_intermediary"
REPORT = OUT_DIR / "cleaning_report.md"

LAG_YEAR = 2014                 # kept only as the GNR lag year
WINDOW = (2015, 2023)           # baseline analysis window
ROBUST_YEAR = 2024              # selectively incomplete; robustness only
BASE_YEAR = 2019

# NAICS2 grouping of the Compustat pipeline (01_Compustat/01_industry_alpha COMPUSTAT_TO_SECTOR_ID).
# `ind2d` carries the first code of each group so R1 gets an integer cell id.
NAICS_GROUP = {31: 31, 32: 31, 33: 31, 44: 44, 45: 44, 48: 48, 49: 48}
SECTOR_LABEL = {31: "31-33", 44: "44-45", 48: "48-49"}
# main.typ "Data and Measurement": 52/53/62/81 follow Hubmer et al., 92/99 out of scope.
# 55 (holding companies / head offices) is outside the paper's 14-cell sector set.
# 49 is kept: its Compustat reason (8-11 public firms, mu_cw < 1) does not apply here.
EXCLUDE_NAICS2 = {52: "finance", 53: "real estate", 55: "management of companies",
                  62: "health care", 81: "other services", 92: "public administration",
                  99: "unclassified"}

# NACE Rev.2 division ranges per section.
NACE_SECTIONS = {
    "A": (1, 3), "B": (5, 9), "C": (10, 33), "D": (35, 35), "E": (36, 39), "F": (41, 43),
    "G": (45, 47), "H": (49, 53), "I": (55, 56), "J": (58, 63), "K": (64, 66), "L": (68, 68),
    "M": (69, 75), "N": (77, 82), "O": (84, 84), "P": (85, 85), "Q": (86, 88), "R": (90, 93),
    "S": (94, 96), "T": (97, 98), "U": (99, 99),
}
# NAICS2 -> main NACE section. Documentation and fallback only: every retained firm has
# its own core NACE code, which assigns the robustness deflator directly.
NAICS2_TO_NACE_SECTION = {
    11: "A", 21: "B", 22: "D", 23: "F", 31: "C", 42: "G", 44: "G", 48: "H", 51: "J",
    52: "K", 53: "L", 54: "M", 55: "M", 56: "N", 61: "P", 62: "Q", 71: "R", 72: "I",
    81: "S", 92: "O",
}

MONEY = ["rev", "mate", "staf", "tfas", "fias"]
STATS_VARS = {"rev": "Revenue", "mate": "Intermediates", "staf": "Wage bill", "cap": "Capital"}
# Hubmer et al. Table A.4, Spain (logs, 2019 EUR, CPI-deflated, FIAS capital).
HUBMER_ES = pd.DataFrame(
    {"Revenue": [13.08, 1.556, 11.27, 12.92, 15.08, 17.57],
     "Intermediates": [12.17, 1.841, 9.922, 12.06, 14.53, 17.08],
     "Wage bill": [11.64, 1.466, 9.897, 11.58, 13.43, 15.74],
     "Capital": [11.07, 2.204, 8.245, 11.12, 13.76, 16.21]},
    index=["Mean", "Std. Dev.", "P10", "P50", "P90", "P99"])


class Waterfall:
    """Row / firm counts after each cleaning step."""

    def __init__(self):
        self.rows = []

    def add(self, step, df, note=""):
        self.rows.append({"step": step, "rows": len(df), "firms": df["bvdid"].nunique(),
                          "note": note})
        print(f"  {step:55s} {len(df):>10,} rows {df['bvdid'].nunique():>10,} firms")

    def table(self):
        t = pd.DataFrame(self.rows)
        t["dropped"] = (-t["rows"].diff()).fillna(0).astype(int)
        return t[["step", "rows", "dropped", "firms", "note"]]


# ---------------------------------------------------------------------------------------
# Steps 1-5: accounts
# ---------------------------------------------------------------------------------------
def load_financials():
    fin = pd.read_parquet(IN_DIR / "orbis_es_financials.parquet")
    for c in ["turn", "opre", "mate", "staf", "tfas", "fias", "empl"]:
        fin[c] = fin[c].astype("float64")
    fin["bvdid"] = fin["bvdid"].astype(str)
    fin["year"] = fin["closdate_year"].astype(int)
    for c in ["conscode", "filing_type", "nr_months", "size_tier"]:
        fin[c] = fin[c].astype(str)
    return fin


def one_account_per_firm_year(fin):
    """Keep one account per (bvdid, year): U1 before U2, then latest closdate, then a
    local-registry filing before an annual report. The raw keys are unique, so the
    order is total and the choice deterministic."""
    order = fin.assign(
        _cons=(fin["conscode"] != "U1").astype(int),
        _filing=(fin["filing_type"] != "Local registry filing").astype(int),
    ).sort_values(["bvdid", "year", "_cons", "closdate", "_filing"],
                  ascending=[True, True, True, False, True])
    n_multi = int(order.duplicated(["bvdid", "year"], keep=False).sum())
    out = order.drop_duplicates(["bvdid", "year"], keep="first")
    return out.drop(columns=["_cons", "_filing"]), n_multi


# ---------------------------------------------------------------------------------------
# Step 6: industry codes
# ---------------------------------------------------------------------------------------
def nace_division_to_section(div):
    for sec, (lo, hi) in NACE_SECTIONS.items():
        if lo <= div <= hi:
            return sec
    return None


def load_firms():
    firms = pd.read_parquet(IN_DIR / "orbis_es_firms.parquet")
    firms["bvdid"] = firms["bvdid"].astype(str)
    firms = firms[firms["bvdid"].str.startswith("ES")]
    out = pd.DataFrame({"bvdid": firms["bvdid"].values})
    out["nace3"] = firms["naceccod2"].astype("string").values          # e.g. "20.1"
    out["naics4"] = firms["naicsccod2017"].astype("string").values     # e.g. "3251"
    out["cnae4"] = firms["cnae2009_pcod"].astype("string").values      # national primary
    out["nace_div"] = pd.to_numeric(out["nace3"].str[:2], errors="coerce")
    out["nace_section"] = out["nace_div"].map(
        lambda d: nace_division_to_section(int(d)) if pd.notna(d) else None)
    out["naics2_raw"] = pd.to_numeric(out["naics4"].str[:2], errors="coerce")
    return out


# ---------------------------------------------------------------------------------------
# Step 7: deflators
# ---------------------------------------------------------------------------------------
def a64_divisions(code):
    """Divisions covered by an A64 code: 'C10-C12' -> 10..12, 'C31_C32' -> {31, 32},
    'G46' -> {46}, single-letter section 'F' -> its division range."""
    if re.fullmatch(r"[A-U]", code):
        lo, hi = NACE_SECTIONS[code]
        return set(range(lo, hi + 1))
    nums = [int(n) for n in re.findall(r"\d+", code)]
    if "-" in code:
        return set(range(nums[0], nums[1] + 1))
    return set(nums)


A64_CODES = ["A01", "A02", "A03", "B", "C10-C12", "C13-C15", "C16", "C17", "C18", "C19",
             "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30",
             "C31_C32", "C33", "D35", "E36", "E37-E39", "F", "G45", "G46", "G47", "H49",
             "H50", "H51", "H52", "H53", "I", "J58", "J59_J60", "J61", "J62_J63", "K64",
             "K65", "K66", "L68", "M69_M70", "M71", "M72", "M73", "M74_M75", "N77", "N78",
             "N79", "N80-N82", "O84", "P85", "Q86", "Q87_Q88", "R90-R92", "R93", "S94",
             "S95", "S96", "T", "U"]
DIV_TO_A64 = {d: code for code in A64_CODES for d in a64_divisions(code)}


def attach_deflators(df):
    """Add deflator columns (2019 = 1). Baseline: HICP. Robustness: A64 VA deflator
    (falling back to the section, then TOTAL, when the A64 cell is missing), section
    GFCF deflator, GDP deflator."""
    defl = pd.read_csv(IN_DIR / "deflators" / "deflators_es.csv")
    defl["p"] = defl["index_2019"] / 100.0

    def series(name):
        return defl[defl["series"] == name].set_index(["code", "year"])["p"]

    hicp = defl[defl["series"] == "hicp"].set_index("year")["p"]
    gdp = defl[defl["series"] == "gdp_defl"].set_index("year")["p"]
    va, gfcf = series("va_defl"), series("gfcf_defl")

    df["p_hicp"] = df["year"].map(hicp)
    df["p_gdp"] = df["year"].map(gdp)
    df["a64"] = df["nace_div"].map(DIV_TO_A64)
    section = df["nace_section"].fillna(df["ind2d"].map(NAICS2_TO_NACE_SECTION))

    def lookup(s, codes):
        return pd.Series(s.reindex(pd.MultiIndex.from_arrays([codes, df["year"]])).values,
                         index=df.index)

    p_va = lookup(va, df["a64"])
    df["va_defl_level"] = np.where(p_va.notna(), "A64", None)
    p_sec = lookup(va, section)
    df.loc[p_va.isna() & p_sec.notna(), "va_defl_level"] = "section"
    p_va = p_va.fillna(p_sec).fillna(lookup(va, pd.Series("TOTAL", index=df.index)))
    df["va_defl_level"] = df["va_defl_level"].fillna("TOTAL")
    df["p_va"] = p_va
    df["p_gfcf"] = lookup(gfcf, section).fillna(lookup(gfcf, pd.Series("TOTAL", index=df.index)))
    missing = df[["p_hicp", "p_gdp", "p_va", "p_gfcf"]].isna().sum()
    if missing.any():
        raise ValueError(f"missing deflator values:\n{missing}")
    return df


def deflate(df):
    """Real money variables in 2019 EUR. `_D` = HICP baseline, `_Ds` = sector robustness.
    Revenue and mate share one index in both variants, so log(mate_D / rev_D) is the
    nominal materials share that GNR's first stage needs."""
    for v in MONEY:
        df[f"{v}_D"] = df[v] / df["p_hicp"]
    df["rev_Ds"] = df["rev"] / df["p_va"]
    df["mate_Ds"] = df["mate"] / df["p_va"]
    df["staf_Ds"] = df["staf"] / df["p_gdp"]
    df["tfas_Ds"] = df["tfas"] / df["p_gfcf"]
    df["fias_Ds"] = df["fias"] / df["p_gfcf"]
    return df


# ---------------------------------------------------------------------------------------
# Step 8: panel structure
# ---------------------------------------------------------------------------------------
def add_panel_flags(df):
    """Consecutive-year indicators within the given sample (R1 rebuilds the lags itself
    and only uses a lag when the previous firm-year is exactly year - 1)."""
    df = df.sort_values(["bvdid", "year"]).reset_index(drop=True)
    g = df.groupby("bvdid", sort=False)["year"]
    df["has_prev_year"] = (g.shift(1) == df["year"] - 1).astype("int8")
    df["has_next_year"] = (g.shift(-1) == df["year"] + 1).astype("int8")
    df["n_years_firm"] = g.transform("size").astype("int16")
    df["lag_only_year"] = (df["year"] == LAG_YEAR).astype("int8")
    df["in_window"] = df["year"].between(*WINDOW).astype("int8")
    return df


GNR_COLS = ["bvdid", "year", "ind2d", "naics2_raw", "sector_id", "naics4", "nace3",
            "nace_section", "cnae4", "size_tier", "conscode", "closdate", "rev_from_opre",
            "rev", "mate", "staf", "tfas", "fias", "empl",
            "rev_D", "mate_D", "staf_D", "cap_D", "wagebill_D", "rev_Ds", "mate_Ds", "staf_Ds", "cap_Ds",
            "has_prev_year", "has_next_year", "n_years_firm", "lag_only_year", "in_window"]


def gnr_ready(clean, cap, years):
    """GNR input for one capital definition: retained sectors, capital > 0, given years."""
    df = clean[(clean["excluded_sector"] == 0) & (clean[f"pos_{cap}"] == 1)
               & clean["year"].between(*years)].copy()
    df["cap_D"] = df[f"{cap}_D"]
    df["cap_Ds"] = df[f"{cap}_Ds"]
    # R1 maps labor and wage_bill through one rename dict, so they need distinct columns.
    df["wagebill_D"] = df["staf_D"]
    df = add_panel_flags(df)
    return df[GNR_COLS]


def write_gnr(df, stem):
    df.to_parquet(OUT_DIR / f"{stem}.parquet", index=False)
    # R1 reads Stata (pd.read_stata), so also write a drop-in .dta.
    dta = df.copy()
    for c in dta.columns:
        if isinstance(dta[c].dtype, (pd.StringDtype, pd.CategoricalDtype)) or dta[c].dtype == object:
            dta[c] = dta[c].astype(object).where(dta[c].notna(), "").astype(str)
    dta.to_stata(OUT_DIR / f"{stem}.dta", write_index=False, version=118)
    print(f"  wrote {stem}.parquet / .dta ({len(df):,} rows)")


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------
def md(df, floatfmt="{:,.3f}"):
    """Minimal markdown table (no tabulate dependency)."""
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, v in zip(cols, r.values):
            if c in ("year", "naics2_raw") or pd.isna(v):
                cells.append("" if pd.isna(v) else str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def log_stats(df, suffix):
    out = {}
    for v, label in STATS_VARS.items():
        x = np.log(df[f"{v}{suffix}"])
        out[label] = [x.mean(), x.std(), *np.percentile(x, [10, 50, 90, 99])]
    return pd.DataFrame(out, index=HUBMER_ES.index)


def build_report(wf, n_multi, n_opre, clean, gnr_t, gnr_f, firms_ref):
    win_t = gnr_t[gnr_t["in_window"] == 1]
    win_f = gnr_f[gnr_f["in_window"] == 1]
    L = ["# Orbis Spain cleaning report", "",
         "Generated by `01_code/02_clean_orbis_es.py`. Decisions: "
         "`02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_CLEANING.md`. "
         "Money in EUR; real values in 2019 EUR, deflated by Spanish HICP "
         "(baseline, as Hubmer et al. App. C.3).", ""]

    L += ["## 1. Row-count waterfall", "",
          "Each row is the sample after the step. Steps 1-6 apply to all years 2014-2024; "
          "the last steps build the baseline GNR input (`gnr_ready_tfas`).", "",
          md(wf.table()), "",
          f"- Firm-years with more than one 12-month account before step 3: {n_multi:,} rows.",
          f"- Revenue taken from `opre` because `turn` is missing: {n_opre:,} rows "
          "(after step 3).", ""]

    years = pd.DataFrame({
        "clean (rev, mate, staf > 0)": clean.groupby("year").size(),
        "  of which tfas > 0": clean[clean["pos_tfas"] == 1].groupby("year").size(),
        "  of which fias > 0": clean[clean["pos_fias"] == 1].groupby("year").size(),
        "GNR tfas (retained sectors)": gnr_t.groupby("year").size(),
        "  with year - 1 present": gnr_t[gnr_t["has_prev_year"] == 1].groupby("year").size(),
        "GNR fias": gnr_f.groupby("year").size(),
    }).fillna(0).astype(int).reset_index()
    L += ["## 2. Firm-years per year", "",
          f"2014 is a lag-only year and {ROBUST_YEAR} is robustness only; neither is in the "
          f"{WINDOW[0]}-{WINDOW[1]} window. The {ROBUST_YEAR} column of the clean panel is shown "
          "for reference.", "", md(years), ""]

    cells = (clean[clean["pos_tfas"] == 1].groupby(["ind2d", "naics2_raw"])
             .agg(firm_years_all=("bvdid", "size")).reset_index())
    sh = win_t.assign(ms=win_t["mate"] / win_t["rev"], ls=win_t["staf"] / win_t["rev"])
    sh["ml_gt1"] = (sh["ms"] + sh["ls"]) > 1
    win_cells = sh.groupby(["ind2d", "naics2_raw"]).agg(
        firm_years=("bvdid", "size"), firms=("bvdid", "nunique"),
        with_lag=("has_prev_year", "sum"),
        med_mate_rev=("ms", "median"), med_staf_rev=("ls", "median"),
        p90_mate_rev=("ms", lambda s: s.quantile(0.9)),
        share_mate_rev_gt1=("ms", lambda s: (s > 1).mean()),
        share_m_plus_l_gt1=("ml_gt1", "mean")).reset_index()
    cell_tab = cells.merge(win_cells, how="left")
    for c in ["firm_years", "firms", "with_lag"]:
        cell_tab[c] = cell_tab[c].astype("Int64")
    cell_tab["retained"] = np.where(cell_tab["naics2_raw"].isin(EXCLUDE_NAICS2), "excluded", "yes")
    cell_tab["sector"] = cell_tab["ind2d"].map(lambda c: SECTOR_LABEL.get(c, str(c)))
    cell_tab = cell_tab[["sector", "naics2_raw", "retained", "firm_years_all", "firm_years",
                         "firms", "with_lag", "med_mate_rev", "med_staf_rev", "p90_mate_rev",
                         "share_mate_rev_gt1", "share_m_plus_l_gt1"]]
    grouped = (win_t.groupby("sector_id")
               .agg(firm_years=("bvdid", "size"), firms=("bvdid", "nunique"),
                    with_lag=("has_prev_year", "sum")).reset_index())
    L += ["## 3. NAICS2 cells", "",
          "`firm_years_all`: clean panel with tfas > 0, all years. Other counts: baseline GNR "
          f"input, {WINDOW[0]}-{WINDOW[1]}. Shares are nominal ratios to revenue in the window.",
          "", md(cell_tab), "", "Grouped GNR cells (`ind2d` / `sector_id`), window:", "",
          md(grouped), ""]

    excl = (clean[(clean["excluded_sector"] == 1) & (clean["pos_tfas"] == 1)
                  & clean["year"].between(*WINDOW)]
            .groupby("naics2_raw").agg(firm_years=("bvdid", "size"), firms=("bvdid", "nunique"))
            .reset_index())
    excl["reason"] = excl["naics2_raw"].map(EXCLUDE_NAICS2)
    L += ["## 4. Sector exclusions", "",
          "Firm-years dropped from the GNR input by NAICS2 (tfas > 0, window):", "",
          md(excl), ""]

    L += ["## 5. Summary statistics (logs, 2019 EUR)", "",
          "Hubmer et al. Table A.4, Spain row (CPI-deflated, FIAS capital, about 1990s-2019):", "",
          md(HUBMER_ES.reset_index().rename(columns={"index": "stat"})), "",
          f"Baseline: GNR tfas input, {WINDOW[0]}-{WINDOW[1]}, HICP-deflated "
          f"({len(win_t):,} firm-years):", "",
          md(log_stats(win_t, "_D").reset_index().rename(columns={"index": "stat"})), "",
          f"Hubmer-comparable capital: GNR fias input, HICP-deflated ({len(win_f):,} firm-years):",
          "", md(log_stats(win_f, "_D").reset_index().rename(columns={"index": "stat"})),
          "", "Sector-deflator robustness (tfas input):", "",
          md(log_stats(win_t, "_Ds").reset_index().rename(columns={"index": "stat"})), ""]

    shares = pd.DataFrame({
        "mate / rev": win_t["mate"] / win_t["rev"],
        "staf / rev": win_t["staf"] / win_t["rev"],
        "(mate + staf) / rev": (win_t["mate"] + win_t["staf"]) / win_t["rev"],
    })
    q = shares.quantile([0.01, 0.1, 0.5, 0.9, 0.99]).T
    q.columns = [f"P{int(c * 100)}" for c in q.columns]
    q["share > 1"] = (shares > 1).mean()
    q["mate/rev outside [0.05, 0.95]"] = [
        ((shares["mate / rev"] < 0.05) | (shares["mate / rev"] > 0.95)).mean(), np.nan, np.nan]
    L += ["## 6. Input shares (nominal, baseline window)", "",
          "R1 later keeps only exp(s) = mate / rev in [0.05, 0.95] and trims the bottom 1% "
          "of r, k, l, m, s; nothing is trimmed here.", "",
          md(q.reset_index().rename(columns={"index": "ratio"})), ""]

    lvl = gnr_t["va_defl_level"].value_counts() if "va_defl_level" in gnr_t else None
    L += ["## 7. Deflators", "",
          "Source: Eurostat API, raw JSON in `00_indata/deflators/`. Baseline HICP "
          "(`prc_hicp_aind`, CP00, 2019 = 100) for all money variables. Robustness `_Ds`: "
          "revenue and mate by the firm's A64 gross-value-added deflator (`nama_10_a64`, B1G, "
          "PD15_NAC), capital by the section GFCF deflator (`nama_10_a64_p5`, P51G, N11G), "
          "staf by the GDP deflator (`nama_10_gdp`, B1GQ). Firms are mapped to A64 by their "
          "own core NACE division; NAICS2 -> NACE section is used only as a fallback.", ""]
    xt = (pd.crosstab(win_t["sector_id"], win_t["nace_section"], normalize="index") * 100)
    top = xt.apply(lambda r: ", ".join(f"{k} {v:.0f}%" for k, v in
                                       r.sort_values(ascending=False).head(3).items() if v >= 1),
                   axis=1)
    cw = pd.DataFrame({"sector_id": top.index,
                       "crosswalk section": [NAICS2_TO_NACE_SECTION[int(s[:2])] for s in top.index],
                       "firm-years by own NACE section (top 3)": top.values})
    L += ["NAICS2 -> NACE section crosswalk vs firms' own core NACE codes (window):", "",
          md(cw), ""]

    nf = firms_ref
    L += ["## 8. Industry-code checks", "",
          f"- Spanish firms in the firms file: {len(nf):,}. Core NACE (`naceccod2`) missing: "
          f"{int(nf['nace3'].isna().sum()):,}; core NAICS missing: {int(nf['naics4'].isna().sum()):,}.",
          "- `naceccod2` is a **3-digit NACE group** (e.g. `20.1`) for every firm, not a 4-digit "
          "class. The only 4-digit code is `cnae2009_pcod` (national CNAE 2009 primary, = NACE "
          f"Rev.2 class), missing for {int(nf['cnae4'].isna().sum()):,} firms. Both are carried "
          "as `nace3` and `cnae4`.",
          "- `naicsccod2017` is a 4-digit NAICS industry group; NAICS2 = its first two digits.", ""]

    REPORT.write_text("\n".join(L))
    print(f"  wrote {REPORT.name}")


# ---------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-2024", action="store_true",
                        help="also write gnr_ready_tfas_with2024 (robustness)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf = Waterfall()

    fin = load_financials()
    wf.add("0. raw extract (2014-2024, U1/U2)", fin)
    fin = fin[fin["bvdid"].str.startswith("ES")]
    wf.add("1. bvdid starts with ES", fin)
    fin = fin[fin["nr_months"] == "12"]
    wf.add("2. nr_months == 12", fin)
    fin, n_multi = one_account_per_firm_year(fin)
    wf.add("3. one account per firm-year", fin, "U1 > U2, latest closdate")

    fin["rev_from_opre"] = (fin["turn"].isna() & fin["opre"].notna()).astype("int8")
    fin["rev"] = fin["turn"].fillna(fin["opre"])
    n_opre = int(fin["rev_from_opre"].sum())
    for i, v in enumerate(["rev", "mate", "staf"]):
        fin = fin[fin[v] > 0]
        wf.add(f"5{'abc'[i]}. {v} > 0", fin, "rev = turn, else opre" if v == "rev" else "")
    fin["pos_tfas"] = (fin["tfas"] > 0).astype("int8")
    fin["pos_fias"] = (fin["fias"] > 0).astype("int8")

    firms = load_firms()
    df = fin.merge(firms, on="bvdid", how="left", validate="many_to_one")
    df = df[df["nace3"].notna() & df["naics2_raw"].notna()]
    df["naics2_raw"] = df["naics2_raw"].astype(int)
    df["ind2d"] = df["naics2_raw"].map(lambda c: NAICS_GROUP.get(c, c)).astype(int)
    df["sector_id"] = df["ind2d"].map(lambda c: SECTOR_LABEL.get(c, str(c)))
    df["excluded_sector"] = df["naics2_raw"].isin(EXCLUDE_NAICS2).astype("int8")
    wf.add("6. core NACE and NAICS present", df)

    df = deflate(attach_deflators(df))
    keep = (["bvdid", "year", "closdate", "conscode", "filing_type", "size_tier",
             "ind2d", "naics2_raw", "sector_id", "naics4", "nace3", "nace_div", "nace_section",
             "cnae4", "a64", "va_defl_level", "excluded_sector", "pos_tfas", "pos_fias",
             "rev_from_opre", "turn", "opre", "empl"] + MONEY
            + [f"{v}_D" for v in MONEY] + ["rev_Ds", "mate_Ds", "staf_Ds", "tfas_Ds", "fias_Ds"]
            + ["p_hicp", "p_gdp", "p_va", "p_gfcf"])
    clean = df[keep].sort_values(["bvdid", "year"]).reset_index(drop=True)
    clean.to_parquet(OUT_DIR / "orbis_es_clean_panel.parquet", index=False)
    print(f"  wrote orbis_es_clean_panel.parquet ({len(clean):,} rows)")

    t = clean[clean["pos_tfas"] == 1]
    wf.add("5d. tfas > 0 (baseline capital)", t, f"fias > 0 instead: {int(clean['pos_fias'].sum()):,} rows")
    t = t[t["excluded_sector"] == 0]
    wf.add("9a. drop excluded NAICS2 sectors", t, ", ".join(map(str, EXCLUDE_NAICS2)))
    t = t[t["year"] <= WINDOW[1]]
    wf.add(f"9b. drop {ROBUST_YEAR} (robustness year) = GNR input", t,
           f"includes {int((t['year'] == LAG_YEAR).sum()):,} lag-only {LAG_YEAR} rows")
    wf.add(f"   memo: window {WINDOW[0]}-{WINDOW[1]} only", t[t["year"] >= WINDOW[0]])

    gnr_t = gnr_ready(clean, "tfas", (LAG_YEAR, WINDOW[1]))
    gnr_f = gnr_ready(clean, "fias", (LAG_YEAR, WINDOW[1]))
    assert len(gnr_t) == len(t)
    write_gnr(gnr_t, "gnr_ready_tfas")
    write_gnr(gnr_f, "gnr_ready_fias")
    if args.include_2024:
        write_gnr(gnr_ready(clean, "tfas", (LAG_YEAR, ROBUST_YEAR)), "gnr_ready_tfas_with2024")

    gnr_t = gnr_t.merge(clean[["bvdid", "year", "va_defl_level"]], on=["bvdid", "year"])
    build_report(wf, n_multi, n_opre, clean, gnr_t, gnr_f, firms)


if __name__ == "__main__":
    main()
