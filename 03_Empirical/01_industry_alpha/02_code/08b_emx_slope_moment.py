"""Compute the EMX-oligopoly slope moment on our Compustat sample.

EMX (2023) oligopoly section (eq. 83 + p. 1661) targets

    b̂ = -(1/η - 1/γ)        from the regression

    Δ(sector inverse cost-weighted markup)  on  Δ(sector HHI)

in changes over time across sectors, with HHI = Σ_j s_j². Their target value is
b̂ = -0.21 from Autor, Dorn, Katz, Patterson, Van Reenen (2020), Table II,
baseline col. 3 — US Census of Manufactures 4-digit, 1982-2012, 5-year LDs.

We compute the analogous moment on our Compustat NAICS2 panel. Specs:

  1. Long-difference (LD): one observation per sector, Δ from window start to end.
     Authoritative version when the window is short (our calibration window has
     N=14 sectors); FE specs are degenerate at this N.
  2. 5-year stacked LD: matches Autor's interval cadence.

Year-on-year FDs are deliberately excluded — they pick up business-cycle
co-movement rather than the secular reallocation the EMX/Autor target
captures, and consistently flip sign vs the LD specs.

Output:
  03_outdata/emx_slope_moment.yaml   — single primary value + provenance
  03_outdata/emx_slope_robustness.csv — table of all (window × spec) combos
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

ROOT = Path(__file__).resolve().parent.parent
INTER = ROOT / "01_intermediary"
OUTDATA = ROOT / "03_outdata"
OUTDATA.mkdir(exist_ok=True)


# ---- Load -----------------------------------------------------------------
firm = pd.read_parquet(INTER / "s2_firm_year_shares.parquet")
sec  = pd.read_parquet(INTER / "s2_sector_year_compustat.parquet")

# HHI per (sector, year) from firm sales shares.
firm = firm[["sector_id", "year", "s_ji"]].dropna()
hhi = (firm.assign(s2=firm.s_ji ** 2)
            .groupby(["sector_id", "year"])["s2"].sum()
            .rename("hhi").reset_index())

# Sector-level inverse cost-weighted markup. 1/mu_cw is the sales-weighted
# harmonic of firm inverse markups (by EMX appendix A identity), so it
# matches their μ_l(s) object exactly.
sec["inv_mu"] = 1.0 / sec["mu_cw_iy"]
panel = sec[["sector_id", "year", "inv_mu"]].merge(hhi, on=["sector_id", "year"])
panel["year"] = panel["year"].astype(int)


# ---- Regressions ----------------------------------------------------------
def _ols(X: pd.Series, y: pd.Series, fe: pd.Series | None = None) -> dict:
    rhs = sm.add_constant(X.rename("d_hhi"))
    if fe is not None:
        D = pd.get_dummies(fe, drop_first=True, prefix="sec").astype(float)
        rhs = pd.concat([rhs, D], axis=1)
    m = sm.OLS(y.astype(float), rhs.astype(float)).fit(cov_type="HC1")
    return {
        "b_hat": float(m.params["d_hhi"]),
        "se_hc1": float(m.bse["d_hhi"]),
        "t_stat": float(m.tvalues["d_hhi"]),
        "n_obs": int(m.nobs),
        "r_squared": float(m.rsquared),
    }


def _long_diff(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    a = df[df.year == y0].set_index("sector_id")
    b = df[df.year == y1].set_index("sector_id")
    return pd.DataFrame({
        "d_inv_mu": b.inv_mu - a.inv_mu,
        "d_hhi":    b.hhi    - a.hhi,
    }).dropna()


def _stacked_lds(df: pd.DataFrame, y0: int, y1: int, gap: int = 5) -> pd.DataFrame:
    rows = []
    for s in df.sector_id.unique():
        ds = df[df.sector_id == s].set_index("year")
        for ystart in range(y0, y1 - gap + 1):
            if ystart in ds.index and (ystart + gap) in ds.index:
                rows.append({
                    "sector_id": s, "y_start": ystart,
                    "d_inv_mu": ds.inv_mu[ystart + gap] - ds.inv_mu[ystart],
                    "d_hhi":    ds.hhi[ystart + gap]    - ds.hhi[ystart],
                })
    return pd.DataFrame(rows).dropna()


def run_window(panel: pd.DataFrame, y0: int, y1: int, tag: str) -> list[dict]:
    w = panel[(panel.year >= y0) & (panel.year <= y1)].copy()
    out: list[dict] = []

    ld = _long_diff(w, y0, y1)
    r = _ols(ld.d_hhi, ld.d_inv_mu)
    out.append({"window": tag, "years": f"{y0}-{y1}", "spec": "long_diff", "fe": "none", **r})

    st = _stacked_lds(w, y0, y1, gap=5)
    if not st.empty:
        r = _ols(st.d_hhi, st.d_inv_mu)
        out.append({"window": tag, "years": f"{y0}-{y1}", "spec": "5yr_stacked_LD", "fe": "none", **r})
        r = _ols(st.d_hhi, st.d_inv_mu, fe=st.sector_id)
        out.append({"window": tag, "years": f"{y0}-{y1}", "spec": "5yr_stacked_LD", "fe": "sector", **r})
    return out


rows: list[dict] = []
rows += run_window(panel, 2010, 2019, "primary")
rows += run_window(panel, 2000, 2019, "extended")
rows += run_window(panel, 1997, 2023, "full")

robust = pd.DataFrame(rows)
robust = robust[["window", "years", "spec", "fe",
                 "b_hat", "se_hc1", "t_stat", "n_obs", "r_squared"]]
print("\n== EMX-oligopoly slope (Δ inv-μ on Δ HHI) — robustness table ==\n")
print(robust.to_string(index=False, float_format="%.4f"))


# ---- Pick the primary moment ---------------------------------------------
primary_row = robust[(robust.window == "primary")
                     & (robust.spec == "long_diff")].iloc[0]
b_primary = float(primary_row["b_hat"])
print(f"\n== Primary moment for calibration ==")
print(f"  spec:    long-difference, 2010-2019, 1 obs/sector")
print(f"  b̂:       {b_primary:+.4f}")
print(f"  se(HC1): {primary_row['se_hc1']:.4f}")
print(f"  t:       {primary_row['t_stat']:+.2f}")
print(f"  N:       {int(primary_row['n_obs'])}")


# ---- Persist --------------------------------------------------------------
robust.to_csv(OUTDATA / "emx_slope_robustness.csv", index=False)
print(f"\nWrote {OUTDATA / 'emx_slope_robustness.csv'}")

moment_yaml = {
    "name": "emx_oligopoly_slope",
    "description":
        "Slope of Δ(sector cost-weighted inverse markup) on Δ(sector HHI) "
        "across sectors. Static-model identification per EMX (2023) eq. 83: "
        "b = -(1/η - 1/γ).",
    "value": b_primary,
    "se_hc1": float(primary_row["se_hc1"]),
    "spec": "long_difference",
    "fe": "none",
    "window": [2010, 2019],
    "n_sectors": int(primary_row["n_obs"]),
    "source_data": "Compustat NAICS2, this project's empirical pipeline (s2_*).",
    "reference_value": {
        "value": -0.21,
        "source": "Autor, Dorn, Katz, Patterson, Van Reenen (2020), Table II col. 3",
        "adopted_by": "Edmond, Midrigan, Xu (2023), oligopoly section, footnote 28",
        "data": "US Census of Manufactures, 4-digit, 1982-2012, 5-year LD",
    },
    "robustness_table_path": "emx_slope_robustness.csv",
}
with open(OUTDATA / "emx_slope_moment.yaml", "w") as f:
    yaml.safe_dump(moment_yaml, f, sort_keys=False)
print(f"Wrote {OUTDATA / 'emx_slope_moment.yaml'}")
