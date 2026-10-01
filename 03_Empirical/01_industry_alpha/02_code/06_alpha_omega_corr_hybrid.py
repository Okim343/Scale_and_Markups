"""
Stage S6 — Within-sector rank correlation between the HYBRID firm α
(`alpha_hybrid_sector_a`) and GNR productivity ω, from the S4b panel.

This is the copula moment ρ̄ the solver consumes (model/pool.py): the fixed,
ex-ante Gaussian-copula α–z rank correlation. It MUST be recomputed on the
hybrid alpha — using the raw-GNR ρ̄ with a hybrid support would be internally
inconsistent (plan §3.3).

Method ported from 03_Empirical/00_Hubmer_RTS/01_code/05_alpha_omega_corr.py
(firm-mean collapse, within-sector winsor, pooled within-sector Spearman,
2·sin(π·ρ_S/6) Gaussian-copula mapping, odd/even split-sample check), reading
`alpha_hybrid_sector_a`/`omega` from the S4b panel over the active window.

Reads:
  01_intermediary/s4b_firm_year_alpha_hybrid.parquet

Writes:
  03_outdata/naics_alpha_omega_corr_hybrid.csv   (per-NAICS2 correlations)
  03_outdata/alpha_omega_corr_hybrid.json        (pooled values + diagnostics)

Run:
  python 06_alpha_omega_corr_hybrid.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import (
    PATHS,
    active_window,
    ensure_dir,
    load_config,
    setup_logger,
)


ALPHA_COL = "alpha_hybrid_sector_a"
CSV_OUT = "naics_alpha_omega_corr_hybrid.csv"
JSON_OUT = "alpha_omega_corr_hybrid.json"


def winsorize_within(df: pd.DataFrame, col: str, by: str, pct) -> pd.Series:
    lo, hi = pct

    def _w(s: pd.Series) -> pd.Series:
        a, b = s.quantile([lo, hi])
        return s.clip(lower=a, upper=b)

    return df.groupby(by)[col].transform(_w)


def collapse_to_firms(fy: pd.DataFrame) -> pd.DataFrame:
    """One unit per (ind2d, gvkey): firm-mean alpha and omega."""
    return fy.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=(ALPHA_COL, "mean"),
        omega=("omega", "mean"),
        n_years=("year", "size"),
    )


def sector_correlations(firms: pd.DataFrame, min_firms: int) -> pd.DataFrame:
    rows = []
    for ind, g in firms.groupby("ind2d"):
        if len(g) < min_firms:
            rows.append((ind, len(g), np.nan, np.nan))
            continue
        rho_s = spearmanr(g["alpha"], g["omega"]).statistic
        rho_p = float(np.corrcoef(g["alpha"], g["omega"])[0, 1])
        rows.append((ind, len(g), float(rho_s), rho_p))
    return pd.DataFrame(
        rows, columns=["NAICS", "n_firms", "rho_spearman", "rho_pearson"]
    )


def pooled_within_spearman(firms: pd.DataFrame) -> float:
    """Rank within sector, then correlate pooled ranks (a within-sector
    pooled rank correlation — sector-level differences are removed)."""
    ra = firms.groupby("ind2d")["alpha"].rank(pct=True)
    ro = firms.groupby("ind2d")["omega"].rank(pct=True)
    return float(np.corrcoef(ra, ro)[0, 1])


def split_sample_firms(fy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Firm units with α from odd years and ω from even years, and the
    reverse. Only firms with at least one year in each half survive."""
    odd = fy[fy["year"] % 2 == 1]
    even = fy[fy["year"] % 2 == 0]
    a_odd = odd.groupby(["ind2d", "gvkey"], as_index=False).agg(alpha=(ALPHA_COL, "mean"))
    o_even = even.groupby(["ind2d", "gvkey"], as_index=False).agg(omega=("omega", "mean"))
    a_even = even.groupby(["ind2d", "gvkey"], as_index=False).agg(alpha=(ALPHA_COL, "mean"))
    o_odd = odd.groupby(["ind2d", "gvkey"], as_index=False).agg(omega=("omega", "mean"))
    fw = a_odd.merge(o_even, on=["ind2d", "gvkey"], how="inner")
    bw = a_even.merge(o_odd, on=["ind2d", "gvkey"], how="inner")
    return fw, bw


def main() -> None:
    config = load_config()
    logger = setup_logger("S6", config.get("log_level", "INFO"))
    cfg_c = config.get("alpha_omega_corr", {})
    min_firms = int(cfg_c.get("min_firms_per_sector", 30))
    winsor_pct = cfg_c.get("winsor_pct", [0.01, 0.99])

    fy_path = PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet"
    if not fy_path.exists():
        raise FileNotFoundError(
            f"{fy_path} missing; run 04b_build_hybrid_alpha.py first."
        )
    fy = pd.read_parquet(
        fy_path, columns=["gvkey", "year", "ind2d", ALPHA_COL, "omega"]
    ).dropna(subset=[ALPHA_COL, "omega"])
    logger.info(f"loaded {fy_path.relative_to(PATHS.project_root)} ({len(fy):,} rows)")

    y0, y1 = active_window(config)
    fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)]
    logger.info(f"  active window [{y0}, {y1}]: {len(fy):,} firm-year rows")
    if len(fy) == 0:
        raise ValueError("no firm-year rows in the correlation window")

    # --- full-sample firm collapse ------------------------------------------
    firms = collapse_to_firms(fy)
    firms["alpha"] = winsorize_within(firms, "alpha", "ind2d", winsor_pct)
    firms["omega"] = winsorize_within(firms, "omega", "ind2d", winsor_pct)
    logger.info(f"  firm units: {len(firms):,} across {firms['ind2d'].nunique()} sectors")

    per_sector = sector_correlations(firms, min_firms)
    pooled_s = pooled_within_spearman(firms)
    covered = per_sector.dropna(subset=["rho_spearman"])
    fw_avg = (
        float(np.average(covered["rho_spearman"], weights=covered["n_firms"]))
        if len(covered)
        else np.nan
    )

    # --- split-sample check (mechanical-correlation diagnostic) -------------
    fw, bw = split_sample_firms(fy)
    split_rhos = {}
    for name, d in [("alpha_odd_omega_even", fw), ("alpha_even_omega_odd", bw)]:
        d["alpha"] = winsorize_within(d, "alpha", "ind2d", winsor_pct)
        d["omega"] = winsorize_within(d, "omega", "ind2d", winsor_pct)
        split_rhos[name] = {
            "pooled_within_spearman": pooled_within_spearman(d),
            "n_firms": int(len(d)),
        }
    split_avg = float(np.mean([v["pooled_within_spearman"] for v in split_rhos.values()]))

    # --- report -------------------------------------------------------------
    logger.info(
        "  per-sector correlations:\n"
        + per_sector.round({"rho_spearman": 3, "rho_pearson": 3}).to_string(index=False)
    )
    logger.info(
        f"  pooled within-sector Spearman: {pooled_s:+.3f}; "
        f"firm-weighted sector average: {fw_avg:+.3f}"
    )
    logger.info(
        f"  split-sample (mechanical-correlation check): "
        f"{split_rhos['alpha_odd_omega_even']['pooled_within_spearman']:+.3f} / "
        f"{split_rhos['alpha_even_omega_odd']['pooled_within_spearman']:+.3f} "
        f"(avg {split_avg:+.3f}; close to full-sample => dependence is real)"
    )
    rho_copula = float(2.0 * np.sin(np.pi * pooled_s / 6.0))
    logger.info(
        f"  implied Gaussian-copula rho (from pooled Spearman): {rho_copula:+.3f}"
    )

    # --- write --------------------------------------------------------------
    out_dir = ensure_dir(PATHS.outdata)
    per_sector.to_csv(out_dir / CSV_OUT, index=False, float_format="%.4f")
    payload = {
        "alpha_object": ALPHA_COL,
        "window": [int(y0), int(y1)],
        "winsor_pct": list(winsor_pct),
        "min_firms_per_sector": min_firms,
        "n_firm_units": int(len(firms)),
        "pooled_within_spearman": pooled_s,
        "firm_weighted_sector_avg_spearman": fw_avg,
        "implied_gaussian_copula_rho": rho_copula,
        "split_sample": {**split_rhos, "average": split_avg},
        "per_sector": {
            int(r.NAICS): {
                "n_firms": int(r.n_firms),
                "rho_spearman": None if np.isnan(r.rho_spearman) else float(r.rho_spearman),
                "rho_pearson": None if np.isnan(r.rho_pearson) else float(r.rho_pearson),
            }
            for r in per_sector.itertuples()
        },
    }
    with open(out_dir / JSON_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        f"wrote {(out_dir / CSV_OUT).relative_to(PATHS.project_root)} and "
        f"{(out_dir / JSON_OUT).relative_to(PATHS.project_root)}"
    )


if __name__ == "__main__":
    sys.exit(main())
