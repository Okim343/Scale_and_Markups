"""
Stage S6b — Within-sector correlation between the HYBRID firm α
(`alpha_hybrid_sector_a`) and firm sales, from the S4b panel merged onto
Compustat real sales (`sale_D`, S1).

This is the empirical moment `corr_alpha_log_sales` in
`calibration/inner.py` should actually be calibrated against, once wired
through the bundle. It is DISTINCT from `alpha_z_copula_rho_bar` (Stage
S6, `06_alpha_omega_corr_hybrid.py`), which measures the alpha-omega
(alpha-productivity) rank correlation feeding the model's exogenous z
copula, not an alpha-sales relationship — see
`02_Drafts/md_files/scale_issue_logs.md` (2026-07-06 and follow-up
entries) for why conflating the two was a moment-definition mismatch.

Also reports the partial correlation of alpha vs. log(sales) controlling
for omega (the GNR log-TFPQ residual — the closest empirical analogue of
the model's productivity draw z), both as a Pearson partial correlation
and as a rank-based (Spearman-style) partial correlation on omega's
within-sector percentile rank. This isolates the alpha-productivity-driven
share of the alpha-sales relationship from whatever is left over (the
model's structural cost/Cournot channels).

Reads:
  01_intermediary/s4b_firm_year_alpha_hybrid.parquet  (alpha_hybrid_sector_a, omega)
  01_intermediary/s1_compustat_firmyear.parquet       (sale_D, real deflated sales)

Writes:
  03_outdata/naics_alpha_sales_corr_hybrid.csv   (per-NAICS2 correlations)
  03_outdata/alpha_sales_corr_hybrid.json        (pooled values + diagnostics)

Run:
  python 06b_alpha_sales_corr_hybrid.py
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
SALES_COL = "sale_D"
CSV_OUT = "naics_alpha_sales_corr_hybrid.csv"
JSON_OUT = "alpha_sales_corr_hybrid.json"


def winsorize_within(df: pd.DataFrame, col: str, by: str, pct) -> pd.Series:
    lo, hi = pct

    def _w(s: pd.Series) -> pd.Series:
        a, b = s.quantile([lo, hi])
        return s.clip(lower=a, upper=b)

    return df.groupby(by)[col].transform(_w)


def collapse_to_firms(fy: pd.DataFrame) -> pd.DataFrame:
    """One unit per (ind2d, gvkey): firm-mean alpha/omega/sales and mean log-sales."""
    fy = fy.copy()
    fy["log_sales"] = np.log(fy[SALES_COL])
    return fy.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=(ALPHA_COL, "mean"),
        omega=("omega", "mean"),
        sales=(SALES_COL, "mean"),
        log_sales=("log_sales", "mean"),
        n_years=("year", "size"),
    )


def pearson_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation of x and y controlling linearly for z."""
    r_xy = float(np.corrcoef(x, y)[0, 1])
    r_xz = float(np.corrcoef(x, z)[0, 1])
    r_yz = float(np.corrcoef(y, z)[0, 1])
    denom = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    return float((r_xy - r_xz * r_yz) / denom) if denom > 0 else float("nan")


def sector_correlations(firms: pd.DataFrame, min_firms: int) -> pd.DataFrame:
    rows = []
    for ind, g in firms.groupby("ind2d"):
        if len(g) < min_firms:
            rows.append((ind, len(g), np.nan, np.nan, np.nan))
            continue
        rho_s = spearmanr(g["alpha"], g["log_sales"]).statistic
        rho_p = float(np.corrcoef(g["alpha"], g["log_sales"])[0, 1])
        omega_rank = g["omega"].rank(pct=True).to_numpy()
        partial_p = pearson_partial(
            g["alpha"].to_numpy(), g["log_sales"].to_numpy(), g["omega"].to_numpy()
        )
        rows.append((ind, len(g), float(rho_s), rho_p, partial_p))
    return pd.DataFrame(
        rows,
        columns=["NAICS", "n_firms", "rho_spearman", "rho_pearson", "partial_pearson_given_omega"],
    )


def pooled_within_spearman(firms: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Rank within sector, then correlate pooled ranks (sector fixed effects removed)."""
    ra = firms.groupby("ind2d")[col_a].rank(pct=True)
    rb = firms.groupby("ind2d")[col_b].rank(pct=True)
    return float(np.corrcoef(ra, rb)[0, 1])


def within_sector_rank_partial(firms: pd.DataFrame) -> float:
    """Partial correlation of within-sector percentile ranks: rank(alpha) &
    rank(log_sales), controlling for rank(omega) — a Spearman-style partial
    correlation on the same within-sector-demeaning basis as `rho_bar`."""
    ra = firms.groupby("ind2d")["alpha"].rank(pct=True).to_numpy()
    rs = firms.groupby("ind2d")["log_sales"].rank(pct=True).to_numpy()
    ro = firms.groupby("ind2d")["omega"].rank(pct=True).to_numpy()
    return pearson_partial(ra, rs, ro)


def main() -> None:
    config = load_config()
    logger = setup_logger("S6b", config.get("log_level", "INFO"))
    cfg_c = config.get("alpha_sales_corr", config.get("alpha_omega_corr", {}))
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

    sales_path = PATHS.intermediary / "s1_compustat_firmyear.parquet"
    sales = pd.read_parquet(sales_path, columns=["gvkey", "year", SALES_COL])
    sales["year"] = sales["year"].astype(fy["year"].dtype)
    fy = fy.merge(sales, on=["gvkey", "year"], how="inner")
    fy = fy[fy[SALES_COL] > 0.0]
    logger.info(
        f"  merged onto {sales_path.relative_to(PATHS.project_root)}::{SALES_COL} "
        f"({len(fy):,} firm-year rows with positive sales)"
    )

    y0, y1 = active_window(config)
    fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)]
    logger.info(f"  active window [{y0}, {y1}]: {len(fy):,} firm-year rows")
    if len(fy) == 0:
        raise ValueError("no firm-year rows in the correlation window")

    # --- full-sample firm collapse ------------------------------------------
    firms = collapse_to_firms(fy)
    firms["alpha"] = winsorize_within(firms, "alpha", "ind2d", winsor_pct)
    firms["omega"] = winsorize_within(firms, "omega", "ind2d", winsor_pct)
    firms["log_sales"] = winsorize_within(firms, "log_sales", "ind2d", winsor_pct)
    logger.info(f"  firm units: {len(firms):,} across {firms['ind2d'].nunique()} sectors")

    per_sector = sector_correlations(firms, min_firms)
    pooled_s = pooled_within_spearman(firms, "alpha", "log_sales")
    pooled_pearson_raw = float(np.corrcoef(firms["alpha"], firms["log_sales"])[0, 1])
    pooled_pearson_raw_level_sales = float(np.corrcoef(firms["alpha"], firms["sales"])[0, 1])
    pooled_partial_pearson = pearson_partial(
        firms["alpha"].to_numpy(), firms["log_sales"].to_numpy(), firms["omega"].to_numpy()
    )
    pooled_partial_rank = within_sector_rank_partial(firms)
    check_corr_alpha_omega = pooled_within_spearman(firms, "alpha", "omega")

    covered = per_sector.dropna(subset=["rho_spearman"])
    fw_avg = (
        float(np.average(covered["rho_spearman"], weights=covered["n_firms"]))
        if len(covered)
        else np.nan
    )

    # --- report -------------------------------------------------------------
    logger.info(
        "  per-sector correlations:\n"
        + per_sector.round(
            {"rho_spearman": 3, "rho_pearson": 3, "partial_pearson_given_omega": 3}
        ).to_string(index=False)
    )
    logger.info(
        f"  pooled raw Pearson corr(alpha, log_sales): {pooled_pearson_raw:+.3f}  "
        f"(model-comparable definition, no sector demeaning)"
    )
    logger.info(f"  pooled raw Pearson corr(alpha, sales) [level]: {pooled_pearson_raw_level_sales:+.3f}")
    logger.info(
        f"  pooled within-sector Spearman corr(alpha, log_sales): {pooled_s:+.3f}; "
        f"firm-weighted sector average: {fw_avg:+.3f}"
    )
    logger.info(
        f"  partial corr(alpha, log_sales | omega): Pearson={pooled_partial_pearson:+.3f}  "
        f"within-sector-rank={pooled_partial_rank:+.3f}"
    )
    logger.info(
        f"  check: pooled within-sector Spearman corr(alpha, omega) = "
        f"{check_corr_alpha_omega:+.3f} (should be close to alpha_z_copula_rho_bar)"
    )

    # --- write --------------------------------------------------------------
    out_dir = ensure_dir(PATHS.outdata)
    per_sector.to_csv(out_dir / CSV_OUT, index=False, float_format="%.4f")
    payload = {
        "alpha_object": ALPHA_COL,
        "sales_object": SALES_COL,
        "window": [int(y0), int(y1)],
        "winsor_pct": list(winsor_pct),
        "min_firms_per_sector": min_firms,
        "n_firm_units": int(len(firms)),
        "pooled_pearson_corr_alpha_log_sales": pooled_pearson_raw,
        "pooled_pearson_corr_alpha_sales_level": pooled_pearson_raw_level_sales,
        "pooled_within_sector_spearman_alpha_log_sales": pooled_s,
        "firm_weighted_sector_avg_spearman_alpha_log_sales": fw_avg,
        "partial_pearson_corr_alpha_log_sales_given_omega": pooled_partial_pearson,
        "partial_within_sector_rank_corr_alpha_log_sales_given_omega": pooled_partial_rank,
        "check_pooled_within_sector_spearman_alpha_omega": check_corr_alpha_omega,
        "per_sector": {
            int(r.NAICS): {
                "n_firms": int(r.n_firms),
                "rho_spearman": None if np.isnan(r.rho_spearman) else float(r.rho_spearman),
                "rho_pearson": None if np.isnan(r.rho_pearson) else float(r.rho_pearson),
                "partial_pearson_given_omega": (
                    None if np.isnan(r.partial_pearson_given_omega)
                    else float(r.partial_pearson_given_omega)
                ),
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
