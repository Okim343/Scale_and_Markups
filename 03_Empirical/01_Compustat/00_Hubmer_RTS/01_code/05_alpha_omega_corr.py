"""
Stage R5 — Within-sector correlation between firm α (RTS) and firm
productivity ω, from the GNR firm-year output.

Purpose: measure the empirical joint dependence of (α, z) that the model's
z-creation rule parameterizes with ρ. This stage only PRODUCES the moment;
nothing downstream consumes it yet (wiring ρ into the calibration bundle is
a separate, deliberate step).

Reads:
  03_outdata/firm_year_rts.parquet            (from 03_postprocess_gnr.py)

Writes:
  03_outdata/naics_alpha_omega_corr.csv       (per-NAICS2 correlations)
  03_outdata/alpha_omega_corr.json            (pooled values + diagnostics)

Method:
  1. Filter to the export window (same window as the sector RTS aggregation).
  2. Collapse to FIRMS: per (gvkey, ind2d), firm-mean alpha_gross_output and
     firm-mean omega over the window years — same cross-section concept as
     the F_alpha firm_mean collapse.
  3. Winsorize both variables within sector.
  4. Per sector: Spearman rank correlation rho_S(ᾱ_f, ω̄_f) (plus Pearson
     for reference). Spearman is scale-invariant (robust to the
     gross-output vs value-added level mismatch) and maps one-to-one into a
     Gaussian-copula ρ: rho_pearson = 2 * sin(pi * rho_S / 6).
  5. Mechanical-correlation check: α̂ and ω̂ are built from the same
     estimated parameters and data, so estimation error is correlated
     across them within a firm-year. The split-sample version computes each
     firm's α from its ODD window years and ω from its EVEN years (and the
     reverse, then averages the two correlations): correlated estimation
     noise across the two halves is broken, so if the split-sample rho is
     close to the full-sample rho, the dependence is real, not mechanical.

Run:
  python 05_alpha_omega_corr.py [--config config_smoke.yaml]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import (
    PATHS,
    ensure_dir,
    outdata_dir,
    parse_config_arg,
    setup_logger,
)

FY_IN = "firm_year_rts.parquet"
CSV_OUT = "naics_alpha_omega_corr.csv"
JSON_OUT = "alpha_omega_corr.json"


def winsorize_within(df: pd.DataFrame, col: str, by: str, pct) -> pd.Series:
    lo, hi = pct

    def _w(s: pd.Series) -> pd.Series:
        a, b = s.quantile([lo, hi])
        return s.clip(lower=a, upper=b)

    return df.groupby(by)[col].transform(_w)


def collapse_to_firms(fy: pd.DataFrame) -> pd.DataFrame:
    """One unit per (ind2d, gvkey): firm-mean alpha and omega."""
    return fy.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=("alpha_gross_output", "mean"),
        omega=("omega", "mean"),
        n_years=("year", "size"),
    )


def sector_correlations(
    firms: pd.DataFrame, min_firms: int
) -> pd.DataFrame:
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
    a_odd = odd.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=("alpha_gross_output", "mean")
    )
    o_even = even.groupby(["ind2d", "gvkey"], as_index=False).agg(
        omega=("omega", "mean")
    )
    a_even = even.groupby(["ind2d", "gvkey"], as_index=False).agg(
        alpha=("alpha_gross_output", "mean")
    )
    o_odd = odd.groupby(["ind2d", "gvkey"], as_index=False).agg(
        omega=("omega", "mean")
    )
    fw = a_odd.merge(o_even, on=["ind2d", "gvkey"], how="inner")
    bw = a_even.merge(o_odd, on=["ind2d", "gvkey"], how="inner")
    return fw, bw


def main() -> None:
    config = parse_config_arg("Stage R5 — firm-level alpha-omega correlation")
    logger = setup_logger("R5", config.get("log_level", "INFO"))
    out_dir = outdata_dir(config)
    cfg_c = config.get("alpha_omega_corr", {})
    min_firms = int(cfg_c.get("min_firms_per_sector", 30))
    winsor_pct = cfg_c.get("winsor_pct", [0.01, 0.99])

    fy_path = out_dir / FY_IN
    if not fy_path.exists():
        raise FileNotFoundError(f"{fy_path} missing; run 03_postprocess_gnr.py first.")
    fy = pd.read_parquet(
        fy_path, columns=["gvkey", "year", "ind2d", "alpha_gross_output", "omega"]
    ).dropna(subset=["alpha_gross_output", "omega"])
    logger.info(f"loaded {fy_path.relative_to(PATHS.project_root)} ({len(fy):,} rows)")

    window = config["export"].get("window")
    if window:
        y0, y1 = int(window[0]), int(window[1])
        fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)]
        logger.info(f"  window [{y0}, {y1}]: {len(fy):,} firm-year rows")
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
    fw_avg = float(
        np.average(covered["rho_spearman"], weights=covered["n_firms"])
    ) if len(covered) else np.nan

    # --- split-sample check ----------------------------------------------------
    fw, bw = split_sample_firms(fy)
    split_rhos = {}
    for name, d in [("alpha_odd_omega_even", fw), ("alpha_even_omega_odd", bw)]:
        d["alpha"] = winsorize_within(d, "alpha", "ind2d", winsor_pct)
        d["omega"] = winsorize_within(d, "omega", "ind2d", winsor_pct)
        split_rhos[name] = {
            "pooled_within_spearman": pooled_within_spearman(d),
            "n_firms": int(len(d)),
        }
    split_avg = float(
        np.mean([v["pooled_within_spearman"] for v in split_rhos.values()])
    )

    # --- report -------------------------------------------------------------------
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
    # Gaussian-copula mapping for the solver's rho, if/when it is wired in.
    rho_copula = float(2.0 * np.sin(np.pi * pooled_s / 6.0))
    logger.info(
        f"  implied Gaussian-copula rho (from pooled Spearman): {rho_copula:+.3f}"
    )

    # --- write -----------------------------------------------------------------------
    ensure_dir(out_dir)
    per_sector.to_csv(out_dir / CSV_OUT, index=False, float_format="%.4f")
    payload = {
        "window": list(window) if window else None,
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
