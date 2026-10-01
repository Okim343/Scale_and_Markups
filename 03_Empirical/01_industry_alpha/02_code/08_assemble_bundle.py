"""
Stage S8 — Final assembly: NATIVELY POOLED calibration bundle.

The bundle has NO sector dimension as a model index. Everything the structural
solver consumes is a pooled scalar or the single pooled F(α). NAICS2 enters
only as calibration data (within-market concentration, the emx_slope
regression, KLEMS capital shares) — see model.typ §"Pooled-Market Steady State"
and pooled_market_baseline_porting.md §7.

Reads:
  01_intermediary/s1_compustat_firmyear.parquet (reference-scale diagnostics)
  01_intermediary/s2_pooled_targets.json    (pooled cr4/cr20/top1pct/top5pct, n_firms_cs)
  01_intermediary/s4_klems_sector_year.parquet (KLEMS a_iy, value_added)
  01_intermediary/s4b_firm_year_alpha_hybrid.parquet (hybrid α for φ_v/diagnostics)
  01_intermediary/s5_F_alpha.parquet         (single pooled F(α))
  01_intermediary/s7_slope_regression.json   (aggregate μ_cw, validation slope)
  03_outdata/emx_slope_moment.yaml           (emx_slope scalar, from 08b)
  03_outdata/alpha_omega_corr_hybrid.json    (copula ρ̄ on the hybrid α, from S6)

Writes (3-file pooled bundle):
  03_outdata/F_alpha.parquet         (copied from s5; single pooled distribution)
  03_outdata/aggregate_moments.yaml  (ALL pooled scalars + validation + flags)
  03_outdata/manifest.yaml           (provenance: hashes, version, paths)

Also writes a reference-only artifact (NOT consumed by the model):
  03_outdata/naics2_exposure_reference.csv (NAICS2 expenditure-share distribution
      for future non-uniform exposure ω_j; the baseline uses uniform ω_j = 1/Q).
  03_outdata/reference_scale_by_sector.csv (Compustat-only sector diagnostics for
      the model-unit operating-scale reference x_ref and diagnostic raw levels).

Run:
  python 08_assemble_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils import PATHS, ensure_dir, load_config, pooled_capital_share, setup_logger


# Checked-in default emx_slope, used only if 08b output is absent on a cold build.
EMX_SLOPE_DEFAULT = -0.5096

SECTOR_LABELS = {
    "11": "Agriculture, forestry, fishing and hunting",
    "21": "Mining, quarrying, and oil and gas extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale trade",
    "44-45": "Retail trade",
    "48-49": "Transportation and warehousing",
    "51": "Information",
    "54": "Professional, scientific, and technical services",
    "56": "Administrative and support and waste management services",
    "61": "Educational services",
    "71": "Arts, entertainment, and recreation",
    "72": "Accommodation and food services",
}


def read_copula_rho_bar(config: dict, logger) -> tuple[float | None, dict | None]:
    r"""Read the fixed Gaussian-copula ρ̄ recomputed on the hybrid alpha (Stage S6)."""
    cfg = config.get("alpha_z_copula", {}) or {}
    if not cfg.get("enabled", True):
        logger.warning(
            "alpha_z_copula.enabled is false — NOT publishing ρ̄; the solver will "
            "error unless ρ̄ is supplied another way."
        )
        return None, None
    source = cfg.get(
        "source",
        "03_Empirical/01_industry_alpha/03_outdata/alpha_omega_corr_hybrid.json",
    )
    key = cfg.get("key", "implied_gaussian_copula_rho")
    src_path = (PATHS.project_root / source).resolve()
    if not src_path.exists():
        raise FileNotFoundError(
            f"alpha_z_copula source missing: {src_path}. ρ̄ is a required external "
            "input; produce it via 03_Empirical/00_Hubmer_RTS, or set "
            "alpha_z_copula.enabled: false."
        )
    with open(src_path) as f:
        doc = json.load(f)
    if key not in doc:
        raise KeyError(f"{src_path} missing key {key!r}")
    rho_bar = float(doc[key])
    if not (0.0 <= rho_bar < 1.0):
        raise ValueError(f"copula ρ̄={rho_bar} from {src_path} not in [0, 1)")
    provenance = {
        "source": source,
        "key": key,
        "rho_bar": rho_bar,
        "pooled_within_spearman": doc.get("pooled_within_spearman"),
        "n_firm_units": doc.get("n_firm_units"),
    }
    logger.info(f"  copula ρ̄ = {rho_bar:.6f} (from {source}, key {key!r})")
    return rho_bar, provenance


def read_alpha_sales_corr_empirical(config: dict, logger) -> tuple[float, float, dict]:
    """Read the empirical corr(alpha, log sales) (Stage S6b) — both the raw
    pooled Pearson value and the partial value controlling for omega (the
    empirical stand-in for the model's z) — distinct from the
    alpha-productivity copula ρ̄ above. This is now a required model input:
    rho_bar is only a calibration warm start/provenance scalar."""
    src_path = (
        PATHS.outdata / "alpha_sales_corr_hybrid.json"
    ).resolve()
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path.relative_to(PATHS.project_root)} missing — run "
            "06b_alpha_sales_corr_hybrid.py to publish alpha_sales_corr_empirical."
        )
    with open(src_path) as f:
        doc = json.load(f)
    key = "pooled_pearson_corr_alpha_log_sales"
    partial_key = "partial_pearson_corr_alpha_log_sales_given_omega"
    value = float(doc[key])
    partial_value = float(doc[partial_key])
    provenance = {
        "source": "03_Empirical/01_industry_alpha/03_outdata/alpha_sales_corr_hybrid.json",
        "key": key,
        "value": value,
        "partial_key": partial_key,
        "partial_value": partial_value,
        "n_firm_units": doc.get("n_firm_units"),
    }
    logger.info(
        f"  empirical corr(alpha, log sales) = {value:.6f}; partial (given omega) = "
        f"{partial_value:.6f} (from Stage S6b)"
    )
    return value, partial_value, provenance


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file; chunked read for large parquets."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hybrid_value_added_weight(logger) -> tuple[float, dict, dict]:
    """Gross-output value-added weight φ_v on the HYBRID alpha, plus GNR-raw diag.

    With y = z (v^φv m^(1-φv))^α the value-added weight satisfies
    φ_v = 1 - melast/α. On the model-facing hybrid alpha this is

        phi_v_hybrid = 1 - mean(melast) / mean(alpha_hybrid_sector_a),

    computed over the S4b panel under the same sample filters the old GNR φ_v
    used (in_step2_sample, non-outlier). The denominator is `alpha_hybrid_sector_a`
    (NOT the raw GNR `rts`) precisely so φ_v and the model-facing α are built from
    the same object — using GNR `rts` would reintroduce the lelast-contaminated
    labor elasticity through the value-added weight.

    Returns (phi_v_hybrid, gnr_elasticities_diag, hybrid_alpha_stats). The raw GNR
    elasticities (melast/kelast/lelast/rts/phi_v_gnr_raw/a_gnr) are kept only as
    diagnostics under `gnr_elasticities` (role: diagnostic_raw_gnr).
    """
    fy = pd.read_parquet(PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet")
    if "in_step2_sample" in fy.columns:
        fy = fy[fy["in_step2_sample"] == True]  # noqa: E712
    if "rts_outlier_flag" in fy.columns:
        fy = fy[fy["rts_outlier_flag"] == False]  # noqa: E712
    fy = fy.dropna(subset=["melast", "alpha_hybrid_sector_a"])

    melast = float(fy["melast"].mean())
    kelast = float(fy["kelast"].mean())
    lelast = float(fy["lelast_gnr_raw"].mean())
    rts = float(fy["alpha_gnr_raw"].mean())
    alpha_hybrid_mean = float(fy["alpha_hybrid_sector_a"].mean())
    alpha_hybrid_std = float(fy["alpha_hybrid_sector_a"].std())
    n_fallback_share = float(fy["a_sector_fallback_flag"].mean())

    phi_v_hybrid = 1.0 - melast / alpha_hybrid_mean
    phi_v_gnr_raw = 1.0 - melast / rts
    a_gnr = kelast / (kelast + lelast)
    logger.info(
        f"  hybrid value-added weight φ_v = 1 - melast/mean(alpha_hybrid) = "
        f"1 - {melast:.4f}/{alpha_hybrid_mean:.4f} = {phi_v_hybrid:.4f}  "
        f"(GNR-raw φ_v={phi_v_gnr_raw:.4f}, a_gnr={a_gnr:.4f})"
    )
    gnr_elasticities = {
        "role": "diagnostic_raw_gnr",
        "melast": melast,
        "kelast": kelast,
        "lelast": lelast,
        "rts": rts,
        "phi_v_gnr_raw": float(phi_v_gnr_raw),
        "a_gnr": float(a_gnr),
        "n_obs": int(len(fy)),
        "source": (
            "Raw GNR firm-year elasticities (lelast Compustat/headcount-"
            "contaminated), DIAGNOSTIC ONLY. φ_v_gnr_raw = 1 - melast/rts and "
            "a_gnr = kelast/(kelast+lelast) were the pre-hybrid model primitives; "
            "they no longer feed the model (see hybrid_alpha block and top-level "
            "phi_v / a)."
        ),
    }
    hybrid_alpha_stats = {
        "formula": "alpha_hybrid_sector_a = melast + kelast + kelast*(1 - a_sector)/a_sector",
        "phi_v_hybrid": float(phi_v_hybrid),
        "phi_v_denominator": "mean(alpha_hybrid_sector_a)",
        "alpha_hybrid_mean": alpha_hybrid_mean,
        "alpha_hybrid_std": alpha_hybrid_std,
        "n_fallback_share": n_fallback_share,
        "n_obs": int(len(fy)),
        "melast_mean": melast,
        "source": (
            "S4b s4b_firm_year_alpha_hybrid.parquet over the step-2, non-outlier "
            "sample; φ_v = 1 - mean(melast)/mean(alpha_hybrid_sector_a). External "
            "primitive, not calibrated."
        ),
    }
    return float(phi_v_hybrid), gnr_elasticities, hybrid_alpha_stats


def naics2_exposure_reference(window: tuple[int, int], logger) -> pd.DataFrame:
    """Reference-only NAICS2 expenditure-share distribution (NOT consumed).

    The baseline uses uniform exposure ω_j = 1/Q. This artifact preserves the
    empirical KLEMS value-added share distribution across NAICS2 groups for a
    future non-uniform exposure sampler; it is documentation, not a model input.
    """
    s4 = pd.read_parquet(PATHS.intermediary / "s4_klems_sector_year.parquet")
    s, e = window
    win = s4[(s4["year"] >= s) & (s4["year"] <= e)]
    ref = win.groupby("sector_id", as_index=False).agg(
        value_added=("value_added", "mean")
    )
    ref["va_share"] = ref["value_added"] / ref["value_added"].sum()
    ref = ref.sort_values("va_share", ascending=False).reset_index(drop=True)
    ref["note"] = "reference_only_not_consumed_by_model"
    logger.info(f"  reference NAICS2 exposure shares over {len(ref)} groups (Σ=1)")
    return ref


def reference_scale(
    window: tuple[int, int],
    *,
    phi_v: float,
    a_bar: float,
    logger,
) -> tuple[dict, pd.DataFrame]:
    """Build the empirical operating-scale reference from Compustat only.

    The model-side use is a dimensionless input-share anchor: x_ref is the
    unweighted median across retained NAICS2 sectors of

        sector median firm x / average annual sector total x.

    The input composite exponents use the hybrid-consistent pair
    (a_bar, phi_v_hybrid) — the SAME capital share and value-added weight the
    model's production function uses — so x_ref is measured in model units.
    Raw Compustat input-index and output levels are retained only as diagnostics.
    Neither object uses alpha, z, calibrated model outcomes, or any fit criterion.
    """
    s, e = window
    path = PATHS.intermediary / "s1_compustat_firmyear.parquet"
    df = pd.read_parquet(
        path,
        columns=["gvkey", "year", "sector_id", "sale_D", "capital_D", "emp", "cogs_D"],
    )
    df = df[(df["year"] >= s) & (df["year"] <= e)].copy()
    for col in ["sale_D", "capital_D", "emp", "cogs_D"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_counts = df.groupby("sector_id").agg(
        raw_firm_year_count=("gvkey", "size"),
        raw_firm_count=("gvkey", "nunique"),
    )
    complete_mask = df[["sale_D", "capital_D", "emp", "cogs_D"]].notna().all(axis=1)
    complete_mask &= (df[["sale_D", "capital_D", "emp", "cogs_D"]] > 0).all(axis=1)
    ref = df.loc[complete_mask].copy()
    if ref.empty:
        raise ValueError("reference-scale sample is empty after complete-positive filter")

    k_exp = float(a_bar * phi_v)
    l_exp = float((1.0 - a_bar) * phi_v)
    m_exp = float(1.0 - phi_v)
    ref["x_ref_input"] = (
        ref["capital_D"].astype(float) ** k_exp
        * ref["emp"].astype(float) ** l_exp
        * ref["cogs_D"].astype(float) ** m_exp
    )
    ref["y_ref_output"] = ref["sale_D"].astype(float)
    by_sector_year = ref.groupby(["sector_id", "year"], as_index=False).agg(
        year_total_x=("x_ref_input", "sum"),
        year_mean_x=("x_ref_input", "mean"),
        year_median_x=("x_ref_input", "median"),
        year_firm_count=("gvkey", "nunique"),
    )
    by_sector_year["year_median_over_total_x"] = (
        by_sector_year["year_median_x"] / by_sector_year["year_total_x"]
    )
    annual_scale = by_sector_year.groupby("sector_id", as_index=False).agg(
        avg_annual_total_x=("year_total_x", "mean"),
        median_annual_total_x=("year_total_x", "median"),
        median_year_median_x=("year_median_x", "median"),
        median_year_median_over_total_x=("year_median_over_total_x", "median"),
        reference_year_count=("year", "nunique"),
    )

    def q25(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q75(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    by_sector = ref.groupby("sector_id", as_index=False).agg(
        firm_year_count=("gvkey", "size"),
        firm_count=("gvkey", "nunique"),
        x_mean=("x_ref_input", "mean"),
        x_total=("x_ref_input", "sum"),
        x_p25=("x_ref_input", q25),
        x_median=("x_ref_input", "median"),
        x_p75=("x_ref_input", q75),
        y_mean=("y_ref_output", "mean"),
        y_p25=("y_ref_output", q25),
        y_median=("y_ref_output", "median"),
        y_p75=("y_ref_output", q75),
        sector_sales=("y_ref_output", "sum"),
    )
    by_sector = by_sector.merge(
        raw_counts.reset_index(), on="sector_id", how="left", validate="one_to_one"
    )
    by_sector = by_sector.merge(
        annual_scale, on="sector_id", how="left", validate="one_to_one"
    )
    by_sector["dropped_firm_year_count"] = (
        by_sector["raw_firm_year_count"] - by_sector["firm_year_count"]
    )
    by_sector["dropped_pct"] = (
        by_sector["dropped_firm_year_count"] / by_sector["raw_firm_year_count"]
    )
    by_sector["label"] = by_sector["sector_id"].map(SECTOR_LABELS).fillna("")
    by_sector["y_over_x_median_ratio"] = by_sector["y_median"] / by_sector["x_median"]
    by_sector["x_over_y_median_ratio"] = by_sector["x_median"] / by_sector["y_median"]
    by_sector["sector_reference_share"] = (
        by_sector["x_median"] / by_sector["avg_annual_total_x"]
    )
    by_sector["median_firm_x_over_mean_firm_x"] = (
        by_sector["x_median"] / by_sector["x_mean"]
    )
    by_sector["notes"] = (
        "complete positive sale_D/capital_D/emp/cogs_D; x_ref is model-unit "
        "median-firm input share; raw x/y levels diagnostic only"
    )
    by_sector = by_sector[
        [
            "sector_id",
            "label",
            "firm_year_count",
            "firm_count",
            "x_mean",
            "x_total",
            "x_p25",
            "x_median",
            "x_p75",
            "avg_annual_total_x",
            "median_annual_total_x",
            "sector_reference_share",
            "median_firm_x_over_mean_firm_x",
            "median_year_median_x",
            "median_year_median_over_total_x",
            "reference_year_count",
            "y_mean",
            "y_p25",
            "y_median",
            "y_p75",
            "y_over_x_median_ratio",
            "x_over_y_median_ratio",
            "raw_firm_year_count",
            "raw_firm_count",
            "dropped_firm_year_count",
            "dropped_pct",
            "sector_sales",
            "notes",
        ]
    ]
    by_sector = by_sector.sort_values("sector_id").reset_index(drop=True)

    n_sectors = len(by_sector)
    if n_sectors == 0:
        raise ValueError("reference-scale sector table is empty")
    weights = by_sector["sector_sales"].to_numpy(dtype=float)
    x_raw_unweighted = float(by_sector["x_median"].median())
    x_ref_share = float(by_sector["sector_reference_share"].median())
    y_unweighted = float(by_sector["y_median"].median())
    pooled_x_median = float(ref["x_ref_input"].median())
    avg_annual_included_sample_x = float(
        by_sector_year.groupby("year")["year_total_x"].sum().mean()
    )
    robustness = {
        "unweighted_naics2_median_reference_share": {
            "x_ref": x_ref_share,
            "aggregation": (
                "median across sectors of sector median firm x divided by "
                "average annual sector total x"
            ),
        },
        "sector_median_firm_over_mean_firm": {
            "x_ref": float(by_sector["median_firm_x_over_mean_firm_x"].median()),
            "aggregation": "median across sectors of sector median firm x / sector mean firm x",
        },
        "sector_year_median_firm_over_total": {
            "x_ref": float(by_sector["median_year_median_over_total_x"].median()),
            "aggregation": (
                "within each sector-year, median firm x / total sector-year x; "
                "then median over years within sector and unweighted median across sectors"
            ),
        },
        "pooled_firm_year_median_over_avg_annual_included_sample_x": {
            "x_ref": float(pooled_x_median / avg_annual_included_sample_x),
            "aggregation": (
                "pooled complete-case firm-year median x divided by average annual "
                "aggregate included-sample x"
            ),
        },
        "raw_unweighted_naics2_median": {
            "x_ref_raw_input_index": x_raw_unweighted,
            "y_ref": y_unweighted,
            "aggregation": "median across sector medians of raw Compustat levels",
        },
        "raw_sales_weighted_naics2_average": {
            "x_ref_raw_input_index": float(np.average(by_sector["x_median"], weights=weights)),
            "y_ref": float(np.average(by_sector["y_median"], weights=weights)),
            "aggregation": "sector-sales-weighted average across sector medians",
        },
        "raw_pooled_firm_year_median": {
            "x_ref_raw_input_index": pooled_x_median,
            "y_ref": float(ref["y_ref_output"].median()),
            "aggregation": "median across complete-case firm-years",
        },
    }
    summary = {
        "type": "pooled_naics2_unweighted_median_reference_share",
        "status": "empirical_input",
        "baseline_years": [int(s), int(e)],
        "sectors": by_sector["sector_id"].tolist(),
        "x_ref_variable": (
            "median_firm_x / average_annual_sector_total_x, where "
            "x = capital_D^(a_bar*phi_v) * emp^((1-a_bar)*phi_v) * "
            "cogs_D^(1-phi_v)"
        ),
        "x_ref_raw_variable": (
            "capital_D^(a_bar*phi_v) * emp^((1-a_bar)*phi_v) * "
            "cogs_D^(1-phi_v)"
        ),
        "y_ref_variable": "sale_D",
        "a_bar": float(a_bar),
        "phi_v": float(phi_v),
        "exponents": {
            "capital_D": k_exp,
            "emp": l_exp,
            "cogs_D": m_exp,
        },
        "x_ref": x_ref_share,
        "x_ref_raw_input_index": x_raw_unweighted,
        "y_ref": y_unweighted,
        "x_ref_role": "model_reference_input_share",
        "x_ref_raw_role": "diagnostic_input_index_level_only",
        "y_ref_role": "diagnostic_output_scale_only",
        "raw_y_ref_over_raw_x_ref": float(y_unweighted / x_raw_unweighted),
        "raw_x_ref_over_y_ref": float(x_raw_unweighted / y_unweighted),
        "n_sectors": int(n_sectors),
        "n_firm_years_complete": int(len(ref)),
        "n_firms_complete": int(ref["gvkey"].nunique()),
        "avg_annual_included_sample_x": avg_annual_included_sample_x,
        "units": {
            "x_ref": "dimensionless share of annual sector input composite",
            "x_ref_raw_input_index": "real input-composite index; not dollars",
            "y_ref": "real 2015 thousand USD from sale_D",
        },
        "robustness": robustness,
        "notes": (
            "Computed from Compustat S1 only; complete positive "
            "sale_D/capital_D/emp/cogs_D firm-years. Model-facing x_ref is a "
            "dimensionless typical-firm input share, not a raw Compustat input "
            "index level. Raw x and y levels are diagnostic only. Not chosen to "
            "match alpha-size correlation or any model outcome."
        ),
    }
    logger.info(
        f"  reference scale: x_ref_share={x_ref_share:.6g}, "
        f"raw_x_ref={x_raw_unweighted:.6g}, y_ref={y_unweighted:.6g}, "
        f"raw y/x={summary['raw_y_ref_over_raw_x_ref']:.4f} "
        f"over {n_sectors} sectors and {len(ref):,} firm-years"
    )
    return summary, by_sector


def build_aggregate_yaml(
    config: dict,
    window: tuple[int, int],
    pipeline_version: str,
    *,
    pooled_targets: dict,
    a: float,
    a_provenance: dict,
    phi_v: float,
    gnr_elasticities: dict,
    hybrid_alpha_stats: dict,
    clip_masses: dict,
    reference_scale_summary: dict,
    emx_slope: float,
    emx_slope_source: str,
    rho_bar: float | None,
    alpha_sales_corr_empirical: float | None,
    alpha_sales_corr_partial_empirical: float | None,
    logger,
) -> dict:
    """Assemble the pooled aggregate_moments.yaml contract."""
    s7 = json.loads((PATHS.intermediary / "s7_slope_regression.json").read_text())
    mu_cw = float(s7["aggregate_markup_cost_weighted"])
    mu_cw_sga = float(s7["aggregate_markup_xsga_cost_weighted"])
    per_sector = pd.DataFrame(pooled_targets["per_sector"])
    if per_sector.empty:
        raise ValueError("s2_pooled_targets.json has empty per_sector table")
    mean_firms_per_sector = float(
        np.average(per_sector["n_firms_cs"], weights=per_sector["tot_sale_cs"])
    )
    targets = {
        "mu_cw": mu_cw,
        "mu_cw_sga": mu_cw_sga,
        "cr4": float(pooled_targets["targets"]["cr4"]),
        "cr20": float(pooled_targets["targets"]["cr20"]),
        "top1pct": float(pooled_targets["targets"]["top1pct"]),
        "top5pct": float(pooled_targets["targets"]["top5pct"]),
        "emx_slope": float(emx_slope),
    }

    agg: dict = {
        "pipeline_version": pipeline_version,
        "active_window": {"key": config["active_window"],
                          "start": window[0], "end": window[1]},
        "naics_exclude": list(config["naics_exclude"]),
        "targets": targets,
        "a": float(a),
        "phi_v": float(phi_v),
        "hybrid_alpha": {
            **hybrid_alpha_stats,
            "a_bar_used": float(a),
            "support_variant": config["F_alpha"].get("support_variant", "rts_shrunk"),
            "shrink_lambda": float(config["F_alpha"].get("shrink_lambda", 0.371)),
            "shrink_target": {
                "source": "Salgado et al. Tables I and A.11",
                "moment": (
                    "compromise between two-digit NAICS RTS dispersion and "
                    "retaining an above-one RTS upper tail"
                ),
                "reference_sd": 0.052,
                "reference_p99": 1.08,
                "role": (
                    "external dispersion discipline for the KLEMS-imputed "
                    "hybrid-alpha support; mean preserving"
                ),
            },
            "alpha_clip": [float(x) for x in config["F_alpha"]["alpha_clip"]],
            "lower_clip_rationale": {
                "source": "Salgado et al. Table A.10",
                "lowest_reported_industry_average_rts": 0.59,
                "industry": "Healthcare",
                "role": (
                    "conservative admissible support floor; not a target for "
                    "Salgado's firm-level lower tail"
                ),
            },
            "support_construction": "winsorize -> mean-preserving shrink -> clip",
            "alpha_clip_lower_mass": float(clip_masses["lower"]),
            "alpha_clip_upper_mass": float(clip_masses["upper"]),
        },
        "gnr_elasticities": gnr_elasticities,
        "reference_scale": reference_scale_summary,
        "n_firms_cs": float(pooled_targets["n_firms_cs"]),
        "n_markets": int(pooled_targets["n_markets"]),
        "mean_firms_per_sector": mean_firms_per_sector,
        "validation": {
            "markup_share_slope_loglog": s7["markup_share_slope_loglog"]["coef"],
            "markup_share_slope_se_hc1": s7["markup_share_slope_loglog"]["se_hc1"],
            "markup_share_slope_n_obs": s7["markup_share_slope_loglog"]["n_obs"],
            "markup_share_slope_r_squared": s7["markup_share_slope_loglog"]["r_squared"],
        },
        "sensitivity_panel": s7["aggregate_markup_sensitivity"],
        "provenance": {
            "mu_cw": "S7 aggregate_markup_cost_weighted (Compustat cost-weighted "
                     "aggregate markup, active-window mean)",
            "mu_cw_sga": "S7 aggregate_markup_xsga_cost_weighted (Compustat "
                         "cost-weighted SG&A-inclusive aggregate markup, "
                         "active-window mean; same source/construction as mu_cw "
                         "but with SG&A in marginal cost)",
            "concentration": pooled_targets["weighting"],
            "n_firms_cs": "pooled total of window-averaged firm counts across "
                          f"{pooled_targets['n_markets']} NAICS2 markets",
            "n_markets": "number of retained NAICS2 sector markets in the pooled concentration target",
            "mean_firms_per_sector": "Compustat-only sector-sales-weighted mean of "
                                     "window-averaged firm counts across retained "
                                     "NAICS2 markets; diagnostic analogue of "
                                     "exogenous-entry N, not a calibration target",
            "a": a_provenance,
            "phi_v": hybrid_alpha_stats["source"],
            "reference_scale": (
                "Compustat S1 active-window complete-positive firm-years; "
                "model-facing x_ref is the unweighted median across sectors of "
                "median firm input divided by average annual sector input; raw "
                "input-index levels and y_ref are diagnostic only."
            ),
            "emx_slope": emx_slope_source,
        },
        "notes": (
            "NATIVELY POOLED bundle (pipeline v0.3.0). No sector dimension. "
            "The model-facing firm-level scalability object is "
            "alpha_hybrid_sector_a (see the hybrid_alpha block and Stage S4b); "
            "gnr_elasticities are raw-GNR diagnostics only. "
            "The calibration target set is mu_cw, cr4, cr20, top1pct, top5pct, "
            "and emx_slope. mean_active_count is intentionally absent: under "
            "the EMX-style exogenous-entry interpretation the model reports the "
            "realized sector firm count as a diagnostic and calibrates N directly "
            "from the concentration moments. The emx_slope target remains the "
            "Autor-style cross-NAICS2 long-difference estimate; the steady-state "
            "model compares it to the confounder-free static cross-sector slope. "
            "alpha_z_copula_rho_bar is the fixed ex-ante Gaussian-copula α–z "
            "correlation ρ̄; the solver derives ρ_z = σ_z·ρ̄/√(1−ρ̄²) "
            "(model.typ eq:rho_loading), not calibrated. Exposure ω_j is uniform "
            "(1/Q) in the baseline; see naics2_exposure_reference.csv for the "
            "deferred empirical exposure distribution (reference only). "
            "alpha_sales_corr_empirical (Stage S6b) is the pooled Pearson "
            "corr(alpha, log sales) in the same panel — a DISTINCT object from "
            "alpha_z_copula_rho_bar (alpha-productivity, not alpha-sales). "
            "alpha_sales_corr_partial_empirical is the same relationship "
            "partialing out omega (the empirical stand-in for the model's z); "
            "it is the valid empirical target for the model's simulated "
            "corr_alpha_log_sales_partial_z moment, NOT the raw "
            "alpha_sales_corr_empirical or alpha_z_copula_rho_bar (see "
            "scale_issue_logs.md, 2026-07-06 and follow-ups)."
        ),
    }
    if rho_bar is not None:
        agg["alpha_z_copula_rho_bar"] = float(rho_bar)
    if alpha_sales_corr_empirical is not None:
        agg["alpha_sales_corr_empirical"] = float(alpha_sales_corr_empirical)
    if alpha_sales_corr_partial_empirical is not None:
        agg["alpha_sales_corr_partial_empirical"] = float(alpha_sales_corr_partial_empirical)
    logger.info(
        "  targets: " + ", ".join(f"{k}={v:.5g}" for k, v in targets.items())
    )
    logger.info(f"  a={a:.5g}  n_firms_cs={agg['n_firms_cs']:.5g}")
    return agg


def build_manifest(config: dict, window: tuple[int, int], outdata: Path) -> dict:
    """Hash all inputs and outputs for reproducibility."""
    input_paths = [
        PATHS.compustat_markup_firmyear,
        PATHS.compustat_markup_cw,
        PATHS.klems_csv,
        PATHS.salgado_firm_year_rts,
    ]
    intermediary_paths = sorted(PATHS.intermediary.glob("s*.parquet")) + sorted(
        PATHS.intermediary.glob("s*.json")
    )
    output_paths = sorted(outdata.glob("*"))

    def _file_record(p: Path) -> dict:
        st = p.stat()
        return {
            "path": str(p.relative_to(PATHS.project_root)),
            "size_bytes": int(st.st_size),
            "sha256": _hash_file(p),
        }

    return {
        "pipeline_version": config.get("pipeline_version"),
        "schema": "pooled-bundle-v1",
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active_window": {"key": config["active_window"],
                          "start": window[0], "end": window[1]},
        "naics_exclude": config["naics_exclude"],
        "inputs": [_file_record(p) for p in input_paths if p.exists()],
        "intermediaries": [_file_record(p) for p in intermediary_paths],
        "outputs": [_file_record(p) for p in output_paths],
    }


def main() -> None:
    config = load_config()
    logger = setup_logger("S8", config.get("log_level", "INFO"))
    aw_key = config["active_window"]
    aw = (int(config["windows"][aw_key]["start"]),
          int(config["windows"][aw_key]["end"]))
    pipeline_version = config.get("pipeline_version", "unknown")
    logger.info(f"assembling POOLED bundle v{pipeline_version}, window {aw_key} = {aw}")

    outdata = ensure_dir(PATHS.outdata)

    # --- Drop any stale per-sector artifact from a previous schema ---------
    stale = outdata / "sector_table.parquet"
    if stale.exists():
        stale.unlink()
        logger.info("  removed stale sector_table.parquet (pooled schema has no sector dim)")

    # --- Inputs ------------------------------------------------------------
    rho_bar, rho_bar_provenance = read_copula_rho_bar(config, logger)
    alpha_sales_corr_empirical, alpha_sales_corr_partial_empirical, alpha_sales_corr_provenance = (
        read_alpha_sales_corr_empirical(config, logger)
    )

    pooled_targets = json.loads(
        (PATHS.intermediary / "s2_pooled_targets.json").read_text()
    )

    a, a_provenance = pooled_capital_share(aw, float(config["a_i_fallback"]), logger)

    # Consistency identity (plan §3.4): S4b's a_bar fallback must equal the
    # exported model-facing `a` (both come from the same pooled_capital_share
    # call on the same window). Fallback rows carry a_sector == a_bar exactly.
    s4b = pd.read_parquet(
        PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet",
        columns=["a_sector", "a_sector_fallback_flag"],
    )
    fb = s4b[s4b["a_sector_fallback_flag"]]
    if len(fb) and not np.allclose(fb["a_sector"].to_numpy(dtype=float), a):
        raise ValueError(
            "S4b a_bar fallback does not match the exported KLEMS `a`; S4b and "
            "S8 must call pooled_capital_share identically."
        )

    phi_v, gnr_elasticities, hybrid_alpha_stats = hybrid_value_added_weight(logger)

    reference_scale_summary, reference_scale_by_sector = reference_scale(
        aw, phi_v=phi_v, a_bar=a, logger=logger
    )

    # emx_slope scalar from 08b (fallback to checked-in default).
    emx_path = outdata / "emx_slope_moment.yaml"
    if emx_path.exists():
        emx_doc = yaml.safe_load(emx_path.read_text())
        emx_slope = float(emx_doc["value"])
        emx_slope_source = (
            f"Stage S8b emx_slope_moment.yaml (spec={emx_doc.get('spec')}, "
            f"window={emx_doc.get('window')})"
        )
    else:
        emx_slope = EMX_SLOPE_DEFAULT
        emx_slope_source = f"checked-in default {EMX_SLOPE_DEFAULT} (08b not run)"
        logger.warning(f"  08b output absent; using emx_slope default {EMX_SLOPE_DEFAULT}")

    # --- F_alpha.parquet (copy the single pooled distribution) -------------
    src_fa = PATHS.intermediary / "s5_F_alpha.parquet"
    out_fa = outdata / "F_alpha.parquet"
    shutil.copyfile(src_fa, out_fa)
    fa = pd.read_parquet(out_fa)
    # Post-shrink clip masses piled at each F(α) bound (audit mandatory
    # reporting). alpha_shrunk is the pre-clip (post-shrink) node value.
    clip_masses = {
        "lower": float(fa.loc[fa["alpha_shrunk"] < fa["alpha"], "weight"].sum()),
        "upper": float(fa.loc[fa["alpha_shrunk"] > fa["alpha"], "weight"].sum()),
    }
    logger.info(
        f"copied {out_fa.relative_to(PATHS.project_root)} "
        f"({out_fa.stat().st_size / 1e3:.2f} KB; {len(fa)} nodes; "
        f"α∈[{fa['alpha'].min():.4f}, {fa['alpha'].max():.4f}]; "
        f"clip mass lower={clip_masses['lower']:.4f}, upper={clip_masses['upper']:.4f})"
    )

    # --- aggregate_moments.yaml --------------------------------------------
    agg = build_aggregate_yaml(
        config, aw, pipeline_version,
        pooled_targets=pooled_targets, a=a, a_provenance=a_provenance,
        phi_v=phi_v, gnr_elasticities=gnr_elasticities,
        hybrid_alpha_stats=hybrid_alpha_stats, clip_masses=clip_masses,
        reference_scale_summary=reference_scale_summary,
        emx_slope=emx_slope, emx_slope_source=emx_slope_source,
        rho_bar=rho_bar, alpha_sales_corr_empirical=alpha_sales_corr_empirical,
        alpha_sales_corr_partial_empirical=alpha_sales_corr_partial_empirical,
        logger=logger,
    )
    out_agg = outdata / "aggregate_moments.yaml"
    with open(out_agg, "w") as f:
        yaml.safe_dump(agg, f, sort_keys=False, default_flow_style=False)
    logger.info(f"wrote {out_agg.relative_to(PATHS.project_root)}")

    # --- reference-only NAICS2 exposure artifact ---------------------------
    ref = naics2_exposure_reference(aw, logger)
    out_ref = outdata / "naics2_exposure_reference.csv"
    ref.to_csv(out_ref, index=False)
    logger.info(f"wrote {out_ref.relative_to(PATHS.project_root)} (reference only)")

    out_scale = outdata / "reference_scale_by_sector.csv"
    reference_scale_by_sector.to_csv(out_scale, index=False)
    logger.info(
        f"wrote {out_scale.relative_to(PATHS.project_root)} "
        "(reference-scale diagnostics)"
    )

    # --- manifest.yaml -----------------------------------------------------
    manifest = build_manifest(config, aw, outdata)
    if rho_bar_provenance is not None:
        manifest["alpha_z_copula"] = rho_bar_provenance
    out_man = outdata / "manifest.yaml"
    with open(out_man, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    logger.info(f"wrote {out_man.relative_to(PATHS.project_root)}")

    logger.info("pooled bundle assembly complete.")


if __name__ == "__main__":
    sys.exit(main())
