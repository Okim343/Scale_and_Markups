"""
Stage S5 — Build the SINGLE pooled scalability distribution F(α).

There is no sector dimension. F(α) is a single pooled distribution built from
the pooled empirical CDF of firm-level returns-to-scale α (the GNR estimates in
firm_year_rts.parquet), pooled across ALL retained NAICS2 firms in the active
window — model.typ Panel B: "pooled scalability distribution; firm-level
returns to scale, not binned by NAICS2".

Construction:
  1. Read s4b_firm_year_alpha_hybrid.parquet (Stage S4b, the hybrid GNR-KLEMS
     alpha panel; the model-facing α column is `alpha_hybrid_sector_a`). Map
     NAICS2 → sector_id and keep only retained sectors (S1's sector set); filter
     to the active window.
  2. Collapse to a cross-section over FIRMS (default `collapse: firm_mean`):
     one unit per gvkey = its window-mean α. "firm_year" keeps the pooled
     firm-year cross-section (robustness).
  3. Winsorize the pooled α at `winsor_pct`.
  4. Discretize the pooled empirical CDF into N equal-mass nodes at the
     quantile midpoints u_k = (k-0.5)/N: α_raw_k = quantile_α(u_k), weight 1/N.
  5. Apply the mean-preserving RTS shrink (`support_variant: rts_shrunk`,
     `shrink_lambda`) to the node support: α_shrunk = mean + λ(α_raw - mean).
     `full_compustat` skips the shrink (robustness).
  6. Clip α_shrunk to `alpha_clip` (default [0.6, 1.20] — the structural
     model/pool.py draw_pool support) with a flag. This shrink→clip order is
     the single standard; the shipped `alpha` IS the model-facing support (the
     solver samples it directly, no further transform).

Reads:
  01_intermediary/s1_compustat_firmyear.parquet    (retained sector_id set)
  01_intermediary/s4b_firm_year_alpha_hybrid.parquet (firm-level hybrid α)

Writes:
  01_intermediary/s5_F_alpha.parquet
      N equal-mass pooled support points. Columns:
        node_id, u, weight, alpha_raw, alpha_shrunk, alpha, alpha_clipped_flag,
        alpha_source.  (alpha_raw = post-winsor pre-shrink; alpha_shrunk =
        post-shrink pre-clip; alpha = final model-facing support.) NO sector_id.

Run:
  python 05_build_F_alpha.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import (
    COMPUSTAT_TO_SECTOR_ID,
    PATHS,
    active_window,
    ensure_dir,
    load_config,
    setup_logger,
)


S5_OUT = "s5_F_alpha.parquet"


def load_pooled_alpha(config: dict, retained_sector_ids: list, logger) -> pd.DataFrame:
    """Pooled firm-level α across all retained NAICS2 firms (active window)."""
    cfg = config["F_alpha"]
    alpha_col = cfg.get("alpha_col", "alpha_hybrid_sector_a")
    collapse = cfg.get("collapse", "firm_mean")
    if collapse not in ("firm_mean", "firm_year"):
        raise ValueError(
            f"F_alpha.collapse must be 'firm_mean' or 'firm_year'; got {collapse!r}"
        )

    path = PATHS.intermediary / "s4b_firm_year_alpha_hybrid.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"s4b_firm_year_alpha_hybrid.parquet missing at {path}; run "
            "04b_build_hybrid_alpha.py first (it builds the hybrid GNR-KLEMS "
            "alpha panel)."
        )
    logger.info(f"reading {path.relative_to(PATHS.project_root)}")
    fy = pd.read_parquet(path)
    needed = {"gvkey", "ind2d", "year", alpha_col}
    missing = needed - set(fy.columns)
    if missing:
        raise KeyError(f"firm_year_rts.parquet missing columns: {sorted(missing)}")

    fy = fy.dropna(subset=[alpha_col])
    logger.info(f"  firm-year rows (non-missing α): {len(fy):,}")

    y0, y1 = active_window(config)
    fy = fy[(fy["year"] >= y0) & (fy["year"] <= y1)]
    logger.info(f"  rows in active window [{y0}, {y1}]: {len(fy):,}")

    # NAICS2 → sector_id; restrict to S1-retained sectors (applies the same
    # NAICS2 exclusions as the rest of the pipeline).
    fy["sector_id"] = fy["ind2d"].astype(int).map(COMPUSTAT_TO_SECTOR_ID)
    fy = fy.dropna(subset=["sector_id"])
    fy = fy[fy["sector_id"].isin(retained_sector_ids)]
    logger.info(f"  rows in retained sectors: {len(fy):,}")

    if collapse == "firm_mean":
        n_rows = len(fy)
        fy = fy.groupby("gvkey", as_index=False).agg(
            **{alpha_col: (alpha_col, "mean")}, n_years=("year", "size")
        )
        logger.info(
            f"  collapsed {n_rows:,} firm-years to {len(fy):,} firm units "
            f"(mean years per firm: {fy['n_years'].mean():.1f})"
        )
    else:
        logger.info("  collapse=firm_year: pooled firm-year cross-section (robustness)")

    return fy.rename(columns={alpha_col: "alpha_unit"})[["alpha_unit"]]


def build_pooled_nodes(alpha: np.ndarray, config: dict, logger) -> pd.DataFrame:
    """Equal-mass discretization of the pooled empirical CDF of α."""
    cfg = config["F_alpha"]
    n_nodes = int(cfg.get("n_nodes", 500))
    w_lo, w_hi = cfg.get("winsor_pct", [0.01, 0.99])
    alpha_lo, alpha_hi = cfg["alpha_clip"]
    min_obs = int(cfg.get("min_obs", 100))

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size < min_obs:
        raise ValueError(
            f"only {alpha.size} pooled α units (< min_obs={min_obs}); refusing "
            "to build an unreliable pooled F(α)."
        )

    # Pooled winsorization.
    lo_q, hi_q = np.quantile(alpha, [w_lo, w_hi])
    n_wins = int(((alpha < lo_q) | (alpha > hi_q)).sum())
    alpha_w = np.clip(alpha, lo_q, hi_q)
    logger.info(
        f"  pooled winsor [{w_lo:.2f}, {w_hi:.2f}] -> [{lo_q:.4f}, {hi_q:.4f}]; "
        f"{n_wins:,} units clipped"
    )

    # Equal-mass nodes: α_raw_k = empirical quantile at u_k = (k-0.5)/N.
    u_nodes = (np.arange(n_nodes) + 0.5) / n_nodes
    alpha_raw = np.quantile(alpha_w, u_nodes)
    weight = np.full(n_nodes, 1.0 / n_nodes)

    # Mean-preserving RTS shrink applied BEFORE the clip. The baseline λ=0.371 is
    # a compromise discipline from Salgado et al.: it substantially shrinks the
    # noisy KLEMS-imputed hybrid-alpha dispersion while preserving the empirical
    # upper-tail feature that some firms have RTS above one (Table I P99 ≈ 1.08).
    # Shrinking first means the clip re-truncates far less lower tail than
    # clipping the raw support would, and the shipped F(α) IS the object the model
    # samples (no further model-side transform). full_compustat = unshrunk
    # robustness variant.
    support_variant = cfg.get("support_variant", "rts_shrunk")
    shrink_lambda = float(cfg.get("shrink_lambda", 0.371))
    if support_variant == "rts_shrunk":
        mean_raw = float(alpha_raw.mean())
        alpha_shrunk = mean_raw + shrink_lambda * (alpha_raw - mean_raw)
    elif support_variant == "full_compustat":
        shrink_lambda = 1.0
        alpha_shrunk = alpha_raw.copy()
    else:
        raise ValueError(
            "F_alpha.support_variant must be 'rts_shrunk' or 'full_compustat'; "
            f"got {support_variant!r}"
        )
    logger.info(
        f"  support_variant={support_variant} (λ={shrink_lambda}): "
        f"mean {alpha_raw.mean():.4f} -> {alpha_shrunk.mean():.4f}, "
        f"std {alpha_raw.std():.4f} -> {alpha_shrunk.std():.4f}"
    )

    # Clip the shrunk support to the model-admissible range (pool.py draw_pool).
    alpha_clipped = np.clip(alpha_shrunk, alpha_lo, alpha_hi)
    clipped_flag = alpha_clipped != alpha_shrunk
    n_clip = int(clipped_flag.sum())
    # Equal-mass nodes => clip mass is the weight piled at each clip bound.
    lower_mass = float(weight[alpha_shrunk < alpha_lo].sum())
    upper_mass = float(weight[alpha_shrunk > alpha_hi].sum())
    if n_clip:
        logger.warning(
            f"  clipped {n_clip} post-shrink α nodes outside [{alpha_lo}, {alpha_hi}]: "
            f"lower-bound mass={lower_mass:.4f}, upper-bound mass={upper_mass:.4f}"
        )
    # A large residual tail pile signals the clip bound / shrink λ needs
    # revisiting; flag loudly but do not fail the build.
    if lower_mass + upper_mass > 0.30:
        logger.warning(
            f"  clip mass {lower_mass + upper_mass:.4f} exceeds 0.30 — revisit "
            "the clip bound or shrink λ."
        )

    nodes = pd.DataFrame(
        {
            "node_id": np.arange(1, n_nodes + 1, dtype="int16"),
            "u": u_nodes.astype("float32"),
            "weight": weight.astype("float32"),
            "alpha_raw": alpha_raw.astype("float32"),
            "alpha_shrunk": alpha_shrunk.astype("float32"),
            "alpha": alpha_clipped.astype("float32"),
            "alpha_clipped_flag": clipped_flag,
            "alpha_source": "alpha_hybrid_sector_a_pooled_empirical",
        }
    )
    nodes["alpha_source"] = nodes["alpha_source"].astype("string")

    logger.info(
        f"  {n_nodes} equal-mass nodes; Σweight={nodes['weight'].sum():.4f}; "
        f"α(clipped) min/mean/max = {nodes['alpha'].min():.4f} / "
        f"{float(np.average(nodes['alpha'], weights=nodes['weight'])):.4f} / "
        f"{nodes['alpha'].max():.4f}"
    )
    return nodes


def main() -> None:
    config = load_config()
    logger = setup_logger("S5", config.get("log_level", "INFO"))

    # Retained sector_ids from S1 (defines the NAICS2 firm set we pool over).
    s1_path = PATHS.intermediary / "s1_compustat_firmyear.parquet"
    if not s1_path.exists():
        raise FileNotFoundError(
            f"Stage S1 output missing at {s1_path}; run 01_load_compustat.py first."
        )
    s1 = pd.read_parquet(s1_path, columns=["sector_id"])
    retained_sector_ids = sorted(s1["sector_id"].dropna().unique().tolist())
    logger.info(f"  retained sector_ids ({len(retained_sector_ids)}): {retained_sector_ids}")

    pooled = load_pooled_alpha(config, retained_sector_ids, logger)
    nodes = build_pooled_nodes(pooled["alpha_unit"].to_numpy(), config, logger)

    out_path = ensure_dir(PATHS.intermediary) / S5_OUT
    nodes.to_parquet(out_path, index=False, compression="snappy")
    logger.info(
        f"wrote {out_path.relative_to(PATHS.project_root)} "
        f"({out_path.stat().st_size / 1e3:.2f} KB; {len(nodes)} rows)"
    )


if __name__ == "__main__":
    sys.exit(main())
