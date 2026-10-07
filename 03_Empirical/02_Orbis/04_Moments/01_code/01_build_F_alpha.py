"""Step 01: pooled scalability distribution F(alpha) from the Orbis Spain GNR estimates.

alpha is the firm's GNR returns to scale, rts = melast + lelast + kelast, estimated per
NAICS2 cell (02_GNR). F(alpha) is one pooled distribution over all Spanish firms in every
cell, with no sector dimension, in the node format the model reads
(04_Code/steady_state/io_bundle.py: node_id, u, weight, alpha; no sector_id).

Construction:
  1. Step-2 firm-years with a finite alpha in the window (config F_alpha.window).
  2. Collapse to one unit per firm, its window-mean alpha (collapse: firm_mean).
  3. Winsorize the pooled firm units at winsor_pct.
  4. Discretize the pooled empirical CDF into n_nodes equal-mass nodes at the quantile
     midpoints u_k = (k - 0.5) / N: alpha_k = quantile(u_k), weight 1 / N.

The Compustat builder (01_Compustat/01_industry_alpha/02_code/05_build_F_alpha.py) also
imputes the labor elasticity from KLEMS, shrinks the dispersion and clips the support. All
three countered Compustat's labor data; Orbis measures the wage bill, so none applies here.

Reads   <gnr.outdata_dir>/{firm_year_rts.parquet, manifest.yaml}
Writes  <output.intermediary_dir>/F_alpha.parquet          node table
        <output.intermediary_dir>/F_alpha_manifest.yaml    provenance and summary moments

    python 01_build_F_alpha.py [--config config_moments_orbis_es.yaml]
"""

import argparse
import datetime
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(__file__).resolve().parent
PROJECT_ROOT = CODE.parents[3]


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_logger(level):
    logger = logging.getLogger("F_alpha")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True).stdout.strip()
        return out or None
    except OSError:
        return None


def load_units(config, logger):
    """Pooled firm units (or firm-years) of alpha in the window; returns (units, sample info)."""
    g, cfg = config["gnr"], config["F_alpha"]
    gnr_dir = resolve(g["outdata_dir"])
    gnr_manifest = yaml.safe_load((gnr_dir / "manifest.yaml").read_text())
    if gnr_manifest.get("code_dirty"):
        logger.warning("GNR estimates were produced from uncommitted code (manifest code_dirty)")
    id_col, alpha_col = g["id_col"], cfg["alpha_col"]
    fy = pd.read_parquet(gnr_dir / "firm_year_rts.parquet",
                         columns=[id_col, "year", "ind2d", "in_step2_sample", alpha_col])
    y0, y1 = map(int, cfg["window"])
    fy = fy[fy["in_step2_sample"] & fy["year"].between(y0, y1) & np.isfinite(fy[alpha_col])]
    logger.info(f"step-2 firm-years in [{y0}, {y1}] with finite {alpha_col}: {len(fy):,} "
                f"({fy[id_col].nunique():,} firms, {fy['ind2d'].nunique()} cells)")
    info = {"firm_years": int(len(fy)), "firms": int(fy[id_col].nunique()),
            "cells": sorted(int(c) for c in fy["ind2d"].unique()),
            "firm_years_by_cell": {int(c): int(n) for c, n in fy["ind2d"].value_counts().sort_index().items()}}

    collapse = cfg["collapse"]
    if collapse == "firm_mean":
        units = fy.groupby(id_col)[alpha_col].mean().to_numpy()
        logger.info(f"collapsed to {len(units):,} firm units "
                    f"(mean {len(fy) / len(units):.2f} years per firm)")
    elif collapse == "firm_year":
        units = fy[alpha_col].to_numpy()
        logger.info("collapse: firm_year, pooled firm-year cross-section (robustness)")
    else:
        raise ValueError(f"F_alpha.collapse must be firm_mean or firm_year, got {collapse!r}")
    return units.astype(float), info, gnr_manifest


def build_nodes(units, cfg, logger):
    """Winsorized equal-mass discretization of the pooled empirical CDF."""
    n_nodes, min_obs = int(cfg["n_nodes"]), int(cfg["min_obs"])
    if units.size < min_obs:
        raise ValueError(f"only {units.size} pooled units (< min_obs = {min_obs})")
    w_lo, w_hi = map(float, cfg["winsor_pct"])
    lo_q, hi_q = np.quantile(units, [w_lo, w_hi])
    n_wins = int(((units < lo_q) | (units > hi_q)).sum())
    alpha_w = np.clip(units, lo_q, hi_q)
    logger.info(f"winsorized at [{w_lo}, {w_hi}] -> [{lo_q:.4f}, {hi_q:.4f}]; {n_wins:,} units moved")

    u = (np.arange(n_nodes) + 0.5) / n_nodes
    nodes = pd.DataFrame({
        "node_id": np.arange(1, n_nodes + 1, dtype="int16"),
        "u": u.astype("float32"),
        "weight": np.full(n_nodes, 1.0 / n_nodes, dtype="float32"),
        "alpha": np.quantile(alpha_w, u).astype("float32"),
        "alpha_source": pd.array([f"orbis_es_gnr_{cfg['alpha_col']}_{cfg['collapse']}"] * n_nodes,
                                 dtype="string"),
    })
    a = nodes["alpha"].to_numpy(dtype=float)
    logger.info(f"{n_nodes} equal-mass nodes: alpha min / mean / max = "
                f"{a.min():.4f} / {a.mean():.4f} / {a.max():.4f}; share of mass above 1: "
                f"{(a > 1).mean():.3f}")
    return nodes, {"winsor_bounds": [float(lo_q), float(hi_q)], "units_winsorized": n_wins}


def summary(x):
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {"mean": float(x.mean()), "sd": float(x.std()),
            **{f"p{p}": float(v) for p, v in zip([1, 10, 25, 50, 75, 90, 99], q)},
            "share_above_1": float((x > 1).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CODE / "config_moments_orbis_es.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    logger = setup_logger(config.get("log_level", "INFO"))
    cfg = config["F_alpha"]

    units, info, gnr_manifest = load_units(config, logger)
    nodes, wins = build_nodes(units, cfg, logger)

    out = resolve(config["output"]["intermediary_dir"])
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out / "F_alpha.parquet", index=False)
    manifest = {
        "description": "Pooled scalability distribution F(alpha) over Orbis Spain firms: "
                       "equal-mass nodes of the window-mean GNR returns to scale.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "settings": cfg,
        "gnr_estimates": {"dir": config["gnr"]["outdata_dir"], "commit": gnr_manifest.get("commit"),
                          "config_sha256": gnr_manifest.get("config_sha256"),
                          "root_selection": gnr_manifest.get("root_selection")},
        "sample": info,
        **wins,
        "units_before_winsor": summary(units),
        "nodes": summary(nodes["alpha"]),
    }
    (out / "F_alpha_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    logger.info(f"wrote {out / 'F_alpha.parquet'} and F_alpha_manifest.yaml")


if __name__ == "__main__":
    main()
