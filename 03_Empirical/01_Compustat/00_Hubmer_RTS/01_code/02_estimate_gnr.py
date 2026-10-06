"""
Stage R2 — Pooled GNR estimation (Step 1 NLS + Step 2 nested GMM).

Reads:
  02_intermediary/gnr_panel_trimmed.parquet   (from 01_build_gnr_panel.py)

Writes:
  02_intermediary/step1_results.json
  02_intermediary/step2_results.json

Step 1 (est_gnr.m / nls_obj.m):
  Multistart constrained NLS of the materials share equation
      s = log(P1 @ gamma') - eps
  over the 10-term polynomial P1 in (k, m, l), subject to P1 @ gamma > 0.
  Starting set: {feasibility-rescaled OLS guess, gamma0, 2*gamma0,
  first n_starts Sobol points}, filtered by nls_const_filter. Each start is
  optimized with L-BFGS-B (analytic gradient, bounds +/- 100*gamma0); the
  best solution is polished with Nelder-Mead (fminsearch analog).
  Then: eps, bigeps = mean(exp(eps)), gamma_scaled = soln / bigeps.

Step 2 (est_gnr.m / step2_obj.m, nested):
  Selection: trimmed sample AND LP1 @ soln > 0 (lag feasibility); the
  lagged-wage filter is ONLY applied when panel.require_wages is true
  (deliberate divergence from the MATLAB script — wages are not a core
  source of identification; see README.md).
  y-tilde:  rt  = r  - eps  - int_melast
            lrt = lr - leps - int_lmelast
  Nested GMM over the 5 integration-constant parameters; the Markov-process
  coefficients (industry dummies + cubic in lagged omega) are concentrated
  out by OLS inside each objective evaluation. Instruments
  IV = [k, k^2, ll, ll^2, k*ll]; W = identity (just-identified: 5 moments,
  5 parameters, so gmm2s is unnecessary, matching the MATLAB default).

Run:
  python 02_estimate_gnr.py [--config config_smoke.yaml]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from utils import (  # utils puts 03_Empirical/00_GNR_core on sys.path
    PATHS,
    ensure_dir,
    intermediary_dir,
    parse_config_arg,
    setup_logger,
)
from gnr_estimate import build_step2_sample, run_step1, run_step2

TRIM_IN = "gnr_panel_trimmed.parquet"
S1_OUT = "step1_results.json"
S2_OUT = "step2_results.json"


# -----------------------------------------------------------------------------
def main() -> None:
    config = parse_config_arg("Stage R2 — pooled GNR estimation (steps 1 and 2)")
    logger = setup_logger("R2", config.get("log_level", "INFO"))
    inter = intermediary_dir(config)

    trim_path = inter / TRIM_IN
    if not trim_path.exists():
        raise FileNotFoundError(
            f"{trim_path} missing; run 01_build_gnr_panel.py first."
        )
    est = pd.read_parquet(trim_path)
    logger.info(
        f"loaded {trim_path.relative_to(PATHS.project_root)} ({len(est):,} rows)"
    )

    # --- Step 1 ---------------------------------------------------------------
    logger.info("=== Step 1: NLS share regression ===")
    s1 = run_step1(est, config["step1"], logger)
    s1["config_snapshot"] = {
        "pipeline_version": config.get("pipeline_version"),
        "step1": config["step1"],
        "trimming": config["trimming"],
        "panel": {k: v for k, v in config["panel"].items()},
        "mapping": config["mapping"],
    }
    s1_path = ensure_dir(inter) / S1_OUT
    with open(s1_path, "w") as f:
        json.dump(s1, f, indent=2)
    logger.info(f"wrote {s1_path.relative_to(PATHS.project_root)}")

    # --- Step 2 ---------------------------------------------------------------
    logger.info("=== Step 2: nested GMM ===")
    require_wages = bool(config["panel"].get("require_wages", False))
    sub, counts = build_step2_sample(
        est, np.asarray(s1["gamma_raw"]), require_wages, logger
    )
    s2 = run_step2(sub, s1, config["step2"], logger)
    s2["sample_counts"] = counts
    s2["require_wages"] = require_wages
    s2_path = inter / S2_OUT
    with open(s2_path, "w") as f:
        json.dump(s2, f, indent=2)
    logger.info(f"wrote {s2_path.relative_to(PATHS.project_root)}")


if __name__ == "__main__":
    sys.exit(main())
