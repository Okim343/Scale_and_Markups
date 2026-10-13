"""Step 02: model moments at given parameters against the Orbis targets.

Ported from ``cmd_validate`` in ``04_Code/steady_state/__main__.py``. Two modes:

* ``--params FILE``: the old ``validate``. Simulates at the params file's
  vector with its frozen ``v_min`` (no anchoring), on the calibration draws
  (``monte_carlo.M`` and ``seed`` unless overridden), and prints model vs target.
* no ``--params``: one objective evaluation at ``initial_values`` of the
  estimation config, through the anchored GE solve that the calibration uses
  (it also solves ``v_min``). Prints model vs target, the weighted residuals and
  the objective value ``0.5 * sum(r^2)``. With ``--write-params PATH`` it writes
  a provenance-stamped params file at that vector (an evaluation, not a fit)
  that 05_Simulation can load.

Reads   <paths.bundle_dir>/{F_alpha.parquet, aggregate_moments.yaml, manifest.yaml}
Writes  nothing, or <PATH> with --write-params

    python 02_validate.py [--config config_estimation.yaml] [--params FILE] [--M N] [--seed S]
                          [--write-params PATH]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd

from orbis_estimation.config import initial_params, load_config
from orbis_estimation.objective import pooled_moments_at_params, weighted_relative_residuals
from orbis_estimation.targets import load_targets
from orbis_model.config import setup_logger
from orbis_model.params import build_provenance, load_params_file, write_params_file
from orbis_model.primitives import load_primitives
from orbis_model.simulation.cross_section import run_simulation
from orbis_model.simulation.moments import economy_moments

EST_DIR = Path(__file__).resolve().parent
SIM_DIR = EST_DIR.parent / "05_Simulation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="default: config_estimation.yaml")
    parser.add_argument("--params", default=None, help="params file; default: evaluate at initial_values")
    parser.add_argument("--M", type=int, default=None, help="override monte_carlo.M")
    parser.add_argument("--seed", type=int, default=None, help="override monte_carlo.seed")
    parser.add_argument("--write-params", default=None,
                        help="without --params: write a params file at the evaluated vector")
    return parser


def comparison_table(model, targets, cfg) -> pd.DataFrame:
    keys = cfg.moment_keys
    resid = weighted_relative_residuals(model, targets, moment_keys=keys, weights=cfg.weights)
    return pd.DataFrame({
        "moment": keys,
        "model": [float(model[k]) for k in keys],
        "target": [float(targets[k]) for k in keys],
        "weight": [cfg.weights[k] for k in keys],
        "weighted_residual": resid,
    })


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.params and args.write_params:
        raise SystemExit("--write-params applies only to the evaluation at initial_values")
    cfg = load_config(args.config)
    sim = cfg.simulation
    logger = setup_logger("validate", cfg.log_level)
    t0 = time.time()
    primitives = load_primitives(cfg.bundle_dir, H=sim.parameters.H)
    targets = load_targets(cfg.bundle_dir)
    M = int(args.M or cfg.M)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    common = dict(
        beta=sim.parameters.beta, delta_K=sim.parameters.delta_K,
        K_switch=sim.participation.K_switch,
        love_of_variety=sim.participation.love_of_variety,
        market_solver_kwargs=sim.market.to_kwargs(),
    )
    if args.params:
        params, wf, initial, _ = load_params_file(args.params, bundle_dir=cfg.bundle_dir)
        result = run_simulation(
            primitives, params, M=M, master_seed=seed, wf_reference=wf, initial=initial, **common,
        )
        model = economy_moments(result, primitives)
        equilibrium = result.equilibrium
        logger.info(f"frozen-v_min simulation at {args.params}")
    else:
        params = initial_params(cfg, primitives)
        model, reference, equilibrium = pooled_moments_at_params(
            params, primitives, M=M, rng=seed, **common,
        )
        logger.info("anchored objective evaluation at initial_values")
    table = comparison_table(model, targets, cfg)
    cost = 0.5 * float(np.sum(table["weighted_residual"] ** 2))
    logger.info(f"(xi, N, gamma, eta, rho_bar) = ({params.xi:.6g}, {params.N:.6g}, {params.gamma:.6g}, "
                f"{params.eta:.6g}, {params.rho_bar:.6g}); v_min={equilibrium.v_min:.6g}; M={M}, seed={seed}")
    logger.info("moments vs targets:\n" + table.to_string(index=False, float_format=lambda v: f"{v: .6g}"))
    logger.info(f"objective 0.5*sum(r^2) = {cost:.6g}")
    logger.info(f"untargeted: mu_cw_alpha={model['mu_cw_alpha']:.6g}, "
                f"mean_active_count={model['mean_active_count']:.6g} "
                f"(data mean_firms_per_sector={primitives.mean_firms_per_sector}); "
                f"y_anchor={equilibrium.y_anchor:.6g}, y_sw={equilibrium.y_sw:.6g}; "
                f"GE converged={equilibrium.converged}, residual={equilibrium.residual:.2e}")
    if args.write_params:
        provenance = build_provenance(
            bundle_dir=cfg.bundle_dir,
            configs={"estimation": cfg.path, "simulation": sim.path},
            code_dirs=(EST_DIR, SIM_DIR), script="04_Estimation/02_validate.py --write-params",
        )
        provenance.update({"M": M, "seed": seed})
        path = write_params_file(
            args.write_params,
            replace(params, v_min=equilibrium.v_min, y_hat=equilibrium.y_hat),
            reference={
                "w": reference.w, "R": reference.R, "X_market": reference.X_market,
                "wf_reference": reference.wf_reference, "Ed_full_pool": reference.Ed_full_pool,
                "P": reference.P, "L": reference.L,
            },
            provenance=provenance,
            fit={
                "note": "evaluation at config initial_values (not a fit); v_min from the anchored solve",
                "cost": cost,
                "moments": {
                    row.moment: {"model": row.model, "target": row.target, "weight": row.weight}
                    for row in table.itertuples()
                },
            },
        )
        logger.info(f"wrote {path}")
    logger.info(f"done ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
