"""Command-line interface for the pooled steady-state model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .calibration import calibrate_pooled, compute_sunk_entry_cost
from .config import load_config
from .io_bundle import load_bundle
from .model.normalization import PooledParams
from .model.pool import draw_pool
from .model.pricing import DEFAULT_PHI_V
from .pooled_inputs import PooledInputs
from .simulation.cross_section import build_firm_panel, run_simulation
from .simulation.moments import economy_moments, pooled_moments_table
from .welfare import solve_market_and_planner, two_channel_decomposition, welfare_decomposition_table


def _pooled_inputs(cfg) -> PooledInputs:
    bundle = load_bundle(cfg.bundle_dir)
    return PooledInputs.from_bundle(
        bundle, eta_init=cfg.calibration.eta_init, H=cfg.parameters.H,
        markup_target=cfg.calibration.markup_target,
    )


def _initial_params(cfg, inputs: PooledInputs) -> PooledParams:
    initial = cfg.raw.get("initial_values", {}) or {}
    # initial_values.eta is authoritative for the warm start when present; it
    # falls back to calibration.eta.init so the plain `calibrate` command keeps
    # its job-4382852 eta default.
    eta = float(initial.get("eta", cfg.calibration.eta_init))
    gamma = max(float(initial.get("gamma", 6.0)), eta + 0.1)
    return PooledParams(
        xi=float(initial.get("xi_default", 8.0)),
        N=float(initial.get("N", inputs.empirical_N or 150.0)), gamma=gamma, eta=eta,
        rho_bar=inputs.rho_bar,
        a=inputs.a, H=inputs.H, phi_v=inputs.phi_v,
    )


def _load_calibration_yaml(
    path: Path, *, a: float = 1.0 / 3.0, H: int = 1000,
    phi_v: float = DEFAULT_PHI_V,
) -> tuple[PooledParams, float | None, tuple[float, float] | None]:
    """Load the common scalar schema; per-sector maps are rejected."""
    with open(path) as stream:
        data = yaml.safe_load(stream)
    schema = data.get("schema")
    if schema != "pooled-emx-v2":
        raise ValueError(f"unsupported calibration schema {schema!r}; rerun calibration for pooled-emx-v2")
    values = data.get("parameters", data)
    if any(isinstance(values.get(key), dict) for key in ("xi", "N", "gamma", "eta")):
        raise ValueError("calibration YAML must contain common scalar parameters")
    params = PooledParams(
        xi=float(values["xi"]), N=float(values["N"]),
        gamma=float(values["gamma"]), eta=float(values["eta"]),
        v_min=float(values.get("v_min", 1.0)),
        y_hat=float(values.get("y_hat", 1.0)),
        rho_bar=float(values.get("rho_bar", 0.0)),
        a=float(values.get("a", a)), H=int(values.get("H", H)),
        phi_v=float(values.get("phi_v", phi_v)),
    )
    reference = data.get("reference", {}) or {}
    wf = reference.get("wf_reference")
    initial = (
        (float(reference["w"]), float(reference["X_market"]))
        if "w" in reference and "X_market" in reference else None
    )
    return params, (float(wf) if wf is not None else None), initial


def _calibration_path(cfg, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    default = cfg.out_results_dir / "calibration_pooled.yaml"
    return default if default.exists() else None


def _params_for_command(cfg, inputs, explicit: str | None):
    path = _calibration_path(cfg, explicit)
    return (
        _load_calibration_yaml(
            path, a=inputs.a, H=inputs.H, phi_v=inputs.phi_v
        )
        if path else (_initial_params(cfg, inputs), None, None)
    )


def _write_yaml(path: Path, value: dict) -> None:
    value = _native(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def _native(item):
    if isinstance(item, dict):
        return {key: _native(val) for key, val in item.items()}
    if isinstance(item, (list, tuple)):
        return [_native(val) for val in item]
    if isinstance(item, np.generic):
        return item.item()
    return item


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    M = int(args.M or cfg.monte_carlo.M)
    # Warm start: prefer the existing calibration YAML's (xi, N, gamma, eta);
    # fall back to config initial_values when no YAML is present.
    warm_path = _calibration_path(cfg, getattr(args, "calibration", None))
    if warm_path is None:
        # The committed warm start lives under the calib_pooled/ subdirectory.
        candidate = cfg.out_results_dir / "calib_pooled" / "calibration_pooled.yaml"
        warm_path = candidate if candidate.exists() else None
    if warm_path is not None:
        initial, _, _ = _load_calibration_yaml(
            warm_path, a=inputs.a, H=inputs.H, phi_v=inputs.phi_v
        )
    else:
        initial = _initial_params(cfg, inputs)
    cal_cfg = cfg.raw.get("calibration", {}) or {}
    iv = cfg.raw.get("initial_values", {}) or {}
    rho_bar_init = float(iv.get("rho_bar", initial.rho_bar or inputs.rho_bar))
    out_dir = Path(args.out_dir) if args.out_dir else cfg.out_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result = calibrate_pooled(
        inputs, M=M, rng=int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed),
        initial=(initial.xi, initial.N, initial.gamma, initial.eta, rho_bar_init),
        xi_bounds=(cfg.calibration.xi_lo, cfg.calibration.xi_hi),
        N_bounds=(cfg.calibration.N_lo, cfg.calibration.N_hi),
        eta_bounds=(cfg.calibration.eta_lo, 10.0),
        rho_bar_bounds=tuple((cal_cfg.get("rho_bar_bounds", {}) or {}).get(k, d) for k, d in (("lo", 0.0), ("hi", 0.95))),
        weights=(cal_cfg.get("weights", {}) or {}).get("pooled"),
        corr_alpha_log_sales_weight=float(
            ((cal_cfg.get("weights", {}) or {}).get("pooled") or {}).get(
                "corr_alpha_log_sales", 0.0
            )
        ),
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(),
        max_nfev=int(args.max_nfev or cfg.calibration.max_nfev),
        checkpoint_path=out_dir / "calibration_checkpoint.jsonl",
    )
    varphi = float((cfg.raw.get("parameters", {}) or {}).get("varphi", 0.04))
    sunk_entry_cost = compute_sunk_entry_cost(
        result.equilibrium, beta=cfg.parameters.beta, varphi=varphi
    )
    payload = {
        "schema": "pooled-emx-v2",
        "parameters": {
            "xi": result.xi, "N": result.N,
            "gamma": result.gamma, "eta": result.eta,
            "rho_bar": result.rho_bar,
            "v_min": result.v_min,
            "y_hat": result.equilibrium.y_hat,
            "a": inputs.a, "H": inputs.H, "phi_v": inputs.phi_v,
        },
        "reference": {
            "w": result.reference.w, "R": result.reference.R,
            "X_market": result.reference.X_market,
            "wf_reference": result.reference.wf_reference,
            "Ed_full_pool": result.reference.Ed_full_pool,
            "P": result.reference.P, "L": result.reference.L,
        },
        "entry": {
            "sunk_entry_cost_F": sunk_entry_cost,
            "varphi": varphi,
            "note": "post-calibration EMX free-entry backout; welfare-only, "
                    "does not enter the static cross-section",
        },
        "fit": {
            "success": result.success, "cost": result.cost, "nfev": result.nfev,
            "moments": {
                k: {
                    "model": float(result.model_moments[k]),
                    "target": float(result.target_moments[k]),
                }
                for k in (*result.moment_keys, "corr_alpha_log_sales", "corr_alpha_log_sales_partial_z")
            },
            "diagnostics": {
                "mu_cw_alpha": float(result.model_moments["mu_cw_alpha"]),
                "mean_active_count": float(result.model_moments["mean_active_count"]),
                "empirical_N_sales_weighted": float(inputs.empirical_N or float("nan")),
                "y_sw": float(result.equilibrium.y_sw),
                "y_anchor": float(result.equilibrium.y_anchor),
                "corr_alpha_log_sales_partial_z": float(result.model_moments["corr_alpha_log_sales_partial_z"]),
                "corr_alpha_log_sales_partial_v": float(result.model_moments["corr_alpha_log_sales_partial_v"]),
            },
        },
        "identification": {
            "eta_jacobian_column_norm": result.eta_jacobian_column_norm,
            "eta_standard_error": float(result.standard_errors[3]),
            "jacobian_rank": result.jacobian_rank,
            "jacobian_condition": result.jacobian_condition,
            "share_std": float(result.model_moments["share_std"]),
            "share_range": float(result.model_moments["share_range"]),
        },
    }
    _write_yaml(out_dir / "calibration_pooled.yaml", payload)
    print(f"calibrated (xi, N, gamma, eta, rho_bar, v_min) = "
          f"({result.xi:.6g}, {result.N:.6g}, {result.gamma:.6g}, {result.eta:.6g}, "
          f"{result.rho_bar:.6g}, {result.v_min:.6g})")
    print(f"anchor (median) y_anchor={result.equilibrium.y_anchor:.6g} "
          f"(y_sw={result.equilibrium.y_sw:.6g})")
    print(f"wrote {out_dir / 'calibration_pooled.yaml'}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    M = int(args.M or cfg.monte_carlo.M_final)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.out_results_dir
    start, end = cfg.active_window
    sim = run_simulation(
        inputs, params, M=M,
        master_seed=int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed),
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        initial=initial,
        out_dir=out_dir, panel_filename=f"sim_panel_{start}_{end}.parquet",
    )
    pooled_moments_table(sim, inputs).to_parquet(out_dir / "sim_pooled_moments.parquet", index=False)
    _write_yaml(out_dir / "sim_economy_moments.yaml", economy_moments(sim, inputs))
    print(f"simulated {M} pooled markets; mean active count={sim.equilibrium.aggregates.mean_active_count:.4g}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    sim = run_simulation(
        inputs, params, M=int(args.M or cfg.monte_carlo.M),
        master_seed=int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed),
        beta=cfg.parameters.beta, delta_K=cfg.parameters.delta_K,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        initial=initial,
    )
    print(yaml.safe_dump(_native(economy_moments(sim, inputs)), sort_keys=False))
    return 0


def cmd_welfare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    M = int(args.M or cfg.monte_carlo.M_final)
    seed = int(args.seed if args.seed is not None else cfg.monte_carlo.master_seed)
    support = inputs.alpha_support if inputs.alpha_support is not None else np.linspace(0.6, 1.20, 500)
    draw = draw_pool(support, xi=params.xi, rho_bar=params.rho_bar, N=params.N, M=M, H=params.H, rng=seed, v_min=params.v_min)
    pair = solve_market_and_planner(
        draw.alpha, draw.v, params, beta=cfg.parameters.beta,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(), wf_reference=wf,
        active_mask=draw.active_mask, initial=initial,
    )
    decomp = two_channel_decomposition(
        pair.me, pair.pe, params=params, chi_me=pair.me.chi,
        beta=cfg.parameters.beta, phi=cfg.parameters.phi,
        delta_K=cfg.parameters.delta_K, n_firms_cs=inputs.n_firms_cs,
        K_switch=cfg.participation.K_switch,
        love_of_variety=cfg.participation.love_of_variety,
        market_solver_kwargs=cfg.market.to_kwargs(),
    )
    out_dir = Path(args.out_dir) if args.out_dir else cfg.out_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    build_firm_panel(pair.me, tilde_alpha=draw.tilde_alpha, regime="market").to_parquet(out_dir / "sim_panel_market.parquet", index=False)
    build_firm_panel(pair.pe, tilde_alpha=draw.tilde_alpha, regime="planner").to_parquet(out_dir / "sim_panel_planner.parquet", index=False)
    welfare_decomposition_table(pair.me, pair.pe, decomp["uniform"], n_firms_cs=inputs.n_firms_cs).to_parquet(out_dir / "welfare_decomposition.parquet", index=False)
    serializable = {k: float(v) for k, v in decomp.items() if k != "uniform"}
    serializable["wf_reference"] = pair.me.participation.wf_reference
    _write_yaml(out_dir / "welfare.yaml", serializable)
    print(f"lambda_total={decomp['lambda_total']:.6g}; identity residual={decomp['identity_residual']:.3e}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steady_state")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--out-dir"); calibrate.add_argument("--M", type=int)
    calibrate.add_argument("--seed", type=int); calibrate.add_argument("--max-nfev", type=int)
    calibrate.set_defaults(func=cmd_calibrate)
    for name, func in (("simulate", cmd_simulate), ("validate", cmd_validate), ("welfare", cmd_welfare)):
        command = sub.add_parser(name)
        command.add_argument("--calibration", default=None)
        command.add_argument("--M", type=int); command.add_argument("--seed", type=int)
        if name != "validate": command.add_argument("--out-dir")
        command.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
