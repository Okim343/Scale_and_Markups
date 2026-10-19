# 05_Simulation: the pooled steady-state model on the Orbis Spain bundle

This folder owns the structural model (package `orbis_model`) and everything that runs it at given
parameters: the cross-section simulation and the market vs planner welfare comparison. The calibration
lives in `04_Estimation/`, which imports `orbis_model`; the dependency runs one way, estimation to
simulation. The code is a port of `04_Code/steady_state/` (model, simulation, welfare, figures) with the
model equations, solvers, anchoring, GE normalization, parallel backend and welfare formulas unchanged.
Nothing here imports from `04_Code/`.

Targets and model inputs come from the Orbis Spain bundle; their definitions and the decisions behind them
are in [`02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md`](../02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md).

## Inputs

| Input | Where | Role |
|---|---|---|
| `F_alpha.parquet` | `03_Empirical/02_Orbis/04_Moments/03_outdata/` | pooled scalability support (`node_id, u, weight, alpha`, no `sector_id`); raw in the bundle, capped at draw time at `alpha_cap_factor * gamma/(gamma-1)` (`parameters.alpha_cap_factor`, 0.99) |
| `aggregate_moments.yaml` | same | model inputs `active_window`, `a`, `phi_v`, `n_firms_cs` (required), `mean_firms_per_sector` (diagnostic only) |
| params file | `paths.params_file`, default `04_Estimation/out_results/calibrated_params.yaml` | calibrated `(xi, N, gamma, eta, rho_bar)`, anchored `v_min`, GE reference prices, provenance |
| `config_simulation.yaml` | this folder | paths, Monte Carlo size and seed, `beta, delta_K, phi, H`, participation, solver, welfare size |

The bundle is gitignored (WRDS-licensed Orbis data); never copy it into tracked files. Both scripts
**refuse to run** when the params file's `provenance.bundle_sha256` (sha256 of `aggregate_moments.yaml` and
`F_alpha.parquet`) differs from the bundle in `paths.bundle_dir`.

## Scripts

Run from this folder; every script takes `--config` (default `config_simulation.yaml`) and `--params`
(overrides `paths.params_file`).

```bash
python 01_simulate.py [--M 500] [--seed S] [--out-dir D]
python 02_welfare.py  [--M 300] [--seed S] [--out-dir D]
python 03_fixed_input_welfare.py [--M 400] [--seed S] [--out-dir D] [--skip-full-planner] [--skip-fixed-k]
python -m orbis_model.figures.simulation_figures --results-dir out_results [--window 2015_2019]
```

| Script | Does | Writes (in `out_results/`, gitignored; `<window>` = bundle active window, e.g. `2015_2019`) |
|---|---|---|
| `01_simulate.py` | draws the common random numbers at the params vector (frozen `v_min`), solves the pooled GE (P = 1, L = 1) on the exogenous Poisson mask | `sim_panel_<window>.parquet` (firm panel), `sim_moments_<window>.parquet`, `sim_economy_moments_<window>.yaml`, `simulate_manifest_<window>.yaml` |
| `02_welfare.py` | market and planner allocations on common draws and mask, uniform-markup allocation at mu_bar = market `mu_cw`, two-channel decomposition | `welfare_<window>.yaml`, `welfare_decomposition_<window>.parquet`, `sim_panel_market_<window>.parquet`, `sim_panel_planner_<window>.parquet`, `welfare_manifest_<window>.yaml` |
| `03_fixed_input_welfare.py` | fixed-input planner chain ME -> PE_I -> PE_K -> PE on the draws and mask of `02_welfare.py` (see below) | in `fixed_input/`: `fixed_input_regimes_<window>.parquet`, `fixed_input_welfare_<window>.yaml`, `sim_panel_planner_fixed_input_<window>.parquet`, `sim_panel_planner_fixed_k_<window>.parquet`, `fixed_input_manifest_<window>.yaml` |
| `orbis_model/figures/simulation_figures.py` | lambda-contribution heatmap and markup distribution from the welfare outputs | `out_figs/welfare/<tag>/` |

Every manifest records the bundle sha256s, the bundle manifest sha256, the config sha256, the params file
sha256, the git commit and dirty flag, M and the seed.

Model moments reported by `01_simulate.py`: `mu_cw` (pure cost-weighted markup `sum TC mu / sum TC`, the
target of the data `mu_cw`), `cr4, cr20, top1pct, top5pct` (within market, market-sales-weighted),
`emx_slope` (cross-market OLS of the sales-weighted inverse markup on HHI), `alpha_rts_gap_top5_bottom50`
(new: within-market sales percentile, mean demeaned alpha of the top 5% minus the bottom 50%, pooled over
all firms of all markets; defined exactly as `rts_gap` in
`03_Empirical/02_Orbis/04_Moments/01_code/06_alpha_moments.py`), and untargeted diagnostics such as
`mu_cw_alpha` (revenue over variable cost) and the GE aggregates.

## Fixed-input planner

`03_fixed_input_welfare.py` ports `04_Code/counterfactuals/fixed_input_welfare` and extends it to
materials. Every allocation imposes P = 1 and L = 1. The chain is

| Regime | Label | Pinned | Freed price |
|---|---|---|---|
| ME | `market` | (market allocation) | |
| PE_I | `planner_fixed_input` | K = K_ME, M = M_ME | R and the shadow materials price P_mat |
| PE_K | `planner_fixed_k` | K = K_ME (the 04_Code object) | R |
| PE | `planner` | nothing beyond L = 1 (the `02_welfare.py` planner) | |

P_mat replaces P = 1 inside the gross-output unit cost `Omega^g(w, R, P_mat)`, as R does for capital;
materials spend stays `(1 - phi_v) TC`, so the materials quantity is `M = (1 - phi_v) TC / P_mat`.
Consumption is `C = Q - M - delta_K K` in every regime; PE_I and PE_K hold K at K_ME, so their investment
equals the market's. The legs are `lambda_realloc` (ME -> PE_I), `lambda_materials` (PE_I -> PE_K) and
`lambda_capital` (PE_K -> PE), with `(1 + l_r)(1 + l_m)(1 + l_c) = 1 + lambda_total` and the additive
`Delta = ln(1 + lambda)`. chi is held at the ME value. With M and seed at the `02_welfare.py` defaults,
`lambda_total` reproduces its value. Solver settings are in the `fixed_input:` config block (`tol`,
`max_nfev`). With `--skip-fixed-k`, PE_I warm-starts from ME and the PE_I -> PE leg is reported as
`lambda_materials_capital`.

`PooledEquilibrium.walras_check` adds the lump-sum rebate `(P_mat - 1) M` to income and reports the
capital rent `(R - R_euler) K` (already inside RK). The budget is not tight in any regime: income minus
C + I is exactly `sales - Q` (`aggregation_gap`), because every market spends the common `X_market`
rather than its CES demand share; `closure_residual` is zero up to rounding.

## Package layout

```
orbis_model/
  config.py        config_simulation.yaml loader, logger
  primitives.py    bundle reader for the model inputs (no targets)
  params.py        params file read/write, bundle sha256 check, provenance
  model/           pool (alpha-v copula, rho_bar in (-1, 1)), market, market_batch, pricing,
                   normalization, participation, aggregator (pooled moments incl. the RTS gap)
  simulation/      cross_section, moments
  welfare/         allocations, welfare_metrics, fixed_input (fixed-input planner chain)
  figures/         house_style, simulation_figures
```

## Tests

```bash
python -m pytest
```

Tests that need the real bundle skip when it is absent; all other tests use small synthetic bundles
written to pytest temporary folders. `tests/fixtures/legacy_anchored_mc_fixture.npz` is synthetic.

## Cluster

`cluster/` (gitignored, local only, like `04_Code/marvin2/`) holds `submit_welfare.sh` and the Marvin2
copy of the config (`config_simulation_marvin2.yaml`, `solver.parallel.n_jobs` pinned to 16). The job reads
the cluster copy, not the local config: edit both. Run python directly, not via `srun`. Sync with
`04_Estimation/cluster/sync_to_marvin2.sh`; submit from this folder with `sbatch cluster/submit_welfare.sh`
after the calibration job. The lab workstation scripts (`submit_*_workstation.sh`) read
`cluster/config_simulation_workstation.yaml`; `submit_fixed_input_workstation.sh M [OUT_DIR]` runs
`03_fixed_input_welfare.py` (24 CPUs, 64G).

## Not ported yet

The counterfactual packages of `04_Code/counterfactuals/` will be ported later, apart from
`fixed_input_welfare` (now `03_fixed_input_welfare.py`).
