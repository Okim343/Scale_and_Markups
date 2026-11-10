# 05_Simulation: the pooled steady-state model on the Orbis Spain bundle

This folder owns the structural model (package `orbis_model`) and everything that runs it at given
parameters: the cross-section simulation and the market vs planner welfare comparison. The calibration
lives in `04_Estimation/`, which imports `orbis_model`; the dependency runs one way, estimation to
simulation. It descends from the retired Compustat package (`04_Code/steady_state`, now archived
outside the repository) with the model equations, solvers, anchoring, GE normalization, parallel backend
and welfare formulas unchanged.

Targets and model inputs come from the Orbis Spain bundle, documented in
[`03_Empirical/README.md`](../03_Empirical/README.md).

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
python 04_sorting_sweep.py [--M 400] [--seed S] [--out-dir D] [--fig-dir D] [--welfare-ref-dir D] [--fixed-input-ref-dir D] [--grid r1,r2] [--labels a,b] [--validation-only]
python 05_planner_expansion.py [--welfare-dir D] [--fixed-input-dir D] [--out-dir D] [--corr-target X | --sorting-dir D]
python 06_level_dispersion.py --reference-draw-policy draw_pool_truncated_v1 [--sorting-dir D] [--workers N] [--out-dir D]
python 07_heterogeneity_anatomy.py --reference-draw-policy draw_pool_truncated_v1 [--level-dispersion-dir D] [--labels a,b] [--workers N]
python 08_recalibrated_anatomy.py --params hom_alpha=FILE --params hom_v=FILE --params hom_both=FILE [--params baseline=FILE] [--labels a,b] [--workers N] [--assemble-only]
python 09_recalibrated_channels_figure.py [--in-dir D] [--fig-stem STEM]
python 10_sorting_markup_moments.py --M N --sweep-dir D [--out-dir D] [--labels a,b]
python -m orbis_model.figures.simulation_figures --results-dir out_results [--window 2015_2019]
```

| Script | Does | Writes (in `out_results/`, gitignored; `<window>` = bundle active window, e.g. `2015_2019`) |
|---|---|---|
| `01_simulate.py` | draws the common random numbers at the params vector (frozen `v_min`), solves the pooled GE (P = 1, L = 1) on the exogenous Poisson mask | `sim_panel_<window>.parquet` (firm panel), `sim_moments_<window>.parquet`, `sim_economy_moments_<window>.yaml`, `simulate_manifest_<window>.yaml` |
| `02_welfare.py` | market and planner allocations on common draws and mask, uniform-markup allocation at mu_bar = market `mu_cw`, two-channel decomposition | `welfare_<window>.yaml`, `welfare_decomposition_<window>.parquet`, `sim_panel_market_<window>.parquet`, `sim_panel_planner_<window>.parquet`, `welfare_manifest_<window>.yaml` |
| `03_fixed_input_welfare.py` | fixed-input planner chain ME -> PE_I -> PE_K -> PE on the draws and mask of `02_welfare.py` (see below) | in `fixed_input/`: `fixed_input_regimes_<window>.parquet`, `fixed_input_welfare_<window>.yaml`, `sim_panel_planner_fixed_input_<window>.parquet`, `sim_panel_planner_fixed_k_<window>.parquet`, `fixed_input_manifest_<window>.yaml` |
| `04_sorting_sweep.py` | both welfare lenses across alpha<->v arrangements: copula re-draws over rho_bar, within-sector permutations, homogeneous alpha (see below) | in `sorting/`: `sorting_economies.parquet`, `sorting.yaml`, `results.md`, `validation.json`, `checkpoint_<label>.json`; `out_figs/sorting/sorting_curve.{pdf,png}` |
| `06_level_dispersion.py` | four market/planner pairs with capability/scalability common or heterogeneous; fresh baseline and common-scalability gates against saved sorting results | in `level_dispersion/`: economy checkpoints, frozen calibration, provenance and validation, `level_dispersion_economies.parquet`, `level_dispersion.yaml`, and the two-panel `results.md` |
| `07_heterogeneity_anatomy.py` | both welfare lenses for the four economies of 06 (heterogeneity collapsed in one draw at the baseline parameters); its baseline checkpoint is a gate for 08 | in `heterogeneity_anatomy/`: checkpoints, legs, `results.md` |
| `08_recalibrated_anatomy.py` | both welfare lenses for the four economies of Table 2, each refit by `04_Estimation/03_calibrate_restricted.py`; the baseline must reproduce 02, 03 and 07 | in `recalibrated_anatomy/`: checkpoints, `recalibrated_anatomy_economies.parquet`, `recalibrated_legs.csv`, `recalibrated_anatomy.yaml`, `results.md` |
| `09_recalibrated_channels_figure.py` | Figure 2 of the paper, the walk from the homogeneous economy to the baseline stacked by both lenses | `02_Drafts/figures/orbis/fig_recalibrated_channels_orbis_zipf.{pdf,png}` |
| `10_sorting_markup_moments.py` | re-solves only the market of every sorting arrangement on the sweep's draws and records markup spread and placement (the markup columns of Table 3); each market is checked against the sweep row | in `<sweep-dir>/markup_moments/`: one JSON per arrangement, `sorting_markup_moments.parquet` |
| `orbis_model/figures/simulation_figures.py` | lambda-contribution heatmap and markup distribution from the welfare outputs | `out_figs/welfare/<tag>/` |

## Zipf calibration (the paper)

`config_simulation_zipf.yaml` runs the suite on the market-free calibration (`04_Estimation/01_calibrate_zipf.py`,
params `04_Estimation/out_results/zipf/job3162/calibrated_params_zipf.yaml`). It is a copy of
`config_simulation.yaml` with only `paths` changed: the params file and the output folders `out_results/zipf`
and `out_figs/zipf`, so the outputs of the current calibration are never overwritten. Model and solver
settings must stay identical in both files (checked in `tests/test_zipf_plumbing.py`). Pass
`--config config_simulation_zipf.yaml` to 01 to 04 and 06; smoke runs add `--M` and explicit output folders.
`05_planner_expansion.py --sorting-dir D` takes the corr(alpha, v) gate target from the baseline of a sorting
run on the same M, seed and params file instead of the default (the calibrated clipped M = 10000 value).
`06_level_dispersion.py` takes M and seed from its sorting reference (`--sorting-dir`) and rebuilds that
draw with `draw_pool` (policy `draw_pool_truncated_v1`); sorting results saved before the switch from
clipped to truncated alpha draws (commit d7c8300) cannot pass its gates.

Run order for the paper, all with `--config config_simulation_zipf.yaml`:

| Paper object | Chain |
|---|---|
| Table 3, Figure 3, Figure B1 | `02_welfare.py --M 10000` and `03_fixed_input_welfare.py --M 10000` (gate references), `04_sorting_sweep.py --M 10000`, then `10_sorting_markup_moments.py --M 10000 --sweep-dir <sorting out>` |
| Table 2, Figure 2 | `06_level_dispersion.py` and `07_heterogeneity_anatomy.py` on a sorting reference, `08_recalibrated_anatomy.py` with the restricted params files, then `09_recalibrated_channels_figure.py`. The gates compare the baseline across steps, so 02, 03, 04, 06, 07 and 08 run at the same M and seed (M = 5000 in the paper) |

The sweep writes `sorting_curve` and `sorting_lenses` to `out_figs/zipf/sorting/`. The paper uses copies of
them in `02_Drafts/figures/orbis/` named `fig_sorting_curve_orbis_zipf.pdf` and `fig_sorting_lenses_orbis_zipf.pdf`.

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

`03_fixed_input_welfare.py` extends the fixed-capital planner of the Compustat version to
materials. Every allocation imposes P = 1 and L = 1. The chain is

| Regime | Label | Pinned | Freed price |
|---|---|---|---|
| ME | `market` | (market allocation) | |
| PE_I | `planner_fixed_input` | K = K_ME, M = M_ME | R and the shadow materials price P_mat |
| PE_K | `planner_fixed_k` | K = K_ME | R |
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

## Sorting sweep

On the draws, seed and
Poisson mask of `02_welfare.py`, each arrangement solves ME, PE_I, PE_K and PE
(`solve_fixed_input_chain`) and the uniform-markup U (`two_channel_decomposition`, mu_bar = that
arrangement's market `mu_cw`), and reports Lens A (`realloc`, `materials`, `capital`) and Lens B (`disp`,
`level`), with `Delta = (1 - beta)(W_b - W_a)` summing exactly to `Delta_total` in each lens. chi is frozen
at the baseline market value; `v_min` and `y_hat` stay at the calibration. The default plan:

| Labels | Kind | Arrangement |
|---|---|---|
| `baseline` | copula | the calibrated draw |
| `rho_<r>`, r in {-0.9, -0.6, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9} | copula | `draw_pool` at rho_bar = r, same seed: alpha, tilde_alpha, eps and the mask are common random numbers (asserted bitwise); v is equal in distribution only |
| `reverse`, `shuffle_0`, `shuffle_1`, `sorted` | permutation | within-sector pairings of the baseline draw; both marginals bitwise identical (asserted) |
| `hom_sector`, `hom_cost` | homogeneous | sector harmonic mean of alpha; one economy-wide cost-weighted harmonic mean at the baseline ME |

The baseline runs first and must reproduce the `02_welfare.py` and `03_fixed_input_welfare.py` outputs at
the same M, seed and params file (their manifests are checked before any solve) to relative tolerance
1e-6, or the run stops with `validation.json`. Every arrangement asserts convergence, L = 1 within 1e-4,
the fixed K and M targets, the uniform markup, the consumption-versus-welfare gap and both lens
identities (1e-12), and is checkpointed atomically; a rerun resumes only if inputs and solver code are
unchanged (`run_manifest.json`). `sorting.yaml:summary` holds the sorting split
`Delta_base = Delta_hom + (Delta_shuf - Delta_hom) + (Delta_base - Delta_shuf)`, secant slopes of the copula
curve and the gaps of the permutation points to it. Smoke runs need references at the same M, e.g.
`02_welfare.py --M 400 --out-dir out_results/smoke_sorting_ref` and the matching `03` call, passed via
`--welfare-ref-dir` and `--fixed-input-ref-dir`.

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
  counterfactuals/ sorting (alpha arrangements, per-arrangement solve and gate, readout)
  figures/         house_style, simulation_figures
```

## Tests

```bash
python -m pytest
```

Tests that need the real bundle skip when it is absent; all other tests use small synthetic bundles
written to pytest temporary folders. `tests/fixtures/legacy_anchored_mc_fixture.npz` is synthetic.

## Cluster

`cluster/` (gitignored, local only) holds `submit_welfare.sh` and the Marvin2
copy of the config (`config_simulation_marvin2.yaml`, `solver.parallel.n_jobs` pinned to 16). The job reads
the cluster copy, not the local config: edit both. Run python directly, not via `srun`. Sync with
`04_Estimation/cluster/sync_to_marvin2.sh`; submit from this folder with `sbatch cluster/submit_welfare.sh`
after the calibration job. The lab workstation scripts (`submit_*_workstation.sh`) read
`cluster/config_simulation_workstation.yaml`; `submit_fixed_input_workstation.sh M [OUT_DIR]` runs
`03_fixed_input_welfare.py` (24 CPUs, 64G); `submit_sorting_workstation.sh M [OUT_DIR]` reruns
`02_welfare.py` and `03_fixed_input_welfare.py` as gate references (skip with `SKIP_REFS=1`) and then
`04_sorting_sweep.py` (24 CPUs, 64G; about 1.5 to 2 h at M = 10000).
