# 04_Estimation: calibration of the pooled model to the Orbis Spain targets

This folder calibrates the five common parameters `(xi, N, gamma, eta, rho_bar)` of the model in
`05_Simulation/orbis_model` to the Orbis Spain targets, with one joint `trf` least-squares fit evaluated
through the anchored GE solve (which also pins the capability scale `v_min`). The package
`orbis_estimation` is a port of `04_Code/steady_state/calibration/`; it imports the model from
`05_Simulation` (importing `orbis_estimation` puts that folder on `sys.path`) and never from `04_Code/`.

Targets, data definitions and the decisions behind them:
[`02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md`](../02_Drafts/md_files/ORBIS_migration/ORBIS_SPAIN_MOMENTS.md).

## Inputs

| Input | Role |
|---|---|
| `03_Empirical/02_Orbis/04_Moments/03_outdata/aggregate_moments.yaml` | targets `mu_cw, cr4, cr20, top1pct, top5pct, emx_slope` and `alpha_rts_gap_top5_bottom50` (read by `orbis_estimation/targets.py`); model inputs (read by `orbis_model/primitives.py`) |
| `.../F_alpha.parquet`, `.../manifest.yaml` | alpha support; bundle provenance (its sha256 goes into the outputs) |
| `config_estimation.yaml` | bundle and simulation-config paths, M and seed, targeted moments and weights, bounds, initial values, inner solver settings, output folder |
| `05_Simulation/config_simulation.yaml` | model settings shared with the simulation: `beta, delta_K, H`, participation, market solver |

The two configs must point at the same bundle (checked). The bundle is gitignored (WRDS-licensed).

Targeted moments are the keys of `weights` in `config_estimation.yaml` (weight 0 keeps a moment
computed and reported). Starting weights: `mu_cw 10, cr4 1, cr20 1, top1pct 0, top5pct 0,
emx_slope 3.162, alpha_rts_gap_top5_bottom50 3.162` (EMX weighting from 04_Code; the RTS gap inherits the
weight of the dropped alpha-sales correlation). Residuals are `(model - target) / (1 + |target|)`.

The data `mu_cw` ((V + kexp)-weighted DLEU markup, V = mate + staf, no SG&A) is the target of the model's
pure cost-weighted markup `sum TC mu / sum TC`; the model's `mu_cw_alpha` (revenue over variable cost) is
reported but not targeted. `alpha_rts_gap_top5_bottom50` is the rho_bar target; `rho_bar` is calibrated in
`[-0.95, 0.95]` (negative values allowed, Salgado et al. find -0.25) and starts at 0.

**Market definition mismatch (accepted).** The bundle measures concentration (`cr4, cr20, top1pct,
top5pct`) and the RTS gap within NAICS4 markets, but `emx_slope` across 14 NAICS2 cells. The model
computes all of them on the same simulated markets, as in 04_Code; there is no nested market structure.
Likewise the bundle's `mean_firms_per_sector` (about 4,083 per NAICS4 market) is a diagnostic only: it is
above the N bounds [5, 1000] and the pool size H = 2500, and it is never used as an initial N.

## Scripts

Run from this folder; every script takes `--config` (default `config_estimation.yaml`).

```bash
python 01_calibrate.py [--M 300] [--max-nfev 3] [--seed S] [--out-dir D]
python 02_validate.py  [--M 500] [--write-params out_results/params_warm_start.yaml]
python 02_validate.py  --params out_results/calibrated_params.yaml [--M 500]
```

| Script | Does | Writes (in `out_results/`, gitignored) |
|---|---|---|
| `01_calibrate.py` | joint fit from `initial_values` within `bounds`; post-calibration sunk entry cost (EMX free entry, `entry.varphi`) | `calibrated_params.yaml` (parameters, reference GE prices, fit, identification, provenance with the bundle sha256 that 05_Simulation checks), `calibration_manifest.yaml` (provenance, output sha256), `calibration_fit_<window>.csv`, `calibration_checkpoint.jsonl` (one row per objective evaluation) |
| `02_validate.py` | without `--params`: one objective evaluation at `initial_values` (anchored solve, solves `v_min`), model vs target, weighted residuals, objective value; `--write-params PATH` writes a params file at that vector (an evaluation, not a fit) for 05_Simulation. With `--params FILE`: the old `validate`, a frozen-`v_min` simulation at the file's vector against the targets | nothing, or the `--write-params` file |

The handoff: `05_Simulation` reads `out_results/calibrated_params.yaml` (its `paths.params_file`) and
refuses to run if the recorded bundle sha256 differs from the bundle it loads. Provenance blocks record the
sha256 of `aggregate_moments.yaml`, `F_alpha.parquet` and `manifest.yaml`, the sha256 of both configs, the
git commit and the dirty flag of `04_Estimation/` and `05_Simulation/`.

There is no `03_identification.py`: the 04_Code identification module is a static map, not a runnable
entry point (ported as `orbis_estimation/identification.py`).

## Package layout

```
orbis_estimation/
  config.py              config_estimation.yaml loader, initial_params (new)
  targets.py             bundle reader for the targets (new)
  objective.py           anchored moments at a parameter vector, weighted residuals
  inner.py               calibrate_pooled: the joint trf fit, delta-method SEs, checkpoints
  anchor_diagnostics.py  untargeted corr(alpha, log sales) diagnostics
  entry_cost.py          post-calibration sunk entry cost
  diagnostics.py, outer.py, identification.py   small helpers ported unchanged
```

## Tests

```bash
python -m pytest
```

Tests that need the real bundle skip when it is absent; the rest use synthetic bundles in pytest temporary
folders.

## Cluster

`cluster/` (gitignored, local only, like `04_Code/marvin2/`) holds `submit_calibrate.sh`,
`sync_to_marvin2.sh` and `config_estimation_marvin2.yaml` (which points at
`05_Simulation/cluster/config_simulation_marvin2.yaml`, `solver.parallel.n_jobs` pinned to 16). Gotchas
carried over from `04_Code/marvin2/README.md`: the job reads the cluster config copies, not the local
configs, so edit both; run python directly, not via `srun`; the cluster folder is not a git checkout, so
sync with `bash 04_Estimation/cluster/sync_to_marvin2.sh`; if a job dies, recover the best iterate from
`calibration_checkpoint.jsonl`. Submit from this folder with `sbatch cluster/submit_calibrate.sh`
(M = 5000, max_nfev = 40).

## Not ported yet

The counterfactual packages of `04_Code/counterfactuals/` will be ported later.
