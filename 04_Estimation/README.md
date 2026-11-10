# 04_Estimation: calibration of the pooled model to the Orbis Spain targets

This folder calibrates the model in `05_Simulation/orbis_model` to the Orbis Spain bundle. The package `orbis_estimation` imports the model from `05_Simulation` (importing it puts that folder on `sys.path`), so the dependency runs one way, from estimation to simulation.

## The paper's calibration (market-free, Zipf)

`01_calibrate_zipf.py` fits four parameters to four moments with $\eta = 1.2$ fixed.

| Parameter | Target | Data |
|---|---|---|
| $\gamma$ (within-market elasticity) | `mu_cw`, the cost-weighted aggregate markup | 1.1815 |
| $\xi$ (Pareto tail of productivity $\nu$) and $N$ (Poisson mean firms per market) | Zipf tail index of firm sales at the top 5% and at the top 2% | 1.1079, 1.1451 |
| $\bar\rho$ (rank copula of $\alpha$ and $\nu$, negative allowed) | RTS gap, top 5% minus bottom 50% of firms by sales within NAICS4 | 0.0460 |

It is one `trf` least-squares fit through the anchored GE solve, which also pins the productivity scale `v_min` through the median-output anchor. Residuals are `weight * (model - target) / (1 + |target|)` with weights 10 on `mu_cw` and 3.162 on the other three. The system is exactly identified, so the weights shape the path and not the solution. Every evaluation draws at $M = 5000$ markets with fixed common random numbers (`monte_carlo` in `config_estimation_zipf.yaml`). The Zipf model counterpart (`orbis_estimation/zipf.py::tail_index`) uses the same estimator as the data step `03_Empirical/02_Orbis/04_Moments/01_code/12_zipf_targets.py`.

**Alpha cap.** Every evaluation caps the drawn alpha at `alpha_cap_factor * gamma/(gamma-1)` at that evaluation's gamma (`parameters.alpha_cap_factor` in the simulation config, 0.99), since above the CES markup floor the anchored GE has no root. Draws come from the support nodes below the cap with their mass renormalized. The model RTS gap is computed on capped alpha, while its target uses raw RTS. The factor is written to the params file (`model.alpha_cap_factor`), and `05_Simulation` refuses a config with a different factor.

## Scripts

Run from this folder. Outputs go to `out_results/` (gitignored).

```bash
python 01_calibrate_zipf.py [--M 5000] [--max-nfev 25] [--seed S] [--out-dir D]
python 02_validate_zipf.py --params baseline=out_results/zipf/<run>/calibrated_params_zipf.yaml [--params LABEL=FILE ...] [--M N]
python 03_calibrate_restricted.py --economy {hom_alpha,hom_v,hom_both} [--scan | --gate0] [--initial-from-scan] [--M N] [--workers W]
```

| Script | Does | Writes |
|---|---|---|
| `01_calibrate_zipf.py` | the calibration above, then the post-calibration sunk entry cost (EMX free entry, `entry.varphi`) | in `out_results/zipf/`: `calibrated_params_zipf.yaml` (parameters, reference GE prices, fit, provenance with the bundle sha256), manifest, fit CSV, `calibration_checkpoint.jsonl` (one row per evaluation) |
| `02_validate_zipf.py` | untargeted validation of one or more params files: Zipf at the top 1% and Hill variants, CR4/CR20 against NAICS2 and NAICS4 data, markup percentiles and variances, the composition of the top 1% of firms, the split of `mu_cw` into floor and share-driven parts | tables at `--out` |
| `03_calibrate_restricted.py` | refits the restricted economies of Table 2 to the baseline targets they can reach, with the baseline weights, bounds, M and seed (`config_estimation_restricted.yaml`) | `out_results/zipf/restricted/<economy>/calibrated_params_<economy>.yaml` with a `restriction` block, manifest, fit CSV, checkpoint |

The restricted economies are the canonical model on modified inputs. `hom_alpha` (the $\nu$-economy) uses one $\alpha$ node at the harmonic mean of the capped $F^\alpha$ and fits $(\xi, N, \gamma)$. `hom_v` (the $\alpha$-economy) sets $\xi = \infty$, so $\nu = $ `v_min`, and `hom_both` (homogeneous) sets both. These two cannot generate a Pareto tail in sales, so they fit only $\gamma$ to `mu_cw` with $N$ at the baseline value. `--gate0` checks that the restricted code path reproduces the baseline fit exactly at the baseline parameters, and `--scan` evaluates a starting grid. Evaluations are checkpointed with their exact `x`, and a restart replays them, so a long fit survives a killed job. An evaluation that runs past the watchdog limit exits with code 75 and is recorded as a timeout.

**Earlier concentration-targeting calibration.** `01_calibrate.py` and `02_validate.py` (with `config_estimation.yaml`) fit $(\xi, N, \gamma, \eta, \bar\rho)$ to `mu_cw`, CR4, CR20, the EMX slope and the RTS gap. They are kept for comparison and are not used in the paper, since Orbis concentration depends on sample coverage and on the market definition.

## Inputs

| Input | Role |
|---|---|
| `03_Empirical/02_Orbis/04_Moments/03_outdata/aggregate_moments.yaml` | `mu_cw` and `alpha_rts_gap_top5_bottom50` (read by `orbis_estimation/targets.py`), model inputs (read by `orbis_model/primitives.py`) |
| `.../zipf_targets.yaml` | the two Zipf indices, plus untargeted data counterparts for `02_validate_zipf.py` |
| `.../F_alpha.parquet`, `.../manifest.yaml` | alpha support, bundle provenance (its sha256 goes into the outputs) |
| `config_estimation_zipf.yaml` | bundle and simulation-config paths, M and seed, fixed $\eta$, weights, bounds, initial values, solver settings |
| `05_Simulation/config_simulation.yaml` | model settings shared with the simulation (`beta, delta_K, H`, alpha cap, participation, market solver) |

The estimation and simulation configs must point at the same bundle (checked). The bundle is gitignored (WRDS-licensed).

**Handoff.** `05_Simulation` reads the params file (`paths.params_file` of its config) and refuses to run if the recorded bundle sha256 differs from the bundle it loads. Provenance blocks record the sha256 of `aggregate_moments.yaml`, `F_alpha.parquet` and `manifest.yaml`, the sha256 of both configs, the git commit and the dirty flag of `04_Estimation/` and `05_Simulation/`.

## Package layout

```
orbis_estimation/
  config.py              config loader, initial_params
  targets.py             bundle reader for the targets
  zipf.py                market-free calibration: Zipf tail index, moments at parameters, the trf fit
  restricted.py          restricted-heterogeneity economies, checkpoint replay, watchdog
  objective.py           anchored moments at a parameter vector, weighted residuals
  inner.py               concentration-targeting fit (01_calibrate.py), delta-method SEs, checkpoints
  anchor_diagnostics.py  untargeted corr(alpha, log sales) diagnostics
  entry_cost.py          post-calibration sunk entry cost
  diagnostics.py, outer.py, identification.py   small helpers
```

## Tests

```bash
python -m pytest
```

Tests that need the real bundle skip when it is absent, and the rest use synthetic bundles in pytest temporary folders.

## Cluster

`cluster/` is gitignored and kept local. It holds the Marvin2 and workstation submit scripts, the sync script and cluster copies of the configs. The jobs read the cluster copies, not the local configs, so edit both. Run python directly, not through `srun`. The cluster folder is not a git checkout and is synced with `bash 04_Estimation/cluster/sync_to_marvin2.sh`. If a job dies, the best iterate can be recovered from the checkpoint file.
