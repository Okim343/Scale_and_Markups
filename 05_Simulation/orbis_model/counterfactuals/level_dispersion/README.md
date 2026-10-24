# Joint capability/scalability experiment

`06_level_dispersion.py` solves market and planner allocations for four economies:
the calibrated joint distribution, sector-common scalability, sector-common
capability, and both primitives sector-common. Harmonic means preserve each
sector's mean `1/alpha` and mean anchor marginal cost `1/v`. All cases share
participation, the output anchor, preference parameters and baseline `chi`.

## Compatibility with Tables 2 and 3

The saved Orbis reference predates the change from clipping to truncation in the
calibration draw. The experiment requires the explicit policy
`--reference-draw-policy integer_nodes_raw_rank_copula_then_clip_v1` to reconstruct
the historical RNG order and clipped alpha values. This compatibility routine is
local to this experiment. It does not alter the current calibration draw routine.

Before calculating either new economy, the script freshly solves the baseline
and common-scalability market/planner pairs. Their quantities, prices, markups,
welfare, firm count and `chi` must agree symmetrically with
`sorting/checkpoint_baseline.json` and `checkpoint_hom_sector.json` at tolerance
`1e-6 + 2e-6 * max(abs(fresh), abs(saved))`. The saved outer solves used tolerance
`1e-5`; the new outer solve uses `1e-7`, so tiny normalization improvements are
expected. An additional, tighter gate reproduces both saved allocations at their
recorded prices (`rtol=atol=1e-8`) to rule out changes in model or firm draws.
Calibration and bundle hashes,
assigned parameters, reconstructed firm counts and alpha-v correlation are
checked first. A failed reference gate stops the experiment.

Inactive padding is discarded only after reconstructing the complete original
draw. Every active firm and market remains unchanged. Each allocation must
converge with fixed participation and labor, positive consumption, and a satisfied
resource constraint. Welfare is computed from utility at frozen `chi` and checked
against the consumption ratio.

## Output

`out_results/level_dispersion/` contains:

- `checkpoint_<economy>.json`: quantities, prices, welfare and solve timings.
- `run_manifest.json`: source hashes, economics, code fingerprint and draw policy.
- `frozen_calibrated_params.yaml`: exact calibration used by this experiment.
- `validation.json`: fresh-versus-saved comparisons and numerical checks.
- `level_dispersion_economies.parquet`, `level_dispersion.yaml`, `results.md`:
  the four economies and both table panels.

Panel A reports `100*(Y_PE/Y_ME-1)` and `100*lambda`, each relative to that
economy's own market. Panel B takes four-way finite differences of those same
percentage changes, so its components are percentage-point contributions:

```
common       = both_common
capability   = alpha_common - both_common
scalability  = v_common - both_common
interaction  = baseline - alpha_common - v_common + both_common
```

Components sum exactly to the calibrated total for both output and welfare.
The interaction is a joint-heterogeneity residual, not the within-sector
reassignment sorting effect. No log-gap shares or nonlinear transformations of
individual log contributions enter either panel.

## Workstation

Run from an isolated copy of `05_Simulation`, with frozen configuration,
calibration and sorting checkpoints, and a read-only path to the bundle. The
submission script uses 24 CPUs, 64 GB RAM and one BLAS thread per worker. It writes
only the isolated experiment outputs and never changes shared calibration files.

```
export TASK_PYTHON=/home/enrico/projects/Scale_and_Markups/.venv/bin/python
sbatch cluster/submit_level_dispersion_workstation.sh \
  --sorting-dir ../inputs/sorting --out-dir ../out_results/level_dispersion
```

For a short functional run, add `--smoke-M 20` and choose a separate output
directory. Smoke results explicitly omit the production reference gates and
must not be used in the paper. Checkpoints resume only with an identical run
fingerprint; use a new output directory after changing code or inputs.
