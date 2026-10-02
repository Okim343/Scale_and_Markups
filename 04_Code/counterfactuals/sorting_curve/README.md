# Sorting curve

I replace the three-point sorting overlay with observed partial-permutation
allocations under both welfare lenses. The paper, deck, and old overlay are untouched.

Run from `04_Code` with the calibration and empirical bundle present:

```bash
python -m counterfactuals.sorting_curve.run_sorting_curve --workers 16
python -m counterfactuals.sorting_curve.figure
```

The production plan first solves identity at q=0 and compares all welfare and
consumption levels, every lambda and Delta, capital, markup, correlation, the
market rental rate, market output diagnostic, and baseline chi against the
saved baseline to relative tolerance 1e-6. Failure writes `validation.json` and
stops before any new arrangement. The frozen anchor is asserted separately.
Then it runs identity q={0.2,0.4,0.6,0.8}, a second q=0.4 draw, reverse
q={0.33,0.67}, and two copula checks rho_bar={0.45,0.70}. Use
`--no-rho-check` to omit the optional checks or `--validation-only` to stop
following baseline validation.

`partial_permute` in the sibling `scalability_sorting.permute` starts with the
identity or reverse assignment, selects round(q*n) active slots without
replacement in each sector, and shuffles their alpha and tilde_alpha together.
The active alpha marginal and the entire capability array remain bit-identical;
inactive slots and participation remain unchanged. The existing modes retain
their behavior. Permutation seeds are JSON-encoded SeedSequence entropy lists in
each row. The realized pooled active correlation supplies the figure x axis.

The base draw uses calibrated params for xi, rho_bar, N, H, and v_min. The
copula checks reuse its seed and assert alpha, tilde_alpha, and active_mask
bitwise equality. The RNG ordering also preserves eps. Capability changes;
its marginal is identical in distribution only, not bitwise. Such a check
changes more features of the joint than its Pearson correlation alone.

All new allocations solve independently through existing helpers. Both planners
inherit the arrangement's market participation and wf_reference. Chi stays at
the saved baseline; y_hat stays at the calibrated value. Convergence, labor,
participation, fixed capital, uniform markup, and both exact log identities
are enforced. The saved reverse and all saved shuffle draws are retained as
cached endpoints. Their nonmarket consumption levels are reconstructed from saved W
with L=1 and the frozen chi, beta, phi, checked against saved market C to
relative tolerance 1e-6. Saved market C and L are retained. Numerical labor
residuals and consumption versus welfare gaps are reported for new solves;
existing solver tolerance permits deviations up to 1e-4. Cached nonmarket
labor residuals are unavailable. Both welfare lens identities still telescope
to machine precision.

Outputs default to `out_results/counterfactuals/sorting_curve/`:

- `sorting_curve_economies.parquet`: one row per solved or saved arrangement,
  all W/C levels, K_ME/K_PE, mu_cw, each lambda and Delta, convergence flags,
  participation diagnostics, design metadata, and source status.
- `sorting_curve.yaml`: provenance, input hashes, summary, secant slopes,
  raw and normalized rankings, replica difference, and copula gaps.
- `results.md`: interpretation and exact file/key citations for reported numbers.
- `validation.json`: baseline relative differences and pass/fail.
- `run_manifest.json` and `checkpoint_<label>.json`: durable per-arrangement
  checkpoints; rerunning skips completed solves only when provenance matches.

The figure is `out_figs/counterfactuals/sorting_convexity_curve.{pdf,png}`.
It uses the existing house style. Four legs are normalized independently to
reverse=0 and baseline=1. Repeated partial draws and saved shuffle draws are
averaged on Delta for connecting lines; every partial replicate is marked.
Copula checks remain in the results table and readout but are omitted from the figure.
Segments guide the eye; no smooth fit is imposed. Sampled secant convexity
cannot establish convexity throughout or a structural kink. Reports distinguish
raw leg ranking from normalized movement ranking. Copula gaps use interpolation,
so they are descriptive, not an exact matched-correlation comparison.

For a quick diagnostic, use `--M-smoke 30 --out-dir /tmp/sorting-curve-smoke`.
A smoke solves its own reverse and shuffle anchors; it cannot validate the
production baseline. Smoke figures must use a separate figure destination.

On the workstation, activate a project Python environment with numpy, scipy,
pandas, pyarrow, pyyaml, and matplotlib. Inspect sinfo and squeue, create the
output directory, then submit `sbatch counterfactuals/sorting_curve/job.slurm`
from `04_Code`. The job requests 16 CPUs and 96 GiB; solver process workers
consume the CPUs while each BLAS worker uses one thread. No GPU is needed.
Transfer the empirical bundle, calibration, and saved counterfactual outputs
without adding them to version control. No commit or push is required.

## Parallel completion after the baseline gate

When the workstation is idle, independent arrangements can run four at a time.
After `checkpoint_baseline.json` and a passing `validation.json` exist, stop
the sequential job with `scancel` and submit `parallel_job.slurm` from `04_Code`.
It requests 64 CPUs and 192 GiB. Four outer processes each use the 16 solver
workers recorded by the validated run. Only the parent publishes tables and
figures, avoiding concurrent writes. Each finished arrangement is checkpointed
immediately. The parallel driver checks source/config/calibration hashes and
revalidates the saved baseline before proceeding. Do not run the sequential
and parallel drivers against the same output folder simultaneously.

## Omit inactive numerical padding

The original draw still uses calibrated H=2500, the same RNG ordering, and the
full arrays for permutations and marginal assertions. `--compact` removes only
trailing columns with no active firms in any sector before numerical evaluation.
Active alpha, tilde_alpha, and capability are checked bitwise. The solver's H
field is adjusted solely to satisfy its array-width contract; economic parameters
and the original draw H remain unchanged. Each new row records `draw_pool_H`
and `numerical_solver_H`. Participation is still seeded from its own market.

First run `sbatch counterfactuals/sorting_curve/compact_validation.slurm`.
It re-solves the full-size baseline and q=0.2 with the shorter numerical arrays,
compares all W, C, lambda, and Delta keys against the full-width solves at
relative tolerance 1e-6, and writes `compact_validation.json` only on success.
Then stop the old driver before submitting `compact_job.slurm`; never let two
parents publish into the same output directory. The compact job resumes saved
points using four arrangements with four solver workers each, 16 CPUs total,
and 48 GiB. Failed validation blocks this path. Original completed points are
retained and input hashes still checked. This changes numerical storage, not
the draw or counterfactual design.
