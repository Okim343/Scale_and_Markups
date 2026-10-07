# GNR core: shared estimator

Source-agnostic Python implementation of the Gandhi-Navarro-Rivers (2020) gross-output
production-function estimator. It is used by both data sources:

- `01_Compustat/00_Hubmer_RTS/01_code` (stages R1-R3): one pooled estimation across NAICS2 industries.
- `02_Orbis/02_GNR`: one estimation per NAICS2 cell, as in Hubmer et al. (2025).

The modules hold no paths, I/O or data-source logic. Callers handle reading, renaming and
writing, then pass DataFrames and config blocks (the `trimming`, `panel`, `step1`, `step2` and
`postprocess` sections of a GNR `config.yaml`).

| Module | Content | Moved from |
|---|---|---|
| `00_original_code/` | MATLAB research code by Hubmer et al. that the Python translates (`est_gnr.m` and helpers). | `00_Hubmer_RTS/00_original_code/` |
| `gnr_model.py` | Polynomials, Step 1 NLS objective, nested Step 2 GMM objective, elasticities, Sobol starts, parallel multistart. | `00_Hubmer_RTS/01_code/gnr_model.py` (unchanged) |
| `gnr_panel.py` | `construct_gnr_vars` (logs, share `s = m - r`, consecutive-year lags), `trim_panel` (est_gnr.m trimming). | stage R1 |
| `gnr_estimate.py` | `run_step1`, `build_step2_sample`, `run_step2`. | stage R2 |
| `gnr_postprocess.py` | `reconstruct_firm_year`: eps, melast, omega, eta, kelast, lelast, rts from saved parameters. | stage R3 |

The refactor moved code without changing logic. The Compustat smoke run (`config_smoke.yaml`,
R1-R3) gives identical panels and `firm_year_rts.parquet` (equal DataFrames) and identical
step 1 / step 2 result JSONs before and after the move.

## Search modes (`search` in the `step1` / `step2` config blocks)

- `extended` (default; used by Compustat, whose configs do not set the key): the original
  Python multistart. Step 1 adds 20 jittered OLS starts; Step 2 adds 2 economic starts and 50
  wide Sobol starts in [-3, 3], solves each with scipy BFGS, and keeps the polish only if it
  improves the objective. Output is identical to before the switch existed (Compustat smoke
  run: equal step 1 / step 2 JSONs and `firm_year_rts`).
- `matlab` (Orbis config): `est_gnr.m`'s search. Its start sets only; Step 2 local solves are
  `gnr_model.fminunc_like` (R2026a `fminunc` quasi-newton defaults: forward differences with
  step sqrt(eps)*max(|x|,1), relative optimality / step tolerance 1e-6, 400 iterations, 100
  evaluations per variable); the best solution is taken among converged runs (MultiStart's
  positive exit flags, including flag 5, "cannot decrease along the search direction"); then
  an `fminsearch`-like polish (TolX = TolFun = 1e-4) whose result is kept. The line search is
  scipy's, so this approximates `fminunc` rather than replicating it. On Orbis cell 51 it
  reproduces the MATLAB code's seed-invariant answer (RTS 0.94 vs 0.96; the gap is a trimming
  difference, below).

Known remaining divergence from `est_gnr.m`: `gnr_panel.trim_panel` applies the 1% percentile
trim after the materials-share band and counts industry / year sizes after trimming; the
MATLAB code computes percentiles and counts on the untrimmed sample. On Orbis cell 51 this
keeps 35,474 vs 36,119 rows.

## Using it

Put this directory on `sys.path`; it is not a package because its name starts with a digit.
The Compustat `utils.py` does this on import:

```python
GNR_CORE = PROJECT_ROOT / "03_Empirical" / "00_GNR_core"
sys.path.insert(0, str(GNR_CORE))
from gnr_panel import construct_gnr_vars, trim_panel
from gnr_estimate import build_step2_sample, run_step1, run_step2
from gnr_postprocess import reconstruct_firm_year
```

Conventions: the caller renames levels to `rev_lvl, cap_lvl, mat_lvl, lab_lvl, wagebill_lvl` and
provides an id column (default `gvkey`), `year` and an integer `ind2d`. The parallel multistart
uses spawned processes, which inherit `sys.path` from the parent, so workers find these modules
without further setup.

Within-cell estimation needs no special handling. Step 2 drops the first industry dummy, so a
single `ind2d` leaves zero dummy columns, and the productivity process is a constant plus the
cubic in lagged omega.
