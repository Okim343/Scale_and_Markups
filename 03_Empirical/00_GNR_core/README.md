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
