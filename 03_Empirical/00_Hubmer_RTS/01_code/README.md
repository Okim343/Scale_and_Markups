# Hubmer/GNR Pooled Firm-Year RTS Estimation Pipeline

Python reimplementation of the Gandhi–Navarro–Rivers (2020, JPE) gross-output
production-function estimator, adapted from the MATLAB research code in
`../00_original_code/`. Spec / source of truth:
`02_Drafts/md_files/hubmer_rts_implementation.md`.

The pipeline produces a **pooled firm-year RTS panel** (not per-NAICS2
estimates) on the broad post-1997 curated Compustat panel, and exports it for
the `01_industry_alpha` pipeline, which converts it into the model's
`F_i^alpha` object.

## Run order

```bash
cd 03_Empirical/00_Hubmer_RTS/01_code
python 01_build_gnr_panel.py                 # panel + trimming      (~minutes)
python 02_estimate_gnr.py                    # GNR steps 1 and 2     (~30 min - 2 h)
python 03_postprocess_gnr.py                 # firm-year RTS panel   (~minutes)
python 04_export_rts_for_alpha_pipeline.py   # export + publish      (~seconds)
```

**Parallelism.** The multistart local solves in stage 02 run on
`step1.n_workers` / `step2.n_workers` processes (default 10; set to 1 for
serial). Worker BLAS pools are pinned to one thread automatically — without
that, oversubscription makes the parallel run slower than serial. On a
10–12-core machine expect roughly an 8–10× speedup of the estimation stage.

Every stage accepts `--config <path>`. A fast end-to-end smoke test
(5% firm subsample, few multistart points, isolated output dirs, never
publishes) is:

```bash
for s in 01_build_gnr_panel 02_estimate_gnr 03_postprocess_gnr 04_export_rts_for_alpha_pipeline; do
  python $s.py --config config_smoke.yaml
done
```

## Files

| File | Role |
| --- | --- |
| `config.yaml` | All settings: variable mapping, trimming, optimizer, export flags. |
| `config_smoke.yaml` | Small-sample smoke-test config (isolated outputs, no publishing). |
| `utils.py` | Paths / config / logging. Self-contained — no imports from `01_industry_alpha`. |
| `gnr_model.py` | Core GNR math (polynomials, Step-1 NLS objective, nested Step-2 GMM, elasticities, Sobol starts). |
| `01_build_gnr_panel.py` | Compustat → GNR input panel (logs, shares, firm-level lags) + MATLAB trimming. |
| `02_estimate_gnr.py` | Step 1 multistart NLS; Step 2 nested multistart GMM. Saves `step1/step2_results.json`. |
| `03_postprocess_gnr.py` | Firm-year `eps, melast, kelast, lelast, omega, eta, rts (= alpha_gross_output)`. |
| `04_export_rts_for_alpha_pipeline.py` | Sector aggregation, diagnostics, publishing to the alpha pipeline. |
| `05_alpha_omega_corr.py` | Within-sector firm-level Spearman corr(α, ω) on the **raw** GNR alpha + split-sample check. **Legacy diagnostic:** the bundle now consumes the copula ρ̄ recomputed on the hybrid alpha (`01_industry_alpha` Stage S6, `06_alpha_omega_corr_hybrid.py`); this script and its `alpha_omega_corr.json` are preserved for comparison only. |

Intermediates land in `../02_intermediary/`, final outputs in `../03_outdata/`.

## Chosen v1 mapping (fixed by the spec)

| GNR field | Mapping |
| --- | --- |
| `id` | `gvkey` |
| `year` | fiscal year (curated `year`) |
| `ind` | `ind2d` |
| `r` | `log(sale_D)` |
| `k` | `log(capital_D)` |
| `m` | `log(cogs_D)` |
| `l` | `log(emp)` |
| `w` | `log(xlr_D / emp)` when available — **optional in the baseline** |
| `s` | `m − r` |
| lags | firm-level one-period lags within `gvkey`, consecutive years only |

Robustness columns carried through the panel: `m_alt1 = log(cogs_D − xlr_D)`
(when positive) and `s_alt1 = m_alt1 − r`.

## Estimator summary

- **Step 1** — constrained NLS of the materials-share equation
  `s = log(P1 @ γ) − ε` over the 10-term polynomial
  `P1 = [1, k, m, l, k², m², l², km, kl, ml]`, with feasibility constraint
  `P1 @ γ > 0`. Multistart (rescaled OLS guess, γ₀, 2γ₀, scrambled Sobol
  points filtered for feasibility) → L-BFGS-B with analytic gradient →
  Nelder-Mead polish. Then `ε̂`, `bigeps = mean(exp ε̂)`, `γ̂ = soln / bigeps`,
  `melast = P1 @ γ̂`.
- **Step 2 (nested)** — selection keeps trimmed rows with `LP1 @ soln > 0`.
  `rt = r − ε − ∫melast`; integration constant `C(k,l) = P2 @ a` with
  `P2 = [k, l, k², l², kl]`; Markov process of productivity = OLS of ω on
  `[industry dummies, 1, ω₋₁, ω₋₁², ω₋₁³]` concentrated out inside the GMM
  objective; instruments `IV = [k, k², l₋₁, l₋₁², k·l₋₁]`, `W = I`
  (just-identified, so no 2-step GMM — matches the MATLAB default).
- **Post-processing** — analytic capital/labor elasticities and
  `rts = melast + kelast + lelast`, exported as `alpha_gross_output`.
  Note: `alpha_gross_output` and the GNR `lelast` are **raw/diagnostic** — they
  are no longer model-facing. The model consumes `alpha_hybrid_sector_a`
  (`melast`/`kelast` + KLEMS-imputed labor), built downstream in
  `01_industry_alpha` Stage S4b; the contaminated GNR `lelast` never enters it.

## Deliberate divergences from the MATLAB script

1. **Wages are optional.** `est_gnr.m` drops all observations with missing
   lagged log wages before Step 2; in the curated panel that would cut the
   consecutive-year sample from ~160k to ~26k rows, and wages are not a core
   source of GNR identification. Baseline: `panel.require_wages: false`.
   Setting it to `true` reproduces the MATLAB restriction as a robustness run.
2. **Analytic elasticity derivatives.** MATLAB computes ∂/∂k by multiplying
   selected `P1`/`P2` columns and dividing by `k` (valid only when log capital
   ≠ 0). We evaluate the same derivative analytically (`gnr_model.kelast/lelast`)
   — algebraically identical, well-defined everywhere.
3. **Feasibility penalty instead of NaN.** The Step 1 objective returns a large
   finite penalty when `P1 @ γ ≤ 0` so scipy line searches backtrack; MATLAB
   returned NaN, which fmincon tolerates but scipy does not.
4. **Trimming counts.** Industry / year minimum-size filters are computed on the
   post-trim sample (the MATLAB script mixed pre-trim row filters with
   post-trim dummy-column filters, which can leave observations without an
   industry dummy — a research-code quirk we did not replicate).
5. **Sobol generation.** `scipy.stats.qmc.Sobol` (scrambled, seeded) replaces
   MATLAB's `sobolset(..., 'Skip', 1000, 'Leap', 100)`; the exact points differ
   but play the same role (a quasi-random multistart pool).
6. **Year dummies** are omitted, matching the final state of the MATLAB script
   (it constructs them, then explicitly replaces them with a constant).

## Outputs

- `02_intermediary/gnr_panel.parquet` — raw constructed panel with lags & flags.
- `02_intermediary/gnr_panel_trimmed.parquet` — final estimation sample.
- `02_intermediary/step1_results.json`, `step2_results.json` — parameters,
  diagnostics, sample counts, config snapshot, industry-dummy layout.
- `02_intermediary/firm_year_rts_raw.parquet` — all trimmed rows (Step-2
  objects NaN outside the Step-2 sample).
- `03_outdata/firm_year_rts.parquet` — final firm-year RTS panel:
  `gvkey, year, ind2d, eps, melast, kelast, lelast, omega, lomega, eta, rts,
  alpha_gross_output, in_step2_sample, rts_outlier_flag`.
- `03_outdata/naics_rts_gnr.csv` — `NAICS, RTS` sector means over the export
  window (default 2010–2019), same format as the legacy Salgado file.
- `03_outdata/rts_diagnostics.json` — counts by year/sector, RTS moments.

## Integration with `01_industry_alpha` (backward compatible)

`04_export_rts_for_alpha_pipeline.py` (with
`export.publish_firm_year_to_alpha_pipeline: true`, the default) copies
`firm_year_rts.parquet` to
`01_industry_alpha/00_indata/04_salgado_data/firm_year_rts.parquet`.

`01_industry_alpha/02_code/05_build_F_alpha.py` then **auto-detects** that file
(`F_alpha_firm_year.enabled: auto` in the alpha pipeline's `config.yaml`) and
builds the within-sector `F_i^α` from empirical firm-year RTS quantiles,
filtered to the active window. If the file is absent — or `enabled: false` —
the legacy Hubmer-prior PCHIP path runs exactly as before. The downstream
`s5_F_alpha.parquet` schema is identical in both branches.

**Safety:** the original Salgado `naics_rts.csv` (legacy fallback input) is
never overwritten by default. The GNR sector means go to `naics_rts_gnr.csv`;
set `export.overwrite_legacy_naics_rts: true` to replace the legacy file (a
timestamped backup is made first).

To revert the alpha pipeline to the legacy behavior at any time: delete
`04_salgado_data/firm_year_rts.parquet` (or set `F_alpha_firm_year.enabled:
false`) and re-run `05_build_F_alpha.py` onward.
