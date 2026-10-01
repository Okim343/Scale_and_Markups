# Empirical calibration pipeline

This directory turns raw Compustat, KLEMS, and GNR returns-to-scale data into the **natively pooled finite-market calibration bundle** read by `04_Code/steady_state`.

The structural model has **no sector dimension**. NAICS2, represented in the pipeline by the derived `sector_id` calibration-market labels, is data used to measure within-market concentration and the retained `emx_slope` regression. It is never a structural model index. The current mapping combines manufacturing as `31-33`, retail as `44-45`, and transportation as `48-49`, leaving 14 empirical market cells in the active sample.

```text
03_Empirical/
├── 00_Hubmer_RTS/
│   ├── 01_code/          # pooled GNR estimation and alpha-omega dependence
│   └── 03_outdata/       # firm_year_rts.parquet, alpha_omega_corr.json (legacy diagnostic)
└── 01_industry_alpha/
    ├── 00_indata/        # raw/published inputs
    ├── 01_intermediary/  # restartable S-stage outputs
    ├── 02_code/          # bundle assembly pipeline
    └── 03_outdata/       # model-facing bundle and reference outputs

04_Code/steady_state/
├── io_bundle.py          # validates and loads the three-file contract
├── pooled_inputs.py      # copies pooled values into structural inputs
└── config.yaml           # paths.bundle_dir -> empirical 03_outdata/
```

## Two sub-pipelines

### `00_Hubmer_RTS`: estimate firm-level alpha and alpha-omega dependence

This pipeline estimates a pooled Gandhi-Navarro-Rivers production function from the curated Compustat panel. In order, its stages build and trim the GNR panel, estimate the materials-share NLS and nested GMM, reconstruct firm-year elasticities and productivity, publish the firm-year RTS panel, and estimate within-sector alpha-omega rank dependence.

Its outputs used downstream are:

- `03_outdata/firm_year_rts.parquet`: firm-year gross-output elasticities (`melast`, `kelast`, `lelast`, `omega`). Its `alpha_gross_output = rts = melast + kelast + lelast` and the raw GNR `lelast` are **diagnostic only** — the GNR labor elasticity is Compustat/headcount-contaminated. The model-facing firm-level scalability object is now `alpha_hybrid_sector_a`, built in `01_industry_alpha` Stage S4b from `melast`/`kelast` and the KLEMS sector capital share (it never uses the raw GNR `lelast`).
- `03_outdata/alpha_omega_corr.json`: pooled within-sector Spearman on the **raw** GNR alpha. **Legacy diagnostic** — the bundle no longer reads this file at all. The copula correlation the bundle actually ships (`alpha_z_copula_rho_bar`) is recomputed on the hybrid alpha by `01_industry_alpha` Stage S6 (`alpha_omega_corr_hybrid.json`), which is a different, later file. `alpha_omega_corr.json` is therefore **not** a required input to the bundle pipeline (see "Required raw inputs" below) — it is kept only as a standalone GNR-side diagnostic.

Stage R4 publishes `firm_year_rts.parquet` to `01_industry_alpha/00_indata/04_salgado_data/`, where S4b reads it. These are **inputs to** the bundle pipeline, not bundle products. Do not rerun GNR merely to rebuild the bundle; rerun it only when the curated GNR input, GNR specification, or correlation settings change.

To regenerate them from the repository root:

```bash
(cd 03_Empirical/00_Hubmer_RTS/01_code && python 01_build_gnr_panel.py)
(cd 03_Empirical/00_Hubmer_RTS/01_code && python 02_estimate_gnr.py)
(cd 03_Empirical/00_Hubmer_RTS/01_code && python 03_postprocess_gnr.py)
(cd 03_Empirical/00_Hubmer_RTS/01_code && python 04_export_rts_for_alpha_pipeline.py)
(cd 03_Empirical/00_Hubmer_RTS/01_code && python 05_alpha_omega_corr.py)
```

All five scripts accept an optional `--config config_smoke.yaml` (or any other config path) through their shared parser.

### `01_industry_alpha`: assemble the pooled bundle

The required DAG and run order is:

```text
S1 -> S2 -> S4 -> S4b -> S5 -> S6 -> S6b -> S7 -> S8b -> S8
```

S6 and S6b can run any time after S4b (they read the same S4b hybrid-alpha panel independently of S7/S8b); both must precede S8. S8b **must precede S8**. S8 has a checked-in fallback `-0.5096` when the S8b YAML is absent, but a reproducible full run should produce and consume S8b's current value instead.

| Stage | Script | Reads -> writes | Computation |
|---|---|---|---|
| S1 | `01_load_compustat.py` | `00_indata/03_Compustat/markup_firm_year.dta` -> `01_intermediary/s1_compustat_firmyear.parquet` | Cleans firm-years, applies exclusions and markup trimming, and maps NAICS2 to calibration `sector_id`. |
| S2 | `02_pooled_targets.py` | S1 parquet -> `s2_firm_year_shares.parquet`, `s2_sector_year_compustat.parquet`, `s2_pooled_targets.json` | Computes within-`(year, sector_id)` shares, markups, and concentration, then pools concentration targets across empirical markets. |
| S4 | `04_klems_capital_share.py` | KLEMS CSV plus S1 retained markets -> `s4_klems_sector_year.parquet` | Computes clipped `a_iy = 1 - LS_VA` and KLEMS value-added/gross-output shares. |
| S4b | `04b_build_hybrid_alpha.py` | published GNR `firm_year_rts.parquet`, S1 sectors, S4 KLEMS -> `s4b_firm_year_alpha_hybrid.parquet` | Builds the model-facing hybrid alpha `alpha_hybrid_sector_a = melast + kelast + kelast*(1 - a_sector)/a_sector`, imputing labor from the KLEMS sector capital share. Raw GNR `lelast`/`alpha_gross_output` are kept only as diagnostic columns. |
| S5 | `05_build_F_alpha.py` | S1 retained markets plus S4b hybrid panel -> `s5_F_alpha.parquet` | Builds one pooled, equal-mass `F(alpha)` over `alpha_hybrid_sector_a`: winsorize → mean-preserving RTS shrink (`support_variant`/`shrink_lambda`) → clip to `[0.6, 1.20]`. The shipped `alpha` column IS the final model-facing support (sampled verbatim; no model-side transform). |
| S6 | `06_alpha_omega_corr_hybrid.py` | S4b hybrid panel -> `03_outdata/alpha_omega_corr_hybrid.json`, `naics_alpha_omega_corr_hybrid.csv` | Recomputes the within-sector alpha-omega rank dependence on the hybrid alpha; supplies `alpha_z_copula_rho_bar` — the alpha-*productivity* copula strength — that the bundle exports. |
| S6b | `06b_alpha_sales_corr_hybrid.py` | S4b hybrid panel plus S1/S2 sector data -> `03_outdata/alpha_sales_corr_hybrid.json`, `naics_alpha_sales_corr_hybrid.csv` | Computes the pooled Pearson correlation between hybrid alpha and log(real sales) — `alpha_sales_corr_empirical` — and its partial version controlling for `omega` — `alpha_sales_corr_partial_empirical`. This is a **distinct object from `alpha_z_copula_rho_bar`** (alpha-*sales*, not alpha-productivity) and is a hard-required bundle scalar consumed by `08_assemble_bundle.py`; skipping this stage makes `08_assemble_bundle.py` fail. |
| S7 | `07_aggregate_moments.py` | `markup_cost_weighted.dta` plus S2 firm shares -> `s7_year_aggregates.parquet`, `s7_slope_regression.json` | Computes pooled aggregate markup and the firm-level markup-share regression used for validation. |
| S8b | `08b_emx_slope_moment.py` | both S2 parquets -> `03_outdata/emx_slope_moment.yaml`, `emx_slope_robustness.csv` | Regresses long differences in sector inverse cost-weighted markup on long differences in HHI; current primary `emx_slope = -0.5096321514062336`. |
| S8 | `08_assemble_bundle.py` | S2/S4/S4b/S5/S7, S8b YAML, and the S6/S6b hybrid JSONs -> bundle files, exposure reference, manifest | Assembles pooled scalars (incl. hybrid `phi_v`, the `hybrid_alpha` block, and the S6b sales-correlation scalars), copies `F(alpha)`, removes stale `sector_table.parquet`, and records provenance and hashes. |

Copy-paste run block, from the repository root:

```bash
(cd 03_Empirical/01_industry_alpha/02_code && python 01_load_compustat.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 02_pooled_targets.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 04_klems_capital_share.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 04b_build_hybrid_alpha.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 05_build_F_alpha.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 06_alpha_omega_corr_hybrid.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 06b_alpha_sales_corr_hybrid.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 07_aggregate_moments.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 08b_emx_slope_moment.py)
(cd 03_Empirical/01_industry_alpha/02_code && python 08_assemble_bundle.py)
```

The scripts, current `config.yaml`, emitted YAML, and structural loader are authoritative; there is no separate pipeline-plan design doc in this directory anymore.

## Bundle contract

`04_Code/steady_state/config.yaml` resolves `paths.bundle_dir` to `03_Empirical/01_industry_alpha/03_outdata`. `io_bundle.load_bundle()` requires these three files:

### `F_alpha.parquet`

One pooled equal-mass `F(alpha)`, with 500 rows and no `sector_id`. The file currently contains:

- `node_id`, `u`, `weight`, `alpha`, `alpha_raw`, `alpha_shrunk`, `alpha_source`: support identity, midpoint rank, mass, final/pre-shrink alpha, and provenance. `alpha_raw` = post-winsor pre-shrink node; `alpha_shrunk` = post-shrink pre-clip; `alpha` = final model-facing support.
- `alpha_clipped_flag`: producer diagnostic indicating clipping (of the shrunk value).

The loader (`io_bundle.F_ALPHA_REQUIRED_COLS`) only *requires* `node_id`, `u`, `weight`, and `alpha`, and rejects a `sector_id` column. Current weights are uniform `0.002`; the final support is winsorize → shrink → clip to `[0.6, 1.20]` (18 nodes clipped, all at the lower bound; the upper bound is never hit); `alpha_source` is `alpha_hybrid_sector_a_pooled_empirical`.

### `aggregate_moments.yaml`

`io_bundle.py` enforces three required top-level keys — `active_window`, `targets`, `validation` — plus a fourth-required-scalars tuple `REQUIRED_SCALAR_KEYS = ("a", "n_firms_cs", "alpha_z_copula_rho_bar", "alpha_sales_corr_empirical")`. Required target keys (`REQUIRED_TARGET_KEYS`) are exactly six: `mu_cw, cr4, cr20, top1pct, top5pct, emx_slope` — **not** seven, and `mean_active_count` is **not** one of them (see "mean_active_count" below). A seventh target key, `mu_cw_sga`, is required *conditionally*: `Bundle.target_mu_cw_sga` raises if it's absent, and it's the one the model's `mu_cw` moment actually targets by default (`calibration.markup_target: sga` in `04_Code/steady_state/config.yaml`).

**Naming trap.** The on-disk data key `mu_cw` is the accounting markup (aggregate revenue / variable cost). The model calls that object `mu_cw_alpha` and reserves the unqualified name `mu_cw` for the *pure* cost-weighted markup (`Σ TC·μ / Σ TC`), which is the actual calibration target — remapped from the data's `mu_cw_sga` (SG&A-inclusive) by default. `Bundle.targets`/`Bundle.calibration_targets()` in `io_bundle.py` do this remap at the loader boundary; don't read the data key `mu_cw` as the model's calibration target.

Current values (`03_Empirical/01_industry_alpha/03_outdata/aggregate_moments.yaml`):

```yaml
targets:
  mu_cw: 1.439939260482788          # data key = accounting markup; model calls this mu_cw_alpha
  mu_cw_sga: 1.1847574710845947     # SG&A-inclusive; THIS is what the model's mu_cw is fit to
  cr4: 0.2034884378694585
  cr20: 0.4834767422477661
  top1pct: 0.3238652553492832
  top5pct: 0.6352322028452946
  emx_slope: -0.5096321514062336
a: 0.3945908870695359
phi_v: 0.3482021401114195
n_firms_cs: 5211.4
n_markets: 14
mean_firms_per_sector: 1305.3552240307745   # diagnostic analogue of exogenous-entry N, not a target
alpha_z_copula_rho_bar: 0.22039325927466766          # alpha-productivity copula strength (warm start)
alpha_sales_corr_empirical: 0.5956851694283806       # alpha-log(sales) correlation (Stage S6b)
alpha_sales_corr_partial_empirical: 0.569949216826769
```

The file also carries a `hybrid_alpha` block (shrink/clip provenance, `phi_v` derivation), a `gnr_elasticities` block (raw-GNR diagnostics only), a `reference_scale` block (retired `x_ref`/`y_ref` operating-scale diagnostics — not consumed by the model; see `02_Drafts/model.typ`'s production function, which no longer has an `x_ref` term at all), a `sensitivity_panel` (primary/robust_a/robust_b windows), and a `provenance` block documenting each scalar's source stage. These are metadata for auditability, not part of the required contract.

`validation.*`: the firm-level log-log markup-share slope is `0.025284185238003257` with HC1 SE `0.0011550164574471054`, `52,114` observations, and R-squared `0.06643365669904244`.

**`mean_active_count` is not a target — it is an EMX-style diagnostic.** The bundle's own `notes:` field states this explicitly: *"The calibration target set is mu_cw, cr4, cr20, top1pct, top5pct, and emx_slope. mean_active_count is intentionally absent: under the EMX-style exogenous-entry interpretation the model reports the realized sector firm count as a diagnostic and calibrates N directly from the concentration moments."* There is no `flags:` key in this file (an older version of this README quoted one; it never made it into the code that assembles the bundle — `08_assemble_bundle.py` has no code path that writes a `flags` block). If you need the old assigned-vs-derived framing, it's superseded: `N` (Poisson mean firms per sector) is now a directly calibrated structural parameter in `04_Code`, not an assigned constant.

**Why `emx_slope` regresses inverse markup (not labor share) on HHI.** EMX's Cournot replication regresses the sectoral *labor share* on HHI (`-bhat(2)=0.213`); we regress the sectoral *inverse cost-weighted markup* directly. Both identify the same structural slope `b = -(1/eta - 1/gamma)` — labor share is proportional to `1/mu` in this model class, so the two differ only by a labor-share loading. We use the markup because (i) Compustat labor cost (`xlr`) is reported for only ~20% of firm-years and skews large, biasing a labor-share measure, while the markup has full coverage; and (ii) it identifies `-(1/eta - 1/gamma)` without the loading wedge. The levels-vs-long-difference choice is immaterial here (~`-0.57` levels vs `-0.51` long-diff). See `02_Drafts/md_files/EMX_breakdown.md` (§"Why we regress inverse markup…") and `model.typ` §5 for the full justification.

### `manifest.yaml`

Provenance for `schema: pooled-bundle-v1`: pipeline version, active window, exclusions, UTC run timestamp, file sizes, and SHA-256 hashes for raw inputs (4 files — see "Required raw inputs" below), intermediaries, and outputs (11 files, incl. the S6b JSON/CSV). It also records the `alpha_z_copula` source/diagnostics block (`source`, `key`, `rho_bar`, `pooled_within_spearman`, `n_firm_units`).

`naics2_exposure_reference.csv` is **reference-only** for a deferred non-uniform exposure extension. The model does not read it; baseline exposure is uniform `1/Q`.

There is intentionally **no `sector_table.parquet`**. The pooled refactor removed that artifact and deleted the structural loader's re-pooling shim. S8 removes a stale copy, and the model loader consumes pooled values directly.

## Design choices that are part of the contract

- **Concentration:** compute measures within each `(year, sector_id)`, average each empirical market over 2010-2019, then take the `n_firms_cs`-weighted mean across markets. `n_firms_cs = 5211.4` is the sum of those window-average market firm counts.
- **Capital share:** `a` is the value-added-weighted KLEMS aggregate, `sum_s VA_s * a_s / sum_s VA_s`, using active-window means across 13 retained KLEMS groups. It is not firm-count weighted. This matches `model.typ` Panel A, "KLEMS aggregate, fallback 1/3."
- **Pooled `F(alpha)`:** filter the S4b hybrid firm-years to the retained calibration markets and 2010-2019, collapse to one window-mean `alpha_hybrid_sector_a` per firm, winsorize the pooled firm distribution at `[0.01, 0.99]`, form equal-mass quantile-midpoint nodes, apply the mean-preserving RTS shrink, then clip to `[0.6, 1.20]`. The baseline shrink is `lambda = 0.371`, a compromise discipline from Salgado et al. that materially compresses the hybrid support's dispersion while preserving the Table I upper-tail feature that some firms have RTS above one (P99 about `1.08`). The lower clip at `0.60` is an admissible numerical/economic floor, justified by Salgado's lowest reported industry-average RTS `0.59` (Healthcare); it is not a target for Salgado's firm-level lower tail. The shrink is applied **before** the clip (the single standard) and the shipped `alpha` is the final support the model samples — there is no model-side shrink. The clip matches the support expected by `steady_state/model/pool.py`; do not restore the legacy sector-specific `[0.5, 1.3]` range.
- **Active window:** 2010-2019, the recent pre-COVID decade. S1 retains years from 1997 onward so alternative windows can be run without rebuilding S1.
- **Exclusions:** `49, 52, 53, 62, 81, 92, 99`. The config labels these as transportation support, finance, real estate, healthcare, other services, public administration, and unclassified. It explicitly records the additional reason for 49: `mu_cw_49 < 1` is inconsistent with Cournot and the public-firm sample has only 8-11 firms in 2010-2019.
- **Alpha-productivity dependence (`alpha_z_copula_rho_bar`):** the bundle exports `alpha_z_copula_rho_bar = 0.22039325927466766` (Stage S6, recomputed on the hybrid alpha; the *pooled within-sector Spearman* between alpha and omega, mapped to an implied Gaussian-copula correlation). **This value is a warm start, not a fixed external input.** `04_Code/steady_state/pooled_inputs.py` copies it in as `PooledInputs.rho_bar`, and `04_Code`'s joint calibration (`calibrate_pooled`, fitting `(xi, N, gamma, eta, rho_bar)`) treats it as the *initial* value for a fifth calibrated parameter — the current committed fit lands at `rho_bar ≈ 0.897` (`04_Code/out_results/calib_pooled/calibration_pooled.yaml`), well away from the bundle's 0.220 warm start. The bundle's own `config.yaml`/`manifest.yaml` comments still describe `rho_bar` as "fixed, ex-ante, not calibrated" and cite a `rho_z = sigma_z*rho_bar/sqrt(1-rho_bar^2)` loading formula — that description and formula predate `04_Code`'s current draw rule and calibration and are stale; do not treat them as authoritative for what the solver does today. The actual current draw rule (`steady_state/model/pool.py`) has no `sigma_z`/`log z` object at all: it draws a Pareto(`xi`)-marginal capability `v` directly, `v = v_min*(1-u)^(-1/xi)` with `u = Phi(rho_bar*tilde_alpha + sqrt(1-rho_bar^2)*eps)`.
- **Alpha-sales correlation (`alpha_sales_corr_empirical`, Stage S6b):** a *separate* empirical moment from `alpha_z_copula_rho_bar` above — the pooled Pearson correlation between hybrid alpha and log(real sales) in the same panel, `0.5956851694283806`, with a partial version controlling for `omega`, `0.569949216826769`. This is the valid empirical target for the model's simulated `corr_alpha_log_sales` diagnostic; `alpha_z_copula_rho_bar` never was (it targets alpha-*productivity* dependence, not alpha-*sales* dependence). Do not conflate the two.

## Required raw inputs

Required main-path raw/external inputs (S8's manifest hashes exactly these four — `alpha_omega_corr.json` is not among them, see the `00_Hubmer_RTS` section above):

```text
01_industry_alpha/00_indata/03_Compustat/markup_firm_year.dta
01_industry_alpha/00_indata/03_Compustat/markup_cost_weighted.dta
01_industry_alpha/00_indata/klems_labor_share_naics2_with_VA_1997_2023.csv
01_industry_alpha/00_indata/04_salgado_data/firm_year_rts.parquet
```

The first Compustat file is also the GNR raw input specified by `00_Hubmer_RTS/01_code/config.yaml`.

## Reproducibility and gotchas

- The cleaning, aggregation, quantile construction, regressions, and assembly are deterministic for fixed raw files, config, and library behavior. GNR multistarts and model Monte Carlo use fixed seeds from their configs.
- S8 writes a fresh UTC `run_timestamp_utc`, so `manifest.yaml` changes on every assembly even when numerical artifacts do not. It hashes existing raw inputs, `s*` intermediaries, and files present in `03_outdata`; use those SHA-256 records to identify changed inputs or outputs.
- A bundle rebuild starts at S1 and uses the already-published GNR files. Rerun `00_Hubmer_RTS` only when its raw input or estimator configuration changes.
- S8b has hard-coded primary/extended/full windows; changing `config.active_window` alone does not change its primary 2010-2019 moment.
- S8 removes any stale `03_outdata/sector_table.parquet`.
- Skipping S6b before S8 makes `08_assemble_bundle.py` fail (it hard-requires `alpha_sales_corr_hybrid.json`).

## End-to-end quickstart

From the repository root, regenerate the bundle and run a fast structural smoke pass:

```bash
(cd 03_Empirical/01_industry_alpha/02_code && \
  python 01_load_compustat.py && \
  python 02_pooled_targets.py && \
  python 04_klems_capital_share.py && \
  python 04b_build_hybrid_alpha.py && \
  python 05_build_F_alpha.py && \
  python 06_alpha_omega_corr_hybrid.py && \
  python 06b_alpha_sales_corr_hybrid.py && \
  python 07_aggregate_moments.py && \
  python 08b_emx_slope_moment.py && \
  python 08_assemble_bundle.py)

(cd 04_Code && python -m steady_state calibrate --M 200 --max-nfev 2)
(cd 04_Code && python -m steady_state simulate --M 200)
(cd 04_Code && python -m steady_state welfare --M 200)
```

`simulate` and `welfare` automatically use `04_Code/out_results/calibration_pooled.yaml` when calibration has produced it (falling back to `04_Code/out_results/calib_pooled/calibration_pooled.yaml` as a committed warm start otherwise). See `04_Code/solver_scaffold.md` for what the structural side of this pipeline actually does with these inputs.
