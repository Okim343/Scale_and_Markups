# 03_Empirical: from Orbis Spain to the calibration bundle

This folder turns Spanish Orbis firm accounts into the pooled calibration bundle read by `04_Estimation` and `05_Simulation`. The bundle holds the scalability distribution $F^\alpha$, the model inputs $(a, \phi_v)$ and every calibration target.

```text
03_Empirical/
├── 00_GNR_core/          # shared GNR estimator (Python port of the Hubmer et al. MATLAB code)
│   └── 00_original_code/ # MATLAB originals, for reference
├── 02_Orbis/             # current pipeline (WRDS-licensed data, all gitignored)
│   ├── 01_Cleaning/      # WRDS extraction, deflators, cleaning to a GNR-ready panel
│   ├── 02_GNR/           # GNR per NAICS2 cell -> firm-year returns to scale
│   ├── 03_Markups/       # DLEU production-function markups
│   ├── 04_Moments/       # F(alpha), targets and the bundle
│   └── 06_Age_Size/      # side analysis of costs and labour by firm age and size (not in the paper)
└── 01_Compustat/         # earlier Compustat pipeline, reference only (see 01_Compustat/README.md)
```

Each stage folder follows the same convention. `01_code/` is versioned, while `00_indata/`, `01_intermediary/` and `03_outdata/` are gitignored and created by the scripts. Paths in the configs are relative to the repository root. Every output folder carries a `manifest.yaml` with the commit, the dirty flag, the config sha256 and the fingerprints of the inputs, so each stage can check that its inputs are the ones the next stage was built on.

## Sample

Spanish unconsolidated accounts (U1, U2) of industrial companies, all size tiers, extracted for closing years 2014 to 2024. Money variables are deflated to 2019 euros with the Spanish HICP. The calibration window is 2015 to 2019. NAICS2 codes 52, 53, 62 and 81 are excluded following Hubmer et al., 92 and 99 as out of scope and 55 as holding companies. Manufacturing (31 to 33), retail (44 to 45) and transportation (48 to 49) are grouped, which leaves 14 NAICS2 cells. The markup sample has 2,000,342 firm-years from 568,920 firms, and the GNR sample behind $F^\alpha$ has 1,391,061 firm-years.

## Run order

The scripts that query WRDS take credentials from the `wrds` package (`~/.pgpass` and the `WRDS_USERNAME` environment variable, or an interactive prompt). Nothing is stored in the repository.

### 01_Cleaning (from the repository root)

```bash
python 03_Empirical/02_Orbis/01_Cleaning/01_code/00_check_access.py         # optional access check
python 03_Empirical/02_Orbis/01_Cleaning/01_code/00b_check_spain_counts.py  # optional size check
python 03_Empirical/02_Orbis/01_Cleaning/01_code/01_extract_orbis_es.py     # WRDS pull, restartable
python 03_Empirical/02_Orbis/01_Cleaning/01_code/01b_fetch_deflators_es.py  # Eurostat, public
python 03_Empirical/02_Orbis/01_Cleaning/01_code/02_clean_orbis_es.py
```

Writes `01_Cleaning/01_intermediary/orbis_es_clean_panel.parquet` and the GNR input `gnr_ready_tfas.parquet`, plus `cleaning_report.md`.

### 02_GNR (from `02_Orbis/02_GNR/01_code/`)

```bash
python 01_run_gnr_cell.py --cell 21 --workers 16    # once per cell in config cells.list
python 03_seed_check_cell.py --cell 21 --seeds 1982 7 123 9999   # optional, multistart stability
python 02_combine_cells.py
```

GNR (Step 1 NLS on the materials share, Step 2 nested GMM) is estimated separately for each NAICS2 cell, as in Hubmer et al., with the estimator in `00_GNR_core`. Step 2 is exactly identified and its root is selected by the group of implied elasticities that most starts reach. `01_code/run_all_cells.slurm` runs every cell, the seed checks and the combine step as one cluster job. Writes `02_GNR/03_outdata/firm_year_rts.parquet` (`melast`, `lelast`, `kelast`, `rts`) and `gnr_cells_report.md`.

### 03_Markups (from `02_Orbis/03_Markups/01_code/`)

```bash
python 01_fetch_rates.py        # ECB overnight rate (EONIA basis), public
python 02_build_sample.py
python 03_estimate_thetas.py    # rolling five-year IV elasticities by cell
python 04_build_markups.py
python 05_stata_crosscheck.py prepare   # optional, with check_stata.do, then `compare`
python 06_plot_aggregate_markup.py      # optional figure
```

Markups follow De Loecker, Eeckhout and Unger (2020). The variable input is $V$ = materials + staff costs, and the firm-year markup is the cell-year output elasticity of $V$ times sales over $V$. Writes `03_Markups/03_outdata/firm_year_markups.parquet`.

### 04_Moments (from `02_Orbis/04_Moments/01_code/`)

| Step | Script | Produces |
|---|---|---|
| 01 | `01_build_F_alpha.py` | pooled $F^\alpha$, firm window means of GNR `rts`, winsorized at 1/99, 500 equal-mass nodes |
| 02 | `02_build_market_panel.py` | per (year, market) firm counts, concentration, HHI and markups |
| 03 | `03_pooled_targets.py` | pooled concentration and firm counts (reported, not targeted in the paper) |
| 04 | `04_aggregate_markup.py` | `mu_cw`, the (V + capital user cost)-weighted markup, mean over 2015 to 2019 (1.1815) |
| 05 | `05_emx_slope.py` | EMX markup-HHI slope (reported, not targeted in the paper) |
| 06 | `06_alpha_moments.py` | RTS gap top 5% minus bottom 50% within NAICS4 (0.046, the $\bar\rho$ target) and $\phi_v$ (0.517) |
| 07 | `07_assemble_bundle.py` | `F_alpha.parquet`, `aggregate_moments.yaml`, `manifest.yaml`, checked with the model and calibration readers |
| 08 | `08_report.py` | `moments_report.md` |
| 09 | `09_crosscheck_compustat.py` | optional, runs the step 02/03/05 functions on the Compustat file and checks them against the published values |
| 10 | `10_fig_rts_by_size.py` | Figure 1 of the paper, `02_Drafts/figures/orbis/fig_rts_by_size_orbis.pdf` |
| 11 | `11_fig_F_alpha.py` | figure of $F^\alpha$ and its truncation (needs the calibrated $\gamma$) |
| 12 | `12_zipf_targets.py` | `zipf_targets.yaml`, Zipf tail indices of firm sales at the top 5% (1.1079) and top 2% (1.1451) |

Every script takes `--config` (default `config_moments_orbis_es.yaml`).

## Bundle contract

`02_Orbis/04_Moments/03_outdata/` is the only empirical input of the model. Both `04_Estimation` and `05_Simulation` point `paths.bundle_dir` at it, and the parameter file records its sha256.

- `F_alpha.parquet` has columns `node_id, u, weight, alpha` and no `sector_id`. The `alpha` column is raw GNR returns to scale (no clip or shrink). The model truncates the draws at $0.99\,\gamma/(\gamma-1)$ at draw time.
- `aggregate_moments.yaml` holds `active_window`, `targets` (`mu_cw`, `cr4`, `cr20`, `top1pct`, `top5pct`, `emx_slope`), `a` (1/3, from EMX), `phi_v`, `n_firms_cs`, `mean_firms_per_sector` (diagnostic only) and `alpha_rts_gap_top5_bottom50`.
- `zipf_targets.yaml` holds the two Zipf tail indices targeted by the market-free calibration, plus untargeted data counterparts for `04_Estimation/02_validate_zipf.py`. It is written separately, so it does not change the bundle hash.
- `manifest.yaml` records provenance, definitions, sample sizes and diagnostics.

The paper targets `mu_cw`, the two Zipf indices and the RTS gap. Concentration and the EMX slope stay in the bundle as reported moments, since Orbis concentration depends on sample coverage and market definition.

## 06_Age_Size

A self-contained side analysis of cost shares and the labour share by firm age and size, with a benchmark against Karabarbounis and Neiman (2014). It reads the clean panel and never writes to the main pipeline. `01_code/04_build_html.py` renders its summaries as HTML pages in `06_Age_Size/`, which are kept local.

## Data availability

Orbis is licensed through WRDS and cannot be redistributed, so no raw, intermediate or output file of this folder is versioned (`.gitignore` blocks every `00_indata/`, `01_intermediary/` and `03_outdata/` folder, and every parquet and dta file). Deflators, overnight rates and the national-accounts benchmark are public and fetched by the scripts.
