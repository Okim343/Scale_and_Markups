# Orbis Spain markups (DLEU method): plan, decisions and approach

Companion to `ORBIS_SPAIN_CLEANING.md`. This stage builds firm-year markups for Orbis Spain following
De Loecker, Eeckhout and Unger (2020, QJE; "DLEU") as closely as the data allow. Status: **coded and run** (2026-10-12); results in Section 9. Results and the commit hash will be added below once the code runs.

References:
- DLEU replication code (Compustat, Stata), read-only and gitignored:
  `03_Empirical/02_Orbis/03_Markups/00_DLEU_Code/` (`01_Create_Data.do`, `02_Produce_Thetas_st.do`,
  `03_Calibration_Moments.do`).
- Paper: `01_Docs/risemkt_power.pdf` (Sections II.B-C, III.A-B, VI.A-B, Appendices A and B).
- Worldscope comparison: De Loecker and Eeckhout, "Global Market Power" (10 Feb 2021 draft),
  `01_Docs/Global_Markups.pdf`, Table 1.

## 1. What the DLEU code does (facts the plan rests on)

1. **Sample.** Compustat; the last data year is dropped as incomplete (`keep if year <= max - 1`); firms
   without NAICS dropped; negative sale/cogs/xsga dropped; `s_g = sale/cogs > 0`.
2. **Deflation.** One deflator for every money variable (US GDP deflator, 2015 = 100), so ratios stay
   nominal. The paper's text mentions PIRIC for capital; the code uses the GDP deflator.
3. **User cost.** `usercost = (FFR - GDP-deflator inflation + 12) / 100`, annual means;
   `kexp = usercost * capital_D` with `capital_D = ppegt / deflator`.
4. **Trim 1** (`01_Create_Data.do`): keep `s_g` **strictly** inside its (p1, p99), percentiles **by year**,
   pooled over sectors.
5. **Trim 2** (`02_Produce_Thetas_st.do` and again in `03_Calibration_Moments.do`):
   `costshare1 = cogs/(cogs + kexp)`, `costshare2 = cogs/(cogs + xsga + kexp)`; for each in turn (cs1, then
   cs2 on the survivors): drop if 0 or missing, keep inside **[p1, p99]** by year. The paper's text also
   mentions an SG&A/sales trim; the code does not apply it.
6. **Lags after trimming.** The panel is `xtset` after the trims, so a trimmed predecessor year removes the
   lag. Lags require consecutive years.
7. **Estimator (baseline, `theta_WI1_ct`).** By NAICS2 cell (48 and 49 merged), on a 5-year window centred
   on t (t-2..t+2, truncated at the data edges):
   `ivregress gmm r (c = L.c) k L.i L.k2 L.k`, with `r = ln(sale_D)`, `c = ln(cogs_D)`, `k = ln(capital_D)`,
   `L.k2 = (L.k)^2`, `i = ln(K - 0.9 L.K)` (missing when investment <= 0). Just-identified, so the GMM
   point estimate equals 2SLS. No year dummies, no constant restrictions, **no guardrails** on the estimate.
   The capital coefficient is stored as `theta_WI1_kt`.
8. **Edge years.** Years before the first window centre (1970) get one pooled estimate on all years
   before 1972; sector 61 gets a longer early period because it is thin.
9. **Markup** (`03_Calibration_Moments.do`): `markup = theta_WI1_ct * sale_D / cogs_D`, on the trimmed
   sample (every firm-year with a theta, not only the estimation rows). Weights: sale, cogs, cogs+xsga,
   cogs+kexp, cogs+kexp+xsga.
10. **Not in the code, only in the paper.** The cost-share elasticity (Section VI.A): theta_st = median over
    firms in sector s, year t of `V / (V + rK)`.
11. **ACF** (`02_Produce_Thetas_st.do`, Section 03): Cobb-Douglas per sector, constant over time, with
    guardrails (theta_c > 1 or theta_k outside [0, 1] replaced by the cross-sector mean). It feeds only
    `theta_unified_const` (a constant gamma = theta_c + theta_k for their IRS model). It is **not** used in
    the markup.
12. **Spain.** The QJE paper reports US numbers only. Spanish markups are in De Loecker and Eeckhout (2018),
    "Global Market Power" (Worldscope): Spain 1.34 in 2016, +0.24 since 1980 (Table 1). That paper
    applies the **US** DLEU elasticities (NAICS2 x year) to COGS, sales weights, all sectors, mostly
    listed firms with consolidated statements. Used only as a loose level check (D6c).

## 2. Mapping DLEU to Orbis

| Step | DLEU (Compustat) | Orbis Spain |
|---|---|---|
| Input | Compustat, all years | `01_Cleaning/01_intermediary/orbis_es_clean_panel.parquet` |
| Years | drop last (incomplete) year | 2014 lag-only, markups 2015-2023, 2024 dropped |
| Sectors | all NAICS2 with a code | the 14 GNR cells: 11 21 22 23 31 42 44 48 51 54 56 61 71 72 |
| Output (sale) | `sale` | `rev_D` (turnover, else operating revenue) |
| Variable input (cogs) | `cogs` | **`V = mate + staf`** (`V_D = mate_D + staf_D`) |
| Overhead (xsga) | `xsga` | not available; every xsga step is skipped |
| Capital | `ppegt` (gross PP&E) | `tfas_D` (net book tangible fixed assets), `pos_tfas == 1` |
| Deflator | US GDP deflator | HICP (`_D`, 2019 = 1), one deflator for all money |
| User cost | FFR - US GDP-deflator inflation + 12% | euro overnight rate - Spanish GDP-deflator inflation + 12% |
| Trim 1 | `sale/cogs` strictly in (p1, p99) by year | `rev/V` strictly in (p1, p99) by year |
| Trim 2 | cs1 then cs2 in [p1, p99] by year | cs1 = `V/(V + kexp)` in [p1, p99] by year; cs2 skipped |
| Elasticity | WI-IV by NAICS2, 5-year window | same estimator by `ind2d` cell, windows below |
| Edge years | pooled early estimate; sector 61 special | 2015 gets the 2016-window theta (flagged) |
| Markup | `theta * sale/cogs` | `theta * rev_D / V_D` |

## 3. Decisions

Who: **U** = user, **C** = Claude (proposed, accepted by the user unless marked otherwise).

| # | Decision | Choice | Who |
|---|---|---|---|
| D1 | Variable input | `V = mate + staf` (COGS analogue). No mate-only run. No PF2/CS2 (overhead) analogue. | U |
| D2a | Elasticity | WI-IV `r (c = L.c) k L.i L.k2 L.k` by `ind2d`, as in the code; no guardrails | U (C rec.) |
| D2b | Windows | centres 2016-2023, truncated: 2016: 2016-18; 2017: 2016-19; 2018-2021: full; 2022: 2020-23; 2023: 2021-23 | U (C rec.) |
| D2c | 2015 | theta of the 2016 window, `theta_imputed = True` | U (C rec.) |
| D2d | Cost-share theta | built as an alternative column: median of `V/(V + kexp)` by `ind2d` x year (paper VI.A) | U (C rec.) |
| D2e | ACF | **not built** (see Section 4) | C, answering U's question; U may override |
| D3 | User cost | `(r_t - pi_t + 12) / 100`; r_t = EONIA-basis overnight rate, one consistent series with no break: published EONIA annual mean of daily values 2014-2021, and €STR + 8.5 bp for 2022-2023 (EONIA's own definition from Oct 2019 to its end on 3 Jan 2022); ECB Data Portal, public API, fetch approved. pi_t = growth of Spanish GDP deflator (`gdp_defl` in `01_Cleaning/00_indata/deflators/deflators_es.csv`, already on disk, from 2010). 12% = depreciation + risk premium as DLEU. `kexp = usercost * tfas_D` | U (C proposal, revised 2026-10-12) |
| D4 | Sample | 14 GNR cells; `tfas` capital; 2014 lag-only, 2015-2023; `_D` HICP. No all-sector, fias or `_Ds` runs | U (C rec.) |
| D5 | Trimming | by year, pooled over cells, exactly as the code: strict (p1, p99) on rev/V; drop cs1 = 0 or missing, keep [p1, p99]; lags built after trimming; Stata `egen pctile` percentile definition; no extra (mate+staf)/rev cap | U |
| D6a | Investment proxy | faithful: `i = ln(K - 0.9 L.K)`, rows with K - 0.9 L.K <= 0 lose `i` (out of the IV sample, kept for markups); the drop is reported | U |
| D6b | Verification | numpy 2SLS checked against statsmodels IV2SLS and a hand-coded solve; Stata `ivregress gmm` on the lab workstation if Stata is installed there | U |
| D6c | Comparisons | (1) main: GNR materials-based markup `melast * rev/mate` on the same firm-years (02_GNR/03_outdata); (2) loose level check: Spain figures from De Loecker and Eeckhout (2018, Worldscope), PDF to be added by the user | U |
| D6d | Output | firm-year file in the format of Section 6 | U |
| D6e | Git | commit the `.gitignore` line ignoring `00_DLEU_Code/` with this work | U |

Notes on the choices:
- **D1, level and comparability.** Pooled sales-weighted `rev/V` is 1.24 over 2015-2023 in the 14 cells
  (cell medians 1.15 in 42/44 to 1.73 in 22), so with theta near 0.85-0.95 the aggregate markup should be
  roughly 1.1-1.2. Two opposite biases against COGS: `staf` includes all labour, part of which Compustat
  books in SG&A (V too large); `mate` excludes external services, energy and rents, part of COGS (V too
  small). 9.3% of window firm-years have V > rev, so their markup is below theta.
- **D2b, why 2016.** `L.i` needs `i_{t-1}`, which needs K in t-1 and t-2. The first year with `L.i` is 2016
  (K in 2014 and 2015). The windows are truncated at both ends, as DLEU's last window is.
- **D3, where kexp matters.** Only in the cs1 trim, the cost-share theta and the `+kexp` weights. The IV
  theta does not use it. Why EONIA-basis: it is the analogue of the Fed funds rate (an unsecured overnight
  market rate, not an administered policy rate). €STR starts only in Oct 2019 and EONIA was €STR + 8.5 bp
  from then on, so splicing €STR + 8.5 bp after EONIA ends gives one series without a break. The ECB deposit
  facility rate was the alternative (administered; EONIA traded about 10 bp below it in 2015-2021).
- **D6c, why Worldscope is only a loose check.** (Implemented in `04_build_markups.py`, config
  `compare.worldscope`; level only, 2016, with two bridges: DLEU's constant 0.85 elasticity, and the
  200 / 1,000 largest firms by sales.) Worldscope covers a few hundred listed Spanish firms with true
  COGS and a different sector mix; this sample is about 400k mostly small private firms with V = mate + staf.
  Data source, firm set and cost concept all differ. The sales-weighted mean, dominated by the largest firms,
  is the closest object to compare. The GNR comparison holds the data fixed and isolates the method.
- **D6a, the size of the loss.** Among consecutive tfas > 0 firm-years, only 62% have K - 0.9 L.K > 0
  (48% in 2022: HICP deflation of a nominal book stock). Compustat's gross ppegt rarely shrinks this way.

## 4. Answer: do we need ACF?

No. In the DLEU code the markup uses only `theta_WI1_ct` (`03_Calibration_Moments.do`, section 02). ACF
produces a single constant-over-time elasticity per sector that feeds `theta_unified_const`, i.e. a
constant returns-to-scale gamma for their IRS model's calibration. In this project the returns to scale
already come from the GNR stage (F(alpha)), so an ACF gamma would duplicate that object with a weaker
method. It is a backup at most. **Not built**; if wanted later it is a self-contained addition (one
optimizer per cell on the same sample).

The IV capital coefficient `theta_WI1_kt` is kept in the output at no cost, so the time-varying gamma
(`theta_c + theta_k`, DLEU section 04.1) can be formed later if needed.

## 5. Code layout

`03_Empirical/02_Orbis/03_Markups/01_code/`, mirroring `02_GNR/01_code/` (Python, numbered scripts, one
YAML config holding every choice):

| File | Role |
|---|---|
| `config_markups_orbis_es.yaml` | every choice of Section 3 (input, V definition, years, cells, user-cost parameters, trims, windows, edge rule) |
| `01_fetch_rates.py` | download EONIA and €STR daily series from the ECB Data Portal, build the EONIA-basis annual series, save raw CSV and annual means under `03_Markups/00_indata/rates/` (restartable, `--refresh`) |
| `02_build_sample.py` | load panel, select cells/years, build V, deflated levels, user cost and kexp, trim 1 and trim 2 with Stata percentiles, lags after trimming; writes `01_intermediary/markup_sample.parquet` plus a trim waterfall |
| `03_estimate_thetas.py` | rolling-window WI-IV per cell, edge rule for 2015, cost-share theta; writes `01_intermediary/thetas.parquet` (cell x year: theta_c, theta_k, n_obs, window, first-stage diagnostics) |
| `04_build_markups.py` | merge thetas, compute markups, write `03_outdata/firm_year_markups.parquet`, `manifest.yaml`, `markups_report.md` and `cell_year_summary.csv`; GNR comparison |
| `05_stata_crosscheck.py`, `check_stata.do` | verification: `prepare` exports one cell-window and synthetic percentile groups, `check_stata.do` runs DLEU's `ivregress gmm` line and `egen pctile` in Stata, `compare` reports the differences |
| `06_plot_aggregate_markup.py` | time series of the aggregate markup, sales- and cost-weighted (cost = V, DLEU `mu_cogs_w`), house style with a centred 3-year moving average; writes `03_outdata/aggregate_markup.{png,csv}` |
| `markups_common.py` | paths, config, logging, git state, `stata_pctile` (Stata `egen pctile` definition), sales-weighted quantile |

Run (about 40 s locally, no workstation needed):

```bash
cd 03_Empirical/02_Orbis/03_Markups/01_code
python 01_fetch_rates.py && python 02_build_sample.py && python 03_estimate_thetas.py && python 04_build_markups.py && python 06_plot_aggregate_markup.py
```

All firm-level files stay in the gitignored `01_intermediary/` and `03_outdata/` (WRDS derivatives). No
data are ever written outside the repo's gitignored folders.

## 6. Output for 04_Moments

`03_Markups/03_outdata/firm_year_markups.parquet`, one row per firm-year in the trimmed 2015-2023 sample:
- keys and codes: `bvdid, year, ind2d, naics4, nace3, cnae4`
- levels (HICP, 2019 EUR): `rev_D, V_D, mate_D, staf_D, cap_D, kexp`; plus `usercost`
- elasticities: `theta_wi_c, theta_wi_k, theta_imputed, theta_cs`
- markups: `markup_wi = theta_wi_c * rev_D / V_D`, `markup_cs = theta_cs * rev_D / V_D`
- flags: `in_iv_sample` (row used in its own year's window regression), `has_lag`, `has_lag_inv`

Weights the moments step can form directly: sales (`rev_D`), V (`V_D`), V + kexp. The xsga-based weights
have no analogue (D1). `manifest.yaml`: commit, config hash, input fingerprint, row counts per trim and per
cell-year.

## 7. Verification and report

- Estimator check: on one cell-window, numpy 2SLS vs statsmodels IV2SLS vs a hand-coded solve; Stata
  `ivregress gmm` on the workstation if available.
- Report per cell x year: theta_c, theta_k, n_obs in the IV, sales- and V-weighted mean markup,
  unweighted P10/P50/P90 and sales-weighted P10/P50/P90 (DLEU III.B ranks by sales share), counts dropped
  by each trim. Pooled numbers per year and over 2015-2023; comparison with the GNR materials-based markup
  (same firm-years) and, as a loose level check, De Loecker and Eeckhout (2018) Spain.
- pyflakes on the new files, `graphify update .`, local commit on `orbis-spain-cleaning`.

## 8. Open items

None.

## 9. Results (run of 2026-10-12)

Full tables: `03_Markups/03_outdata/markups_report.md` and `cell_year_summary.csv` (gitignored).

**Rates.** EONIA-basis rate: -0.11% (2015) to -0.48% (2021), 0.08% (2022), 3.29% (2023). The overlap
check gives EONIA = €STR + 8.5 bp exactly on all 578 days. User cost 11.3% (2015) falling to 7.4% (2022,
GDP-deflator inflation 4.7%) and 9.0% (2023).

**Sample waterfall** (14 cells, 2014-2023): 4.35M firm-years; trim 1 (rev/V) drops 86.9k; tfas <= 0 or
missing drops 354k (146k missing, 191k zero, 16k negative); the cost-share trim drops 78.1k. Final 3.83M,
of which 3.66M in 2015-2023 (685k firms) carry a markup. Rev/V cutoffs about 0.30 and 7.0 by year;
costshare1 cutoffs about 0.35 and 0.9999. **IV sample**: 1.48M firm-years (67.5k in 2016, 200-216k a year
in 2017-2022, 157k in 2023), the loss coming from L.i (K - 0.9 L.K <= 0 for 32-40% of firm-years with a lag, 53% in 2022).

**Verification.** numpy 2SLS = statsmodels IV2SLS = explicit two-stage OLS (max difference 9e-11); vs
Stata `ivregress gmm` on the workstation (cell 21, window 2017-2021, 3,627 rows): coefficients within
3e-7, robust SEs within 3e-9. `stata_pctile` = Stata `egen pctile` exactly on 16 synthetic cases (group
sizes divisible and not divisible by 100, with ties).

**Elasticities.** theta_wi_c ranges 0.83 (48 transport) to 0.97 (44 retail); minimum first-stage robust t
101. Very stable over time within a cell (largest move: 72, 0.943 to 0.912). Sales-weighted average 0.942
(2015) to 0.936 (2023), against about 0.85-0.90 in DLEU (Compustat COGS). The cost-share theta is higher,
0.90-0.99, because kexp on net book tangibles is small relative to V: the median costshare1 is above 0.97
in most cells.

**Markups, pooled by year** (weights within year over all cells):

| year | sales-w. mean | V-w. mean | (V+kexp)-w. mean | cost-share, sales-w. | P50 | P90 | P90 sales-w. |
|---|---|---|---|---|---|---|---|
| 2015* | 1.246 | 1.163 | 1.173 | 1.293 | 1.167 | 1.827 | 1.614 |
| 2016 | 1.284 | 1.180 | 1.192 | 1.334 | 1.167 | 1.827 | 1.695 |
| 2017 | 1.273 | 1.175 | 1.186 | 1.328 | 1.169 | 1.843 | 1.671 |
| 2018 | 1.258 | 1.167 | 1.176 | 1.312 | 1.172 | 1.856 | 1.646 |
| 2019 | 1.264 | 1.172 | 1.180 | 1.321 | 1.169 | 1.843 | 1.652 |
| 2020 | 1.241 | 1.156 | 1.163 | 1.293 | 1.147 | 1.782 | 1.615 |
| 2021 | 1.231 | 1.151 | 1.157 | 1.291 | 1.159 | 1.855 | 1.583 |
| 2022 | 1.240 | 1.154 | 1.160 | 1.309 | 1.168 | 1.882 | 1.603 |
| 2023 | 1.265 | 1.170 | 1.177 | 1.333 | 1.176 | 1.890 | 1.629 |

`*` theta imputed from the 2016 window. No trend over 2015-2023; the dip in 2020-2021 is COVID.

**By cell** (averages of the yearly within-cell figures): sales-weighted mean markup 1.12 (42 wholesale),
1.16 (44), 1.17 (56), 1.23 (23), 1.24 (22), 1.26 (11), 1.28 (31), 1.33 (54), 1.38 (48), 1.54 (61),
1.61 (72), 1.64 (71), 1.79 (51) and 2.14 (21 mining, 10k firm-years, a few large firms). High values in
72/71/61 reflect costs outside mate + staf (rents, external services), the D1 caveat.

**GNR comparison** (2.70M common firm-years, 74% of the markup sample): the GNR materials-based markup
melast * rev / mate is lower, sales-weighted 1.06-1.10 vs 1.22-1.26 and median 0.93-0.96 vs 1.14-1.16,
with Spearman rank correlation 0.70-0.74 by year (0.58 in 42 to 0.85 in 72 by cell).

**Worldscope level check, 2016** (sales-weighted; loose, see D6c; Orbis has unconsolidated accounts only):

| series | firms | markup |
|---|---|---|
| Worldscope Spain (De Loecker and Eeckhout, Table 1) | | 1.34 |
| Orbis baseline | 387,839 | 1.284 |
| Orbis, constant theta 0.85 | 387,839 | 1.164 |
| Orbis top 200 firms by sales, baseline | 200 | 1.320 |
| Orbis top 200 firms by sales, theta 0.85 | 200 | 1.202 |
| Orbis top 1,000 firms by sales, baseline | 1,000 | 1.303 |
| Orbis top 1,000 firms by sales, theta 0.85 | 1,000 | 1.183 |

The baseline level is close to Worldscope (1.28 vs 1.34; 1.32 for the 200 largest firms). The closeness
partly reflects offsetting differences: V = mate + staf includes all labour, so rev/V is lower than
sales/COGS, while the Spanish elasticities (sales-weighted 0.94) are above the US ones Worldscope uses.
With an elasticity near the US level (0.85) the Orbis markup is 1.16-1.20, below Worldscope. The level
match is therefore not evidence that the two measures agree firm by firm.
