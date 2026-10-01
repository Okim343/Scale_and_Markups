# Planner expansion

Descriptive companion to [`scalability_sorting`](../scalability_sorting/) and
[`scale_channel_decomposition`](../scale_channel_decomposition/). Those packages
show that positive alpha-nu sorting raises the welfare cost of markups in both
Lens-A legs. This package asks *where in the cross-section* the planner puts the
extra inputs, to test the mechanism the paper hedges in the sorting section:
the most scalable technologies sit in the most capable firms, which are the
largest and carry the highest markups, so the markup wedge holds back input
demand where the planner would expand it most.

Nothing is re-solved. The script reads the stored baseline panels, so it runs in
seconds.

## Why TC shares are input shares

`cost` in the panels is true total cost `TC = alpha*revenue/mu` (not `MC*y`).
Within a regime, firm inputs are `k = a*phi_v*TC/R`, `l = (1-a)*phi_v*TC/W`,
`m = (1-phi_v)*TC/P` with scalar `a` and `phi_v`, so a firm's share of every
aggregate input equals its share of aggregate `TC`, and `TC_pl/TC_mkt` is the
firm's input expansion (up to the regime's factor prices). The free planner and
the market share `R`, so `sum TC_pl / sum TC_mkt = K_PE/K_ME`. The fixed-K
planner shares `K`, so the same ratio equals `R_PE_I/R_ME`.

## Inputs (all postdate the rho_bar fix, commit 0eab400)

| file | regime |
|---|---|
| `out_results/welfare/sim_panel_market.parquet` | market |
| `out_results/welfare/sim_panel_planner.parquet` | free-capital planner |
| `out_results/counterfactuals/fixed_capital_planner/sim_panel_planner_fixed_k.parquet` | fixed-K planner (optional) |
| `out_results/welfare/welfare.yaml` | `mu_bar` check |
| `out_results/counterfactuals/fixed_capital_planner/fixed_capital_welfare.yaml` | `R` ratio check |
| `out_results/counterfactuals/scale_channel_decomposition/scale_channel_economies.parquet` | baseline `K_ratio` check |

Only active firms are read; panels are merged on `(market_id, firm_id)`. The
script refuses any path under `ARCHIVE/`.

## Validation (aborts on failure)

1. Free planner: `sum TC_pl / sum TC_mkt` equals the baseline `K_ratio` (rel. tol 1e-6).
2. Fixed-K planner: `sum TC_fk / sum TC_mkt` equals `R_planner_fixed_k / R_market` (rel. tol 1e-6).
3. `corr(alpha, v)` over active firms equals 0.78486 to five decimals.
4. Market cost-weighted markup equals `welfare.yaml: mu_bar`.
5. Active sets, `alpha` and `v` are identical across the panels.

## Tables

All tables are built for both planners (`free`, `fixed_k`). Quintiles use
tie-broken ranks, `rank(method='first')` on rows sorted by
`(market_id, firm_id)`, because alpha has a discrete support (479 values,
including mass points at 0.6 and at the top value 1.0879). `pooled` ranks across
all active firms (headline); `within` ranks inside each sector (robustness).

| `table` | cells | content |
|---|---|---|
| `quintile_alpha` | alpha quintile | mean alpha, mean ln v, firm share, cost-weighted market mu, share of market TC, share of planner TC, TC ratio, share of the TC gain |
| `quintile_v` | nu quintile | same |
| `double_alpha_v` | 5x5 alpha by nu | same fields per cell |
| `double_mu_alpha` | 5x5 market-markup by alpha | same fields per cell |
| `size_rank` | within-sector sales rank: leader, 2nd, 3rd to 5th, rest | same, plus within-sector percentile ranks of alpha and v |

Every cell also carries `sales_ratio` and the TC-weighted means
`wmean_log_mu_mkt` and `wmean_log_tc_ratio`. Because
`log(TC_pl/TC_mkt) = log(mu_mkt) + log(sales_pl/sales_mkt)`, their ratio is the
part of the cell's log expansion that is direct removal of the wedge.

The regression is `log(TC_pl/TC_mkt)` on `log(mu_mkt)`, `alpha`, `log(v)` with
sector fixed effects (within transformation), firm-weighted and
TC_mkt-weighted, reporting coefficients, coefficient times within-sector SD,
partial R2, R2 of each regressor alone, within R2 and R2 with FE. It is
**descriptive, not causal**: `mu_mkt` is an equilibrium function of market
share, which is itself a function of `(alpha, v)`. The logit-alpha variant is
skipped because about 3.4% of active firms have `alpha >= 1`, and `1/alpha` is
used instead. Standard errors are not reported, since the sample is the full
simulated population.

`alpha_support` records how much of the expansion falls on firms with
`alpha >= 1` and at the top mass point.

## How to run

From `04_Code/`:

```bash
python -m counterfactuals.planner_expansion.run_planner_expansion
```

Pass `--fixed-k-dir ''` to skip the fixed-K tables.

## Outputs

`out_results/counterfactuals/planner_expansion/`:

- `planner_expansion_cells.parquet`: long cell table, keyed by `table`,
  `scheme` (`pooled`/`within`), `planner` (`free`/`fixed_k`), `var1`/`bin1`,
  `var2`/`bin2`.
- `planner_expansion.yaml`: `provenance`, `validation`, every cell under
  `<table>.<scheme>.<planner>.<bin>` (bins `Q1..Q5`, `r<i>_c<j>` for double
  sorts with rows = first variable, or the size-rank label), `regression`,
  `alpha_support`.
- `results.md`: hand-written verdict above the marker line, auto-generated
  tables below it. Reruns regenerate only the part below the marker.

`counterfactuals/planner_expansion/table_snippet.typ`: a `#figure` in the format
of `tab:sorting` (alpha quintiles, nu quintiles, size rank, regression). It is
generated by the script and is not inserted into `model.typ`.

## Headline (see `results.md`)

Pooled alpha quintile TC ratios are 1.42 / 1.50 / 1.56 / 1.62 / 3.14, and the
nu quintiles look the same. The double sort puts 67.6% of the TC gain in the
(Q5 alpha, Q5 nu) cell, and sector leaders alone (1.75% of firms, mean alpha
1.05, mu 1.26) take 47.5% at a ratio of 5.57. Conditional on the other two, the
markup is the strongest predictor and alpha comes second. Log nu adds nothing,
with a slightly negative sign. Within sectors, the expansion requires both a
high markup and a high alpha. The direct wedge is a small part of the leaders'
expansion (0.23 of 1.69 log points), and the rest is the output response of
their scalable technology.
