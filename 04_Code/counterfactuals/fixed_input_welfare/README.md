# Fixed-capital-envelope planner counterfactual

A non-invasive `counterfactuals/` package that adds a **fixed-capital-envelope
planner** regime to the pooled steady-state model. Everything here imports
`steady_state` code unchanged — nothing under `steady_state/` is modified.

## 1. Motivation

Comparing the existing MARKET and PLANNER regimes in the production welfare run
(`out_results/welfare.yaml`, M=10000) gave a much larger productivity-loss
headline — **~41% gross output, ~24% value-added** (`lambda_total ≈ 0.319`) —
than a reference paper's static Table 6 at a comparable aggregate markup
(**~3% / ~7%**).

The gap is a real mechanical difference, not a bug. In *every* existing regime
the rental rate `R` is fixed at the Euler rate
(`R = euler_R(beta, delta_K)`, `steady_state/model/normalization.py:120-121`),
so capital is supplied perfectly elastically
(`K = a·phi_v·variable_cost/R`, `aggregator.py`). The PLANNER regime therefore
lets aggregate capital roughly **double** (`K_planner/K_market ≈ 2.14×`) as it
re-optimizes freely — a large first-order **scale** effect. The reference
paper's static exercise instead holds the aggregate factor envelope **fixed**
and only reallocates it more efficiently across firms — a much smaller,
second-order **misallocation** effect.

## 2. What this counterfactual computes

A PLANNER-priced equilibrium (`μ ≡ 1`) with aggregate capital `K` **pinned to
the MARKET equilibrium's realized `K`**. It does this by promoting the rental
rate `R` to a *third solved unknown* (alongside `w` and `X_market`), instead of
fixing it at the Euler rate. The third GE residual drives realized `K_agg` to
`K_target = K_market`.

This isolates the smaller **reallocation-only** loss (`lambda_reallocation`),
comparable in spirit to a Hsieh–Klenow-style static misallocation number, and
separates it from the **scale** loss (`lambda_scale`) that inflates the
headline.

## 3. Mechanism, precisely

`R` is not just aggregate bookkeeping. It feeds every firm's marginal cost via
`_omega_gross(w, R, a_i, phi_v)` (`steady_state/model/market.py:72-88`), called
inside the actual Cournot/planner fixed-point solve (`market_batch.py`,
`market.py`). So making `R` a solved unknown genuinely **re-prices every firm**
— it is a new GE object, not a relabeling. To pin `K` below the free-capital
planner's level, `R` must rise **above** the Euler rate to choke capital demand
down (in the sanity run `R` rose from `0.1017` to `0.166`).

No new `PricingRegime` enum member is added: this is a PLANNER-priced
equilibrium with a non-Euler `R`, distinguished purely by the free-text
`regime="planner_fixed_k"` label used in the output panel/table and by which
files it lands in.

## 4. Caveat vs. the reference paper's construction

This is **not** an exact replication of the paper's static exercise. The paper
holds the *entire* K+L+M envelope fixed and reallocates strictly within it. This
counterfactual pins only **aggregate capital `K`**; labor `w` and the materials
aggregator `X_market` still clear through the usual GE conditions (`P=1`,
`L=1`). So `lambda_reallocation` is the *closest available analogue* to the
paper's static number within this model's machinery — structurally comparable in
spirit, not identical in construction. Read it as "reallocation at a pinned
capital stock," not "the paper's exact Table 6 number."

### Why leave materials `M` free?

We deliberately pin only the **primary factors** (`K` here, `L` via the `L=1`
condition) and let materials `M` clear. EMX itself *does* hold materials fixed,
so this is a conscious deviation, motivated by what the number is meant to be
comparable to:

- **`M` is intermediate output, not a primal endowment.** In this gross-output
  model materials are the same composite the firms produce, recycled as input
  (`C = Q − M`; `M = (1−φ_v)·TC`). A Hsieh–Klenow-style static misallocation
  number holds the *primary-input envelope* (capital and labor) fixed and asks
  how much more output efficient reallocation of those endowments would yield.
  Pinning `K` and `L` is exactly that envelope; pinning `M` on top would freeze
  an endogenously-produced intermediate alongside a primary factor, which is a
  different (and non-standard) object.
- **No clean instrument, and it would distort the wedge.** Materials carry no
  separate factor price in the solved forward pass — they are bought at the
  numeraire (`_omega_gross`'s materials price is hardwired to `P=1` in the batch
  path). Pinning `M` would require introducing a materials-price wedge `P_M` as
  a 4th solved unknown and threading it through `steady_state/model/market.py`
  and the aggregator's `M`/`C` accounting — an invasive model extension, not a
  config of the existing machinery. It would also move the comparison *further*
  from the Hsieh–Klenow primal-factor benchmark, not closer.

Because `φ_v` is fixed, materials move only in proportion to total variable cost
(`M = (1−φ_v)·TC`); with `K` and `L` pinned, the residual movement in `M`
reflects the same reallocation the primary-factor envelope already captures, so
letting it clear does not smuggle in a separate "scale" channel. If an
EMX-exact, materials-fixed variant is ever needed as a robustness bound, it
should be built as a deliberate model extension (a `P_M` wedge with an
`M = M^{ME}` residual), not folded into this non-invasive package.

## 5. The decomposition it enables

Splitting the existing MARKET → PLANNER `lambda_total` into two legs through the
intermediate FIXED_K_PLANNER allocation:

- `lambda_reallocation` (MARKET → FIXED_K_PLANNER): reallocation gain at pinned
  aggregate capital.
- `lambda_scale` (FIXED_K_PLANNER → PLANNER): incremental gain from letting
  capital additionally expand at the Euler rate — the ~2.14× `K` effect.

with the multiplicative identity check
`(1 + lambda_reallocation)·(1 + lambda_scale) = (1 + lambda_total)`
(`identity_residual` should be ~0), mirroring the check in
`steady_state.welfare.welfare_metrics.two_channel_decomposition`.

## 6. How to run

From `04_Code/`:

```bash
# full run (MARKET + FIXED_K_PLANNER + full PLANNER; ~10-13 min at M=10000)
python -m counterfactuals.fixed_input_welfare.run_fixed_capital_planner \
  --calibration out_results/calib_pooled/calibration_pooled.yaml --M 10000

# faster: skip the full free-capital planner leg
# (reports only lambda_reallocation — no lambda_scale / identity check)
python -m counterfactuals.fixed_input_welfare.run_fixed_capital_planner \
  --calibration out_results/calib_pooled/calibration_pooled.yaml --M 10000 --skip-full-planner
```

CLI flags: `--config`, `--calibration` (required in practice — no silent
fallback to a missing default path), `--M`, `--seed`, `--out-dir`,
`--skip-full-planner`.

**Runtime** (~M=10000): MARKET cold-started ~4-5 min, FIXED_K_PLANNER
warm-started ~3-5 min, optional full PLANNER warm-started ~2-3 min →
**~10-13 min** total (~7-10 min with `--skip-full-planner`). Cost scales
roughly linearly in `M`.

**MARKET is re-solved from scratch** (not reconstructed from
`sim_panel_market.parquet`): that panel carries no
`w`/`R`/`X_market`/`wf_reference` needed to warm-start the 3-unknown solve, so
re-solving is both simpler and safer and gives full reproducibility. As a cheap
cross-check the run writes `market_recompute_check.yaml` to diff against the
production `welfare_decomposition.parquet` market row.

## 7. Output artifacts

Written to `out_results/counterfactuals/fixed_capital_planner/`:

| File | Contents |
|---|---|
| `sim_panel_planner_fixed_k.parquet` | firm-level panel, same schema as `sim_panel_planner.parquet`, `regime="planner_fixed_k"` |
| `fixed_capital_decomposition.parquet` | regime-level table: `market`, `planner_fixed_k`, (`planner`) rows × `C, L, K, R, profits, mu_cw, mu_cw_alpha, mean_active_count, wf_reference` |
| `fixed_capital_welfare.yaml` | `lambda_reallocation`, (`lambda_scale`, `lambda_total`, `identity_residual` if full planner), `K_target`, `K_target_gap`, `R_market`, `R_planner_fixed_k`, `mu_cw_planner_fixed_k`, `M`, `seed` |
| `market_recompute_check.yaml` | fresh MARKET leg's `C, L, K, R, mu_cw`, to diff against the production run |

## 8. Known risks

- **Convergence of the 3-unknown solve.** The extra `R` unknown makes the solve
  harder than the 2-unknown baseline. **Always sanity-check at small `M`
  (e.g. 200-500) first**, confirming `K_target_gap ≈ 0` and
  `mu_cw_planner_fixed_k ≈ 1.0`, before committing to a full M=10000 run.
- **`R` is unbounded** in `least_squares` here. If convergence is flaky, bound
  `log R` (e.g. `R ∈ [euler_R, 10·euler_R]`) as a fallback: `R` must rise above
  the Euler rate in this counterfactual, since the target `K` is smaller than
  the freely-optimizing planner's `K`.
- The run prints a `WARNING` if any leg fails to fully converge; treat a
  non-trivial `K_target_gap` or `identity_residual` as a red flag.
