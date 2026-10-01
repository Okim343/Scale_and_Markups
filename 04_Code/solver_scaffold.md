# Steady-State Solver: Current `04_Code` State

This file documents the solver that is currently implemented in
`04_Code/steady_state/`. It is no longer a proposal for a sector-by-sector
solver. The codebase is a pooled steady-state model with common structural
parameters, a pooled empirical bundle, pooled market draws, EMX-style exogenous
entry, and a pooled GE normalization.

The current design, relative to a plain lognormal-productivity/Berry-entry
starting point:

- the model is pooled, not sector-by-sector;
- the empirical bundle supplies a single pooled `F_alpha` node table and pooled
  scalar targets directly;
- firm capability is a Pareto-tailed inverse-marginal-cost primitive `v`
  (not a lognormal productivity `z`), anchored at a common output `y_hat`;
  the `alpha`-`v` rank dependence is imposed through a Gaussian-copula
  correlation `rho_bar`;
- market structure is EMX-style **exogenous** Poisson entry, not an endogenous
  Berry/Bresnahan participation game: every allocated firm produces, and there
  is no per-period operating-cost cutoff in the static cross section;
- calibration is a single joint `trf` least-squares fit over
  `(xi, N, gamma, eta, rho_bar)` — five parameters, with `rho_bar` calibrated
  rather than fixed — evaluated through an **anchored** GE solve that also
  pins the capability scale `v_min`;
- welfare, simulation, and tests are implemented around the same pooled solve
  stack; a separate `04_Code/counterfactuals/` package reuses this stack for
  targeted reallocation exercises (fixed-capital planner, scalability-sorting)
  outside the baseline welfare comparison.

---

## 1. Top-Level Layout

```text
04_Code/
├── steady_state/
│   ├── __main__.py
│   ├── config.py
│   ├── config.yaml
│   ├── io_bundle.py
│   ├── pooled_inputs.py
│   ├── model/
│   │   ├── pool.py
│   │   ├── pricing.py
│   │   ├── market.py
│   │   ├── market_batch.py
│   │   ├── participation.py
│   │   ├── aggregator.py
│   │   └── normalization.py
│   ├── calibration/
│   │   ├── inner.py
│   │   ├── objective.py
│   │   ├── anchor_diagnostics.py
│   │   ├── entry_cost.py
│   │   ├── identification.py     # orphaned: not imported by __main__.py or tests
│   │   ├── diagnostics.py        # orphaned: not imported by __main__.py or tests
│   │   └── outer.py              # orphaned: not imported by __main__.py or tests
│   ├── simulation/
│   │   ├── cross_section.py
│   │   └── moments.py
│   ├── welfare/
│   │   ├── allocations.py
│   │   └── welfare_metrics.py
│   ├── figures/
│   │   ├── house_style.py
│   │   ├── simulation_figures.py  # not wired into __main__.py
│   │   └── generated/
│   └── utils/
├── counterfactuals/               # sibling package, reuses steady_state.{model,welfare}
│   ├── fixed_input_welfare/       # fixed-capital planner counterfactual
│   └── scalability_sorting/       # alpha-permutation / contribution-grid counterfactuals
├── tests/
├── marvin2/
├── out_results/
├── out_figs/
└── solver_scaffold.md
```

The main split is:

- `model/`: pooled draws, market solve, batched solve backends, exogenous
  entry, aggregation, and GE normalization;
- `calibration/`: pooled objective evaluation and the joint five-parameter
  fit (`inner.py`, `objective.py`), plus post-fit diagnostics
  (`anchor_diagnostics.py`) and the post-calibration sunk-entry-cost backout
  (`entry_cost.py`). `identification.py`, `diagnostics.py`, and `outer.py`
  still exist and are self-consistent, but nothing in `__main__.py` or
  `tests/` imports them — treat them as unmaintained scratch modules, not
  part of the live calibration path (the live path computes its own
  jacobian/identification diagnostics inline in `inner.py`);
- `simulation/`: pooled equilibrium runs and rectangular firm-panel output;
- `welfare/`: market/planner/uniform-markup comparisons on common draws;
- `tests/`: regression and contract checks for the pooled pipeline;
- `counterfactuals/`: a separate top-level package (not under `steady_state/`)
  that reuses the pooled model/welfare stack for targeted reallocation
  exercises outside the baseline market-vs-planner comparison.

---

## 2. User-Facing Commands

The package is run from `04_Code`:

```bash
python -m steady_state calibrate
python -m steady_state simulate
python -m steady_state validate
python -m steady_state welfare
```

`--config` is a **top-level** parser flag and must precede the subcommand
(e.g. `python -m steady_state --config my_config.yaml calibrate`), not a
per-subcommand flag.

Implemented per-subcommand flags:

- `calibrate`: `--out-dir`, `--M`, `--seed`, `--max-nfev`
- `simulate`: `--calibration`, `--M`, `--seed`, `--out-dir`
- `validate`: `--calibration`, `--M`, `--seed` (no `--out-dir`)
- `welfare`: `--calibration`, `--M`, `--seed`, `--out-dir`

There is no sector-by-sector calibration loop, no per-sector gamma
floor flag, no two-stage `--ge-M` path, and no plotting subcommand in the
active CLI (`figures/simulation_figures.py` exists but is not wired into
`__main__.py`).

---

## 3. Configuration Contract

`steady_state/config.py` loads the typed config from
`steady_state/config.yaml`.

The active config blocks are:

- `paths`: empirical bundle and output directories;
- `window`: active reporting window used for output file names;
- `monte_carlo`: `M`, `M_final`, `master_seed`;
- `parameters`: assigned `beta`, `delta_K`, `phi`, `H` (`phi_v` and `a` are
  **not** set here — they are external primitives read from the empirical
  bundle; see §4);
- `participation`: `K_switch`, `love_of_variety` (`K_switch` is accepted for
  signature compatibility by the exogenous-entry path but no longer changes
  behavior — see §8);
- `solver.market`: inner-market tolerances and backend selection;
- `solver.parallel`: market-chunk process parallelism for the batched solver;
- `calibration`: `markup_target` (`sga` or `cogs` — which empirical value the
  model's `mu_cw` moment is fit to), `eta.{init,lo}`, `xi.{lo,hi}` (Pareto tail
  bounds), `N.{lo,hi}` (Poisson-mean bounds), `rho_bar.{lo,hi}` (copula-strength
  bounds — `rho_bar` is now a calibrated bound pair, not a fixed value),
  `inner.{method,tol,max_nfev}`, `weights.pooled`.

The current local default backend is:

```yaml
solver:
  market:
    backend: vectorized
```

(`market_batch.solve_batch`'s own Python default is `backend="scalar"`; the
shipped `config.yaml` is what actually selects `vectorized` at runtime.)

`config.py` validates only the currently used pooled settings. `solver.sector_fp`
and `solver.outer_fp` still appear in the YAML as historical leftovers but are
not read by `config.py`. The YAML's top-level `welfare:` block
(`M`, `panel_filename_me`, `panel_filename_pe`, `out_yaml`) is itself stale —
its filenames (e.g. `sim_panel_2010_2019.parquet`) don't match what
`cmd_welfare` actually writes (`sim_panel_market.parquet`,
`sim_panel_planner.parquet`, `welfare_decomposition.parquet`, `welfare.yaml`,
all fixed names — see §12); it is not read by `config.py` either.

---

## 4. Empirical Bundle Contract

`steady_state/io_bundle.py` reads a pooled empirical bundle from:

```text
03_Empirical/01_industry_alpha/03_outdata/
```

The current bundle contains exactly three required artifacts:

- `F_alpha.parquet`
- `aggregate_moments.yaml`
- `manifest.yaml`

### 4.1 `F_alpha.parquet`

Required columns (`F_ALPHA_REQUIRED_COLS`):

- `node_id`
- `u`
- `weight`
- `alpha`

This is a single pooled scalability distribution. A `sector_id` column is now
treated as an error.

### 4.2 `aggregate_moments.yaml`

Required top-level keys (`load_bundle`): `active_window`, `targets`,
`validation`, plus `REQUIRED_SCALAR_KEYS = ("a", "n_firms_cs",
"alpha_z_copula_rho_bar", "alpha_sales_corr_empirical")`.

Required target keys (`REQUIRED_TARGET_KEYS`, exactly six): `mu_cw`, `cr4`,
`cr20`, `top1pct`, `top5pct`, `emx_slope`. A seventh key, `mu_cw_sga`, is
required *conditionally* — `Bundle.target_mu_cw_sga` raises if it's absent —
and is the value the model's `mu_cw` calibration moment is actually fit to by
default (`calibration.markup_target: sga`).

**Naming trap carried across the bundle boundary.** The on-disk data key
`mu_cw` is the accounting markup (aggregate revenue / variable cost).
`Bundle.targets` remaps this at load time: the model-facing dict renames the
data's `mu_cw` to `mu_cw_alpha` and reserves the unqualified `mu_cw` name for
the pure cost-weighted markup that the model actually calibrates against
(sourced from the data's `mu_cw_sga` when `markup_target="sga"`). Downstream
code (aggregator, calibration, welfare) consistently uses the *model* naming;
only `io_bundle.py`'s properties and the raw YAML use the *data* naming.

`io_bundle.py` exposes:

- pooled capital share `a`;
- pooled firm count `n_firms_cs` and `mean_firms_per_sector` (a diagnostic
  analogue of the exogenous-entry `N`, not itself a target);
- `phi_v` (externally-assigned gross-output value-added weight);
- pooled target dictionary (`targets` / `calibration_targets(markup_target=...)`);
- the alpha-`v` copula correlation `rho_bar` (`alpha_z_copula_rho_bar` in the
  YAML — see §5 for what this is actually used for now);
- `alpha_sales_corr_empirical` / `alpha_sales_corr_partial_empirical` (Stage
  S6b of the empirical pipeline) — a *separate* empirical moment
  (alpha-vs-log-sales correlation) from `rho_bar` (alpha-vs-productivity
  copula strength); do not conflate the two.

Legacy bundle shapes that expected `sector_table.parquet` or per-sector pooled
objects are no longer valid for this loader.

---

## 5. Capability Support and the Copula Draw Rule

The model now works with a single pooled support for `alpha` and a Pareto-tailed
capability primitive `v` (inverse marginal cost at the common anchor `y_hat`) —
**not** a lognormal productivity `z`. There is no `sigma_z` anywhere in the
current code.

`steady_state/pooled_inputs.py` copies the already-pooled support and targets
out of the bundle into a stable `PooledInputs` object. There is no re-pooling
inside the solver.

`steady_state/model/pool.py` draws the common random-number pool
(`draw_pool`), in two stages:

1. **Ladder position.** Combine the firm's alpha-rank score
   `tilde_alpha = Phi^{-1}(rank(alpha))` with independent luck `eps` into a
   standardized ladder:

   ```text
   ladder = rho_bar * tilde_alpha + sqrt(1 - rho_bar^2) * eps
   u = Phi(ladder)
   ```

   `rho_bar` is the alpha-`v` rank-copula strength; this is the *only* place
   `alpha` and `v` are linked.

2. **Pareto marginal.** Map the ladder percentile through a Pareto(`xi`)
   quantile:

   ```text
   v = v_min * (1 - u) ** (-1 / xi)
   ```

   `xi` is the Pareto tail index of `v` (must exceed 1); `v_min` is the
   capability scale, solved by the baseline anchoring condition (see §9.3),
   not configured or calibrated directly.

Entry is folded into the same draw call: `draw_pool` also calls
`allocate_poisson(N, M, H, rng)`, which draws `n_m = max(1, Poisson(N))`
producing firms per market and returns an `(M, H)` boolean `active_mask`
activating the first `n_m` pool slots (equivalent to a random `n_m`-firm
sample, since pool columns are i.i.d.). `N` is a calibrated parameter. All
allocated firms produce — see §8.

The pooled draw object, `PoolDraw`, has fields:

- `alpha`: shape `(M, H)`
- `v`: shape `(M, H)`
- `tilde_alpha`: shape `(M, H)`
- `active_mask`: shape `(M, H)`, bool

`model/pool.py` also ships `derived_z(alpha, v, y_hat)`, an explicitly
labeled **diagnostic** implied legacy productivity `z = (v/alpha)^alpha *
y_hat^(1-alpha)` — used only by calibration anchor diagnostics (§10), never by
the market solve itself.

---

## 6. Inner Market Solve

`steady_state/model/market.py` solves one product market jointly in:

- firm shares `s_j`;
- the market output index `Y_im`.

Inputs are:

- firm draws `(alpha_j, v_j)`;
- elasticities `(eta, gamma)`;
- factor prices `(w, R)`, the materials price `P` (normalized to 1 in the
  steady state, but threaded explicitly);
- pooled capital share `a`, gross-output value-added weight `phi_v`, and
  output anchor `y_hat`;
- market expenditure `X_market`.

The key point is unchanged from the older memo: with heterogeneous `alpha_j`,
the problem is not scale-free, so shares and levels must be solved jointly.

Marginal cost uses the **gross-output** unit cost `Omega^g(w, R, P) =
(Omega(w,R)/phi_v)^phi_v * (P/(1-phi_v))^(1-phi_v)` — the value-added cost
`Omega(w,R)` combined with a materials layer priced at `P` — not the plain
value-added cost. As `phi_v -> 1` this collapses back to the value-added-only
case.

The implemented algorithm is:

- damped fixed-point iteration on `(s, Y_im)`;
- Newton fallback after a configurable number of stalled iterations.

The returned `MarketSolution` stores:

- `y`, `p`, `s`, `mu`, `d`, `cost`;
- `Y_im`, `P_im`;
- variety-normalized `Y_j_normalized`, `P_j_normalized`;
- convergence diagnostics.

`cost` holds **true total variable cost** `TC = alpha * MC * y = alpha *
revenue / mu` (the gross-output migration redefined this from the older
`MC * y`) — `profit = revenue - cost = (1 - alpha/mu) * revenue`.

Markup logic is centralized in `steady_state/model/pricing.py` through three
regimes:

- `MARKET`: EMX markup rule;
- `PLANNER`: `mu = 1`;
- `UNIFORM`: common scalar markup `mu_bar`.

---

## 7. Batched Market Solve

`steady_state/model/market_batch.py` is now the main entry point used by the
rest of the model. It solves a stack of `M` markets with arrays that have a
leading market axis.

Implemented backends:

- `scalar`: loop over markets and call `model.market.solve`;
- `vectorized`: NumPy-vectorized fixed point with scalar fallback on
  unconverged markets.

The scalar solver remains the correctness oracle. `solve_batch`'s own Python
default is `backend="scalar"`; the shipped `config.yaml` (`solver.market.backend:
vectorized`) is what makes vectorized the effective default in practice. The
vectorized backend is designed to reproduce the scalar arithmetic while
speeding up the pooled Monte Carlo, and also supports optional process-level
chunking across markets (`solver.parallel`).

The batched output `BatchSolution` stores:

- per-firm arrays of shape `(M, H)`:
  - `s`, `mu`, `sales`, `cost`, `output`, `price`
- per-market arrays of shape `(M,)`:
  - `Y_im`, `P_im`, `Y_j_normalized`, `P_j_normalized`
  - `residual`, `converged`, `used_newton`, `iterations`

---

## 8. Entry: Exogenous EMX Allocation, Not Endogenous Participation

`steady_state/model/participation.py` no longer implements a Berry/Bresnahan
participation game. Following Edmond-Midrigan-Xu (2023, §VI), the per-sector
firm count is **exogenous**: `n_m = max(1, Poisson(N))` (drawn in
`model.pool.allocate_poisson`, folded into the pool draw — see §5), `N` is a
directly calibrated parameter, and **all** allocated firms produce. There is
no per-period operating cost and no within-sector profitability cutoff in the
static cross section.

`allocate_exogenous(alpha, v, active_mask, ...)` is the entire module: it runs
exactly one `solve_batch` call over the given mask — no loss-maker deletion,
no geometric/one-at-a-time elimination, no re-entry testing. `wf_reference` is
hard-set to `0.0`.

The returned `ParticipationResult` container is kept for downstream
compatibility, with fields:

- `active_mask`
- `solution` (a `BatchSolution`)
- `n_active`
- `wf_reference` — always `0.0`
- `Ed_full_pool` — always `nan`
- `pool_binds` — always all-`False`
- `empty_market` — always all-`False`
- `converged`

Only `active_mask`, `solution`, `n_active`, and `converged` carry real
content; the rest are vestigial fields kept so callers written against the
older container don't need to change.

The sunk entry cost `F` is **not** used for selection. It lives only in the
dynamic free-entry block and is backed out *after* calibration
(`calibration/entry_cost.py`, `compute_sunk_entry_cost`), from
`F = beta * E[pi] / (W * (1/beta - 1 + varphi))` (EMX's free-entry condition,
`varphi` the exogenous exit rate, default `0.04`). It never feeds back into
the static cross-section markups or shares.

Planner-specific social-surplus participation is still explicitly deferred
(`pooled_inputs.planner_selection_stub` raises `NotImplementedError`).

---

## 9. Aggregation and GE Normalization

### 9.1 Pooled Moments

`steady_state/model/aggregator.py` computes the pooled calibration moments
(`pooled_moments`, keyed by `POOLED_MOMENT_KEYS = ("mu_cw", "cr4", "cr20",
"top1pct", "top5pct", "emx_slope")`):

- `mu_cw` — the **pure** cost-weighted markup `Σ TC·mu / Σ TC`, the actual
  calibration target;
- `cr4`, `cr20`, `top1pct`, `top5pct`;
- `emx_slope` (and `emx_intercept`) — the sector-level `1/mu ~ HHI` slope.

It also reports, as diagnostics (not calibration targets):

- `mu_cw_alpha` — the accounting object aggregate revenue / true variable
  cost (`Σ sales / Σ cost`); **note the naming is easy to get backwards**:
  `mu_cw` is the pure markup, `mu_cw_alpha` is the accounting ratio — the
  reverse of what the bundle's on-disk data key `mu_cw` means (§4.2);
- `mean_active_count` (exogenous under EMX entry, ≈ `N`);
- `share_std`, `share_range`, `materials_share`.

### 9.2 Aggregate Resource Accounting

`aggregate_pooled_markets(...)` builds the top-level pooled CES aggregate over
markets and returns a `PooledAggregate` with:

- `C` (net consumption), `P`, `L`, `K`
- `Q` (top-level gross composite the demand system loads on) and `M`
  (aggregate materials use); the gross-output closure is `C = Q - M`, with
  `M = (1 - phi_v) * (population-scaled total variable cost)` — as `phi_v ->
  1`, `M -> 0` and `C -> Q`;
- aggregate profits, mean active count, population scaling, exposure weights.

Operating overhead is handled as per-active-firm labor cost:

```text
operating_cost_labor = wf_reference / w
```

— which is always `0.0` given the exogenous-entry `wf_reference` in §8.

### 9.3 GE Normalization

`steady_state/model/normalization.py` closes the pooled steady state.

Assigned objects:

- `R = 1 / beta - (1 - delta_K)` from the Euler equation;
- `chi = w / C` after convergence.

There are **three** related normalization calls, not two:

- `solve_full_pool_reference(...)`: normalizes the full-pool allocation with
  `wf=0` (retained for diagnostics only; the live calibration/simulation path
  builds its reference directly from the GE solve on the exogenous Poisson
  mask instead);
- `solve_pooled_ge(...)`: solves `(log w, log X_market)` in log-space least
  squares s.t. `P=1, L=1` on a given exogenous active mask — the frozen-anchor
  entry point used by simulation, welfare (planner/uniform-markup), and
  validation;
- `solve_pooled_ge_anchored(...)`: solves `(log w, log X_market, log v_min)`
  jointly — a third residual drives the **median** active-firm output
  (`y_anchor`, not the sales-weighted mean `y_sw`) to the common anchor
  `y_hat`. This is the anchoring condition described in `model.typ`
  ("the median active-firm output equals the common anchor"), and it is
  what `calibration/objective.py` actually calls at every evaluation
  (see §10.2) — `solve_pooled_ge` alone does *not* solve for `v_min`.

The main model object returned downstream is `PooledEquilibrium`, which also
carries `v_min`, `y_hat`, `y_sw` (sales-weighted mean output, diagnostic), and
`y_anchor` (median output, the anchoring statistic).

---

## 10. Calibration

### 10.1 Parameters

The current calibration is a single pooled `trf` fit over **five**
parameters:

- `xi` — Pareto tail index of capability `v`
- `N` — Poisson mean firms per sector
- `gamma`, `eta` — demand elasticities
- `rho_bar` — the alpha-`v` copula strength

`xi` and `N` replace an older `(sigma_z, phi_f)` pair; `rho_bar` is now
calibrated (with a logit reparameterization to its bounds), not fixed at the
bundle's warm-start value.

Externally assigned inputs: `a`, `H`, `phi_v`.

There are no per-sector parameters in the active calibration path.

Why is `H` not estimated (e.g. as in Eeckhout, Fu, Yeh, and Yoo, "Quantifying
Market Power," QMP)? Unlike QMP, who estimate the potential-entrant count `M`
jointly with `(sigma, phi)` using a business-dynamism moment, our static
steady state has no reallocation moment that separately identifies the pool
size from the entry parameters. We therefore fix `H` well above the
calibrated `N`'s upper bound and verify results are insensitive to `H` over a
range and that the pool never binds.

### 10.2 Objective Evaluation

`steady_state/calibration/objective.py`'s `pooled_moments_at_params` evaluates
moments at a parameter vector:

1. draw the pooled CRN pool via `model.pool.draw_pool` (this also draws the
   exogenous Poisson entry mask, so there is no separate "full pool vs
   selected" split — entry is exogenous, so they coincide);
2. solve `solve_pooled_ge_anchored` (jointly solving `w`, `X_market`,
   `v_min`) on that mask;
3. compute pooled moments (`aggregator.pooled_moments`) on the solved
   equilibrium.

The core active calibration moments (`MOMENT_KEYS = POOLED_MOMENT_KEYS`) are:

- `mu_cw`, `cr4`, `cr20`, `top1pct`, `top5pct`, `emx_slope`

`calibrate_pooled` (in `inner.py`) adds a **seventh** residual,
`corr_alpha_log_sales`, whenever `corr_alpha_log_sales_weight > 0` — the
shipped `config.yaml` sets that weight to `3.162` by default, so the
committed calibration actually fits seven moments, not six. This residual
targets the bundle's `alpha_sales_corr_empirical` (Stage S6b of the empirical
pipeline — see `03_Empirical/README.md`), computed via
`calibration/anchor_diagnostics.py`'s `anchor_diagnostics(equilibrium)`, which
also reports (but does not target) `corr_alpha_log_sales_partial_z` and
`corr_alpha_log_sales_partial_v` as overidentification diagnostics.

### 10.3 Optimizer

`steady_state/calibration/inner.py` fits the five common parameters jointly
via

```python
scipy.optimize.least_squares(..., method="trf")
```

using the transformed parameterization (`_decode`):

- `xi = exp(x0)`
- `N = exp(x1)`
- `eta = 1 + exp(x2)`
- `gamma = eta + exp(x3)`
- `rho_bar = logistic(x4)` rescaled into `[rho_bar_lo, rho_bar_hi]`

so the model-imposed restriction `1 < eta < gamma`, `xi > 1`, `N >= 1`, and
`rho_bar` in bounds always hold.

Default/config bounds: `xi in [1.05, 60.0]`, `N in [5.0, 1000.0]` (both from
`config.yaml`), `eta in [1.01, 10.0]` (lower bound from `config.yaml`, upper
hard-coded in `cmd_calibrate`), `gamma - eta in [0.01, 24.0]`, `rho_bar in
[0.0, 0.95]` (from `config.yaml`).

Residuals are target-relative errors (EMX-style normalizer `(model - target) /
(1 + |target|)`), reweighted by `calibration.weights.pooled` (shipped weights:
`mu_cw=10, cr4=1, cr20=1, top1pct=0, top5pct=0, emx_slope=sqrt(10),
corr_alpha_log_sales=sqrt(10)`).

### 10.4 Output Schema

`python -m steady_state calibrate` writes `calibration_pooled.yaml` with
schema `pooled-emx-v2` (the loader rejects any other schema string, including
the older `pooled-common-v1`):

```yaml
schema: pooled-emx-v2
parameters:
  xi: ...
  N: ...
  gamma: ...
  eta: ...
  rho_bar: ...
  v_min: ...
  y_hat: ...
  a: ...
  H: ...
  phi_v: ...
reference:
  w: ...
  R: ...
  X_market: ...
  wf_reference: ...   # always 0.0 under exogenous entry
  Ed_full_pool: ...
  P: ...
  L: ...
entry:
  sunk_entry_cost_F: ...   # post-calibration EMX free-entry backout; welfare-only
  varphi: ...
  note: ...
fit:
  success: ...
  cost: ...
  nfev: ...
  moments:
    mu_cw: {model: ..., target: ...}
    cr4: {model: ..., target: ...}
    cr20: {model: ..., target: ...}
    top1pct: {model: ..., target: ...}
    top5pct: {model: ..., target: ...}
    emx_slope: {model: ..., target: ...}
    corr_alpha_log_sales: {model: ..., target: ...}
    corr_alpha_log_sales_partial_z: {model: ..., target: ...}
  diagnostics:
    mu_cw_alpha: ...
    mean_active_count: ...
    empirical_N_sales_weighted: ...
    y_sw: ...
    y_anchor: ...
    corr_alpha_log_sales_partial_z: ...
    corr_alpha_log_sales_partial_v: ...
identification:
  eta_jacobian_column_norm: ...
  eta_standard_error: ...
  jacobian_rank: ...
  jacobian_condition: ...
  share_std: ...
  share_range: ...
```

The committed warm-start fit (`04_Code/out_results/calib_pooled/calibration_pooled.yaml`)
currently converges to `(xi, N, gamma, eta, rho_bar) ≈ (16.8, 57.1, 7.25, 1.54,
0.897)` with `v_min ≈ 195.5`, `H=2500`, `a≈0.395`, `phi_v≈0.348` — well away
from the empirical `rho_bar` warm start of `0.220` (see
`03_Empirical/README.md`'s "Alpha-productivity dependence" note).

Older calibration schemas with per-sector maps, or the `pooled-common-v1`
schema, are not accepted by the active loader in `__main__.py`.

---

## 11. Simulation Outputs

`steady_state/simulation/cross_section.py` runs the pooled equilibrium and
builds a rectangular firm panel.

The canonical panel columns (`SIM_PANEL_COLUMNS`) are:

- `market_id`
- `firm_id`
- `alpha`
- `v`
- `tilde_alpha`
- `active`
- `n_active`
- `share`
- `markup`
- `sales`
- `cost`
- `output`
- `price`

`python -m steady_state simulate` writes:

- `sim_panel_<start>_<end>.parquet`
- `sim_pooled_moments.parquet`
- `sim_economy_moments.yaml`

`steady_state/simulation/moments.py` augments the targeted moments with
normalization and entry-mask diagnostics such as:

- `w`, `R`, `C`, `chi`
- `P_agg`, `L_agg`, `K_agg`, `Pi_agg`
- `wf_reference` (always 0.0), `pool_binds`, `empty_markets` (both always
  vacuous under exogenous entry — kept for container compatibility)
- `converged`
- `normalization_residual`

---

## 12. Welfare Stack

`steady_state/welfare/allocations.py` solves market and planner allocations on
identical draws. The planner solve reuses:

- the market allocation's `wf_reference` (always 0.0);
- the market allocation's `active_mask` (the same exogenous Poisson entry
  mask);
- the market allocation's `(w, X_market)` as the GE initial condition.

`steady_state/welfare/welfare_metrics.py` then builds the fixed-participation
two-channel decomposition:

- market equilibrium;
- uniform-markup equilibrium with `mu_bar = mu_cw^market` (the **pure**
  cost-weighted markup — the aggregate that coincided with the accounting
  markup before the gross-output migration, but no longer does, since `cost`
  now holds true total variable cost rather than `MC*y`; see §6, §9.1);
- planner equilibrium.

Reported welfare objects include:

- `lambda_total`
- `lambda_level`
- `lambda_dispersion`
- `identity_residual`

`python -m steady_state welfare` writes:

- `sim_panel_market.parquet`
- `sim_panel_planner.parquet`
- `welfare_decomposition.parquet`
- `welfare.yaml`

`04_Code/counterfactuals/` (a sibling package to `steady_state/`) reuses this
market/planner allocation machinery for targeted reallocation exercises —
e.g. a fixed-capital planner counterfactual and scalability-sorting/
alpha-permutation counterfactuals — that sit outside the baseline welfare
comparison documented here; see that package's own `README.md` for details.

---

## 13. Tests and Current Reality Check

`04_Code/tests/` currently contains:

`test_aggregator.py`, `test_anchored_mc_relabeling.py`, `test_calibration.py`,
`test_gross_output.py`, `test_market.py`, `test_market_backends.py`,
`test_normalization.py`, `test_participation.py`, `test_pooled_structural.py`,
`test_simulation_cli.py` (plus `tests/fixtures/`).

These cover: normalization, the anchored-MC/`v_min` relabeling
(`test_anchored_mc_relabeling.py`), the gross-output cost migration
(`test_gross_output.py`), the scalar market oracle (`test_market.py`) and
batched backends (`test_market_backends.py`), calibration, exogenous entry
(`test_participation.py`), aggregation, and end-to-end CLI runs including
`simulate`/`validate`/`calibrate`/`welfare` (`test_simulation_cli.py` — this
one file covers "welfare plumbing" too; there is no separate welfare test
file).

So the correct mental model for `04_Code` is:

- one pooled empirical bundle;
- one common calibrated parameter vector — now five parameters,
  `(xi, N, gamma, eta, rho_bar)`, not four;
- a Pareto-tailed capability primitive `v` with an anchored relabeling to a
  diagnostic legacy `z`, not a lognormal `z` drawn directly;
- EMX-style exogenous Poisson entry, not an endogenous participation game;
- an anchored GE normalization (`solve_pooled_ge_anchored`) that jointly pins
  `v_min` to a median-output anchoring condition, used by calibration;
- one pooled market Monte Carlo;
- one pooled GE normalization stack reused by simulation, calibration, and
  welfare;
- a separate `counterfactuals/` package built on top of the same stack.

Any documentation that still refers to sector-level objects such as
`sector_table.parquet`, per-sector `gamma_i`, `n_i`, sector fixed points, an
outer eta loop, a lognormal `(sigma_z, phi_f)` parameterization, an
endogenous loss-maker-deletion participation game, a fixed (non-calibrated)
`rho_bar`, or the `pooled-common-v1` calibration schema is documenting an
older design, not the current code.
