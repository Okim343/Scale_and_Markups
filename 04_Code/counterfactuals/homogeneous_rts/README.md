# Homogeneous-scalability counterfactual

A non-invasive `counterfactuals/` package that adds a third, curvature-preserving
economy to the `scalability_sorting` family. Everything here imports
`steady_state` and `fixed_input_welfare` code unchanged — nothing under either
is modified, and no recalibration happens.

## 1. Motivation

`counterfactuals/scalability_sorting/` shows that positive α–capability
sorting inflates the fixed-capital markup-reallocation loss `λ^K`: shuffling α
within each sector (a marginal-preserving re-pairing of α with capability `v`)
collapses `λ^K` from ~5.76% (baseline) to ~3.04% (shuffle average). That
experiment isolates the *sorting* channel, but it leaves open how much of the
*baseline* loss survives even with no sorting at all — i.e. how much comes from
firm-level scalability heterogeneity itself, holding each sector's *average*
curvature fixed.

This package answers that by adding a **homogeneous economy**: within each
sector, every active firm's α is replaced by that sector's harmonic mean of α.
Because the shuffle already preserves each sector's active-α multiset (hence
its mean curvature `κ̄_j = mean(1/α_j) − 1`), the homogeneous economy differs
from the shuffle economies in exactly one dimension — within-sector α
variance — while holding `κ̄_j` bit-for-bit identical, sector by sector, to the
baseline and every shuffle draw.

## 2. What this counterfactual computes

Two homogenized draws, each run through the same MARKET → FIXED-K-PLANNER →
`fixed_capital_decomposition` pipeline as `scalability_sorting`:

- **Primitive (headline).** Per-sector α_j^hom = ((1/n_j) Σ_{i∈active_j} 1/α_ji)^-1.
  Conserves `κ̄_j` exactly; the resulting `λ^K` feeds the headline decomposition.
- **Cost-weighted (robustness only).** A single economy-wide
  α^hom = (Σ_i c_i/α_i)^-1, weighted by baseline true-variable-cost shares
  `c_i = cost_i / Σcost` (`MarketSolution.cost = α·revenue/μ`). This holds the
  *aggregate* MC elasticity fixed rather than each sector's mean curvature — a
  different question, reported alongside but never substituted into the
  headline split.

Because `consumption_equivalent_lambda` returns `expm1((1-β)·ΔW)`
(`steady_state/welfare/welfare_metrics.py`), `Δ ≡ log(1+λ^K) = (1-β)·ΔW`
exactly, so the three legs decompose additively (not approximately):

```
Delta_base = Delta_hom + (Delta_shuffle - Delta_hom) + (Delta_base - Delta_shuffle)
             markup_common     dispersion                    sorting
```

- `markup_common = Δ_hom`: the loss that survives with a common curvature per
  sector and zero within-sector dispersion — "how costly are markups even with
  no scalability heterogeneity at all."
- `dispersion = Δ_shuffle − Δ_hom`: the extra loss from within-sector α
  dispersion alone, with sorting (the α↔v pairing) still switched off.
- `sorting = Δ_base − Δ_shuffle`: the extra loss from assortatively pairing
  that dispersion with capability — the object `scalability_sorting` already
  reports as `object_C_markup_sorting_premium`, restated on the `log(1+λ)`
  scale.

All three terms are **within-sector**: cross-sector variation in `κ̄_j` is held
fixed across baseline, shuffle, and homogeneous alike.

## 3. Mechanism, precisely

α enters the model only through marginal-cost curvature
(`steady_state/model/market.py`: `mc = (Ω/v)·(y/ŷ)^{1/α−1}`; at `y=ŷ`, MC is
α-free). `κ = 1/α − 1` is therefore the primitive curvature statistic, and
`homogenize_alpha_primitive` is defined directly on it: it is the unique
per-sector constant that preserves `mean(1/α)` (hence `κ̄_j`) while forcing
`var(α) = 0` within every active sector. `homogenize_alpha_cost_weighted`
instead requires `me_base` (the baseline MARKET equilibrium) to read
`participation.solution.cost`.

Both transforms only touch a `PoolDraw`'s `alpha`/`tilde_alpha` fields
(`tilde_alpha → NaN`: it is alpha's rank score, meaningless once alpha is
constant, and only feeds a diagnostic panel column, never the solver). `v` and
`active_mask` are untouched, mirroring
`counterfactuals.scalability_sorting.permute.permute_alpha`.

## 4. Design: reads the shuffle run, computes only the new leg

This package does **not** regenerate the shuffle family — it reads
`sorting_objects.yaml`/`sorting_economies.parquet` from a prior
`scalability_sorting` run for `Δ_base`/`Δ_shuffle`, and rebuilds only the base
draw and the two homogeneous economies itself.

Because the two runs are decoupled processes, the only thing tying them
together is that both build `base_draw = draw_pool(...)` from the same
`(support, params.xi, params.rho_bar, params.N, M, params.H, seed, params
.v_min)`. Any drift in calibration file, config, `M`, or seed between the two
runs would silently produce a homogeneous leg computed on a *different*
economy than the shuffle legs it's being compared against — invalidating the
whole decomposition without necessarily raising an error downstream.

**Consistency guard.** Before solving anything, the driver rebuilds
`base_draw`/`me_base` and diffs `C`, `K`, `R`, `y_anchor`, `corr(α,v)`, and
`chi` against the stored `scalability_sorting` baseline row at ~1e-6 relative
tolerance, writing `consistency_check.yaml` and raising loudly on any
mismatch. This is the one thing protecting the two-run design from silent
drift.

## 5. How to run

Prerequisite: a `scalability_sorting` run must already have written its
outputs (`sorting_objects.yaml`, `sorting_economies.parquet`) to
`--sorting-dir` (default `out_results/counterfactuals/scalability_sorting`).

From `04_Code/`:

```bash
python -m counterfactuals.homogeneous_rts.run_homogeneous_rts \
  --calibration out_results/calib_pooled/calibration_pooled.yaml
```

Flags: `--config`, `--calibration` (default
`out_results/calib_pooled/calibration_pooled.yaml`), `--sorting-dir`,
`--out-dir`. There is no `--M`/`--seed` — both are read from the sorting run's
`sorting_objects.yaml` so the two runs cannot drift apart on those dimensions.

To sanity-check quickly, first run `scalability_sorting` at a small `M`:

```bash
python -m counterfactuals.scalability_sorting.run_scalability_sorting \
  --calibration out_results/calib_pooled/calibration_pooled.yaml \
  --M 200 --n-shuffle 5 --out-dir out_results/counterfactuals/scalability_sorting_smoke
python -m counterfactuals.homogeneous_rts.run_homogeneous_rts \
  --calibration out_results/calib_pooled/calibration_pooled.yaml \
  --sorting-dir out_results/counterfactuals/scalability_sorting_smoke \
  --out-dir out_results/counterfactuals/homogeneous_rts_smoke
```

## 6. Output artifacts

Written to `out_results/counterfactuals/homogeneous_rts/`:

| File | Contents |
|---|---|
| `homogeneous_rts_decomposition.yaml` | `markup_common`/`dispersion`/`sorting` + `identity_residual`, `Delta_base`/`Delta_shuffle`(+se)/`Delta_hom_primitive`/`Delta_hom_cost_weighted`, `lambda_K_*`, provenance (`M`, `seed`, `chi_baseline`, `y_hat`, `sorting_dir`), curvature-disclosure table |
| `homogeneous_economies.parquet` | one row per variant (`primitive`, `cost_weighted`): `alpha_hom_min/max/mean`, `lambda_K`, `W_market`, `W_planner_fixed_k`, `mean_inv_alpha`, `lambda_K_ownchi`, `chi_own`, `converged`, `residual` |
| `sim_panel_homogeneous_primitive.parquet` | `build_firm_panel` of the fixed-K planner homogeneous (primitive) economy, `regime="planner_fixed_k_homogeneous_primitive"` |
| `consistency_check.yaml` | recomputed-vs-stored baseline residuals (`C`, `K`, `R`, `y_anchor`, `corr_av`, `chi_baseline`) |

## 7. Results (M=10000 production run)

From the full production run (`M=10000`, `seed=20260525`, calibration
`out_results/calib_pooled/calibration_pooled.yaml`, reading
`out_results/counterfactuals/scalability_sorting/`; consistency guard passed
with exact agreement, both homogeneous legs converged):

| Quantity | `λ^K` | `Δ = log(1+λ^K)` |
|---|---:|---:|
| Baseline (sorted) | 5.763% | 0.056033 |
| Shuffle average | 3.038% | 0.029925 |
| Homogeneous, primitive | 2.436% | 0.024064 |
| Homogeneous, cost-weighted (robustness) | 2.914% | 0.028728 |

| Term | `Δ` | Share of `Δ_base` |
|---|---:|---:|
| `markup_common` (= `Δ_hom` primitive) | 0.024064 | 43.0% |
| `dispersion` (= `Δ_shuffle − Δ_hom`) | 0.005862 | 10.5% |
| `sorting` (= `Δ_base − Δ_shuffle`) | 0.026108 | 46.6% |
| identity residual | 0.0 | — |

Reading: even with a common per-sector curvature and zero within-sector α
dispersion, **~43% of the baseline markup-reallocation loss survives**
(`markup_common`) — markups are costly on their own, independent of any
scalability heterogeneity to exploit. Restoring within-sector α dispersion
while keeping the α↔v pairing random adds only a modest further **~10.5%**
(`dispersion`). The remaining **~46.6%** is the assortative sorting of that
dispersion onto capable firms (`sorting`) — consistent with
`scalability_sorting/results.md`'s "sorting removes about half the loss"
headline, now showing that sorting is doing almost all of the non-common
work; dispersion by itself, absent sorting, is a comparatively small effect.

The cost-weighted robustness variant's `λ^K` (2.914%) sits close to the
primitive variant's (2.436%) — same ballpark — so the `markup_common` finding
is not an artifact of the specific per-sector harmonic-mean construction.

Full numbers: `out_results/counterfactuals/homogeneous_rts/homogeneous_rts_decomposition.yaml`.

## 8. Known risks

- **CRN/seed/ŷ/χ drift between the two runs** is the central risk of this
  decoupled design — see the consistency guard above. If it fires, rerun
  `scalability_sorting` with the same `--calibration` (and, if used,
  `--reanchor-yhat-baseline-median`) before rerunning this package.
- **Convergence of the fixed-K 3-unknown solve** carries the same risks as
  `fixed_input_welfare`/`scalability_sorting` (see their READMEs): check
  `K_target_gap`/`residual` and `converged` on every leg before trusting the
  numbers.
- **The cost-weighted variant is not comparable to the primitive one.** It
  answers "hold the aggregate MC elasticity fixed," not "hold each sector's
  mean curvature fixed" — do not report its `λ^K` as an alternative headline
  number.
- **Untargeted markup dispersion** (same caveat as
  `scalability_sorting/results.md`): the reallocation loss rides on the fat
  μ-tail that no calibration moment targets, so absolute `λ^K` levels — on
  every leg here, not just the homogeneous ones — inherit that robustness
  frontier. The *relative* decomposition is still clean under common random
  numbers.
