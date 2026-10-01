# Scale-channel decomposition

Companion counterfactual to [`scalability_sorting`](../scalability_sorting/) and
[`homogeneous_rts`](../homogeneous_rts/). Those two put only the **reallocation**
leg `λ^K` (MARKET → FIXED-K PLANNER) on the α-arrangement axis. This package runs
the **full free-capital PLANNER** leg — the one they deliberately skip
(`pe=None`) — under each same arrangement, obtains `λ_total`, and backs out the
**scale** leg `λ_scale`.

## What it produces

The additive 3×3 table on the `Δ = log(1+λ)` scale:

|            | common-α (`Δ_hom`) | heterogeneity (`Δ_shuf − Δ_hom`) | sorting (`Δ_base − Δ_shuf`) |
|------------|--------------------|----------------------------------|-----------------------------|
| **`Δ^K`**    | published (reallocation) | published | published |
| **`Δ_total`** | *new* | *new* | *new* |
| **`Δ_scale`** | = `Δ_total − Δ^K` | = `Δ_total − Δ^K` | = `Δ_total − Δ^K` |

Because `log(1+λ) = (1−β)·ΔW` is linear in welfare and the multiplicative Lens-A
identity `(1+λ^K)(1+λ_scale) = (1+λ_total)` holds arrangement by arrangement, the
`Δ_scale` row is the `Δ_total` row minus the (already published) `Δ^K` row,
term by term. The only genuinely new estimand is the **`Δ_total` row**.

**Headline expected result** (SPEC §6-i): the `Δ_scale` **sorting** term is
≈ 0 — the scale channel is capital deepening (`K_planner/K_market ≈ 2.14×`)
governed by aggregate curvature, which the marginal-preserving shuffle holds
fixed. That is a positive finding: the α↔ν sorting story is confined to the
reallocation leg and does not contaminate the scale gain (the two Lens-A legs are
channel-orthogonal in the sorting dimension). Reported, not asserted.

## Design (a): reuse cached λ^K + consistency guard

The reallocation welfare levels (`W_market`, `W_planner_fixed_k`) and `λ^K` are
**read from the prior parquets** — this package re-solves only the 14 **market**
legs (for the CRN warm point + drift guard) and the 14 **full-planner** legs
(new). `λ_total` is anchored on the *cached* `W_market`, so the log identity
`Δ_scale = Δ_total − Δ^K` closes to machine precision and cross-run CRN drift is
isolated into `consistency_check.yaml` rather than into `identity_residual`.

The two runs share only a base draw built from the same `M`/`seed`
(inherited verbatim from `sorting_objects.yaml` — there are **no** `--M`/`--seed`
flags). A consistency guard recomputes the baseline market economy and diffs
`C / K / R / y_anchor / corr_av / chi_baseline` against the stored sorting-run
baseline row (rel-tol `1e-6`), and asserts the 10 shuffle `perm_seed`s align 1:1,
**before** anything else runs.

## Protocol invariants enforced

1. **χ frozen at baseline** — `chi_baseline = float(me_base.chi)` computed once,
   passed as `chi` into every `steady_state_welfare` call (cached-market and new
   planner). Checked against provenance to `1e-6`.
2. **Common random numbers** — one `base_draw`; arrangements only permute α↔ν
   (`permute_alpha` self-asserts marginals preserved, `v` bit-identical).
3. **Frozen ŷ / anchor** — `params.y_hat` asserted equal to the cached `y_hat`;
   `solve_pooled_ge` never re-anchors.
4. **Frozen marginals** — covered by (2) plus an explicit `active_mask` recheck.
5. **Shared `active_mask` + `wf_reference` on the new planner leg**
   (correctness-critical): `_solve_full_planner` seeds the planner with
   `me.participation.wf_reference` / `me.participation.active_mask` from the
   arrangement's own market solve, warm-started at `(me.w, me.X_market)`. An
   assertion confirms `pe.participation.active_mask` matches `me`'s for all 14
   arrangements.
6. **Per-draw λ_scale differencing** (§6-iii) — `Δ_scale_i = Δ_total_i − Δ^K_i`
   per matched shuffle draw, then averaged (`per_draw_scale_delta`); never a
   ratio of averaged λ's.
7. **Two-run drift guard** — `consistency_check.yaml` written before any raise.

## How to run

From `04_Code/`, after `scalability_sorting` **and** `homogeneous_rts` runs exist:

```bash
python -m counterfactuals.scale_channel_decomposition.run_scale_channel \
  --calibration out_results/calib_pooled/calibration_pooled.yaml
```

Production compute (M=10000): 14 market (warm) + 14 full-planner solves, the
full-planner being the cheapest leg (~2–3 min warm). Total ≈ 1.5–2 h.

### Smoke run

```bash
python -m counterfactuals.scale_channel_decomposition.run_scale_channel \
  --calibration out_results/calib_pooled/calibration_pooled.yaml --M-smoke 200
```

The cached parquets are **M=10000 only**, so a low-M smoke cannot reuse them.
When `--M-smoke` differs from the cached `M`, the driver falls back to a
**self-contained in-process** run: it solves the fixed-K reallocation leg
in-process for every arrangement (design (b) at low M) so the full identity chain
is still validated end-to-end. `consistency_check.yaml` records `mode:
in_process` (no cross-run drift surface in that mode). ~1–2 min total.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--config` | package default | steady-state config |
| `--calibration` | `out_results/calib_pooled/calibration_pooled.yaml` | calibration YAML (must match the cached runs') |
| `--sorting-dir` | `…/counterfactuals/scalability_sorting` | cached reallocation parquets |
| `--homogeneous-dir` | `…/counterfactuals/homogeneous_rts` | cached hom-α parquets |
| `--out-dir` | `…/counterfactuals/scale_channel_decomposition` | outputs |
| `--M-smoke N` | — | override M; triggers the in-process fallback if `N ≠` cached M |
| `--lens-b` | off | **inert stub** this release (reserved for the Lens-B level channel) |

## Outputs (`out_results/counterfactuals/scale_channel_decomposition/`)

- **`scale_channel_economies.parquet`** — one row per arrangement (14 rows):
  `label`, `perm_seed`, `corr_av`, `W_market_cached`, `W_market_resolved`,
  `drift_W_market_rel`, `W_planner`, `W_planner_fixed_k_cached`,
  `lambda_K_cached`, `lambda_total`, `lambda_scale`, `delta_K`, `delta_total`,
  `delta_scale`, `delta_scale_termwise_gap`, `K_market`, `K_planner`, `K_ratio`,
  `K_target_gap_cached`, `mu_cw_market`, `mu_cw_planner`, `identity_residual`,
  `market_converged`, `planner_converged`.
- **`scale_channel_decomposition.yaml`** — `provenance`, the `three_by_three`
  table (values, %-shares, per-draw SEs on the shuffle-derived terms),
  `per_arrangement` rows, `robustness` (primitive vs cost-weighted common-α scale
  term), `diagnostics`, and `caveats`.
- **`consistency_check.yaml`** — per-field recomputed-vs-cached rel-diffs,
  perm_seed alignment, mode, overall pass/fail.

## Acceptance checks (validated at the smoke M)

`identity_residual < 1e-9`; `Δ_scale = Δ_total − Δ^K` termwise `< 1e-9`; planner
`mu_cw ≈ 1.0` (p = MC); consistency guard passes; `active_mask` invariant holds
for all 14 arrangements; baseline `λ_scale ≈ 24.8%` / `K_ratio ≈ 2.14×`
(order-of-magnitude); `Δ_scale` sorting term near-zero (reported).

## Results (M=10000 production run)

Provenance: `M=10000`, `seed=20260525`, `shuffle_seed=20260709`, `n_shuffle=10`,
`chi_baseline=0.281430`, `y_hat=1.0`, design `reuse_cached_lambda_k`. All asserted
checks passed: `identity_residual` max `2.2e-16`, `Δ_scale = Δ_total − Δ^K`
termwise max `2.8e-17`, planner `mu_cw ≡ 1.0`, `drift_W_market_rel` max `0.0`
(cache reuse exact), guard passed (rel-diffs `0.0`, perm_seeds aligned),
`active_mask` invariant held for all 14 arrangements. The baseline reproduces the
paper headline exactly: `λ_total = 31.94%`, `λ_scale = 24.75%`,
`K_planner/K_market = 2.141×`.

### The 3×3 (`Δ = log(1+λ)`)

|            | common-α        | heterogeneity   | sorting          |
|------------|-----------------|-----------------|------------------|
| **`Δ^K`**    | 0.02406 (42.9%) | 0.00586 (10.5%) | 0.02611 (46.6%)  |
| **`Δ_total`** | 0.12088 (43.6%) | 0.03963 (14.3%) | 0.11668 (42.1%)  |
| **`Δ_scale`** | 0.09682 (43.8%) | 0.03377 (15.3%) | **0.09057 (41.0%)** |

The `Δ^K` row reproduces the published `homogeneous_rts` split
(43.0 / 10.5 / 46.6%) to within rounding — the cache-reuse design is validated
against the prior run. Shuffle-derived SEs (per-draw CRN differencing, §6-iii):
`Δ^K` sorting/heterogeneity `2.8e-5`, `Δ_total` `1.0e-4`, `Δ_scale` `7.4e-5` —
the scale SE is tighter than the naive `Δ_total` SE because the CRN pairing
cancels the shared draw noise, and is the same order as the published `λ^K`
shuffle SE (`2.91e-5`).

### Headline: the SPEC §6-i orthogonality hypothesis is overturned

The `Δ_scale` **sorting** term is **+0.0906 (41.0% of the scale channel, SE
7.4e-5)** — decisively *not* the ≈0 that SPEC §6-i predicted, and confirmed at
full sample (the M=200 smoke showed the same). Per-arrangement `λ_scale` falls
monotonically with the α↔ν correlation:

| arrangement | corr(α,ν) | `λ_scale` | `K_planner/K_market` |
|---|---|---|---|
| baseline          | +0.785 | 24.75% | 2.141× |
| shuffle (mean)    | +0.013 | 13.96% | 1.697× |
| reverse           | −0.825 | 12.01% | 1.621× |
| hom-primitive     | +0.094 | 10.17% | 1.533× |
| hom-cost-weighted | (n/a)  | 12.96% | 1.608× |

So the α↔z sorting mechanism is **not** confined to reallocation: it amplifies
the capital-deepening (scale) channel almost as strongly (41% sorting share in
`Δ_scale` vs 47% in `Δ^K`). The SPEC premise — "the shuffle preserves each
sector's κ̄_j, so aggregate curvature and hence the planner's K expansion are held
fixed" — is incomplete. The market's capital under-accumulation (the gap the
scale channel closes) depends on the α↔z *joint*, which the marginal-preserving
shuffle destroys; the K expansion drops from 2.14× (sorted) to ~1.70× (shuffled).
The two Lens-A legs are therefore **not** channel-orthogonal in the sorting
dimension.

### Robustness caveat (§6-ii)

The common-α *scale* term is primitive `0.0968` vs cost-weighted `0.1218` —
**26% apart**, noticeably less stable than the `λ^K` case (2.44% vs 2.91%). So
part of the common-α scale term carries the curvature-level confound: the
per-sector harmonic-mean collapse shifts aggregate MC curvature, which itself
moves the planner's optimal K. The **sorting** and **heterogeneity** terms are
unaffected (they difference out the hom leg), so the headline finding is clean;
only the common-α column's *level* interpretation inherits the confound.

## Not in scope this release

- **Lens-B** level-channel per-arrangement decomposition
  (`two_channel_decomposition`, UNIFORM-regime solve) — `--lens-b` reserved as an
  inert stub.
- A drift-proof self-contained design-(b) production path — the in-process
  fallback exists only for the smoke; production uses design (a) + the guard.
