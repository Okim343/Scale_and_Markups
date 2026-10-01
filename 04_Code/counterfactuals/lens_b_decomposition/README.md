# Lens-B decomposition (dispersion × level)

Companion to [`scale_channel_decomposition`](../scale_channel_decomposition/)
(Lens A: reallocation × scale). This package puts the **Lens-B** split of the
market→planner welfare gain — **dispersion × level** — on the same 14
α-arrangement axis (baseline / shuffle ×10 / reverse / hom-primitive /
hom-cost-weighted).

It **runs after** the scale-channel run and **reuses that run's cached welfare
levels** (`W_market`, `W_planner`) from `scale_channel_economies.parquet`, so the
expensive full free-capital **planner is never re-solved**. The only new solves
are the **UNIFORM legs** (μ ≡ μ̄), which give the dispersion leg; the level leg
falls out residually.

## What it produces

The additive 3×3 table on the `Δ = log(1+λ)` scale, sharing the cached
`Δ_total` row:

|                    | common-α (`Δ_hom`) | heterogeneity (`Δ_shuf − Δ_hom`) | sorting (`Δ_base − Δ_shuf`) |
|--------------------|--------------------|----------------------------------|-----------------------------|
| **`Δ_total`**       | cached (scale-channel) | cached | cached |
| **`Δ_dispersion`**  | *new* | *new* | *new* |
| **`Δ_level`**       | `= Δ_total − Δ_disp` | `= Δ_total − Δ_disp` | `= Δ_total − Δ_disp` |

Because `log(1+λ) = (1−β)·ΔW` is linear in welfare and the multiplicative Lens-B
identity `(1+λ_disp)(1+λ_level) = (1+λ_total)` holds arrangement by arrangement,
the `Δ_level` row is the `Δ_total` row minus the `Δ_disp` row, term by term. The
only genuinely new estimand is the **`Δ_dispersion` row**.

## Design (minimal / time-boxed): reuse cached planner, reconstruct the rest

`solve_pooled_ge`'s `active_mask` and `wf_reference` are **exogenous
passthroughs** ("no selection"; `normalization.py:271,281`), so they are
reconstructed for free from the base draw + calibration — bit-identical to what
the scale-channel market/planner legs used. Consequently:

- `W_market`, `W_planner`, `λ_total`, `Δ_total` are **read from the cache**.
- μ̄ = `mu_cw_market` is **read from the cache** for 12 of 14 arrangements.
- Only **3 MARKET solves** run: the baseline (to build the cost-weighted
  homogeneous draw and to warm the UNIFORM chain) plus the two homogeneous-α
  arrangements (whose `mu_cw_market` is NaN in the cache, so μ̄ is recomputed).
- **14 UNIFORM solves** run (chained warm-start; cold-retry fallback).
- **0 PLANNER solves.**

Everything is anchored on the cached `W_market`/`W_planner` (both on the frozen
`chi_baseline` from provenance) and the UNIFORM economy is a deterministic
fixpoint on the reconstructed draw, so both Lens-B identities close to machine
precision **with no cross-run drift guard**. A soft check reports the recomputed
baseline `chi` vs the cached `chi_baseline` (warns above rel-tol `1e-4`); welfare
always uses the cached `chi_baseline` for consistency with `W_market`/`W_planner`.

`M` / `seed` / `shuffle_seed` / `chi_baseline` are inherited **verbatim** from the
scale-channel `provenance` block (no `--M` / `--seed` flags): the base draw must
be bit-identical to the cached run for the reuse to be valid.

> **Unification note.** This is a deliberately standalone package (its own pure
> `decomposition.py`, no import of the scale-channel *code*; only its *data*).
> It is intended to be folded into a unified counterfactual driver later.

## How to run

From `04_Code/`, after `scale_channel_decomposition` has produced its outputs:

```bash
python -m counterfactuals.lens_b_decomposition.run_lens_b \
  --calibration out_results/calib_pooled/calibration_pooled.yaml
```

Production compute (M=10000): ~3 market + 14 uniform solves, **no planner**;
≈ 30–45 min.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--config` | package default | steady-state config |
| `--calibration` | `out_results/calib_pooled/calibration_pooled.yaml` | calibration YAML (**must match** the scale-channel run's) |
| `--scale-channel-dir` | `…/counterfactuals/scale_channel_decomposition` | cached parquet + provenance yaml |
| `--out-dir` | `…/counterfactuals/lens_b_decomposition` | outputs |

## Outputs (`out_results/counterfactuals/lens_b_decomposition/`)

- **`lens_b_economies.parquet`** — one row per arrangement (14 rows): `label`,
  `perm_seed`, `corr_av`, `W_market_cached`, `W_uniform`, `W_planner_cached`,
  `mu_bar`, `mu_cw_uniform`, `lambda_total`, `lambda_dispersion`, `lambda_level`,
  `delta_total`, `delta_dispersion`, `delta_level`, `lambda_total_cached`,
  `delta_total_cached_gap`, `delta_level_termwise_gap`, `identity_residual`,
  `uniform_converged`.
- **`lens_b_decomposition.yaml`** — `provenance`, the `three_by_three` table
  (values, %-shares, per-draw SEs on the shuffle-derived terms),
  `per_arrangement` rows, `robustness` (primitive vs cost-weighted common-α
  dispersion term), `diagnostics`, and `caveats`.

## Acceptance checks

`identity_residual` ≈ 0 (machine precision); `Δ_level = Δ_total − Δ_disp`
termwise ≈ 0; `delta_total_cached_gap` ≈ 0 (recomputed total matches cache);
`mu_cw_uniform ≈ mu_bar` (target realized); baseline `chi` rel-diff vs cache
`< 1e-4`; all UNIFORM legs converged.

## Interpretation & caveats

- **The dispersion leg is expected to be genuinely sorting-sensitive** — unlike
  the scale-channel's a-priori orthogonality hypothesis. Sorting reshapes which
  firms bear the wedge dispersion, so a non-trivial sorting share in
  `Δ_dispersion` means the cost of dispersion depends on *which* technologies
  carry the wedges.
- **Homogeneous-α confound** — the common-α column inherits the curvature-level
  confound; see the robustness block. Expected milder than the scale case
  (dispersion removal drives no capital expansion).
- **Untargeted μ-dispersion frontier** — absolute λ levels ride the uncalibrated
  markup tail, which bites the dispersion leg most directly; the **relative** 3×3
  under common random numbers is the clean object.
