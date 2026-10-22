# Level and dispersion of TFP and RTS

Splits every welfare leg (Lens A `realloc`, `scale`; Lens B `disp`, `level`; `total`) into a level and a
dispersion term for capability `v` (TFP) and scalability `alpha` (RTS). Nothing under `steady_state/` or
the other counterfactual packages is modified.

Run from `04_Code` after a production `sorting_curve` run (its `run_manifest.json`, `checkpoint_baseline.json`)
and the `scale_channel_decomposition` / `lens_b_decomposition` outputs exist:

```bash
python -m counterfactuals.level_dispersion.run_level_dispersion --workers 10
```

## Economies

All four use the sorting-curve draw (same seed, M, calibration, active set), frozen `chi` and `y_hat`, and are
solved by `sorting_curve.run_sorting_curve.solve_row` (ME, PE_I, U, PE) on the compact solver arrays.

| Label | Transform |
|---|---|
| `baseline` | none; gated against `sorting_curve/checkpoint_baseline.json` at 1e-6 |
| `hom_alpha` | sector harmonic mean of alpha (`homogeneous_rts.homogenize_alpha_primitive`, keeps mean curvature); gated against the saved `hom_primitive` rows |
| `hom_v` | sector harmonic mean of v (`homogenize_v_primitive`, keeps mean anchor marginal cost `Omega/v`) |
| `hom_both` | both |

## Decomposition (on `Delta = ln(1 + lambda)`, per leg)

```
one at a time:  Delta_base = Delta_hom_v     + (Delta_base - Delta_hom_v)       TFP level + TFP dispersion
                Delta_base = Delta_hom_alpha + (Delta_base - Delta_hom_alpha)   RTS level + RTS dispersion
joint:          Delta_base = Delta_hom_both + (Delta_hom_alpha - Delta_hom_both) + (Delta_hom_v - Delta_hom_both)
                             + (Delta_base - Delta_hom_alpha - Delta_hom_v + Delta_hom_both)
                             common level     TFP dispersion | common alpha       RTS dispersion | common v
                             interaction (the alpha-v pairing)
```

Collapsing either primitive within sector also removes within-sector sorting, so each one-at-a-time
dispersion term includes the sorting term; the joint split isolates it in the interaction.

Outputs in `out_results/counterfactuals/level_dispersion/`: `checkpoint_<label>.json`,
`validation_<label>.json`, `level_dispersion_economies.parquet`, `level_dispersion.yaml`.
