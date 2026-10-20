"""Sorting sweep: arrangements of the alpha<->v pairing and the 04_sorting_sweep.py script.

* Every permutation mode keeps each sector's active-alpha multiset and the whole
  v array bitwise; ``sorted`` and ``reverse`` are the extreme within-sector pairings.
* ``partial_permute`` with q = 0 returns its base arrangement exactly.
* Copula re-draws share alpha, tilde_alpha and active_mask bitwise with the base.
* A two-arrangement sweep at tiny M passes the baseline gate and every identity
  check, and a rerun resumes from checkpoints without solving.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import SIM_DIR, write_bundle, write_sim_config
from orbis_model.counterfactuals.sorting import (
    PERMUTE_MODES,
    active_corr_av,
    assert_copula_crn,
    build_plan,
    homogenize_alpha_cost_weighted,
    homogenize_alpha_sector,
    partial_permute,
    permute_alpha,
    within_sector_rank_corr,
)
from orbis_model.model.normalization import PooledParams
from orbis_model.model.pool import draw_pool
from orbis_model.params import build_provenance, write_params_file

SUPPORT = np.linspace(0.7, 1.2, 25)


def _draw(rho_bar=-0.3, M=40, H=15, seed=11, **kwargs):
    return draw_pool(SUPPORT, xi=6.0, rho_bar=rho_bar, N=8.0, M=M, H=H, rng=seed, **kwargs)


def _assert_same_multisets(new, base):
    np.testing.assert_array_equal(new.v, base.v)
    np.testing.assert_array_equal(new.active_mask, base.active_mask)
    np.testing.assert_array_equal(new.alpha[~base.active_mask], base.alpha[~base.active_mask])
    for m in range(base.alpha.shape[0]):
        active = base.active_mask[m]
        np.testing.assert_array_equal(np.sort(new.alpha[m, active]), np.sort(base.alpha[m, active]))


@pytest.mark.parametrize("mode", PERMUTE_MODES)
def test_permutation_modes_preserve_marginals_bitwise(mode):
    base = _draw(alpha_max=1.1)
    new = permute_alpha(base, mode, rng=3)
    _assert_same_multisets(new, base)
    # tilde_alpha travels with alpha: the (alpha, tilde_alpha) pairs are unchanged.
    pairs = lambda d: sorted(zip(d.alpha[d.active_mask], d.tilde_alpha[d.active_mask]))
    assert pairs(new) == pairs(base)
    if mode == "identity":
        assert new is base


def test_sorted_and_reverse_are_the_extreme_pairings():
    base = _draw()
    srt, rev = permute_alpha(base, "sorted"), permute_alpha(base, "reverse")
    for m in range(base.alpha.shape[0]):
        active = np.flatnonzero(base.active_mask[m])
        by_v = active[np.argsort(base.v[m, active], kind="stable")]
        assert np.all(np.diff(srt.alpha[m, by_v]) >= 0.0)   # alpha rises with v
        assert np.all(np.diff(rev.alpha[m, by_v]) <= 0.0)   # alpha falls with v
    others = [base, permute_alpha(base, "shuffle", rng=1), partial_permute(base, "sorted", 0.5, rng=2),
              partial_permute(base, "reverse", 0.5, rng=2)]
    hi, lo = within_sector_rank_corr(srt), within_sector_rank_corr(rev)
    for other in others:
        assert lo <= within_sector_rank_corr(other) <= hi
    assert active_corr_av(rev) < active_corr_av(base) < active_corr_av(srt)


@pytest.mark.parametrize("base_mode", ["identity", "reverse", "sorted"])
def test_partial_permute_q0_is_the_base_arrangement(base_mode):
    base = _draw()
    zero = partial_permute(base, base_mode, 0.0, rng=5)
    ref = permute_alpha(base, base_mode)
    np.testing.assert_array_equal(zero.alpha, ref.alpha)
    np.testing.assert_array_equal(zero.tilde_alpha, ref.tilde_alpha)
    _assert_same_multisets(partial_permute(base, base_mode, 1.0, rng=5), base)


def test_copula_redraws_share_common_random_numbers():
    cap = 1.1
    base = _draw(rho_bar=-0.3, alpha_max=cap)
    for rho in (-0.9, 0.0, 0.6, 0.9):
        other = _draw(rho_bar=rho, alpha_max=cap)
        assert_copula_crn(other, base)
        for name in ("alpha", "tilde_alpha", "active_mask"):
            np.testing.assert_array_equal(getattr(other, name), getattr(base, name))
        assert not np.array_equal(other.v, base.v)
    tampered = dataclasses.replace(base, alpha=base.alpha * 1.0001)
    with pytest.raises(AssertionError, match="alpha"):
        assert_copula_crn(tampered, base)


def test_homogeneous_arrangements():
    base = _draw()
    hom = homogenize_alpha_sector(base)
    np.testing.assert_array_equal(hom.v, base.v)
    for m in range(base.alpha.shape[0]):
        active = base.active_mask[m]
        assert np.ptp(hom.alpha[m, active]) == 0.0
        assert np.mean(1 / hom.alpha[m, active]) == pytest.approx(np.mean(1 / base.alpha[m, active]), rel=1e-13)
    cw = homogenize_alpha_cost_weighted(base, 0.93)
    assert set(np.unique(cw.alpha[base.active_mask])) == {0.93}
    np.testing.assert_array_equal(cw.alpha[~base.active_mask], base.alpha[~base.active_mask])


def test_plan_starts_with_baseline_and_skips_the_calibrated_grid_point():
    plan = build_plan(0.3, grid=(-0.5, 0.3, 0.6), n_shuffle=2)
    labels = [a.label for a in plan]
    assert labels[0] == "baseline" and plan[0].rho_bar == 0.3
    assert labels == ["baseline", "rho_-0.50", "rho_+0.60", "reverse", "shuffle_0", "shuffle_1",
                      "sorted", "hom_sector", "hom_cost"]


# ---------------------------------------------------------------------------
# 04_sorting_sweep.py end to end at tiny M
# ---------------------------------------------------------------------------


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  SIM_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sweep_setup(tmp_path):
    bundle = write_bundle(tmp_path / "bundle")
    params_file = tmp_path / "params.yaml"
    out = tmp_path / "out"
    cfg = write_sim_config(tmp_path / "config_simulation.yaml", bundle_dir=bundle,
                           params_file=params_file, out_dir=out, H=6)
    prov = build_provenance(bundle_dir=bundle, configs={"simulation": cfg}, code_dirs=(), script="test")
    params = PooledParams(xi=8.0, N=3.0, gamma=3.8, eta=2.0, rho_bar=-0.3, a=0.35, H=6, phi_v=0.45)
    write_params_file(params_file, params, reference={"w": 1.0, "X_market": 1.0, "wf_reference": 0.0},
                      provenance=prov, model={"alpha_cap_factor": 0.99})
    assert _script("02_welfare.py").main(["--config", str(cfg)]) == 0
    assert _script("03_fixed_input_welfare.py").main(["--config", str(cfg)]) == 0
    return cfg, out


def test_sweep_smoke_validates_and_resumes(sweep_setup, monkeypatch):
    cfg, out = sweep_setup
    argv = ["--config", str(cfg), "--labels", "baseline,reverse", "--fig-dir", str(out / "figs")]
    sweep = _script("04_sorting_sweep.py")
    assert sweep.main(argv) == 0
    res = out / "sorting"
    validation = json.loads((res / "validation.json").read_text())
    assert validation["passed"] and validation["max_relative_difference"] <= 1e-6
    df = pd.read_parquet(res / "sorting_economies.parquet")
    assert list(df.label) == ["baseline", "reverse"]
    assert df[[f"converged_{k}" for k in ("ME", "PE_I", "PE_K", "PE", "U")]].all().all()
    assert df[["identity_A", "identity_B"]].abs().max().max() <= 1e-12
    assert df.chi.nunique() == 1 and df.chi.iloc[0] == df.chi_own_ME.iloc[0]
    for leg in ("realloc", "materials", "capital", "total", "disp", "level"):
        np.testing.assert_allclose(df[f"lambda_{leg}"], np.expm1(df[f"Delta_{leg}"]), rtol=0, atol=0)
    summary = yaml.safe_load((res / "sorting.yaml").read_text())["summary"]
    assert summary["complete"] and summary["all_converged"]
    assert (res / "results.md").read_text().startswith("# Sorting sweep (Orbis Spain)")

    # Resume: every arrangement is checkpointed, so nothing may be solved again.
    def no_solve(*args, **kwargs):
        raise AssertionError("resumed sweep re-solved an arrangement")

    monkeypatch.setattr(sweep, "solve_arrangement", no_solve)
    before = {p.name: p.read_text() for p in res.glob("checkpoint_*.json")}
    assert sweep.main(argv) == 0
    assert {p.name: p.read_text() for p in res.glob("checkpoint_*.json")} == before
    pd.testing.assert_frame_equal(pd.read_parquet(res / "sorting_economies.parquet"), df)


def test_sweep_gate_rejects_mismatched_references(sweep_setup, monkeypatch):
    cfg, out = sweep_setup
    sweep = _script("04_sorting_sweep.py")
    # A reference produced at another M is refused before anything is solved.
    monkeypatch.setattr(sweep, "solve_arrangement",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("solved")))
    with pytest.raises(RuntimeError, match="Rerun it"):
        sweep.main(["--config", str(cfg), "--M", "4", "--validation-only"])
    monkeypatch.undo()
    # A reference whose numbers differ fails the gate and records it.
    path = out / "welfare_2015_2019.yaml"
    ref = yaml.safe_load(path.read_text())
    ref["W_uniform"] *= 1.001
    path.write_text(yaml.safe_dump(ref))
    with pytest.raises(RuntimeError, match="baseline validation failed"):
        sweep.main(["--config", str(cfg), "--validation-only"])
    validation = json.loads((out / "sorting" / "validation.json").read_text())
    assert not validation["passed"]
    assert not (out / "sorting" / "checkpoint_baseline.json").exists()
