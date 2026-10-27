"""Restricted-heterogeneity recalibration: inputs, coordinates, checkpoint replay.

Synthetic support only; the real bundle is never read here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from orbis_estimation.restricted import (
    ECONOMIES, Checkpoint, common_alpha, coordinate_bounds, decode, economy_inputs, encode, fingerprint,
    scan_points,
)
from orbis_model.model.normalization import PooledParams
from orbis_model.model.pool import alpha_cap, draw_pool
from orbis_model.primitives import ModelPrimitives

SUPPORT = np.linspace(0.8, 1.3, 50)
BOUNDS = {"xi": (1.05, 60.0), "N": (5.0, 2000.0), "gamma_gap": (0.01, 30.0), "rho_bar": (-0.95, 0.95)}
RHO = (-0.95, 0.95)


@pytest.fixture
def primitives():
    return ModelPrimitives(alpha_support=SUPPORT, a=1 / 3, phi_v=0.5, n_firms_cs=100.0, H=40,
                           active_window=(2015, 2019))


@pytest.fixture
def params():
    return PooledParams(xi=6.0, N=20.0, gamma=7.0, eta=1.2, rho_bar=-0.2, a=1 / 3, H=40, phi_v=0.5)


def test_common_alpha_is_harmonic_mean_below_cap():
    gamma = 7.0
    cap = alpha_cap(gamma, 0.99)
    kept = SUPPORT[SUPPORT < cap]
    assert common_alpha(SUPPORT, gamma, 0.99) == pytest.approx(1 / np.mean(1 / kept), rel=1e-15)
    assert common_alpha(SUPPORT, gamma, 0.99) < cap
    assert common_alpha(SUPPORT, 3.0, 0.99) == pytest.approx(1 / np.mean(1 / SUPPORT), rel=1e-15)


def test_baseline_inputs_are_unchanged(primitives, params):
    prim, par = economy_inputs(ECONOMIES["baseline"], primitives, params, alpha_cap_factor=0.99)
    assert prim is primitives and par is params


@pytest.mark.parametrize("label", ["hom_alpha", "hom_v", "hom_both"])
def test_restricted_draw_has_one_common_value(label, primitives, params):
    economy = ECONOMIES[label]
    prim, par = economy_inputs(economy, primitives, params, alpha_cap_factor=0.99)
    assert par.rho_bar == 0.0
    draw = draw_pool(prim.alpha_support, xi=par.xi, rho_bar=par.rho_bar, N=par.N, M=30, H=par.H,
                     rng=1, v_min=3.0, alpha_max=alpha_cap(par.gamma, 0.99))
    mask = draw.active_mask
    assert (np.unique(draw.alpha[mask]).size == 1) == economy.common_alpha
    assert (np.unique(draw.v[mask]).size == 1) == economy.common_v
    if economy.common_v:
        assert np.isinf(par.xi) and np.all(draw.v == 3.0)
    if economy.common_alpha:
        assert np.all(draw.alpha == common_alpha(SUPPORT, par.gamma, 0.99))


@pytest.mark.parametrize("label", list(ECONOMIES))
def test_encode_decode_round_trip(label, primitives, params):
    free = ECONOMIES[label].free
    values = {"xi": 6.0, "N": 20.0, "gamma": 7.0, "rho_bar": -0.2}
    x = encode(values, free, eta=1.2, rho_bar_bounds=RHO)
    out = decode(x, free, values, eta=1.2, rho_bar_bounds=RHO, primitives=primitives)
    for name in free:
        assert getattr(out, name) == pytest.approx(values[name], rel=1e-12)
    lo, hi = coordinate_bounds(free, BOUNDS)
    assert lo.shape == hi.shape == (len(free),) and np.all(lo < hi)


def test_checkpoint_replays_and_refuses_another_run(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    ck = Checkpoint(path, fingerprint({"run": 1}), {"run": 1})
    x = np.array([0.1, np.log(20.0)])
    ck.append({"x_hex": [float(v).hex() for v in x], "residuals_hex": [float(0.25).hex()], "status": "ok"})
    with path.open("a") as fh:
        fh.write('{"x_hex": ["cut')       # a line cut by a kill is skipped
    again = Checkpoint(path, fingerprint({"run": 1}), {"run": 1})
    assert again.get(x)["status"] == "ok"
    assert again.get(np.nextafter(x, np.inf)) is None   # replay keys on the exact bits
    assert json.loads(path.read_text().splitlines()[0])["run"] == 1
    with pytest.raises(RuntimeError, match="another run"):
        Checkpoint(path, fingerprint({"run": 2}), {"run": 2})


def test_scan_points_grid():
    points = scan_points({"xi": [3.0, 5.0], "N": [10.0, 20.0, 40.0]}, ("xi", "N", "gamma"),
                         {"xi": 6.0, "N": 20.0, "gamma": 7.0})
    assert len(points) == 6 and all(p["gamma"] == 7.0 for p in points)
    with pytest.raises(ValueError, match="not free"):
        scan_points({"rho_bar": [0.0]}, ("N", "gamma"), {"N": 20.0, "gamma": 7.0})
