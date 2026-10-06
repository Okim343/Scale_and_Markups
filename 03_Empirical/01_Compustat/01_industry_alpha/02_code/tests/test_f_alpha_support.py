"""Unit tests for the Stage S5 alpha-support construction order.

The single standard is winsorize -> mean-preserving shrink -> clip. The shipped
`alpha` column is the final model-facing support (sampled verbatim by the model).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]


def _load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s5 = _load_module("05_build_F_alpha.py", "s5_build_f_alpha")
LOG = logging.getLogger("s5-test")
LOG.addHandler(logging.NullHandler())

LO, HI = 0.6, 1.20


def _cfg(variant="rts_shrunk", lam=0.55):
    return {"F_alpha": {
        "n_nodes": 500, "winsor_pct": [0.01, 0.99], "alpha_clip": [LO, HI],
        "min_obs": 100, "support_variant": variant, "shrink_lambda": lam,
    }}


def _alpha():
    rng = np.random.default_rng(0)
    return rng.normal(0.8, 0.3, 5000)


def test_shrink_then_clip_order_and_columns():
    nodes = s5.build_pooled_nodes(_alpha(), _cfg(), LOG)
    assert {"alpha_raw", "alpha_shrunk", "alpha"}.issubset(nodes.columns)
    raw = nodes["alpha_raw"].to_numpy(float)
    shr = nodes["alpha_shrunk"].to_numpy(float)
    fin = nodes["alpha"].to_numpy(float)

    # shrink is mean-preserving on the pre-clip node support, and compresses it
    assert shr.mean() == pytest.approx(raw.mean(), abs=1e-3)
    assert shr.std() < raw.std()

    # final support = clip(shrunk) to [0.6, 1.20] — clip is applied AFTER shrink
    np.testing.assert_allclose(fin, np.clip(shr, LO, HI), atol=1e-5)
    assert fin.min() >= LO - 1e-6 and fin.max() <= HI + 1e-6


def test_lower_clip_bound_is_0p6():
    nodes = s5.build_pooled_nodes(_alpha(), _cfg(), LOG)
    # nothing below 0.6 survives; some mass reaches the 0.6 floor for this draw
    assert nodes["alpha"].min() == pytest.approx(LO, abs=1e-6)


def test_full_compustat_variant_skips_shrink():
    nodes = s5.build_pooled_nodes(_alpha(), _cfg(variant="full_compustat"), LOG)
    np.testing.assert_allclose(
        nodes["alpha_shrunk"].to_numpy(float), nodes["alpha_raw"].to_numpy(float), atol=1e-6
    )
