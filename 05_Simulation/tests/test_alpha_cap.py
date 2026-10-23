"""The gamma-dependent alpha cap: helper, draw, config default and params-file check."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from conftest import SIM_DIR
from orbis_model.config import load_config
from orbis_model.model.pool import DEFAULT_ALPHA_CAP_FACTOR, alpha_cap, draw_pool
from orbis_model.params import check_alpha_cap_factor

SUPPORT = np.linspace(0.7, 1.3, 50)


def test_alpha_cap_is_factor_times_min_markup():
    assert alpha_cap(5.0, 0.99) == pytest.approx(0.99 * 5.0 / 4.0)
    assert alpha_cap(5.0) == pytest.approx(DEFAULT_ALPHA_CAP_FACTOR * 1.25)
    assert alpha_cap(5.0, None) is None
    for bad in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError):
            alpha_cap(5.0, bad)
    with pytest.raises(ValueError):
        alpha_cap(1.0, 0.99)


def test_cap_truncates_without_mass_point():
    kw = dict(xi=8.0, rho_bar=0.5, N=6.0, M=400, H=12, rng=3)
    raw = draw_pool(SUPPORT, **kw)
    cap = alpha_cap(7.25, 0.99)
    capped = draw_pool(SUPPORT, alpha_max=cap, **kw)
    assert raw.alpha.max() > cap  # the cap binds on this support
    assert capped.alpha.max() < cap  # truncation, not clipping: nothing sits at the cap
    kept = SUPPORT[SUPPORT < cap]
    assert set(np.unique(capped.alpha)) <= set(kept)
    assert capped.alpha.max() == kept.max()
    # tilde_alpha is the rank within the truncated support, still standard normal
    assert abs(capped.tilde_alpha.mean()) < 0.05 and abs(capped.tilde_alpha.std() - 1.0) < 0.05
    np.testing.assert_array_equal(capped.active_mask, raw.active_mask)


def test_draw_pool_rejects_cap_below_support():
    with pytest.raises(ValueError, match="alpha_max"):
        draw_pool(SUPPORT, xi=8.0, rho_bar=0.0, N=3.0, M=2, H=4, rng=0, alpha_max=0.5)


def test_draw_pool_rejects_nonpositive_alpha_max():
    with pytest.raises(ValueError, match="alpha_max"):
        draw_pool(SUPPORT, xi=8.0, rho_bar=0.0, N=3.0, M=2, H=4, rng=0, alpha_max=0.0)


def test_shipped_config_sets_the_cap():
    assert load_config().parameters.alpha_cap_factor == 0.99


def test_config_cap_default_null_and_bounds(tmp_path):
    raw = yaml.safe_load((SIM_DIR / "config_simulation.yaml").read_text())
    raw["paths"]["bundle_dir"] = str(tmp_path)
    path = tmp_path / "cfg.yaml"
    del raw["parameters"]["alpha_cap_factor"]
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).parameters.alpha_cap_factor == DEFAULT_ALPHA_CAP_FACTOR
    raw["parameters"]["alpha_cap_factor"] = None
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).parameters.alpha_cap_factor is None
    raw["parameters"]["alpha_cap_factor"] = 1.5
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="alpha_cap_factor"):
        load_config(path)


def test_params_file_cap_factor_must_match_config():
    check_alpha_cap_factor({"model": {"alpha_cap_factor": 0.99}}, 0.99)
    check_alpha_cap_factor({}, None)  # pre-cap file: raw support
    with pytest.raises(ValueError, match="alpha_cap_factor"):
        check_alpha_cap_factor({"model": {"alpha_cap_factor": 0.99}}, 0.95)
    with pytest.raises(ValueError, match="alpha_cap_factor"):
        check_alpha_cap_factor({}, 0.99)
