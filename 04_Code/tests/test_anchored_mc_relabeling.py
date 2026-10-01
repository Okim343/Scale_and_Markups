"""Legacy oracle for the anchored-MC relabeling migration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from steady_state.model.market import solve
from steady_state.model.market_batch import solve_batch


FIXTURE = Path(__file__).with_name("fixtures") / "legacy_anchored_mc_fixture.npz"


def _common_kwargs(data):
    return dict(
        eta=float(data["eta"]),
        gamma=float(data["gamma"]),
        w=float(data["w"]),
        R=float(data["R"]),
        X_market=float(data["X_market"]),
        a_i=float(data["a_i"]),
        phi_v=float(data["phi_v"]),
        tol=float(data["tol"]),
        max_iter=int(data["max_iter"]),
        damping=float(data["damping"]),
        newton_fallback_after=int(data["newton_fallback_after"]),
    )


def _v_from_legacy_z(alpha, z, *, x_ref, y_hat):
    return (alpha / x_ref) * z ** (1.0 / alpha) * y_hat ** (1.0 - 1.0 / alpha)


def _legacy_z_from_v(alpha, v, *, x_ref, y_hat):
    return (x_ref * v / alpha) ** alpha * y_hat ** (1.0 - alpha)


def test_anchored_scalar_solver_matches_legacy_fixture_under_relabeling():
    data = np.load(FIXTURE)
    x_ref = float(data["x_ref"])

    for y_hat in data["y_hat_values"]:
        v = _v_from_legacy_z(data["alpha"], data["z"], x_ref=x_ref, y_hat=float(y_hat))
        scalar = solve(
            n=data["alpha"].size,
            alpha=data["alpha"],
            v=v,
            y_hat=float(y_hat),
            **_common_kwargs(data),
        )
        assert scalar.converged
        np.testing.assert_allclose(scalar.y, data["scalar_y"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scalar.p, data["scalar_p"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scalar.s, data["scalar_s"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scalar.mu, data["scalar_mu"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            scalar.p * scalar.y,
            data["scalar_sales"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(scalar.cost, data["scalar_cost"], rtol=1e-12, atol=1e-12)


def test_anchored_batch_solvers_match_legacy_fixture_under_relabeling():
    data = np.load(FIXTURE)
    x_ref = float(data["x_ref"])

    for y_hat in data["y_hat_values"]:
        v = _v_from_legacy_z(
            data["alpha_batch"],
            data["z_batch"],
            x_ref=x_ref,
            y_hat=float(y_hat),
        )
        for backend in ("scalar", "vectorized"):
            batch = solve_batch(
                data["alpha_batch"],
                v,
                active_mask=data["active_mask"],
                backend=backend,
                y_hat=float(y_hat),
                **_common_kwargs(data),
            )
            assert np.all(batch.converged)
            prefix = f"batch_{backend}"
            np.testing.assert_allclose(batch.output, data[f"{prefix}_output"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(batch.price, data[f"{prefix}_price"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(batch.s, data[f"{prefix}_s"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(batch.mu, data[f"{prefix}_mu"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(batch.sales, data[f"{prefix}_sales"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(batch.cost, data[f"{prefix}_cost"], rtol=1e-12, atol=1e-12)


def test_relabeling_identity_round_trips_for_non_unit_anchors():
    data = np.load(FIXTURE)
    x_ref = float(data["x_ref"])

    for y_hat in data["y_hat_values"]:
        v = _v_from_legacy_z(data["alpha"], data["z"], x_ref=x_ref, y_hat=float(y_hat))
        z_round_trip = _legacy_z_from_v(data["alpha"], v, x_ref=x_ref, y_hat=float(y_hat))
        np.testing.assert_allclose(z_round_trip, data["z"], rtol=1e-14, atol=1e-14)

        v_batch = _v_from_legacy_z(
            data["alpha_batch"],
            data["z_batch"],
            x_ref=x_ref,
            y_hat=float(y_hat),
        )
        z_batch_round_trip = _legacy_z_from_v(
            data["alpha_batch"],
            v_batch,
            x_ref=x_ref,
            y_hat=float(y_hat),
        )
        np.testing.assert_allclose(
            z_batch_round_trip,
            data["z_batch"],
            rtol=1e-14,
            atol=1e-14,
        )
