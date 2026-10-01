"""Stable pooled-input interface layered over the transitional bundle loader."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .io_bundle import Bundle
from .model.pricing import DEFAULT_PHI_V


@dataclass(frozen=True)
class PooledInputs:
    targets: dict[str, float]
    rho_bar: float = 0.50
    alpha_sales_corr_empirical: float | None = None
    alpha_sales_corr_partial_empirical: float | None = None
    a: float = 1.0 / 3.0
    eta_init: float = 2.0
    H: int = 300
    alpha_support: np.ndarray | None = None
    n_firms_cs: float = 1_000.0
    empirical_N: float | None = None
    phi_v: float = DEFAULT_PHI_V
    markup_target: str = "sga"
    alpha_support_variant: str = "rts_shrunk"
    alpha_support_shrink_lambda: float = 0.371

    @classmethod
    def from_bundle(
        cls,
        bundle: Bundle,
        *,
        eta_init: float = 2.0,
        H: int = 300,
        markup_target: str = "sga",
    ) -> "PooledInputs":
        """Copy the already-pooled scalars and support from the bundle.

        The empirical pipeline now emits pooled targets, the pooled capital
        share ``a``, the pooled firm count, the externally-assigned value-added
        weight ``phi_v``, and a single pooled ``alpha_support`` directly — so
        this is a straight copy with no re-pooling.

        ``markup_target`` selects which empirical value the pure-markup moment
        ``mu_cw`` is fit to: ``"sga"`` (SG&A-inclusive ≈1.18, the primary
        target) or ``"cogs"`` (COGS-only ≈1.44). The model's ``mu_cw_alpha``
        remains the accounting object aggregate revenue / true variable cost.

        The RTS-dispersion shrink and clip are now applied in the empirical
        pipeline (Stage S5, winsorize -> shrink -> clip): ``bundle.alpha_support``
        IS the final model-facing support and is sampled verbatim — the loader
        performs no support transform. The shrink variant/λ the bundle was built
        with are read back from ``aggregate_moments["hybrid_alpha"]`` for
        provenance only.
        """
        support = np.asarray(bundle.alpha_support, dtype=float)
        hybrid = bundle.aggregate_moments.get("hybrid_alpha") or {}
        return cls(
            targets=bundle.calibration_targets(markup_target=markup_target),
            rho_bar=bundle.rho_bar,
            alpha_sales_corr_empirical=bundle.alpha_sales_corr_empirical,
            alpha_sales_corr_partial_empirical=bundle.alpha_sales_corr_partial_empirical,
            a=bundle.a,
            eta_init=eta_init,
            H=H,
            alpha_support=support,
            n_firms_cs=bundle.n_firms_cs,
            empirical_N=bundle.mean_firms_per_sector,
            phi_v=bundle.phi_v,
            markup_target=markup_target,
            alpha_support_variant=str(hybrid.get("support_variant", "rts_shrunk")),
            alpha_support_shrink_lambda=float(hybrid.get("shrink_lambda", 0.371)),
        )


def f_alpha_sampler(rng, size, *, alpha_support: np.ndarray | None = None) -> np.ndarray:
    """Draw from the pooled scalability support; uniform stub if omitted."""
    gen = np.random.default_rng(rng)
    support = (
        np.asarray(alpha_support, dtype=float)
        if alpha_support is not None
        else np.linspace(0.6, 1.20, 500)
    )
    return gen.choice(support, size=size, replace=True)


def exposure_weights(Q: int, *, uniform: bool = True, rng=None) -> np.ndarray:
    """Return market exposure weights; empirical sampling is intentionally deferred."""
    if Q <= 0:
        raise ValueError("Q must be positive")
    if not uniform:
        raise NotImplementedError("empirical exposure sampling is deferred")
    return np.full(Q, 1.0 / Q)


def planner_selection_stub(*args, **kwargs):
    raise NotImplementedError(
        "planner social-surplus selection deferred (berry_entry section 3.1)"
    )
