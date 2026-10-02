"""Marginal-preserving permutation of the alpha<->v assignment in a PoolDraw.

The alpha-v sorting is a property of the *joint* assignment, not of the solver
(``pool.py:117`` is the only place alpha and v are linked). A post-hoc
permutation of the calibrated :class:`~steady_state.model.pool.PoolDraw` changes
only the joint and leaves both marginals bit-identical -- a pure sorting
experiment under common random numbers. See ``IMPLEMENTATION_PLAN.md`` sec. 2-3.

``steady_state/`` is untouched; this module only imports ``PoolDraw``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from steady_state.model.pool import PoolDraw

_MODES = ("identity", "shuffle", "reverse")


def active_corr_av(draw: PoolDraw) -> float:
    """Pearson corr(alpha, v) over the active firms of every sector (pooled).

    Diagnostic that confirms the sorting lever fired: ~0.596 for the calibrated
    baseline, ~0 for a shuffle, <0 for reverse.
    """
    mask = np.asarray(draw.active_mask, dtype=bool)
    a = np.asarray(draw.alpha, dtype=np.float64)[mask]
    v = np.asarray(draw.v, dtype=np.float64)[mask]
    if a.size < 2 or np.std(a) == 0.0 or np.std(v) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, v)[0, 1])


def permute_alpha(
    draw: PoolDraw,
    mode: str,
    rng: np.random.Generator | int | None = None,
) -> PoolDraw:
    """Return a new PoolDraw with ``alpha`` (and ``tilde_alpha``) re-assigned to ``v``.

    ``mode="identity"``  -> baseline; the input draw is returned unchanged.
    ``mode="shuffle"``   -> random permutation of alpha among each sector's ACTIVE
                            firms (corr(alpha, v) ~ 0).
    ``mode="reverse"``   -> negative-assortative: within each sector's active
                            firms the largest alpha pairs with the smallest v.

    ``v`` and ``active_mask`` are untouched; ``tilde_alpha`` (alpha's rank score,
    consumed by ``build_firm_panel``) carries the *identical* reordering as alpha
    so the firm panel stays consistent. Permutation is within-sector and over
    active firms only -- the active-alpha multiset per sector is preserved
    exactly, so the marginals (and hence the anchored calibration) are unchanged.

    Runtime assertions fail loudly if any sector's active-alpha multiset or the
    entire ``v`` array is not bit-identical to the input.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")
    if mode == "identity":
        return draw

    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    alpha_base = np.asarray(draw.alpha, dtype=np.float64)
    tilde_base = np.asarray(draw.tilde_alpha, dtype=np.float64)
    v = np.asarray(draw.v, dtype=np.float64)
    mask = np.asarray(draw.active_mask, dtype=bool)
    M, _H = alpha_base.shape

    alpha = alpha_base.copy()
    tilde = tilde_base.copy()

    for m in range(M):
        active = np.nonzero(mask[m])[0]
        if active.size <= 1:
            continue
        if mode == "shuffle":
            dest = active
            src = gen.permutation(active)
        else:  # reverse
            # destinations ordered by v ascending; sources by alpha descending,
            # so the smallest-v slot receives the largest alpha (and lockstep
            # tilde_alpha, which is monotone in alpha).
            dest = active[np.argsort(v[m, active], kind="stable")]
            src = active[np.argsort(alpha_base[m, active], kind="stable")[::-1]]
        alpha[m, dest] = alpha_base[m, src]
        tilde[m, dest] = tilde_base[m, src]

    _assert_marginals_preserved(alpha, alpha_base, v, draw.v, mask)

    return dataclasses.replace(draw, alpha=alpha, tilde_alpha=tilde)


def _assert_marginals_preserved(
    alpha: np.ndarray,
    alpha_base: np.ndarray,
    v: np.ndarray,
    v_base: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not np.array_equal(v, np.asarray(v_base, dtype=np.float64)):
        raise AssertionError("permute_alpha changed v; marginal not preserved")
    M = alpha.shape[0]
    for m in range(M):
        active = np.nonzero(mask[m])[0]
        if active.size == 0:
            continue
        if not np.array_equal(
            np.sort(alpha[m, active]), np.sort(alpha_base[m, active])
        ):
            raise AssertionError(
                f"permute_alpha changed the active-alpha multiset in sector {m}"
            )


def partial_permute(draw: PoolDraw, base_mode: str, q: float,
                    rng: np.random.Generator | int | None = None) -> PoolDraw:
    """Shuffle round(q*n) active slots within each sector of identity/reverse.

    Alpha and its rank score move together. Inactive slots, capability, and
    participation are unchanged. q=0 exactly returns the chosen base assignment.
    """
    if base_mode not in ("identity", "reverse"):
        raise ValueError("base_mode must be identity or reverse")
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be finite and in [0, 1]")
    base = permute_alpha(draw, base_mode)
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    alpha, tilde = base.alpha.copy(), base.tilde_alpha.copy()
    for m, mask in enumerate(base.active_mask):
        active = np.flatnonzero(mask)
        n = round(q * len(active))
        if n < 2:
            continue
        dest = gen.choice(active, size=n, replace=False)
        src = gen.permutation(dest)
        alpha[m, dest] = base.alpha[m, src]
        tilde[m, dest] = base.tilde_alpha[m, src]
    _assert_marginals_preserved(alpha, draw.alpha, base.v, draw.v, draw.active_mask)
    return dataclasses.replace(base, alpha=alpha, tilde_alpha=tilde)
