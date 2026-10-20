"""Alternative alpha<->v arrangements of a calibrated :class:`PoolDraw`.

Ported from ``04_Code/counterfactuals/scalability_sorting/permute.py`` and
``04_Code/counterfactuals/homogeneous_rts/homogenize.py``.

``draw_pool`` links alpha and v in one place only (the copula ladder), so the
sorting of scalability onto capability can be changed in two ways:

* **Permutations** re-assign alpha among each sector's active firms and leave
  ``v`` untouched. Both marginals stay bitwise identical; only the pairing
  moves. Modes: ``identity``, ``shuffle``, ``reverse`` (largest alpha with
  smallest v) and ``sorted`` (largest alpha with largest v). These attain the
  minimal and maximal within-sector rank correlation.
* **Copula re-draws** call ``draw_pool`` with another ``rho_bar`` and the same
  seed. The RNG order makes alpha, tilde_alpha, eps and active_mask common
  random numbers; v changes and its marginal is equal in distribution only.
  See :func:`assert_copula_crn`.

The homogeneous-alpha arrangements remove within-sector alpha dispersion:
:func:`homogenize_alpha_sector` collapses each sector to its harmonic mean
(preserving mean 1/alpha, hence mean marginal-cost curvature), and
:func:`homogenize_alpha_cost_weighted` uses one economy-wide alpha, the
cost-weighted harmonic mean at the baseline market allocation.

``tilde_alpha`` (alpha's copula rank score, only read by ``build_firm_panel``)
moves in lockstep with alpha under every permutation; it is NaN under
homogenization, where alpha has no rank.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ...model.pool import PoolDraw

PERMUTE_MODES = ("identity", "shuffle", "reverse", "sorted")
PARTIAL_BASE_MODES = ("identity", "reverse", "sorted")


def active_corr_av(draw: PoolDraw) -> float:
    """Pearson corr(alpha, v) over the active firms of every sector (pooled)."""
    mask = np.asarray(draw.active_mask, dtype=bool)
    a = np.asarray(draw.alpha, dtype=np.float64)[mask]
    v = np.asarray(draw.v, dtype=np.float64)[mask]
    # ptp, not std: the std of identical floats can round to a tiny nonzero value.
    if a.size < 2 or np.ptp(a) == 0.0 or np.ptp(v) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, v)[0, 1])


def within_sector_rank_corr(draw: PoolDraw) -> float:
    """Mean over sectors (with at least two active firms) of Spearman corr(alpha, v).

    Ties in alpha (repeated support nodes, the cap) take average ranks. Used by
    the tests to check that ``sorted`` and ``reverse`` are the extremes.
    """
    from scipy.stats import rankdata

    mask = np.asarray(draw.active_mask, dtype=bool)
    out = []
    for m in range(mask.shape[0]):
        active = np.flatnonzero(mask[m])
        if active.size < 2:
            continue
        ra = rankdata(draw.alpha[m, active])
        rv = rankdata(draw.v[m, active])
        if np.std(ra) == 0.0 or np.std(rv) == 0.0:
            continue
        out.append(np.corrcoef(ra, rv)[0, 1])
    return float(np.mean(out)) if out else float("nan")


def _gen(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def permute_alpha(
    draw: PoolDraw,
    mode: str,
    rng: np.random.Generator | int | list[int] | None = None,
) -> PoolDraw:
    """Return a new PoolDraw with alpha (and tilde_alpha) re-assigned to v.

    ``identity`` returns the input unchanged. ``shuffle`` draws a random
    permutation of alpha among each sector's active firms. ``reverse`` pairs
    the largest alpha with the smallest v within each sector and ``sorted``
    the largest alpha with the largest v. ``v`` and ``active_mask`` are never
    touched, and each sector's active-alpha multiset is preserved exactly
    (asserted).
    """
    if mode not in PERMUTE_MODES:
        raise ValueError(f"mode must be one of {PERMUTE_MODES}; got {mode!r}")
    if mode == "identity":
        return draw
    gen = _gen(rng)
    alpha_base = np.asarray(draw.alpha, dtype=np.float64)
    tilde_base = np.asarray(draw.tilde_alpha, dtype=np.float64)
    v = np.asarray(draw.v, dtype=np.float64)
    mask = np.asarray(draw.active_mask, dtype=bool)
    alpha = alpha_base.copy()
    tilde = tilde_base.copy()
    for m in range(alpha.shape[0]):
        active = np.flatnonzero(mask[m])
        if active.size <= 1:
            continue
        if mode == "shuffle":
            dest = active
            src = gen.permutation(active)
        else:
            # Destinations by v ascending; sources by alpha ascending (sorted)
            # or descending (reverse). tilde_alpha follows its own alpha.
            dest = active[np.argsort(v[m, active], kind="stable")]
            src = active[np.argsort(alpha_base[m, active], kind="stable")]
            if mode == "reverse":
                src = src[::-1]
        alpha[m, dest] = alpha_base[m, src]
        tilde[m, dest] = tilde_base[m, src]
    assert_marginals_preserved(alpha, alpha_base, v, draw.v, mask)
    return dataclasses.replace(draw, alpha=alpha, tilde_alpha=tilde)


def partial_permute(
    draw: PoolDraw,
    base_mode: str,
    q: float,
    rng: np.random.Generator | int | list[int] | None = None,
) -> PoolDraw:
    """Shuffle round(q*n) active slots within each sector of an extreme arrangement.

    ``base_mode`` is ``identity``, ``reverse`` or ``sorted``. Alpha and its rank
    score move together; inactive slots, v and participation are unchanged.
    ``q = 0`` returns the base arrangement exactly.
    """
    if base_mode not in PARTIAL_BASE_MODES:
        raise ValueError(f"base_mode must be one of {PARTIAL_BASE_MODES}; got {base_mode!r}")
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be finite and in [0, 1]")
    base = permute_alpha(draw, base_mode)
    gen = _gen(rng)
    alpha, tilde = base.alpha.copy(), base.tilde_alpha.copy()
    for m, row_mask in enumerate(base.active_mask):
        active = np.flatnonzero(row_mask)
        n = round(q * active.size)
        if n < 2:
            continue
        dest = gen.choice(active, size=n, replace=False)
        src = gen.permutation(dest)
        alpha[m, dest] = base.alpha[m, src]
        tilde[m, dest] = base.tilde_alpha[m, src]
    assert_marginals_preserved(alpha, draw.alpha, base.v, draw.v, draw.active_mask)
    return dataclasses.replace(base, alpha=alpha, tilde_alpha=tilde)


def assert_marginals_preserved(
    alpha: np.ndarray,
    alpha_base: np.ndarray,
    v: np.ndarray,
    v_base: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Raise unless v is bitwise unchanged and every sector keeps its active-alpha multiset."""
    if not np.array_equal(np.asarray(v, dtype=np.float64), np.asarray(v_base, dtype=np.float64)):
        raise AssertionError("permutation changed v; marginal not preserved")
    mask = np.asarray(mask, dtype=bool)
    for m in range(alpha.shape[0]):
        active = np.flatnonzero(mask[m])
        if active.size and not np.array_equal(
            np.sort(alpha[m, active]), np.sort(alpha_base[m, active])
        ):
            raise AssertionError(f"permutation changed the active-alpha multiset in sector {m}")


def assert_copula_crn(draw: PoolDraw, base: PoolDraw) -> None:
    """Raise unless a copula re-draw shares alpha, tilde_alpha and active_mask bitwise."""
    for name in ("alpha", "tilde_alpha", "active_mask"):
        if not np.array_equal(getattr(draw, name), getattr(base, name)):
            raise AssertionError(f"copula re-draw broke common random numbers: {name}")


def homogenize_alpha_sector(draw: PoolDraw) -> PoolDraw:
    """Collapse alpha within each sector to the harmonic mean of its active firms.

    Preserves each sector's mean 1/alpha exactly. Singleton sectors keep their
    alpha bitwise; inactive slots are left as drawn (never solver-consumed).
    """
    alpha_base = np.asarray(draw.alpha, dtype=np.float64)
    mask = np.asarray(draw.active_mask, dtype=bool)
    alpha = alpha_base.copy()
    for m in range(alpha.shape[0]):
        active = np.flatnonzero(mask[m])
        if active.size <= 1:
            continue
        alpha[m, active] = 1.0 / np.mean(1.0 / alpha_base[m, active])
    new = dataclasses.replace(draw, alpha=alpha, tilde_alpha=np.full_like(alpha, np.nan))
    _assert_homogenized(new, draw, sector_mean=True)
    return new


def cost_weighted_alpha(alpha: np.ndarray, cost: np.ndarray, mask: np.ndarray) -> float:
    """Economy-wide cost-weighted harmonic mean ``(sum_i c_i / alpha_i)^-1`` over active firms."""
    mask = np.asarray(mask, dtype=bool)
    c = np.asarray(cost, dtype=np.float64)[mask]
    a = np.asarray(alpha, dtype=np.float64)[mask]
    return float(1.0 / np.sum((c / np.sum(c)) / a))


def homogenize_alpha_cost_weighted(draw: PoolDraw, alpha_hom: float) -> PoolDraw:
    """Set every active firm's alpha to one economy-wide value ``alpha_hom``.

    ``alpha_hom`` is :func:`cost_weighted_alpha` at the baseline market
    allocation, which holds the aggregate marginal-cost elasticity fixed.
    """
    alpha = np.asarray(draw.alpha, dtype=np.float64).copy()
    mask = np.asarray(draw.active_mask, dtype=bool)
    alpha[mask] = float(alpha_hom)
    new = dataclasses.replace(draw, alpha=alpha, tilde_alpha=np.full_like(alpha, np.nan))
    _assert_homogenized(new, draw, sector_mean=False)
    return new


def _assert_homogenized(new: PoolDraw, base: PoolDraw, *, sector_mean: bool) -> None:
    if not np.array_equal(new.v, base.v):
        raise AssertionError("homogenization changed v")
    if not np.array_equal(new.active_mask, base.active_mask):
        raise AssertionError("homogenization changed active_mask")
    mask = np.asarray(new.active_mask, dtype=bool)
    for m in range(mask.shape[0]):
        active = np.flatnonzero(mask[m])
        if active.size == 0:
            continue
        if np.ptp(new.alpha[m, active]) != 0.0:
            raise AssertionError(f"homogenization left alpha dispersion in sector {m}")
        if sector_mean and not np.isclose(
            np.mean(1.0 / new.alpha[m, active]), np.mean(1.0 / base.alpha[m, active]),
            rtol=1e-12, atol=0.0,
        ):
            raise AssertionError(f"homogenization changed mean(1/alpha) in sector {m}")
