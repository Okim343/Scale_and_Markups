"""Pooled bundle loader.

Reads the three artifacts produced by the natively pooled empirical pipeline
(``03_Empirical/01_industry_alpha/03_outdata/``):

* ``F_alpha.parquet``        — a SINGLE pooled scalability distribution
  (N equal-mass nodes; no sector dimension).
* ``aggregate_moments.yaml`` — all pooled scalar targets (mu_cw, cr4, cr20,
  top1pct, top5pct, emx_slope), the pooled capital share ``a``, the pooled
  firm count ``n_firms_cs``, the Gaussian-copula ρ̄, and validation moments.
* ``manifest.yaml``          — provenance + input/output hashes.

There is no ``sector_table`` and no per-sector α grid: the empirical pipeline
now emits pooled scalars directly, so the loader no longer re-pools anything.
The asserts below pin the pooled contract so a stale or partial bundle fails
fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .model.pricing import DEFAULT_PHI_V


# Single pooled F(α) node table. No sector_id.
F_ALPHA_REQUIRED_COLS = ("node_id", "u", "weight", "alpha")

# Pooled scalar targets that must be present under aggregate_moments["targets"].
REQUIRED_TARGET_KEYS = (
    "mu_cw", "cr4", "cr20", "top1pct", "top5pct", "emx_slope",
)

# Top-level pooled scalars the solver reads straight out of aggregate_moments.
# The model capital share is the top-level KLEMS VA-weighted `a` (= a_bar),
# consistent with the hybrid alpha (see Bundle.a). The GNR estimate
# gnr_elasticities.a_gnr is a diagnostic only and is no longer required.
REQUIRED_SCALAR_KEYS = ("a", "n_firms_cs", "alpha_z_copula_rho_bar", "alpha_sales_corr_empirical")


@dataclass(frozen=True)
class Bundle:
    """Natively pooled calibration bundle in memory.

    Holds the pooled α support as a plain array plus the raw F(α) frame and the
    parsed YAML mappings. All model-facing quantities are pooled scalars.
    """

    f_alpha: pd.DataFrame
    alpha_support: np.ndarray
    aggregate_moments: Mapping[str, Any]
    manifest: Mapping[str, Any]
    bundle_dir: Path

    @property
    def active_window(self) -> tuple[int, int]:
        w = self.aggregate_moments["active_window"]
        return int(w["start"]), int(w["end"])

    @property
    def targets(self) -> dict[str, float]:
        """Pooled calibration target scalars, keyed by MODEL moment names.

        The empirical bundle stores the accounting markup under the data key
        ``mu_cw`` (= aggregate revenue / variable cost = cost-weighted μ/α). The
        model names that object ``mu_cw_alpha``; the unqualified ``mu_cw`` now
        denotes the PURE cost-weighted markup. Remap at this boundary so the rest
        of the code speaks only model names — the on-disk data keys are unchanged.
        """
        tgt = self.aggregate_moments["targets"]
        out = {k: float(tgt[k]) for k in REQUIRED_TARGET_KEYS if k != "mu_cw"}
        out["mu_cw_alpha"] = float(tgt["mu_cw"])
        return out

    @property
    def a(self) -> float:
        """Pooled capital share of value added — the top-level KLEMS ``a`` (a_bar).

        Reads ``aggregate_moments["a"]``, the KLEMS VA-weighted aggregate
        capital share. This is the hybrid-consistent model capital share: the
        hybrid alpha imputes labor from ``a_sector`` (KLEMS), so the aggregate
        capital share the model uses is the KLEMS ``a_bar``, NOT the old GNR
        estimate ``gnr_elasticities.a_gnr`` (which was contaminated by the GNR
        ``lelast`` and is now kept only as a diagnostic).
        """
        if "a" not in self.aggregate_moments:
            raise KeyError(
                "aggregate_moments.yaml missing top-level 'a' — the model capital "
                "share is the KLEMS VA-weighted a_bar. Rerun the empirical "
                "pipeline's 08_assemble_bundle.py to export it."
            )
        val = float(self.aggregate_moments["a"])
        if not (0.0 < val < 1.0):
            raise ValueError(f"capital share a must lie in (0, 1); got {val}")
        return val

    @property
    def n_firms_cs(self) -> float:
        """Pooled total firm count (population-scaling input)."""
        return float(self.aggregate_moments["n_firms_cs"])

    @property
    def mean_firms_per_sector(self) -> float:
        """Sector-sales-weighted firm count analogue for model ``N``."""
        if "mean_firms_per_sector" not in self.aggregate_moments:
            n_markets = self.aggregate_moments.get("n_markets")
            if n_markets is None:
                return self.n_firms_cs
            return self.n_firms_cs / float(n_markets)
        return float(self.aggregate_moments["mean_firms_per_sector"])

    @property
    def phi_v(self) -> float:
        r"""Externally-assigned gross-output value-added weight $\phi_v$.

        Read from ``aggregate_moments["phi_v"]`` when the empirical pipeline
        exports it (``phi_v = 1 - melast/rts`` from the Hubmer/GNR run); falls
        back to :data:`steady_state.model.pricing.DEFAULT_PHI_V` so existing
        bundles still load. Not part of the calibrated vector; validated in
        ``(0, 1]``.
        """
        val = float(self.aggregate_moments.get("phi_v", DEFAULT_PHI_V))
        if not (0.0 < val <= 1.0):
            raise ValueError(f"phi_v must lie in (0, 1]; got {val}")
        return val

    @property
    def reference_scale(self) -> dict[str, Any]:
        """Optional retired operating-scale metadata from aggregate_moments.yaml."""
        block = self.aggregate_moments.get("reference_scale")
        if not isinstance(block, Mapping):
            return {}
        return deepcopy(dict(block))

    @property
    def rho_bar(self) -> float:
        r"""Fixed, ex-ante Gaussian-copula α–z rank correlation $\bar\rho$.

        Read from ``aggregate_moments["alpha_z_copula_rho_bar"]``. It is the
        correlation between the α-rank score and the standardized productivity
        ladder ``ρ̄·tilde_alpha + √(1−ρ̄²)·ε`` (model/pool.py); not calibrated.
        Validated to lie in [0, 1).
        """
        if "alpha_z_copula_rho_bar" not in self.aggregate_moments:
            raise KeyError(
                "aggregate_moments.yaml missing 'alpha_z_copula_rho_bar' — the "
                "Gaussian-copula ρ̄ is a required external input. Rerun the "
                "empirical pipeline's 08_assemble_bundle.py (it copies "
                "implied_gaussian_copula_rho from the Hubmer-RTS "
                "alpha_omega_corr.json), or set alpha_z_copula.enabled: false."
            )
        val = float(self.aggregate_moments["alpha_z_copula_rho_bar"])
        if not (0.0 <= val < 1.0):
            raise ValueError(f"alpha_z_copula_rho_bar must lie in [0, 1); got {val}")
        return val

    @property
    def alpha_sales_corr_empirical(self) -> float | None:
        """Empirical corr(alpha, log sales), distinct from ``rho_bar``.

        Read from ``aggregate_moments["alpha_sales_corr_empirical"]`` (Stage
        S6b, ``06b_alpha_sales_corr_hybrid.py``): the pooled Pearson
        correlation between the hybrid alpha and log(real sales) in the same
        Compustat/GNR panel used for ``rho_bar`` — NOT the alpha-productivity
        copula correlation. This is the valid empirical target for the
        model's simulated ``corr_alpha_log_sales`` moment; ``rho_bar`` never
        was (see ``02_Drafts/md_files/scale_issue_logs.md``, 2026-07-06).
        Returns ``None`` on an older bundle that predates this stage.
        """
        if "alpha_sales_corr_empirical" not in self.aggregate_moments:
            return None
        return float(self.aggregate_moments["alpha_sales_corr_empirical"])

    @property
    def alpha_sales_corr_partial_empirical(self) -> float | None:
        """Empirical corr(alpha, log sales | omega) — the partial correlation
        controlling for omega (the empirical stand-in for the model's z).

        Read from ``aggregate_moments["alpha_sales_corr_partial_empirical"]``
        (Stage S6b). This is the valid empirical target for the model's
        simulated ``corr_alpha_log_sales_partial_z`` moment
        (calibration anchor diagnostics) — comparing partial
        to partial, unlike the raw ``alpha_sales_corr_empirical`` above or
        ``rho_bar``, neither of which is on the same conditioning basis as
        the model's own partial correlation. Returns ``None`` on an older
        bundle that predates this stage.
        """
        if "alpha_sales_corr_partial_empirical" not in self.aggregate_moments:
            return None
        return float(self.aggregate_moments["alpha_sales_corr_partial_empirical"])

    @property
    def target_mu_cw_alpha(self) -> float:
        """Empirical accounting markup (data key ``mu_cw`` = revenue / variable cost)."""
        return float(self.aggregate_moments["targets"]["mu_cw"])

    @property
    def target_mu_cw_sga(self) -> float:
        """SG&A-inclusive cost-weighted aggregate markup (theta_WI2 basis).

        This is the primary markup target for the model's pure-markup aggregate
        ``mu_cw``. The accounting/welfare object ``mu_cw_alpha`` (= aggregate
        revenue / aggregate variable cost) is reported separately and is not
        remapped to this target. Raises if the empirical bundle predates the
        SG&A export.
        """
        tgt = self.aggregate_moments["targets"]
        if "mu_cw_sga" not in tgt:
            raise KeyError(
                "aggregate_moments.yaml targets lack 'mu_cw_sga' — rerun the "
                "empirical pipeline (07_aggregate_moments.py / 08_assemble_bundle.py) "
                "or set calibration.markup_target: cogs."
            )
        return float(tgt["mu_cw_sga"])

    def calibration_targets(self, *, markup_target: str = "sga") -> dict[str, float]:
        """Pooled calibration targets with the markup cost base selected.

        The calibration markup moment is ``mu_cw``, the pure firm-markup
        aggregate. ``mu_cw_alpha`` is retained in the returned mapping as the
        empirical COGS accounting diagnostic, matching the model-side accounting
        object name (aggregate revenue / true variable cost). ``markup_target``
        selects which empirical value ``mu_cw`` is fit to: ``"sga"`` (SG&A-
        inclusive ≈1.18, the primary target) or ``"cogs"`` (COGS-only ≈1.44).
        All other targets are unchanged.
        """
        if markup_target not in {"sga", "cogs"}:
            raise ValueError(
                f"markup_target must be 'sga' or 'cogs'; got {markup_target!r}"
            )
        targets = dict(self.targets)
        targets["mu_cw"] = (
            self.target_mu_cw_sga if markup_target == "sga" else self.target_mu_cw_alpha
        )
        return targets

    @property
    def target_emx_slope(self) -> float:
        """Pooled EMX-style slope target b = -(1/η - 1/γ) (calibration input)."""
        return float(self.aggregate_moments["targets"]["emx_slope"])

    @property
    def target_slope_validation(self) -> float:
        """Firm-level cross-sectional log-log markup-share slope (validation only)."""
        val = self.aggregate_moments.get("validation", {})
        if "markup_share_slope_loglog" in val:
            return float(val["markup_share_slope_loglog"])
        return float("nan")


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_bundle(bundle_dir: Path | str) -> Bundle:
    """Read the three pooled bundle files; validate the columns/scalar contract.

    Raises ``FileNotFoundError`` if any artifact is missing, ``ValueError`` if
    the schema, node table, or required scalars deviate from the pooled
    contract.
    """
    bundle_dir = Path(bundle_dir).resolve()

    f_alpha_path = bundle_dir / "F_alpha.parquet"
    agg_moments_path = bundle_dir / "aggregate_moments.yaml"
    manifest_path = bundle_dir / "manifest.yaml"

    for p in (f_alpha_path, agg_moments_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"Bundle artifact missing: {p}")

    f_alpha = pd.read_parquet(f_alpha_path)
    aggregate_moments = _load_yaml(agg_moments_path)
    manifest = _load_yaml(manifest_path)

    # --- F_alpha contract: single pooled distribution -------------------------
    missing = [c for c in F_ALPHA_REQUIRED_COLS if c not in f_alpha.columns]
    if missing:
        raise ValueError(f"F_alpha.parquet missing columns: {missing}")
    if "sector_id" in f_alpha.columns:
        raise ValueError(
            "F_alpha.parquet carries a 'sector_id' column — this loader expects "
            "a SINGLE pooled distribution (no sector dimension). Rebuild the "
            "bundle with the pooled pipeline (Stage S5)."
        )
    if len(f_alpha) < 1:
        raise ValueError("F_alpha.parquet is empty")
    f_alpha = f_alpha.copy().sort_values("node_id").reset_index(drop=True)
    alpha_support = f_alpha["alpha"].to_numpy(dtype=np.float64)

    # --- aggregate_moments contract: pooled scalars --------------------------
    for k in ("active_window", "targets", "validation"):
        if k not in aggregate_moments:
            raise ValueError(f"aggregate_moments.yaml missing key: {k}")
    missing_t = [k for k in REQUIRED_TARGET_KEYS if k not in aggregate_moments["targets"]]
    if missing_t:
        raise ValueError(f"aggregate_moments.yaml: targets missing {missing_t}")
    missing_s = [k for k in REQUIRED_SCALAR_KEYS if k not in aggregate_moments]
    if missing_s:
        raise ValueError(f"aggregate_moments.yaml: top-level scalars missing {missing_s}")
    # Model capital share is the top-level KLEMS VA-weighted `a` (a_bar); the GNR
    # a_gnr is an optional diagnostic (see Bundle.a).
    a_val = float(aggregate_moments["a"])
    if not (0.0 < a_val < 1.0):
        raise ValueError(f"aggregate_moments.yaml: top-level 'a' must lie in (0, 1); got {a_val}")

    return Bundle(
        f_alpha=f_alpha,
        alpha_support=alpha_support,
        aggregate_moments=aggregate_moments,
        manifest=manifest,
        bundle_dir=bundle_dir,
    )
