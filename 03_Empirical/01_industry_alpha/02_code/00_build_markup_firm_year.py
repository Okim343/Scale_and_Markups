"""
Build the normalized Compustat firm-year markup panel used by S1.

This script makes explicit the upstream normalization that produced
00_indata/03_Compustat/markup_firm_year.dta.

Inputs:
  - data_main_upd_trim_1.dta: curated firm-year Compustat base panel
  - theta/markup source: updated theta output keyed by (gvkey, year)

Output:
  - markup_firm_year.dta with the canonical broad firm-year schema expected by
    01_load_compustat.py.
  - optional QJE-cost-share-trimmed aggregate markup files matching
    03_Calculate_Markups.do.

Run examples:
  python 00_build_markup_firm_year.py --dry-run
  python 00_build_markup_firm_year.py --theta-source ../00_indata/03_Compustat/theta_unified_time.dta --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import PATHS, setup_logger


COMPUSTAT_DIR = PATHS.indata / "03_Compustat"
DEFAULT_BASE = COMPUSTAT_DIR / "data_main_upd_trim_1.dta"
DEFAULT_OUTPUT = COMPUSTAT_DIR / "markup_firm_year.dta"

THETA_CANDIDATES = [
    COMPUSTAT_DIR / "theta_unified_time.dta",
    COMPUSTAT_DIR / "theta_unified_const.dta",
    COMPUSTAT_DIR / "ACF_theta_ALLsectors.dta",
    COMPUSTAT_DIR / "markup_firm_year.dta",
]

CANONICAL_COLUMNS = [
    "gvkey",
    "year",
    "sale",
    "cogs",
    "xsga",
    "xlr",
    "xrd",
    "xad",
    "dvt",
    "ppegt",
    "emp",
    "mkvalt",
    "ind2d",
    "ind3d",
    "ind4d",
    "USGDP",
    "usercost",
    "sale_D",
    "cogs_D",
    "xsga_D",
    "mkvalt_D",
    "dividend_D",
    "capital_D",
    "xlr_D",
    "kexp",
    "s_g",
    "theta_WI1_ct",
    "theta_WI2_ct",
    "theta_WI2_xt",
    "theta_WI1_kt",
    "theta_WI2_kt",
    "markup",
    "markup_xsga",
    "total_sales_year",
    "weight_s",
]

THETA_COLUMNS = [
    "theta_WI1_ct",
    "theta_WI2_ct",
    "theta_WI2_xt",
    "theta_WI1_kt",
    "theta_WI2_kt",
]

OPTIONAL_MERGE_COLUMNS = [
    *THETA_COLUMNS,
    "markup",
    "markup_xsga",
    "total_sales_year",
    "weight_s",
]

RENAME_MAP = {
    "fyear": "year",
    "naics2": "ind2d",
    "real_sale": "sale_D",
    "real_cogs": "cogs_D",
    "real_xsga": "xsga_D",
    "real_ppegt": "capital_D",
    "gdp_def2009": "USGDP",
    "user_cost_capital": "usercost",
    "markups": "markup",
    "weights": "weight_s",
}

NUMERIC_COLUMNS = [c for c in CANONICAL_COLUMNS if c != "gvkey"]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input type: {path}")


def normalize_gvkey(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    out = raw.copy()
    mask = numeric.notna()
    out.loc[mask] = numeric.loc[mask].astype("Int64").astype(str).str.zfill(6)
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.astype(str)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {old: new for old, new in RENAME_MAP.items() if old in out.columns and new not in out.columns}
    out = out.rename(columns=rename)
    if "gvkey" in out.columns:
        out["gvkey"] = normalize_gvkey(out["gvkey"])
    if "year" in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out["year"]):
            out["year"] = out["year"].dt.year
        else:
            out["year"] = pd.to_numeric(out["year"], errors="coerce")
    if "ind2d" in out.columns:
        out["ind2d"] = pd.to_numeric(out["ind2d"], errors="coerce")
    return out


def load_base(path: Path) -> pd.DataFrame:
    df = normalize_columns(read_table(path))
    missing = [c for c in CANONICAL_COLUMNS[:26] if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing base columns: {missing}")
    require_unique_keys(df, path.name)
    return df[CANONICAL_COLUMNS[:26]].copy()


def apply_qje_costshare_trim(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Apply the Stage-3 QJE cost-share trim from 03_Calculate_Markups.do.

    Stata logic:
      costshare1 = cogs_D/(cogs_D+kexp)
      costshare2 = cogs_D/(cogs_D+xsga_D+kexp)
      for s in 1/2:
        bysort year: egen p1/p99 = pctile(costshare`s'), p(1/99)
        drop if costshare`s'==0 | costshare`s'==.
        drop if costshare`s' > p99
        drop if costshare`s' < p1
    """
    out = df.copy()
    n0 = len(out)
    out["costshare1"] = out["cogs_D"] / (out["cogs_D"] + out["kexp"])
    out["costshare2"] = out["cogs_D"] / (out["cogs_D"] + out["xsga_D"] + out["kexp"])

    for col in ["costshare1", "costshare2"]:
        before = len(out)
        bounds = (
            out.groupby("year")[col]
            .quantile([0.01, 0.99])
            .unstack()
            .rename(columns={0.01: "p1", 0.99: "p99"})
        )
        out = out.merge(bounds, left_on="year", right_index=True, how="left")
        keep = (
            out[col].notna()
            & out[col].ne(0)
            & out[col].le(out["p99"])
            & out[col].ge(out["p1"])
        )
        out = out.loc[keep].drop(columns=["p1", "p99"]).copy()
        logger.info(
            f"QJE trim {col}: dropped {before - len(out):,d} rows "
            f"({100.0 * (before - len(out)) / before:.2f}%)"
        )

    out = out.drop(columns=["costshare1", "costshare2"])
    logger.info(f"QJE cost-share trim total: {n0:,d} -> {len(out):,d} rows")
    return out


def choose_theta_source(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg).expanduser()
        if not path.is_absolute():
            path = (PATHS.code / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for path in THETA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "no theta source found; pass --theta-source with an updated theta/markup file"
    )


def load_theta(path: Path) -> pd.DataFrame:
    df = normalize_columns(read_table(path))
    if {"gvkey", "year"}.issubset(df.columns):
        keys = ["gvkey", "year"]
    elif {"ind2d", "year"}.issubset(df.columns):
        keys = ["ind2d", "year"]
    else:
        raise KeyError(
            f"{path.name} must contain either gvkey-year or ind2d-year keys"
        )
    available = [c for c in OPTIONAL_MERGE_COLUMNS if c in df.columns]
    missing_theta = [c for c in THETA_COLUMNS if c not in available]
    if missing_theta:
        raise KeyError(f"{path.name} missing theta columns: {missing_theta}")
    out = df[[*keys, *available]].copy()
    require_unique_keys(out, path.name, keys=keys)
    out.attrs["merge_keys"] = keys
    return out


def require_unique_keys(
    df: pd.DataFrame, name: str, *, keys: list[str] | tuple[str, ...] = ("gvkey", "year")
) -> None:
    key_list = list(keys)
    if df[key_list].isna().any().any():
        raise ValueError(f"{name} has missing key values for {key_list}")
    dup = int(df.duplicated(key_list).sum())
    if dup:
        raise ValueError(f"{name} has {dup:,d} duplicate rows for keys {key_list}")


def build_panel(base: pd.DataFrame, theta: pd.DataFrame) -> pd.DataFrame:
    merge_keys = theta.attrs.get("merge_keys", ["gvkey", "year"])
    validate = "one_to_one" if merge_keys == ["gvkey", "year"] else "many_to_one"
    df = base.merge(theta, on=merge_keys, how="left", validate=validate)
    matched = df[THETA_COLUMNS].notna().any(axis=1)

    if "markup" not in df.columns:
        df["markup"] = np.nan
    markup_missing = df["markup"].isna() & matched
    if markup_missing.any():
        df.loc[markup_missing, "markup"] = (
            df.loc[markup_missing, "theta_WI1_ct"]
            * (df.loc[markup_missing, "sale_D"] / df.loc[markup_missing, "cogs_D"])
        )
    if "markup_xsga" not in df.columns:
        df["markup_xsga"] = np.nan
    markup_xsga_missing = df["markup_xsga"].isna() & matched
    if markup_xsga_missing.any():
        # This follows the existing panel's convention: theta_WI2_ct divided by
        # the COGS/sales share, not by (COGS+XSGA)/sales.
        df.loc[markup_xsga_missing, "markup_xsga"] = (
            df.loc[markup_xsga_missing, "theta_WI2_ct"]
            * (df.loc[markup_xsga_missing, "sale_D"] / df.loc[markup_xsga_missing, "cogs_D"])
        )

    # Recompute these from the selected row universe. Older carried values can
    # be based on a different vintage/window and may include Stata-incompatible
    # infinities in early years.
    df["total_sales_year"] = df.groupby("year")["sale_D"].transform("sum")
    df["weight_s"] = np.where(df["total_sales_year"] > 0, df["sale_D"] / df["total_sales_year"], np.nan)

    df = df[CANONICAL_COLUMNS].copy()
    cast_types(df)
    validate_panel(df)
    return df


def cast_types(df: pd.DataFrame) -> None:
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["ind2d", "ind3d", "ind4d"]:
        if col in df.columns:
            df[col] = df[col].astype("Int32")
    float32_cols = [
        "sale_D",
        "cogs_D",
        "xsga_D",
        "mkvalt_D",
        "dividend_D",
        "capital_D",
        "xlr_D",
        "kexp",
        "s_g",
        *THETA_COLUMNS,
        "markup",
        "markup_xsga",
        "total_sales_year",
        "weight_s",
    ]
    for col in float32_cols:
        df[col] = df[col].astype("float32")


def validate_panel(df: pd.DataFrame) -> None:
    require_unique_keys(df, "rebuilt markup_firm_year")
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"rebuilt panel missing columns: {missing}")
    if df[["sale", "cogs"]].isna().any().any():
        raise ValueError("rebuilt panel has missing sale or cogs")
    if (df["sale"] <= 0).any() or (df["cogs"] <= 0).any():
        raise ValueError("rebuilt panel has non-positive sale or cogs")
    if (df["markup"].dropna() <= 0).any() or (df["markup_xsga"].dropna() <= 0).any():
        raise ValueError("rebuilt panel has non-positive markups")
    numeric = df.select_dtypes(include=[np.number])
    infs = np.isinf(numeric).sum()
    bad = {str(k): int(v) for k, v in infs.items() if int(v)}
    if bad:
        raise ValueError(f"rebuilt panel has infinite numeric values: {bad}")


def weighted_mean(group: pd.DataFrame, value: str, weight: str) -> float:
    weights = group[weight].astype("float64")
    values = group[value].astype("float64")
    wsum = weights.sum(skipna=True)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.nan
    return float((values * weights).sum(skipna=True) / wsum)


def build_sales_weighted(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.groupby("year", sort=True):
        rows.append(
            {
                "year": year,
                "markup": weighted_mean(group, "markup", "weight_s"),
                "markup_xsga": weighted_mean(group, "markup_xsga", "weight_s"),
            }
        )
    return pd.DataFrame(rows)


def build_cost_weighted(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["total_cost"] = work["cogs_D"] + work["kexp"] + work["xsga_D"]
    work["total_cost_year"] = work.groupby("year")["total_cost"].transform("sum")
    work["weight_c"] = np.where(
        work["total_cost_year"] > 0,
        work["total_cost"] / work["total_cost_year"],
        np.nan,
    )
    rows = []
    for year, group in work.groupby("year", sort=True):
        rows.append(
            {
                "year": year,
                "markup": weighted_mean(group, "markup", "weight_c"),
                "markup_xsga": weighted_mean(group, "markup_xsga", "weight_c"),
            }
        )
    return pd.DataFrame(rows)


def build_percentiles(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.sort_values(["year", "markup"]).groupby("year", sort=True):
        g = group[["markup", "weight_s"]].dropna().copy()
        if g.empty:
            rows.append(
                {
                    "year": year,
                    "markup_mean": np.nan,
                    "markup_p50": np.nan,
                    "markup_p75": np.nan,
                    "markup_p90": np.nan,
                }
            )
            continue
        wsum = g["weight_s"].sum()
        weights = g["weight_s"] / wsum if wsum > 0 else g["weight_s"]
        cumshare = weights.cumsum()

        def weighted_pct(cutoff: float) -> float:
            hit = g.loc[cumshare >= cutoff, "markup"]
            return float(hit.min()) if not hit.empty else np.nan

        rows.append(
            {
                "year": year,
                "markup_mean": float((g["markup"] * weights).sum()) if wsum > 0 else np.nan,
                "markup_p50": weighted_pct(0.50),
                "markup_p75": weighted_pct(0.75),
                "markup_p90": weighted_pct(0.90),
            }
        )
    return pd.DataFrame(rows)


def write_stata_checked(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_stata(path, write_index=False, version=118)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="curated base .dta/.parquet/.csv")
    parser.add_argument("--theta-source", default=None, help="theta/markup source keyed by gvkey-year")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output markup_firm_year.dta path")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    parser.add_argument("--dry-run", action="store_true", help="build and validate but do not write")
    parser.add_argument(
        "--costshare-trim-firm-year",
        action="store_true",
        help=(
            "write a QJE-cost-share-trimmed firm-year file. This is not the "
            "default because GNR and S1 need the broad firm-year universe; "
            "the output filename must contain 'qje', 'costshare', or 'trim'."
        ),
    )
    parser.add_argument(
        "--write-aggregates",
        action="store_true",
        help="also write markup_sales_weighted.dta, markup_cost_weighted.dta, and markup_percentiles.dta",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("BUILD_MARKUP_FIRM_YEAR", "INFO")

    base_path = Path(args.base).expanduser()
    if not base_path.is_absolute():
        base_path = (PATHS.code / base_path).resolve()
    theta_path = choose_theta_source(args.theta_source)
    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (PATHS.code / out_path).resolve()

    logger.info(f"base source : {base_path}")
    logger.info(f"theta source: {theta_path}")
    logger.info(f"output      : {out_path}")

    base = load_base(base_path)
    theta = load_theta(theta_path)
    if args.costshare_trim_firm_year:
        name = out_path.name.lower()
        if not any(token in name for token in ("qje", "costshare", "trim")):
            raise ValueError(
                "--costshare-trim-firm-year requires an explicit trimmed output "
                "filename containing 'qje', 'costshare', or 'trim'. Do not "
                "overwrite the canonical broad markup_firm_year.dta with the "
                "QJE aggregation-trimmed universe."
            )
        logger.warning(
            "writing a QJE-cost-share-trimmed firm-year file by explicit request"
        )
        panel_base = apply_qje_costshare_trim(base, logger)
    else:
        panel_base = base
    panel = build_panel(panel_base, theta)

    logger.info(
        f"rebuilt shape: {panel.shape}; years {int(panel['year'].min())}-{int(panel['year'].max())}; "
        f"firms {panel['gvkey'].nunique():,d}"
    )
    logger.info(
        "markup p1/p50/p99: "
        f"{panel['markup'].quantile(0.01):.4f} / "
        f"{panel['markup'].quantile(0.50):.4f} / "
        f"{panel['markup'].quantile(0.99):.4f}"
    )

    if args.write_aggregates:
        logger.info(
            "building aggregate markup files on the QJE cost-share-trimmed "
            "row universe only"
        )
        agg_base = apply_qje_costshare_trim(base, logger)
        agg_panel = build_panel(agg_base, theta)
        sw = build_sales_weighted(agg_panel)
        cw = build_cost_weighted(agg_panel)
        pct = build_percentiles(agg_panel)
        logger.info(
            "aggregate preview: "
            f"sales-weighted years {int(sw['year'].min())}-{int(sw['year'].max())}; "
            f"cost-weighted active mean 2010-2019="
            f"{cw.loc[(cw['year'] >= 2010) & (cw['year'] <= 2019), 'markup'].mean():.4f}"
        )

    if args.dry_run:
        logger.info("dry run requested; not writing output")
        return 0
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_stata(out_path, write_index=False, version=118)
    logger.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    if args.write_aggregates:
        aggregate_paths = {
            "markup_sales_weighted.dta": sw,
            "markup_cost_weighted.dta": cw,
            "markup_percentiles.dta": pct,
        }
        for name, data in aggregate_paths.items():
            path = out_path.parent / name
            write_stata_checked(data, path, args.overwrite)
            logger.info(f"wrote {path} ({path.stat().st_size / 1e3:.2f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
