"""Step 05: where the planner expands input demand (descriptive, no re-solve).

Ported from ``04_Code/counterfactuals/planner_expansion/run_planner_expansion.py``
(cell tables, sector fixed-effects regression and alpha-support summary reused
unchanged). Reads the baseline firm panels written by ``02_welfare.py`` (market,
free planner) and ``03_fixed_input_welfare.py`` (fixed-capital planner PE_K and
fixed-input planner PE_I) and asks which firms absorb the planner's extra inputs.

Within a regime every input (k, l, m) is a scalar fraction of firm total cost
``TC = alpha * sales / mu``, so input shares are TC shares and the firm's input
expansion is ``TC_planner / TC_market``.

Validation runs first and aborts on failure:

* free planner: ``sum TC_PE / sum TC_ME == K_PE / K_ME`` (same R);
* PE_K and PE_I: ``sum TC_pl / sum TC_ME == R_pl / R_ME`` (same K);
* corr(alpha, v) over active firms equals the sorting-sweep baseline value;
* the market cost-weighted markup equals ``welfare_<window>.yaml: mu_bar``.

Reads   <welfare-dir>/{sim_panel_market,sim_panel_planner,welfare}_<window>.*
        <fixed-input-dir>/{sim_panel_planner_fixed_k,sim_panel_planner_fixed_input,
                           fixed_input_regimes,fixed_input_welfare}_<window>.*
Writes  <out-dir>/planner_expansion_cells.parquet, planner_expansion.yaml, results.md
(defaults: <welfare-dir> = out_results, <fixed-input-dir> = out_results/fixed_input,
 <out-dir> = out_results/planner_expansion)

    python 05_planner_expansion.py [--window 2015_2019] [--corr-target X] [--skip-fixed-input]
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from orbis_model.params import write_yaml as _write_yaml

KEYS = ["market_id", "firm_id"]
N_Q = 5
# corr(alpha, v) over active firms of the calibrated M = 10000 draw
# (out_results/sorting/sorting_economies.parquet, label baseline).
CORR_AV_TARGET = -0.247519
CORR_AV_TOL = 5e-6
RATIO_REL_TOL = 1e-6
AUTO_MARKER = "<!-- AUTO-GENERATED TABLES BELOW: do not edit by hand -->"
FIXED_PLANNERS = {"fixed_k": ("sim_panel_planner_fixed_k", "planner_fixed_k"),
                  "fixed_input": ("sim_panel_planner_fixed_input", "planner_fixed_input")}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--welfare-dir", default="out_results")
    p.add_argument("--fixed-input-dir", default="out_results/fixed_input")
    p.add_argument("--out-dir", default="out_results/planner_expansion")
    p.add_argument("--window", default="2015_2019")
    p.add_argument("--corr-target", type=float, default=CORR_AV_TARGET)
    p.add_argument("--skip-fixed-input", action="store_true",
                   help="free planner only (skip PE_K and PE_I)")
    return p


def _read_active(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns + ["active"], filters=[("active", "==", True)])
    return df.drop(columns="active")


def load_panel(welfare_dir: Path, fixed_dir: Path | None, tag: str) -> pd.DataFrame:
    """One row per active firm, market and planner outcomes side by side."""
    mkt = _read_active(welfare_dir / f"sim_panel_market_{tag}.parquet",
                       KEYS + ["alpha", "v", "markup", "cost", "sales", "share"])
    pl = _read_active(welfare_dir / f"sim_panel_planner_{tag}.parquet",
                      KEYS + ["alpha", "v", "cost", "sales", "share"])
    df = mkt.merge(pl, on=KEYS, how="outer", suffixes=("", "_pl"),
                   validate="one_to_one", indicator=True)
    if not (df["_merge"] == "both").all():
        raise SystemExit(f"market/planner active sets differ: {df['_merge'].value_counts().to_dict()}")
    for col in ("alpha", "v"):
        if not np.array_equal(df[col].to_numpy(), df[f"{col}_pl"].to_numpy()):
            raise SystemExit(f"{col} differs between market and planner panels")
    df = df.drop(columns=["_merge", "alpha_pl", "v_pl"]).rename(columns={
        "cost": "tc_mkt", "sales": "sales_mkt", "share": "share_mkt",
        "markup": "mu_mkt", "cost_pl": "tc_pl", "sales_pl": "sales_pl", "share_pl": "share_pl",
    })
    if fixed_dir is not None:
        for name, (stem, _) in FIXED_PLANNERS.items():
            fk = _read_active(fixed_dir / f"{stem}_{tag}.parquet", KEYS + ["alpha", "cost", "sales", "share"])
            fk = fk.rename(columns={"cost": f"tc_{name}", "sales": f"sales_{name}",
                                    "share": f"share_{name}", "alpha": f"alpha_{name}"})
            df = df.merge(fk, on=KEYS, how="left", validate="one_to_one")
            if df[f"tc_{name}"].isna().any():
                raise SystemExit(f"{name} planner active set differs from market")
            if not np.array_equal(df["alpha"].to_numpy(), df[f"alpha_{name}"].to_numpy()):
                raise SystemExit(f"alpha differs between market and {name} panels")
            df = df.drop(columns=f"alpha_{name}")
    return df.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled and within-sector quintiles plus within-sector size rank."""
    n = len(df)
    by_mkt = df.groupby("market_id", sort=False)
    n_mkt = by_mkt["firm_id"].transform("size").to_numpy()
    df["log_v"] = np.log(df["v"])
    df["log_mu_mkt"] = np.log(df["mu_mkt"])
    for col, tag in (("alpha", "a"), ("v", "v"), ("mu_mkt", "mu")):
        r_pool = df[col].rank(method="first").to_numpy()
        r_with = by_mkt[col].rank(method="first").to_numpy()
        df[f"q_{tag}_pooled"] = np.ceil(N_Q * r_pool / n).astype(int)
        df[f"q_{tag}_within"] = np.ceil(N_Q * r_with / n_mkt).astype(int)
        df[f"pct_{tag}_within"] = r_with / n_mkt
    rank = by_mkt["sales_mkt"].rank(ascending=False, method="first").to_numpy()
    df["size_rank"] = np.select([rank == 1, rank == 2, rank <= 5],
                                ["1_leader", "2_second", "3_third_fifth"], default="4_rest")
    return df


def validate(df: pd.DataFrame, welfare_dir: Path, fixed_dir: Path | None, tag: str,
             corr_target: float) -> dict:
    welfare = yaml.safe_load(open(welfare_dir / f"welfare_{tag}.yaml"))
    regimes = pd.read_parquet(welfare_dir / f"welfare_decomposition_{tag}.parquet").set_index("regime")
    k_ratio = regimes.loc["planner", "K"] / regimes.loc["market", "K"]
    tc_ratio = df["tc_pl"].sum() / df["tc_mkt"].sum()
    corr_av = float(np.corrcoef(df["alpha"], df["v"])[0, 1])
    mu_cw = float((df["mu_mkt"] * df["tc_mkt"]).sum() / df["tc_mkt"].sum())
    out = {
        "n_active": int(len(df)), "n_markets": int(df["market_id"].nunique()),
        "free_planner": {"tc_ratio": float(tc_ratio), "K_ratio": float(k_ratio),
                         "rel_gap": float(tc_ratio / k_ratio - 1.0)},
        "corr_alpha_v": {"value": corr_av, "target": corr_target, "abs_gap": abs(corr_av - corr_target)},
        "mu_cw_market": {"value": mu_cw, "welfare_yaml_mu_bar": welfare["mu_bar"],
                         "abs_gap": abs(mu_cw - welfare["mu_bar"])},
    }
    checks = [abs(out["free_planner"]["rel_gap"]) < RATIO_REL_TOL,
              out["corr_alpha_v"]["abs_gap"] < CORR_AV_TOL,
              out["mu_cw_market"]["abs_gap"] < 1e-9]
    if fixed_dir is not None:
        fixed = yaml.safe_load(open(fixed_dir / f"fixed_input_welfare_{tag}.yaml"))
        for name, (_, regime) in FIXED_PLANNERS.items():
            ratio = df[f"tc_{name}"].sum() / df["tc_mkt"].sum()
            r_ratio = fixed[f"R_{regime}"] / fixed["R_market"]
            out[name] = {"tc_ratio": float(ratio), "R_ratio": float(r_ratio),
                         "rel_gap": float(ratio / r_ratio - 1.0)}
            # K is pinned to solver tolerance (gap <= 1e-4), so allow that here.
            checks.append(abs(out[name]["rel_gap"]) < 1e-4)
    out["passed"] = bool(all(checks))
    return out


# --------------------------------------------------------------------------
# Cell tables
# --------------------------------------------------------------------------

def cell_table(df: pd.DataFrame, by: list[str], cols: dict[str, str],
               table: str, scheme: str, planner: str) -> pd.DataFrame:
    """Aggregate firm outcomes into cells defined by ``by``.

    ``cols`` maps ``tc``/``sales``/``share`` to the planner-regime columns.
    ``wmean_log_mu_mkt`` and ``wmean_log_tc_ratio`` are TC_mkt-weighted means of
    firm logs; since ``log(TC_pl/TC_mkt) = log(mu_mkt) + log(sales_pl/sales_mkt)``
    their ratio is the part of the cell's log expansion that is the direct
    removal of the wedge.
    """
    log_ratio = np.log(df[cols["tc"]] / df["tc_mkt"])
    tmp = df.assign(_mu_tc=df["mu_mkt"] * df["tc_mkt"], _tc_pl=df[cols["tc"]],
                    _sales_pl=df[cols["sales"]], _share_pl=df[cols["share"]],
                    _lmu_tc=df["log_mu_mkt"] * df["tc_mkt"],
                    _lr_tc=log_ratio * df["tc_mkt"])
    g = tmp.groupby(by, sort=True)
    agg = g.agg(
        n_firms=("firm_id", "size"),
        mean_alpha=("alpha", "mean"),
        mean_log_v=("log_v", "mean"),
        mean_pct_alpha_within=("pct_a_within", "mean"),
        mean_pct_v_within=("pct_v_within", "mean"),
        tc_mkt=("tc_mkt", "sum"),
        tc_pl=("_tc_pl", "sum"),
        mu_tc=("_mu_tc", "sum"),
        lmu_tc=("_lmu_tc", "sum"),
        lr_tc=("_lr_tc", "sum"),
        sales_mkt=("sales_mkt", "sum"),
        sales_pl=("_sales_pl", "sum"),
        mean_share_mkt=("share_mkt", "mean"),
        mean_share_pl=("_share_pl", "mean"),
    ).reset_index()
    tot_mkt, tot_pl = tmp["tc_mkt"].sum(), tmp["_tc_pl"].sum()
    agg["firm_share"] = agg["n_firms"] / len(tmp)
    agg["mu_cw_mkt"] = agg["mu_tc"] / agg["tc_mkt"]
    agg["share_tc_mkt"] = agg["tc_mkt"] / tot_mkt
    agg["share_tc_pl"] = agg["tc_pl"] / tot_pl
    agg["tc_ratio"] = agg["tc_pl"] / agg["tc_mkt"]
    agg["gain_share"] = (agg["tc_pl"] - agg["tc_mkt"]) / (tot_pl - tot_mkt)
    agg["sales_ratio"] = agg["sales_pl"] / agg["sales_mkt"]
    agg["wmean_log_mu_mkt"] = agg["lmu_tc"] / agg["tc_mkt"]
    agg["wmean_log_tc_ratio"] = agg["lr_tc"] / agg["tc_mkt"]
    agg = agg.drop(columns=["mu_tc", "lmu_tc", "lr_tc"])

    rename = {}
    for i, col in enumerate(by):
        rename[col] = f"bin{i + 1}"
        agg[f"var{i + 1}"] = col
    agg = agg.rename(columns=rename)
    if len(by) == 1:
        agg["var2"], agg["bin2"] = "", ""
    agg["bin1"] = agg["bin1"].astype(str)
    agg["bin2"] = agg["bin2"].astype(str)
    agg.insert(0, "planner", planner)
    agg.insert(0, "scheme", scheme)
    agg.insert(0, "table", table)
    return agg


def build_cells(df: pd.DataFrame, planners: dict[str, dict]) -> pd.DataFrame:
    frames = []
    for planner, cols in planners.items():
        for scheme in ("pooled", "within"):
            frames.append(cell_table(df, [f"q_a_{scheme}"], cols,
                                     "quintile_alpha", scheme, planner))
            frames.append(cell_table(df, [f"q_v_{scheme}"], cols,
                                     "quintile_v", scheme, planner))
            frames.append(cell_table(df, [f"q_a_{scheme}", f"q_v_{scheme}"],
                                     cols, "double_alpha_v", scheme, planner))
            frames.append(cell_table(df, [f"q_mu_{scheme}", f"q_a_{scheme}"],
                                     cols, "double_mu_alpha", scheme, planner))
        frames.append(cell_table(df, ["size_rank"], cols, "size_rank",
                                 "within", planner))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Sector fixed-effects regression
# --------------------------------------------------------------------------

def _demean(x: np.ndarray, codes: np.ndarray, w: np.ndarray) -> np.ndarray:
    num = np.bincount(codes, weights=w * x)
    den = np.bincount(codes, weights=w)
    return x - (num / den)[codes]


def fe_regression(df: pd.DataFrame, y: np.ndarray, regressors: list[str],
                  weights: np.ndarray | None) -> dict:
    """OLS of ``y`` on ``regressors`` with sector fixed effects (FWL)."""
    codes = pd.factorize(df["market_id"])[0]
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    sw = np.sqrt(w)
    Y = _demean(y, codes, w)
    X = np.column_stack([_demean(df[c].to_numpy(float), codes, w)
                         for c in regressors])

    def ols(A, b):
        return np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]

    beta = ols(X, Y)
    resid = Y - X @ beta
    ssr = float((w * resid**2).sum())
    y_bar = float((w * y).sum() / w.sum())
    r2_total = 1.0 - ssr / float((w * (y - y_bar) ** 2).sum())
    r2_within = 1.0 - ssr / float((w * Y**2).sum())

    out = {"n": int(len(df)), "r2_with_fe": r2_total, "r2_within": r2_within,
           "coef": {}, "coef_x_sd_within": {}, "partial_r2": {},
           "within_r2_alone": {}}
    for j, name in enumerate(regressors):
        others = [k for k in range(len(regressors)) if k != j]
        ry = Y - X[:, others] @ ols(X[:, others], Y)
        rx = X[:, j] - X[:, others] @ ols(X[:, others], X[:, j])
        bj = float((w * rx * ry).sum() / (w * rx * rx).sum())
        partial = 1.0 - float((w * (ry - bj * rx) ** 2).sum()) / float((w * ry**2).sum())
        b_alone = float((w * X[:, j] * Y).sum() / (w * X[:, j] ** 2).sum())
        alone = 1.0 - float((w * (Y - b_alone * X[:, j]) ** 2).sum()) / float((w * Y**2).sum())
        sd = float(np.sqrt((w * X[:, j] ** 2).sum() / w.sum()))
        out["coef"][name] = float(beta[j])
        out["coef_x_sd_within"][name] = float(beta[j]) * sd
        out["partial_r2"][name] = partial
        out["within_r2_alone"][name] = alone
    return out


def build_regressions(df: pd.DataFrame, planners: dict[str, dict]) -> dict:
    df["inv_alpha"] = 1.0 / df["alpha"]
    logit_ok = bool(((df["alpha"] > 0) & (df["alpha"] < 1)).all())
    specs = {"baseline": ["log_mu_mkt", "alpha", "log_v"],
             "inv_alpha": ["log_mu_mkt", "inv_alpha", "log_v"]}
    if logit_ok:
        df["logit_alpha"] = np.log(df["alpha"] / (1.0 - df["alpha"]))
        specs["logit_alpha"] = ["log_mu_mkt", "logit_alpha", "log_v"]

    out = {
        "label": "DESCRIPTIVE, NOT CAUSAL. log(mu_mkt) is itself an "
                 "equilibrium function of (alpha, v) through the market share.",
        "dependent": "log(TC_planner / TC_market)",
        "fixed_effects": "sector (market_id)",
        "standard_errors": "not reported: the sample is the full simulated "
                           "population of active firms, not a sample",
        "logit_alpha_note": (
            "used" if logit_ok else
            f"skipped: {int((df['alpha'] >= 1).sum())} active firms "
            f"({(df['alpha'] >= 1).mean():.4f} of the total) have alpha >= 1, "
            f"so log(alpha/(1-alpha)) is undefined; 1/alpha is used instead"),
        "mechanical_identity": "log(TC_pl/TC_mkt) = log(mu_mkt) + "
                               "log(sales_pl/sales_mkt) since TC = alpha*sales/mu "
                               "and mu_pl = 1, so the log(mu_mkt) coefficient "
                               "carries a mechanical +1",
    }
    for planner, cols in planners.items():
        y = np.log(df[cols["tc"]].to_numpy() / df["tc_mkt"].to_numpy())
        out[planner] = {}
        for spec, regs in specs.items():
            out[planner][spec] = {
                "firm_weighted": fe_regression(df, y, regs, None),
                "tc_mkt_weighted": fe_regression(df, y, regs, df["tc_mkt"].to_numpy()),
            }
        # Between- vs within-sector split of the TC-weighted dispersion of y.
        w = df["tc_mkt"].to_numpy()
        codes = pd.factorize(df["market_id"])[0]
        y_sec = (np.bincount(codes, weights=w * y) / np.bincount(codes, weights=w))[codes]
        y_bar = (w * y).sum() / w.sum()
        out[planner]["variance_split_tc_weighted"] = {
            "total": float((w * (y - y_bar) ** 2).sum() / w.sum()),
            "between_sector": float((w * (y_sec - y_bar) ** 2).sum() / w.sum()),
            "within_sector": float((w * (y - y_sec) ** 2).sum() / w.sum()),
        }
    return out


def alpha_support(df: pd.DataFrame, planners: dict[str, dict]) -> dict:
    """How much of the expansion sits on firms with alpha >= 1 (IRS at the margin)."""
    irs = df["alpha"] >= 1.0
    a_max = float(df["alpha"].max())
    at_max = df["alpha"] == a_max
    leader = df["size_rank"] == "1_leader"
    out = {
        "alpha_min": float(df["alpha"].min()), "alpha_max": a_max,
        "n_unique_alpha": int(df["alpha"].nunique()),
        "firm_share_alpha_ge_1": float(irs.mean()),
        "firm_share_at_alpha_max": float(at_max.mean()),
        "leader_share_alpha_ge_1": float(irs[leader].mean()),
        "leader_share_at_alpha_max": float(at_max[leader].mean()),
        "tc_mkt_share_alpha_ge_1": float(df.loc[irs, "tc_mkt"].sum() / df["tc_mkt"].sum()),
    }
    for planner, cols in planners.items():
        gain = df[cols["tc"]] - df["tc_mkt"]
        out[f"gain_share_alpha_ge_1_{planner}"] = float(gain[irs].sum() / gain.sum())
        out[f"tc_pl_share_alpha_ge_1_{planner}"] = float(
            df.loc[irs, cols["tc"]].sum() / df[cols["tc"]].sum())
    return out


# --------------------------------------------------------------------------
# YAML, markdown, Typst
# --------------------------------------------------------------------------

ROW_FIELDS = ["n_firms", "firm_share", "mean_alpha", "mean_log_v",
              "mean_pct_alpha_within", "mean_pct_v_within", "mu_cw_mkt",
              "share_tc_mkt", "share_tc_pl", "tc_ratio", "gain_share",
              "sales_ratio", "wmean_log_mu_mkt", "wmean_log_tc_ratio",
              "mean_share_mkt", "mean_share_pl"]


def _row_dict(row: pd.Series) -> dict:
    return {f: (int(row[f]) if f == "n_firms" else float(row[f])) for f in ROW_FIELDS}


def cells_to_yaml(cells: pd.DataFrame) -> dict:
    out: dict = {}
    for (table, scheme, planner), sub in cells.groupby(
            ["table", "scheme", "planner"], sort=False):
        node = out.setdefault(table, {}).setdefault(scheme, {}).setdefault(planner, {})
        for _, row in sub.iterrows():
            if row["var2"]:
                key = f"r{row['bin1']}_c{row['bin2']}"
            elif table == "size_rank":
                key = row["bin1"]
            else:
                key = f"Q{row['bin1']}"
            node[key] = _row_dict(row)
    return out


def _md_quintile(cells, table, scheme, planner) -> str:
    sub = cells.query("table == @table and scheme == @scheme and planner == @planner")
    head = ("| bin | firm share | mean alpha | mean ln v | mu_cw (mkt) | "
            "share TC mkt | share TC pl | TC ratio | share of gain |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    rows = "".join(
        f"| {r.bin1} | {r.firm_share:.4f} | {r.mean_alpha:.4f} | {r.mean_log_v:.4f} | "
        f"{r.mu_cw_mkt:.4f} | {r.share_tc_mkt:.4f} | {r.share_tc_pl:.4f} | "
        f"{r.tc_ratio:.4f} | {r.gain_share:.4f} |\n" for r in sub.itertuples())
    return head + rows


def _md_grid(cells, table, scheme, planner, field, fmt) -> str:
    sub = cells.query("table == @table and scheme == @scheme and planner == @planner")
    grid = sub.pivot(index="bin1", columns="bin2", values=field)
    v1, v2 = sub["var1"].iloc[0], sub["var2"].iloc[0]
    head = (f"| {v1} \\ {v2} | " + " | ".join(grid.columns) + " |\n|---|"
            + "---|" * len(grid.columns) + "\n")
    rows = "".join(f"| {idx} | " + " | ".join(fmt.format(x) for x in grid.loc[idx])
                   + " |\n" for idx in grid.index)
    return head + rows


def _md_regression(reg: dict, planner: str) -> str:
    lines = ["| spec | weights | coef ln mu | coef alpha-term | coef ln v | "
             "partial R2 ln mu | partial R2 alpha-term | partial R2 ln v | "
             "within R2 | R2 (with FE) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for spec, by_w in reg[planner].items():
        if spec == "variance_split_tc_weighted":
            continue
        for wname, r in by_w.items():
            names = list(r["coef"])
            lines.append(
                f"| {spec} | {wname} | " + " | ".join(f"{r['coef'][n]:.4f}" for n in names)
                + " | " + " | ".join(f"{r['partial_r2'][n]:.4f}" for n in names)
                + f" | {r['r2_within']:.4f} | {r['r2_with_fe']:.4f} |")
    return "\n".join(lines) + "\n"


def write_results_md(path: Path, cells: pd.DataFrame, reg: dict, valid: dict,
                     support: dict, planners: list[str]) -> None:
    parts = [AUTO_MARKER, "",
             "All numbers below are in `planner_expansion.yaml` "
             "(same table/scheme/planner/bin keys) and "
             "`planner_expansion_cells.parquet` (columns `table`, `scheme`, "
             "`planner`, `bin1`, `bin2`).", "",
             "### Validation (`planner_expansion.yaml: validation`)", "",
             "```yaml", yaml.safe_dump(valid, sort_keys=False).rstrip(), "```", "",
             "### Alpha support (`planner_expansion.yaml: alpha_support`)", "",
             "```yaml", yaml.safe_dump(support, sort_keys=False).rstrip(), "```", ""]
    for planner in planners:
        for scheme in ("pooled", "within"):
            for table in ("quintile_alpha", "quintile_v"):
                parts += [f"### `{table}.{scheme}.{planner}`", "",
                          _md_quintile(cells, table, scheme, planner)]
        for scheme in ("pooled", "within"):
            for field, fmt in (("tc_ratio", "{:.3f}"), ("gain_share", "{:.4f}"),
                               ("mu_cw_mkt", "{:.4f}"), ("firm_share", "{:.5f}")):
                parts += [f"### `double_alpha_v.{scheme}.{planner}`: {field}", "",
                          _md_grid(cells, "double_alpha_v", scheme, planner, field, fmt)]
            for field, fmt in (("tc_ratio", "{:.3f}"), ("firm_share", "{:.5f}")):
                parts += [f"### `double_mu_alpha.{scheme}.{planner}`: {field}", "",
                          _md_grid(cells, "double_mu_alpha", scheme, planner, field, fmt)]
        sub = cells.query("table == 'size_rank' and planner == @planner")
        parts += [f"### `size_rank.within.{planner}`", "",
                  "| rank | firm share | mean alpha | pct alpha (within) | "
                  "pct v (within) | mu_cw (mkt) | mean share mkt | mean share pl | "
                  "share TC mkt | share TC pl | TC ratio | share of gain | "
                  "wmean ln mu | wmean ln TC ratio |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        parts += [f"| {r.bin1} | {r.firm_share:.4f} | {r.mean_alpha:.4f} | "
                  f"{r.mean_pct_alpha_within:.3f} | {r.mean_pct_v_within:.3f} | "
                  f"{r.mu_cw_mkt:.4f} | {r.mean_share_mkt:.4f} | {r.mean_share_pl:.4f} | "
                  f"{r.share_tc_mkt:.4f} | {r.share_tc_pl:.4f} | {r.tc_ratio:.4f} | "
                  f"{r.gain_share:.4f} | {r.wmean_log_mu_mkt:.4f} | "
                  f"{r.wmean_log_tc_ratio:.4f} |" for r in sub.itertuples()]
        parts += ["", f"### `regression.{planner}` (descriptive, sector FE)", "",
                  _md_regression(reg, planner),
                  "Variance split (TC-weighted): "
                  + ", ".join(f"{k} {v:.4f}" for k, v in
                              reg[planner]["variance_split_tc_weighted"].items()), ""]
    auto = "\n".join(parts)

    head = ""
    if path.exists():
        text = path.read_text()
        if AUTO_MARKER in text:
            head = text.split(AUTO_MARKER)[0]
    if not head:
        head = "# Planner expansion: results\n\n(verdict to be written)\n\n"
    path.write_text(head + auto)




def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    welfare_dir = Path(args.welfare_dir)
    fixed_dir = None if args.skip_fixed_input else Path(args.fixed_input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.window

    df = load_panel(welfare_dir, fixed_dir, tag)
    valid = validate(df, welfare_dir, fixed_dir, tag, args.corr_target)
    print(yaml.safe_dump({"validation": valid}, sort_keys=False))
    if not valid["passed"]:
        _write_yaml(out_dir / "planner_expansion.yaml", {"validation": valid})
        raise SystemExit("validation failed; see planner_expansion.yaml")

    df = add_bins(df)
    planners = {"free": {"tc": "tc_pl", "sales": "sales_pl", "share": "share_pl"}}
    if fixed_dir is not None:
        for name in FIXED_PLANNERS:
            planners[name] = {"tc": f"tc_{name}", "sales": f"sales_{name}", "share": f"share_{name}"}

    cells = build_cells(df, planners)
    reg = build_regressions(df, planners)
    support = alpha_support(df, planners)
    inputs = [welfare_dir / f"sim_panel_market_{tag}.parquet", welfare_dir / f"sim_panel_planner_{tag}.parquet",
              welfare_dir / f"welfare_{tag}.yaml"]
    if fixed_dir is not None:
        inputs += [fixed_dir / f"{stem}_{tag}.parquet" for stem, _ in FIXED_PLANNERS.values()]
        inputs.append(fixed_dir / f"fixed_input_welfare_{tag}.yaml")
    result = {
        "provenance": {
            "inputs_mtime": {str(p): dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
                             for p in inputs},
            "ranking": "tie-broken ranks, rank(method='first') after sorting on (market_id, firm_id); "
                       "pooled = across all active firms, within = within market_id",
            "input_shares": "within a regime k, l, m are scalar fractions of TC, so input shares equal TC shares",
        },
        "validation": valid, **cells_to_yaml(cells), "regression": reg, "alpha_support": support,
    }
    cells.to_parquet(out_dir / "planner_expansion_cells.parquet", index=False)
    _write_yaml(out_dir / "planner_expansion.yaml", result)
    write_results_md(out_dir / "results.md", cells, reg, valid, support, list(planners))
    print(f"wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
