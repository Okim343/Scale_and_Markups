"""CLI entry point for the planner-expansion incidence tables.

Descriptive post-processing of the stored baseline firm panels: the MARKET and
free-capital PLANNER panels of the welfare run, plus (optionally) the
fixed-capital PLANNER panel of ``fixed_input_welfare``. It tests, on existing
outputs only, the mechanism the paper currently hedges in the sorting section:
positive sorting puts the most scalable technologies in the most capable firms,
which are the largest and carry the highest markups, so the markup wedge holds
back input demand where the planner would expand it most.

Because ``k``, ``l`` and ``m`` are each a scalar fraction of firm total cost
``TC`` within a regime, every input share below is a ``TC`` share, and the
firm-level input expansion is ``TC_planner / TC_market``. ``cost`` in the
panels is true total cost ``TC = alpha*revenue/mu``.

Validation runs first and aborts the run on failure:

* free planner: ``sum TC_pl / sum TC_mkt == K_PE / K_ME`` (same ``R``);
* fixed-K planner: ``sum TC_pl / sum TC_mkt == R_PE_I / R_ME`` (same ``K``);
* ``corr(alpha, v)`` over active firms equals the published ``0.78486``;
* market cost-weighted markup equals ``welfare.yaml: mu_bar``.

Quantiles use tie-broken ranks (``rank(method='first')`` after sorting on
``(market_id, firm_id)``) because alpha has a discrete support. Pooled ranks are
the headline; within-sector ranks are the robustness check.

Run from ``04_Code/``::

    python -m counterfactuals.planner_expansion.run_planner_expansion
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from steady_state.__main__ import _write_yaml

PACKAGE_DIR = Path(__file__).resolve().parent
KEYS = ["market_id", "firm_id"]
N_Q = 5

# Published targets the stored panels must reproduce.
CORR_AV_TARGET = 0.78486
CORR_AV_TOL = 5e-6
RATIO_REL_TOL = 1e-6

AUTO_MARKER = "<!-- AUTO-GENERATED TABLES BELOW: do not edit by hand -->"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.planner_expansion.run_planner_expansion",
        description="Where does the planner expand input demand? Descriptive "
                    "incidence tables on the stored baseline firm panels.",
    )
    parser.add_argument(
        "--welfare-dir", default="out_results/welfare",
        help="holds sim_panel_market.parquet, sim_panel_planner.parquet, "
             "welfare.yaml",
    )
    parser.add_argument(
        "--fixed-k-dir",
        default="out_results/counterfactuals/fixed_capital_planner",
        help="holds sim_panel_planner_fixed_k.parquet and "
             "fixed_capital_welfare.yaml; pass '' to skip the fixed-K tables",
    )
    parser.add_argument(
        "--k-ratio-source",
        default="out_results/counterfactuals/scale_channel_decomposition/"
                "scale_channel_economies.parquet",
        help="parquet whose 'baseline' row carries K_market, K_planner, K_ratio",
    )
    parser.add_argument(
        "--out-dir", default="out_results/counterfactuals/planner_expansion",
    )
    parser.add_argument(
        "--typst-out", default=str(PACKAGE_DIR / "table_snippet.typ"),
    )
    return parser


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _read_active(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns + ["active"],
                         filters=[("active", "==", True)])
    return df.drop(columns="active")


def load_panel(welfare_dir: Path, fixed_k_dir: Path | None) -> pd.DataFrame:
    """One row per active firm, market and planner outcomes side by side."""
    mkt = _read_active(welfare_dir / "sim_panel_market.parquet",
                       KEYS + ["alpha", "v", "markup", "cost", "sales", "share"])
    pl = _read_active(welfare_dir / "sim_panel_planner.parquet",
                      KEYS + ["alpha", "v", "cost", "sales", "share"])
    df = mkt.merge(pl, on=KEYS, how="outer", suffixes=("", "_pl"),
                   validate="one_to_one", indicator=True)
    if not (df["_merge"] == "both").all():
        raise SystemExit("market/planner active sets differ: "
                         f"{df['_merge'].value_counts().to_dict()}")
    for col in ("alpha", "v"):
        if not np.array_equal(df[col].to_numpy(), df[f"{col}_pl"].to_numpy()):
            raise SystemExit(f"{col} differs between market and planner panels")
    df = df.drop(columns=["_merge", "alpha_pl", "v_pl"]).rename(columns={
        "cost": "tc_mkt", "sales": "sales_mkt", "share": "share_mkt",
        "markup": "mu_mkt", "cost_pl": "tc_pl", "sales_pl": "sales_pl",
        "share_pl": "share_pl",
    })

    if fixed_k_dir is not None:
        fk = _read_active(fixed_k_dir / "sim_panel_planner_fixed_k.parquet",
                          KEYS + ["alpha", "cost", "sales", "share"])
        fk = fk.rename(columns={"cost": "tc_fk", "sales": "sales_fk",
                                "share": "share_fk", "alpha": "alpha_fk"})
        df = df.merge(fk, on=KEYS, how="left", validate="one_to_one")
        if df["tc_fk"].isna().any():
            raise SystemExit("fixed-K planner active set differs from market")
        if not np.array_equal(df["alpha"].to_numpy(), df["alpha_fk"].to_numpy()):
            raise SystemExit("alpha differs between market and fixed-K panels")
        df = df.drop(columns="alpha_fk")

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
    df["size_rank"] = np.select(
        [rank == 1, rank == 2, rank <= 5], ["1_leader", "2_second", "3_third_fifth"],
        default="4_rest",
    )
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df: pd.DataFrame, welfare_dir: Path, fixed_k_dir: Path | None,
             k_ratio_source: Path) -> dict:
    welfare = yaml.safe_load(open(welfare_dir / "welfare.yaml"))
    econ = pd.read_parquet(k_ratio_source)
    base = econ.loc[econ["label"] == "baseline"].iloc[0]

    tc_ratio = df["tc_pl"].sum() / df["tc_mkt"].sum()
    corr_av = float(np.corrcoef(df["alpha"], df["v"])[0, 1])
    mu_cw = float((df["mu_mkt"] * df["tc_mkt"]).sum() / df["tc_mkt"].sum())

    out = {
        "n_active": int(len(df)),
        "n_markets": int(df["market_id"].nunique()),
        "free_planner": {
            "tc_ratio": float(tc_ratio),
            "K_market": float(base["K_market"]),
            "K_planner": float(base["K_planner"]),
            "K_ratio": float(base["K_ratio"]),
            "rel_gap": float(tc_ratio / base["K_ratio"] - 1.0),
            "source": str(k_ratio_source),
        },
        "corr_alpha_v": {"value": corr_av, "target": CORR_AV_TARGET,
                         "abs_gap": abs(corr_av - CORR_AV_TARGET)},
        "mu_cw_market": {"value": mu_cw, "welfare_yaml_mu_bar": welfare["mu_bar"],
                         "abs_gap": abs(mu_cw - welfare["mu_bar"])},
    }
    checks = [
        abs(out["free_planner"]["rel_gap"]) < RATIO_REL_TOL,
        out["corr_alpha_v"]["abs_gap"] < CORR_AV_TOL,
        out["mu_cw_market"]["abs_gap"] < 1e-9,
    ]
    if fixed_k_dir is not None:
        fkw = yaml.safe_load(open(fixed_k_dir / "fixed_capital_welfare.yaml"))
        fk_ratio = df["tc_fk"].sum() / df["tc_mkt"].sum()
        r_ratio = fkw["R_planner_fixed_k"] / fkw["R_market"]
        out["fixed_k_planner"] = {
            "tc_ratio": float(fk_ratio), "R_market": fkw["R_market"],
            "R_planner_fixed_k": fkw["R_planner_fixed_k"],
            "R_ratio": float(r_ratio), "rel_gap": float(fk_ratio / r_ratio - 1.0),
        }
        checks.append(abs(out["fixed_k_planner"]["rel_gap"]) < RATIO_REL_TOL)
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


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_typst(cells: pd.DataFrame, reg: dict, valid: dict,
                support: dict) -> str:
    """Three-panel table in the format of ``tab:sorting`` in model.typ."""
    def quint_rows(table):
        sub = cells.query("table == @table and scheme == 'pooled' and planner == 'free'")
        rows = []
        for r in sub.itertuples():
            rows.append(f"      [#h(0.8em)Q{r.bin1}], [{r.mean_alpha:.3f}], "
                        f"[{r.mean_log_v:.2f}], [{r.mu_cw_mkt:.3f}], "
                        f"[{_pct(r.share_tc_mkt)}], [{_pct(r.share_tc_pl)}], "
                        f"[{r.tc_ratio:.2f}]")
        return ",\n".join(rows) + ","

    sub = cells.query("table == 'size_rank' and planner == 'free'").set_index("bin1")
    labels = {"1_leader": "Largest firm", "2_second": "Second",
              "3_third_fifth": "Third to fifth", "4_rest": "All others"}
    size_rows = ",\n".join(
        f"      [#h(0.8em){labels[k]}], [{sub.loc[k, 'mean_alpha']:.3f}], "
        f"[{_pct(sub.loc[k, 'firm_share'])}], [{sub.loc[k, 'mu_cw_mkt']:.3f}], "
        f"[{_pct(sub.loc[k, 'share_tc_mkt'])}], [{_pct(sub.loc[k, 'share_tc_pl'])}], "
        f"[{sub.loc[k, 'tc_ratio']:.2f}]" for k in labels) + ","

    rf = reg["free"]["baseline"]["firm_weighted"]
    rw = reg["free"]["baseline"]["tc_mkt_weighted"]
    reg_rows = []
    for name, lab in (("log_mu_mkt", "$ln mu^(M E)$"), ("alpha", "$alpha$"),
                      ("log_v", "$ln nu$")):
        reg_rows.append(
            f"      [#h(0.8em){lab}], [{rf['coef'][name]:.2f}], "
            f"[{rf['partial_r2'][name]:.3f}], [{rw['coef'][name]:.2f}], "
            f"[{rw['partial_r2'][name]:.3f}]")
    reg_rows.append(f"      [#h(0.8em)Within $R^2$], [{rf['r2_within']:.3f}], [], "
                    f"[{rw['r2_within']:.3f}], []")
    reg_rows = ",\n".join(reg_rows) + ","

    tc_ratio = valid["free_planner"]["tc_ratio"]
    lead_irs = _pct(support["leader_share_alpha_ge_1"])
    corr = valid["corr_alpha_v"]["value"]
    return f"""// Generated by counterfactuals/planner_expansion/run_planner_expansion.py.
// Not inserted into model.typ. Numbers: out_results/counterfactuals/planner_expansion/planner_expansion.yaml
#figure(
  block(width: 100%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (2.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      column-gutter: 0.25cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.cell(colspan: 7, inset: (top: 6pt, bottom: 4pt))[*Panel A.* By $alpha$ quintile],
      table.hline(stroke: 0.4pt),
      [*Quintile*], [Mean $alpha$], [Mean $ln nu$], [$mu^(M E)$], [Share $T C^(M E)$], [Share $T C^(P E)$], [Ratio],
{quint_rows("quintile_alpha")}
      table.cell(colspan: 7, inset: (top: 10pt, bottom: 4pt))[*Panel B.* By $nu$ quintile],
      table.hline(stroke: 0.4pt),
      [*Quintile*], [Mean $alpha$], [Mean $ln nu$], [$mu^(M E)$], [Share $T C^(M E)$], [Share $T C^(P E)$], [Ratio],
{quint_rows("quintile_v")}
      table.cell(colspan: 7, inset: (top: 10pt, bottom: 4pt))[*Panel C.* By size rank within the sector],
      table.hline(stroke: 0.4pt),
      [*Rank*], [Mean $alpha$], [Firms], [$mu^(M E)$], [Share $T C^(M E)$], [Share $T C^(P E)$], [Ratio],
{size_rows}
      table.hline(stroke: 0.7pt),
    )
    #v(0.1cm)
    #table(
      columns: (2.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
      column-gutter: 0.25cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.cell(colspan: 5, inset: (top: 6pt, bottom: 4pt))[*Panel D.* Cross-firm regression of $ln(T C^(P E) slash T C^(M E))$, sector fixed effects],
      table.hline(stroke: 0.4pt),
      [], table.cell(colspan: 2, align: center)[_Firm-weighted_], table.cell(colspan: 2, align: center)[_$T C^(M E)$-weighted_],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*Regressor*], [Coef.], [Partial $R^2$], [Coef.], [Partial $R^2$],
{reg_rows}
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ Baseline market ($M E$) and free-capital planner ($P E$) allocations over all {valid['n_active']:,} active firms, where corr$(alpha, nu) = {corr:.3f}$. Within a regime every input is a fixed fraction of firm total cost $T C$, so $T C$ shares are input shares, and aggregate $T C^(P E) slash T C^(M E) = {tc_ratio:.2f}$ equals $K^(P E) slash K^(M E)$. Quintiles are pooled across sectors, each holds $20%$ of firms, and ties in the discrete support of $alpha$ are broken by order. $mu^(M E)$ is the cost-weighted market markup of the cell, and Ratio is $T C^(P E) slash T C^(M E)$ within the cell. In Panel C firms are ranked by market sales within their sector, and {lead_irs} of sector leaders have $alpha >= 1$. Panel D is descriptive, not causal, since $mu^(M E)$ is itself an equilibrium function of $alpha$ and $nu$ through the market share, and the partial $R^2$ is the share of residual variance a regressor explains given the other two.])
    #v(0.5em)
  ],
  caption: [*Where the planner expands input demand.*],
)<tab:planner_expansion>
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _provenance(paths: list[Path]) -> dict:
    return {str(p): dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            for p in paths}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    welfare_dir = Path(args.welfare_dir)
    fixed_k_dir = Path(args.fixed_k_dir) if args.fixed_k_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [welfare_dir, fixed_k_dir]:
        if p is not None and "ARCHIVE" in p.resolve().parts:
            raise SystemExit(f"refusing to read from ARCHIVE: {p}")

    df = load_panel(welfare_dir, fixed_k_dir)
    valid = validate(df, welfare_dir, fixed_k_dir, Path(args.k_ratio_source))
    print(yaml.safe_dump({"validation": valid}, sort_keys=False))
    if not valid["passed"]:
        _write_yaml(out_dir / "planner_expansion.yaml", {"validation": valid})
        raise SystemExit("validation failed; see planner_expansion.yaml")

    df = add_bins(df)
    planners = {"free": {"tc": "tc_pl", "sales": "sales_pl", "share": "share_pl"}}
    if fixed_k_dir is not None:
        planners["fixed_k"] = {"tc": "tc_fk", "sales": "sales_fk",
                               "share": "share_fk"}

    cells = build_cells(df, planners)
    reg = build_regressions(df, planners)
    support = alpha_support(df, planners)

    inputs = [welfare_dir / "sim_panel_market.parquet",
              welfare_dir / "sim_panel_planner.parquet",
              welfare_dir / "welfare.yaml", Path(args.k_ratio_source)]
    if fixed_k_dir is not None:
        inputs += [fixed_k_dir / "sim_panel_planner_fixed_k.parquet",
                   fixed_k_dir / "fixed_capital_welfare.yaml"]
    result = {
        "provenance": {
            "inputs_mtime": _provenance(inputs),
            "rho_bar_fix_commit": "0eab400 (2026-07-09 10:46)",
            "ranking": "tie-broken ranks, rank(method='first') after sorting on "
                       "(market_id, firm_id); pooled = across all active firms, "
                       "within = within market_id",
            "input_shares": "within a regime k, l, m are scalar fractions of TC, "
                            "so input shares equal TC shares",
        },
        "validation": valid,
        **cells_to_yaml(cells),
        "regression": reg,
        "alpha_support": support,
    }
    cells.to_parquet(out_dir / "planner_expansion_cells.parquet", index=False)
    _write_yaml(out_dir / "planner_expansion.yaml", result)
    write_results_md(out_dir / "results.md", cells, reg, valid, support,
                     list(planners))
    Path(args.typst_out).write_text(build_typst(cells, reg, valid, support))
    print(f"wrote outputs to {out_dir} and {args.typst_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
