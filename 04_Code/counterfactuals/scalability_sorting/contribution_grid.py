"""Where the fixed-K markup loss lives, by scalability quintile.

The fixed-K loss lambda^K is a pure *reallocation* loss, whose sufficient
statistic is the cost-weighted markup variance ``Var_cost(mu) = sum_i w_i
(mu_i - mu_bar)^2`` (``w_i`` = firm cost share, ``mu_bar`` = cost-weighted mean).
This grouping attributes the **full** variance to alpha-rank quintiles --
including within-quintile / within-sector dispersion -- unlike a 2D cell-mean
heatmap, which keeps only the small between-cell part and understates the
high-alpha firms.

We solve the MARKET economies (baseline, reverse) -- the markup wedge only exists
there -- and plot each alpha-quintile's contribution to ``Var_cost(mu)``,
baseline vs reverse, on a common scale.

Read the figure as a *level* story, not a *concentration* story. The top-alpha
quintile carries ~90% of ``Var_cost(mu)`` in ANY matching (90.8% baseline, 93.2%
under a random shuffle) because markup is convex in market share
(``corr(share, mu) = 0.999``) and the biggest firms sit at the top of the size
tail -- so the quintile share is mechanical and is NOT itself a sorting effect.
What sorting changes is the total height of the bars: breaking the sorting
collapses total ``Var_cost(mu)`` ~13-fold (reverse is ~0.4% of baseline), which is
what maps onto lambda^K falling. Do not quote "90% in the top quintile" as a
headline sorting result.

Nothing under ``steady_state/`` is modified; ``_alpha_rank_percentile`` and the
house style are imported unchanged.

Run from ``04_Code/``::

    python -m counterfactuals.scalability_sorting.contribution_grid \\
      --calibration out_results/calib_pooled/calibration_pooled.yaml --M 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steady_state.__main__ import _params_for_command, _pooled_inputs, load_config
from steady_state.figures.house_style import configure_style, save_pdf_and_png, style_axes
from steady_state.figures.simulation_figures import _alpha_rank_percentile
from steady_state.model.pool import draw_pool
from steady_state.simulation.cross_section import build_firm_panel

from .permute import active_corr_av, permute_alpha
from .run_scalability_sorting import _solve_market

N_BINS = 5


def _var_by_quintile(panel: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Contribution of each alpha-rank quintile to cost-weighted Var(mu).

    Returns ``(contrib_by_q, mu_bar, cost_wtd_mu_by_q, cost_share_by_q)`` where
    ``contrib_by_q`` sums to the total cost-weighted variance (full attribution,
    within-quintile dispersion included).
    """
    df = panel[panel["active"].astype(bool)]
    markup = df["markup"].to_numpy(float)
    cost = df["cost"].to_numpy(float)
    rank = _alpha_rank_percentile(df["tilde_alpha"].to_numpy(float))

    w = cost / cost.sum()
    mu_bar = float(np.sum(w * markup))
    qbin = np.clip(np.digitize(rank, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]), 0, n_bins - 1)

    contrib = np.zeros(n_bins)
    cw_mu = np.full(n_bins, np.nan)
    cost_share = np.zeros(n_bins)
    for q in range(n_bins):
        sel = qbin == q
        contrib[q] = float(np.sum(w[sel] * (markup[sel] - mu_bar) ** 2))
        cost_share[q] = float(w[sel].sum())
        if np.any(sel):
            cw_mu[q] = float(np.sum(cost[sel] * markup[sel]) / cost[sel].sum())
    return contrib, mu_bar, cw_mu, cost_share


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="counterfactuals.scalability_sorting.contribution_grid")
    parser.add_argument("--config", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--M", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--economies-parquet", default=None,
                        help="sorting_economies.parquet with lambda_K per mode "
                             "(panel titles only). Defaults to the headline run.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    inputs = _pooled_inputs(cfg)
    params, wf, initial = _params_for_command(cfg, inputs, args.calibration)
    n = N_BINS

    support = (inputs.alpha_support if inputs.alpha_support is not None
               else np.linspace(0.6, 1.20, 500))
    base = draw_pool(support, xi=params.xi, rho_bar=params.rho_bar, N=params.N,
                     M=args.M, H=params.H, rng=args.seed, v_min=params.v_min)
    rev = permute_alpha(base, "reverse", None)

    print(f"solving MARKET legs at M={args.M} "
          f"(corr_av base={active_corr_av(base):+.3f} rev={active_corr_av(rev):+.3f})")
    me_b = _solve_market(base, params, cfg, inputs, wf, initial)
    me_r = _solve_market(rev, params, cfg, inputs, wf, (me_b.w, me_b.X_market))
    if not me_r.converged:
        me_r = _solve_market(rev, params, cfg, inputs, wf, initial)

    pan_b = build_firm_panel(me_b, tilde_alpha=base.tilde_alpha, regime="market")
    pan_r = build_firm_panel(me_r, tilde_alpha=rev.tilde_alpha, regime="market")
    cb, mub_b, cwmu_b, cs_b = _var_by_quintile(pan_b, n)
    cr, mub_r, cwmu_r, cs_r = _var_by_quintile(pan_r, n)
    var_b, var_r = float(cb.sum()), float(cr.sum())

    # lambda^K per economy (panel annotation only).
    econ_path = Path(args.economies_parquet) if args.economies_parquet else (
        cfg.out_results_dir / "counterfactuals" / "scalability_sorting"
        / "sorting_economies.parquet")
    lam_b = lam_r = None
    if econ_path.exists():
        econ = pd.read_parquet(econ_path)
        lam_b = float(econ.loc[econ["mode"] == "baseline", "lambda_K"].mean())
        lam_r = float(econ.loc[econ["mode"] == "reverse", "lambda_K"].mean())

    # ---- numeric report ----
    print(f"\ncost-weighted Var(mu): baseline {var_b:.5f}  reverse {var_r:.5f} "
          f"(reverse = {var_r/var_b*100:.1f}% of baseline)")
    print("\ncontribution to cost-weighted Var(mu) by scalability quintile:")
    print(f"{'scal Q':>7}{'base x1e3':>11}{'base %':>8}{'cw mu':>8}{'cost%':>7}"
          f"{'rev x1e3':>11}{'rev %':>8}")
    for q in range(n):
        print(f"{'Q'+str(q+1):>7}{cb[q]*1e3:>11.3f}{cb[q]/var_b*100:>8.1f}"
              f"{cwmu_b[q]:>8.3f}{cs_b[q]*100:>7.1f}"
              f"{cr[q]*1e3:>11.3f}{(cr[q]/var_r*100 if var_r>0 else 0):>8.1f}")
    print(f"\nTOTAL cost-weighted Var(mu): reverse is {var_r/var_b*100:.1f}% of "
          f"baseline (the sorting effect is this ~level collapse). "
          f"NB the top-alpha quintile carries {cb[-1]/var_b*100:.1f}% of the "
          f"baseline variance, but that share is ~sorting-invariant (mechanical: "
          f"markup convex in share), so it is NOT itself a sorting result.")

    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.out_figs_dir / "counterfactuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV: contribution to cost-weighted Var(mu) by scalability quintile ----
    csv_rows = []
    for q in range(n):
        csv_rows.append({
            "scal_quintile": f"Q{q+1}",
            "alpha_rank_range": f"{int(100*q/n)}-{int(100*(q+1)/n)}%",
            "baseline_var_contrib_x1e3": round(cb[q] * 1e3, 5),
            "baseline_pct_of_total": round(cb[q] / var_b * 100, 2),
            "baseline_costwt_mu": round(float(cwmu_b[q]), 5),
            "baseline_cost_share_pct": round(cs_b[q] * 100, 2),
            "reverse_var_contrib_x1e3": round(cr[q] * 1e3, 5),
            "reverse_pct_of_total": round(cr[q] / var_r * 100, 2) if var_r > 0 else 0.0,
            "reverse_costwt_mu": round(float(cwmu_r[q]), 5),
            "reverse_cost_share_pct": round(cs_r[q] * 100, 2),
        })
    csv_rows.append({
        "scal_quintile": "TOTAL", "alpha_rank_range": "",
        "baseline_var_contrib_x1e3": round(var_b * 1e3, 5),
        "baseline_pct_of_total": 100.0, "baseline_costwt_mu": round(mub_b, 5),
        "baseline_cost_share_pct": 100.0,
        "reverse_var_contrib_x1e3": round(var_r * 1e3, 5),
        "reverse_pct_of_total": 100.0 if var_r > 0 else 0.0,
        "reverse_costwt_mu": round(mub_r, 5), "reverse_cost_share_pct": 100.0,
    })
    csv_path = out_dir / "quintile_contribution.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"wrote table to {csv_path}")

    # ---- figure: grouped bars, common scale ----
    configure_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    x = np.arange(n); wbar = 0.4
    c_base, c_rev = "#2b3a67", "#c98a3c"
    ax.bar(x - wbar / 2, cb * 1e3, wbar, label="Baseline (assortative)", color=c_base)
    ax.bar(x + wbar / 2, cr * 1e3, wbar, label="Reverse sorted", color=c_rev)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q+1}\n{int(100*q/n)}-{int(100*(q+1)/n)}%" for q in range(n)],
                       fontsize=11)
    ax.set_xlabel("Scalability quantile (alpha rank)", fontsize=12)
    style_axes(ax)
    ax.legend(fontsize=11, loc="upper left")
    title = ("Reverse sorting collapses total markup dispersion "
             "(the top quintile dominates in either regime)")
    sub = (r"Contribution to cost-weighted $\mathrm{Var}(\mu)$ ($\times 10^{3}$) "
           "by scalability quintile. ")
    if lam_b is not None:
        sub += (f"$\\lambda^K$: baseline {lam_b*100:.1f}%, reverse {lam_r*100:.1f}%. ")
    sub += (f"Reverse total is {var_r/var_b*100:.1f}% of baseline. "
            "Full attribution (within-quintile dispersion included).")
    ax.text(0.0, 1.075, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=15)
    ax.text(0.0, 1.02, sub, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color="0.35")

    stem = out_dir / "scalability_sorting_markup_variance_by_quintile"
    save_pdf_and_png(fig, stem)
    plt.close(fig)
    print(f"\nwrote figure to {stem}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
