# How Costly are Scalable Markups?

**Enrico Truzzi** · Universitat Pompeu Fabra · [enrico.truzzi@upf.edu](mailto:enrico.truzzi@upf.edu)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

Working draft: [`02_Drafts/model.typ`](02_Drafts/model.typ) (Typst source)

## Overview

Classic oligopoly models assign every firm the same returns-to-scale parameter, such that firms differ in productivity but not in how easily they can scale. With heterogeneous returns to scale, capability and scalability jointly determine equilibrium size, and, under oligopoly, size determines markups. This project argues that the sorting between capability and scalability, and not only their marginal distributions, plays a major role in shaping the aggregate welfare cost of market power, since a given markup wedge is more distortionary when it is attached to a firm that can absorb large quantities with only a limited increase in cost.

The quantitative environment is a nested-CES Cournot economy in the tradition of Edmond, Midrigan and Xu (2023, henceforth EMX), where markups are equilibrium outcomes of finite-firm competition inside each sector and larger market shares relax the perceived demand elasticity. Additionally, I combine this endogenous-market-power mechanism with firm-level returns to scale disciplined by gross-output production function estimates on Compustat, using the approach of Gandhi, Navarro and Rivers (2020, henceforth GNR) in the spirit of Hubmer, Chan, Ozkan, Salgado and Hong (2025). The research question is the following: how does capability-scalability sorting shape endogenous markups and their aggregate welfare cost?

## Main Results

The calibrated economy matches the SG&A-inclusive aggregate markup, within-sector concentration (CR4 and CR20), the EMX markup-concentration slope and the empirical correlation between returns to scale and log sales, and generates a markup distribution that is considerably more compressed than that of EMX, with the remaining dispersion concentrated in a thin upper tail of dominant firms. Moving from the market to the efficient allocation yields a large consumption-equivalent gain, most of which comes from the aggregate scale of production rather than from the reallocation of a given stock of inputs, and from the markup level rather than from markup dispersion. That is, the two lenses used in the paper (a fixed-input misallocation view in the style of Hsieh and Klenow, and a uniform-markup view in the style of EMX) both place the bulk of the loss on the capital-accumulation margin.

Positive sorting of scalability onto capability nearly doubles the welfare cost of markups relative to an economy where the same returns to scale are assigned at random, whereas reverse sorting and a common returns-to-scale parameter lower it further. In particular, the amplification operates through a small set of sector leaders that combine the highest capability, the highest scalability and the highest markups, and that absorb a disproportionate share of the inputs the planner adds, with the markup acting as the trigger and the scalable technology as the source of the expansion. Reading this as evidence that sorting is harmful would, however, be an incorrect interpretation, since breaking sorting lowers consumption in every allocation: sorting is productive, and raises the cost of markups only because the planner gains more from it than the market does.

## The Model in Brief

Each firm $i$ in sector $j$ is characterised by its capability $\nu_{ji}$ and its scalability $\alpha_{ji}$, which enter an anchored marginal cost

$$MC_{ji}(y) = \frac{\Omega^g}{\nu_{ji}} \left(\frac{y}{\hat y}\right)^{1/\alpha_{ji} - 1},$$

where $\Omega^g$ is the gross-output unit cost and $\hat y$ a common anchor output, such that capability alone ranks firms at the anchor and $\alpha_{ji}$ only governs how fast marginal cost rises away from it. Finite-firm Cournot competition delivers the EMX inverse-markup rule

$$\frac{1}{\mu_{ji}} = 1 - \frac{1}{\gamma} - \left(\frac{1}{\eta} - \frac{1}{\gamma}\right) s_{ji}, \qquad \gamma > \eta,$$

which makes markups increasing in the market share $s_{ji}$. The dependence between the two margins is introduced only through ranks, with a Gaussian copula of strength $\bar\rho$ mapped into a Pareto capability draw,

$$\ell_{ji} = \bar\rho\, \tilde\alpha_{ji} + \sqrt{1 - \bar\rho^2}\, \epsilon_{ji}, \qquad \nu_{ji} = \underline{\nu}\, \big(1 - \Phi(\ell_{ji})\big)^{-1/\xi},$$

where $\tilde\alpha_{ji}$ is the standard-normal score of the firm's scalability rank in its sector. Re-solving the economy under alternative within-sector arrangements of $\alpha$, which leave both marginal distributions unchanged, splits any welfare loss $\Delta$ (a log consumption ratio) additively into a common-$\alpha$ term, a heterogeneity term and a sorting term,

$$\Delta_{\text{base}} = \Delta_{\text{hom}} + (\Delta_{\text{shuf}} - \Delta_{\text{hom}}) + (\Delta_{\text{base}} - \Delta_{\text{shuf}}).$$

## Repository Layout

```text
.
├── 02_Drafts/
│   ├── model.typ                     # working draft (Typst)
│   └── literature.bib
├── 03_Empirical/                     # data -> calibration bundle (see 03_Empirical/README.md)
│   ├── 00_Hubmer_RTS/01_code/        # GNR gross-output estimation, firm-level returns to scale
│   └── 01_industry_alpha/02_code/    # Compustat + KLEMS -> pooled targets, F(alpha), bundle
├── 04_Code/
│   ├── steady_state/                 # structural model package
│   │   ├── model/                    # pool draws, Cournot pricing, market solver, aggregation
│   │   ├── calibration/              # inner/outer loop, SMM objective, identification checks
│   │   ├── simulation/               # cross-section and model moments
│   │   ├── welfare/                  # market and planner allocations, welfare metrics
│   │   ├── figures/
│   │   └── config.yaml
│   ├── counterfactuals/              # welfare decompositions and sorting experiments
│   └── tests/                        # pytest suite
├── run_steady_state.py               # project-root launcher for the steady_state CLI
└── environment.yml
```

## From Paper to Code

| Draft section | Object | Code |
|---|---|---|
| Data and Measurement | Firm-level returns to scale (GNR) | `03_Empirical/00_Hubmer_RTS/01_code/` (stages R1 to R5) |
| Data and Measurement | Pooled targets, $F(\alpha)$, copula inputs, calibration bundle | `03_Empirical/01_industry_alpha/02_code/` (stages S1 to S8) |
| Pooled-Market Steady State | Market equilibrium and sector firm counts | `04_Code/steady_state/model/` |
| Calibration | Simulated method of moments | `04_Code/steady_state/calibration/` |
| Welfare Cost of Markups, Lens A | Reallocation and scale legs | `04_Code/counterfactuals/fixed_input_welfare/` |
| Welfare Cost of Markups, Lens B | Dispersion and level legs | `04_Code/counterfactuals/lens_b_decomposition/` |
| Sorting and the Cost of Markups | Baseline, shuffled and reverse-sorted arrangements | `04_Code/counterfactuals/scalability_sorting/` |
| Sorting and the Cost of Markups | Homogeneous returns to scale | `04_Code/counterfactuals/homogeneous_rts/` |
| Sorting and the Cost of Markups | Scale-channel and planner-expansion diagnostics | `04_Code/counterfactuals/scale_channel_decomposition/`, `04_Code/counterfactuals/planner_expansion/` |

Each counterfactual package carries its own README documenting its convention and outputs.

## Replication

**Environment.** The code is written in Python 3.10 and relies on NumPy, SciPy, pandas, Numba and joblib.

```bash
conda env create -f environment.yml
conda activate scale_markups
```

**Empirical bundle.** The two empirical sub-pipelines, their stage DAG and the full input/output contract are documented in [`03_Empirical/README.md`](03_Empirical/README.md). The bundle they produce is the only empirical input read by the structural model.

**Calibration and simulation.** From the project root:

```bash
python run_steady_state.py calibrate --M 1000 --outer-maxiter 5
python run_steady_state.py simulate --calibration 04_Code/out_results/calibration_2010_2019.yaml
python run_steady_state.py validate --calibration 04_Code/out_results/calibration_2010_2019.yaml
```

**Counterfactuals.** From `04_Code/`, each experiment reads the calibrated parameter file, for instance:

```bash
python -m counterfactuals.fixed_input_welfare.run_fixed_capital_planner --calibration out_results/calib_pooled/calibration_pooled.yaml
python -m counterfactuals.lens_b_decomposition.run_lens_b --calibration out_results/calib_pooled/calibration_pooled.yaml
python -m counterfactuals.scalability_sorting.run_scalability_sorting --calibration out_results/calib_pooled/calibration_pooled.yaml
```

**Tests.**

```bash
pytest 04_Code/tests 03_Empirical/01_industry_alpha/02_code/tests
```

**Computational note.** The market solver is vectorised and parallelised across simulated sectors. A full calibration with 5,000 simulated sectors takes around two hours parallelised over 16 Haswell cores, whereas a single counterfactual at 10,000 sectors runs in around ten to fifteen minutes.

## Data Availability

Firm-level data come from Compustat North America (2010 to 2019), accessed through WRDS under a university licence, and cannot be redistributed, whereas sector capital shares come from the publicly available KLEMS accounts. Raw, intermediate and output data folders are therefore excluded from version control by design, and the empirical pipeline expects them under the paths listed in [`03_Empirical/README.md`](03_Empirical/README.md).

## Citation

```bibtex
@unpublished{truzzi2026scalable,
  author = {Truzzi, Enrico},
  title  = {How Costly are Scalable Markups?},
  note   = {Working paper, Universitat Pompeu Fabra},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE).
