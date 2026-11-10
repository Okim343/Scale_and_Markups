# Who Carries the Markups? Scalability, Sorting, and the Welfare Cost of Market Power

**Enrico Truzzi** · Universitat Pompeu Fabra · [enrico.truzzi@upf.edu](mailto:enrico.truzzi@upf.edu)

![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Status: working paper](https://img.shields.io/badge/status-working%20paper-orange)

Paper: [`02_Drafts/main_orbis.pdf`](02_Drafts/main_orbis.pdf) (compiled from the Typst source [`02_Drafts/main_orbis.typ`](02_Drafts/main_orbis.typ))

## Overview

Standard oligopoly models let firms differ in productivity but assign every firm the same returns to scale. This project introduces firm-level heterogeneity in returns to scale (RTS), disciplined by gross-output production function estimates for Spanish firms in Orbis, into an Atkeson-Burstein-style Cournot oligopoly, and studies how the sorting between productivity and scalability shapes the welfare cost of market power. With heterogeneous RTS, productivity and scalability jointly determine equilibrium size, and under oligopoly size determines markups. A given markup wedge is therefore more distortionary when it sits on a firm that can absorb large quantities with only a limited increase in cost.

Returns to scale are estimated with the method of Gandhi, Navarro and Rivers (2020, GNR) in the spirit of Hubmer, Chan, Ozkan, Salgado and Hong (2025), on 1,391,061 Spanish Orbis firm-years from 2015 to 2019. Markups follow the production-function approach of De Loecker, Eeckhout and Unger (2020).

## Main Results

- **Data.** Returns to scale are U-shaped in firm size, both within NAICS2 industry-years and in the pooled sample, with the top 5% of firms by sales reaching an RTS about 0.074 above the industry mean. As in Hubmer et al., the steep rise at the top is driven by the output elasticity of intermediate inputs.
- **Calibration.** The calibrated vector $(\xi, N, \gamma, \bar\rho)$ matches the cost-weighted aggregate markup (1.1815), two Zipf tail indices of firm sales (top 5% and top 2%) and the RTS gap between the top 5% and the bottom 50% of firms by sales (0.046), all within 0.015%. The implied productivity-scalability correlation is negative, about -0.14.
- **Welfare cost of markups.** Removing every markup raises steady-state consumption by $\lambda = 32.3\%$, more than twice the 14.3% of an economy recalibrated to the same targets with common RTS. Each margin on its own adds little, and the joint interaction of productivity and scalability accounts for 48.7% of the gap, such that the two act as complements.
- **Channels.** Reallocating the inputs the market already uses would raise consumption by only 5.5%, while capital accounts for 63% of the gap. Markups lower welfare mostly by holding back the expansion of scalable firms, whose output responds most to a given markup.
- **Sorting.** Holding both marginal distributions fixed, $\lambda$ is S-shaped in the correlation between the two margins, rising from 9.0% under perfect negative pairing to 50.3% under random reassignment and 152.3% under perfect positive pairing. Sorting raises welfare in both the market and the planner allocation, and widens the gap because the planner gains faster.

## The Model in Brief

Each firm $i$ in sector $j$ draws a scalability $\alpha_{ji}$ from the empirical distribution $F^\alpha$ and a productivity $\nu_{ji}$, which enter an anchored marginal cost

$$MC_{ji}(y) = \frac{\Omega^g}{\nu_{ji}} \left(\frac{y}{\hat y}\right)^{1/\alpha_{ji} - 1},$$

where $\Omega^g$ is the gross-output unit cost and $\hat y$ the median active-firm output. Productivity alone ranks firms at the anchor, and $\alpha_{ji}$ only governs how fast marginal cost rises away from it. Finite-firm Cournot competition inside each sector delivers the inverse-markup rule

$$\frac{1}{\mu_{ji}} = 1 - \frac{1}{\gamma} - \left(\frac{1}{\eta} - \frac{1}{\gamma}\right) s_{ji}, \qquad \gamma > \eta.$$

The dependence between the two margins runs through ranks, with a Gaussian copula of strength $\bar\rho$ mapped into a Pareto productivity draw,

$$\ell_{ji} = \bar\rho\, \tilde\alpha_{ji} + \sqrt{1 - \bar\rho^2}\, \epsilon_{ji}, \qquad \nu_{ji} = \underline{\nu}\, \big(1 - \Phi(\ell_{ji})\big)^{-1/\xi},$$

with the number of firms per sector drawn from a Poisson distribution with mean $N$. Scalability draws are truncated at $0.99\,\gamma/(\gamma-1)$, since cost minimization requires returns to scale not to exceed the markup. The welfare gain is split in two non-nested ways, into markup dispersion and level (through a uniform-markup allocation), and into misallocation, materials and capital (through fixed-input and fixed-capital planners).

## Repository Layout

```text
.
├── 02_Drafts/
│   ├── main_orbis.typ             # paper source (Typst)
│   ├── main_orbis.pdf             # compiled paper
│   └── literature.bib
├── 03_Empirical/                  # data -> calibration bundle (see 03_Empirical/README.md)
│   ├── 00_GNR_core/               # shared GNR estimator (Python port of the Hubmer et al. MATLAB code)
│   ├── 02_Orbis/                  # Orbis Spain: cleaning, GNR, markups, moments, age/size side analysis
│   └── 01_Compustat/              # earlier Compustat pipeline, kept as a reference only
├── 04_Estimation/                 # calibration (package orbis_estimation)
├── 05_Simulation/                 # structural model, welfare and counterfactuals (package orbis_model)
├── environment.yml
└── LICENSE
```

The dependency runs one way. `03_Empirical` writes the bundle, `04_Estimation` reads it and writes the calibrated parameters, and `05_Simulation` reads both. Every output records the sha256 of its inputs, and the simulation refuses to run on a parameter file calibrated on a different bundle.

## From Paper to Code

| Paper object | Producer |
|---|---|
| RTS estimates, $F^\alpha$, $\phi_v$ (Section 3.2) | `03_Empirical/02_Orbis/02_GNR`, then `04_Moments/01_code/01_build_F_alpha.py` and `06_alpha_moments.py` |
| Aggregate markup, Zipf tail indices, RTS gap (Section 3.2) | `03_Empirical/02_Orbis/03_Markups`, `04_Moments/01_code/04_aggregate_markup.py`, `12_zipf_targets.py`, `06_alpha_moments.py` |
| Figure 1, RTS by sales percentile | `03_Empirical/02_Orbis/04_Moments/01_code/10_fig_rts_by_size.py` |
| Table 1, calibration and fit | `04_Estimation/01_calibrate_zipf.py` |
| Table 2 and Table B2, four recalibrated economies | `04_Estimation/03_calibrate_restricted.py`, then `05_Simulation/08_recalibrated_anatomy.py` |
| Figure 2, channels from the homogeneous economy to the baseline | `05_Simulation/09_recalibrated_channels_figure.py` |
| Table 3, sorting sweep | `05_Simulation/04_sorting_sweep.py` (welfare columns), `10_sorting_markup_moments.py` (markup columns) |
| Figure 3 and Figure B1, decomposition and welfare along the sweep | `05_Simulation/04_sorting_sweep.py` (`sorting_lenses`, `sorting_curve`) |

All results in the paper use the market-free (Zipf) calibration, that is `04_Estimation/config_estimation_zipf.yaml` and `05_Simulation/config_simulation_zipf.yaml`.

## Replication

**Environment.**

```bash
conda env create -f environment.yml
conda activate scale_markups
```

The paper is compiled with [Typst](https://typst.app) (`typst compile 02_Drafts/main_orbis.typ`). Figures are not present, so recompiling requires rerunning the pipeline below, whose figure scripts write into `02_Drafts/figures/orbis/`.

**1. Empirical bundle.** The Orbis stages, their order and the bundle contract are documented in [`03_Empirical/README.md`](03_Empirical/README.md). The bundle in `03_Empirical/02_Orbis/04_Moments/03_outdata/` is the only empirical input read by the model.

**2. Calibration.** From `04_Estimation/` (see [`04_Estimation/README.md`](04_Estimation/README.md)):

```bash
python 01_calibrate_zipf.py
python 03_calibrate_restricted.py --economy hom_alpha
python 03_calibrate_restricted.py --economy hom_v
python 03_calibrate_restricted.py --economy hom_both
```

**3. Welfare and counterfactuals.** From `05_Simulation/`, each script with `--config config_simulation_zipf.yaml` (see [`05_Simulation/README.md`](05_Simulation/README.md) for the arguments, the reference gates and the run order):

```text
02_welfare.py -> 03_fixed_input_welfare.py -> 04_sorting_sweep.py -> 10_sorting_markup_moments.py
                                           -> 06_level_dispersion.py -> 07_heterogeneity_anatomy.py
                                              -> 08_recalibrated_anatomy.py -> 09_recalibrated_channels_figure.py
```

**Tests.**

```bash
(cd 04_Estimation && python -m pytest)
(cd 05_Simulation && python -m pytest)
```

Tests that need the licensed bundle skip when it is absent, and the rest run on synthetic bundles.

**Computational note.** The market solver is vectorized and parallelized over simulated markets. The calibration runs at $M = 5000$ markets and the sorting sweep at $M = 10000$, each taking a few hours on 16 to 24 cores. The cluster scripts are kept local.

## Data Availability

Firm-level data come from Orbis (Bureau van Dijk), accessed through WRDS under a university licence, and cannot be redistributed. Deflators (Eurostat) and the euro overnight rate (ECB) are public and fetched by the pipeline. Raw, intermediate and output data, model outputs and figures are excluded from version control by design, and the pipeline expects them under the paths listed in [`03_Empirical/README.md`](03_Empirical/README.md).

## Citation

```bibtex
@unpublished{truzzi2026carries,
  author = {Truzzi, Enrico},
  title  = {Who Carries the Markups? Scalability, Sorting, and the Welfare Cost of Market Power},
  note   = {Working paper, Universitat Pompeu Fabra},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE).
