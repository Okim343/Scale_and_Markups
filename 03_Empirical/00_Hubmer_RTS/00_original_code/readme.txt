README: GNR Production Function Estimation Code
================================================

OVERVIEW
--------
MATLAB code implementing the nonparametric production function and TFP
estimation of Gandhi, Navarro, and Rivers (2020, JPE). The estimation has
two steps: Step 1 recovers the materials output elasticity from the
materials share equation via constrained nonlinear least squares; Step 2
recovers the constant of integration (functions of capital and labor) and
the productivity Markov process via GMM. Outputs are firm-level
productivity (omega), output elasticities for capital, labor, and
materials, and returns to scale.

FILES
-----
est_gnr.m
    Main script. Loads and trims the firm-level data, runs both estimation
    steps using MultiStart global optimization, computes output
    elasticities and returns to scale, and exports results to CSV.

nls_obj.m
    Step 1 objective function (value and Jacobian). Supports least-squares
    (vector and sum), derivative-based, and GMM formulations.

nls_const_filter.m
    Filters candidate starting parameter vectors, keeping those that
    satisfy the Step 1 positivity constraint (P1*gamma' > 0).

step2_obj.m
    Step 2 GMM objective. In the nested version, the Markov-process
    coefficients (delta) are concentrated out by OLS given a guess of the
    integration-constant parameters (alpha).

gensobol.m
    Generates scrambled Sobol sequences of starting values on a
    user-specified hypercube.

rowupdate.m
    Utility to add or remove rows from a matrix of stored best-guess
    parameter vectors (tolerance-based duplicate checking).

INPUT DATA
----------
est_gnr.m expects a comma-delimited CSV (path set at the top of the
script) with: id (firm identifier), year, ind (industry), s (log
materials expenditure share of revenue) and its lag ls, and logs of
revenue, capital, materials, labor, and wages (r, k, m, l, w) plus their
one-period lags (lr, lk, lm, ll, lw). Drop missing observations
beforehand. The script trims extreme materials-share observations
(m/r outside [0.05, 0.95]) and small industries.

OUTPUT
------
A CSV (and .mat file) with, for each firm-year: the Step 1 residual
(eps), the materials, capital, and labor output elasticities,
productivity (omega and its lag), the productivity innovation (eta), and
returns to scale. A guess file storing the best parameter estimates is
saved for use as starting values in later runs.

REQUIREMENTS
------------
MATLAB with the Optimization, Global Optimization, and Statistics and
Machine Learning Toolboxes. Parallel Computing Toolbox optional
(MultiStart is set to UseParallel = true).

REFERENCE
---------
Gandhi, A., S. Navarro, and D. Rivers (2020). "On the Identification of
Gross Output Production Functions." Journal of Political Economy,
128(8), 2973-3016.
