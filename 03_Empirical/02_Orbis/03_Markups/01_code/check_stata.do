* Cross-check for 05_stata_crosscheck.py: run in <intermediary>/stata_check/ with
*   stata-se -b do check_stata.do
* 1) DLEU's estimator line (02_Produce_Thetas_st.do) on one exported cell-window.
* 2) egen pctile (the DLEU trims) on synthetic groups.

clear all
set more off

import delimited using "check_iv.csv", clear asdouble
gen k_lag2 = k_lag^2
* DLEU: ivregress gmm r (c =L.c) k L.i L.k2 L.k
ivregress gmm r (c = c_lag) k i_lag k_lag2 k_lag
matrix B = e(b)
matrix VV = e(V)
local names "c k i_lag k_lag2 k_lag _cons"
clear
set obs 6
gen str10 term = ""
gen double b = .
gen double se = .
local j = 1
foreach t of local names {
    replace term = "`t'" in `j'
    replace b = B[1, `j'] in `j'
    replace se = sqrt(VV[`j', `j']) in `j'
    local ++j
}
format b se %21.15g
export delimited using "iv_stata.csv", replace

import delimited using "pctile_synth.csv", clear asdouble
bysort grp: egen double p1 = pctile(x), p(1)
bysort grp: egen double p99 = pctile(x), p(99)
bysort grp: keep if _n == 1
keep grp p1 p99
format p1 p99 %21.15g
export delimited using "pctile_stata.csv", replace
