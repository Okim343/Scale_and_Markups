"""Step 05 (verification): cross-check the Python estimator and percentiles against Stata.

  prepare   writes <intermediary>/stata_check/check_iv.csv (one cell-window of estimation rows,
            via 03_estimate_thetas.py --check) and pctile_synth.csv (synthetic groups with
            sizes divisible and not divisible by 100, and ties)
  compare   reads Stata's iv_stata.csv and pctile_stata.csv (written by check_stata.do) and
            reports the differences with numpy 2SLS and markups_common.stata_pctile

Run check_stata.do with Stata in <intermediary>/stata_check/ between the two steps
(on the lab workstation: stata-se -b do check_stata.do).

    python 05_stata_crosscheck.py prepare [--cell 21 --year 2019]
    python 05_stata_crosscheck.py compare
"""

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from markups_common import CODE, load_config, resolve, setup_logger, stata_pctile

GROUP_SIZES = [100, 200, 250, 999, 1000, 1037, 4321, 5000]
PCTS = [1, 99]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "compare"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--cell", type=int, default=21)
    parser.add_argument("--year", type=int, default=2019)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    logger = setup_logger("stata_check", config.get("log_level", "INFO"))
    d = resolve(config["output"]["intermediary_dir"]) / "stata_check"
    d.mkdir(parents=True, exist_ok=True)

    if args.step == "prepare":
        subprocess.run([sys.executable, str(CODE / "03_estimate_thetas.py"), "--config", str(config_path),
                        "--check", str(args.cell), str(args.year),
                        "--export-check", str(d / "check_iv.csv")], check=True)
        rng = np.random.default_rng(20261012)
        frames = []
        for g, n in enumerate(GROUP_SIZES):
            x = np.round(rng.lognormal(0.2, 0.6, n), 2 if g % 2 else 6)   # odd groups have ties
            frames.append(pd.DataFrame({"grp": g, "x": x}))
        pd.concat(frames).to_csv(d / "pctile_synth.csv", index=False)
        logger.info(f"wrote check_iv.csv and pctile_synth.csv to {d}")
        return

    # compare
    from importlib import import_module
    est = import_module("03_estimate_thetas")
    rows = pd.read_csv(d / "check_iv.csv")
    rows["k_lag2"] = rows["k_lag"] ** 2
    b, se, _ = est.iv_gmm(*est.design(rows))
    st = pd.read_csv(d / "iv_stata.csv").set_index("term")
    names = ["c", *est.EXOG, "_cons"]
    tab = pd.DataFrame({"b_python": b, "b_stata": st.loc[names, "b"].to_numpy(),
                        "se_python": se, "se_stata": st.loc[names, "se"].to_numpy()}, index=names)
    logger.info("IV (Stata ivregress gmm vs numpy):\n" + tab.to_string(float_format=lambda x: f"{x:.10f}"))
    logger.info(f"max |b diff| {np.abs(tab.b_python - tab.b_stata).max():.2e}, "
                f"max |se diff| {np.abs(tab.se_python - tab.se_stata).max():.2e}")

    syn = pd.read_csv(d / "pctile_synth.csv")
    sp = pd.read_csv(d / "pctile_stata.csv").set_index("grp")
    out = []
    for g, x in syn.groupby("grp")["x"]:
        for p in PCTS:
            out.append({"grp": g, "n": len(x), "p": p, "python": stata_pctile(x.to_numpy(), p),
                        "stata": sp.loc[g, f"p{p}"]})
    out = pd.DataFrame(out)
    out["diff"] = (out["python"] - out["stata"]).abs()
    logger.info("egen pctile vs stata_pctile:\n" + out.to_string(index=False))
    logger.info(f"max |pctile diff| {out['diff'].max():.2e}")


if __name__ == "__main__":
    main()
