"""Export one Orbis Spain cell in the CSV layout of Hubmer et al.'s est_gnr.m.

For running their original MATLAB estimator (03_Empirical/00_GNR_core/00_original_code) on
our data, to compare with the Python translation. Builds the same variables as the Python
driver (shared gnr_panel.construct_gnr_vars) and keeps complete cases, current and lagged,
as est_gnr.m requires ("cleaned such that there are no missing observations"). Trimming is
left to est_gnr.m itself.

Columns: id (int), year, ind, s, ls, r, k, m, l, w, lr, lk, lm, ll, lw. Labor l is the log
wage bill, so w = log(wage bill / labor) = 0; est_gnr.m only requires lw to be non-missing.

    python 04_export_matlab_input.py --cell 51 --out <dir>   # writes <dir>/firmdata_tfp.csv
"""

import argparse
import importlib.util
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parent
COLS = ["id", "year", "ind", "s", "ls", "r", "k", "m", "l", "w", "lr", "lk", "lm", "ll", "lw"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=CODE / "config_gnr_orbis_es.yaml")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("gnr_cell_driver", CODE / "01_run_gnr_cell.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    from gnr_panel import construct_gnr_vars

    config = yaml.safe_load(args.config.read_text())
    logger = drv.setup_logger(f"{args.cell}-export", "/dev/null", "INFO")
    df, _ = drv.load_cell(config, args.cell, logger)
    df = construct_gnr_vars(df, logger, build_materials_alt1=False, id_col=drv.CORE_ID)
    df["id"] = df[drv.CORE_ID].astype("category").cat.codes.astype(int) + 1
    df["ind"] = df["ind2d"]
    out = df[COLS].dropna().reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "firmdata_tfp.csv", index=False)
    print(f"cell {args.cell}: wrote {len(out):,} complete-case rows "
          f"({out['id'].nunique():,} firms) to {args.out / 'firmdata_tfp.csv'}")


if __name__ == "__main__":
    main()
