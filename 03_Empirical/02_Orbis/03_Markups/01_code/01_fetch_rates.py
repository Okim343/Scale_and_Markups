"""Step 01: euro overnight rate on an EONIA basis, from the ECB Data Portal (public, no credentials).

DLEU's user cost uses the Fed funds rate. The euro-area analogue is the unsecured overnight
interbank rate. EONIA ends on 2021-12-31; from 2019-10-02 the ECB computed it as the euro
short-term rate (€STR) plus 8.5 bp. One series without a break (ORBIS_SPAIN_MARKUPS.md, D3):
published EONIA, annual mean of daily values, through user_cost.eonia_last_year, then
€STR + user_cost.estr_spread_pp. The overlap 2019-10 to 2021-12 is used to check the spread.

    python 01_fetch_rates.py [--config ...] [--refresh]

Writes (public data; ignored by the repo-wide *indata*/ rule, rerun this script to rebuild):
    <user_cost.rates_dir>/{eonia,estr}_daily.csv    raw ECB downloads
    <user_cost.rates_dir>/overnight_rate_annual.csv year, eonia, estr, rate_eonia_basis, source
"""

import argparse
import io

import pandas as pd
import requests

from markups_common import load_config, resolve, setup_logger

API = "https://data-api.ecb.europa.eu/service/data"
SERIES = {
    "eonia": ("EON", "D.EONIA_TO.RATE"),
    "estr": ("EST", "B.EU000A2X2A25.WT"),
}
START = "2010-01-01"


def fetch(name, out_dir, refresh, logger):
    """Daily series as (date, value); reuse the saved CSV unless refresh."""
    path = out_dir / f"{name}_daily.csv"
    if not path.exists() or refresh:
        flow, key = SERIES[name]
        r = requests.get(f"{API}/{flow}/{key}", params={"format": "csvdata", "startPeriod": START},
                         timeout=120)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
        raw[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"}) \
            .to_csv(path, index=False)
        logger.info(f"downloaded {name}: {len(raw):,} daily obs")
    d = pd.read_csv(path, parse_dates=["date"])
    return d.dropna(subset=["value"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="re-download saved series")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    logger = setup_logger("rates", config.get("log_level", "INFO"))
    uc = config["user_cost"]
    out_dir = resolve(uc["rates_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eonia, estr = (fetch(n, out_dir, args.refresh, logger) for n in ("eonia", "estr"))
    spread = uc["estr_spread_pp"]

    # Overlap check: EONIA should equal €STR + spread day by day from 2019-10-02.
    ov = eonia.merge(estr, on="date", suffixes=("_eonia", "_estr"))
    ov = ov[ov["date"] >= "2019-10-02"]
    gap = (ov["value_eonia"] - ov["value_estr"] - spread).abs()
    logger.info(f"overlap {len(ov)} days: max |EONIA - (€STR + {spread})| = {gap.max():.4f} pp")
    if len(ov) and gap.max() > 0.0015:
        raise RuntimeError("EONIA and €STR + spread disagree on the overlap; check the series")

    ann = pd.DataFrame({
        "eonia": eonia.groupby(eonia["date"].dt.year)["value"].mean(),
        "estr": estr.groupby(estr["date"].dt.year)["value"].mean(),
    })
    ann.index.name = "year"
    last = int(uc["eonia_last_year"])
    ann["rate_eonia_basis"] = ann["eonia"].where(ann.index <= last, ann["estr"] + spread)
    ann["source"] = ["eonia" if y <= last else f"estr+{spread}" for y in ann.index]
    # A year with only part of €STR (2019) is never spliced: it uses EONIA.
    ann = ann.dropna(subset=["rate_eonia_basis"]).reset_index()
    ann.to_csv(out_dir / "overnight_rate_annual.csv", index=False)
    logger.info("annual EONIA-basis rate (%):\n" + ann.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
