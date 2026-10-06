"""Second access check: can we actually SELECT Spanish data, and how much is there?

Runs only server-side aggregates (counts) plus a 3-row sample; downloads no panel.
Credentials are prompted by the wrds package, never stored in the repo.

    python 03_Empirical/02_Orbis/01_Cleaning/01_code/00b_check_spain_counts.py

Writes 03_Empirical/02_Orbis/01_Cleaning/00_indata/spain_counts_log.md.
"""

import os
from pathlib import Path

import wrds

OUT_DIR = Path(__file__).resolve().parents[1] / "00_indata"
LIB = "bvd"
SIZES = ["l", "m", "s"]
VARS = ["turn", "opre", "mate", "staf", "tfas", "fias", "empl"]
FILTER = "contact_ctryiso = 'ES' AND conscode IN ('U1', 'U2')"


def num(col):
    """Cast a column that may be stored as text to numeric; non-numeric text -> NULL."""
    return (f"(CASE WHEN {col}::text ~ '^-?[0-9]+(\\.[0-9]+)?([eE][-+]?[0-9]+)?$' "
            f"THEN {col}::text::numeric END)")


def main():
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
    out = ["# Spain counts log", ""]

    # 1. Column types of the variables we use (several are stored as text).
    types = db.describe_table(library=LIB, table="ob_ind_g_fins_l")
    types = types[types["name"].str.lower().isin(
        VARS + ["closdate", "closdate_year", "nr_months", "conscode", "filing_type",
                "emp_orig_range_value", "op_rev_original_range_value"])]
    out += ["## Column types, ob_ind_g_fins_l", "",
            types[["name", "type"]].astype(str).to_markdown(index=False), ""]

    # 2. Spanish unconsolidated rows by year, with shares of positive values.
    # coalesce: a missing value counts as "not positive" (avg would otherwise skip NULLs).
    pos = ", ".join(f"avg(coalesce(({num(v)} > 0)::int, 0)) AS {v}_pos" for v in VARS)
    pos += (", avg(coalesce((coalesce(turn, opre) > 0 AND mate > 0 AND staf > 0 "
            "AND tfas > 0)::int, 0)) AS gnr_all")
    flags = ("avg((emp_orig_range_value IS NOT NULL)::int) AS empl_range, "
             "avg((op_rev_original_range_value IS NOT NULL)::int) AS opre_range")
    for size in SIZES:
        table = f"{LIB}.ob_ind_g_fins_{size}"
        sql = (f"SELECT closdate_year AS year, count(*) AS n, count(DISTINCT bvdid) AS firms, "
               f"avg(coalesce(({num('nr_months')} = 12)::int, 0)) AS m12, {pos}, {flags} "
               f"FROM {table} WHERE {FILTER} GROUP BY 1 ORDER BY 1")
        try:
            res = db.raw_sql(sql)
            out += [f"## {table}", "", res.round(3).to_markdown(index=False), ""]
        except Exception as err:
            out += [f"## {table}: query failed", f"{err}", ""]

    # 3. Industry-code coverage: Spanish firms with a core / primary / national code.
    codes = ["naceccod2", "nacepcod2", "naicsccod2017", "naicspcod2017", "natpcod"]
    has = ", ".join(f"count(DISTINCT CASE WHEN {c} IS NOT NULL THEN bvdid END) AS {c}"
                    for c in codes)
    for size in SIZES:
        table = f"{LIB}.ob_industry_classifications_{size}"
        sql = (f"SELECT count(DISTINCT bvdid) AS firms, count(*) AS rows, {has}, "
               f"string_agg(DISTINCT natclass, ' | ') AS natclass "
               f"FROM {table} WHERE contact_ctryiso = 'ES'")
        try:
            out += [f"## {table} (ES firms with each code)", "",
                    db.raw_sql(sql).to_markdown(index=False), ""]
        except Exception as err:
            out += [f"## {table}: query failed", f"{err}", ""]

    db.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "spain_counts_log.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
