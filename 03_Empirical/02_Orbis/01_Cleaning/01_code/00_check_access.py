"""Check WRDS Orbis access for the Spain extraction (discovery only, pulls no data).

Run interactively; the wrds package prompts for username and password and can
save them to ~/.pgpass (outside the repo). No credentials are read from or
written to this repository.

    python 03_Empirical/02_Orbis/01_Cleaning/01_code/00_check_access.py

Writes 03_Empirical/02_Orbis/01_Cleaning/00_indata/discovery_log.md.
"""

import os
from pathlib import Path

import wrds

OUT_DIR = Path(__file__).resolve().parents[1] / "00_indata"
REQUIRED_FIN = ["bvdid", "conscode", "filing_type", "closdate",
                "turn", "opre", "mate", "staf", "tfas", "fias", "empl"]
TABLE_HINTS = ["ind_g_fins", "contact_info", "industry", "activ", "company_id"]


def main():
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
    lines = ["# Orbis discovery log", ""]

    libs = [l for l in db.list_libraries() if "bvd" in l.lower() or "orbis" in l.lower()]
    lines += ["## Libraries", *[f"- {l}" for l in libs], ""]
    if not libs:
        lines.append("No BvD/Orbis library visible: subscription not reachable via PostgreSQL.")

    for lib in libs:
        try:
            tables = db.list_tables(library=lib)
        except Exception as err:  # permission errors show up here
            lines += [f"## {lib}: cannot list tables ({err})", ""]
            continue
        hits = [t for t in tables if any(h in t.lower() for h in TABLE_HINTS)]
        lines += [f"## {lib} ({len(tables)} tables, {len(hits)} relevant)", ""]
        for t in hits:
            try:
                cols = set(db.describe_table(library=lib, table=t)["name"].str.lower())
            except Exception as err:
                lines.append(f"- `{t}`: describe failed ({err})")
                continue
            missing = [c for c in REQUIRED_FIN if c not in cols] if "fins" in t else []
            note = f" missing: {missing}" if missing else ""
            flags = [c for c in ("contact_ctryiso", "nr_months") if c in cols]
            lines.append(f"- `{t}` ({len(cols)} cols){note} {flags if flags else ''}")
        lines.append("")

    db.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "discovery_log.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
