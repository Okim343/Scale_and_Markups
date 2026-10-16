"""Fill findings_template.html with the aggregate results and write the findings page.

    python3 01_code/04_build_html.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CODE = HERE / "01_code"
RES = HERE / "03_outdata"


def main():
    page = (CODE / "findings_template.html").read_text()
    for token, name in [("__DATA__", "viz_data.json"), ("__LAB__", "labour_results.json")]:
        assert page.count(token) == 1, token
        page = page.replace(token, json.dumps(json.loads((RES / name).read_text())))
    out = HERE / "orbis_age_size_findings.html"
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
