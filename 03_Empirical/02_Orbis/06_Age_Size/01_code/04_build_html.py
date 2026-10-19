"""Fill the page templates with the aggregate results and write the pages.

    python3 01_code/04_build_html.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CODE = HERE / "01_code"
RES = HERE / "03_outdata"


PAGES = [  # template, output page, {token: results file}
    ("findings_template.html", "orbis_age_size_findings.html", {"__DATA__": "viz_data.json"}),
    ("kn_template.html", "orbis_labour_share_kn.html", {"__KN__": "kn_results.json"}),
]


def main():
    for template, page_name, tokens in PAGES:
        if not (CODE / template).exists():
            continue
        page = (CODE / template).read_text()
        for token, name in tokens.items():
            assert page.count(token) == 1, token
            page = page.replace(token, json.dumps(json.loads((RES / name).read_text())))
        (HERE / page_name).write_text(page)
        print(f"wrote {HERE / page_name}")


if __name__ == "__main__":
    main()
