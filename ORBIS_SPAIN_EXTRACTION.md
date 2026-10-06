# Orbis Spain extraction: task spec for agents

Pull Spanish firm-level financials from BvD Orbis on WRDS. The download must contain exactly what the
GNR production-function estimation and the materials-based markup need, and nothing else.
This file is the full spec: read all of it before connecting to WRDS.

## 0. Context (why this data)

The paper (`02_Drafts/main.typ`) currently calibrates on Compustat. Compustat has three problems:
labor cost (`xlr`) covers about 20% of firm-years, it contains only public firms (so concentration is too low),
and it is too thin for sectors finer than NAICS2. Spanish Orbis fixes all three. Spain has mandatory
filing, the wage bill (`STAF`) is reported, and it is one of the three largest Orbis samples in our main
reference, Hubmer, Chan, Ozkan, Salgado & Hong, *Scalable versus Productive Technologies*
(`../Scale_and_Markups_static/01_Docs/salgado_rts_firm_specific.pdf`, Appendix C.3).
We replicate their Orbis variable choices.

Downstream uses:
- **GNR** (Gandhi-Navarro-Rivers) gross-output production function. Inputs: revenue, intermediates
  (flexible), labor (wage bill), capital. Estimated within NAICS2 cells.
- **Markup**: μ = ε^m · revenue / `MATE`, where ε^m is the GNR materials elasticity.
  Cost weights are `MATE + STAF + R·K`. **No COGS or SG&A is needed.**
- **Concentration**: CR4, CR20 and HHI within NACE Rev.2 4-digit sectors, sales-based.

## 1. Credentials and security (read first)

- Access uses a WRDS account that belongs to someone else. **Never write the username or password
  into any file in this repo**, any script, any log, or the chat. Use the `wrds` Python package's
  standard mechanism (`~/.pgpass` created by `wrds.Connection()` on first login, or the
  `WRDS_USERNAME` env var with the password prompted interactively).
- Make sure `~/.pgpass` is outside the repo. Do not commit anything containing credentials.
- WRDS data is licensed. Raw extracts must not be committed. Add the output directory to `.gitignore`
  (see §5) **before** writing any data.
- Read-only: only `SELECT` queries. Never query anything outside the tables listed below.

## 2. Discover exact table and column names (do not guess)

WRDS PostgreSQL library and table names for Orbis are **not hard-coded here on purpose**, because
they must be verified. Use:

```python
import wrds
db = wrds.Connection()                      # prompts; creates ~/.pgpass if asked
libs = [l for l in db.list_libraries() if 'bvd' in l.lower() or 'orbis' in l.lower()]
for lib in libs:
    print(lib, db.list_tables(library=lib))
db.describe_table(library=LIB, table=TABLE)  # column names and types
```

What to look for, based on the WRDS Orbis overview naming conventions:
- Size suffixes: `_l` (very large and large), `_m`, `_s`, `_lm`, `_lms`. **Use `_lms` (all sizes).**
  If `_lms` does not exist, use `_l`, `_m` and `_s` and append them.
- Currency: names with `_usd` or `_eur` are converted. Names without them are in **local currency,
  which for Spain is EUR. Prefer the local-currency table.** If only `_eur` exists, that is fine too.
  Do not use `_usd`.
- Financials: industrial-company template, "global financials and ratios". On the SAS server it is
  `ob_ind_g_fins_*` or `ob_w_ind_g_fins_cfl_*`. Use the industrial template only, not banks or insurance.
- Country: `contact_ctryiso` lives in `ob_contact_info_*`.
- Industry codes: the industry-classifications table in the "Activity" category, which holds the
  NACE Rev.2 and NAICS 2017 core codes.
- Keys: `bvdid`, `conscode`, `filing_type`, `closdate` jointly identify a financials record.

Before extracting anything, **write a short discovery log** to
`03_Empirical/02_Orbis/01_Cleaning/00_indata/discovery_log.md`. It should list the chosen library and tables,
and each required column → actual column name. If any required column cannot be found, stop and report
to the user. Do not substitute silently.

### Discovery results (2026-10-06, from `00_check_access.py`)
- Library `bvd` (mirrored as `bvd_orbis_large`, `bvd_orbis_medium`, `bvd_orbis_small`). All three size tiers are accessible.
- **No `_lms` table exists in local currency or EUR**; combined `_lms` tables are USD-only. Use the
  local-currency (EUR for Spain) tables **`bvd.ob_ind_g_fins_l`, `bvd.ob_ind_g_fins_m`, `bvd.ob_ind_g_fins_s`**
  and append them.
- All required financials columns are present, and `contact_ctryiso` and `nr_months` are in the financials
  tables themselves (no join needed for the country filter).
- Industry codes: `bvd.ob_industry_classifications_{l,m,s}`.
- Open question: what the `_int` twin tables (`ob_ind_g_fins_int_*`, 6 fewer columns) are.
  `00b_check_spain_counts.py` diffs them; use the plain tables unless that check says otherwise.
- `00b_check_spain_counts.py` also confirms SELECT permission and reports Spanish row counts by year/size.

### Findings from `00b` (first run, 2026-10-06)
- SELECT works. `_int` tables are the plain tables minus `closdate_char`, `closdate_year`, `df_employees`,
  `df_oprev`, `emp_orig_range_value`, `op_rev_original_range_value`. **Use the plain tables** and also
  extract `closdate_year`, `emp_orig_range_value` and `op_rev_original_range_value`. The range flags mark
  employment and revenue reported as ranges (the OECD rounding problem) and are used in cleaning.
- `nr_months` is stored as **text**; cast before comparing. Do not assume the financial columns are numeric
  either: check `describe_table` types and cast explicitly.
- ES conscode counts (size l): U1 491,905 · U2 80,302 · C2 53,976 · C1 5,722 · LF 44.
- `ob_industry_classifications_*` has **several rows per firm** (core, primary and secondary codes on
  separate rows). Columns: NACE `naceccod2` (core) / `nacepcod2` (primary) / `nacescod2` (secondary);
  NAICS `naicsccod2017` / `naicspcod2017` / `naicsscod2017`; national `natpcod` (Spanish CNAE, which
  follows NACE Rev.2 at 4 digits). Collapse to one row per firm by priority: core → primary → national
  primary. **Never use secondary codes.**

### Findings from `00b` (second run, 2026-10-06; ES, U1/U2)
- Financial columns are DOUBLE PRECISION; only `nr_months` is text. `closdate_year` is numeric.
- **Coverage break in 2015**: rows jump about 3× (size l 11.6k → 38.4k; m 69k → 188k; s 361k → 578k),
  consistent with the about 10-year rolling window. Pre-2015 rows are a surviving subset. **Baseline window:
  2015–2024** (2025 has only about 1–1.5k rows per tier; 2024 is slightly incomplete for small firms).
  Still extract all years (cheap), but analyse 2015–2024.
- Rows with revenue, `mate`, `staf` and `tfas` all > 0, 2015–2024: size l ≈ 285k, m ≈ 1.49M,
  s ≈ 3.12M, **total ≈ 4.9M firm-years** (shares about 0.68 / 0.72 / 0.53 of rows). `mate` is the binding
  variable (about 0.72–0.80 positive for l/m, about 0.68 for s).
- `nr_months == 12` for about 97–98% of rows. Range flags are never set for Spain (no rounding problem).
- Industry codes: `naceccod2` and `naicsccod2017` are present for more than 99.5% of Spanish firms in every tier.

### Extraction done (2026-10-06, `01_extract_orbis_es.py`, closdate_year 2014–2024)
- `orbis_es_financials.parquet` (353 MB): 8.78M rows, 1.39M firms, no duplicate keys.
  `orbis_es_firms.parquet` (17 MB): one row per firm, matched for all financial firms.
- Usable firm-years (revenue, mate, staf, tfas > 0, 12-month accounts): 5.02M in total; about 450–500k a year
  for 2015–2023, 204k in 2014, 450k in 2024.
- **Window decision: baseline 2015–2023, with 2014 used only for GNR lags.** 2024 is about 91% of 2023 and the
  shortfall is concentrated in small firms (late filers), so it is selectively incomplete. It stays in the file
  as a robustness year.
- Cleaning notes: 2,382 rows with a bvdid not starting `ES` (drop); 2,464 firm-years with more than one account
  (deduplicate); use the **core** codes `naceccod2` / `naicsccod2017` (unique per firm). The primary codes
  `nacepcod2` / `naicspcod2017` have several values for 19k / 598k firms, so don't use them.

## 3. What to extract

### Global filters (apply in SQL, server side)
| Filter | Value |
|---|---|
| Country | `contact_ctryiso = 'ES'` (join via `bvdid`). As a sanity check, Spanish `bvdid`s start with `ES`. |
| Template | industrial companies (implied by the financials table chosen) |
| Consolidation | `conscode IN ('U1','U2')` (unconsolidated only, no double counting with group accounts) |
| Size | all (`_lms`) |
| Years | all available (WRDS has about a 10-year rolling window; expect roughly 2016–2025) |

Filter on the server: first pull the list of Spanish `bvdid`s, then pull financials restricted to them.
**Chunk by year of `closdate`** to avoid timeouts. Never pull the global table.

### File A: `orbis_es_financials.parquet` (one row per bvdid × conscode × filing_type × closdate)
| Concept | Orbis mnemonic | Use |
|---|---|---|
| firm ID | `bvdid` | key |
| consolidation code | `conscode` | key |
| filing type | `filing_type` | key |
| closing date | `closdate` | key, year |
| months in accounting period | `nr_months` (verify name) | keep only 12-month accounts |
| sales | `TURN` | revenue |
| operating revenue | `OPRE` | revenue fallback |
| material costs | `MATE` | flexible input and markup denominator |
| cost of employees | `STAF` | labor (wage bill) |
| tangible fixed assets | `TFAS` | capital, baseline |
| total fixed assets | `FIAS` | capital, Hubmer et al. definition (robustness) |
| number of employees | `EMPL` | optional robustness (≥10 employees sample) |

Column names on WRDS may be lower case (`turn`, `opre`, ...). Keep them lower case in the output.

### File B: `orbis_es_firms.parquet` (one row per bvdid)
| Concept | Use |
|---|---|
| `bvdid` | key |
| `contact_ctryiso` | check (all must equal `ES`) |
| NACE Rev.2 core code (4-digit) | concentration sectors |
| NAICS 2017 core code | NAICS2 cells for GNR (comparable with Hubmer et al.) |

If a firm has several codes, keep the **core/primary** code only.

### Do NOT extract
Ownership or shareholders, subsidiaries, directors, advisors, cash-flow-only variables, ratios,
listed-company detail, interim accounts, bank or insurance tables, `_usd` tables, exchange-rate
columns, Amadeus, Bank Focus, or any other country.

## 4. Raw-file validation (before handing back)

Write `03_Empirical/02_Orbis/01_Cleaning/00_indata/extraction_report.md` with:
1. Row counts of A and B, number of unique `bvdid`, and row counts by `closdate` year.
2. Key uniqueness: `(bvdid, conscode, filing_type, closdate)` is unique in A, and `bvdid` is unique in B.
3. Share of firms in A that are matched in B, and the share with a non-missing NACE/NAICS code.
4. For each of `turn, opre, mate, staf, tfas, fias, empl`: the share non-missing, and the share > 0, by year.
5. Share of rows with `nr_months == 12`.
6. Expected magnitude: Hubmer et al. report about **3.8M Spanish firm-years after cleaning** over a longer
   window (about 1990s–2019). With a roughly 10-year window, expect on the order of 1–2.5M usable
   firm-years. If the count is far off, flag it, don't "fix" it.
7. The exact SQL actually run (credentials stripped).

## 5. Where things go

```text
03_Empirical/02_Orbis/
├── 01_Cleaning/
│   ├── 00_indata/          # raw extracts (GITIGNORED) + discovery_log.md + extraction_report.md
│   ├── 01_code/            # 01_extract_orbis_es.py (the extraction script, committed)
│   └── 01_intermediary/    # cleaned panel (GITIGNORED), created in the next stage, not this one
├── 02_GNR/                 # per-cell GNR on the shared 03_Empirical/00_GNR_core estimator
└── 03_Markups/
```

Data stays out of git through the repo-wide `.gitignore` rules for any `*indata*/` and
`*intermediary*/` directory (the original `02_orbis_spain` rules predate the 2026-10-06 restructure).

The extraction script must be rerunnable, read no credentials from the repo, and follow repo style
(`snake_case`, `pathlib`/`os.path.join`, `mkdir(parents=True, exist_ok=True)`).

## 6. Next stage (separate task; do not do during extraction)

**Done (2026-10-06): see `ORBIS_SPAIN_CLEANING.md`.** Deviation from the list below: deflation uses
Spanish HICP for every money variable (Hubmer et al. App. C.3), because Eurostat has no Spanish
output or intermediate-consumption deflator at A64. Sector deflators are kept as robustness columns.

These steps are listed so the extraction does not drop something they need.
Cleaning follows Hubmer et al. Appendix C.3:
- Drop duplicate records, missing industry code, and firms whose `bvdid` country prefix ≠ `ES`.
- Keep `nr_months == 12`. If a firm has several accounts per year, keep one deterministically
  (prefer U1 over U2, then the latest `closdate`).
- Revenue = `turn`, or `opre` if `turn` is missing.
- Drop missing, zero or negative revenue, `mate`, `staf`, or capital (`tfas` baseline; `fias` robustness).
- Deflate with Spanish sector output and intermediate price indices and a capital deflator
  (INE or EU KLEMS). These are **not from WRDS** and are fetched separately.
- GNR within NAICS2 cells, then the markup μ = ε^m · revenue / `mate`, aggregated with cost weights
  `mate + staf + R·K`.
