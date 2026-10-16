# Financial Domain

## Boundary

The Financial domain covers **reported and adjusted/non-GAAP metrics from company filings** for a small HK-listed universe:

- `0700.HK` Tencent Holdings Limited
- `0005.HK` HSBC Holdings plc
- `9988.HK` Alibaba Group Holding Limited

It does **not** cover market prices, news headlines, or LLM-written opinions.

Primary tools:

- `get_financial_metric`
- `get_filing_summary`

Data file: `data/mock/financial.json` (structured snapshots only; **no PDFs in git**).

## Annual vs interim

| `report_type` | Meaning |
|---------------|---------|
| `annual` | Full-year report / annual results |
| `interim` | Half-year / interim results |

Default when `period` is omitted: **latest annual** report. Interim must be requested explicitly via `report_type="interim"` and/or an interim `fiscal_year_label` / `period_end`.

## `fiscal_year_label` vs `period_end`

- `fiscal_year_label`: human label such as `FY2025`, `FY2026`, `1H2026`
- `period_end`: actual period end date (`YYYY-MM-DD`) — **required for unambiguous grounding**

Never rely on the label alone across companies: Alibaba’s fiscal year differs from Tencent/HSBC.

## Company period conventions (v1)

| Ticker | Latest annual | `period_end` | Latest interim in v1 | `period_end` |
|--------|---------------|--------------|----------------------|--------------|
| 0700.HK | FY2025 | 2025-12-31 | 1H2026 | 2026-06-30 |
| 0005.HK | FY2025 | 2025-12-31 | 1H2026 | 2026-06-30 |
| 9988.HK | FY2026 | **2026-03-31** | *(not in v1; schema allows future add)* | — |

Alibaba FY2026 is the year ended **31 March 2026**, not calendar year 2026.

## Data source rules

Allowed ground-truth sources:

- Company Investor Relations official site
- HKEX official disclosures

Not allowed as ground truth:

- News websites, blogs, search snippets
- Memory / guessed figures

## Reported vs adjusted

| `metric_type` | Examples |
|---------------|----------|
| `reported` | IFRS / US GAAP headline figures |
| `adjusted` | non-IFRS, non-GAAP, constant-currency excl. notable items |

Tools never silently mix these. Request `metric_type` explicitly when asking for adjusted figures.

## `verification_status`

| Status | Meaning |
|--------|---------|
| `verified` | Number taken from an official IR/HKEX document text in this project pass |
| `pending` | Not yet manually verified (or page-level check incomplete) |

`source_page` may be `null` when the PDF page was not manually recorded; the metric can still be `verified` if the value itself was confirmed from the official release text.

## Benchmark ambiguity rules

Benchmark questions must state:

- full year ended `<date>`, or
- six months / half-year ended `<date>`

Avoid vague wording like “2026 revenue” unless the item is an **error/ambiguity** test.

Formal scorable set: `benchmarks/questions/financial_questions_v1.jsonl`  
Draft/pending: `benchmarks/questions/financial_questions_draft_pending.jsonl`  
The evaluation runner **skips** rows with `verification_status: pending`.

## Real data later (out of scope for this PR)

Future REST/MCP connectors can replace the JSON snapshot loader. Keep the same report/metric schema and source URL fields so evaluation remains comparable. Do not add paid APIs in this phase.
