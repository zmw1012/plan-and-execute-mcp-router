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

## Document types (do not mix names)

| `document_type` | Meaning |
|-----------------|---------|
| `annual_report` | Formal Annual Report |
| `interim_report` | Formal Interim Report |
| `annual_results_announcement` | Earnings / results announcement (not a formal annual report) |
| `interim_results_announcement` | Interim/quarterly results announcement |

v1 sources use formal **Annual Report** / **Interim Report** only. Do not label a results announcement as an “annual report”.

`audit_status` must match the cited document:

- Formal annual report with audited statements → `audited`
- Interim report with review conclusion → `reviewed`
- Results announcement whose statements are unaudited → `unaudited` (not used in v1 after Alibaba switched to the Annual Report)

## Annual vs interim

| `report_type` | Meaning |
|---------------|---------|
| `annual` | Full-year report / annual results period |
| `interim` | Half-year / interim results period |

Default when `period` is omitted: **latest annual** report. Interim must be requested explicitly via `report_type="interim"` and/or an interim `fiscal_year_label` / `period_end`.

## `fiscal_year_label` vs `period_end`

- `fiscal_year_label`: human label such as `FY2025`, `FY2026`, `1H2026`
- `period_end`: actual period end date (`YYYY-MM-DD`) — **required for unambiguous grounding**

Never rely on the label alone across companies: Alibaba’s fiscal year differs from Tencent/HSBC.

## Snapshot metadata (not period ends)

In `financial.json` → `meta`:

| Field | Meaning |
|-------|---------|
| `snapshot_created_date` | When this local JSON snapshot was assembled |
| `latest_source_publication_date` | Max `publication_date` among included source documents |

Do **not** use a report `period_end` as the snapshot creation date.

## Company period conventions (v1)

| Ticker | Latest annual | `period_end` | Latest interim in v1 | `period_end` |
|--------|---------------|--------------|----------------------|--------------|
| 0700.HK | FY2025 Annual Report | 2025-12-31 | 2026 Interim Report | 2026-06-30 |
| 0005.HK | FY2025 Annual Report & Accounts | 2025-12-31 | 2026 Interim Report | 2026-06-30 |
| 9988.HK | FY2026 Annual Report | **2026-03-31** | *(schema allows future add)* | — |

Alibaba FY2026 is the year ended **31 March 2026**, not calendar year 2026.

## Metric口径 fields

Tool API still accepts `metric` (`revenue` / `net_profit` / `eps`) for compatibility, but each stored row also has:

| Field | Purpose |
|-------|---------|
| `canonical_metric` | Precise company-specific metric name |
| `metric_definition` | Human-readable definition |
| `metric_type` | `reported` or `adjusted` |
| `adjustment_basis` | Full adjustment description when `metric_type=adjusted` |
| `eps_basis` | `basic` or `diluted` when metric is EPS |

Examples:

- Tencent `net_profit` reported → `profit_attributable_to_equity_holders`
- HSBC `net_profit` reported → `profit_after_tax`
- Alibaba `net_profit` reported → `net_income_attributable_to_ordinary_shareholders`
- HSBC adjusted revenue → `constant_currency_revenue_excluding_notable_items` with full `adjustment_basis`

These companies’ `net_profit` values are **not interchangeable** without reading `canonical_metric`.

## Evidence pages

For every `verification_status=verified` metric:

- `source_url` required
- `source_page_pdf` required (PDF page index starting at 1)
- `source_page_printed` optional
- `evidence_text` must be locatable on that PDF page

## Data source rules

Allowed ground-truth sources:

- Company Investor Relations official site
- HKEX official disclosures

Not allowed as ground truth:

- News websites, blogs, search snippets
- Memory / guessed figures

## Benchmark ambiguity rules

Benchmark questions must state:

- full year ended `<date>`, or
- six months / half-year ended `<date>`

Avoid vague wording like “2026 revenue” unless the item is an **error/ambiguity** test.

Formal scorable set: `benchmarks/questions/financial_questions_v1.jsonl`  
Draft/pending: `benchmarks/questions/financial_questions_draft_pending.jsonl`  
The evaluation runner **skips** rows with `verification_status: pending`.

## Evaluation limitation (important)

`EvaluationRunner` currently passes **`gold_tools` and `gold_arguments` directly** into the agent/router.

Therefore a `strict_task_success_rate` of 1.0 on financial_questions_v1 is only a **gold-path execution smoke test**. It does **not** measure agent tool-selection accuracy or end-to-end autonomous performance.

True tool-selection / routed-mode experiments require a later agent that chooses tools without gold arguments.

## Real data later (out of scope for this PR)

Future REST/MCP connectors can replace the JSON snapshot loader. Keep the same report/metric schema and source URL fields so evaluation remains comparable. Do not add paid APIs in this phase.
