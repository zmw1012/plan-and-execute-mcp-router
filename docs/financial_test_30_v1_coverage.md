# Financial Held-out Test 30 v1 — Coverage

Catalog under audit: `configs/operations/financial_real_mcp.yaml`  
Frozen inventory: **2 servers**, **9 raw MCP tools**, **21 Operation Resolver entries**.

## Distribution tables

### ETOM level

| Level | Count | IDs |
|---|---|---|
| L1 | 6 | `fin_ht_001`, `002`, `011`, `012`, `021`, `022` |
| L2 | 6 | `fin_ht_003`, `004`, `013`, `014`, `023`, `024` |
| L3 | 6 | `fin_ht_005`, `006`, `015`, `016`, `025`, `026` |
| L4 | 6 | `fin_ht_007`, `008`, `017`, `018`, `027`, `028` |
| L5 | 6 | `fin_ht_009`, `010`, `019`, `020`, `029`, `030` |

### Company

| Company | Ticker | Count |
|---|---|---|
| Tencent | 0700.HK | 10 |
| HSBC | 0005.HK | 10 |
| Alibaba | 9988.HK | 10 |

### Answerability

| Class | Count |
|---|---|
| `structured_answerable` | 10 |
| `evidence_answerable` | 14 |
| `intentionally_infeasible` | 6 |

### Company × ETOM

|  | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Tencent | 2 | 2 | 2 | 2 | 2 |
| HSBC | 2 | 2 | 2 | 2 | 2 |
| Alibaba | 2 | 2 | 2 | 2 | 2 |

## Per-question coverage map

| ID | Co | L | Class | Gold servers | Primary gold ops (abbrev.) | Period | Notes |
|---|---|---|---|---|---|---|---|
| fin_ht_001 | T | L1 | structured | aktools | analysis_indicator | 1H2026 | Basic EPS |
| fin_ht_002 | T | L1 | structured | aktools | analysis/report | 1H2026 | IFRS profit only |
| fin_ht_003 | T | L2 | evidence | hkexnews | list/download/keyword/pages | FY2025 | Non-IFRS basic EPS |
| fin_ht_004 | T | L2 | evidence | hkexnews | list/keyword/pages | 1H2026 | Non-IFRS profit only |
| fin_ht_005 | T | L3 | structured | aktools | analysis_indicator | 1H2026 | Interim IFRS margin |
| fin_ht_006 | T | L3 | structured | aktools | analysis_indicator | 1H2026/FY2025 | Interim/annual revenue % |
| fin_ht_007 | T | L4 | evidence | aktools+hkexnews | analysis + keyword/pages | FY2025 | IFRS + non-IFRS basic EPS |
| fin_ht_008 | T | L4 | evidence | aktools+hkexnews | analysis + list/keyword | FY2025 | Revenue corroboration |
| fin_ht_009 | T | L5 | infeasible | — | — | FY2025 | VAS segment refuse |
| fin_ht_010 | T | L5 | infeasible | — | — | 9M2026 | Unavailable period refuse |
| fin_ht_011 | H | L1 | evidence | hkexnews | list/keyword/pages | 1H2026 | Basic EPS |
| fin_ht_012 | H | L1 | evidence | hkexnews | list/keyword/pages | 1H2026 | PAT only |
| fin_ht_013 | H | L2 | evidence | hkexnews | list/download/keyword | FY2025 | Announcement-path EPS |
| fin_ht_014 | H | L2 | evidence | hkexnews | list/keyword/pages | 1H2026 | Evidence-only revenue |
| fin_ht_015 | H | L3 | evidence | hkexnews | keyword/pages | 1H2026 | Interim PAT margin |
| fin_ht_016 | H | L3 | evidence | hkexnews | keyword/pages | FY2025 | Fetch-then APM−reported |
| fin_ht_017 | H | L4 | evidence | aktools+hkexnews | profile + keyword | FY2025 | YE + revenue |
| fin_ht_018 | H | L4 | evidence | aktools+hkexnews | profile + keyword/pages | 1H2026 | Name + interim EPS |
| fin_ht_019 | H | L5 | infeasible | — | — | FY2025 | CET1 refuse |
| fin_ht_020 | H | L5 | infeasible | — | — | FY2025 | Unsafe gross-margin compare |
| fin_ht_021 | A | L1 | structured | aktools | analysis_indicator | FY2026 | Basic EPS only |
| fin_ht_022 | A | L1 | structured | aktools | profile/analysis | FY2026 | Period-end 2026-03-31 |
| fin_ht_023 | A | L2 | structured | aktools | analysis_indicator | FY2026 | REPORT_DATE→revenue |
| fin_ht_024 | A | L2 | structured | aktools | analysis/report | FY2026 | REPORT_DATE→reported NI |
| fin_ht_025 | A | L3 | structured | aktools | analysis_indicator | FY2026 | Revenue + NI pair |
| fin_ht_026 | A | L3 | structured | aktools | analysis_indicator | FY2026 | Revenue + basic EPS |
| fin_ht_027 | A | L4 | evidence | aktools+hkexnews | analysis + keyword/pages | FY2026 | Reported NI + non-GAAP NI |
| fin_ht_028 | A | L4 | evidence | aktools+hkexnews | analysis + list/keyword | FY2026 | Revenue corroboration |
| fin_ht_029 | A | L5 | infeasible | — | — | FY2026 | Cloud segment refuse |
| fin_ht_030 | A | L5 | infeasible | — | — | 6M to 2025-09-30 | Unavailable interim refuse |

## Catalog answerability summary

Under frozen Financial v1:

- **Structured path strong for:** Tencent/Alibaba reported consolidated P&L / EPS via `aktools.stock_financial_hk_analysis_indicator_em` (+ report_em when needed); profile year-end helpers.
- **Evidence path required for:** all non-IFRS / non-GAAP amounts; all HSBC absolute USD amounts and APM figures; page/keyword corroboration.
- **Cross-server L4:** six items deliberately combine AKTools + `mcp_hkexnews`.
- **Intentionally out of scope (L5):** segment revenues, CET1, unavailable periods, unsafe bank gross-margin comparisons.

## Duplicate check (Development vs Held-out)

Against `financial_mcp_free_30_v1.jsonl` and `financial_real_mcp_coverage_v1.jsonl`:

- Exact normalized question match: **0**
- Token Jaccard ≥ 0.85 near-duplicate: **0**

Development remains untouched.

## Official evidence basis

Numeric golds for answerable items are aligned to the verified mock snapshot in `data/mock/financial.json` (official IR / annual & interim report figures previously locked for the project). L5 items have no numeric gold by design.
