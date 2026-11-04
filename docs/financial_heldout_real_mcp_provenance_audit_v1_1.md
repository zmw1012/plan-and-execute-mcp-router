# Financial Held-out Real-MCP Provenance Audit v1.1

**Branch:** `feature/financial-heldout-test-v1`  
**Base candidate:** `benchmarks/heldout/financial_test_30_v1.jsonl` @ `f85d53f`  
**Active freeze:** `benchmarks/heldout/financial_test_30_v1_1.jsonl`  
**Catalog:** frozen Financial Real MCP v1 (unchanged)  
**Method:** direct Real-MCP stdio calls only (no LLM / embedding / agent / experiment runner)

## Executive finding

v1 evidence provenance was **invalid for mcp-hkexnews v0.1**: gold `source_document` / `source_url` / `source_page_pdf` pointed at formal Annual/Interim **Reports**, while gold operations require results **announcements**. v1.1 re-anchors every evidence item to a live-retrieved results announcement and live-verifies structured items against raw AKTools fields.

## A. Structured-answerable (10) — live AKTools

All 10 structured items were invoked via `aktools-mcp` `get_data` → `stock_financial_hk_analysis_indicator_em` / company profile interfaces (no LLM).

| ID | Live raw check | Result |
|---|---|---|
| fin_ht_001 | 00700 `报告期` REPORT_DATE 2026-06-30 `BASIC_EPS=12.639` 人民币 | passed |
| fin_ht_002 | same row `HOLDER_PROFIT=114115000000` | passed |
| fin_ht_005 | interim margin from OPERATE_INCOME/HOLDER_PROFIT → 28.44% | passed |
| fin_ht_006 | 1H/FY OPERATE_INCOME ratio → 53.37% | passed |
| fin_ht_021 | 09988 `年度` 2026-03-31 `BASIC_EPS=5.7` | passed |
| fin_ht_022 | profile `年结日=03-31` + REPORT_DATE 2026-03-31 | passed |
| fin_ht_023 | OPERATE_INCOME=1023670000000 | passed |
| fin_ht_024 | **blocker then corrected** — see below | corrected |
| fin_ht_025 | **blocker then corrected** — see below | corrected |
| fin_ht_026 | OPERATE_INCOME + BASIC_EPS | passed |

### Alibaba HOLDER_PROFIT semantic blocker (corrected, not silently reconciled)

Live East Money `HOLDER_PROFIT` for FY2026 = **103,592,000,000**.  
v1 gold used mock **105,904** (= “net income attributable to ordinary shareholders” in the results announcement).

Official FY2026 results announcement confirms **both** lines as distinct:

- Net income attributable to Alibaba Group Holding Limited: **103,592**
- Net income attributable to ordinary shareholders: **105,904**

v1.1 therefore **changed the question wording and gold** for `fin_ht_024`, `fin_ht_025`, and the structured leg of `fin_ht_027` to the HOLDER_PROFIT / Group Holding Limited metric that AKTools actually returns. Difficulty/company/ETOM/answerability counts preserved. No silent numeric overwrite of ordinary-shareholders gold as if it were HOLDER_PROFIT.

## B. Evidence-answerable (14) — live mcp-hkexnews

Minimum distinct downloads (release_date + short title_keyword; no bulk; cache under gitignored `outputs/hkexnews_cache`):

| Doc key | Retrieved title | Type | Release date | Official URL |
|---|---|---|---|---|
| tencent_fy2025 | ANNOUNCEMENT OF THE ANNUAL RESULTS FOR THE YEAR ENDED 31 DECEMBER 2025 | annual_results_announcement | 2026-03-18 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0318/2026031800388.pdf |
| tencent_1h2026 | ANNOUNCEMENT OF THE RESULTS FOR THE THREE AND SIX MONTHS ENDED 30 JUNE 2026 | interim_results_announcement | 2026-08-12 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0812/2026081200296.pdf |
| hsbc_fy2025 | 2025 results | annual_results_announcement | 2026-02-25 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0225/2026022500111.pdf |
| hsbc_1h2026 | 2026 Interim results | interim_results_announcement | 2026-08-04 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0804/2026080400325.pdf |
| alibaba_fy2026 | ANNOUNCEMENT OF THE MARCH QUARTER 2026 RESULTS AND FISCAL YEAR 2026 ANNUAL RESULTS | annual_results_announcement | 2026-05-13 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0513/2026051300653.pdf |

Every evidence item was re-anchored to one of these documents with verified page + snippet via `keyword_search` / `get_announcement_pages`. **No evidence item claims a full Annual/Interim Report was retrieved through mcp-hkexnews.**

## C. L4 cross-server semantic consistency (6)

| ID | AKTools leg | HKEX leg | Compatible |
|---|---|---|---|
| fin_ht_007 | FY2025 IFRS basic EPS 24.749 | FY2025 results ann. non-IFRS basic EPS 28.577 | yes |
| fin_ht_008 | FY2025 OPERATE_INCOME 751766e6 | same revenue in results ann. | yes |
| fin_ht_017 | HSBC profile `年结日=12-31` / English name | FY2025 results revenue 68,274 | yes |
| fin_ht_018 | HSBC profile English name contains HSBC | 1H2026 basic EPS 0.85 | yes |
| fin_ht_027 | HOLDER_PROFIT 103592 | non-GAAP NI 60,658 in results ann. | yes (after NI correction) |
| fin_ht_028 | OPERATE_INCOME 1023670e6 | revenue in results ann. | yes |

Initial audit false-failures on HSBC profile used English field names; live profile rows expose Chinese keys (`年结日`, `英文名称`). Reconfirmed live.

## D. L5 infeasibility (6)

| ID | Kind | Reason |
|---|---|---|
| fin_ht_009 | unsupported_capability | VAS segment revenue |
| fin_ht_010 | unavailable_period | 9M2026 |
| fin_ht_019 | unsupported_capability | CET1 |
| fin_ht_020 | ambiguous_unsafe_comparison | bank gross-margin cross-company |
| fin_ht_029 | unsupported_capability | Cloud segment |
| fin_ht_030 | unavailable_period | Alibaba 6M to 2025-09-30 |

All remain infeasible for baseline and proposed methods under the **same** frozen catalog. No Segment/CET1/NIM tools added.

## E. Versioning artifacts

| Artifact | Location |
|---|---|
| Active benchmark | `benchmarks/heldout/financial_test_30_v1_1.jsonl` |
| Superseded candidate | `benchmarks/heldout/financial_test_30_v1.jsonl` |
| Held-out README | `benchmarks/heldout/README.md` |
| This audit | `docs/financial_heldout_real_mcp_provenance_audit_v1_1.md` |
| Machine audit log | `outputs/financial_heldout_provenance_audit_v1_1/audit.jsonl` (gitignored) |
| Validator | `scripts/validate_financial_heldout_test_v1_1.py` |
| Tests | `tests/test_financial_heldout_benchmark_v1_1.py` |
| Live audit driver | `scripts/audit_financial_heldout_real_mcp_v1_1.py` |

## Confirmations

- Catalog v1 YAML not modified.
- No agent run on Development or Held-out.
- No generation LLM / embedding / DeepSeek / OpenAI calls.
- Direct free/local Real-MCP calls only.
- `.env` untouched.
