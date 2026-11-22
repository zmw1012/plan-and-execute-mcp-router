# News MCP Candidate Audit (Discovery v1)

**Branch:** `feature/real-news-mcp-discovery-v1`  
**Worktree:** `hk-equity-mcp-agent-news`  
**Base:** `ac49c48`  
**Method:** direct free MCP / interface smoke only. No Agent, LLM, embedding, or paid API.  
**Temporal policy:** max gold date **2026-08-12** Asia/Hong_Kong.

Financial v1.3 and Market v1.1 artifacts are **readonly** and were not modified.

## Audited servers

| Server | Package / repo | Transport | Auth | Free | Live | Decision |
|---|---|---|---|---|---|---|
| **mcp-hkexnews** | `agentladle-mcp-hkexnews` 0.1.0 | stdio | none | yes | yes | **accepted primary** |
| **AKTools** | `aktools` 0.1.4 + `akshare` 1.20.22 | stdio | none | yes | yes | **accepted supporting** |
| **ah-disclosure-kit** | MIT kit, 40 tools | stdio | none | yes | yes | **supporting with risks** |
| **FTShare-MCP** | hosted gateway | HTTP | key recommended | no (quota) | tools/list only historically | **blocked** |
| **mock NewsServer** | in-repo Python | n/a | n/a | n/a | mock | **rejected** |

## Raw MCP tools

| Server | Raw tools |
|---|---|
| AKTools | `search_interfaces`, `describe_interface`, `get_data` |
| mcp-hkexnews | `lookup_stock_code`, `list_hkexnews_announcements`, `download_hkexnews_announcement`, `parse_hkexnews_announcement`, `keyword_search`, `get_announcement_pages` |
| ah-disclosure | 40 tools (filings/annual/ingest/profile/LLM/cleanup, …) |

## Logical News operations (candidate catalog)

See `configs/operations/news_real_mcp_candidate_v1.yaml`.

| Decision | Count | Examples |
|---|---|---|
| primary | 5 | hkexnews list / download / parse / keyword / pages |
| supporting | 5 | lookup; AKTools meta×2; ah `resolve_company`; ah `search_annual_report` |
| distractor | 7 | **`stock_news_em` (runtime-only)**; ah `search_filings`; A-share notice/gsrl; global EM; CCTV; Caixin |
| blocked | 1 | FTShare announcements list |
| rejected | 2 | `stock_individual_notice_report`; mock NewsServer |
| **total logical ops** | **20** | |

`equal_function_sets: []` — media ≠ official disclosure ⇒ **Exact L2 = 0**.  
`stock_news_em`: `formal_gold_eligible=false` (supporting_runtime_only / distractor_for_formal_gold).

### Dependency smoke correction

Re-ran list→download→parse→`keyword_search(keywords=[...])`→pages for Tencent/HSBC/Alibaba with fixed ≤2026-08-12 corporate-event announcements. All three `success=true`. See `docs/news_mcp_discovery_v1/hkexnews_dependency_chain_summary.json`.

## Live coverage (≤2026-08-12 window where applicable)

| Path | Tencent 00700 | HSBC 00005 | Alibaba 09988 |
|---|---|---|---|
| `list_hkexnews_announcements` 2026-07-01..08-12 | ✅ n=8 | ✅ n=32 | ✅ n=13 |
| download→parse→keyword (`keywords` list) | — | ✅ sample | — |
| `stock_news_em` floating | ✅ rows, **0 before cutoff** | ✅ / 0 | ✅ / 0 |
| ah `search_annual_report` | ✅ title+URL (2025 AR, publish 2026-04-09) | prior | prior |
| ah `search_filings` + date args | ⚠️ returns post-cutoff | ⚠️ | ⚠️ |

## Source-class separation (enforced)

| Class | What | News gold? |
|---|---|---|
| `official_disclosure` | HKEXnews announcements (NDDR, AGM notice, monthly return, results **title/date**, …) | yes (event/title/date/snippet) |
| `company_ir` / annual report PDF discovery | ah `search_annual_report` | title/URL/date only — **not** Financial numbers |
| `financial_media` | East Money / Caixin / CCTV / global flash | live OK; **not** historical cutoff gold via `stock_news_em` |
| Results announcement numeric tables | revenue/EPS/etc. | **Financial domain — out of News gold** |

## Fail-closed observations

- Invalid stock `99999` → `Stock code not found` (hkexnews).
- Post-cutoff list window still returns data — policy must exclude from gold, not rely on empty API.
- `stock_news_em` has no date params → cannot fail closed by API; offline filter yields empty before cutoff → treat as capability gap / refuse for historical questions.
- Keyword arg must be `keywords: [str]` (list).

## Evidence paths

- `docs/news_mcp_discovery_v1/*.json`
- `scripts/news_mcp_discovery_smoke.py`
