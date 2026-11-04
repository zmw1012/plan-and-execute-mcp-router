# News / Corporate Disclosure MCP Capability Matrix (Discovery v1 — corrected)

Temporal policy: **max gold date = 2026-08-12**, timezone **Asia/Hong_Kong**.  
Legend: ✅ live-verified usable for **formal gold** · ⚠️ runtime-only / constrained · ❌ unsupported · D distractor

## HKEXnews dependency smoke (corrected)

Full live chain for **Tencent / HSBC / Alibaba** (fixed announcements ≤ cutoff; corporate-event keywords; **not** revenue/EPS extraction):

`list → download → parse → keyword_search(keywords=[...]) → get_announcement_pages(start_page=page_number)`

| Issuer | Announcement (≤2026-08-12) | keyword_search | pages |
|---|---|---|---|
| Tencent 00700 | Monthly Return … 31 July 2026 (`2026-08-06`) | ✅ `keywords=["Monthly","Return","Issuer"]` | ✅ `start_page=4` |
| HSBC 00005 | Transaction in own shares (`2026-08-12`) | ✅ `keywords=["Transaction","shares","HSBC"]` | ✅ `start_page=2` |
| Alibaba 09988 | DATE OF BOARD MEETING (`2026-08-07`) | ✅ `keywords=["Board","Meeting","Director"]` | ✅ `start_page=1` |

Evidence: `docs/news_mcp_discovery_v1/hkexnews_dependency_chain_{tencent,hsbc,alibaba}.json`, `hkexnews_dependency_chain_summary.json`, `observed_dependency_edges.json`.  
Prior failed `keyword` (singular) probe is **not** treated as success; re-run used schema-correct `keywords:list`.

### Verified field edges

- `list.local_key|news_id|release_date|title` → `download.local_key|news_id|release_date|title_keyword` (short title_keyword required)
- `download.local_key` → `parse.local_key`
- `parse.local_key` → `keyword_search.local_key` / `get_announcement_pages.local_key`
- `keyword_search.page_number` → `get_announcement_pages.start_page`

## Capability × operation

| # | Capability | Primary formal-gold path | Supporting | Distractor / gap |
|---|---|---|---|---|
| 1 | Search by stock code | ✅ hkexnews list | — | D A-share notice; ⚠️ media runtime-only |
| 2 | Search by company name | ⚠️ weak for official | — | D media name keyword (not formal gold) |
| 3 | Date-range filter | ✅ hkexnews start/end | — | D ah `search_filings` (date-unreliable) |
| 4 | Publication timestamp | ✅ `release_datetime` | — | ⚠️ media `发布时间` floating |
| 5 | Source name + URL | ✅ `file_link` | ah annual `pdf_url` | ⚠️ media URL runtime-only |
| 6 | Headline | ✅ `title` | ah annual title | ⚠️ media headline runtime-only |
| 7 | Snippet / page text | ✅ keyword/pages chain | — | ⚠️ media snippet runtime-only |
| 8 | HKEX official list | ✅ list | — | D ah filings |
| 9 | Download/parse/keyword/pages | ✅ live 3-issuer chain | — | — |
| 10 | Dedup | ⚠️ offline on `news_id`/`local_key` | — | ❌ no MCP dedup tool |
| 11 | ZH + EN | ✅ EN list live | — | — |
| 12 | Chronological ordering | ✅ list DateTime order | offline sort | — |
| 13 | Latest-before-cutoff | ✅ max datetime on official list | — | ❌ media (n_before_cutoff=0) |
| 14 | Fail closed | ✅ invalid code; temporal policy | — | media historical → capability_gap |

## `stock_news_em` role (corrected)

| Attribute | Value |
|---|---|
| Live feed Tencent/HSBC/Alibaba | yes (floating ~10) |
| `n_before_cutoff` | **0** |
| Date-range params | **none** |
| Catalog role | `supporting_runtime_only` / `distractor_for_formal_gold` |
| `formal_gold_eligible` | **false** |
| Allowed | tool-selection distractor; capability-gap; fail-closed; informal demo |
| Forbidden | formal gold; held-out evidence; official-vs-media factual gold; adapted L4 formal; latest-before-cutoff gold |

## Exact ETOM planning constraints

- `equal_function_sets: []` ⇒ **Exact L2 = 0**
- Official disclosure ≠ financial media ≠ Annual Report ≠ results announcement (as equal tools)
- Exact L3 only via HKEXnews field DAG above
- Exact L4 = 0 until a real cross-server field transfer exists
- Exact L5 only for true `capability_gap` (e.g. historical media under frozen catalog)

## Counts (candidate catalog after correction)

| Class | Count |
|---|---|
| primary | 5 |
| supporting | 5 |
| distractor | 7 (includes `stock_news_em`) |
| blocked | 1 |
| rejected | 2 |
| **logical ops** | **20** |
| equal_function_sets | **0** |

## Remaining gaps

1. No free historical HK **media** archive with date filter.  
2. ah `search_filings` date window untrustworthy.  
3. No cross-source dedup MCP.  
4. FTShare still blocked.  
5. Company-name → official disclosure without code remains weak.
