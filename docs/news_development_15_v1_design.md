# News Development 15 v1 Design

**File:** `benchmarks/questions/news_development_15_v1.jsonl`  
**Catalog:** `news_real_mcp_v1`  
**Domain:** News / Corporate Disclosure  
**Cutoff:** 2026-08-12 Asia/Hong_Kong  

## Company mix

Tencent 5 · HSBC 5 · Alibaba 5

## ETOM / task buckets

| Bucket | Count |
|---|---|
| Exact L1 | 5 |
| Exact L2 | 0 |
| Exact L3 | 4 |
| Exact L4 | 0 |
| Exact L5 capability_gap | 2 |
| Adapted robustness | 2 |
| Project reasoning | 2 |

## Notes

- Formal gold from live HKEXnews / ah annual-report metadata only.
- Exact L3 uses live-verified field edges (`keywords:list`, `page_number→start_page`).
- Results announcement used only for title/date event identification (no EPS/revenue).
- `stock_news_em` appears only in Exact L5 gap probes as unavailable historical media.
