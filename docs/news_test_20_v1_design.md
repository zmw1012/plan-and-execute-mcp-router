# News Held-out 20 v1 Design

**File:** `benchmarks/heldout/news_test_20_v1.jsonl`  
**Catalog:** `news_real_mcp_v1`  
**Role:** `final_evaluation` · `locked=true` · `heldout_agent_run_status=never_agent_run`  
**Cutoff:** 2026-08-12 Asia/Hong_Kong  

## Company mix

Tencent 7 · HSBC 7 · Alibaba 6

## ETOM / task buckets

| Bucket | Count |
|---|---|
| Exact L1 | 6 |
| Exact L2 | 0 |
| Exact L3 | 5 |
| Exact L4 | 0 |
| Exact L5 capability_gap | 3 |
| Adapted robustness | 3 |
| Project reasoning | 3 |

## Notes

- Do not expose held-out gold to Agent prompts.
- No Agent/LLM run has been performed on `news_ht_*`.
- Annual-report questions are existence/title/date/URL only.
