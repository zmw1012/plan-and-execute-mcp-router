# News Real-MCP Catalog Freeze v1

**Catalog:** `configs/operations/news_real_mcp_v1.yaml`  
**Status:** `frozen_v1`  
**Source candidate:** `configs/operations/news_real_mcp_candidate_v1.yaml` (retained as provenance trail)  
**Source commit:** `1d5fd858426ba0e99a0a969208bf00222a9ab23c`  
**Freeze date:** 2026-11-04 (Asia/Hong_Kong)  
**Cutoff:** 2026-08-12 Asia/Hong_Kong  

## Package / server versions (live-audited)

| Server | Version / note |
|---|---|
| agentladle-mcp-hkexnews | 0.1.0 |
| aktools | 0.1.4 |
| akshare | 1.20.22 |
| ah-disclosure-kit | vendor stdio `ah-disclosure-mcp` |

Checksum: see `catalog_checksum_sha256` in the YAML (validator recomputes).

## Exposure policy

| Class | exposed_to_agent | runtime_registry | formal_gold_eligible |
|---|---|---|---|
| primary | true | true | true |
| supporting (selected) | true | true | true/false by op |
| distractor (incl. `stock_news_em`) | true | true | **false** |
| blocked / rejected | **false** | **false** | false |

`stock_news_em`: runtime distractor only; never formal/held-out gold.
