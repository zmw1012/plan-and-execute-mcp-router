# Market Development 15 v1.1 Design

- **File:** `benchmarks/questions/market_development_15_v1_1.jsonl`
- **Supersedes:** `market_development_15_v1.jsonl` (`superseded_before_any_agent_run`)
- **Membership / numerical gold:** identical to v1
- **Catalog:** `market_real_mcp_v1_1`

## Project difficulty distribution (not ETOM distribution)

| Level | Count |
| --- | --- |
| D1 | 4 |
| D2 | 3 |
| D3 | 3 |
| D4 | 3 |
| D5 | 2 |

## Exact ETOM-compatible assignments

Assigned only when orchestration semantics hold:

- **L1 exact:** foundational single-tool retrieval (close/volume/OHLC/HSI)
- **L2 adapted:** previous-valid-day / field-path / adjustment-mode disambiguation
- **L3/L4:** none forced; local calc and same-server dual retrieval are `etom_level=null`
- **L5 exact:** minute-bar capability gap (`mkt_dev_015`)
- **temporal_policy (not exact L5):** latest/current (`mkt_dev_014`) because spot distractors exist

Project difficulty levels are balanced for analysis, whereas ETOM levels are assigned only when the original orchestration semantics are satisfied. Market-only tasks are not forced into ETOM L4 because the frozen Market catalog contains one executable MCP server. Cross-server ETOM L4 is evaluated later in the cross-domain benchmark.
