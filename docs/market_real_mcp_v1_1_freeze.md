# Market Real-MCP v1.1 Freeze

- **Active catalog:** `configs/operations/market_real_mcp_v1_1.yaml`
- **Supersedes:** `configs/operations/market_real_mcp_v1.yaml` (`superseded_before_any_agent_run`)
- **Reason:** package + blocked FTShare operation-ID provenance correction **before any Agent/LLM run**
- **Cutoff / formulas / previous-valid-day / adjustment rules:** unchanged from v1
- **Numerical gold:** unchanged; v1 provenance hashes reused (raw invocations unchanged)

## Provenance corrections

| Field | v1 (incorrect) | v1.1 (corrected) |
| --- | --- | --- |
| Python package | `mcp-server-aktools` | `aktools` `0.1.4` (`pip show` verified) |
| Blocked FTShare IDs | `ftshare.hk_hist`, `ftshare.stock_hk_daily` | `ftshare.ft_v2_hk_hk_candlesticks`, `ftshare.ft_v1_hk_hsi_daily_weight` (candidate-audit exact) |
| Minute hist | deferred | still deferred (`aktools.stock_hk_hist_min_em`) |

## Active role distribution (unchanged)

primary 2 / supporting 6 / distractor 5 → **13 active**

## Taxonomy policy

Project difficulty levels are balanced for analysis, whereas ETOM levels are assigned only when the original orchestration semantics are satisfied. Market-only tasks are not forced into ETOM L4 because the frozen Market catalog contains one executable MCP server. Cross-server ETOM L4 is evaluated later in the cross-domain benchmark.

## Benchmarks

| File | Status |
| --- | --- |
| `market_development_15_v1.jsonl` / `market_test_20_v1.jsonl` | `superseded_before_any_agent_run` |
| `market_development_15_v1_1.jsonl` / `market_test_20_v1_1.jsonl` | **active** |

## Gateway

`MarketRealMcpGateway` defaults to `market_real_mcp_v1_1.yaml`. Fail-closed; no mock FinancialServer fallback.
