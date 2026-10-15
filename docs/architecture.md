# Architecture

## Goal

Compare three agent settings for Hong Kong equity question answering that use multiple MCP-style tool servers:

1. **flat** — all tools available in one list
2. **routed** — domain router then tool selection
3. **routed_structured** — routed plus structured-output constraints

Current repository ships **mock servers only**. No paid APIs or real LLM calls.

## Components

```
User / Eval Runner
        │
        ▼
   AgentRouter (modes: flat | routed | routed_structured)
        │
        ├── FinancialServer  (get_financial_metric, get_filing_summary)
        ├── MarketServer     (get_price_series, calculate_return)
        └── NewsServer       (search_announcements, get_news_summary)
```

## Unified tool response

Every tool returns:

| Field | Meaning |
|-------|---------|
| `success` | Whether the call succeeded |
| `source` | Server / data source id |
| `ticker` | Normalized ticker (e.g. `0700.HK`) |
| `as_of_date` | Snapshot date |
| `data` | Payload |
| `evidence` | Supporting snippets |
| `error` | Error message when `success=false` |

## Data

Mock tables live in `src/hk_equity_mcp/common/mock_data.py` and are mirrored under `data/mock/` for inspection. Universe: `0700.HK`, `0005.HK`, `9988.HK`.

## Evaluation

`EvaluationRunner` reads JSONL questions, executes gold tools against mock servers (placeholder agent), writes per-example CSV rows, and computes reserved metrics.
