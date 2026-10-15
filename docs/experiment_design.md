# Experiment Design (Draft)

## Research question

Does domain routing (and structured outputs) improve tool use quality and answer faithfulness for multi-domain HK equity QA, compared with a flat tool list?

## Conditions

| Condition | Description |
|-----------|-------------|
| `flat` | Model sees all tools |
| `routed` | Router selects domain, then tools in that domain |
| `routed_structured` | Same as routed, with structured argument / answer schema |

## Domains / tools

- **Financial**: `get_financial_metric`, `get_filing_summary`
- **Market**: `get_price_series`, `calculate_return`
- **News**: `search_announcements`, `get_news_summary`

## Metrics (interfaces reserved)

- Strict Task Success Rate
- Tool Selection Accuracy / Tool F1
- Argument Accuracy
- Execution Success Rate
- Answer Correctness
- Evidence Correctness
- Wrong or Unnecessary Tool Rate
- Tool Calls / Tokens / Latency

## Current status

Mock end-to-end path works with gold tools for smoke evaluation. Real LLM integration is intentionally deferred.
