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

Mock end-to-end path works for a **gold-path execution smoke test**: the runner injects `gold_tools` / `gold_arguments`, so a 1.0 `strict_task_success_rate` does **not** mean the agent can autonomously select tools. Real LLM integration and true tool-selection evaluation are intentionally deferred. See also `docs/financial_domain.md`.
