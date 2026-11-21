# HK-155 MCP-Free Final Audit v1

- Audit version: `mcp_free_final_audit_v1`
- Generated (UTC): `2026-11-06T06:18:12.497117+00:00`
- Run dir: `/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-hk-mcp-free/outputs/hk_unified_exploratory155_mcp_free_live_v1`
- Condition: `mcp_free` / execution_backend=`none`
- Provider: `openai_compatible`
- Provider host: `api.zhizengzeng.com`
- Model: `deepseek-v4.1-flash`
- Catalog fingerprint: `null` (none — MCP-Free has no operation catalog)

## Accuracy

- Correct (final): `20/155`
- Strict / overall accuracy (denominator 155): `0.12903225806451613` / `0.12903225806451613`
- Auto-gradable accuracy (denominator 154): `0.12337662337662338`
- Auto correct / auto gradable: `19/154`
- Manual required: `1` (confirmed `1`)
- Conservative lower bound (auto-only correct/155): `0.12258064516129032`
- Financial final: `{'correct': 8, 'n': 60, 'final_accuracy': 0.13333333333333333}`
- D2 final: `{'correct': 2, 'n': 24, 'final_accuracy': 0.08333333333333333}`
- Development-origin final: `{'correct': 10, 'n': 70, 'final_accuracy': 0.14285714285714285}`

## Status / failure rates

- Unavailable rate: `0.9225806451612903`
- Answered rate: `0.05806451612903226`
- Clarification rate: `0.01935483870967742`
- Hallucinated exact-value rate: `0.08387096774193549`
- Parse failure rate: `0.0064516129032258064`

## Parse-error task

- task_id: `fin_ht_030`
- domain: `financial`
- raw response saved: `True`
- error stage: `parse`
- offline recoverable: `True`
- recovery method: `truncated_completed_fields`
- needs manual scoring: `False`
- formal treatment: `incorrect_evaluation_failure_formal;offline_parser_delta_reported_only`

## Cost / latency

- Input tokens: `36682`
- Output tokens: `42451`
- Total tokens: `79133`
- LLM calls: `157`
- Mean / median / p95 latency_ms: `3865.1640258064517` / `3415.284` / `5937.596499999998`
- Estimated USD cost: `None` (No frozen provider price table for openai_compatible/api.zhizengzeng.com; USD cost not estimated.)

## Tool / embedding / MCP

- tool metrics: **N/A** (not vacuous 1.0)
- embedding calls: **0**
- resolver calls: **0**
- real MCP executions: **0**
- tool calls: **0**

## Notes

- No LLM / Embedding / MCP called during this audit.
- Raw predictions.jsonl and run artifacts were not modified.
- Empty tool sets are not scored as tool accuracy 1.0.
