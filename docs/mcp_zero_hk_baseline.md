# HK-155 MCP-Zero Baseline

Condition: `mcp_zero`  
Config: `configs/mcp_zero_hk_v1.yaml`  
Runner: `scripts/run_hk_unified_mcp_zero_experiment.py`

## Architecture

```
User query
  → LLM Active Tool Request (<tool_assistant>)
  → Server-level retrieval (top-k servers)
  → Tool-level retrieval inside selected servers only (top-k tools)
  → Inject retrieved schemas into context
  → Agent selects operation + constructs arguments
  → UnifiedRealMcpGateway.invoke (Real MCP)
  → If capability gap remains → Iterative Capability Extension
  → Final answer (shared OV JSON contract)
```

## Retrieval knobs

| Knob | Value | Source |
|---|---|---|
| `server_top_k` | 3 | ETOM MCP0 pin default |
| `tool_top_k` | 5 | ETOM MCP0 pin default |
| `score_mode` | `official_combined` | Paper / official matcher |
| `similarity_threshold` | 0.0 | HK default (no silent drop) |
| `max_extension_rounds` | 3 | HK adapted termination bound |
| Tie-break | `(-score, server_id, tool_name, operation_id)` | Deterministic |

## Extension termination policy

Stop further retrieval when any of:

1. `max_extension_rounds` reached  
2. Duplicate Active Tool Request key  
3. No new tools beyond already-seen set  
4. Empty / unparsable request  
5. No servers matched  
6. Model proceeds to `invoke_api` / `final_answer`  

## Models

| Slot | Value |
|---|---|
| Generation | `deepseek-v4.1-flash` |
| Embedding | `text-embedding-3-small` |
| Temperature | `0.0` |
| Max tokens | `1024` |
| Max LLM turns | `8` |
| Max invocations | `6` |

## Offline commands

```bash
# Dry-run (no APIs)
python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split development --dry-run --allow-dirty \
  --output-dir outputs/mcp_zero_hk_dryrun

# Mock smoke (hash embeddings + mock transport; no paid APIs)
python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split development --smoke --mock --allow-dirty \
  --output-dir outputs/mcp_zero_hk_mock_smoke
```

## Future real smoke (DO NOT run in this round)

```bash
# Requires paid env profiles; refuse dirty tree unless --allow-dirty
python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split development --smoke \
  --env-file .env.aliyun \
  --embedding-env-file .env.embedding-relay \
  --output-dir outputs/mcp_zero_hk_smoke_real
```

## Outputs (OV-compatible)

- `predictions.jsonl`
- `per_task_metrics.csv`
- `manual_grading.csv`
- `cost_summary.json`
- `run_metadata.json`
- `retrieval_analysis.json`
- `error_summary.json`
- `grouped_aggregates.json`

Extra MCP-Zero fields on predictions / retrieval analysis:

- `active_tool_request`
- `retrieved_servers` / ranks / scores
- `retrieved_tools` / ranks / scores
- `selected_operations` / `invoked_operations`
- `extension_rounds` / `termination_reason`
- `llm_calls` / embedding calls+tokens
- `retrieval_latency_ms` / `generation_latency_ms`
