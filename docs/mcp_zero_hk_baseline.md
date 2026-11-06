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

## Frozen method card

Frozen before any HK-155 real result. sha256 `77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f`.

The paper / official matcher default is `server_top_k=5`, `tool_top_k=3`. This baseline uses the ETOM pinned adaptation `3 / 5`. That 3/5 setting is not the paper's original exact configuration. Do not retune top-k from HK-155 results. A later 5/3 run is an independent ablation and must not replace this baseline.

| Knob | Frozen value |
|---|---|
| `server_top_k` | 3 (ETOM adaptation; paper default is 5) |
| `tool_top_k` | 5 (ETOM adaptation; paper default is 3) |
| `score_mode` | `official_combined` |
| `similarity_threshold` | 0.0 |
| `max_extension_rounds` | 3 |
| Tie-break | `(-score, server_id, tool_name, operation_id)` |
| Generation | `deepseek-v4.1-flash` |
| Embedding | `text-embedding-3-small` |
| Temperature / max tokens / max LLM turns | `0.0` / `1024` / `8` |
| Evaluator | `hk_unified_evaluator_v1.2` |

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

## Real smoke and HK-155 commands

Do not run these until a generation profile is confirmed. No file in the main repo is both the long-term relay and `LLM_MODEL=deepseek-v4.1-flash`.

Candidates, names and non-secret fields only:

| File | Host | Model |
|---|---|---|
| `.env` | `api.zhizengzeng.com` | `deepseek-flash` |
| `.env.relay` | `api.zhizengzeng.com` | `deepseek-flash` |
| `.env.aliyun` | `token-plan.cn-beijing.maas.aliyuncs.com` | `deepseek-v4.1-flash` |

`.env` and `.env.relay` also omit `LLM_PROVIDER`. The runner requires `LLM_PROVIDER=aliyun` because that is the frozen config provider pin, and it requires `LLM_MODEL=deepseek-v4.1-flash`. `.env.aliyun` has the right model and provider, and the wrong host for a post-token-plan run. Embedding stays on the main repo file `.env.embedding-relay` (`api.zhizengzeng.com`, `text-embedding-3-small`). Pass that path absolutely; this worktree does not contain it.

`--max-tasks` is the cumulative locked-order prefix, not the number of new tasks in the current process. It is recorded on every run and is intentionally outside resume identity, so 20 → 50 → 100 → 155 can resume. Resume still checks the other method identity fields, and the new `question_ids` list must extend the previous list as a prefix. Completed task ids are skipped and are not sent to the model again.

The four smoke ids are unchanged HK-155 development-source items:

| ID | Role |
|---|---|
| `fin_cf_001` | Financial, single operation |
| `mkt_dev_014` | Market, refusal / infeasible (`should_refuse_or_clarify`, zero gold operations) |
| `news_dev_006` | News, multi-step dependency (5 operations, dependency edges) |
| `xd_dev_007` | Cross-domain, output-input dependency across servers |

A real `--split exploratory_all` run writes `retirement_effective=true` into that output directory. These four ids do not include previously reserved questions. The formal 155 directory is separate. Frozen JSONL is not modified.

```bash
# NOT EXECUTABLE until GENERATION_ENV is a confirmed profile with
# LLM_MODEL=deepseek-v4.1-flash and LLM_PROVIDER=aliyun.
# Real smoke. Do not point this at the formal 155 directory.
python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --ids fin_cf_001,mkt_dev_014,news_dev_006,xd_dev_007 \
  --env-file <UNRESOLVED_GENERATION_ENV> \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay" \
  --output-dir outputs/hk_unified_mcp_zero_real_smoke4_v1

# Formal HK-155. First wave has no --resume.
python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --max-tasks 20 \
  --env-file <UNRESOLVED_GENERATION_ENV> \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay" \
  --output-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1

# Cumulative 50, 100, then 155. Same directory. --resume skips completed ids.
python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --max-tasks 50 \
  --resume \
  --env-file <UNRESOLVED_GENERATION_ENV> \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay" \
  --output-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1

python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --max-tasks 100 \
  --resume \
  --env-file <UNRESOLVED_GENERATION_ENV> \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay" \
  --output-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1

python scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --max-tasks 155 \
  --resume \
  --env-file <UNRESOLVED_GENERATION_ENV> \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay" \
  --output-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1

# Offline postprocess. Does not call LLM, embedding, or MCP.
python scripts/postprocess_hk_unified_run.py \
  --run-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1
```

Embedding indexes are written under `<output-dir>/.cache/mcp_zero_hk_v1/`. The cache key includes the embedding model, unified catalog fingerprint, server/tool metadata hash, and scoring config. A missing index is built explicitly. A partial file for that same identity is resumed. A complete index is required before task execution. The writer refuses `operation_embeddings` and ETOM cache directories, and it refuses to overwrite a file from another namespace. Mock hash vectors use model name `mock-hash-embedding-v1`, so they do not occupy the `text-embedding-3-small` index.

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
- index build calls, tokens, latency, and `cache_reused` on `run_metadata.json` (`embedding_index`) and `cost_summary.json`
