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

Generation uses an OpenAI-compatible relay, not Alibaba token-plan. Runtime env pins (`generation_provider`, host, model) enter run/resume identity. They are **not** part of the routing method-card hash (`method_identity_sha256` stays `77cc3b07…`).

Required generation profile fields in gitignored main-repo `.env.relay-v41`:

- `LLM_PROVIDER=openai_compatible`
- `LLM_MODEL=deepseek-v4.1-flash`
- `LLM_BASE_URL` host `api.zhizengzeng.com`

| Role | File | Host | Model / provider pin |
|---|---|---|---|
| Generation | main-repo `.env.relay-v41` | `api.zhizengzeng.com` | `deepseek-v4.1-flash` / `openai_compatible` |
| Embedding | main-repo `.env.embedding-relay` | `api.zhizengzeng.com` | `text-embedding-3-small` |

Always run from this worktree with:

```bash
export PYTHONPATH=src
PY="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
GEN_ENV="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.relay-v41"
EMB_ENV="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay"
EMB_CACHE=".cache/mcp_zero_hk_v1"
```

Catalog embedding indexes are **shared** under `$EMB_CACHE` (worktree `.cache/mcp_zero_hk_v1`), not under each `output_dir`. Check status offline with `--index-status`. Formal agent runs should pass `--require-complete-index` so a missing index fails closed instead of rebuilding mid-run.

`--allow-hkex-download` is required for real smoke and formal 155. `news_dev_006` gold includes `hkexnews.download_hkexnews_announcement`. `xd_dev_007` itself only needs list + `stock_hk_hist`, but the four-task smoke set includes `news_dev_006`. Formal HK-155 has 24 download-gold tasks. Without the flag, `UnifiedRealMcpGateway` constructs `HkexnewsAdapter(allow_download=False)` and download fails closed. Separately, the adapter still enforces a session budget of one successful download; that is a pre-existing gateway safety limit, not part of the frozen MCP-Zero method card, and it must be resolved before treating a continuous 155-run as complete for download-heavy news/financial tasks.

`--max-tasks` is the cumulative locked-order prefix, not the number of new tasks in the current process. It is recorded on every run and is intentionally outside resume identity. Resume still checks the other method identity fields, and the new `question_ids` list must extend the previous list as a prefix. Completed task ids are skipped and are not sent to the model again.

The four smoke ids are unchanged HK-155 development-source items:

| ID | Role |
|---|---|
| `fin_cf_001` | Financial, single operation |
| `mkt_dev_014` | Market, refusal / infeasible (`should_refuse_or_clarify`, zero gold operations) |
| `news_dev_006` | News, multi-step dependency (includes download; needs `--allow-hkex-download`) |
| `xd_dev_007` | Cross-domain, list→`stock_hk_hist` output-input dependency |

A real `--split exploratory_all` run writes `retirement_effective=true` into that output directory. These four ids do not include previously reserved questions. The formal 155 directory is separate. Frozen JSONL is not modified.

```bash
# Real smoke v3 (after ATR/json_mode protocol correction). Do NOT overwrite v1/v2.
# v1: invalidated_engineering_smoke_due_to_runner_protocol_defects
# v2: invalidated_engineering_smoke_due_to_atr_json_mode_protocol_conflict
# Requires .env.relay-v41 with LLM_PROVIDER=openai_compatible.
# json_mode must be false so raw <tool_assistant> ATR turns are legal.
PYTHONPATH=src "$PY" scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --ids fin_cf_001,mkt_dev_014,news_dev_006,xd_dev_007 \
  --allow-hkex-download \
  --env-file "$GEN_ENV" \
  --embedding-env-file "$EMB_ENV" \
  --output-dir outputs/hk_unified_mcp_zero_real_smoke4_v3

# Continuous full HK-155. First process has no --resume.
PYTHONPATH=src "$PY" scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --max-tasks 155 \
  --allow-hkex-download \
  --env-file "$GEN_ENV" \
  --embedding-env-file "$EMB_ENV" \
  --output-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1

# After an interrupt: same directory, same identity, resume cumulative 155.
PYTHONPATH=src "$PY" scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all \
  --confirm-retire-heldout \
  --max-tasks 155 \
  --resume \
  --allow-hkex-download \
  --env-file "$GEN_ENV" \
  --embedding-env-file "$EMB_ENV" \
  --output-dir outputs/hk_unified_exploratory155_mcp_zero_live_v1

# Optional staged prefixes (same semantics; max-tasks is cumulative):
# --max-tasks 20
# --max-tasks 50 --resume
# --max-tasks 100 --resume
# --max-tasks 155 --resume

# Offline postprocess. Does not call LLM, embedding, or MCP.
PYTHONPATH=src "$PY" scripts/postprocess_hk_unified_run.py \
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
- `selected_operations` / `invalid_selections` / `invoked_operations` / `successful_invoked_operations`
- `extension_rounds` / `termination_reason`
- `llm_calls` / embedding calls+tokens (per-task query-time deltas)
- `retrieval_latency_ms` / `generation_latency_ms`
- `error_type` / `error_message` / `failure_stage` / `protocol_repair_count`
- index build calls, tokens, latency, and `cache_reused` on `run_metadata.json` (`embedding_index`) and `cost_summary.json`

### Tool / retrieval field semantics (comparable to Operation-Visible)

| Field | Meaning |
|---|---|
| `retrieved_tools` | All HierarchicalRouter Top-K candidates. Use for **retrieval Recall@K** only. |
| `selected_operations` | `operation_id`s the model explicitly chose in `invoke_api` tool_calls. Formal **tool-selection** precision/recall/F1 should use this set (or the OV-compatible `predicted_operations` derived from invoke_api in the trace). Never auto-copy retrieved candidates into selected. |
| `invoked_operations` | Selected ops that passed basic validation and were submitted to the Gateway. |
| `successful_invoked_operations` | Invoked ops where Gateway returned success. |
| `invalid_selections` | `invoke_api` requests for ops not in the retrieved set (`invoked=0`). |

Operation-Visible `operation_precision` / `operation_recall` / `operation_f1` continue to use invoke_api order from the trace (`extract_predicted_operations`), which aligns with `invoked_operations` / successful selection — not with raw retrieval candidates.
