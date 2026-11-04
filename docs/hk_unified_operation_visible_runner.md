# Unified HK Operation-Visible Real-MCP Runner

Branch: `feature/unified-hk-operation-visible-runner-v1`  
Worktree: `../hk-equity-mcp-agent-hk-unified`

## Interpreter / MCP binaries

Do **not** use system `python`. Use the main repo discovery venv:

```bash
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
```

MCP executables are resolved fail-closed in order:

1. `AKTOOLS_MCP_COMMAND` / `HKEXNEWS_MCP_COMMAND` / `AH_DISCLOSURE_MCP_COMMAND`
2. current worktree `.venv-*-discovery/bin/…`
3. sibling `../hk-equity-mcp-agent/.venv-*-discovery/bin/…`
4. `PATH`

Do not copy venvs into this worktree.

## Dry-run (no paid APIs)

```bash
cd ../hk-equity-mcp-agent-hk-unified
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --dry-run --allow-dirty \
  --output-dir outputs/hk_unified_dry_run_v1
```

## Real Smoke 8 (separate output-dir)

```bash
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --smoke \
  --output-dir outputs/hk_unified_real_smoke8_v1
```

Smoke and full Development **must** use different `--output-dir` values.

## Development 70 — correct batching

Do **not** seed the Dev70 directory with `--smoke`.

```bash
# Batch 1: first 20 development tasks
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split development --max-tasks 20 \
  --output-dir outputs/hk_unified_dev70_v1

# Batch 2: first 40 (resume skips completed)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split development --max-tasks 40 --resume \
  --output-dir outputs/hk_unified_dev70_v1

# Batch 3: remaining development via resume
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split development --resume \
  --output-dir outputs/hk_unified_dev70_v1
```

## Offline postprocess (no Agent / LLM / Embedding / MCP)

Recompute metrics/error/retrieval summaries from an existing `predictions.jsonl`
without modifying predictions or run_metadata identity:

```bash
PYTHONPATH=src "$PYTHON" scripts/postprocess_hk_unified_run.py \
  --run-dir outputs/hk_unified_real_smoke8_v1
```

Writes/refreshes: `per_task_metrics.csv`, `manual_grading.csv`, `cost_summary.json`,
`retrieval_analysis.json`, `error_summary.json`, `evaluation_metadata.json`.

## Metric notes

- `error_counts`: non-empty `error` / `error_type` only. **`status=full` is not an error.**
- `result_status_counts`: independent status histogram (`full` / `failed` / …).
- `tool_execution_success` ≡ `all_tool_invocations_successful` (all invokes succeeded; vacuous True if zero invokes).
- `recovered_after_tool_error`: early invoke failed, later invoke succeeded, complete final answer — execution recovery only, not answer correctness.

## Held-out (legacy)

Locked unless both `--split heldout` and `--confirm-heldout-final` are provided.
This project now prefers exploratory_all for method development.

## Exploratory 155 (method development + difficulty audit)

Pointer-only manifest: `benchmarks/manifests/hk_unified_exploratory_all_v1.json`  
Locked run-order: `benchmarks/manifests/hk_unified_exploratory_all_run_order_v1.json`

Requires both `--split exploratory_all` and `--confirm-retire-heldout`.
Dry-run/mock show retirement policy but do **not** count as first real exposure.
Only a real (non-mock, non-dry-run) run sets
`heldout_status=retired_into_exploratory_development`.

Use a dedicated output directory (never smoke / never dev70):

```bash
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
ENV_GEN="../hk-equity-mcp-agent/.env.aliyun"
ENV_EMB="../hk-equity-mcp-agent/.env.embedding-relay"
OUT=outputs/hk_unified_exploratory155_operation_visible_v1

# dry-run
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split exploratory_all --confirm-retire-heldout --dry-run --allow-dirty \
  --output-dir "$OUT"

# real batches (do not execute in offline hardening rounds)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --embedding-env-file "$ENV_EMB" \
  --allow-hkex-download --max-tasks 20 \
  --output-dir "$OUT"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --embedding-env-file "$ENV_EMB" \
  --allow-hkex-download --max-tasks 50 --resume \
  --output-dir "$OUT"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --embedding-env-file "$ENV_EMB" \
  --allow-hkex-download --max-tasks 100 --resume \
  --output-dir "$OUT"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --embedding-env-file "$ENV_EMB" \
  --allow-hkex-download --resume \
  --output-dir "$OUT"

# offline postprocess
PYTHONPATH=src "$PYTHON" scripts/postprocess_hk_unified_run.py --run-dir "$OUT"
```
