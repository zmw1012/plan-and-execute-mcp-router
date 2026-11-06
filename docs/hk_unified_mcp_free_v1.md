# Unified HK MCP-Free (Closed-Book) Baseline v1

Branch: `feature/hk-unified-mcp-free-v1`  
Worktree: `../hk-equity-mcp-agent-hk-mcp-free`

## Condition

| Field | Value |
|---|---|
| `condition` | `mcp_free` |
| `execution_backend` | `none` |
| Model | `deepseek-v4.1-flash` |
| Temperature | `0` |
| Max tokens | `1024` |
| JSON mode | `true` |
| Dataset | Same exploratory HK-155 as Operation-Visible |

## MCP-Free invariants

- No MCP tools / function calling / operation catalog
- No find / describe / invoke
- No Embedding / RAG / Web
- No local financial mock data / HKEX / AKTools
- No external knowledge base
- Only DeepSeek model-parameter knowledge + user question
- Never fall back to tools on failure
- Gold / reference / domain / answerability never enter the model context

## Frozen system prompt

Recorded in `run_metadata.json` (`system_prompt` + `system_prompt_hash`).

Semantic content (domain-agnostic, pre-frozen):

> You have no external tools, web access, retrieval system, database, filings,
> market feed, or news source. Answer only from model-internal knowledge. For
> time-sensitive or exact company data that you cannot verify, return unavailable
> instead of fabricating. Follow the required JSON answer contract.

Plus the shared `FINAL_ANSWER_CONTRACT`.

## Formal output directory

```text
outputs/hk_unified_exploratory155_mcp_free_v1
```

Artifacts aligned with Operation-Visible:

- `predictions.jsonl`
- `per_task_metrics.csv`
- `manual_grading.csv`
- `cost_summary.json`
- `error_summary.json`
- `run_metadata.json`
- `grouped_aggregates.json`

Tool-related fields are explicitly `N/A` / zero (not vacuous tool-success).

## Offline dry-run / mock

```bash
cd ../hk-equity-mcp-agent-hk-mcp-free
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
OUT=outputs/hk_unified_exploratory155_mcp_free_v1

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout --dry-run --allow-dirty \
  --output-dir "$OUT"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout --mock --allow-dirty \
  --max-tasks 3 --output-dir outputs/hk_unified_mcp_free_mock_smoke_v1
```

## Formal 155 run (do not execute in this offline round)

```bash
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
ENV_GEN="../hk-equity-mcp-agent/.env.aliyun"
OUT=outputs/hk_unified_exploratory155_mcp_free_v1

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" \
  --max-tasks 20 \
  --output-dir "$OUT"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --resume \
  --output-dir "$OUT"
```

No `--embedding-env-file` (Embedding is disabled).

## Scoring

Reuses deterministic evaluator / grader **v1.2** without modifying gold.

Reported separately:

- answer accuracy
- refusal accuracy
- unavailable rate
- hallucinated exact-value rate
- tokens / latency / LLM calls

Grouped by: overall, domain, D1–D5 (`project_difficulty`), answerability class,
company, development / previously-reserved origin.

## ETOM Empty-Route Control (offline; not an MCP-Free agent)

```bash
PYTHONPATH=src "$PYTHON" scripts/evaluate_etom_empty_route_control.py \
  --lock "../hk-equity-mcp-agent-etom/benchmarks/etom/formal_100_v1.lock.json" \
  --output-dir outputs/etom_empty_route_control_v1
```

Predicts an empty operation set for every locked task. Expected:
L1–L4 exact match = 0; L5 empty-route exact match = 1.
