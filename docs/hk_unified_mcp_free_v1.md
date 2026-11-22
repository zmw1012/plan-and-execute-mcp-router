# Unified HK MCP-Free (Closed-Book) Baseline v1

Branch: `feature/hk-unified-mcp-free-v1`  
Worktree: `../hk-equity-mcp-agent-hk-mcp-free`

## Condition

| Field | Value |
|---|---|
| `condition` | `mcp_free` |
| `execution_backend` | `none` |
| Generation provider | `openai_compatible` (OpenAI-compatible relay; **not** Aliyun Token Plan) |
| Provider host | `api.zhizengzeng.com` |
| Model | `deepseek-v4.1-flash` |
| Temperature | `0` |
| Max tokens | `1024` |
| JSON mode | `true` |
| Dataset | Same exploratory HK-155 as Operation-Visible |
| Formal env | `../hk-equity-mcp-agent/.env.relay-v41` |

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

## Output directories (must stay separate)

| Role | Directory |
|---|---|
| Dry-run only | `outputs/hk_unified_exploratory155_mcp_free_dry_run_v1` |
| Real smoke (3 ids) | `outputs/hk_unified_mcp_free_real_smoke3_v1` |
| Formal live 155 | `outputs/hk_unified_exploratory155_mcp_free_live_v1` |

Do **not** reuse a dry-run directory for live/smoke. The runner refuses mode mixing
and refuses a non-`--resume` start when `predictions.jsonl` / live metadata already
exists (no silent overwrite or delete).

Legacy offline dry-run at `outputs/hk_unified_exploratory155_mcp_free_v1` is left
untouched; new dry-runs should use the `*_dry_run_v1` path above.

## Formal artifacts

- `predictions.jsonl` (per-task append + fsync)
- `per_task_metrics.csv`
- `manual_grading.csv`
- `cost_summary.json`
- `error_summary.json`
- `run_metadata.json`
- `grouped_aggregates.json`
- `grading_summary.json` (also written)

Tool-related fields are explicitly `N/A` / zero (not vacuous tool-success).

## `--max-tasks` semantics

**Cumulative prefix**, not “add N more this round”.

- Tasks follow the locked exploratory run order (or `--ids` order).
- `--max-tasks N` keeps the first N of that ordered list.
- With `--resume`, already-completed task IDs in `predictions.jsonl` are skipped.
- Expanding `20 → 50 → 100 → 155` with `--resume` only runs the not-yet-done suffix.
- `max_tasks` is **not** part of resume identity. Resume identity **does** lock
  provider / provider host / model / prompt / question / run-order /
  temperature / max_tokens / split (prevents mid-run provider switching).

## Offline dry-run / mock

```bash
cd ../hk-equity-mcp-agent-hk-mcp-free
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
DRY=outputs/hk_unified_exploratory155_mcp_free_dry_run_v1

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout --dry-run --allow-dirty \
  --output-dir "$DRY"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout --mock --allow-dirty \
  --max-tasks 3 --output-dir outputs/hk_unified_mcp_free_mock_smoke_v1
```

## Real smoke (3 tasks; paid generation only — do not run in offline audit)

Locked HK-155 IDs (Financial / Market / Cross-domain), reused from Dev smoke-8
coverage and present in the exploratory manifest:

`fin_cf_001,mkt_dev_003,xd_dev_007`

Required env keys in `.env.relay-v41` (no Aliyun Token Plan):

- `LLM_PROVIDER=openai_compatible`
- `LLM_MODEL=deepseek-v4.1-flash`
- `LLM_BASE_URL` host = `api.zhizengzeng.com`

```bash
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
ENV_GEN="../hk-equity-mcp-agent/.env.relay-v41"
SMOKE=outputs/hk_unified_mcp_free_real_smoke3_v1

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --ids fin_cf_001,mkt_dev_003,xd_dev_007 \
  --env-file "$ENV_GEN" \
  --output-dir "$SMOKE"
```

No `--embedding-env-file`. First smoke start has **no** `--resume`.

## Formal 155 run (batched cumulative prefix)

```bash
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
ENV_GEN="../hk-equity-mcp-agent/.env.relay-v41"
LIVE=outputs/hk_unified_exploratory155_mcp_free_live_v1

# First formal batch — NO --resume, dedicated live dir (not dry-run)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" \
  --max-tasks 20 \
  --output-dir "$LIVE"

# Expand cumulative target with --resume (skips completed)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --resume --max-tasks 50 \
  --output-dir "$LIVE"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --resume --max-tasks 100 \
  --output-dir "$LIVE"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$ENV_GEN" --resume --max-tasks 155 \
  --output-dir "$LIVE"
```

Why the first formal command must **not** use `--resume`: `--resume` requires an
existing same-mode `run_metadata.json` and would incorrectly try to continue a
dry-run directory. First live start uses a clean live output dir.

## Progress / logs / offline eval

```bash
LIVE=outputs/hk_unified_exploratory155_mcp_free_live_v1

# completed count
wc -l "$LIVE/predictions.jsonl"
python - <<PY
import json
from pathlib import Path
p=Path("$LIVE/predictions.jsonl")
n=sum(1 for line in p.open() if line.strip()) if p.exists() else 0
meta=json.loads((Path("$LIVE")/"run_metadata.json").read_text())
print("completed", n, "target_question_count", meta.get("question_count"),
      "run_status", meta.get("run_status"))
PY

# error summary
python -m json.tool "$LIVE/error_summary.json" | head -80

# offline postprocess (no LLM)
PYTHONPATH=src "$PYTHON" scripts/postprocess_hk_unified_run.py --run-dir "$LIVE"

# optional final audit v2 (offline)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_final_audit_v2.py --run-dir "$LIVE"
```

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

## ETOM Empty-Route Control (scope — NOT part of MCP-Free Agent)

**Status:** deterministic offline control only. **Do not re-run** for the HK-155
MCP-Free campaign. **Do not** place it in the main HK-155 comparison table.
**Do not** use it as an ETOM formal baseline comparator or as a Proposed Method
baseline.

| Property | Value |
|---|---|
| Role | Offline empty-route control (predicts `[]` ops for every locked ETOM task) |
| Network / LLM / embedding / MCP calls | **0** |
| Included in HK-155 MCP-Free table | **No** |
| Included in ETOM formal baseline comparison | **No** |
| Proposed Method contrast baseline | **No** |
| Code retained | Yes (for reproducibility); results not refreshed this round |

```bash
# Historical / reproducibility only — not part of formal MCP-Free Agent runs
PYTHONPATH=src "$PYTHON" scripts/evaluate_etom_empty_route_control.py \
  --lock "../hk-equity-mcp-agent-etom/benchmarks/etom/formal_100_v1.lock.json" \
  --output-dir outputs/etom_empty_route_control_v1
```

Expected when executed offline: L1–L4 exact match = 0; L5 empty-route exact match = 1.
