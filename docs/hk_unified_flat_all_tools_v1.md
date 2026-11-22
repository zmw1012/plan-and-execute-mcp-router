# Unified HK Flat All Tools Baseline v1

Branch: `feature/flat-all-tools-baseline-v1`  
Worktree: `../hk-equity-mcp-agent-flat-all-tools`  
Base: `1eb3315` (HK-155 final audit v2 + grader v1.2)

## Method definition

**Flat All Tools** is a true **no-routing / full-tool exposure** control for HK-155:

- Every Agent planning turn exposes **all 39 runtime operations** simultaneously
  (operation_id, name, description, complete input JSON schema, constraints).
- The model selects one or more underlying operations directly and supplies arguments.
- Execution goes through `UnifiedRealMcpGateway` only (fail closed for unknown /
  blocked / schema-invalid selections).
- There is **no** embedding, `find_operations`, `describe_operation`, operation
  retrieval, server retrieval, MCP-Zero hierarchical routing, or pre-filtered Top-K.

### Sole major difference vs Operation-Visible

| Dimension | Operation-Visible | Flat All Tools |
|---|---|---|
| Tool exposure | Meta-tools only (`find` / `describe` / `invoke_api`) | All 39 runtime ops directly |
| Routing / retrieval | Embedding + lexical Top-K | None |
| Catalog | Same frozen 42 unique / 39 runtime / 3 blocked | Same |
| Model / temp / tokens / turns | `deepseek-v4.1-flash`, 0.0, 1024, 8 | Same |
| Answer contract | `FINAL_ANSWER_CONTRACT` | Same |
| Gateway | `UnifiedRealMcpGateway` | Same |
| Evaluator | `hk_unified_evaluator_v1.2` | Same (Flat-compatible traces) |
| Tool protocol | Prompt-JSON ReAct (not provider-native FC) | Same harness |

Flat is therefore a **routing ablation**: it holds catalog, generation, contract,
gateway, and grading fixed while removing the Operation-Visible retrieval boundary.

## Catalog identity

| Count | Meaning |
|---|---|
| 42 | Unique logical `operation_id`s after domain merge |
| 39 | Runtime-exposed operations (Agent-visible) |
| 3 | Blocked/rejected — never exposed, never callable |

Namespaced Cross-domain rows may exceed 42, but runtime registration is
**deduplicated by `operation_id`**. Blocked/rejected ops must not appear in the
Flat schema bundle or action mapping.

## Why Flat is a routing ablation

Operation-Visible measures whether an agent can **find then invoke** the right
tool under a meta-tool boundary. Flat removes that search problem: the correct
tool is always in context. Comparing Flat vs Operation-Visible isolates the
cost/benefit of retrieval routing under an otherwise identical stack.

## No retrieval / embedding

Per Flat task and run:

- `retrieval_metrics = N/A` (never coerced to `1.0`)
- `retrieval_calls = 0`
- `embedding_calls = 0`, `embedding_tokens = 0`
- `resolver_calls.find_operations = 0`
- `resolver_calls.describe_operation = 0`

Tool-selection metrics (precision / recall / exact_tool_set) **do** apply.

## Context and token-cost risk

All 39 complete schemas are injected into the system prompt every turn. Recorded
in `run_metadata.json`:

- `system_prompt_hash`
- `schema_bundle_sha256`
- `serialized_schema_char_count`
- `schema_bundle_token_estimate` (`ceil(char_count/4)`, reproducible offline)
- `action_mapping_fingerprint`
- `catalog_fingerprint`

This is intentional: Flat trades retrieval complexity for **context overhead**.
Large schema bundles increase input tokens and may degrade tool selection when
the prompt becomes very long.

## Why ETOM (~2,375 tools) does not run Flat

ETOM-scale catalogs make full-schema exposure impractical:

1. Prompt / context limits — thousands of complete JSON schemas exceed model
   windows and explode cost.
2. Selection noise — with thousands of near-duplicate tools, direct exposure
   ceases to be a fair routing ablation and becomes a stress test of context
   packing.
3. HK-155 scope — this baseline uses the frozen 39-runtime HK catalog shared
   with Operation-Visible. ETOM Flat would be a different experiment.

## Provider-safe action names

When `operation_id` is not a safe action name (dots, length, etc.):

1. Build a deterministic, reversible `action_name ↔ operation_id` map.
2. Resolve collisions with stable digests.
3. Freeze and fingerprint the mapping (`action_mapping_fingerprint`).
4. Keep original `operation_id` visible in the schema bundle and in evidence.
5. Evaluator / gold identity uses **operation_id only**, never internal aliases.

## Artifacts (HK unified compatible)

- `predictions.jsonl`
- `per_task_metrics.csv`
- `manual_grading.csv`
- `cost_summary.json`
- `run_metadata.json`
- `error_summary.json`
- `grouped_aggregates.json`
- `retrieval_analysis.json` (Flat: N/A semantics)
- `evaluation_metadata.json`

Report separately:

- **A.** Final answer accuracy  
- **B.** Exact tool-set success  
- **C.** Strict overall task success  

## Offline dry-run / mock (allowed)

```bash
cd ../hk-equity-mcp-agent-flat-all-tools
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"

# Dry-run (no LLM / MCP / embedding)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_flat_all_tools_experiment.py \
  --split exploratory_all --confirm-retire-heldout --dry-run --allow-dirty \
  --output-dir outputs/hk_unified_exploratory155_flat_all_tools_dry_run_v1

# Four-domain mock (offline)
PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_flat_all_tools_experiment.py \
  --split development \
  --ids fin_cf_001,mkt_dev_014,news_dev_006,xd_dev_007 \
  --mock --allow-dirty \
  --output-dir outputs/hk_unified_flat_all_tools_mock_4domain_v1
```

**Do not report mock / dry-run as experimental results.**

## Generation identity (formal)

| Field | Value |
|---|---|
| `LLM_PROVIDER` | `openai_compatible` |
| Host | `api.zhizengzeng.com` |
| `LLM_MODEL` | `deepseek-v4.1-flash` |
| Env file | `../hk-equity-mcp-agent/.env.relay-v41` |
| `json_mode` | `true` (JSON action/final only; no ATR) |
| Embedding | **not used** (no embedding env) |

Wrong provider / host / model fail closed. Aliyun Token Plan is not accepted.

## Future real smoke (do NOT run in this offline round)

```bash
PYTHON="../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
ENV_GEN="../hk-equity-mcp-agent/.env.relay-v41"

PYTHONPATH=src "$PYTHON" scripts/run_hk_unified_flat_all_tools_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --ids fin_cf_001,mkt_dev_014,news_dev_006,xd_dev_007 \
  --env-file "$ENV_GEN" \
  --output-dir outputs/hk_unified_flat_all_tools_real_smoke4_v1
```

Use a **separate** `--output-dir` from dry-run and mock. Never mix modes in one
directory. Without `--resume`, existing `predictions.jsonl` is refused (no
overwrite).

## Safety

- Git dirty / catalog drift / schema drift / provider / host / model mismatch → fail closed.
- API keys never written to metadata, logs, or git.
- No silent fallback to mock `FinancialServer` / local gold tools.
