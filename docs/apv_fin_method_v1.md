# APV-Fin Method v1

Adaptive Plan-Visible Finance Router (APV-Fin) is the **proposed method**, separated from the Operation-Visible baseline.

## Reused baseline components (not rewritten)

1. `UnifiedRealMcpGateway`
2. Frozen unified HK operation catalog (`build_unified_catalog`)
3. Operation-Visible find / describe / invoke boundary (`OperationResolverServer`)
4. HK unified dataset loader
5. Evaluator / grader v1.2
6. Existing answer contract (`answered` / `clarification_needed` / `unavailable`)
7. Resume / checkpoint / output artifact conventions
8. Provider metadata + fail-closed environment loading

Baseline OV behavior, prompts, configs, and raw outputs are **not** modified.

## Architecture (canonical live order)

```
question
  → initial structured planner (task_spec + initial plan DAG)
  → scheduler selects ready steps
  → execute retrieve / compute / reason / verify
       retrieve → find_operations → hard-filter → Tier A/B/C rerank → describe → invoke
       compute  → explicit expression/operands → deterministic safe arithmetic (no eval)
       reason/verify → concise summary only (no private CoT persistence)
  → write evidence ledger
  → monitor (continue | replan | verify | answer | clarify | refuse | fail)
  → final validator
  → controlled failure / justified refusal when budgets are exhausted
```

Monitor runs **after** observations. Idle monitor without new observations is bounded (`max_idle_monitor_rounds`). Live mode forbids `forced_select` / mock hooks. No MCP-Zero ATR.

Only **retrieve** steps may enter the MCP operation resolver. Plans never hard-code `operation_id`.

## Formal live identity (fail-closed)

- `generation_provider`: `openai_compatible`
- `generation_host`: `api.zhizengzeng.com`
- `generation_model`: `deepseek-v4.1-flash`
- `embedding_model`: `text-embedding-3-small` (host recorded from embedding env; never log API keys)

Formal env (sibling checkout):

- GEN: `../hk-equity-mcp-agent/.env.relay-v41`
- EMB: `../hk-equity-mcp-agent/.env.embedding-relay`
- Python: `../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python`

Formal output dirs:

- `outputs/apv_fin_dry_run_v1`
- `outputs/apv_fin_real_smoke4_v1`
- `outputs/apv_fin_diagnostic12_live_v1`

## Schemas

Versioned JSON Schemas (`additionalProperties=false`):

- `apv_fin_plan_v1`
- `apv_fin_monitor_v1`
- `apv_fin_compute_v1`
- `apv_fin_final_v1`

If provider structured output is unavailable: JSON mode → deterministic parse → at most one format repair → fail closed.

## Budget limits (config; do not retune for mock)

From `configs/hk_unified_apv_fin_v1.yaml` (worst-case call estimate derived offline without retuning):

- `max_plan_revisions: 2`
- `max_targeted_retrievals_per_step: 2`
- `max_total_mcp_invocations: 8`
- `max_llm_monitor_rounds: 6`
- `max_same_operation_repeats: 1`
- `max_final_repair_rounds: 1`
- `max_idle_monitor_rounds: 1`
- `max_total_llm_calls_per_task: 14` (all generation calls incl. repairs; excludes embedding/MCP)

## Metadata + reranking

Sidecar: `configs/operations/hk_unified_operation_metadata_v1.yaml` (39 runtime ops).

Company-generic scopes only (`hk_listed_issuer` / `hk_equity` / `index`), not issuer whitelists.

Compatibility tiers: A (compatible) → B (unknown/partial) → C (conflict, fallback only).

## Offline commands

```bash
PYTHONPATH=src python scripts/run_hk_unified_apv_fin_experiment.py \
  --smoke --dry-run --allow-dirty \
  --output-dir outputs/apv_fin_dry_smoke_v1

PYTHONPATH=src python scripts/run_hk_unified_apv_fin_experiment.py \
  --smoke --mock --allow-dirty --max-tasks 4 \
  --output-dir outputs/apv_fin_mock_smoke4_v1

PYTHONPATH=src python scripts/run_hk_unified_apv_fin_experiment.py \
  --diagnostic12 --dry-run --allow-dirty \
  --split exploratory_all --confirm-retire-heldout \
  --output-dir outputs/apv_fin_dry_run_v1

PYTHONPATH=src python scripts/run_hk_unified_apv_fin_experiment.py \
  --smoke4 --dry-run --allow-dirty \
  --split exploratory_all --confirm-retire-heldout \
  --output-dir outputs/apv_fin_dry_run_v1
```

## Real Smoke-4 commands (paid; run only after explicit confirmation)

- v1: invalidated (embedding index timeout before tasks).
- v2: invalidated engineering smoke (`planner_fail_closed` / schema contract mismatch).
- v3: invalidated engineering smoke (empty-arg invoke + monitor contract; not an accuracy result). Raw preserved under `outputs/apv_fin_real_smoke4_v3/`.
- Next paid run: **v4** after `tool_binding_and_monitor_protocol_v3_before_valid_method_evaluation`.

Shared catalog index cache: `.cache/apv_fin_operation_embeddings_v1/`

```bash
cd "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-apv-fin"

# (a) build/resume shared operation embedding index only
PYTHONPATH=src \
"/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/python" \
  scripts/run_hk_unified_apv_fin_experiment.py \
  --build-index-only \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay"

# (b) local status (zero network)
PYTHONPATH=src \
"/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/python" \
  scripts/run_hk_unified_apv_fin_experiment.py \
  --index-status \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay"

# (c) Smoke-4 live v4 — requires complete index; no --allow-dirty
PYTHONPATH=src \
"/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/python" \
  scripts/run_hk_unified_apv_fin_experiment.py \
  --smoke4 \
  --split exploratory_all \
  --confirm-retire-heldout \
  --allow-hkex-download \
  --require-complete-index \
  --env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.relay-v41" \
  --embedding-env-file "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay" \
  --output-dir "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-apv-fin/outputs/apv_fin_real_smoke4_v4"
```
