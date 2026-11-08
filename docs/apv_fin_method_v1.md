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

## Architecture

```
question
  → initial structured planner (task_spec + initial plan DAG)
  → monitor loop (continue | replan | verify | answer | clarify | refuse | fail)
  → step scheduler
       retrieve → find_operations → hard-filter → Tier A/B/C rerank → describe → invoke
       compute  → explicit expression/operands → deterministic safe arithmetic (no eval)
       reason/verify → concise summary only (no private CoT persistence)
       answer/clarify/refuse → terminal nodes
  → evidence ledger
  → final validator
  → controlled failure / justified refusal when budgets are exhausted
```

Only **retrieve** steps may enter the MCP operation resolver. Plans never hard-code `operation_id`.

## Schemas

Versioned JSON Schemas (`additionalProperties=false`):

- `apv_fin_plan_v1`
- `apv_fin_monitor_v1`
- `apv_fin_compute_v1`
- `apv_fin_final_v1`

If provider structured output is unavailable: JSON mode → deterministic parse → at most one format repair → fail closed.

## Budgets (config-driven)

From `configs/hk_unified_apv_fin_v1.yaml`:

- `max_plan_revisions: 2`
- `max_targeted_retrievals_per_step: 2`
- `max_total_mcp_invocations: 8`
- `max_llm_monitor_rounds: 6`
- `max_same_operation_repeats: 1`
- `max_final_repair_rounds: 1`

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
  --output-dir outputs/apv_fin_diagnostic12_dry_v1
```

## Future real smoke (DO NOT RUN in this implementation round)

```bash
PYTHONPATH=src python scripts/run_hk_unified_apv_fin_experiment.py \
  --smoke --allow-dirty \
  --env-file .env.aliyun \
  --embedding-env-file .env.embedding \
  --output-dir outputs/apv_fin_real_smoke8_v1
```
