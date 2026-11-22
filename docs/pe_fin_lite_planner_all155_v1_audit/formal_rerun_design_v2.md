# Formal Planner-only v2 Rerun Design

## Output directory

`outputs/pe_fin_lite_retrieval_all155_planner_v2`

Do **not** overwrite `outputs/pe_fin_lite_retrieval_all155_live_v1`.

## Reuse (identity must match exactly)

Allowed reuse from live_v1 / shared cache:

- 39-operation catalog embedding index (`.cache/pe_fin_lite_operation_embeddings_v1`)
- R0 question-query vector cache (copy or re-point only if identity matches)
- R0 question-level rankings (remain valid; not invalidated by Planner v1 mark)

## Must wait for Planner v2

- `--embed-planner-step-queries`
- R1/R2/R3 `--rank-from-planner-cache`
- `--select-formula --formal-lock`
- reserved-85 / all-155 frozen formula evaluation

## Fail-closed gates

1. Planner rows == 155 and unique task IDs == 155
2. Metadata records `n_failed` and `failed_task_ids`
3. Step embedding / ranking require complete Planner cache by default
4. Diagnostic only: `--allow-incomplete-planner-cache`
5. Formal formula selection **never** allows incomplete Planner cache
6. v1 cache identity cannot resume into v2 (protocol + canonicalizer hash)

## Protocol

- `planner_only_protocol_version = pe_fin_lite_planner_only_v2`
- Structural canonicalizer hash included in resume identity
- Budgets unchanged: Top-K=5, Describe-N=3, Primary+1 Fallback, LLM≤8, MCP≤8, Monitor=0, global replan=0
