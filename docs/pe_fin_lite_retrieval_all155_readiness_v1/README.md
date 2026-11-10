# PE-Fin Lite all-155 retrieval audit — offline cache readiness

Zero-network readiness audit only. No LLM / Embedding / MCP / HK-155 agent run.

Audit commit context: method freeze `pe_fin_lite_v1.2` @ `baf572461dce670fddb71291128e82360dbef2f0`.

## Inventory checked (read-only)

| Path | Role |
|---|---|
| `outputs/pe_fin_lite_retrieval_all155_live_v1/` | early live retrieval + planner-only v1 |
| `outputs/pe_fin_lite_planner_protocol_diagnostic10_v3/` | Planner protocol Diagnostic-10 only |
| `outputs/pe_fin_lite_retrieval_all155_planner_v2/` | **missing** (never created) |
| `outputs/pe_fin_lite_retrieval_all155_v1/` | mock/offline scaffolding + formula schema |
| `.cache/pe_fin_lite_operation_embeddings_v1/` | shared catalog embedding index |
| current retrieval audit script/config/identity | `scripts/run_pe_fin_lite_retrieval_audit.py` + `pe_fin_lite_retrieval_audit/*` |

## Itemized findings

### 1. Catalog index vs latest operation documents

| Field | Value |
|---|---|
| current `operation_documents_sha256` | `5989196c1c2765cfe5ec6c094c80189cb9b3b18ed9cc2072ab057b29d3f0dd27` |
| current catalog fingerprint | `...financial:5d5881b6e2b5...` (matches D12 v3) |
| current embedding identity digest | **`ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df`** |
| index complete for current identity | **yes** (39/39 vectors) |
| note on `439b4c61…` | **not found** in caches or recomputation; live current digest is `ac4c109b…` (same as D12 v3) |

Old live_v1 catalog snapshot pointed at `420290c7…` / financial fingerprint `18634ccfb733` → **stale relative to current docs**.

### 2. Question-query embeddings reuse

| Field | Value |
|---|---|
| path | `outputs/pe_fin_lite_retrieval_all155_live_v1/question_query_cache/` |
| complete | yes (`n=155/155`) |
| binds to operation_documents? | **no** (question/manifest/model only) |
| reusable for later offline re-rank? | **conditionally yes** if question identity / query-construction version unchanged; copy into a **new** output dir rather than mixing with invalid planner cache |

### 3. Old planner-only all155 v1 cache

| Field | Value |
|---|---|
| path | `.../retrieval_all155_live_v1/planner_cache/` |
| protocol | `pe_fin_lite_planner_only_v1` |
| n_tasks / n_errors | 155 / 41 |
| status | **`invalid_for_formal_step_level_retrieval_due_to_planner_protocol_failure_rate`** |
| prompt/schema vs current | mismatch (`ed778588…` / `9eb087e2…` vs current `b347eed8…` / `fdf72836…`) |
| current expected planner identity | `13a26b812b4565b9f969dc3eb78950d58d50789757a199384f0e9aa9a97b39c8` (`planner_only_v3`) |
| verdict | **INVALID — do not resume into formal R1–R3 / formula lock** |

### 4. Complete planner-only v3 all155 cache?

**No.** Only Diagnostic-10 planner v3 exists (`n_tasks=10`). No all-155 planner-only v3 directory.

### 5. Diagnostic-10 planner cache must not impersonate all155

| Field | Value |
|---|---|
| path | `outputs/pe_fin_lite_planner_protocol_diagnostic10_v3/planner_cache/` |
| n_tasks | 10 |
| `cannot_resume_all155` | **true** |
| `cannot_formal_lock` | **true** |
| role | engineering protocol diagnostic only |

### 6. Step-query embeddings

**Absent.** No `step_query_cache/` under live_v1 or elsewhere with identity matching current docs/index.

### 7. R0 question-level ranking

| Field | Value |
|---|---|
| live_v1 R0 present | yes (`question_level/`) |
| cached `operation_documents_sha256` | `01e0bf46…` (**≠** current `5989196c…`) |
| cached catalog fingerprint | financial `18634ccfb733` (**≠** current `5d5881b6e2b5`) |
| verdict | **must recompute** against current catalog index (`ac4c109b…`) |

Question-query vectors may be reused as inputs; the ranking artifact itself is stale.

### 8. R1 / R2 / R3

**Not formally computed** on a valid planner-only v3 all155 cache. Mock formula preview under `retrieval_all155_v1` is schema/machinery only.

### 9. formula_lock

| Artifact | Status |
|---|---|
| `formula_lock.json` (formal) | **does not exist** |
| `formula_lock_schema.json` | schema-only / `schema_only_no_formal_selection` |

### 10. What needs API vs offline

| Stage | Network? |
|---|---|
| `--index-status` | offline |
| `--build-index-only` (only if incomplete; currently complete) | Embedding API |
| `--embed-question-queries` | Embedding API (skip if reusing complete QQ cache with matching identity) |
| `--planner-only --live` all155 v3 | **Generation API** (required; new output dir) |
| `--embed-planner-step-queries` | Embedding API |
| `--question-level` (R0) | offline ranking |
| `--rank-from-planner-cache` (R1/R2/R3) | offline ranking |
| `--select-formula --formal-lock` (dev70 only) | offline |
| `--evaluate-frozen-formula` | offline |
| `--catalog-coverage` / `--report-all` | offline |
| full PE-Fin Lite agent155 | **out of scope / do not run here** |

## Reusable vs invalid summary

| Cache | Verdict |
|---|---|
| catalog index `ac4c109b…` | **reusable / current** |
| question-query vectors (155) | **conditionally reusable** (question identity) |
| live_v1 R0 rankings | **invalid vs current docs/index** → recompute |
| live_v1 planner-only v1 | **invalid** |
| Diagnostic-10 planner v3 | **not all155** |
| planner_v2 output dir | **missing** |
| step-query embeddings | **missing** |
| formal formula_lock | **missing** |

## Required next output directory

Use a **new** directory, e.g.:

`outputs/pe_fin_lite_retrieval_all155_planner_v3/`

Do **not** resume into:

- `outputs/pe_fin_lite_retrieval_all155_live_v1/`
- `outputs/pe_fin_lite_retrieval_all155_planner_v2/` (missing / obsolete design target)
- `outputs/pe_fin_lite_planner_protocol_diagnostic10_v3/`

Exact commands: `future_commands.md`.
