# Future retrieval-audit commands (DO NOT EXECUTE in the readiness/prep round)

Verified against CLI `--help` / dry-run patterns. Use the absolute paths below (not system Python).

```bash
PY="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
ROOT="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-lite"
GEN="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.relay-v41"
EMB="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.env.embedding-relay"
OUT="$ROOT/outputs/pe_fin_lite_retrieval_all155_planner_v3"
CACHE="$ROOT/.cache/pe_fin_lite_operation_embeddings_v1"
```

Constraints:

- independent `OUT` (do not mix invalid planner v1 / Diagnostic-10 trees into formal selection)
- keep INVALIDATED mixed-dim archive under `question_level_invalid_mixed_dim64_v1/`
- reuse complete QQ 1536 cache + real catalog `ac4c109b…` (no re-embed if identities match)
- `no_unseen_heldout_claim=True`
- real offline stages **must** pass `--embedding-env-file "$EMB"` (identity-only; zero network)
- do **not** run full PE-Fin Lite agent155 in this sequence

Protocol: `retrieval_experiment_protocol_v1.md`

---

## 1) real index-status (zero network; emb identity)

```bash
cd "$ROOT"
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --index-status \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

Expect selected identity `ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df`
(host=`api.zhizengzeng.com`, model=`text-embedding-3-small`, dim=1536).

## 2) Planner-only all155 v3 live (generation LLM)

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --planner-only \
  --live \
  --require-complete-index \
  --env-file "$GEN" \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 3) zero-network Planner gate/audit

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --planner-cache-audit \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

If `systemic_schema_or_protocol_failure=true` → stop and inspect.  
Sparse provider empty/truncated failures → report only; do not invent Planner v4.

If audit shows `ok=154` / `failed=1` (or any `ok=false` rows) and
`systemic_schema_or_protocol_failure=false`, run the generic failed-row retry
**before** step-query embedding. Do **not** pass `--allow-incomplete-planner-cache`
for formal step embedding / formula lock.

### 3b) Retry failed Planner rows only (generation LLM; auto-derive failed IDs)

Freezes `planner_cache_attempt1/` (immutable snapshot of the incomplete cache),
retries every `ok=false` task once with the same prompt/schema/canonicalizer /
provider/model, and atomically promotes `planner_cache/planner_outputs.jsonl`
only if the merge is 155/155 ok. Original attempt1 SHA remains unchanged.
No task-id hardcode.

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --retry-failed-planner-cache \
  --live \
  --env-file "$GEN" \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

Then re-run step 3 (`--planner-cache-audit`). Expect:

- `rows=155`, `unique_task_ids=155`, `ok=155`, `failed=0`
- `complete_155_ok=true`
- `formal_step_pipeline_allowed=true`

Optional dry-run (no LLM; lists failed IDs + freezes/verifies snapshot):

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --retry-failed-planner-cache \
  --dry-run \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 4) Step-query embeddings live (Embedding API)

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --embed-planner-step-queries \
  --live \
  --require-complete-index \
  --env-file "$GEN" \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 5) zero-network R0 / R1 / R2 / R3 / R_runtime ranking

Question-level R0 (reuse QQ; do not pass `--live`):

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --question-level \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

Step-level R1/R2/R3@α/R_runtime:

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --rank-from-planner-cache \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 6–7) development70 formula selection + formal lock

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --select-formula \
  --formal-lock \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

R0 is excluded from Plan-first selection. If R1/R2/R3 wins → `runtime_sync_required=true` (sync later; do not patch method now).

## 8) reserved85 frozen evaluation

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --evaluate-frozen-formula \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 9) all155 descriptive report

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --report-all \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 10) catalog coverage

```bash
PYTHONPATH=src "$PY" scripts/run_pe_fin_lite_retrieval_audit.py \
  --catalog-coverage \
  --require-complete-index \
  --embedding-env-file "$EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

---

## Explicitly out of scope here

```text
# full PE-Fin Lite agent over HK-155
# re-embed question queries when QQ identity already matches
# treat INVALIDATED mixed-dim R0 as valid
# retune alpha on reserved-85
# modify Planner prompt/schema/method runtime in this sequence
```
