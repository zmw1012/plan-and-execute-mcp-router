# Future retrieval-audit commands (DO NOT EXECUTE in this freeze round)

Absolute paths:

```text
PYTHON=/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group\ project/hk-equity-mcp-agent-pe-fin-lite/.venv/bin/python
ROOT=/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group\ project/hk-equity-mcp-agent-pe-fin-lite
GEN_ENV=/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group\ project/hk-equity-mcp-agent/.env.relay-v41
EMB_ENV=/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group\ project/hk-equity-mcp-agent/.env.embedding-relay
OUT=$ROOT/outputs/pe_fin_lite_retrieval_all155_planner_v3
CACHE=$ROOT/.cache/pe_fin_lite_operation_embeddings_v1
```

Constraints for all steps:

- independent `OUT` (never mix with invalid planner v1 / Diagnostic-10 caches)
- resume/checkpoint protections remain enabled by runner defaults
- `no_unseen_heldout_claim=True`
- **do not** run full PE-Fin Lite agent155

---

## 1) index-status (offline)

```bash
cd "$ROOT"
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --index-status \
  --output-dir "$OUT" \
  --embedding-cache-dir "$CACHE"
```

Expect: complete identity `ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df`.  
If incomplete only then run `--build-index-only --live` with `$EMB_ENV` (not required if status is complete).

## 2) embed-question-queries (Embedding API — only if not reusing)

If reusing live_v1 QQ vectors, copy/re-point only after identity checks; otherwise:

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --embed-question-queries \
  --live \
  --require-complete-index \
  --env-file "$GEN_ENV" \
  --embedding-env-file "$EMB_ENV" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 3) planner-only all155 v3 (Generation API — required)

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --planner-only \
  --live \
  --require-complete-index \
  --env-file "$GEN_ENV" \
  --embedding-env-file "$EMB_ENV" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

Must produce 155 unique rows under `planner_only_v3` identity.  
Do **not** pass `--planner-diagnostic10` here.

## 4) embed-planner-step-queries (Embedding API)

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --embed-planner-step-queries \
  --live \
  --require-complete-index \
  --env-file "$GEN_ENV" \
  --embedding-env-file "$EMB_ENV" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 5) question-level R0 offline ranking

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --question-level \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 6) step-level R1/R2/R3 offline ranking

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --rank-from-planner-cache \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 7) formula selection on development70 only

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --select-formula \
  --formal-lock \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

Selection must use development-origin 70 only; no retune on reserved85.

## 8) freeze selected formula

Produced by step 7 when formal gates pass (`formula_lock.json`).  
Do not hand-edit lock contents.

## 9) descriptive evaluation on development70 / reserved85 / all155

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --evaluate-frozen-formula \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

Optional combined report:

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --report-all \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 10) catalog coverage report

```bash
PYTHONPATH=src "$PYTHON" scripts/run_pe_fin_lite_retrieval_audit.py \
  --catalog-coverage \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

---

## Explicitly out of scope

```text
# DO NOT RUN as part of retrieval audit readiness
# full PE-Fin Lite agent over HK-155
```
