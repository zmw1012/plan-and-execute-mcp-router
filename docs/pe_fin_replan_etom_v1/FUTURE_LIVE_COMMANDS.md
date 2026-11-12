# Future Live Commands (DO NOT RUN while HK-155 is active)

Wait until HK-155 is complete and frozen. Do not concurrently call generation
relay, embedding relay, or MCP.

## Verified paths

```bash
ROOT="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-replan-etom"
ORIG="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent"
PY="$ORIG/.venv/bin/python"   # verified existing venv; ETOM baseline historically used PYTHON/python3
ENV_GEN="$ORIG/.env.relay-v41"
ENV_EMB="$ORIG/.env.embedding-relay"
OUT="$ROOT/outputs/pe_fin_replan_etom_formal100_v1"
CACHE="$ROOT/.cache/pe_fin_replan_etom_operation_embeddings_v1"
cd "$ROOT"
```

Ensure pinned ETOM raw checkout exists (`data/etom/raw` symlink or
`bash scripts/fetch_etom_source.sh`).

## 1) Index status (safe anytime offline)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --index-status \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 2) Build index only (AFTER HK-155 freeze; uses embedding relay)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --build-index-only \
  --formal100 \
  --require-complete-index \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## 3) Engineering smoke / mock formal100 (offline OK)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --mock \
  --formal100 \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$ROOT/outputs/pe_fin_replan_etom_formal100_v1_mock"
```

## 4) Dry-run formal100 (offline OK; no live pass claim)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --dry-run \
  --formal100 \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --require-complete-index \
  --output-dir "$OUT"
```

## 5) Live formal100 (FUTURE ONLY)

```bash
caffeinate -dimsu "$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --formal100 \
  --require-complete-index \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
```

## 6) Resume (FUTURE ONLY)

```bash
caffeinate -dimsu "$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --formal100 \
  --resume \
  --require-complete-index \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
```

## 7) Final offline audit (after live completes)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --dry-run \
  --formal100 \
  --output-dir "$OUT" \
  --embedding-cache-dir "$CACHE"
# plus dedicated audit script once live artifacts exist
```

## Safety reminders

- mock/dry-run must never be labeled as live pass
- do not reuse HK embedding index directories
- do not modify `hk-equity-mcp-agent-pe-fin-lite` or its outputs
- no push / PR / merge from this offline scaffolding turn
