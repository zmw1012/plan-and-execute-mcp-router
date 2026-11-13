# Future Live Commands (DO NOT RUN while HK-155 is active)

Wait until HK-155 is complete and frozen. Do not concurrently call generation
relay, embedding relay, or MCP.

## Dual live gate (mandatory for any live path)

All live embedding builds and live generation runs require **both**:

1. CLI: `--allow-live-formal100`
2. Env: `PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1`

Truth table (verified offline; exit 3 = gated):

| CLI `--allow-live-formal100` | Env `=1` | Live formal100 / live smoke | Live build-index |
| --- | --- | --- | --- |
| no | no | refuse (exit 3) | refuse (exit 3) |
| yes | no | refuse (exit 3) | refuse (exit 3) |
| no | yes | refuse (exit 3) | refuse (exit 3) |
| yes | yes | allowed (future) | allowed (future; embedding API only) |

Offline mock / dry-run / index-status / `--build-index-only --mock-embeddings`
do **not** need the live gate.

Gate scopes:

- **build-index-only (live)**: authorizes the **embedding API** only. Does not
  claim formal100 accuracy. Does not call generation LLM / MCP.
- **formal100 / Smoke-5 live**: authorizes generation LLM + routing + query
  embeddings. Still routing-only for ETOM tool invoke (no real ETOM MCP tool
  business execution in this condition).

Do **not** pass `--require-complete-index` together with `--build-index-only`
(the build creates the index; completeness is checked afterwards).

## Verified paths

```bash
ROOT="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-replan-etom"
ORIG="/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent"
# Prefer the MCP-capable sibling venv (has `mcp` + `httpx`; used by ETOM MCP discovery).
# `.venv` imports the package but lacks `mcp` and is NOT preferred for live path.
PY="$ORIG/.venv-aktools-discovery/bin/python"
# Proposed-method generation identity: openai_compatible + api.zhizengzeng.com
# + deepseek-v4.1-flash via .env.relay-v41 (NOT baseline .env.aliyun).
ENV_GEN="$ORIG/.env.relay-v41"
ENV_EMB="$ORIG/.env.embedding-relay"
OUT="$ROOT/outputs/pe_fin_replan_etom_formal100_v1"
SMOKE_OUT="$ROOT/outputs/pe_fin_replan_etom_smoke5_v4"
# Smoke-v2 (aliyun/HTTP 403) and Smoke-v3 (grounding/CLI defects) are INVALIDATED.
# Do not resume those directories.
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

## 2) Build index only (AFTER HK-155 freeze; embedding relay only)

Live build needs the dual gate. Do **not** add `--require-complete-index` here.

```bash
PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1 caffeinate -dimsu "$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --build-index-only \
  --allow-live-formal100 \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/build_index.log"
```

Offline hash-index engineering (never under shared `.cache` by default):

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --build-index-only \
  --mock-embeddings \
  --output-dir "$ROOT/outputs/pe_fin_replan_etom_mock_index_v1"
```

## 3) Index status with require-complete-index (after build)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --index-status \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --embedding-env-file "$ENV_EMB" \
  --output-dir "$OUT"
```

Fails closed (exit 3) if the live index is incomplete / identity-mismatched.

## 4) ETOM Smoke-5 live (AFTER HK-155 freeze; path smoke only)

First-per-level in frozen order: `L1-q556`, `L2-q766`, `L3-q108`, `L4-q57`,
`L5-q68`. Role=`engineering_live_path_smoke_only`.
`explicitly_not_formal_accuracy=true`. **Never** resume into formal100 output.

```bash
PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1 caffeinate -dimsu "$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --engineering-smoke5 \
  --allow-live-formal100 \
  --require-complete-index \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$SMOKE_OUT" \
  2>&1 | tee -a "$SMOKE_OUT/run.log"
```

## 5) Smoke-5 offline audit (after smoke completes)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --dry-run \
  --engineering-smoke5 \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$SMOKE_OUT"
```

Inspect `$SMOKE_OUT/run_metadata.json` / `predictions.jsonl`:
`live_pass_claimed=false`, `run_role=engineering_live_path_smoke_only`,
`explicitly_not_formal_accuracy=true`, task_ids length 5.

## 6) ETOM formal100 live (FUTURE ONLY; after Smoke-5 OK)

```bash
PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1 caffeinate -dimsu "$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --formal100 \
  --allow-live-formal100 \
  --require-complete-index \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
```

## 7) formal100 resume (FUTURE ONLY)

```bash
PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1 caffeinate -dimsu "$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --formal100 \
  --allow-live-formal100 \
  --resume \
  --require-complete-index \
  --env-file "$ENV_GEN" \
  --embedding-env-file "$ENV_EMB" \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
```

## 8) formal100 final offline audit (after live completes)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --dry-run \
  --formal100 \
  --require-complete-index \
  --embedding-cache-dir "$CACHE" \
  --embedding-env-file "$ENV_EMB" \
  --output-dir "$OUT"
```

Plus inspect `$OUT/run_metadata.json`, `grouped_results.json`,
`predictions.jsonl` (`live_pass_claimed` must remain false for post-hoc /
exploratory labeling; do not treat as HK primary).

## Offline engineering (OK anytime; no live gate)

```bash
"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --mock \
  --formal100 \
  --output-dir "$ROOT/outputs/pe_fin_replan_etom_formal100_v1_mock"

"$PY" scripts/run_etom_pe_fin_replan_experiment.py \
  --dry-run \
  --formal100 \
  --embedding-cache-dir "$CACHE" \
  --output-dir "$OUT"
```

## Safety reminders

- mock/dry-run must never be labeled as live pass (`live_pass_claimed` is always false in artifacts)
- Smoke-5 output dir must stay separate from formal100; `run_role` is in resume identity
- a non-resume run into an output dir that already has predictions is refused; use `--resume` (identity-checked) or a new dir
- Ctrl-C/SIGTERM => `run_status=aborted`, rows kept (exit 130); unknown runner errors also finalize `aborted` and re-raise
- do not reuse HK embedding index directories (`pe_fin_lite_operation_embeddings_v1`) or baseline `etom_embeddings`
- do not modify `hk-equity-mcp-agent-pe-fin-lite` or its outputs
- no push / PR / merge from this offline scaffolding / readiness audit turn
