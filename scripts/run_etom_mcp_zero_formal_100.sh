#!/usr/bin/env bash
# MCP-Zero × ETOM formal-100 helper.
# Paid modes require ENV_FILE + EMBEDDING_ENV_FILE + ETOM_SOURCE_DIR.
# Cursor agents must not execute live/build-index; run those in a Terminal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-preflight}"
OUT="${OUT:-$ROOT/outputs/etom_external_formal100_mcp_zero_v1}"
CACHE="${EMBEDDING_CACHE_DIR:-$ROOT/.cache/mcp_zero_etom_v1}"
# Use nonempty placeholder-free arrays; expand via ${arr[@]+"${arr[@]}"} for set -u.
MAX_TASKS_ARGS=()
SOURCE_ARGS=()
FIXTURE_ARGS=()
ENV_ARGS=()
DIRTY_ARGS=()
if [[ -n "${MAX_TASKS:-}" ]]; then
  MAX_TASKS_ARGS=(--max-tasks "$MAX_TASKS")
fi
if [[ -n "${ETOM_SOURCE_DIR:-}" ]]; then
  SOURCE_ARGS=(--etom-source-dir "$ETOM_SOURCE_DIR")
fi
if [[ -n "${ETOM_FIXTURE_DIR:-}" ]]; then
  FIXTURE_ARGS=(--etom-fixture-dir "$ETOM_FIXTURE_DIR")
fi
if [[ -n "${ENV_FILE:-}" ]]; then
  ENV_ARGS+=(--env-file "$ENV_FILE")
fi
if [[ -n "${EMBEDDING_ENV_FILE:-}" ]]; then
  ENV_ARGS+=(--embedding-env-file "$EMBEDDING_ENV_FILE")
fi
if [[ "${ALLOW_DIRTY:-0}" == "1" ]]; then
  DIRTY_ARGS=(--allow-dirty)
fi

_require_paid_env() {
  if [[ -z "${ENV_FILE:-}" || -z "${EMBEDDING_ENV_FILE:-}" || -z "${ETOM_SOURCE_DIR:-}" ]]; then
    cat >&2 <<EOF
Paid mode requires:
  ENV_FILE=.../.env.relay-v41
  EMBEDDING_ENV_FILE=.../.env.embedding-relay
  ETOM_SOURCE_DIR=.../hk-equity-mcp-agent-etom/data/etom/raw
EOF
    exit 2
  fi
}

case "$MODE" in
  dry-run)
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --dry-run \
      --output-dir "$OUT" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${FIXTURE_ARGS[@]+"${FIXTURE_ARGS[@]}"} ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  preflight)
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --preflight \
      --output-dir "$OUT" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${FIXTURE_ARGS[@]+"${FIXTURE_ARGS[@]}"} ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  mock)
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --mock \
      --output-dir "${OUT}_mock" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${FIXTURE_ARGS[@]+"${FIXTURE_ARGS[@]}"} ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  index-status)
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --index-status \
      --output-dir "$OUT" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${FIXTURE_ARGS[@]+"${FIXTURE_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  build-index)
    _require_paid_env
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --build-index-only \
      --output-dir "$OUT" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  run)
    _require_paid_env
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --live \
      --require-complete-index --output-dir "$OUT" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  resume)
    _require_paid_env
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --live --resume \
      --require-complete-index --output-dir "$OUT" --embedding-cache-dir "$CACHE" \
      ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"} ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} ${DIRTY_ARGS[@]+"${DIRTY_ARGS[@]}"}
    ;;
  log)
    echo "=== $OUT/run_metadata.json ==="
    if [[ -f "$OUT/run_metadata.json" ]]; then
      python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2)[:4000])" "$OUT/run_metadata.json"
    else
      echo "(missing)"
    fi
    echo "=== predictions tail ==="
    if [[ -f "$OUT/predictions.jsonl" ]]; then
      tail -n 5 "$OUT/predictions.jsonl"
    else
      echo "(missing)"
    fi
    ;;
  *)
    echo "usage: $0 dry-run|preflight|mock|index-status|build-index|run|resume|log" >&2
    exit 2
    ;;
esac
