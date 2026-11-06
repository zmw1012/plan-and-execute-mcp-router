#!/usr/bin/env bash
# Operation-Visible ETOM-100 formal evaluation wrapper.
# Reads the frozen formal_100_v1.lock.json only. Never resamples.
# Never prints API keys or env-file values.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:-}"
ORIG="$(cd "$ROOT/.." && pwd)/hk-equity-mcp-agent"
ENV_FILE="${ENV_FILE:-$ORIG/.env.aliyun}"
EMBEDDING_ENV_FILE="${EMBEDDING_ENV_FILE:-$ORIG/.env.embedding-relay}"
OUT="${OUT:-$ROOT/outputs/etom_external_formal100_operation_visible_v1}"
LOG="$OUT/run.log"
PY="${PYTHON:-python3}"
MAX_TASKS="${MAX_TASKS:-100}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

require_generation_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Generation env file is missing. Set ENV_FILE." >&2
    exit 2
  fi
}

require_embedding_env() {
  if [[ ! -f "$EMBEDDING_ENV_FILE" ]]; then
    echo "Embedding env file is missing. Set EMBEDDING_ENV_FILE." >&2
    exit 2
  fi
}

pilot() {
  "$PY" "$ROOT/scripts/run_etom_operation_visible_formal_100.py" \
    --output-dir "$OUT" \
    --max-tasks "$MAX_TASKS" \
    "$@"
}

case "$MODE" in
  preflight|dry-run)
    require_generation_env
    require_embedding_env
    pilot --dry-run --env-file "$ENV_FILE" --embedding-env-file "$EMBEDDING_ENV_FILE"
    ;;
  mock)
    OUT="${MOCK_OUT:-$ROOT/outputs/etom_external_formal100_operation_visible_v1_mock}"
    mkdir -p "$OUT"
    pilot --mock
    ;;
  index-status)
    pilot --index-status
    ;;
  run)
    require_generation_env
    require_embedding_env
    # Requires complete final catalog index. Does not rebuild embeddings.
    # Do not start this from Cursor in this setup turn.
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_formal_100.py" \
      --env-file "$ENV_FILE" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks "$MAX_TASKS" \
      --live \
      --require-complete-index \
      2>&1 | tee -a "$LOG"
    ;;
  resume)
    require_generation_env
    require_embedding_env
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_formal_100.py" \
      --env-file "$ENV_FILE" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks "$MAX_TASKS" \
      --live \
      --require-complete-index \
      --resume \
      2>&1 | tee -a "$LOG"
    ;;
  log)
    touch "$LOG"
    tail -f "$LOG"
    ;;
  *)
    echo "Usage: $0 {preflight|mock|index-status|run|resume|log}" >&2
    exit 2
    ;;
esac
