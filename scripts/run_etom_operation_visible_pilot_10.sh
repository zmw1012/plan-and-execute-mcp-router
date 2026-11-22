#!/usr/bin/env bash
# Operation-Visible adapted for ETOM routing-only evaluation.
# 10-task cost pilot only. Never prints API keys or env-file values.
#
# Stages:
#   preflight / mock / index-status  -> offline (no paid API calls)
#   build-index                      -> embedding API only (no DeepSeek)
#   run / resume                     -> require complete final index; 10 tasks
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:-}"
ORIG="$(cd "$ROOT/.." && pwd)/hk-equity-mcp-agent"
ENV_FILE="${ENV_FILE:-$ORIG/.env.aliyun}"
EMBEDDING_ENV_FILE="${EMBEDDING_ENV_FILE:-$ORIG/.env.embedding-relay}"
OUT="${OUT:-$ROOT/outputs/etom_external_pilot_v1}"
LOG="$OUT/run.log"
INDEX_LOG="$OUT/build_index.log"
PY="${PYTHON:-python3}"
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
  "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
    --output-dir "$OUT" \
    --max-tasks 10 \
    "$@"
}

pilot_with_envs() {
  pilot \
    --env-file "$ENV_FILE" \
    --embedding-env-file "$EMBEDDING_ENV_FILE" \
    "$@"
}

case "$MODE" in
  preflight|dry-run)
    require_generation_env
    require_embedding_env
    pilot_with_envs --dry-run
    ;;
  mock)
    OUT="${MOCK_OUT:-$ROOT/outputs/etom_external_pilot_v1_mock}"
    mkdir -p "$OUT"
    pilot --mock-llm
    ;;
  build-index)
    require_embedding_env
    # Embedding provider only. Do not load or call DeepSeek.
    # Resume automatically from the partial JSONL checkpoint.
    # Do not loop this command from Cursor; run once in Terminal and interrupt with Ctrl+C if needed.
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks 10 \
      --build-index-only \
      --allow-dirty \
      2>&1 | tee -a "$INDEX_LOG"
    ;;
  index-status)
    # Offline. No network. No API keys printed.
    pilot --index-status
    ;;
  run)
    require_generation_env
    require_embedding_env
    # Requires a complete final catalog index. Does not rebuild embeddings.
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
      --env-file "$ENV_FILE" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks 10 \
      --live \
      --require-complete-index \
      2>&1 | tee -a "$LOG"
    ;;
  resume)
    require_generation_env
    require_embedding_env
    # Resumes ETOM task predictions only. Incomplete embedding index is rejected.
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
      --env-file "$ENV_FILE" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks 10 \
      --live \
      --require-complete-index \
      --resume \
      2>&1 | tee -a "$LOG"
    ;;
  log)
    touch "$LOG"
    if [[ -f "$INDEX_LOG" ]]; then
      echo "=== build_index.log (tail) ==="
      tail -n 40 "$INDEX_LOG" || true
      echo "=== run.log (follow) ==="
    fi
    tail -f "$LOG"
    ;;
  *)
    echo "Usage: $0 {preflight|mock|build-index|index-status|run|resume|log}" >&2
    exit 2
    ;;
esac
