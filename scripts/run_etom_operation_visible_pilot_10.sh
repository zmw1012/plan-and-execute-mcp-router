#!/usr/bin/env bash
# Operation-Visible adapted for ETOM routing-only evaluation.
# 10-task cost pilot only. Never prints API keys or env-file values.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:-}"
ORIG="$(cd "$ROOT/.." && pwd)/hk-equity-mcp-agent"
ENV_FILE="${ENV_FILE:-$ORIG/.env.aliyun}"
EMBEDDING_ENV_FILE="${EMBEDDING_ENV_FILE:-$ORIG/.env.embedding-relay}"
OUT="${OUT:-$ROOT/outputs/etom_external_pilot_v1}"
LOG="$OUT/run.log"
PY="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

if [[ ! -f "$ENV_FILE" || ! -f "$EMBEDDING_ENV_FILE" ]]; then
  echo "Generation or embedding env file is missing. Set ENV_FILE and EMBEDDING_ENV_FILE." >&2
  exit 2
fi

pilot() {
  "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
    --env-file "$ENV_FILE" \
    --embedding-env-file "$EMBEDDING_ENV_FILE" \
    --output-dir "$OUT" \
    --max-tasks 10 \
    "$@"
}

case "$MODE" in
  preflight|dry-run)
    pilot --dry-run
    ;;
  mock)
    OUT="${MOCK_OUT:-$ROOT/outputs/etom_external_pilot_v1_mock}"
    mkdir -p "$OUT"
    pilot --mock-llm
    ;;
  run)
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
      --env-file "$ENV_FILE" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks 10 \
      --live 2>&1 | tee -a "$LOG"
    ;;
  resume)
    caffeinate -dimsu "$PY" "$ROOT/scripts/run_etom_operation_visible_pilot.py" \
      --env-file "$ENV_FILE" \
      --embedding-env-file "$EMBEDDING_ENV_FILE" \
      --output-dir "$OUT" \
      --max-tasks 10 \
      --live --resume 2>&1 | tee -a "$LOG"
    ;;
  log)
    touch "$LOG"
    tail -f "$LOG"
    ;;
  *)
    echo "Usage: $0 {preflight|mock|run|resume|log}" >&2
    exit 2
    ;;
esac
