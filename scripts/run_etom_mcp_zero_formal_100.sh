#!/usr/bin/env bash
# MCP-Zero × ETOM formal100 helper.
# This construction round: offline only. Do NOT run live paid APIs from Cursor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-dry-run}"
OUT="${OUT:-$ROOT/outputs/etom_external_formal100_mcp_zero_v1_offline}"
case "$MODE" in
  dry-run|preflight|mock|index-status)
    python3 "$ROOT/scripts/run_etom_mcp_zero.py" --split formal100 --"$MODE" --output-dir "$OUT" ${ETOM_FIXTURE_DIR:+--etom-fixture-dir "$ETOM_FIXTURE_DIR"} ${ETOM_SOURCE_DIR:+--etom-source-dir "$ETOM_SOURCE_DIR"}
    ;;
  build-index|run|resume)
    cat <<EOF
Future paid commands (NOT run in this round):
  # 1) index-status
  python3 scripts/run_etom_mcp_zero.py --split formal100 --index-status --require-complete-index --etom-source-dir "$ETOM_SOURCE_DIR"
  # 2) build-index (paid embeddings)
  python3 scripts/run_etom_mcp_zero.py --split formal100 --build-index-only --etom-source-dir "$ETOM_SOURCE_DIR"
  # 3) pilot-10 / formal-100 live with resume batches
  python3 scripts/run_etom_mcp_zero.py --split formal100 --live --require-complete-index --resume --max-tasks N
EOF
    exit 2
    ;;
  *)
    echo "usage: $0 dry-run|preflight|mock|index-status|build-index|run|resume" >&2
    exit 2
    ;;
esac
