# HK Unified Operation-Visible 155 — Final Audit v2

Offline audit of baseline commit `39bd31b651deb1d064942ee5c6f5e9068946ac15`
(`feature/unified-hk-operation-visible-runner-v1`).

## How to reproduce (offline)

```bash
PYTHONPATH=src python3 scripts/run_hk_unified_final_audit_v2.py \
  --run-dir outputs/hk_unified_exploratory155_operation_visible_v1 \
  --baseline-git-commit 39bd31b651deb1d064942ee5c6f5e9068946ac15
```

Writes to `outputs/.../final_audit_v2/` without modifying raw predictions.
No LLM / Embedding / MCP / paid API calls.

## Tracked copies

This directory mirrors the non-sensitive summary artifacts from
`outputs/hk_unified_exploratory155_operation_visible_v1/final_audit_v2/`
for version control (`outputs/` is gitignored).

## Manual review

Open `manual_review_queue.csv`. Fill only `human_confirmed_*` and `human_notes`.
Suggested fields are pre-filled by the auditor and are not authoritative.
