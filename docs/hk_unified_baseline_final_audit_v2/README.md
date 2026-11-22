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

Human decisions are stored **only** in:

`manual_review_decisions_v1.csv`

This file is immutable input for the audit runner (never overwritten).
Do **not** write decisions back into `outputs/.../manual_grading.csv` or
`predictions.jsonl`.

```bash
PYTHONPATH=src python3 scripts/run_hk_unified_final_audit_v2.py \
  --run-dir outputs/hk_unified_exploratory155_operation_visible_v1 \
  --baseline-git-commit 39bd31b651deb1d064942ee5c6f5e9068946ac15 \
  --manual-decisions docs/hk_unified_baseline_final_audit_v2/manual_review_decisions_v1.csv
```

Final human-merged artifacts:

- `final_grading_summary_v2.json`
- `final_grouped_aggregates_v2.json`
- `final_per_task_audit_v2.csv`
- `final_baseline_error_analysis_v2.md`
- `final_report_ready_baseline_tables_v2.csv`

Headline metrics A/B/C are reported separately and must not be conflated.
