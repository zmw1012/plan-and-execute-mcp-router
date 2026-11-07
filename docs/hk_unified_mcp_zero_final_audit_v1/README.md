# HK-155 MCP-Zero Final Audit v1

Offline mirror of `outputs/hk_unified_mcp_zero_exploratory155_v2/final_audit_mcp_zero_v1/`.

Reproduce:

```bash
PYTHONPATH=src python3 scripts/run_hk_unified_mcp_zero_final_audit_v1.py \
  --run-dir outputs/hk_unified_mcp_zero_exploratory155_v2
```

Does not modify raw predictions. No LLM/Embedding/MCP calls.
Manual review: fill `human_grades.json` / `manual_review_queue.csv` for `fin_cf_022` only after independent human grading.
