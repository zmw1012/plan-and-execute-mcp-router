# Future LIVE Diagnostic-20 (PE-Fin RePlan v1.2) — do not run until approved

**PE-Fin Lite v1.2 remains the frozen formal HK-155 result.**  
**PE-Fin Lite v1.2-EEA remains a post-hoc execution/evidence ablation.**  
**PE-Fin RePlan v1.2 is a separate post-hoc exploratory method** after the
invalidated live v1 engineering diagnostic.

## Invalidated prior run

`outputs/pe_fin_replan_diagnostic20_live_v1/` is frozen and marked
`INVALIDATED.md` — do not modify, resume, or use for accuracy.

**Before Diagnostic-20 live v2**, run live engineering Smoke-5 first
(`docs/pe_fin_replan_v1/FUTURE_LIVE_SMOKE5_COMMAND.md` →
`outputs/pe_fin_replan_smoke5_live_v2`).

## Identity

- `condition=pe_fin_replan`
- `method_version=pe_fin_replan_v1_2`
- `protocol_version=planner_executor_replanner_contract_v2`
- `replanner_schema_version=pe_fin_replan_v1_2`
- `max_execution_rounds=7` / `max_replan_rounds=7`
- `max_total_llm_calls_per_task=16` (hard ceiling; transport retries count)
- `max_total_mcp_invocations=8`
- `decision_reason` optional

## Exact command (when approved)

```bash
cd "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-lite"

PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all \
  --diagnostic20-replan \
  --replan \
  --confirm-retire-heldout \
  --allow-hkex-download \
  --env-file .env \
  --embedding-env-file .env \
  --require-complete-index \
  --output-dir outputs/pe_fin_replan_diagnostic20_live_v2
```

## Explicitly not this round

- No live Diagnostic-20 until user confirms
- No overwrite of invalidated v1
- No HK-155 rerun / ETOM / push / PR / merge
