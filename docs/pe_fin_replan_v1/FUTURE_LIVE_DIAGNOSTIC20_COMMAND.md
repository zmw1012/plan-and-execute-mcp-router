# Future LIVE Diagnostic-20 (PE-Fin RePlan v1.1) — do not run until approved

**PE-Fin Lite v1.2 remains the frozen formal HK-155 result.**  
**PE-Fin Lite v1.2-EEA remains a post-hoc execution/evidence ablation.**  
**PE-Fin RePlan v1.1 is a separate post-hoc exploratory method**  
(`Planner → Execute-one-step → Replanner × ≤7 → Response`).

Identity:
- `condition=pe_fin_replan`
- `method_version=pe_fin_replan_v1_1`
- `protocol_version=planner_executor_replanner_seven_round_v1`
- `replanner_schema_version=pe_fin_replan_v1_1`
- `max_execution_rounds=7` / `max_replan_rounds=7`
- `max_total_llm_calls_per_task=16` (hard ceiling)
- `max_total_mcp_invocations=8`

Diagnostic-20 is **not** formal accuracy and **cannot** replace HK-155.

## Prerequisites

- Clean git tree (or explicit `--allow-dirty` only if intentionally used)
- Complete shared embedding index (same identity as formal R5 run)
- Valid generation + embedding env files
- Fresh output dir (do **not** resume EEA-only or RePlan v1 outputs)

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
  --output-dir outputs/pe_fin_replan_diagnostic20_live_v1
```

## Explicitly not this round

- No live Diagnostic-20 until user confirms
- No HK-155 rerun
- No ETOM
- No push / PR / merge
