# PE-Fin RePlan ETOM Formal-100 — Report-ready summary

Audit: `pe_fin_replan_etom_formal100_final_audit_v1`
Run commit: `9e0fb8ee1874ee7ef84c006d1c361ff7f231293b`

## Headline (three lenses; do not collapse)

1. **All-task routing score**: Exact Match **45/100** (0.4500); P=0.6573 R=0.6711 F1=0.6429
2. **Failures-as-zero E2E (primary conservative)**: Exact Match **35/100** (0.3500); P=0.4548 R=0.4685 F1=0.4460
3. **Completed-only diagnostic (NOT overall)**: Exact Match **35/65** (0.5385); selection bias; exclude from headline accuracy.

## Completeness

- predictions=100; completed=65; failed=35; run_status=`completed`
- Controlled task failures do **not** make the run partial under finalization v2.

## Failures-as-zero per level (Exact Match)

- L1: 15/20 = 75.00% (completed=17, failed=3)
- L2: 10/20 = 50.00% (completed=19, failed=1)
- L3: 4/20 = 20.00% (completed=10, failed=10)
- L4: 0/20 = 0.00% (completed=5, failed=15)
- L5: 6/20 = 30.00% (completed=14, failed=6)

## Failure taxonomy (status=failed only)

- n_failed=35
- by_error_type={'replanner_grounding_failure': 24, None: 10, 'resource_limit_failure': 1}
- by_stopping_reason={'ungrounded_replanner_response': 24, 'max_total_llm_calls': 7, 'resource_limit_failure': 1, 'two_consecutive_no_progress_rounds': 3}

## Resources

- LLM calls=745; embedding calls=232; routing records=208
- tokens=1669390; latency≈4562.7s
- unique predicted servers=121; real_tool_executions=0
- estimated_cost_usd=None

## Output-directory incident

- Raw written to repository root: `/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-replan-etom`
- Expected: `/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-replan-etom/outputs/pe_fin_replan_etom_formal100_v1`
- Confirmed code defect: empty/`'.'` `--output-dir` resolves to repo root (future-run fix only).
- This audit does **not** require Formal-100 rerun.

## Interpretation

- Routing-only evaluation; not real MCP business execution.
- Post-hoc exploratory transfer; equal-level stratified Formal-100.
- `cross_condition_comparability_not_established` vs other ETOM conditions without frozen shared baseline tables.

