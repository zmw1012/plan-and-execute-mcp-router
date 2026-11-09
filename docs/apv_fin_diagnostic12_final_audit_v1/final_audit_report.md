# APV-Fin Diagnostic-12 Final Offline Audit v1

**Read-only descriptive audit. Not a re-run. Not population accuracy. No API calls.**

## Identity

- method_version: `apv_fin_v1_4_1_crash_safe_runtime`
- git_commit: `3411a28f1c05dad87007626e31dfa3f4ecadfcc4`
- run_status: `{'n_rows_12': True, 'unique_ids_12': True, 'ordered_ids_match_manifest': True, 'manifest_hash_match': True, 'ordered_id_hash_match': True, 'run_status_completed': True, 'processed_task_count_12': True, 'method_version': True, 'git_commit': True, 'git_dirty_false': True, 'no_catalog_index_rebuild': True, 'resume_no_duplicate_ids': True}` all_passed=`True`
- ordered IDs match manifest: `True`
- generation: `{'model': 'deepseek-v4.1-flash', 'provider': 'openai_compatible', 'host': 'api.zhizengzeng.com'}`
- embedding: `{'model': 'text-embedding-3-small', 'provider': 'openai_compatible', 'host': 'api.zhizengzeng.com'}`

## Strict metrics

- strict answer success: **0/12**
- descriptive auto-grade correct: **1/12**
- status full/failed: **1/11**
- resource-limit failures: **8/12**
- exact tool set: **3/12**
- mean operation F1: **0.44880950000000003**
- notes: apv_fin_metrics.descriptive_auto_grade_accuracy disagrees with prediction-level answer_correct (0.0 vs 0.08333333333333333); grading_summary.auto_graded_correct=1. Audit uses prediction answer_correct for descriptive.

## Error taxonomy (primary)

- `exact_success`: 0
- `planner_parse_or_schema`: 1
- `capability_or_tool_selection`: 0
- `incomplete_disclosure_chain`: 8
- `required_output_not_satisfied`: 0
- `evidence_conflict_or_verification`: 0
- `calculation_not_reached`: 0
- `final_schema_or_synthesis`: 2
- `resource_limit_exception_replan`: 0
- `incorrect_refusal`: 1
- `tool_execution_failure`: 0

## Mechanism funnel (counts / 12)

- `planner_valid`: 11
- `capability_compiled`: 11
- `compatible_candidate_found`: 10
- `operation_selected`: 10
- `mcp_invoked`: 10
- `mcp_successful`: 9
- `adapter_successful`: 6
- `required_outputs_satisfied`: 4
- `retrieve_steps_resolved`: 5
- `compute_executed`: 1
- `verify_resolved`: 2
- `final_contract_valid`: 1
- `strict_answer_correct`: 0

## Resources

- LLM calls total/mean/median: 60 / 5 / 4.5
- tokens in/out/total: 189386 / 67903 / 257289
- embedding query calls/tokens: 108 / 0
- MCP executions: 16
- latency mean/median/p95 ms: 62357.33125 / 62989.9905 / 94370.1827
- estimated_cost_usd: `None`

## Same-12 baseline descriptive comparison

| Condition | Strict/desc correct | Full | Failed | Exact tools | F1 mean | LLM mean | MCP mean | Resource-like |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| operation_visible | 2/12 | 9 | 3 | 3 | 0.5844878333333333 | 5.666666666666667 | 3.0 | 3 |
| mcp_zero | 2/12 | 8 | 4 | 2 | 0.49464291666666665 | 5.5 | 2.5833333333333335 | 4 |
| flat_all_tools | 2/12 | 7 | 5 | 1 | 0.5171296666666666 | 2.9166666666666665 | 4.666666666666667 | 5 |
| mcp_free | 1/12 | 12 | 0 | None | None | 1.0 | 0.0 | 0 |
| apv_fin | 0/12 | 1 | 11 | 3 | 0.44880950000000003 | 5.0 | 1.3333333333333333 | 8 |

Verification vs expected correct counts:

- `operation_visible`: expected 2, observed 2, match=True
- `mcp_zero`: expected 2, observed 2, match=True
- `flat_all_tools`: expected 2, observed 2, match=True
- `mcp_free`: expected 1, observed 1, match=True
- `apv_fin`: expected 0, observed 0, match=True

## Research conclusions (evidence-backed)

### 1. Capability-aware routing improved candidate/tool selection selectively

Compatible candidates and selections appeared on 10/12 and 10/12 tasks; exact tool-set reached 3/12 and mean F1=0.44880950000000003. Versus OV exact=3/12 and F1=0.5844878333333333, APV-Fin is not uniformly better, but several tasks show capability injection/list prefix (dependency expansions) that OV-style free selection lacked. This is descriptive only (n=12).

### 2. Better selection did not convert into answer correctness

Strict answer success is 0/12 while MCP success occurred on 9/12 and adapters on 6/12. Required outputs were satisfied on only 4/12; final contract valid 1/12. The drop is concentrated after retrieve: missing disclosure fields, unfinished chains, and final schema/synthesis failures.

### 3. Plan/Monitor/Verify/Final control cost is high relative to yield

Mean LLM calls=5 (median 4.5), total tokens=257289, monitor+repair=19+3. Strict yield=0/12. OV mean LLM calls=5.666666666666667 with 2/12 descriptive correct on the same IDs. APV-Fin spends more control tokens without converting them into correct answers on this diagnostic slice.

### 4. Disclosure dependency chains still do not complete

Primary incomplete_disclosure_chain=8; dependency expansions sum=9. Typical pattern: list/prefix executes, then keyword/page/narrative retrieve hits max_targeted_retrievals or missing required outputs, then exceptional Monitor replans until the replan cap. Producer chain starts more often than in v7/v8 engineering smokes, but still fails to finish download→parse→evidence.

### 5. Resource-limit is usually a downstream stop, not the first root cause

resource_limit error_type count=8/12, but primary taxonomy assigns resource_limit_exception_replan as primary only 0 times. Most capped tasks already failed retrieve/disclosure/required-output stages; the replan budget then expires. Treating all failures as 'budget too small' would mis-locate the defect.

### 6. Current APV-Fin behaves like an over-engineered controller on this slice

With planner/monitor/binding repairs and multi-step verify/final machinery, APV-Fin reaches 0/12 strict vs baselines' descriptive corrects (OV/MCP-Zero/Flat=2, MCP-Free=1) on the same 12 IDs. Control complexity and token spend rise while answer yield falls — consistent with over-engineering for this diagnostic regime, not a claim about all future tasks.

### 7. Simpler follow-on direction (not implemented)

Evidence suggests retaining capability-aware routing and evidence/result normalization, while removing multi-round Monitor/Replan and collapsing to a shallow retrieve→compute→verify pipeline. This audit does not implement that redesign.

## Integrity

- raw hashes before/after unchanged: `True`
- no API / no rerun / no raw mutation
