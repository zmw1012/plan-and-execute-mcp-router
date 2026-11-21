# HK-155 MCP-Zero Final Audit v1

Offline audit only. Raw predictions were not modified. No APIs called.

## Identity
- condition=`mcp_zero`
- provider/host/model=`openai_compatible` / `api.zhizengzeng.com` / `deepseek-v4.1-flash`
- embedding=`text-embedding-3-small`
- prompt hash=`0879f47079d8ba345cda167829fa3474375aefd647f92efc139014d7a2885ab8`
- method-card hash=`77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f`
- git_commit=`1c027c3405d7d70cbdbc82d270a1251969420405`
- tasks=`155` unique; domains={'financial': 60, 'market': 35, 'news': 35, 'cross_domain': 25}
- status={'full': 130, 'failed': 25}
- embedding index reused=`True` build_calls=`0`

## Auto / strict grading (v1.2)
- auto_gradable_count=`154`
- auto_graded_correct=`89`
- auto_graded_accuracy=`0.577922077922078`
- final_answer_correct / final_answer_accuracy=`90`/`0.5806451612903226`
- strict_answer_correct / strict_answer_accuracy=`90`/`0.5806451612903226`
- strict_overall_task_success_n / rate=`57`/`0.36774193548387096` (**condition-specific**; condition-specific)
- common_strict_answer_success_n / rate=`90`/`0.5806451612903226` (cross-baseline comparable)
- manual_required_count=`1` (fin_cf_022)
- human_grades_applied=`['fin_cf_022']`

## Metric formulas
- `final_answer_accuracy`: count(answer_correct IS True) / 155 after human grades for manual_required; failed forced incorrect; auto-only rate remains auto_graded_accuracy
- `strict_answer_accuracy`: answer_correct IS True over all 155 tasks; failed forced False; human grades applied for manual_required
- `strict_overall_task_success`: status == 'full' AND answer_correct IS True AND tool_selection_correct IS True AND all_tool_invocations_successful IS True; denominator = 155; failed/partial forced incorrect; does NOT require evidence_correct or terminal_contract_valid; human overall_task_success does NOT override this conjunction
- `common_strict_answer_success`: status == 'full' AND answer_correct IS True; failed/partial forced incorrect; denominator = 155; cross-baseline comparable answer+completion metric

## Cross-condition comparability vs Flat
- Flat final answer accuracy=`0.567742` (88/155); Flat reported strict answer/overall=`0.5483870967741935` (85/155)
- MCP-Zero `strict_overall_task_success` vs Flat: **condition-specific** — MCP-Zero strict_overall_task_success additionally requires tool_selection_correct and all_tool_invocations_successful; Flat's published 85/155 matches an answer-level rate, so the two must not share a comparison column.
- Use `common_strict_answer_success` for same-column cross-baseline comparison.

## Descriptive baseline comparison (not significance-tested)
- MCP-Zero strict answer accuracy ≈ `58.0645%` vs MCP-Free `12.9032%` (20/155) vs Flat `56.7742%` (88/155)
- MCP-Zero common_strict_answer_success ≈ `58.0645%` (do not put MCP-Zero strict_overall next to Flat 85/155)
- MCP-Zero tokens=`4050513` vs Flat `9464124`; MCP execs=`332` vs Flat `572`
- MCP-Zero exact_tool_set≈`0.439` (Flat≈0.232); op F1≈`0.641` (Flat≈0.573)
- First-20 vs full-155 differences reflect locked-order sample composition, not model regression.

## Domain / difficulty (strict answer accuracy)
### domain
- `cross_domain`: n=25 auto_acc=0.52 strict=0.52
- `financial`: n=60 auto_acc=0.3898 strict=0.4
- `market`: n=35 auto_acc=0.7714 strict=0.7714
- `news`: n=35 auto_acc=0.7429 strict=0.7429

### project_difficulty
- `D1`: n=35 auto_acc=0.8 strict=0.8
- `D2`: n=24 auto_acc=0.6522 strict=0.6667
- `D3`: n=42 auto_acc=0.6667 strict=0.6667
- `D4`: n=28 auto_acc=0.4286 strict=0.4286
- `D5`: n=26 auto_acc=0.2308 strict=0.2308

### source_split
- `development`: n=70 auto_acc=0.5362 strict=0.5429
- `previously_reserved`: n=85 auto_acc=0.6118 strict=0.6118

### answerability_class
- `capability_gap`: n=5 auto_acc=1.0 strict=1.0
- `evidence_answerable`: n=29 auto_acc=0.2069 strict=0.2069
- `intentionally_infeasible`: n=11 auto_acc=0.0 strict=0.0
- `robustness`: n=8 auto_acc=0.125 strict=0.125
- `should_refuse_or_clarify`: n=6 auto_acc=0.0 strict=0.0
- `structured_answerable`: n=96 auto_acc=0.8105 strict=0.8125

### status
- `failed`: n=25 auto_acc=0.0 strict=0.0
- `full`: n=130 auto_acc=0.6899 strict=0.6923

## MCP-Zero error taxonomy (primary)
- `success`: 90
- `method_resource_limit_failure`: 22
- `incomplete_tool_chain`: 8
- `answer_synthesis_failure`: 3
- `retrieval_failure`: 10
- `mcp_tool_execution_failure`: 7
- `llm_provider_transport_failure`: 3
- `feasibility_refusal_failure`: 12

## Failed-task error breakdown (raw error field)
- `max_llm_turns`: 19
- `RuntimeError`: 3
- `timeout`: 2
- `max_invocations`: 1

## Manual review / human grade provenance
- Only `fin_cf_022` is manual_required under grader v1.2.
- human_grades_source=`/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-mcp-zero/docs/hk_unified_mcp_zero_final_audit_v1/human_grades.json`
- tasks_applied=`['fin_cf_022']`
- sealed=`True`
- Independent MCP-Zero review; not copied from MCP-Free/Flat.

## Resources
- tokens in/out/total=`3817113`/`233400`/`4050513`
- llm_calls=`761`
- embedding query calls/tokens=`412`/`9275`
- index build_calls=`0` cache_reused=`True`
- MCP executions=`332`
- latency mean/median/p95 ms=`47350.03088387097`/`39619.269`/`90990.29`
- estimated_usd=`None` (No frozen provider price table in-repo; estimated_usd left null.)

## Raw artifact integrity
- pre/post SHA256 match: see `raw_artifact_fingerprints.json`
