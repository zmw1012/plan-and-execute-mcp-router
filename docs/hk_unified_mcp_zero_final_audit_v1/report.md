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
- final_answer_accuracy (auto)=`0.577922077922078`
- strict_answer_accuracy /155=`0.5741935483870968`
- strict_overall_task_success /155=`0.36774193548387096`
- manual_required_count=`1` (fin_cf_022 queue)

## Descriptive baseline comparison (not significance-tested)
- MCP-Zero strict answer accuracy ≈ `57.4194%` vs MCP-Free `12.9032%` (20/155) vs Flat `56.7742%` (88/155)
- MCP-Zero tokens=`4050513` vs Flat `9464124`; MCP execs=`332` vs Flat `572`
- MCP-Zero exact_tool_set≈`0.439` (Flat≈0.232); op F1≈`0.641` (Flat≈0.573)
- First-20 vs full-155 differences reflect locked-order sample composition, not model regression.

## Domain / difficulty (strict answer accuracy)
### domain
- `cross_domain`: n=25 auto_acc=0.52 strict=0.52
- `financial`: n=60 auto_acc=0.3898 strict=0.3833
- `market`: n=35 auto_acc=0.7714 strict=0.7714
- `news`: n=35 auto_acc=0.7429 strict=0.7429

### project_difficulty
- `D1`: n=35 auto_acc=0.8 strict=0.8
- `D2`: n=24 auto_acc=0.6522 strict=0.625
- `D3`: n=42 auto_acc=0.6667 strict=0.6667
- `D4`: n=28 auto_acc=0.4286 strict=0.4286
- `D5`: n=26 auto_acc=0.2308 strict=0.2308

### source_split
- `development`: n=70 auto_acc=0.5362 strict=0.5286
- `previously_reserved`: n=85 auto_acc=0.6118 strict=0.6118

### answerability_class
- `capability_gap`: n=5 auto_acc=1.0 strict=1.0
- `evidence_answerable`: n=29 auto_acc=0.2069 strict=0.2069
- `intentionally_infeasible`: n=11 auto_acc=0.0 strict=0.0
- `robustness`: n=8 auto_acc=0.125 strict=0.125
- `should_refuse_or_clarify`: n=6 auto_acc=0.0 strict=0.0
- `structured_answerable`: n=96 auto_acc=0.8105 strict=0.8021

### status
- `failed`: n=25 auto_acc=0.0 strict=0.0
- `full`: n=130 auto_acc=0.6899 strict=0.6846

## MCP-Zero error taxonomy (primary)
- `success`: 89
- `method_resource_limit_failure`: 22
- `incomplete_tool_chain`: 8
- `answer_synthesis_failure`: 3
- `retrieval_failure`: 10
- `mcp_tool_execution_failure`: 7
- `evaluation_manual_required`: 1
- `llm_provider_transport_failure`: 3
- `feasibility_refusal_failure`: 12

## Failed-task error breakdown (raw error field)
- `max_llm_turns`: 19
- `RuntimeError`: 3
- `timeout`: 2
- `max_invocations`: 1

## Manual review
- Only `fin_cf_022` is manual_required under grader v1.2.
- See `manual_review_queue.csv` / `human_grades.json` (human fields blank).
- Do **not** copy MCP-Free/Flat human grades for this condition.

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
