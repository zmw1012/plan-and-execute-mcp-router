# HK-155 Flat All Tools Final Audit v1

Offline audit only. No LLM / Embedding / MCP / paid API. Raw predictions and run_metadata were not modified.

## Identity
- Condition / architecture: `flat` / `flat_all_tools`
- Provider / host / model: `openai_compatible` / `api.zhizengzeng.com` / `deepseek-v4.1-flash`
- Implementation commit: `9c4c904311d72c78ea169c1ac9e452966366ca66`
- system_prompt_hash: `fdecb85f16b506c6ed0e5e2c8b2c248929bd7342c68d11a2e6ac7f604ed1198c`
- N tasks: **155**
- human_grades_applied: `True`

## Status
- full / partial / failed: `{'full': 118, 'failed': 36, 'partial': 1}`
- explicit errors: `{'max_invocations': 34, 'max_llm_turns': 2}`

## Answer scoring
- auto_gradable: `154`
- auto correct: `87`
- auto accuracy (do NOT treat as final overall): `0.564935064935065`
- manual_required: `1` → `['fin_cf_022']`
- final answer accuracy: `88` / `155` = `0.567741935483871`
- overall_accuracy (= final answer accuracy when human closed): `0.567741935483871`
- conservative lower bound (auto-only correct / 155): `0.5612903225806452`
- upper bound if manual correct: `0.567741935483871`
- strict answer accuracy (failed/partial forced false): `85` / `155` = `0.5483870967741935`
- strict overall task success: `85` / `155` = `0.5483870967741935`

## Retrieval
- retrieval_metrics: `N/A` (not 1.0)
- embedding / find / describe calls: `0`

## Tool selection / execution
- exact tool-set: `0.23225806451612904`
- operation P/R/F1: `0.5116897225806452` / `0.729032264516129` / `0.5731674645161287`
- wrong-tool rate: `0.45605221290322584`
- all invocations successful rate: `0.5870967741935483`
- recovered after tool error rate: `0.16129032258064516`

## Domain final / auto accuracy
- cross_domain: final=14/25=0.56; auto=14/25 = 0.56; manual=0
- financial: final=19/60=0.31666666666666665; auto=18/59 = 0.3050847457627119; manual=1
- market: final=26/35=0.7428571428571429; auto=26/35 = 0.7428571428571429; manual=0
- news: final=29/35=0.8285714285714286; auto=29/35 = 0.8285714285714286; manual=0

## Difficulty final / auto accuracy
- D1: final=23/35=0.6571428571428571; auto=23/35 = 0.6571428571428571
- D2: final=16/24=0.6666666666666666; auto=15/23 = 0.6521739130434783
- D3: final=27/42=0.6428571428571429; auto=27/42 = 0.6428571428571429
- D4: final=12/28=0.42857142857142855; auto=12/28 = 0.42857142857142855
- D5: final=10/26=0.38461538461538464; auto=10/26 = 0.38461538461538464

## Answerability final / auto accuracy
- capability_gap: final=5/5=1.0; auto=5/5 = 1.0
- evidence_answerable: final=3/29=0.10344827586206896; auto=3/29 = 0.10344827586206896
- intentionally_infeasible: final=3/11=0.2727272727272727; auto=3/11 = 0.2727272727272727
- robustness: final=3/8=0.375; auto=3/8 = 0.375
- should_refuse_or_clarify: final=0/6=0.0; auto=0/6 = 0.0
- structured_answerable: final=74/96=0.7708333333333334; auto=73/95 = 0.7684210526315789

## Error taxonomy (primary)
- counts: `{'success': 85, 'incomplete_tool_chain': 3, 'resource_limit_max_invocations': 34, 'redundant_tool_selection': 7, 'answer_synthesis': 3, 'protocol_or_parse_failure': 1, 'refusal_or_feasibility_error': 19, 'resource_limit_max_llm_turns': 2, 'wrong_tool_selection': 1}`

## Resources
- tokens in/out/total: `9231332` / `232792` / `9464124`
- LLM calls / real MCP: `460` / `572`
- mean latency ms: `25185.31666451613`
- estimated_usd: `null`

## Partial task
- `{'task_id': 'xd_ht_002', 'status': 'partial', 'error': None, 'predicted_status': 'unavailable', 'final_answer': 'No tool calls and no answer.', 'answer_correct': False, 'answer_correct_strict': False, 'primary_error_category': 'protocol_or_parse_failure', 'selected_operations': ['hkexnews.lookup_stock_code', 'hkexnews.list_hkexnews_announcements', 'aktools.stock_hk_hist', 'hkexnews.download_hkexnews_announcement'], 'invoked_operations': ['hkexnews.lookup_stock_code', 'hkexnews.list_hkexnews_announcements', 'aktools.stock_hk_hist', 'hkexnews.download_hkexnews_announcement'], 'llm_calls': 3, 'real_mcp_execution_count': 6, 'treatment': 'strict incorrect; remains in denominator; no rerun; classified as protocol_or_parse_failure'}`

Human grades closed from `human_grades.json` (no task-id scoring hardcode).
