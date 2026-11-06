# HK Unified Operation-Visible Baseline — Final Audit v2

## Experimental identity
- Baseline git commit: `39bd31b651deb1d064942ee5c6f5e9068946ac15`
- Audit version: `final_audit_v2`
- Evaluator: `hk_unified_evaluator_v1.2` / grader `deterministic_v1.2`
- Model: `deepseek-v4.1-flash`
- Prompt hash: `9674844a98c9f438ae02d19572f173a7cdc27e1c365ed27dd39ea7e0963eec1b`
- Catalog fingerprint: `unique=42|runtime=39|namespaced=54|cross_domain:a21a0cdf6c73|financial:18634ccfb733|market:2b201872c109|news:9121e9b7ca8d`
- Provider segment: tasks 1–65: Aliyun token-plan host; tasks 66–155: api.zhizengzeng.com relay; requested model id is deepseek-v4.1-flash for both. Difficulty mix differs by segment, so accuracy gaps must NOT be interpreted as a provider effect.

## Dataset composition
- N tasks: **155** (exploratory_all development lock)
- Domain counts: `{"cross_domain": 25, "financial": 60, "market": 35, "news": 35}`
- Difficulty counts: `{"D1": 35, "D2": 24, "D3": 42, "D4": 28, "D5": 26}`
- Answerability counts: `{"capability_gap": 5, "evidence_answerable": 29, "intentionally_infeasible": 11, "robustness": 8, "should_refuse_or_clarify": 6, "structured_answerable": 96}`

## Overall metrics
- Auto-gradable: 154 / 155; auto-graded accuracy: **0.6103896103896104** (94 correct)
- Manual required (answer): **1**
- Exact tool-set rate: **0.4129032258064516**
- Mean operation P/R/F1: 0.6547311999999998 / 0.7274193677419354 / 0.6693134580645159
- Tool execution success rate: **0.7677419354838709**
- Recovered-after-tool-error rate: 0.12258064516129032
- Mean dependency accuracy: None
- Refusal accuracy (auto subset): 0.2727272727272727
- Mean latency_ms / llm_calls / total_tokens / mcp_exec: 37629.65738064515 / 5.380645161290323 / 64256.49677419355 / 2.554838709677419

## Metrics by domain
- **cross_domain**: n=25, auto_acc=0.64, exact_tool=0.28, tool_ok=0.8
- **financial**: n=60, auto_acc=0.4576271186440678, exact_tool=0.13333333333333333, tool_ok=0.7
- **market**: n=35, auto_acc=0.7428571428571429, exact_tool=0.8571428571428571, tool_ok=0.9142857142857143
- **news**: n=35, auto_acc=0.7142857142857143, exact_tool=0.5428571428571428, tool_ok=0.7142857142857143

## Metrics by D1–D5
- **D1**: n=35, auto_acc=0.7714285714285715, exact_tool=0.5428571428571428
- **D2**: n=24, auto_acc=0.7391304347826086, exact_tool=0.4583333333333333
- **D3**: n=42, auto_acc=0.6904761904761905, exact_tool=0.40476190476190477
- **D4**: n=28, auto_acc=0.5, exact_tool=0.35714285714285715
- **D5**: n=26, auto_acc=0.2692307692307692, exact_tool=0.2692307692307692

## Metrics by answerability class
- **capability_gap**: n=5, auto_acc=0.6, refusal_acc=0.6
- **evidence_answerable**: n=29, auto_acc=0.27586206896551724, refusal_acc=0.0
- **intentionally_infeasible**: n=11, auto_acc=0.18181818181818182, refusal_acc=0.5454545454545454
- **robustness**: n=8, auto_acc=0.375, refusal_acc=0.5
- **should_refuse_or_clarify**: n=6, auto_acc=0.0, refusal_acc=0.3333333333333333
- **structured_answerable**: n=96, auto_acc=0.8210526315789474, refusal_acc=0.0

## Retrieval Recall@K
- Applicable tasks with gold ops: 133
- Recall@K: `{"R@1": 0.6992481203007519, "R@10": 0.849624060150376, "R@20": 0.9548872180451128, "R@5": 0.8195488721804511}`
- Agent selection given retrieval hit: `{"exact_tool_set_rate": 0.4566929133858268, "gold_subset_rate": 0.7007874015748031, "n": 127}`

## Agent tool-set exact match / precision / recall / F1
- exact_tool_set_rate=0.4129032258064516; P=0.6547311999999998; R=0.7274193677419354; F1=0.6693134580645159

## Dependency accuracy
- mean graph_dependency_accuracy=None

## Tool execution success
- all_tool_invocations_successful_rate=0.7677419354838709; recovered_after_tool_error_rate=0.12258064516129032

## Final answer accuracy
- v1.2 auto-graded accuracy=0.6103896103896104 (v1.1 was 0.3877551020408163)
- Overall accuracy still pending manual confirmation for evidence/overall fields.

## Refusal accuracy
- 0.2727272727272727

## Token / latency / tool-call efficiency
- mean_total_tokens=64256.49677419355, mean_latency_ms=37629.65738064515, mean_llm_calls=5.380645161290323, mean_real_mcp_execution_count=2.554838709677419

## Failure taxonomy
- Taxonomy version: `error_taxonomy_v1`
- Primary counts: `{"agent_selection": 8, "answer_synthesis_calculation": 6, "dependency_order": 0, "description_argument_grounding": 0, "evaluation_failure": 1, "feasibility_refusal": 22, "resource_limit": 23, "retrieval": 1, "success": 94, "tool_execution": 0}`
- Primary rates: `{"agent_selection": 0.05161290322580645, "answer_synthesis_calculation": 0.03870967741935484, "dependency_order": 0.0, "description_argument_grounding": 0.0, "evaluation_failure": 0.0064516129032258064, "feasibility_refusal": 0.14193548387096774, "resource_limit": 0.14838709677419354, "retrieval": 0.0064516129032258064, "success": 0.6064516129032258, "tool_execution": 0.0}`

## Representative success cases
- `fin_ht_022` (financial/D1): exact_tool=False
- `mkt_ht_002` (market/D1): exact_tool=True
- `news_ht_006` (news/D1): exact_tool=True
- `fin_ht_021` (financial/D1): exact_tool=False
- `mkt_dev_003` (market/D1): exact_tool=True
- `news_dev_003` (news/D1): exact_tool=False
- `xd_ht_004` (cross_domain/D2): exact_tool=True
- `fin_cf_018` (financial/D1): exact_tool=False

## Representative failure cases
- `xd_ht_007` (cross_domain/D2): primary=resource_limit, error=max_search_reformulations
- `fin_cf_019` (financial/D1): primary=answer_synthesis_calculation, error=n/a
- `fin_cf_011` (financial/D1): primary=agent_selection, error=n/a
- `xd_dev_006` (cross_domain/D3): primary=answer_synthesis_calculation, error=n/a
- `fin_ht_011` (financial/D1): primary=resource_limit, error=max_llm_turns
- `fin_cf_028` (financial/D1): primary=resource_limit, error=max_llm_turns
- `news_ht_005` (news/D1): primary=answer_synthesis_calculation, error=n/a
- `fin_cf_010` (financial/D1): primary=resource_limit, error=max_llm_turns

## Operation-Visible strengths
- Visible find→describe→invoke loop makes retrieval and tool-set metrics auditable.
- Structured refusal channel exists and is measurable via refusal_correctness.
- Non-trivial retrieval recall and tool execution success (R@top_k and tool_ok in artifacts).

## Operation-Visible weaknesses
- Resource-limit failures (max_llm_turns / reformulations / invocations) truncate answerable tasks.
- Feasibility/refusal leakage: agents often answer with substitutes under capability gaps.
- Even with gold ops retrieved, agent_selection and answer_synthesis remain major residual errors.

## Concrete design requirements for the Proposed Method
Requirements below are emitted only when taxonomy rates support them. No task-id hard rules.
- **Explicit evidence-chain planning** (from `agent_selection` rate=0.05161290322580645): Gold ops are often retrieved but not selected as a complete set; planning over evidence chains is supported.
- **Calculation/synthesis verification** (from `answer_synthesis_calculation` rate=0.03870967741935484): Tool results succeed but final numeric/unit synthesis fails.
- **Feasibility and temporal-policy gate** (from `feasibility_refusal` rate=0.14193548387096774): Refusal tasks are answered with substitutes or live/cutoff violations.
- **Structured evidence compression** (from `resource_limit` rate=0.14838709677419354): max_llm_turns / reformulations / invocations dominate hard failures; compression/dedup can reduce budget burn.
- **Operation deduplication** (from `resource_limit` rate=0.14838709677419354): Repeated find/describe/invoke under budget caps is a supported weakness.
- **Refusal validation** (from `feasibility_refusal` rate=0.14193548387096774): Capability-gap / blocked-provider items need post-answer refusal validation.

## Limitations
- Exploratory dataset / development lock — not an unseen final test set.
- Provider transition after task 65; do not attribute segment gaps to provider effect.
- Automatic grading limitations remain (evidence/overall still human).
- Manual review remaining (see `manual_review_queue.csv`).
- ETOM external evaluation not yet run.

## v1.1 → v1.2 grading delta
- `{"change_explanations": [{"bucket": "semantic_and_currency_units", "explanation": "v1.1 required literal unit tokens (e.g. 'percent', 'cny'+'million'); v1.2 accepts % / RMB↔CNY aliases and semantic units such as decimal_return."}, {"bucket": "unicode_minus_and_scale", "explanation": "v1.2 normalizes Unicode minus and matches percent↔decimal and million↔raw/billion scale forms with scaled tolerances."}, {"bucket": "labeled_multi_value", "explanation": "OHLC / unadjusted-qfq-hfq / MA+return dict gold_values are now auto-gradable via label binding instead of manual_required."}, {"bucket": "refusal_substitute", "explanation": "Refusal items that answer with numeric substitutes remain incorrect; status-only refusals without fabricated targets remain correct."}], "flips_false_to_true": [{"reason_v1_2": "numeric_match", "task_id": "fin_ht_021"}, {"reason_v1_2": "numeric_match;numeric_match", "task_id": "xd_ht_004"}, {"reason_v1_2": "numeric_match", "task_id": "fin_cf_018"}, {"reason_v1_2": "numeric_match;string_match;string_match", "task_id": "xd_ht_008"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_001"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_002"}, {"reason_v1_2": "numeric_match;numeric_match", "task_id": "xd_dev_001"}, {"reason_v1_2": "numeric_match", "task_id": "fin_cf_026"}, {"reason_v1_2": "numeric_match;numeric_match", "task_id": "xd_ht_001"}, {"reason_v1_2": "numeric_match", "task_id": "fin_cf_001"}, {"reason_v1_2": "numeric_match;string_match", "task_id": "xd_dev_004"}, {"reason_v1_2": "numeric_match", "task_id": "fin_cf_002"}, {"reason_v1_2": "numeric_match", "task_id": "fin_cf_004"}, {"reason_v1_2": "numeric_match;string_match", "task_id": "xd_ht_005"}, {"reason_v1_2": "numeric_match", "task_id": "mkt_ht_011"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_024"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_013"}, {"reason_v1_2": "numeric_match", "task_id": "mkt_dev_008"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_014"}, {"reason_v1_2": "numeric_match", "task_id": "mkt_ht_015"}, {"reason_v1_2": "numeric_match;string_match;numeric_match", "task_id": "xd_dev_009"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_004"}, {"reason_v1_2": "numeric_match", "task_id": "mkt_ht_014"}, {"reason_v1_2": "numeric_match", "task_id": "mkt_dev_013"}, {"reason_v1_2": "numeric_match;numeric_match", "task_id": "fin_ht_025"}, {"reason_v1_2": "numeric_match;numeric_match", "task_id": "fin_ht_026"}, {"reason_v1_2": "numeric_match", "task_id": "fin_cf_007"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_005"}, {"reason_v1_2": "numeric_match", "task_id": "fin_ht_006"}, {"reason_v1_2": "numeric_match;numeric_match", "task_id": "fin_ht_007"}], "flips_false_to_true_count": 30, "flips_true_to_false": [], "flips_true_to_false_count": 0, "newly_auto_gradable_from_manual": [{"answer_correct_v1_2": true, "reason": "labeled_multi_match:open:match,high:match,low:match,close:match", "task_id": "mkt_dev_003"}, {"answer_correct_v1_2": true, "reason": "structured_part_match;numeric_match", "task_id": "xd_ht_003"}, {"answer_correct_v1_2": true, "reason": "labeled_multi_match:open:match,high:match,low:match,close:match", "task_id": "mkt_ht_004"}, {"answer_correct_v1_2": true, "reason": "labeled_multi_match:unadjusted:match,qfq:match", "task_id": "mkt_dev_007"}, {"answer_correct_v1_2": true, "reason": "labeled_multi_match:unadjusted:match,hfq:match", "task_id": "mkt_ht_006"}, {"answer_correct_v1_2": true, "reason": "identity_match", "task_id": "mkt_ht_008"}, {"answer_correct_v1_2": true, "reason": "labeled_multi_match:ma5:match,simple_return:match", "task_id": "mkt_ht_016"}], "still_manual_required": ["fin_cf_022"], "v1_1_auto_gradable_count": 147, "v1_1_auto_graded_accuracy": 0.3877551020408163, "v1_1_auto_graded_correct": 57, "v1_1_manual_required_count": 8, "v1_2_auto_gradable_count": 154, "v1_2_auto_graded_accuracy": 0.6103896103896104, "v1_2_auto_graded_correct": 94, "v1_2_manual_required_count": 1}`

## Provenance / offline guarantees
- No LLM / Embedding / MCP / paid API calls during this audit.
- Raw predictions.jsonl, gold, prompt, and catalog were not modified.
- Raw SHA256: `{"manual_grading.csv": "e46f04c708dc040e9b61bc5c0f99b706596ff445d99d64568bf4a40902894d25", "per_task_metrics.csv": "783a9cd4c7ef567dbc658445a0fa9db3673a9d4a02c8ad546f96cf976b9c80e9", "predictions.jsonl": "e273d25a41470ca1fcdaa979469a2c18de1c0bf5eade46e969e33c42526cb50c", "retrieval_analysis.json": "aee1da3519d9056aacd96a2cc60fc54deab61d1ba5663a43e57177deed99e7f1", "run_metadata.json": "c164c3af206b072ba42d76b808aa076ec4b3868ebe33099a8576209b6cfaf4b8"}`

