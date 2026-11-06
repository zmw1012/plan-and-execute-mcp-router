# HK Unified Operation-Visible Baseline — Final Audit v2 (Human-Merged)

## Experimental identity
- Baseline git commit: `39bd31b651deb1d064942ee5c6f5e9068946ac15`
- Manual decisions: `/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-hk-unified/docs/hk_unified_baseline_final_audit_v2/manual_review_decisions_v1.csv`
- Manual decisions SHA256: `4291e86b8ea3a3077a56e28ad1898d27c89e007abe3676bb0cbd6712ea63dafc`
- Evaluator/grader: `hk_unified_evaluator_v1.2` / `deterministic_v1.2`
- Model: `deepseek-v4.1-flash`
- Provider segment: tasks 1–65: Aliyun token-plan host; tasks 66–155: api.zhizengzeng.com relay; requested model id is deepseek-v4.1-flash for both. Difficulty mix differs by segment, so accuracy gaps must NOT be interpreted as a provider effect.

## Separated headline metrics (do not conflate)
- **A. Final answer accuracy**: 95/155 = **0.6129032258064516**
- **B. Exact tool-set success**: 64/155 = **0.4129032258064516**
- **C. Strict overall task success**: 45/155 = **0.2903225806451613**

### Metric definitions
- `A_final_answer_accuracy`: Share of tasks with correct final answer (auto v1.2 or human decision). Independent of tool-set.
- `B_exact_tool_set_success`: Share of tasks where predicted invoke set exactly equals gold set.
- `C_strict_overall_task_success`: answer correct AND exact tool set AND all invocations successful AND dependency OK when applicable AND refusal OK when applicable AND evidence correct or N/A.

## Metrics by domain (final answer / exact tool / strict)
- **cross_domain**: n=25, answer_acc=0.64, exact_tool=0.28, strict=0.2
- **financial**: n=60, answer_acc=0.4666666666666667, exact_tool=0.13333333333333333, strict=0.08333333333333333
- **market**: n=35, answer_acc=0.7428571428571429, exact_tool=0.8571428571428571, strict=0.6571428571428571
- **news**: n=35, answer_acc=0.7142857142857143, exact_tool=0.5428571428571428, strict=0.34285714285714286

## Metrics by D1–D5
- **D1**: n=35, answer_acc=0.7714285714285715, exact_tool=0.5428571428571428, strict=0.45714285714285713
- **D2**: n=24, answer_acc=0.75, exact_tool=0.4583333333333333, strict=0.4583333333333333
- **D3**: n=42, answer_acc=0.6904761904761905, exact_tool=0.40476190476190477, strict=0.2619047619047619
- **D4**: n=28, answer_acc=0.5, exact_tool=0.35714285714285715, strict=0.17857142857142858
- **D5**: n=26, answer_acc=0.2692307692307692, exact_tool=0.2692307692307692, strict=0.07692307692307693

## Metrics by answerability class
- **capability_gap**: n=5, answer_acc=0.6, strict=0.4
- **evidence_answerable**: n=29, answer_acc=0.27586206896551724, strict=0.0
- **intentionally_infeasible**: n=11, answer_acc=0.18181818181818182, strict=0.0
- **robustness**: n=8, answer_acc=0.375, strict=0.0
- **should_refuse_or_clarify**: n=6, answer_acc=0.0, strict=0.0
- **structured_answerable**: n=96, answer_acc=0.8229166666666666, strict=0.4479166666666667

## Retrieval / tools / refusal (process)
- Recall@K: `{"R@1": 0.6992481203007519, "R@10": 0.849624060150376, "R@20": 0.9548872180451128, "R@5": 0.8195488721804511}`
- Mean operation P/R/F1: 0.6547311999999998 / 0.7274193677419354 / 0.6693134580645159
- Tool execution success rate: 0.7677419354838709
- Refusal accuracy (auto subset): 0.2727272727272727

## Failure taxonomy (answer_success ≠ strict_task_success)
- Taxonomy version: `error_taxonomy_v1.1`
- Primary counts: `{"agent_selection": 8, "answer_success": 50, "answer_synthesis_calculation": 6, "dependency_order": 0, "description_argument_grounding": 0, "evaluation_failure": 0, "feasibility_refusal": 22, "resource_limit": 23, "retrieval": 1, "strict_task_success": 45, "tool_execution": 0}`
- Primary rates: `{"agent_selection": 0.05161290322580645, "answer_success": 0.3225806451612903, "answer_synthesis_calculation": 0.03870967741935484, "dependency_order": 0.0, "description_argument_grounding": 0.0, "evaluation_failure": 0.0, "feasibility_refusal": 0.14193548387096774, "resource_limit": 0.14838709677419354, "retrieval": 0.0064516129032258064, "strict_task_success": 0.2903225806451613, "tool_execution": 0.0}`

## Manual review merge
- Queue size: 8; decisions applied: 8
- Human decisions are authoritative for queued tasks; `suggested_*` fields are never used as substitutes.
- `evaluation_failure` is not treated as an Agent error after human confirmation.

## Representative answer_success (not strict)
- `fin_ht_022` (financial/D1): failed=['exact_tool_set']
- `fin_ht_021` (financial/D1): failed=['exact_tool_set']
- `news_dev_003` (news/D1): failed=['exact_tool_set']
- `fin_cf_018` (financial/D1): failed=['exact_tool_set']
- `xd_ht_003` (cross_domain/D3): failed=['exact_tool_set']
- `news_dev_005` (news/D1): failed=['exact_tool_set']
- `xd_ht_008` (cross_domain/D3): failed=['exact_tool_set']
- `news_ht_002` (news/D1): failed=['exact_tool_set']
- `xd_dev_003` (cross_domain/D3): failed=['exact_tool_set']
- `news_dev_004` (news/D1): failed=['all_tool_invocations_successful']

## Representative strict_task_success
- `mkt_ht_002` (market/D1)
- `news_ht_006` (news/D1)
- `mkt_dev_003` (market/D1)
- `xd_ht_004` (cross_domain/D2)
- `mkt_dev_002` (market/D1)
- `news_ht_003` (news/D1)
- `mkt_ht_001` (market/D1)
- `mkt_ht_003` (market/D1)
- `news_dev_002` (news/D1)
- `mkt_dev_004` (market/D1)

## Proposed Method requirements (taxonomy-supported)
- **Explicit evidence-chain planning** (`agent_selection` rate=0.05161290322580645): Gold ops are often retrieved but not selected as a complete set; planning over evidence chains is supported.
- **Calculation/synthesis verification** (`answer_synthesis_calculation` rate=0.03870967741935484): Tool results succeed but final numeric/unit synthesis fails.
- **Feasibility and temporal-policy gate** (`feasibility_refusal` rate=0.14193548387096774): Refusal tasks are answered with substitutes or live/cutoff violations.
- **Structured evidence compression** (`resource_limit` rate=0.14838709677419354): max_llm_turns / reformulations / invocations dominate hard failures; compression/dedup can reduce budget burn.
- **Operation deduplication** (`resource_limit` rate=0.14838709677419354): Repeated find/describe/invoke under budget caps is a supported weakness.
- **Refusal validation** (`feasibility_refusal` rate=0.14193548387096774): Capability-gap / blocked-provider items need post-answer refusal validation.

## Limitations
- Exploratory dataset / development lock — not an unseen final test set.
- Provider transition after task 65; do not attribute segment gaps to provider effect.
- Automatic grading limitations remain for evidence on non-queued tasks (treated N/A in strict rubric unless human-set).
- ETOM external evaluation not yet run.

## Provenance / offline guarantees
- No LLM / Embedding / MCP / paid API calls during this audit.
- Raw predictions.jsonl, gold, prompt, catalog, and raw manual_grading.csv were not modified.
- Manual decision file is read-only input and is never overwritten by the audit runner.
- Raw SHA256: `{"manual_grading.csv": "e46f04c708dc040e9b61bc5c0f99b706596ff445d99d64568bf4a40902894d25", "per_task_metrics.csv": "783a9cd4c7ef567dbc658445a0fa9db3673a9d4a02c8ad546f96cf976b9c80e9", "predictions.jsonl": "e273d25a41470ca1fcdaa979469a2c18de1c0bf5eade46e969e33c42526cb50c", "retrieval_analysis.json": "aee1da3519d9056aacd96a2cc60fc54deab61d1ba5663a43e57177deed99e7f1", "run_metadata.json": "c164c3af206b072ba42d76b808aa076ec4b3868ebe33099a8576209b6cfaf4b8"}`

