# PE-Fin Lite Smoke-5 live v3 final audit

Offline engineering audit. Raw predictions/metadata were not modified.

## Freeze checks
- unique predictions: 5
- full=2 partial=1 failed=2
- answer_correct=0
- LLM calls=19 tokens=41465 gateway/MCP=11
- run_status=completed git=ddae351eb5eb07a3ba4465f7ebcbbb339bb0362a
- predictions sha256: `512e9e59a68ab192b6c0818c25be7a60a0ab4ef0d73549852a889a74abc6122f`
- run_metadata sha256: `520cd970c783de85c50184767b1342e1a4e487adaa36ed478d737fac6d68e83a`

## Validity classification

| task_id | status | validity | taxonomy |
|---|---|---|---|
| fin_cf_003 | failed | disclosure_engineering_failure | disclosure_local_key_not_passed_to_parse |
| fin_cf_023 | partial | method_outcome | method_metric_semantics_mismatch |
| xd_dev_002 | full | method_outcome | method_data_semantics_eps_mismatch |
| mkt_dev_014 | full | method_outcome | method_feasibility_temporal_policy |
| news_ht_004 | failed | disclosure_engineering_failure | disclosure_shortest_prefix_not_stopped |

## Method outcomes (do not retune ranking)
- `fin_cf_023`: financial path completed; metric/semantics mismatch vs gold.
- `xd_dev_002`: financial+market completed; EPS semantics mismatch.
- `mkt_dev_014`: hist price retrieved on temporal-policy question (feasibility failure).

## Disclosure engineering failures (not pure method failures)
- `fin_cf_003`: list/download succeeded; `local_key` not extracted from real list envelope → parse missing `local_key`.
- `news_ht_004`: list succeeded and should have satisfied list-level required outputs; normalization miss caused unnecessary download failure.

## Mechanism funnel
- planner_valid: 5
- retrieve_step_completed_mean: 0.5
- required_output_satisfied_mean: 0.5
- final_contract_valid: 5
- strict_answer_correct: 0

## Note
v3 remains a valid engineering smoke record after separating Disclosure defects from method outcomes.
