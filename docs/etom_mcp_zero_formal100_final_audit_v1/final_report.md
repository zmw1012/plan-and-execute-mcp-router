# MCP-Zero × ETOM Formal-100 Final Audit v1

- Audit version: `etom_mcp_zero_formal100_final_audit_v1`
- Generated (UTC): `2026-11-08T03:27:00.984826+00:00`
- Raw aggregate SHA256: `41827cff3ad70a783dbc3e13b72f76b4acee2006ddaad5455a61de2f4ab351ee`
- Raw SHA unchanged after audit: `True`

## Official headline (strict denominator)

- Overall strict exact route: **49/100 = 0.49**
- Completed-only diagnostic: 49/84 = 0.5833 (must not replace strict)
- Completion: 84/100

| Level | Strict EM | Completed-only EM |
|---|---:|---:|
| L1 | 17/20 = 0.85 | 0.8500 |
| L2 | 10/20 = 0.50 | 0.5000 |
| L3 | 9/20 = 0.45 | 0.5000 |
| L4 | 2/20 = 0.10 | 0.1818 |
| L5 | 11/20 = 0.55 | 0.7333 |

### L4 separate metrics (strict /20)

- official_union_node_set_em: 1/20
- efs_covering_node_set: 3/20
- cross_server_dependency_correctness: 2/20
- complete_route_success: 2/20

## Identity

- Identity checks ok: `True`
- sample_sha256: `99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1`
- ordered_task_ids_sha256: `3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479`
- source_commit: `790f28727f567b88918420d163c524d46644a694`
- models: `deepseek-v4.1-flash` / `text-embedding-3-small`
- host: `api.zhizengzeng.com`
- top-k: server=3 tool=5
- catalog: 491 servers / 2375 tools
- embedding index: {'identity_sha256': '019cbd1ca4e69a6cb5c063ae54a6f851c8cbcf1b158b5a811ad9e8fadbb8ef1b', 'completed_units': 3357, 'expected_units': 3357, 'final_complete': True, 'cache_reused': True, 'build_calls': 0}

## Retrieval

- server_recall=0.775
- tool_recall=0.8625
- gold_in_candidates_tasks=69
- routed_gold_not_selected_tasks=37
- selected_not_retrieved_tasks=0

## Protocol / failures

- completed=84 failed=16
- all failures max_llm_turns: True
- parse_repairs=63 protocol_repairs=47

## Error taxonomy (exclusive primary)

- exact_route_success: 49
- server_retrieval_miss: 12
- tool_retrieval_miss: 1
- gold_retrieved_but_not_selected: 4
- incomplete_tool_chain: 14
- dependency_order_error: 0
- unnecessary_tool_or_l5_incorrect_rejection: 4
- protocol_parse_failure: 0
- max_turn_resource_limit: 16

## Cost

- LLM calls=451; tokens in/out/total=1277207/142939/1420146
- query embeddings=308/5868
- this_run_estimated_usd=0.554806
- retry_count=0; timeouts=0

## Operation-Visible descriptive comparison

- OV overall strict 66/100=0.66; completion 92/100
- MCP-Zero overall strict 49/100=0.49; completion 84/100
- L4 column aligned on complete_route_success (OV 0.25 vs MCP-Zero 0.10)
- No statistical significance claims.

## Notes

- Route exact match is not final-answer accuracy.
- Raw grouped_results.json pooled exact_match_rate is completed-only; corrected in this audit.
- No raw artifacts were modified; no APIs were called.
