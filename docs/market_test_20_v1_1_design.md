# Market Held-out Test 20 v1.1 Design

- **File:** `benchmarks/heldout/market_test_20_v1_1.jsonl`
- **Flags:** `locked=true`, `frozen=true`, `dataset_split=heldout`, `benchmark_role=final_evaluation_only`
- **Supersedes:** `market_test_20_v1.jsonl` (`superseded_before_any_agent_run`; never agent-run)
- **Membership / numerical gold:** identical to v1

## Project difficulty distribution (not ETOM distribution)

| Level | Count |
| --- | --- |
| D1 | 4 |
| D2 | 4 |
| D3 | 4 |
| D4 | 4 |
| D5 | 4 |

## Robustness / ETOM notes for D5 items

| ID | robustness_subtype | exact ETOM L5? |
| --- | --- | --- |
| mkt_ht_017 | temporal_policy | no (spot tools exist) |
| mkt_ht_018 | capability_gap | yes (no HKEX official calendar) |
| mkt_ht_019 | temporal_policy | no (post-cutoff policy) |
| mkt_ht_020 | unavailable_provider | no (AKTools hist path exists) |

Project difficulty levels are balanced for analysis, whereas ETOM levels are assigned only when the original orchestration semantics are satisfied. Market-only tasks are not forced into ETOM L4 because the frozen Market catalog contains one executable MCP server. Cross-server ETOM L4 is evaluated later in the cross-domain benchmark.
