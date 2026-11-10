# PE-Fin Lite Diagnostic-12 Checkpoint Live v3 — Final Freeze

## Verdict

**engineering checkpoint diagnostic only**  
`explicitly_not_formal_accuracy=true`  
`engineering_pass=true`  
`raw_hashes_unchanged=true`  
`zero_network audit=true`

This freeze is the final Diagnostic-12 engineering mirror for proposed method **`pe_fin_lite_v1.2`**.  
It is **not** formal accuracy, not significance testing, and not a license to create v1.3 / new controllers from D12 residuals.

## Frozen identity

| Field | Value |
|---|---|
| rows / unique IDs | 12 / 12 |
| run_status | completed |
| method_version | `pe_fin_lite_v1.2` |
| git_commit | `baf572461dce670fddb71291128e82360dbef2f0` |
| git_dirty | false |
| architecture | `plan_first_step_level_retrieval_v1` |
| Monitor / global replan | 0 / 0 |
| TopK / Describe / fallback | 5 / 2–3 / ≤1 |
| LLM≤8 / MCP≤8 | yes |
| embedding_index_identity | `ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df` |
| system_prompt_hash | `b347eed8c28d75d27ba92c1500da3b5393cee68ce5d24f36954b3fcabf4d1380` |
| schema_hash | `fdf728361fd74c3d9e6c5807b35a51ac8f87123cc0e3b3e8ea6139627da42d68` |
| disclosure_workflow_hash | `c3145b45afc9fa174e9b2c9968b63a1fd59498769625f9935c2b00ac4cb4a4a9` |

## Frozen results

| Metric | Value |
|---|---|
| full / partial / failed | **5 / 5 / 2** |
| answer_correct | **3/12** |
| LLM calls | 42 |
| tokens | 95419 |
| MCP calls | 33 |
| embedding queries | 10 |

Full IDs: `fin_cf_023`, `xd_dev_002`, `xd_ht_007`, `mkt_dev_001`, `news_ht_004`  
Partial IDs: `fin_cf_003`, `fin_cf_011`, `xd_ht_014`, `mkt_dev_014`, `fin_cf_024`  
Failed IDs: `fin_cf_020`, `fin_ht_016`

## Checkpoint funnel (headline = stage_outcome_status)

| Checkpoint | Stage outcome |
|---|---|
| C01 Planner | 12 pass |
| C02 Filter quality | not_evaluated in D12 (11 not_evaluated / 1 N/A) |
| C03 Embedding recall | not_evaluated in D12 (11 not_evaluated / 1 N/A) |
| C04 Describe | 11 pass / 1 N/A |
| C05 Binder | 11 pass / 1 N/A |
| C06 Arguments | 11 pass / 1 N/A |
| C07 MCP | 10 pass / 1 fail / 1 N/A |
| C08 Normalize | 11 pass / 1 N/A |
| C09 Evidence | 6 pass / 5 fail / 1 N/A |
| C10 Fallback | 11 pass / 1 N/A |
| C11 Compute | 1 pass / 10 N/A_not_planned / 1 N/A_upstream_blocked |
| C12 Reason | 4 pass / 1 fail / 7 N/A |
| C13 Final contract | 12 pass |
| C14 Evaluator execution | 12 pass |

## Primary blockers

| Primary | Count |
|---|---|
| C09 evidence acceptance | 4 |
| C12 reason | 1 |
| C07 MCP | 1 |
| none / all non-fail | 6 |

## Retained research limitations

### `xd_dev_002` Reason failure (kept)

- `reason_parse_failed=true`
- `runtime_recovered=false`
- `repair_count=1`
- `initial_parse_error="$: unexpected property 'basic_eps'"`
- Stage outcome = fail (Reason must not count pass)

### EPS evidence semantic limitation (kept)

Reported FY2025 basic EPS **8.504848** vs benchmark **1.21**.  
Operation execution and required-field completeness do **not** guarantee metric-basis / unit semantic correctness.  
No HSBC / gold-specific adapter patch in v1.2.

## Raw SHA256 (immutable)

| File | SHA256 |
|---|---|
| predictions.jsonl | `e1420f3c4d4bbb8832cc1bdcdc5e52bb2e80193e35f8fb1d9e40edbdb7ad573c` |
| run_metadata.json | `b024cc2b35e50c8c1405e011857546d7b4cf7c58b0088a0de92c918215e0d6ba` |
| error_summary.json | `5d7955c3c010ca2561531c8e85438683493b07bb44727a5561b4e4bbe17fe3d8` |
| cost_summary.json | `00f72ac9966ce3a273a7a644805f02f8f9a6038f0ee1a20bfa712751525b5aee` |
| grading_summary.json | `bb4b3d33cea466f9b54c5143f02c67a69c31ce885a91428e3045ebbcc9a09c7f` |
| per_task_metrics.csv | `85179fe73da6051f322b1a9e60c6df74cf8d24d32fe6724f8c0fc10f9e3feef1` |

before == after at freeze verification; raw run was **not** mutated.

## Source paths (read-only)

- Run: `outputs/pe_fin_lite_diagnostic12_checkpoint_live_v3/`
- Audit: `outputs/pe_fin_lite_diagnostic12_checkpoint_live_v3/final_checkpoint_audit_v1/`

See also:

- `v1_v2_v3_comparison.md`
- `method_freeze_declaration.md`
- `raw_hashes_freeze_verification.json`
