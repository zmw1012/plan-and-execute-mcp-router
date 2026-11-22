# PE-Fin RePlan ETOM-100 Identity Freeze

Offline transfer condition. Not a formal primary HK-155 result.
ETOM-100 task IDs / order / questions / gold / evaluator stay frozen.

## Method identity

| Field | Value |
| --- | --- |
| condition | `pe_fin_replan_etom` |
| method_version | `pe_fin_replan_etom_v1` |
| protocol_version | `planner_executor_replanner_etom_seven_round_v1` |
| architecture | `plan_execute_replan` |
| base_method | `pe_fin_replan_v1_4_1` |
| transfer_source | `hk155_pe_fin_replan` |
| max_execution_rounds | 7 |
| max_replan_rounds | 7 |
| max_executed_steps | 7 |
| max_total_llm_calls_per_task | 16 |
| max_total_mcp_invocations | 8 |
| posthoc_exploratory | true |
| formal_primary_result | false |
| cross_benchmark_transfer_evaluation | true |
| Monitor | disabled |
| global replan | disabled |
| RePlan-4 | disabled on ETOM |

## ETOM-100 dataset identity

| Field | Value |
| --- | --- |
| dataset / sampler | `etom_stratified_external_100_v1` |
| seed | `20261106` |
| sample_size | 100 (20 per L1–L5) |
| lock path | `benchmarks/etom/formal_100_v1.lock.json` |
| lock SHA256 | `2bda1a2f16dd1b36b5a73b4057b7305b4901ed83eba0b537db2f187a56ed309b` |
| sample_sha256 | `99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1` |
| ordered_task_ids_sha256 | `3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479` |
| ordered_ids newline join SHA256 | `528eb682db72025933752f85e63f03b09ab51b93113e79baf3eff3fcc058e895` |
| questions hash (ordered query_sha256 join) | `4c5b4962aaab874b2c2923e77066b64bc0c67830e82bf010476438f3228f893b` |
| gold/reference hash (ordered ground_truth_sha256 join) | `c8541379212f98ffae628290128dd8a610470e85584465d6e6389c057e354c02` |
| source repository | `https://github.com/snooow1029/ETOM` |
| source_commit | `790f28727f567b88918420d163c524d46644a694` |
| source_pin_sha256 | `5f1ae09e7a8fbb7b554ed1302433feacaffb8645ccc560c7bb87ff64893f2692` |
| registry_sha256 | `3c362877ec5733cf0fa1d3451b2f353fb97674e66075da0d7a6f41c0f98cc608` |
| server count | 491 |
| tool / operation count | 2375 |
| evaluator module | `hk_equity_mcp.etom.evaluator` |
| evaluator.py SHA256 | `1f2d45e01db2879a59298a001473f15b9dddb2e643d1276a7dc92ab9116dd482` |

Gold / accepted tools / evaluator are evaluation-only. They must not construct
Planner prompts, retrieval queries, operation documents, or runtime routing.

## Catalog / index identity

| Field | Value |
| --- | --- |
| capability profile | `configs/capabilities/etom100_capability_profile_v1.yaml` |
| operation docs namespace | ETOM registry-derived documents (condition-local) |
| embedding index cache | `.cache/pe_fin_replan_etom_operation_embeddings_v1` |
| forbidden reuse | HK `pe_fin_lite_operation_embeddings_v1`, baseline `etom_embeddings` |
| retrieval formula | R5 `cosine_similarity + 0.12 * lexical_score` |
| Top-K / Describe-N | 5 / 3 |

## Existing Operation-Visible baseline run (reference only)

Baseline condition `operation_visible` formal100 live identity (do not overwrite):

| Field | Value |
| --- | --- |
| formal_sample_sha256 | `99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1` |
| ordered_task_ids_sha256 | `3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479` |
| catalog completed/total | 2375 / 2375 |
| generation model | `deepseek-v4.1-flash` |
| embedding model | `text-embedding-3-small` |
| baseline branch (historical) | `feature/etom-formal100-v1` |

## Offline safety

This worktree round allows mock / dry-run / unit tests / hash embeddings only.
No real generation LLM, embedding API, ETOM MCP, or live formal100 until
HK-155 is frozen and the user explicitly authorizes ETOM live.
