# PE-Fin Lite R5 HK-155 Final Audit (pe_fin_lite_r5_hk155_final_audit_v1)

## 1. Branch / commit / status
- run git_branch: `feature/pe-fin-lite-proposed-method-v1`
- run git_commit: `7500a948d9a0a47483d7fc826178b1b894e8745f`
- run git_dirty: `False`
- audit timestamp_utc: `2026-11-12T05:46:30.179540+00:00`
- identity_all_pass: `True`
- preliminary_facts_match: `True`

## 2. Raw SHA256 before/after
- See `raw_hashes.json` (`unchanged=true` required).
- audit raw_hashes_unchanged flag (post-write): sealed in artifact.

## 3. Identity verdict
- all_required_pass: `True`
- failed_checks: `[]`
- domain_counts: `{'financial': 60, 'market': 35, 'news': 35, 'cross_domain': 25}`
- duplicate_count: `0`
- missing_task_ids: `[]`
- formula_hash_recomputed: `7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e`

## 4. 155 completeness
- rows/unique: `155` / `155`
- order matches frozen exploratory_all: `True`
- processed_task_count / run_status via identity checks: ok if all_required_pass

## 5. Full / partial / failed
- `{'failed': 94, 'full': 14, 'partial': 47}`

## 6. Auto / manual / final answer metrics
- auto_gradable_accuracy: `0.22727272727272727` (n=154)
- final_answer lower/upper: `0.22580645161290322` / `0.23225806451612904`
- final_answer_accuracy_pending_manual: `True`
- manual_required_n: `1`

## 7. Common strict answer success
- `{'definition': 'status==full AND answer_correct==true; partial/failed forced incorrect; denominator=155', 'count': 9, 'rate': 0.05806451612903226}`

## 8. Exact tool-set and operation F1
- exact_tool_set: `{'count': 22, 'rate': 0.14193548387096774}`
- operation_precision_mean: `0.14193548387096774`
- operation_recall_mean: `0.14193548387096774`
- operation_f1_mean: `0.14193548387096774`
- all_mcp_invocations_successful_rate: `0.5655172413793104`
- evidence_acceptance_proxy: `0.21563218390804598`
- condition-specific E2E (PE-Fin only): `{'formula': "PE-Fin Lite strict_answer_success_rate from aggregate_pe_fin_lite_metrics: answer_correct AND predicted_status=='answered' AND status in {full,partial}; NOT comparable to other conditions' condition-specific strict.", 'rate': 0.09032258064516129}`
- refusal/clarification: `{'n_should_refuse_or_clarify': 6, 'predicted_unavailable_or_clarify': 133}`

## 9. C01–C14 funnel
*N/A ≠ pass; not_evaluated ≠ pass; evaluator executed ≠ answer correct.*
- `C01_planner`: pass=153 fail=2 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C02_filter`: pass=0 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=145 unknown=0
- `C03_embedding`: pass=0 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=145 unknown=0
- `C04_describe`: pass=145 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C05_binder`: pass=145 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C06_arguments`: pass=145 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C07_mcp`: pass=82 fail=63 N/A_not_planned=8 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C08_normalize`: pass=145 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C09_evidence_acceptance`: pass=15 fail=130 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C10_fallback_unavailable`: pass=145 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C11_compute`: pass=1 fail=0 N/A_not_planned=122 N/A_not_reached=0 N/A_upstream_blocked=32 not_evaluated=0 unknown=0
- `C12_reason`: pass=17 fail=2 N/A_not_planned=50 N/A_not_reached=0 N/A_upstream_blocked=86 not_evaluated=0 unknown=0
- `C13_final`: pass=155 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0
- `C14_evaluator`: pass=155 fail=0 N/A_not_planned=0 N/A_not_reached=0 N/A_upstream_blocked=0 not_evaluated=0 unknown=0

## 10. Primary blockers
```json
{
  "primary_blocking_checkpoint_counts": {
    "C07_mcp": 63,
    "C09_evidence_acceptance": 67,
    "none_or_all_non_fail": 21,
    "C12_reason": 2,
    "C01_planner": 2
  },
  "secondary_failure_total": 63,
  "by_domain": {
    "financial": {
      "C07_mcp": 6,
      "none_or_all_non_fail": 8,
      "C12_reason": 1,
      "C09_evidence_acceptance": 45
    },
    "market": {
      "C07_mcp": 28,
      "none_or_all_non_fail": 3,
      "C09_evidence_acceptance": 2,
      "C01_planner": 2
    },
    "news": {
      "C07_mcp": 13,
      "none_or_all_non_fail": 8,
      "C09_evidence_acceptance": 13,
      "C12_reason": 1
    },
    "cross_domain": {
      "C09_evidence_acceptance": 7,
      "C07_mcp": 16,
      "none_or_all_non_fail": 2
    }
  },
  "by_difficulty": {
    "D1": {
      "C07_mcp": 13,
      "none_or_all_non_fail": 7,
      "C12_reason": 1,
      "C09_evidence_acceptance": 14
    },
    "D2": {
      "C09_evidence_acceptance": 10,
      "C07_mcp": 10,
      "none_or_all_non_fail": 4
    },
    "D3": {
      "C09_evidence_acceptance": 21,
      "C07_mcp": 16,
      "none_or_all_non_fail": 3,
      "C01_planner": 1,
      "C12_reason": 1
    },
    "D4": {
      "C07_mcp": 15,
      "none_or_all_non_fail": 4,
      "C01_planner": 1,
      "C09_evidence_acceptance": 8
    },
    "D5": {
      "C07_mcp": 9,
      "C09_evidence_acceptance": 14,
      "none_or_all_non_fail": 3
    }
  },
  "by_status": {
    "full": {
      "none_or_all_non_fail": 12,
      "C12_reason": 2
    },
    "partial": {
      "C07_mcp": 19,
      "C09_evidence_acceptance": 19,
      "none_or_all_non_fail": 9
    },
    "failed": {
      "C07_mcp": 44,
      "C09_evidence_acceptance": 48,
      "C01_planner": 2
    }
  }
}
```

## 11. Domain / difficulty / answerability
```json
{
  "by_domain": {
    "financial": {
      "n": 60,
      "answer_correct_true": 15,
      "answer_correct_false": 44,
      "answer_correct_null": 1,
      "auto_gradable_accuracy": 0.2542372881355932,
      "common_strict_answer_success": 0.08333333333333333,
      "exact_tool_set": 0.18333333333333332,
      "status_counts": {
        "failed": 32,
        "full": 6,
        "partial": 22
      },
      "primary_blocker_counts": {
        "C07_mcp": 6,
        "none_or_all_non_fail": 8,
        "C12_reason": 1,
        "C09_evidence_acceptance": 45
      },
      "mean_llm_calls": 3.283333333333333,
      "mean_mcp_calls": 3.066666666666667,
      "mean_tokens": 7322.716666666666,
      "mean_latency_ms": 30415.69348537984
    },
    "market": {
      "n": 35,
      "answer_correct_true": 6,
      "answer_correct_false": 29,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.17142857142857143,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.17142857142857143,
      "status_counts": {
        "failed": 32,
        "partial": 3
      },
      "primary_blocker_counts": {
        "C07_mcp": 28,
        "none_or_all_non_fail": 3,
        "C09_evidence_acceptance": 2,
        "C01_planner": 2
      },
      "mean_llm_calls": 3.1714285714285713,
      "mean_mcp_calls": 1.5714285714285714,
      "mean_tokens": 7299.142857142857,
      "mean_latency_ms": 28657.47732483516
    },
    "news": {
      "n": 35,
      "answer_correct_true": 13,
      "answer_correct_false": 22,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.37142857142857144,
      "common_strict_answer_success": 0.11428571428571428,
      "exact_tool_set": 0.14285714285714285,
      "status_counts": {
        "failed": 20,
        "full": 7,
        "partial": 8
      },
      "primary_blocker_counts": {
        "C07_mcp": 13,
        "none_or_all_non_fail": 8,
        "C09_evidence_acceptance": 13,
        "C12_reason": 1
      },
      "mean_llm_calls": 2.6857142857142855,
      "mean_mcp_calls": 1.9428571428571428,
      "mean_tokens": 6188.0,
      "mean_latency_ms": 23786.17276066382
    },
    "cross_domain": {
      "n": 25,
      "answer_correct_true": 1,
      "answer_correct_false": 24,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.04,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.0,
      "status_counts": {
        "failed": 10,
        "partial": 14,
        "full": 1
      },
      "primary_blocker_counts": {
        "C09_evidence_acceptance": 7,
        "C07_mcp": 16,
        "none_or_all_non_fail": 2
      },
      "mean_llm_calls": 3.64,
      "mean_mcp_calls": 2.68,
      "mean_tokens": 9412.12,
      "mean_latency_ms": 35259.72828004509
    }
  },
  "by_difficulty": {
    "D1": {
      "n": 35,
      "answer_correct_true": 8,
      "answer_correct_false": 27,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.22857142857142856,
      "common_strict_answer_success": 0.17142857142857143,
      "exact_tool_set": 0.0,
      "status_counts": {
        "failed": 20,
        "full": 7,
        "partial": 8
      },
      "primary_blocker_counts": {
        "C07_mcp": 13,
        "none_or_all_non_fail": 7,
        "C12_reason": 1,
        "C09_evidence_acceptance": 14
      },
      "mean_llm_calls": 3.0,
      "mean_mcp_calls": 2.4857142857142858,
      "mean_tokens": 6100.285714285715,
      "mean_latency_ms": 20902.972629672982
    },
    "D2": {
      "n": 24,
      "answer_correct_true": 2,
      "answer_correct_false": 21,
      "answer_correct_null": 1,
      "auto_gradable_accuracy": 0.08695652173913043,
      "common_strict_answer_success": 0.08333333333333333,
      "exact_tool_set": 0.0,
      "status_counts": {
        "failed": 18,
        "partial": 4,
        "full": 2
      },
      "primary_blocker_counts": {
        "C09_evidence_acceptance": 10,
        "C07_mcp": 10,
        "none_or_all_non_fail": 4
      },
      "mean_llm_calls": 3.0,
      "mean_mcp_calls": 2.3333333333333335,
      "mean_tokens": 6674.958333333333,
      "mean_latency_ms": 24188.912880092783
    },
    "D3": {
      "n": 42,
      "answer_correct_true": 3,
      "answer_correct_false": 39,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.07142857142857142,
      "common_strict_answer_success": 0.023809523809523808,
      "exact_tool_set": 0.0,
      "status_counts": {
        "failed": 26,
        "full": 4,
        "partial": 12
      },
      "primary_blocker_counts": {
        "C09_evidence_acceptance": 21,
        "C07_mcp": 16,
        "none_or_all_non_fail": 3,
        "C01_planner": 1,
        "C12_reason": 1
      },
      "mean_llm_calls": 3.2857142857142856,
      "mean_mcp_calls": 2.761904761904762,
      "mean_tokens": 8600.02380952381,
      "mean_latency_ms": 33570.641618011345
    },
    "D4": {
      "n": 28,
      "answer_correct_true": 6,
      "answer_correct_false": 22,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.21428571428571427,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.0,
      "status_counts": {
        "partial": 16,
        "failed": 11,
        "full": 1
      },
      "primary_blocker_counts": {
        "C07_mcp": 15,
        "none_or_all_non_fail": 4,
        "C01_planner": 1,
        "C09_evidence_acceptance": 8
      },
      "mean_llm_calls": 3.357142857142857,
      "mean_mcp_calls": 2.142857142857143,
      "mean_tokens": 8222.392857142857,
      "mean_latency_ms": 34963.02938986004
    },
    "D5": {
      "n": 26,
      "answer_correct_true": 16,
      "answer_correct_false": 10,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.6153846153846154,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.8461538461538461,
      "status_counts": {
        "partial": 7,
        "failed": 19
      },
      "primary_blocker_counts": {
        "C07_mcp": 9,
        "C09_evidence_acceptance": 14,
        "none_or_all_non_fail": 3
      },
      "mean_llm_calls": 3.230769230769231,
      "mean_mcp_calls": 2.1153846153846154,
      "mean_tokens": 6983.807692307692,
      "mean_latency_ms": 32342.03258011705
    }
  },
  "by_source_split": {
    "development": {
      "n": 70,
      "answer_correct_true": 16,
      "answer_correct_false": 53,
      "answer_correct_null": 1,
      "auto_gradable_accuracy": 0.2318840579710145,
      "common_strict_answer_success": 0.05714285714285714,
      "exact_tool_set": 0.12857142857142856,
      "status_counts": {
        "failed": 39,
        "full": 8,
        "partial": 23
      },
      "primary_blocker_counts": {
        "C07_mcp": 23,
        "none_or_all_non_fail": 9,
        "C12_reason": 2,
        "C09_evidence_acceptance": 35,
        "C01_planner": 1
      },
      "mean_llm_calls": 3.2857142857142856,
      "mean_mcp_calls": 2.6142857142857143,
      "mean_tokens": 7702.6,
      "mean_latency_ms": 29936.44637254599
    },
    "previously_reserved": {
      "n": 85,
      "answer_correct_true": 19,
      "answer_correct_false": 66,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.2235294117647059,
      "common_strict_answer_success": 0.058823529411764705,
      "exact_tool_set": 0.15294117647058825,
      "status_counts": {
        "failed": 55,
        "full": 6,
        "partial": 24
      },
      "primary_blocker_counts": {
        "C07_mcp": 40,
        "C09_evidence_acceptance": 32,
        "none_or_all_non_fail": 12,
        "C01_planner": 1
      },
      "mean_llm_calls": 3.0941176470588236,
      "mean_mcp_calls": 2.2470588235294118,
      "mean_tokens": 7147.458823529411,
      "mean_latency_ms": 28781.309682801915
    }
  },
  "by_answerability": {
    "capability_gap": {
      "n": 5,
      "answer_correct_true": 5,
      "answer_correct_false": 0,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 1.0,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 1.0,
      "status_counts": {
        "failed": 5
      },
      "primary_blocker_counts": {
        "C09_evidence_acceptance": 5
      },
      "mean_llm_calls": 3.2,
      "mean_mcp_calls": 2.0,
      "mean_tokens": 6457.6,
      "mean_latency_ms": 37036.27801700495
    },
    "evidence_answerable": {
      "n": 29,
      "answer_correct_true": 1,
      "answer_correct_false": 28,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.034482758620689655,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.0,
      "status_counts": {
        "partial": 15,
        "failed": 14
      },
      "primary_blocker_counts": {
        "C09_evidence_acceptance": 23,
        "none_or_all_non_fail": 2,
        "C07_mcp": 4
      },
      "mean_llm_calls": 3.1379310344827585,
      "mean_mcp_calls": 3.5172413793103448,
      "mean_tokens": 7220.172413793103,
      "mean_latency_ms": 30839.488071766842
    },
    "intentionally_infeasible": {
      "n": 11,
      "answer_correct_true": 5,
      "answer_correct_false": 6,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.45454545454545453,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 1.0,
      "status_counts": {
        "partial": 2,
        "failed": 9
      },
      "primary_blocker_counts": {
        "none_or_all_non_fail": 1,
        "C09_evidence_acceptance": 9,
        "C07_mcp": 1
      },
      "mean_llm_calls": 3.272727272727273,
      "mean_mcp_calls": 2.272727272727273,
      "mean_tokens": 6862.090909090909,
      "mean_latency_ms": 31861.095712139188
    },
    "robustness": {
      "n": 8,
      "answer_correct_true": 5,
      "answer_correct_false": 3,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.625,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.0,
      "status_counts": {
        "partial": 6,
        "failed": 1,
        "full": 1
      },
      "primary_blocker_counts": {
        "none_or_all_non_fail": 4,
        "C07_mcp": 4
      },
      "mean_llm_calls": 3.25,
      "mean_mcp_calls": 1.0,
      "mean_tokens": 6194.625,
      "mean_latency_ms": 23838.4878175857
    },
    "should_refuse_or_clarify": {
      "n": 6,
      "answer_correct_true": 6,
      "answer_correct_false": 0,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 1.0,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 1.0,
      "status_counts": {
        "failed": 4,
        "partial": 2
      },
      "primary_blocker_counts": {
        "C07_mcp": 4,
        "none_or_all_non_fail": 2
      },
      "mean_llm_calls": 2.6666666666666665,
      "mean_mcp_calls": 1.1666666666666667,
      "mean_tokens": 4727.5,
      "mean_latency_ms": 20649.115228792652
    },
    "structured_answerable": {
      "n": 96,
      "answer_correct_true": 13,
      "answer_correct_false": 82,
      "answer_correct_null": 1,
      "auto_gradable_accuracy": 0.1368421052631579,
      "common_strict_answer_success": 0.09375,
      "exact_tool_set": 0.0,
      "status_counts": {
        "failed": 61,
        "full": 13,
        "partial": 22
      },
      "primary_blocker_counts": {
        "C07_mcp": 50,
        "C09_evidence_acceptance": 30,
        "none_or_all_non_fail": 12,
        "C12_reason": 2,
        "C01_planner": 2
      },
      "mean_llm_calls": 3.2083333333333335,
      "mean_mcp_calls": 2.3125,
      "mean_tokens": 7829.5625,
      "mean_latency_ms": 29139.18102294944
    }
  },
  "by_status": {
    "full": {
      "n": 14,
      "answer_correct_true": 9,
      "answer_correct_false": 5,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.6428571428571429,
      "common_strict_answer_success": 0.6428571428571429,
      "exact_tool_set": 0.0,
      "status_counts": {
        "full": 14
      },
      "primary_blocker_counts": {
        "none_or_all_non_fail": 12,
        "C12_reason": 2
      },
      "mean_llm_calls": 3.9285714285714284,
      "mean_mcp_calls": 1.1428571428571428,
      "mean_tokens": 8854.5,
      "mean_latency_ms": 25658.899711045837
    },
    "partial": {
      "n": 47,
      "answer_correct_true": 11,
      "answer_correct_false": 36,
      "answer_correct_null": 0,
      "auto_gradable_accuracy": 0.23404255319148937,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.0851063829787234,
      "status_counts": {
        "partial": 47
      },
      "primary_blocker_counts": {
        "C07_mcp": 19,
        "C09_evidence_acceptance": 19,
        "none_or_all_non_fail": 9
      },
      "mean_llm_calls": 3.595744680851064,
      "mean_mcp_calls": 3.1914893617021276,
      "mean_tokens": 9175.063829787234,
      "mean_latency_ms": 35118.76922781936
    },
    "failed": {
      "n": 94,
      "answer_correct_true": 15,
      "answer_correct_false": 78,
      "answer_correct_null": 1,
      "auto_gradable_accuracy": 0.16129032258064516,
      "common_strict_answer_success": 0.0,
      "exact_tool_set": 0.19148936170212766,
      "status_counts": {
        "failed": 94
      },
      "primary_blocker_counts": {
        "C07_mcp": 44,
        "C09_evidence_acceptance": 48,
        "C01_planner": 2
      },
      "mean_llm_calls": 2.8617021276595747,
      "mean_mcp_calls": 2.2127659574468086,
      "mean_tokens": 6292.81914893617,
      "mean_latency_ms": 26937.827866534364
    }
  }
}
```

## 12. Retrieval R5
- r5_runtime_effective: `True`
```json
{
  "embedding_retrieve_steps": 131,
  "disclosure_retrieve_steps": 65,
  "score_rows_checked": 655,
  "score_rows_ok": 655,
  "score_formula_ok": true,
  "domain_incompatible_rejects": 0,
  "filter_anomalies": 0,
  "safety_only_r5_steps": 131,
  "pool39_steps": 131,
  "topk_gt5_steps": 0,
  "describe_gt3_steps": 0,
  "topk_candidate_rows": 655,
  "described_candidate_rows": 393,
  "primary_selections": 131,
  "fallback_selections": 117,
  "fallback_attempts": 82,
  "rank4_5_not_described_entries": 262,
  "zero_candidate_steps": 0,
  "zero_invocation_steps": 0,
  "binder_outside_described": 0,
  "formula_name_mismatches": 0,
  "note_zero_invocation": "zero_invocation_steps != tool success; counted separately",
  "r5_runtime_effective": true
}
```
- evaluation-only gold metrics: `{"n_tasks_with_gold_ops": 133, "hit@1": 0.518796992481203, "hit@3": 0.5789473684210527, "hit@5": 0.5789473684210527, "recall@1": 0.3681704260651629, "recall@3": 0.4523809523809524, "recall@5": 0.46516290726817044, "mrr": 0.5463659147869674, "complete_route@3": 0.3458646616541353, "complete_route@5": 0.3533834586466165, "note": "evaluation-only; gold ops never enter runtime ranking"}`

## 13. Disclosure / compute / reason / final
- disclosure: `{"disclosure_steps": 65, "list": 65, "chain_started_but_incomplete_proxy": 46, "evidence_found_but_not_accepted_proxy": 46, "stopped_after_list": 19, "execution_ok": 19, "list_success_not_body_evidence": 19, "download": 24, "parse": 24, "keyword": 24, "pages": 24}`
- compute/reason/final: `{"compute": {"planned": 33, "executed_status_rows": 40, "completed": 1}, "reason": {"planned": 105, "completed_status_rows": 17, "repair_tasks": 12, "parse_failed_flags": 2}, "final": {"unavailable": 132, "answered": 22, "semantically_correct_full": 9, "clarification_needed": 1}}`

## 14. Resources
```json
{
  "llm_calls_total": 493,
  "mcp_invocations_total": 374,
  "embedding_query_calls_total": 131,
  "tokens_total_sum_per_task": 1146716.0,
  "tokens_total_from_cost_summary": 1146716,
  "latency_ms_mean": 29302.984316879883,
  "latency_ms_median": 26821.417500032112,
  "latency_ms_p95": 53392.20945793204,
  "estimated_usd": null,
  "calls_by_stage_totals": {
    "planner": 155,
    "final": 151,
    "tool_binding": 131,
    "reason": 19,
    "reason_repair": 12,
    "final_repair": 6,
    "tool_binding_repair": 8,
    "planner_repair": 11
  },
  "fallback_total": 82,
  "budget_violations": [],
  "catalog_index_build_calls": 0,
  "note": "index reused complete cache; no build during this run expected"
}
```

## 15. Baseline comparison (descriptive only)
```json
{
  "pe_fin_lite_r5": {
    "final_answer_accuracy_lower_bound": 0.22580645161290322,
    "final_answer_accuracy_upper_bound": 0.23225806451612904,
    "common_strict_answer_success": 0.05806451612903226,
    "exact_tool_set": 0.14193548387096774,
    "tokens_total": 1146716,
    "condition_specific_strict": 0.09032258064516129,
    "condition_specific_note": "not comparable across methods"
  },
  "baselines_frozen": {
    "MCP-Free": {
      "final_answer_accuracy": 0.129,
      "common_strict_answer_success": null,
      "exact_tool_set": null,
      "tokens_total": 79133,
      "note": "frozen baseline figures from protocol; not re-derived here"
    },
    "Flat All Tools": {
      "final_answer_accuracy": 0.5677,
      "common_strict_answer_success": 0.5484,
      "exact_tool_set": 0.2323,
      "tokens_total": 9464124
    },
    "Operation-Visible": {
      "final_answer_accuracy": 0.6129,
      "common_strict_answer_success": null,
      "exact_tool_set": 0.4129,
      "condition_specific_strict": 0.2903,
      "tokens_total": null,
      "note": "condition-specific strict not comparable across methods"
    },
    "MCP-Zero": {
      "final_answer_accuracy": 0.5806,
      "common_strict_answer_success": 0.5806,
      "exact_tool_set": 0.439,
      "condition_specific_strict": 0.3677,
      "tokens_total": 4050513,
      "note": "condition-specific strict not comparable across methods"
    }
  },
  "descriptive_deltas_vs_baselines_final_answer_lower": {
    "MCP-Free": 0.09680645161290322,
    "Flat All Tools": -0.34189354838709674,
    "Operation-Visible": -0.38709354838709675,
    "MCP-Zero": -0.35479354838709676
  },
  "descriptive_deltas_vs_baselines_common_strict": {
    "MCP-Free": null,
    "Flat All Tools": -0.49033548387096776,
    "Operation-Visible": null,
    "MCP-Zero": -0.5225354838709677
  },
  "no_significance_claims": true
}
```

## 16. engineering_valid
```json
{
  "engineering_valid": true,
  "engineering_blockers": [],
  "method_valid_failure_note": "High full/partial/failed counts and low answer accuracy are treated as method-valid outcomes under the frozen R5 PE-Fin Lite contract unless an identity/R5-wiring/budget-implementation inconsistency is proven.",
  "evidence_r5_wired": true,
  "evidence_domain_filter_off": true
}
```

## 17. Manual scoring required?
- yes if pending: `['fin_cf_022']`
- lower/upper answer accuracy bounds above; do not seal until human_grades filled.

## 18. Constraints confirmation
- no API
- no rerun
- no raw mutation
- no method change
- no push

## 19. Stop and wait
Stop and wait for human grades before sealing final answer accuracy.
