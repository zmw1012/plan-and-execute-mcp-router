# PE-Fin Lite End-to-End Checkpoint Audit v1

- audit_version: `pe_fin_lite_checkpoint_audit_v1`
- result_role: `engineering_checkpoint_audit_only`
- explicitly_not_formal_accuracy: `True`
- engineering_pass: `True`
- n_predictions: 12
- unique_task_ids: 12
- raw_hashes_unchanged: `True`
- checkpoint_trace_contract_version: `pe_fin_lite_checkpoint_trace_v1`

## Engineering pass (not accuracy)

```json
{
  "engineering_pass": true,
  "explicitly_not_formal_accuracy": true,
  "not_accuracy_pass": true,
  "checks": {
    "predictions_complete": true,
    "unique_ids": true,
    "run_status_completed": true,
    "no_runner_crash": true,
    "raw_hashes_unchanged": true,
    "primary_error_localizable": true,
    "na_not_counted_as_pass": true,
    "evaluator_separated": true,
    "zero_invocation_not_tool_success": true,
    "failed_partial_not_strict_success": true,
    "no_silent_mock_fallback": true,
    "trace_coverage_adequate": true
  },
  "failing_checks": [],
  "n_rows": 12,
  "unique_ids": 12,
  "expected_n": 12,
  "notes": [
    "Answer may be incorrect while engineering_pass is true.",
    "N/A / N/A_* is not counted as pass.",
    "Evaluator errors are separated from agent/method errors.",
    "engineering_pass is derived only from invariant checks (never hard-coded)."
  ]
}
```

## Checkpoint funnel

- C01_planner: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 12})
- C02_filter: pass=0 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=0.000; headline=stage_outcome; stage={'not_evaluated': 11, 'N/A': 1})
- C03_embedding: pass=0 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=0.000; headline=stage_outcome; stage={'not_evaluated': 11, 'N/A': 1})
- C04_describe: pass=11 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 11, 'N/A': 1})
- C05_binder: pass=11 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 11, 'N/A': 1})
- C06_arguments: pass=11 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 11, 'N/A': 1})
- C07_mcp: pass=10 fail=1 N/A=1 unknown=0 (pass_rate_excl_NA=0.909; headline=stage_outcome; stage={'pass': 10, 'fail': 1, 'N/A_not_planned': 1})
- C08_normalize: pass=11 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 11, 'N/A': 1})
- C09_evidence_acceptance: pass=6 fail=5 N/A=1 unknown=0 (pass_rate_excl_NA=0.545; headline=stage_outcome; stage={'fail': 5, 'pass': 6, 'N/A': 1})
- C10_fallback_unavailable: pass=11 fail=0 N/A=1 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 11, 'N/A': 1})
- C11_compute: pass=1 fail=0 N/A=11 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'N/A_not_planned': 10, 'pass': 1, 'N/A_upstream_blocked': 1})
- C12_reason: pass=4 fail=1 N/A=7 unknown=0 (pass_rate_excl_NA=0.800; headline=stage_outcome; stage={'N/A_upstream_blocked': 3, 'N/A_not_planned': 4, 'fail': 1, 'pass': 4})
- C13_final: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 12})
- C14_evaluator: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000; headline=stage_outcome; stage={'pass': 12})

## Primary blocking taxonomy

```json
{
  "C09_evidence_acceptance": 4,
  "none_or_all_non_fail": 6,
  "C12_reason": 1,
  "C07_mcp": 1
}
```

## Notes

- N/A is not counted as pass.
- Upstream failures remain failures even if final answer happens to look correct.
- Evaluator (C14) is separated from agent/method errors.
- Zero MCP invocation is never recorded as tool success.

