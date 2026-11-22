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
  "n_rows": 12,
  "unique_ids": 12,
  "expected_n": 12,
  "notes": [
    "Answer may be incorrect while engineering_pass is true.",
    "N/A is not counted as pass.",
    "Evaluator errors are separated from agent/method errors."
  ]
}
```

## Checkpoint funnel

- C01_planner: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C02_filter: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C03_embedding: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C04_describe: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C05_binder: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C06_arguments: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C07_mcp: pass=9 fail=3 N/A=0 unknown=0 (pass_rate_excl_NA=0.750)
- C08_normalize: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C09_evidence_acceptance: pass=4 fail=8 N/A=0 unknown=0 (pass_rate_excl_NA=0.333)
- C10_fallback_unavailable: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C11_compute: pass=2 fail=1 N/A=9 unknown=0 (pass_rate_excl_NA=0.667)
- C12_reason: pass=7 fail=2 N/A=3 unknown=0 (pass_rate_excl_NA=0.778)
- C13_final: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)
- C14_evaluator: pass=12 fail=0 N/A=0 unknown=0 (pass_rate_excl_NA=1.000)

## Primary blocking taxonomy

```json
{
  "C09_evidence_acceptance": 5,
  "C07_mcp": 3,
  "C12_reason": 2,
  "C11_compute": 1,
  "none_or_all_non_fail": 1
}
```

## Notes

- N/A is not counted as pass.
- Upstream failures remain failures even if final answer happens to look correct.
- Evaluator (C14) is separated from agent/method errors.
- Zero MCP invocation is never recorded as tool success.

