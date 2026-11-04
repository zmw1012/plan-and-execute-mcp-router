"""Output schemas for the unified HK Operation-Visible baseline."""

from __future__ import annotations

PREDICTION_FIELDS = (
    "task_id",
    "id",
    "domain",
    "domain_combination",
    "project_difficulty",
    "exact_etom_level",
    "exact_etom_alignment",
    "etom_level",
    "etom_alignment",
    "question",
    "status",
    "error",
    "final_answer",
    "predicted_status",
    "predicted_operations",
    "retrieved_candidates",
    "describe_calls",
    "invoke_calls",
    "dependency_edges_predicted",
    # tool_execution_success == all_tool_invocations_successful
    # (True iff every invoke succeeded; vacuous True when zero invokes).
    "tool_execution_success",
    "all_tool_invocations_successful",
    "successful_invocation_count",
    "failed_invocation_count",
    "recovered_after_tool_error",
    "protocol_success_rate",
    "final_operation_success_by_operation_id",
    "evidence",
    "citations",
    "llm_calls",
    "tokens",
    "embedding_calls",
    "embedding_tokens",
    "latency_ms",
    "retry_count",
    "timeout",
    "parse_repair_count",
    "answerability_class",
    "gold_operations",
    "operation_precision",
    "operation_recall",
    "operation_f1",
    "exact_tool_set",
    "graph_dependency_accuracy",
    "wrong_tool_rate",
    "refusal_correctness",
    "real_mcp_execution_count",
    "trace",
    "resolver_calls",
    "underlying_operation_calls",
    "provider_metadata",
    "condition",
    # Manual grading placeholders (filled later / copied to CSV)
    "manual_answer_correctness",
    "manual_evidence_correctness",
    "manual_unsupported_claim",
    "manual_causal_overclaim",
    "manual_overall_task_success",
)

PER_TASK_METRIC_FIELDS = (
    "task_id",
    "domain",
    "domain_combination",
    "project_difficulty",
    "exact_etom_level",
    "exact_etom_alignment",
    "answerability_class",
    "status",
    "operation_precision",
    "operation_recall",
    "operation_f1",
    "exact_tool_set",
    "graph_dependency_accuracy",
    "tool_execution_success",
    "all_tool_invocations_successful",
    "successful_invocation_count",
    "failed_invocation_count",
    "recovered_after_tool_error",
    "protocol_success_rate",
    "wrong_tool_rate",
    "refusal_correctness",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "embedding_calls",
    "embedding_tokens",
    "latency_ms",
    "real_mcp_execution_count",
    "retry_count",
    "timeout",
    "parse_repair_count",
)

METRIC_FIELD_DEFINITIONS = {
    "tool_execution_success": (
        "Alias of all_tool_invocations_successful: True iff every invoke_api "
        "succeeded. Vacuous True when there were zero invocations "
        "(refusal without tools is not an execution failure)."
    ),
    "all_tool_invocations_successful": (
        "True iff every invoke_api succeeded (vacuous True if none)."
    ),
    "recovered_after_tool_error": (
        "True when an earlier invoke failed, a later invoke succeeded, and the "
        "agent still produced a complete final answer (status=full). "
        "Execution recovery only — not answer correctness."
    ),
    "error_counts": (
        "Counts of non-empty prediction error/error_type only. "
        "status=full is never counted as an error."
    ),
    "result_status_counts": (
        "Independent counts of agent result status (full/failed/…). "
        "Not an error taxonomy."
    ),
}

MANUAL_GRADING_FIELDS = (
    "task_id",
    "domain",
    "domain_combination",
    "company",
    "project_difficulty",
    "exact_etom_level",
    "exact_etom_alignment",
    "question",
    "gold_status",
    "answerability_class",
    "gold_operations",
    "reference_answer",
    "predicted_status",
    "predicted_answer",
    "predicted_operations",
    "evidence",
    "manual_answer_correctness",
    "manual_evidence_correctness",
    "manual_unsupported_claim",
    "manual_causal_overclaim",
    "manual_overall_task_success",
    "notes",
)

BANNED_PROMPT_SUBSTRINGS = (
    "gold_operations",
    "reference_answer",
    "expected_tool_path",
    "gold_values",
    "known_data_risks",
)
