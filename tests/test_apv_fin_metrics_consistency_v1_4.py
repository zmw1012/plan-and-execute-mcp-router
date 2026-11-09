"""Metrics consistency: invocation vs evidence satisfaction; aggregate/budget counters."""

from __future__ import annotations

from hk_equity_mcp.apv_fin.agent import _invoke_counts, _tool_execution_success
from hk_equity_mcp.experiment.unified_apv_fin_metrics import (
    aggregate_apv_fin_metrics,
    strict_answer_success,
    tool_execution_success_flag,
)


def test_zero_invoke_null_semantics():
    trace = [
        {"type": "step_execution", "result": {"ok": False, "error": "binding_expand_search_exhausted"}},
        {"type": "tool_result", "name": "find_operations", "success": True},
    ]
    assert _invoke_counts(trace) == (0, 0)
    assert _tool_execution_success(trace) is None
    row = {
        "status": "failed",
        "all_tool_invocations_successful": None,
        "successful_invocation_count": 0,
        "failed_invocation_count": 0,
        "real_mcp_execution_count": 0,
        "answer_correct": True,
        "auto_grade_status": "graded",
    }
    assert tool_execution_success_flag(row) is None
    assert strict_answer_success(row) is False


def test_successful_operation_list_uses_mcp_success_not_step_ok():
    # Simulate agent accounting: MCP success even when required outputs missing.
    successful_invoked = []
    out = {
        "invoke": {"success": True, "operation_id": "aktools.stock_hk_hist"},
        "chosen_operation_id": "aktools.stock_hk_hist",
        "ok": False,
        "mcp_execution_success": True,
    }
    if out.get("invoke") is not None:
        oid = out.get("chosen_operation_id")
        if out.get("mcp_execution_success") and oid:
            successful_invoked.append(oid)
    assert successful_invoked == ["aktools.stock_hk_hist"]


def test_aggregate_reads_budget_counters_when_top_level_missing():
    rows = [
        {
            "status": "failed",
            "llm_calls": 7,
            "real_mcp_execution_count": 1,
            "tokens": {"total_tokens": 1000},
            "budget": {
                "deterministic_step_transitions": 2,
                "dependency_auto_expansion_count": 1,
                "schema_description_expansion_count": 1,
                "repeated_signature_block_count": 1,
                "raw_observation_chars": 10000,
                "projected_observation_chars": 200,
                "compression_ratio": 0.02,
                "deterministic_binding_count": 1,
                "llm_binding_count": 2,
                "capability_injected_candidate_count": 3,
                "capability_filter_candidate_count": 5,
                "output_adapter_success_count": 1,
                "required_output_missing_count": 1,
                "deterministic_compute_count": 0,
                "exception_replan_calls": 2,
                "pagination_count": 1,
                "rows_scanned": 50,
            },
            "monitor_decisions": [{}, {}],
            "plan_state": {"steps": {}},
        }
    ]
    agg = aggregate_apv_fin_metrics(rows)
    assert agg["deterministic_step_transitions_mean"] == 2.0
    assert agg["dependency_auto_expansion_count_sum"] == 1
    assert agg["schema_description_expansion_count_sum"] == 1
    assert agg["repeated_signature_block_count_sum"] == 1
    assert agg["raw_observation_chars_mean"] == 10000.0
    assert agg["projected_observation_chars_mean"] == 200.0
    assert agg["deterministic_binding_count_sum"] == 1
    assert agg["llm_binding_count_sum"] == 2
    assert agg["capability_injected_candidate_count_sum"] == 3
    assert agg["capability_filter_candidate_count_sum"] == 5
    assert agg["output_adapter_success_count_sum"] == 1
    assert agg["required_output_missing_count_sum"] == 1
    assert agg["exception_replan_calls_sum"] == 2
    assert agg["pagination_count_sum"] == 1
    assert agg["rows_scanned_sum"] == 50


def test_invoke_count_matches_trace():
    trace = [
        {
            "type": "step_execution",
            "result": {"invoke": {"success": True, "operation_id": "a"}},
        },
        {
            "type": "step_execution",
            "result": {"invoke": {"success": False, "operation_id": "b"}},
        },
        {"type": "step_execution", "result": {"ok": False}},
    ]
    ok, fail = _invoke_counts(trace)
    assert ok == 1 and fail == 1
    assert _tool_execution_success(trace) is False
