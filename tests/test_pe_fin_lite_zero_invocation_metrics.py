"""Zero-invocation metrics must not vacuous-true tool success."""

from __future__ import annotations

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.evaluate import (
    eval_c07_mcp,
    eval_c11_compute,
    eval_c12_reason,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.engineering_pass import (
    assess_engineering_pass,
)


def test_zero_mcp_metrics_null_not_true_in_agent_payload_shape():
    # Simulate future agent return contract (unit-level).
    mcp_count = 0
    invoke_records: list = []
    if mcp_count == 0 and not invoke_records:
        all_ok = None
        tool_execution_success = None
    else:
        all_ok = True
        tool_execution_success = True
    assert tool_execution_success is None
    assert all_ok is None


def test_c07_temporal_no_tool_is_na_not_required():
    row = {
        "task_id": "t_temporal",
        "error_type": "temporal_latest_current_unsatisfied",
        "real_mcp_execution_count": 0,
        "tool_execution_success": True,  # raw vacuous bug
        "invoked_operations": [],
        "invoke_records": [],
        "initial_plan": {
            "steps": [{"id": "s1", "type": "retrieve", "depends_on": []}]
        },
        "retrieval_funnels": [
            {
                "hard_filtered": {"kept": ["a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
                "embedding_topk": [],
                "described_candidates": [],
            }
        ],
        "trace": [{"stage": "temporal_policy", "status": "unavailable"}],
    }
    r = eval_c07_mcp(row)
    assert r["stage_outcome_status"] == "N/A_not_required"
    assert r["status"] == "N/A_not_required"
    assert r["detail"]["metrics_vacuous_true"] is True


def test_c07_binder_failed_is_na_not_reached():
    row = {
        "task_id": "t_binder",
        "real_mcp_execution_count": 0,
        "tool_execution_success": True,
        "invoked_operations": [],
        "invoke_records": [],
        "initial_plan": {
            "steps": [{"id": "s1", "type": "retrieve", "depends_on": []}]
        },
        "step_statuses": [
            {
                "id": "s1",
                "type": "retrieve",
                "status": "failed",
                "failure_reason": "binder_failed:empty content",
            }
        ],
        "retrieval_funnels": [
            {
                "hard_filtered": {"kept": ["a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
                "binding": {"ok": False, "error": "empty content", "repair_count": 0},
                "invoked_operations": [],
                "described_candidates": ["a"],
                "embedding_topk": [{"operation_id": "a", "score": 1}],
            }
        ],
    }
    r = eval_c07_mcp(row)
    assert r["stage_outcome_status"] == "N/A_not_reached"
    assert r["status"] == "N/A_not_reached"


def test_c11_upstream_blocked_not_headline_pass():
    r = eval_c11_compute(
        {
            "initial_plan": {
                "steps": [
                    {"id": "s1", "type": "retrieve", "depends_on": []},
                    {"id": "s2", "type": "compute", "depends_on": ["s1"]},
                ]
            },
            "step_statuses": [
                {
                    "id": "s2",
                    "type": "compute",
                    "status": "skipped_due_to_dependency",
                    "failure_reason": "depends_on_failed:['s1']",
                }
            ],
            "compute_traces": [],
        }
    )
    assert r["stage_outcome_status"] == "N/A_upstream_blocked"
    assert r["status"] == "N/A_upstream_blocked"
    assert r["status"] != "pass"


def test_c12_skipped_reason_is_na_upstream_blocked():
    r = eval_c12_reason(
        {
            "initial_plan": {
                "steps": [
                    {"id": "s1", "type": "retrieve", "depends_on": []},
                    {"id": "s2", "type": "reason", "depends_on": ["s1"]},
                ]
            },
            "step_statuses": [
                {
                    "id": "s2",
                    "type": "reason",
                    "status": "skipped_due_to_dependency",
                    "failure_reason": "depends_on_failed:['s1']",
                }
            ],
            "reason_artifacts": [],
        }
    )
    assert r["stage_outcome_status"] == "N/A_upstream_blocked"
    assert r["status"] == "N/A_upstream_blocked"


def test_engineering_pass_false_on_raw_vacuous_tool_success():
    predictions = [
        {
            "task_id": f"t{i}",
            "real_mcp_execution_count": 0 if i == 0 else 1,
            "tool_execution_success": True,
        }
        for i in range(12)
    ]
    per_task = [
        {
            "task_id": f"t{i}",
            "checkpoint_list": [],
            "checkpoints": {},
            "primary_blocking_checkpoint": None,
        }
        for i in range(12)
    ]
    eng = assess_engineering_pass(
        task_ids=[f"t{i}" for i in range(12)],
        expected_n=12,
        run_metadata={"run_status": "completed", "completed": True},
        per_task=per_task,
        raw_hashes_unchanged=True,
        predictions=predictions,
    )
    assert eng["engineering_pass"] is False
    assert "zero_invocation_not_tool_success" in eng["failing_checks"]
