"""Build a synthetic Diagnostic-12-shaped fixture run for offline checkpoint audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "fixture_run"


def _base_row(task_id: str, domain: str, **extra):
    row = {
        "task_id": task_id,
        "id": task_id,
        "domain": domain,
        "status": "completed",
        "predicted_status": "answered",
        "final_answer": "1",
        "answer_correct": False,
        "auto_grade_status": "graded",
        "auto_grade_reason": "fixture",
        "manual_answer_correctness": "",
        "llm_calls": 2,
        "tokens": {"total_tokens": 100},
        "real_mcp_execution_count": 1,
        "embedding_calls": 1,
        "parse_repair_count": 0,
        "initial_plan": {
            "steps": [
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": domain if domain != "cross_domain" else "financial",
                    "objective": "retrieve",
                    "required_outputs": ["revenue"],
                    "depends_on": [],
                }
            ]
        },
        "step_statuses": [
            {
                "id": "s1",
                "type": "retrieve",
                "status": "completed",
                "execution_ok": True,
                "gateway_success": True,
                "normalization_success": True,
                "required_outputs_satisfied": True,
                "missing_outputs": [],
                "scope_match": True,
                "step_completed": True,
            }
        ],
        "retrieval_funnels": [
            {
                "step_id": "s1",
                "search_query": "financial | retrieve",
                "hard_filtered": {
                    "kept": ["op.a", "op.b", "op.c", "op.d", "op.e"],
                    "rejected": [{"operation_id": "op.blocked", "reason": "blocked_or_rejected"}],
                    "kept_count": 5,
                    "rejected_count": 1,
                },
                "embedding_topk": [
                    {"operation_id": "op.a", "score": 0.9},
                    {"operation_id": "op.b", "score": 0.8},
                    {"operation_id": "op.c", "score": 0.7},
                    {"operation_id": "op.d", "score": 0.6},
                    {"operation_id": "op.e", "score": 0.5},
                ],
                "described_candidates": ["op.a", "op.b", "op.c"],
                "primary": {"operation_id": "op.a", "arguments": {"stock_code": "0700"}},
                "fallback": {"operation_id": "op.b", "arguments": {"stock_code": "0700"}},
                "invoked_operations": ["op.a"],
                "invoke_gateway_success": {"op.a": True},
                "binding": {"ok": True, "repair_count": 0},
                "grounding": {"stock_code": "0700"},
            }
        ],
        "invoked_operations": ["op.a"],
        "successful_invoked_operations": ["op.a"],
        "invoke_records": [
            {
                "operation_id": "op.a",
                "gateway_success": True,
                "success": True,
                "step_completed": True,
            }
        ],
        "all_tool_invocations_successful": True,
        "failed_invocation_count": 0,
        "successful_invocation_count": 1,
        "fallback_used_count": 0,
        "local_fallback_count": 0,
        "compute_traces": [],
        "reason_artifacts": [],
        "final_payload": {"status": "answered", "final_answer": "1", "evidence_ids": ["e1"]},
    }
    row.update(extra)
    return row


def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    # 12 tasks covering domains; most pass-shaped. Inject targeted failures for unit docs.
    ids = [
        ("fin_cf_003", "financial"),
        ("fin_cf_011", "financial"),
        ("fin_cf_020", "financial"),
        ("fin_cf_023", "financial"),
        ("fin_ht_016", "financial"),
        ("xd_dev_002", "cross_domain"),
        ("xd_ht_007", "cross_domain"),
        ("xd_ht_014", "cross_domain"),
        ("mkt_dev_014", "market"),
        ("fin_cf_024", "financial"),
        ("mkt_dev_001", "market"),
        ("news_ht_004", "news"),
    ]
    rows = [_base_row(tid, dom) for tid, dom in ids]

    # C05 fail: binder selected outside described
    rows[0]["retrieval_funnels"][0]["primary"] = {
        "operation_id": "op.outside",
        "arguments": {"stock_code": "0700"},
    }
    rows[0]["retrieval_funnels"][0]["selected_outside_described"] = True

    # C07 fail: no invocation
    rows[1]["invoked_operations"] = []
    rows[1]["invoke_records"] = []
    rows[1]["successful_invocation_count"] = 0
    rows[1]["real_mcp_execution_count"] = 0
    rows[1]["retrieval_funnels"][0]["invoked_operations"] = []
    rows[1]["tool_execution_success"] = True  # incorrect claim

    # C08/C09: gateway success but missing outputs / normalize fail
    rows[2]["step_statuses"][0].update(
        {
            "normalization_success": False,
            "missing_outputs": ["revenue"],
            "required_outputs_satisfied": False,
            "step_completed": False,
            "status": "failed",
        }
    )
    rows[2]["retrieval_funnels"][0]["invoke_gateway_success"] = {"op.a": True}

    # C11 compute applicable
    rows[3]["initial_plan"]["steps"].append(
        {
            "id": "s2",
            "type": "compute",
            "domain": "compute",
            "objective": "ratio",
            "required_outputs": [],
            "depends_on": ["s1"],
            "compute_spec": {
                "expression": "(a-b)/b",
                "operand_refs": {"a": "s1.revenue", "b": "s1.basic_eps"},
            },
        }
    )
    rows[3]["step_statuses"].append(
        {
            "id": "s2",
            "type": "compute",
            "status": "failed",
            "execution_ok": False,
            "failure_reason": "missing_operand",
        }
    )
    rows[3]["compute_traces"] = [{"error": "missing_operand", "expression": "(a-b)/b"}]

    # C12 reason
    rows[4]["initial_plan"]["steps"].append(
        {
            "id": "s2",
            "type": "reason",
            "domain": "general",
            "objective": "summarize",
            "required_outputs": [],
            "depends_on": ["s1"],
        }
    )
    rows[4]["step_statuses"].append(
        {"id": "s2", "type": "reason", "status": "completed", "execution_ok": True}
    )
    rows[4]["reason_artifacts"] = [
        {"summary": "ok", "unsupported_claim": True, "gaps": ["missing revenue"]}
    ]

    # C10 refusal task
    rows[8]["answerability_class"] = "should_refuse_or_clarify"
    rows[8]["predicted_status"] = "unavailable"
    rows[8]["final_payload"] = {"status": "unavailable", "final_answer": ""}

    # C14: final looks answered but evaluator fields present with False
    rows[10]["answer_correct"] = False
    rows[10]["auto_grade_reason"] = "numeric_mismatch_fixture"

    pred = OUT / "predictions.jsonl"
    with pred.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "run_status": "completed",
        "mock": False,
        "dry_run": False,
        "condition": "pe_fin_lite",
        "architecture": "plan_first_step_level_retrieval_v1",
        "checkpoint_trace_contract_version": "pe_fin_lite_checkpoint_trace_v1",
        "n_tasks": 12,
        "completed": True,
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "error_summary.json").write_text("{}\n", encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
