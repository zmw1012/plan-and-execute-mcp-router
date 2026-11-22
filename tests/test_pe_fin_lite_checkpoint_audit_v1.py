"""Offline Checkpoint Audit v1 tests (fake backend; no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_gates import hk_artifact_root, require_hk_artifact

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit import (
    AUDIT_VERSION,
    CHECKPOINT_IDS,
    CHECKPOINT_TRACE_CONTRACT_VERSION,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.evaluate import (
    eval_c01_planner,
    eval_c02_filter,
    eval_c03_embedding,
    eval_c04_describe,
    eval_c05_binder,
    eval_c06_arguments,
    eval_c07_mcp,
    eval_c08_normalize,
    eval_c09_evidence,
    eval_c10_fallback,
    eval_c11_compute,
    eval_c12_reason,
    eval_c13_final,
    eval_c14_evaluator,
    evaluate_task_checkpoints,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.engineering_pass import (
    assess_engineering_pass,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.runner import (
    collect_raw_hashes,
    run_checkpoint_audit,
)
from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
    enrich_retrieve_funnel_passive,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()


def _build_fixture():
    import importlib.util

    path = (
        ROOT
        / "tests"
        / "fixtures"
        / "pe_fin_lite_checkpoint_audit_v1"
        / "build_fixture_run.py"
    )
    spec = importlib.util.spec_from_file_location("build_fixture_run", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.build()
MANIFEST = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_diagnostic12_v1.json"
V3_CACHE = (
    ARTIFACT_ROOT
    / "outputs"
    / "pe_fin_lite_planner_protocol_diagnostic10_v3"
    / "planner_cache"
    / "planner_outputs.jsonl"
)
V3_SHA = "372e78041796cb12c38c4968c39ac285b6c9e2efb8a6ccbe9d49ab8d51e697ff"


def _row(**kwargs):
    base = {
        "task_id": "t1",
        "domain": "financial",
        "status": "completed",
        "predicted_status": "answered",
        "final_answer": "x",
        "answer_correct": False,
        "auto_grade_status": "graded",
        "auto_grade_reason": "unit",
        "manual_answer_correctness": "",
        "initial_plan": {
            "steps": [
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": "financial",
                    "objective": "q",
                    "required_outputs": ["revenue"],
                    "depends_on": [],
                }
            ]
        },
        "step_statuses": [],
        "retrieval_funnels": [],
        "invoked_operations": [],
        "invoke_records": [],
    }
    base.update(kwargs)
    return base


def test_v3_planner_outputs_sha_frozen():
    require_hk_artifact("outputs/pe_fin_lite_planner_protocol_diagnostic10_v3/planner_cache/planner_outputs.jsonl")
    assert V3_CACHE.exists()
    assert hashlib.sha256(V3_CACHE.read_bytes()).hexdigest() == V3_SHA


def test_diagnostic12_manifest_identity_and_coverage():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert man["manifest_id"] == "hk_pe_fin_lite_diagnostic12_v1"
    assert len(man["ordered_task_ids"]) == 12
    assert man["ordered_task_ids"] == [t["task_id"] for t in man["tasks"]]
    domains = {t["domain"] for t in man["tasks"]}
    assert {"financial", "market", "news", "cross_domain"} <= domains
    buckets = {t["selection_bucket"] for t in man["tasks"]}
    assert any("calculation" in b for b in buckets)
    assert any("refusal" in b for b in buckets)
    assert any("news" in b for b in buckets)
    # Pointer-only: does not embed gold answers
    blob = json.dumps(man)
    assert "gold_operations" not in blob
    assert "reference_answer" not in blob


def test_c01_pass_and_fail():
    ok = eval_c01_planner(
        _row(initial_plan={"steps": [{"id": "s1", "type": "retrieve", "depends_on": []}]})
    )
    assert ok["status"] == "pass"
    bad = eval_c01_planner(_row(initial_plan=None, status="failed"))
    assert bad["status"] == "fail"


def test_c02_filter_pass_and_blocked_leak_fail():
    funnel = {
        "hard_filtered": {
            "kept": ["op.a"],
            "rejected": [{"operation_id": "op.blocked", "reason": "blocked_or_rejected"}],
            "kept_count": 1,
            "rejected_count": 1,
        },
        "embedding_topk": [{"operation_id": "op.a", "score": 1}],
        "described_candidates": ["op.a"],
    }
    c02 = eval_c02_filter(_row(retrieval_funnels=[funnel]))
    assert c02["contract_status"] == "pass"
    assert c02["stage_outcome_status"] == "not_evaluated"
    assert c02["status"] == "not_evaluated"
    # N/A without retrieve
    assert (
        eval_c02_filter(_row(initial_plan={"steps": [{"id": "s1", "type": "compute"}]}, retrieval_funnels=[]))[
            "status"
        ]
        == "N/A"
    )


def test_c03_c04_topk_and_describe():
    funnel = {
        "search_query": "q",
        "hard_filtered": {"kept": ["a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
        "embedding_topk": [
            {"operation_id": f"op.{i}", "score": 1.0 - i * 0.1} for i in range(5)
        ],
        "described_candidates": ["op.0", "op.1", "op.2"],
    }
    c03 = eval_c03_embedding(_row(retrieval_funnels=[funnel]))
    assert c03["contract_status"] == "pass"
    assert c03["stage_outcome_status"] == "not_evaluated"
    assert c03["status"] == "not_evaluated"
    d = eval_c04_describe(_row(retrieval_funnels=[funnel]))
    assert d["status"] == "pass"
    assert d["detail"]["funnels"][0]["rank4_5_not_described"] == ["op.3", "op.4"]


def test_c05_binder_outside_described_fails():
    funnel = {
        "hard_filtered": {"kept": ["op.a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
        "embedding_topk": [{"operation_id": "op.a", "score": 1}],
        "described_candidates": ["op.a", "op.b", "op.c"],
        "primary": {"operation_id": "op.wrong", "arguments": {}},
        "binding": {"ok": True},
    }
    assert eval_c05_binder(_row(retrieval_funnels=[funnel]))["status"] == "fail"


def test_c06_missing_arguments_fail():
    funnel = {
        "hard_filtered": {"kept": ["op.a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
        "embedding_topk": [{"operation_id": "op.a", "score": 1}],
        "described_candidates": ["op.a"],
        "primary": {"operation_id": "op.a"},  # arguments missing
        "binding": {"ok": True},
    }
    assert eval_c06_arguments(_row(retrieval_funnels=[funnel]))["status"] == "fail"


def test_c07_no_invocation_not_tool_success():
    r = eval_c07_mcp(
        _row(
            invoked_operations=[],
            invoke_records=[],
            tool_execution_success=True,
            retrieval_funnels=[{"search_query": "q"}],
        )
    )
    assert r["status"] == "fail"
    assert r["detail"]["no_invocation"] is True


def test_c08_normalize_gateway_ok_normalize_fail():
    r = eval_c08_normalize(
        _row(
            step_statuses=[
                {
                    "id": "s1",
                    "type": "retrieve",
                    "gateway_success": True,
                    "normalization_success": False,
                    "missing_outputs": ["revenue"],
                }
            ],
            retrieval_funnels=[
                {
                    "hard_filtered": {"kept": [], "rejected": [], "kept_count": 0, "rejected_count": 0},
                    "invoke_gateway_success": {"op.a": True},
                    "checkpoint_passive": {
                        "normalize": {"normalization_success": False}
                    },
                }
            ],
        )
    )
    assert r["status"] == "fail"


def test_c09_scope_mismatch_and_false_accept():
    r = eval_c09_evidence(
        _row(
            step_statuses=[
                {
                    "id": "s1",
                    "type": "retrieve",
                    "step_completed": True,
                    "missing_outputs": ["revenue"],
                    "scope_match": False,
                }
            ]
        )
    )
    assert r["status"] == "fail"


def test_c10_fallback_recovery_and_na():
    funnel = {
        "hard_filtered": {"kept": ["a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
        "local_fallback": {
            "attempted": True,
            "reason": "required_outputs_missing",
            "merged": True,
            "success": True,
            "merge_conflicts": [],
        },
        "embedding_topk": [],
        "described_candidates": [],
    }
    assert eval_c10_fallback(_row(retrieval_funnels=[funnel]))["status"] == "pass"
    assert (
        eval_c10_fallback(
            _row(initial_plan={"steps": [{"id": "s1", "type": "compute"}]}, retrieval_funnels=[])
        )["status"]
        == "N/A"
    )


def test_c11_compute_missing_operand():
    r = eval_c11_compute(
        _row(
            initial_plan={
                "steps": [
                    {"id": "s1", "type": "retrieve", "depends_on": []},
                    {"id": "s2", "type": "compute", "depends_on": ["s1"]},
                ]
            },
            step_statuses=[
                {
                    "id": "s2",
                    "type": "compute",
                    "status": "failed",
                    "execution_ok": False,
                    "failure_reason": "missing_operand",
                }
            ],
            compute_traces=[{"error": "missing_operand"}],
        )
    )
    assert r["status"] == "fail"
    assert eval_c11_compute(_row())["status"] == "N/A_not_planned"


def test_c12_reason_unsupported_claim():
    r = eval_c12_reason(
        _row(
            initial_plan={
                "steps": [
                    {"id": "s1", "type": "retrieve", "depends_on": []},
                    {"id": "s2", "type": "reason", "depends_on": ["s1"]},
                ]
            },
            reason_artifacts=[{"summary": "x", "unsupported_claim": True}],
            step_statuses=[{"id": "s2", "type": "reason", "status": "completed"}],
        )
    )
    assert r["status"] == "fail"


def test_c13_final_and_c14_evaluator_separation():
    assert eval_c13_final(_row(final_answer="1", predicted_status="answered"))["status"] == "pass"
    # Evaluator present even when answer wrong
    e = eval_c14_evaluator(_row(answer_correct=False, auto_grade_reason="mismatch"))
    assert e["status"] == "pass"
    assert e["detail"]["human_fields_separate"] is True
    # Missing evaluator fields → fail observability
    bare = _row()
    bare.pop("answer_correct", None)
    bare.pop("auto_grade_status", None)
    bare.pop("auto_grade_reason", None)
    bare.pop("manual_answer_correctness", None)
    assert eval_c14_evaluator(bare)["status"] == "fail"


def test_primary_blocking_is_earliest_fail():
    row = _row(
        initial_plan=None,
        status="failed",
        retrieval_funnels=[],
        answer_correct=True,  # must NOT convert upstream fail to pass
    )
    out = evaluate_task_checkpoints(row)
    assert out["primary_blocking_checkpoint"] == "C01_planner"
    assert out["answer_correct"] is True


def test_passive_enrich_has_no_secrets():
    funnel = enrich_retrieve_funnel_passive(
        {
            "hard_filtered": {"kept": ["a"], "rejected": [], "kept_count": 1, "rejected_count": 0},
            "embedding_topk": [{"operation_id": "a", "score": 1}],
            "described_candidates": ["a"],
            "primary": {"operation_id": "a", "arguments": {"x": 1}},
            "api_key": "SECRET",
            "reasoning_content": "hidden",
        }
    )
    blob = json.dumps(funnel.get("checkpoint_passive"))
    assert "SECRET" not in blob
    assert "hidden" not in blob
    assert funnel["checkpoint_trace_contract_version"] == CHECKPOINT_TRACE_CONTRACT_VERSION


def test_fixture_run_audit_does_not_mutate_raw(tmp_path: Path):
    fixture = _build_fixture()
    # Copy fixture to tmp so we don't dirty repo if audit somehow wrote (it shouldn't).
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("predictions.jsonl", "run_metadata.json", "error_summary.json"):
        (run_dir / name).write_bytes((fixture / name).read_bytes())
    before = collect_raw_hashes(run_dir)
    out = tmp_path / "audit"
    result = run_checkpoint_audit(run_dir=run_dir, output_dir=out, expected_n=12)
    after = collect_raw_hashes(run_dir)
    assert before == after
    assert result["raw_hashes_unchanged"] is True
    assert (out / "audit_report.md").exists()
    assert (out / "per_task_checkpoints.jsonl").exists()
    assert (out / "checkpoint_funnel.json").exists()
    # Fixture has intentional failures → engineering may still pass completeness checks.
    assert result["summary"]["n_predictions"] == 12
    assert AUDIT_VERSION in (out / "audit_report.md").read_text()


def test_mock_dry_run_cannot_engineering_pass_live():
    eng = assess_engineering_pass(
        task_ids=[f"t{i}" for i in range(12)],
        expected_n=12,
        run_metadata={"run_status": "completed", "mock": True, "completed": True},
        per_task=[{"checkpoint_list": [], "primary_blocking_checkpoint": None} for _ in range(12)],
        raw_hashes_unchanged=True,
        mock=True,
    )
    assert eng["engineering_pass"] is False
    assert "mock_or_dry_run" in eng["reason"]


def test_all_checkpoint_ids_covered():
    assert len(CHECKPOINT_IDS) == 14
    row = _row(
        retrieval_funnels=[
            {
                "hard_filtered": {
                    "kept": ["op.a"],
                    "rejected": [],
                    "kept_count": 1,
                    "rejected_count": 0,
                },
                "embedding_topk": [{"operation_id": "op.a", "score": 1}],
                "described_candidates": ["op.a"],
                "primary": {"operation_id": "op.a", "arguments": {"x": 1}},
                "binding": {"ok": True},
                "invoked_operations": ["op.a"],
                "invoke_gateway_success": {"op.a": True},
            }
        ],
        invoked_operations=["op.a"],
        invoke_records=[{"operation_id": "op.a", "gateway_success": True}],
        step_statuses=[
            {
                "id": "s1",
                "type": "retrieve",
                "step_completed": True,
                "missing_outputs": [],
                "scope_match": True,
                "gateway_success": True,
                "normalization_success": True,
                "required_outputs_satisfied": True,
            }
        ],
    )
    out = evaluate_task_checkpoints(row)
    assert set(out["checkpoints"]) == set(CHECKPOINT_IDS)
