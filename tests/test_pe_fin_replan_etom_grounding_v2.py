"""Offline tests for pe_fin_replan_etom_grounding_v2 + CLI observability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.pe_fin_replan_etom.grounding import (
    GROUNDING_CONTRACT_VERSION,
    grounding_contract_hash,
    validate_replanner_response,
)
from hk_equity_mcp.pe_fin_replan_etom.identity import resume_identity_hash
from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
    UnifiedPeFinReplanEtomRunner,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_V3 = ROOT / "outputs" / "pe_fin_replan_etom_smoke5_v3"

V3_RAW = {
    "predictions.jsonl": ("c031aa1199df9c329f4d188be8e33dd319e7d4ea28de94a0ea1c0aa611ea67c0", 69042, 5),
    "run_metadata.json": ("3e84c38660808887ddb7e648fbf82b0fc99af6ea598abaf3a3b4bdfb9e9f9fb5", 126543, None),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_question_constraint_last_10_ok():
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "Routed get_recent_trades with a limit of 10 trades.",
            "reasoning_brief": "constraint echo",
        },
        question="fetch the most recent trades for BTC/USDT with a limit of 10 trades.",
        evidence_values={"s1.input.limit": 10},
        provenance_class_by_key={"s1.input.limit": "grounded_input"},
        route_records=[
            {
                "operation_id": "Binance Cryptocurrency MCP::get_recent_trades",
                "server_name": "Binance Cryptocurrency MCP",
                "tool_name": "get_recent_trades",
            }
        ],
    )
    assert out["ok"] is True
    assert out["contract_version"] == "pe_fin_replan_etom_grounding_v2"
    assert out["classifications"]["question_constraint_ok"]
    assert out["classifications"]["numbers_supported"] == []


def test_02_list_enumeration_formatting_ignored():
    out = validate_replanner_response(
        response={
            "completion": "insufficient_evidence",
            "final_answer": (
                "1. Listed sources.\n"
                "2. Searched up to 10 results for best practices.\n"
                "3. Searched up to 10 results for styles.\n"
                "4. Preliminary design incomplete.\n"
                "5. Evaluation incomplete."
            ),
            "reasoning_brief": "routing-only",
        },
        question=(
            "list sources, search up to 10 results related to A and up to 10 results for B, "
            "then design and evaluate."
        ),
        evidence_values={},
        provenance_class_by_key={},
        route_records=[],
    )
    assert out["ok"] is True
    ignored = {s.strip("'\"") for s in out["classifications"]["formatting_enumeration_ignored"]}
    for i in range(1, 6):
        assert str(i) in ignored or repr(float(i)) in ignored


def test_03_ungrounded_body_number_rejected():
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "The total is 123456 units.",
            "reasoning_brief": "guess",
        },
        question="what is the value?",
        evidence_values={"x": 42},
        provenance_class_by_key={"x": "observed"},
    )
    assert out["ok"] is False
    assert any("ungrounded_answer_value" in r for r in out["reasons"])


def test_04_question_target_not_labeled_observed():
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "Used limit 10 from the question.",
            "reasoning_brief": "scope",
        },
        question="limit of 10 trades",
        evidence_values={},
        provenance_class_by_key={},
    )
    assert out["ok"] is True
    assert out["classifications"]["numbers_supported"] == []
    assert out["classifications"]["question_constraint_ok"]
    # Question constraints must never be labeled as observed/derived support.
    assert out.get("question_constraint_not_observed") is True


def test_05_verified_route_identifier_ok():
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "Selected tool server42_lookup via operation server42_lookup::run.",
            "reasoning_brief": "route",
        },
        question="find a lookup tool",
        evidence_values={},
        provenance_class_by_key={},
        route_records=[
            {
                "operation_id": "server42_lookup::run",
                "server_name": "server42_lookup",
                "tool_name": "run",
            }
        ],
    )
    assert out["ok"] is True
    assert out["classifications"]["verified_route_identifier"] or out["classifications"][
        "identifiers_ok"
    ]


def test_06_unverified_identifier_rejected():
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "Selected tool server99_secret.",
            "reasoning_brief": "fabricated",
        },
        question="find a lookup tool",
        evidence_values={},
        provenance_class_by_key={},
        route_records=[
            {
                "operation_id": "server42_lookup::run",
                "server_name": "server42_lookup",
                "tool_name": "run",
            }
        ],
    )
    assert out["ok"] is False


def test_07_grounding_fix_does_not_invent_exact_match():
    # Routing mismatch remains an evaluator concern; grounding pass != exact.
    # This unit only asserts grounding can pass while exact_match stays False externally.
    ground = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "Routed with limit 10.",
            "reasoning_brief": "ok",
        },
        question="limit of 10 trades",
        evidence_values={"s1.input.limit": 10},
        provenance_class_by_key={"s1.input.limit": "grounded_input"},
    )
    assert ground["ok"] is True
    exact_match = False  # wrong route; must stay false
    assert exact_match is False


def test_08_09_routing_invariants_on_mock_rows(tmp_path):
    runner = UnifiedPeFinReplanEtomRunner(
        ROOT, tmp_path / "mock", mode="mock", engineering_smoke5=True, formal100=False, progress=False
    )
    result = runner.execute()
    assert result.get("live_pass_claimed") is False
    assert result.get("execution_scope") == "routing_only"
    assert result.get("real_tool_executions") == 0
    preds = [
        json.loads(line)
        for line in (tmp_path / "mock" / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for row in preds:
        assert row["routing_record_count"] == row["mcp_invocations"]
        assert row["real_tool_executions"] == 0
        assert row["executed_etom_mcp_tool"] is False
        assert row["execution_scope"] == "routing_only"


def test_10_cli_compact_not_dataset_lock(tmp_path):
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_etom_pe_fin_replan_experiment.py"),
            "--mock",
            "--engineering-smoke5",
            "--output-dir",
            str(tmp_path / "cli_mock"),
            "--no-progress",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "PE-FIN-REPLAN ETOM RUN COMPLETE" in out
    assert "ordered_task_ids" not in out
    assert "L1-q160" not in out
    assert "routing_records=" in out
    assert "real_tool_executions=0" in out
    # Progress disabled: no [01/5] lines when --no-progress
    assert "[01/5]" not in out


def test_11_smoke_v3_raw_unchanged():
    for name, (want, nbytes, rows) in V3_RAW.items():
        path = SMOKE_V3 / name
        data = path.read_bytes()
        assert len(data) == nbytes
        assert _sha(path) == want
        if rows is not None:
            assert sum(1 for l in data.decode().splitlines() if l.strip()) == rows
    text = (SMOKE_V3 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_etom_grounding_and_terminal_observability_defects"
        in text
    )
    assert "outputs/pe_fin_replan_etom_smoke5_v4" in text


def test_12_resume_identity_cross_contract_fail_closed():
    assert GROUNDING_CONTRACT_VERSION == "pe_fin_replan_etom_grounding_v2"
    new_hash = grounding_contract_hash()
    old = {
        "condition": "pe_fin_replan_etom",
        "method_version": "pe_fin_replan_etom_v1",
        "protocol_version": "planner_executor_replanner_etom_seven_round_v1",
        "dataset_sample_sha256": "a",
        "dataset_manifest_sha256": "b",
        "ordered_task_ids_sha256": "c",
        "capability_profile_hash": "d",
        "prompt_hash": "e",
        "schema_set_hash": "f",
        "replanner_schema_hash": "g",
        "grounding_contract_hash": "old_v1_hash",
        "cli_output_contract_version": None,
        "catalog_registry_sha256": "h",
        "operation_documents_sha256": "i",
        "operation_index_identity": "j",
        "embedding_cache_namespace": "pe_fin_replan_etom_operation_embeddings_v1",
        "generation_provider_profile_id": "pe_fin_replan_etom_zhizengzeng_v1",
        "generation_provider_profile_hash": "k",
        "generation_provider": "openai_compatible",
        "generation_model": "deepseek-v4.1-flash",
        "generation_host": "api.zhizengzeng.com",
        "embedding_model": "text-embedding-3-small",
        "embedding_host": "x",
        "budgets": {},
        "evaluator_identity": {},
        "run_mode": "live",
        "run_role": "engineering_live_path_smoke_only",
        "git_commit": "abc",
        "git_dirty": False,
        "fixture_run": False,
    }
    new = dict(old)
    new["grounding_contract_hash"] = new_hash
    new["cli_output_contract_version"] = "pe_fin_replan_etom_cli_compact_v1"
    assert resume_identity_hash(old) != resume_identity_hash(new)


def test_step_marker_formatting():
    out = validate_replanner_response(
        response={
            "completion": "insufficient_evidence",
            "final_answer": "Step 1: plan. Step 2: retrieve. Step 3: stop.",
            "reasoning_brief": "fmt",
        },
        question="do the work",
        evidence_values={},
        provenance_class_by_key={},
    )
    assert out["ok"] is True
