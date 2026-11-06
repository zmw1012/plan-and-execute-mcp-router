"""Offline checks for the ETOM-100 Operation-Visible final audit."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from hk_equity_mcp.etom.source import repo_root_from_here

ROOT = repo_root_from_here()
SCRIPT = ROOT / "scripts" / "audit_etom_operation_visible_formal100_final_v1.py"
ORIGINAL = ROOT / "outputs" / "etom_external_formal100_operation_visible_v1"
AUDIT = ORIGINAL / "final_audit_v1"
DOCS = ROOT / "docs" / "etom_operation_visible_formal100_final_audit_v1"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("etom_formal100_final_audit_v1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


def test_parser_v11_rules_are_general_and_task_agnostic(audit):
    plain = '{"tool_calls": [], "final_answer": {"infeasible": true}}'
    assert audit.parse_agent_json_v1(plain) == audit.parse_agent_json_v11(plain)
    fenced = '```json\n{"a": 1}\n```'
    assert audit.parse_agent_json_v1(fenced) == {"a": 1}
    assert audit.parse_agent_json_v11(fenced) == {"a": 1}

    prose = 'Note:\n{"tool_calls": [], "final_answer": {"infeasible": true,}}\n'
    with pytest.raises(Exception):
        audit.parse_agent_json_v1(prose)
    assert audit.parse_agent_json_v11(prose)["final_answer"]["infeasible"] is True

    smart = "{\u201ctool_calls\u201d: [], \u201cfinal_answer\u201d: null}"
    with pytest.raises(Exception):
        audit.parse_agent_json_v1(smart)
    assert audit.parse_agent_json_v11(smart)["final_answer"] is None

    bom = '\ufeff{"tool_calls": []}'
    assert audit.parse_agent_json_v11(bom) == {"tool_calls": []}
    with pytest.raises(Exception):
        audit.parse_agent_json_v11("[1, 2, 3]")

    assert "task_id" not in inspect.signature(audit.parse_agent_json_v11).parameters
    left = audit.parse_agent_json_v11(prose)
    right = audit.parse_agent_json_v11(prose)
    assert left == right


def test_auditor_source_does_not_call_live_clients():
    text = SCRIPT.read_text(encoding="utf-8")
    for banned in ("import requests", "import httpx", "import openai", "hk_equity_mcp.etom.llm", "urllib.request"):
        assert banned not in text


def test_final_audit_preserves_raw_predictions_and_terminal_counts(audit):
    predictions = ORIGINAL / "predictions.jsonl"
    before = audit.sha256_file(predictions)
    metadata_before = audit.sha256_file(ORIGINAL / "run_metadata.json")
    code = audit.run_audit(ROOT, ORIGINAL, AUDIT, DOCS)
    assert code == 0
    assert audit.sha256_file(predictions) == before
    assert audit.sha256_file(ORIGINAL / "run_metadata.json") == metadata_before

    summary = json.loads((DOCS / "error_summary.json").read_text(encoding="utf-8"))
    parser = json.loads((DOCS / "parser_comparison.json").read_text(encoding="utf-8"))
    metrics = json.loads((DOCS / "per_level_metrics.json").read_text(encoding="utf-8"))
    cost = json.loads((DOCS / "cost_summary_audited.json").read_text(encoding="utf-8"))
    taxonomy = [
        json.loads(line)
        for line in (DOCS / "error_taxonomy.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["exact_route_success"] == 66
    assert summary["tasks"] == 100
    assert len(taxonomy) == 100
    assert len({row["task_id"] for row in taxonomy}) == 100
    assert parser["per_task_score_changes"] == []
    assert parser["parse_error_ids_still_failed"] == [
        "L1-q591",
        "L2-q518",
        "L3-q195",
        "L4-q48",
        "L4-q71",
    ]
    assert parser["max_llm_turns_kept_failed"] == ["L2-q766", "L2-q720", "L4-q75"]
    assert metrics["pooled"]["strict_exact_route_count"] == 66
    assert cost["task_llm_calls"] == 434
    assert cost["cost_summary_llm_calls"] == 435
    assert cost["real_tool_executions"] == 0
    primaries = [row["primary_error"] for row in taxonomy]
    assert primaries.count(None) == 66
    assert all(isinstance(item, str) and item for item in primaries if item is not None)

    l4 = metrics["per_level"]["L4"]
    locked = metrics["l4_locked_strict_labels"]
    assert locked["official_union_node_set_em"] == "3/20 = 0.150"
    assert locked["efs_covering_node_set"] == "5/20 = 0.250"
    assert locked["complete_route_success"] == "5/20 = 0.250"
    assert locked["cross_server_dependency_correctness"] == "6/20 = 0.300"
    assert locked["mean_f1"] == 0.385
    assert l4["node_set_exact_match_strict"] == "3/20 = 0.150"
    assert l4["efs_covering_node_set_strict"] == "5/20 = 0.250"
    assert l4["complete_route_success_strict"] == "5/20 = 0.250"
    assert l4["cross_server_dependency_strict_text"] == "6/20 = 0.300"
    assert abs(float(l4["f1_strict"]) - 0.385) < 1e-3
    by_metric = {row["metric"]: row["strict"] for row in l4["table_rows"]}
    assert by_metric["Node Set Exact Match (official union)"] == "3/20 = 0.150"
    assert by_metric["EFS-covering node set"] == "5/20 = 0.250"
    assert by_metric["Complete route success"] == "5/20 = 0.250"
    assert by_metric["Cross-server dependency correctness"] == "6/20 = 0.300"
    assert by_metric["F1"] == "0.385"
    assert by_metric["EFS-covering node set"] != "6/20 = 0.300"
    report = (DOCS / "audit_report.md").read_text(encoding="utf-8")
    assert "| Official union node-set EM | 3/20 = 0.150 |" in report
    assert "| EFS-covering node-set | 5/20 = 0.250 |" in report
    assert "| Cross-server dependency correctness | 6/20 = 0.300 |" in report
    assert "0.300 is cross-server dependency correctness only" in report
    assert "EFS-covering node set | 6/20 = 0.300" not in report

    for name in (
        "error_taxonomy.jsonl",
        "error_summary.json",
        "per_level_metrics.json",
        "per_task_metrics_audited.csv",
        "retrieval_analysis_audited.json",
        "cost_summary_audited.json",
        "audit_report.md",
    ):
        assert (AUDIT / name).read_bytes() == (DOCS / name).read_bytes()
