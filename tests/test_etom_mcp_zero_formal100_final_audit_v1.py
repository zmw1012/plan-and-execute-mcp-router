"""Deterministic regression tests for MCP-Zero × ETOM Formal-100 final audit v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.mcp_zero_etom.final_audit_v1.audit import (
    EXPECTED,
    PRIMARY_LABELS,
    assert_raw_untouched,
    compute_official_metrics,
    hash_raw_artifacts,
    load_json,
    load_jsonl,
    run_final_audit,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "etom_external_formal100_mcp_zero_v1"


pytestmark = pytest.mark.skipif(
    not (RUN_DIR / "predictions.jsonl").exists(),
    reason="Formal-100 raw run artifacts not present",
)


@pytest.fixture(scope="module")
def rows():
    return load_jsonl(RUN_DIR / "predictions.jsonl")


@pytest.fixture(scope="module")
def official(rows):
    return compute_official_metrics(rows)


def test_strict_49_100_not_confused_with_completed_only_49_84(official):
    strict = official["overall"]["strict_exact_route"]
    completed = official["overall"]["completed_only_exact_route"]
    assert strict["successes"] == 49
    assert strict["denominator"] == 100
    assert strict["rate"] == 0.49
    assert completed["successes"] == 49
    assert completed["denominator"] == 84
    assert completed["rate"] == pytest.approx(49 / 84)
    # Labels must remain distinct
    assert strict["label"] != completed["label"]
    assert "84" in completed["label"]
    assert "100" in strict["label"]


def test_per_level_strict_targets(official):
    expected = EXPECTED["strict_per_level"]
    for level, target in expected.items():
        got = official["per_level_strict_exact_route"][level]
        assert got["successes"] == target["successes"]
        assert got["strict_denominator"] == 20
        assert got["strict_rate"] == pytest.approx(target["rate"])


def test_l4_metric_labels_not_conflated(official):
    l4 = official["L4"]
    union = l4["official_union_node_set_em"]["strict"]
    efs = l4["efs_covering_node_set"]["strict"]
    dep = l4["cross_server_dependency_correctness"]["strict"]
    complete = l4["complete_route_success"]["strict"]
    # Values from sealed run
    assert union["successes"] == 1
    assert efs["successes"] == 3
    assert dep["successes"] == 2
    assert complete["successes"] == 2
    # Labels stay distinct and efs != dependency
    assert l4["efs_covering_node_set"]["label"] == "efs_covering_node_set"
    assert (
        l4["cross_server_dependency_correctness"]["label"]
        == "cross_server_dependency_correctness"
    )
    assert l4["label_guard"]["efs_covering_is_not_cross_server_dependency"] is True
    assert efs["successes"] != dep["successes"]
    assert l4["efs_covering_node_set"] is not l4["cross_server_dependency_correctness"]


def test_raw_grouped_results_pooled_is_completed_only_not_strict():
    grouped = load_json(RUN_DIR / "grouped_results.json")
    pooled = grouped["pooled_100"]["exact_match_rate"]
    assert pooled == pytest.approx(49 / 84)
    assert pooled != 0.49
    # Strict block already present in raw artifact
    assert grouped["overall_exact_route"]["rate"] == 0.49
    assert grouped["strict"]["L3"]["strict_exact_route_rate"] == 0.45


def test_raw_sha_unchanged_by_audit(tmp_path):
    before = hash_raw_artifacts(RUN_DIR)
    audit_dir = tmp_path / "final_audit_v1"
    docs_dir = tmp_path / "docs_mirror"
    result = run_final_audit(
        root=ROOT,
        run_dir=RUN_DIR,
        audit_dir=audit_dir,
        docs_mirror_dir=docs_dir,
    )
    after = hash_raw_artifacts(RUN_DIR)
    integrity = assert_raw_untouched(before, after)
    assert integrity["ok"] is True
    assert result["summary"]["raw_sha_unchanged"] is True
    sha_doc = load_json(audit_dir / "raw_artifact_sha256.json")
    assert sha_doc["integrity"]["ok"] is True
    assert (audit_dir / "final_report.md").exists()
    assert (docs_dir / "audit_summary.json").exists()
    # Ensure audit did not create/modify raw sibling files unexpectedly
    assert not (RUN_DIR / "predictions.jsonl.tmp").exists()


def test_taxonomy_exclusive_and_covers_100(rows):
    from hk_equity_mcp.mcp_zero_etom.final_audit_v1.audit import build_error_taxonomy

    tax = build_error_taxonomy(rows)
    assert tax["sum_primary"] == 100
    assert tax["primary_counts"]["exact_route_success"] == 49
    assert tax["primary_counts"]["max_turn_resource_limit"] == 16
    for row in tax["tasks"]:
        assert row["primary"] in PRIMARY_LABELS


def test_protocol_failures_all_max_turns(rows):
    from hk_equity_mcp.mcp_zero_etom.final_audit_v1.audit import audit_protocol_failures

    proto = audit_protocol_failures(rows)
    assert proto["completed"] == 84
    assert proto["failed"] == 16
    assert proto["all_failures_are_max_llm_turns"] is True
    assert proto["parse_repairs"] == 63
    assert proto["protocol_repairs"] == 47
    assert proto["task_protocol_success_rate"] == pytest.approx(0.84)
    assert proto["raw_content_preserved_tasks"] == 100
    assert len(proto["failed_tasks"]) == 16
