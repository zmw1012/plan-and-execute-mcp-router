"""Offline tests for Formal-100 final audit v1 (no network / no rerun)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_gates import ENV_ETOM_ARTIFACT_ROOT, etom_artifact_root, require_paths

from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
    EtomPeFinReplanRunnerError,
    resolve_output_dir,
)
from hk_equity_mcp.pe_fin_replan_etom.formal100_final_audit_v1 import (
    AUDIT_VERSION,
    compute_three_lenses,
    freeze_raw_hashes,
    load_predictions,
    run_audit,
    verify_raw_unchanged,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = etom_artifact_root()  # Formal-100 raw historically at worktree root (incident)
AUDIT_DIR = etom_artifact_root() / "outputs" / "pe_fin_replan_etom_formal100_v1" / "final_audit_v1"

PRED_SHA = "ad8e845a286971a55c6b401127270829b3fcdc250c48f83911cb26b41377e149"
META_SHA = "335ad2cef27f214430f196714f9c96c514b1373bd6b91c413044d9be66a3b19d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def audit_result():
    require_paths("predictions.jsonl", "run_metadata.json", root=etom_artifact_root(), env_var=ENV_ETOM_ARTIFACT_ROOT)
    docs = ROOT / "docs" / "pe_fin_replan_etom_formal100_final_audit_v1"
    return run_audit(
        raw_dir=RAW_DIR,
        audit_dir=AUDIT_DIR,
        repo_root=ROOT,
        docs_dir=docs,
    )


def test_01_raw_present_and_hashes_stable():
    require_paths("predictions.jsonl", "run_metadata.json", root=etom_artifact_root(), env_var=ENV_ETOM_ARTIFACT_ROOT)
    assert (RAW_DIR / "predictions.jsonl").exists()
    assert (RAW_DIR / "run_metadata.json").exists()
    assert _sha(RAW_DIR / "predictions.jsonl") == PRED_SHA
    assert _sha(RAW_DIR / "run_metadata.json") == META_SHA
    freeze = freeze_raw_hashes(RAW_DIR)
    assert freeze["artifacts"]["predictions.jsonl"]["rows"] == 100
    verify = verify_raw_unchanged(freeze)
    assert verify["all_unchanged"] is True


def test_02_three_lenses_separators():
    require_paths("predictions.jsonl", "run_metadata.json", root=etom_artifact_root(), env_var=ENV_ETOM_ARTIFACT_ROOT)
    rows = load_predictions(RAW_DIR / "predictions.jsonl")
    assert len(rows) == 100
    lenses = compute_three_lenses(rows)
    a = lenses["all_task_route_prediction_score"]
    b = lenses["end_to_end_conservative_failures_as_zero"]
    c = lenses["completed_only_diagnostic"]
    assert a["exact_match_count"] == 45
    assert a["denominator"] == 100
    assert abs(a["exact_match_rate"] - 0.45) < 1e-9
    assert b["exact_match_count"] == 35
    assert abs(b["exact_match_rate"] - 0.35) < 1e-9
    assert c["denominator"] == 65
    assert c["exact_match_count"] == 35
    assert abs(c["exact_match_rate"] - (35 / 65)) < 1e-9
    assert c["not_formal_headline"] is True
    # failed exact=true retained in all-route, zeroed in failures-as-zero
    failed_exact = [r for r in rows if r.get("status") == "failed" and r.get("exact_match") is True]
    assert len(failed_exact) == 10
    assert lenses["reporting_policy"]["cross_condition_comparability_not_established"] is True


def test_03_per_level_failures_as_zero_denominators():
    require_paths("predictions.jsonl", "run_metadata.json", root=etom_artifact_root(), env_var=ENV_ETOM_ARTIFACT_ROOT)
    rows = load_predictions(RAW_DIR / "predictions.jsonl")
    b = compute_three_lenses(rows)["end_to_end_conservative_failures_as_zero"]
    for level in range(1, 6):
        entry = b["per_level"][f"L{level}"]
        assert entry["n"] == 20
    assert b["per_level"]["L1"]["exact_match_count"] == 15  # 75%
    assert b["per_level"]["L2"]["exact_match_count"] == 10
    assert b["per_level"]["L3"]["exact_match_count"] == 4
    assert b["per_level"]["L4"]["exact_match_count"] == 0
    assert b["per_level"]["L5"]["exact_match_count"] == 6


def test_04_routing_invariants_and_real_tools():
    require_paths("predictions.jsonl", "run_metadata.json", root=etom_artifact_root(), env_var=ENV_ETOM_ARTIFACT_ROOT)
    rows = load_predictions(RAW_DIR / "predictions.jsonl")
    assert all(int(r.get("real_tool_executions") or 0) == 0 for r in rows)
    assert all(
        int(r.get("routing_record_count") or 0) == int(r.get("mcp_invocations") or 0)
        for r in rows
    )
    assert all(r.get("executed_etom_mcp_tool") is False for r in rows)


def test_05_audit_run_and_pointer(audit_result):
    assert audit_result["ok"] is True
    assert audit_result["audit_dir"].endswith("final_audit_v1")
    assert (AUDIT_DIR / "final_report.json").exists()
    assert json.loads((AUDIT_DIR / "final_report.json").read_text())["audit_version"] == AUDIT_VERSION
    pointer = json.loads(
        (etom_artifact_root() / "outputs" / "pe_fin_replan_etom_formal100_v1" / "FORMAL_RUN_RAW_POINTER.json").read_text()
    )
    assert pointer["no_raw_mutation"] is True
    assert pointer["raw_artifacts"]["predictions.jsonl"]["sha256"] == PRED_SHA
    assert "hk-equity-mcp-agent-pe-fin-replan-etom/predictions.jsonl" in pointer["raw_artifacts"][
        "predictions.jsonl"
    ]["absolute_path"].replace("\\", "/")
    incident = json.loads((AUDIT_DIR / "raw_location_incident.json").read_text())
    assert incident["code_bug"]["confirmed"] is True
    assert incident["code_bug"]["fix_scope"] == "future_runs_only_no_raw_mutation"
    # raw still at root after audit
    assert _sha(RAW_DIR / "predictions.jsonl") == PRED_SHA
    assert _sha(RAW_DIR / "run_metadata.json") == META_SHA


def test_06_completeness_and_order_fail_closed(audit_result):
    comp = json.loads((AUDIT_DIR / "completeness.json").read_text())
    assert comp["pass"] is True
    assert comp["predictions_rows"] == 100
    assert comp["completed_equals_65"] is True
    assert comp["failed_equals_35"] is True
    assert comp["run_status"] == "completed"
    ident = json.loads((AUDIT_DIR / "identity_checks.json").read_text())
    assert ident["pass"] is True
    order_check = next(c for c in ident["checks"] if c["check"] == "order_matches_frozen_lock")
    assert order_check["ok"] is True


def test_07_empty_output_dir_refused_future_fix(tmp_path):
    require_paths("predictions.jsonl", "run_metadata.json", root=etom_artifact_root(), env_var=ENV_ETOM_ARTIFACT_ROOT)
    with pytest.raises(EtomPeFinReplanRunnerError):
        resolve_output_dir(ROOT, "")
    with pytest.raises(EtomPeFinReplanRunnerError):
        resolve_output_dir(ROOT, ".")
    with pytest.raises(EtomPeFinReplanRunnerError):
        resolve_output_dir(ROOT, str(ROOT))
    ok = resolve_output_dir(ROOT, tmp_path / "formal_out")
    assert ok == (tmp_path / "formal_out").resolve()
    # incident does not require Formal-100 rerun
    incident = json.loads((AUDIT_DIR / "raw_location_incident.json").read_text())
    assert "rerun" not in (incident.get("code_bug") or {}).get("fix_scope", "")


def test_08_docs_mirror_present(audit_result):
    docs = ROOT / "docs" / "pe_fin_replan_etom_formal100_final_audit_v1"
    assert (docs / "report_ready_summary.md").exists()
    assert (docs / "FORMAL_RUN_RAW_POINTER.json").exists()
    text = (docs / "report_ready_summary.md").read_text(encoding="utf-8")
    assert "Completed-only diagnostic (NOT overall)" in text
    assert "45/100" in text and "35/100" in text
