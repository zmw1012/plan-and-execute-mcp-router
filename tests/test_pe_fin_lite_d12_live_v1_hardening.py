"""Diagnostic-12 live v1 offline hardening regressions (mock/dry-run only)."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.disclosure_taxonomy import (
    classify_disclosure_stage_failure,
    classify_task_disclosure_failures,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.evaluate import (
    eval_c02_filter,
    eval_c03_embedding,
    eval_c08_normalize,
    eval_c09_evidence,
    eval_c11_compute,
    eval_c12_reason,
    eval_c13_final,
    eval_c14_evaluator,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.runner import (
    collect_raw_hashes,
    run_checkpoint_audit,
)
from hk_equity_mcp.pe_fin_lite import PE_FIN_LITE_METHOD_VERSION
from hk_equity_mcp.pe_fin_lite.calculator import (
    CalculatorError,
    SAFE_FUNCTION_NAMES,
    safe_eval_expression,
)
from hk_equity_mcp.pe_fin_lite.evidence_provenance import (
    grounded_context_from_arguments,
    merge_field_maps,
    observed_field_map,
)
from hk_equity_mcp.pe_fin_lite.envelope_projection import _resolve_required
from hk_equity_mcp.pe_fin_lite.executor import execute_reason_step
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite.issuer_code import (
    canonicalize_hk_issuer_code,
    issuer_codes_equivalent,
)
from hk_equity_mcp.pe_fin_lite.temporal_policy import (
    assess_latest_current_refusal,
    question_requests_latest_current,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_V1 = ROOT / "outputs" / "pe_fin_lite_diagnostic12_checkpoint_live_v1"
RAW_SHA = {
    "predictions.jsonl": "a84762f8d5123db68eaf0445794adfb1c8bced90cbc4b61e2bdddd3b39683131",
    "run_metadata.json": "14a2f5531a1a04e211c3e4fa4b2ef127eae2d1ec5a967c7b39cd5407e2238e61",
}


def _src_blob() -> str:
    paths = list((ROOT / "src" / "hk_equity_mcp" / "pe_fin_lite").rglob("*.py"))
    paths += list(
        (ROOT / "src" / "hk_equity_mcp" / "experiment" / "pe_fin_lite_checkpoint_audit").rglob(
            "*.py"
        )
    )
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(paths))


def test_method_version_bumped_for_runtime_hardening():
    assert PE_FIN_LITE_METHOD_VERSION == "pe_fin_lite_v1.1"
    cfg = yaml.safe_load(
        (ROOT / "configs" / "hk_unified_pe_fin_lite_v1.yaml").read_text(encoding="utf-8")
    )
    assert cfg["method_version"] == PE_FIN_LITE_METHOD_VERSION


def test_issuer_code_canonicalization_and_rejects_statement_label():
    assert canonicalize_hk_issuer_code("700") == "00700"
    assert canonicalize_hk_issuer_code("0700.HK") == "00700"
    assert canonicalize_hk_issuer_code("00005") == "00005"
    assert canonicalize_hk_issuer_code("利润表") is None
    assert canonicalize_hk_issuer_code("资产负债表") is None
    assert canonicalize_hk_issuer_code("income statement") is None
    assert issuer_codes_equivalent("700", "00700.HK")
    assert not issuer_codes_equivalent("利润表", "00005")


def test_issuer_code_grounded_input_prefers_stock_over_statement_symbol():
    fmap = grounded_context_from_arguments(
        arguments={"stock": "00005", "symbol": "利润表", "indicator": "年度"},
        step={},
        question_grounding={"stock_code": "00005"},
    )
    assert fmap["issuer_code"]["value"] == "00005"
    assert "利润表" not in json.dumps(fmap, ensure_ascii=False)


def test_issuer_code_merge_drops_statement_pollution():
    polluted = {
        "issuer_code": {
            "value": "利润表",
            "provenance_class": "grounded_input",
            "source": "validated_argument_or_question",
            "ref": "symbol",
        }
    }
    good = observed_field_map({"issuer_code": "00005"})
    merged, conflicts = merge_field_maps(polluted, good)
    assert merged["issuer_code"]["value"] == "00005"
    assert not any(c.get("field") == "issuer_code" for c in conflicts)


def test_envelope_resolve_rejects_statement_as_issuer_code():
    values, missing = _resolve_required(
        {"symbol": "利润表", "SECURITY_CODE": "09988", "name": "Alibaba"},
        ["issuer_code", "company_name"],
    )
    assert values["issuer_code"] == "09988"
    assert "issuer_code" not in missing


def test_safe_compute_whitelist_round_half_up_and_abs():
    assert SAFE_FUNCTION_NAMES == frozenset({"abs", "min", "max", "round_half_up"})
    assert safe_eval_expression("round_half_up(close,2)", {"close": 475.204}) == 475.20
    assert safe_eval_expression("abs(apm-reported)", {"apm": 10.0, "reported": 12.5}) == 2.5
    with pytest.raises(CalculatorError):
        safe_eval_expression("eval('1')", {})
    with pytest.raises(CalculatorError):
        safe_eval_expression("__import__('os').system('x')", {})
    with pytest.raises(CalculatorError):
        safe_eval_expression("open('/etc/passwd')", {})
    with pytest.raises(CalculatorError):
        safe_eval_expression("math.sqrt(a)", {"a": 4})


def test_reason_parse_failure_and_runtime_recovery():
    store = EvidenceStore()
    llm = MagicMock()
    llm.complete.side_effect = [
        {"content": "NOT_JSON{{{", "usage": {"total_tokens": 1}},
        {"content": "still broken", "usage": {"total_tokens": 1}},
    ]
    out = execute_reason_step(
        step={"id": "s3", "type": "reason", "objective": "summarize", "depends_on": []},
        store=store,
        llm=llm,
        question="q",
    )
    assert out["reason_parse_failed"] is True
    assert out["runtime_recovered"] is True
    assert out["step_result"]["status"] == "failed"
    assert out["repair_count"] == 1
    assert llm.complete.call_count == 2


def test_temporal_latest_current_refuses_historical_substitution():
    assert question_requests_latest_current(
        "What is Tencent's latest / current / today's closing price?"
    )
    refusal = assess_latest_current_refusal(
        question="What is Tencent's latest / current / today's closing price?",
        evidence_store={
            "items": [
                {
                    "values": {"close": 430.2, "trade_date": "2026-07-02", "issuer_code": "00700"}
                }
            ]
        },
    )
    assert refusal and refusal["should_refuse"] is True
    assert "temporal_latest_current" in refusal["reason"]
    assert assess_latest_current_refusal(
        question="What was the close on 2026-07-31?",
        evidence_store={"items": [{"values": {"close": 1}}]},
    ) is None


def test_dual_layer_c08_c09_c11_c12_c13_c14():
    # C08: null normalization is unknown, not auto-pass
    c08 = eval_c08_normalize(
        {
            "initial_plan": {"steps": [{"id": "s1", "type": "retrieve", "depends_on": []}]},
            "step_statuses": [
                {"id": "s1", "type": "retrieve", "gateway_success": True, "normalization_success": None}
            ],
            "retrieval_funnels": [{"hard_filtered": {"kept": [], "rejected": [], "kept_count": 0, "rejected_count": 0}}],
        }
    )
    assert c08["stage_outcome_status"] == "unknown"
    assert c08["detail"].get("reason") == "observability_missing" or "null_normalization" in json.dumps(
        c08
    )

    # C09: correctly rejected incomplete → logic pass, retrieve incomplete
    c09 = eval_c09_evidence(
        {
            "initial_plan": {"steps": [{"id": "s1", "type": "retrieve", "depends_on": []}]},
            "step_statuses": [
                {
                    "id": "s1",
                    "type": "retrieve",
                    "step_completed": False,
                    "missing_outputs": ["snippet"],
                    "scope_match": True,
                }
            ],
            "retrieval_funnels": [],
        }
    )
    assert c09["detail"]["acceptance_logic_pass"] is True
    assert c09["detail"]["retrieve_outcome_complete"] is False
    assert c09["stage_outcome_status"] == "fail"
    assert c09["contract_status"] == "pass"

    # C11 upstream blocked
    c11 = eval_c11_compute(
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
            "compute_traces": [{"error": "skipped_due_to_dependency:s1"}],
        }
    )
    assert c11["stage_outcome_status"] == "N/A_upstream_blocked"
    assert c11["status"] == "N/A_upstream_blocked"
    assert c11["downstream_task_effect"] == "N/A_upstream_blocked"

    # C12 parse fail + recovered ≠ stage pass
    c12 = eval_c12_reason(
        {
            "initial_plan": {
                "steps": [
                    {"id": "s1", "type": "retrieve", "depends_on": []},
                    {"id": "s2", "type": "reason", "depends_on": ["s1"]},
                ]
            },
            "reason_artifacts": [
                {
                    "summary": "fallback",
                    "gaps": ["reason_parse_failed"],
                    "reason_parse_failed": True,
                    "runtime_recovered": True,
                }
            ],
            "step_statuses": [
                {
                    "id": "s2",
                    "type": "reason",
                    "status": "failed",
                    "reason_parse_failed": True,
                    "runtime_recovered": True,
                }
            ],
        }
    )
    assert c12["stage_outcome_status"] == "fail"
    assert c12["detail"]["runtime_recovered"] is True

    # C13 schema vs semantic
    c13 = eval_c13_final(
        {
            "final_answer": "x",
            "predicted_status": "answered",
            "status": "full",
            "answer_correct": False,
        }
    )
    assert c13["contract_status"] == "pass"
    assert c13["downstream_task_effect"] == "schema_valid_not_necessarily_correct"

    # C14 evaluator executed ≠ answer_correct
    c14 = eval_c14_evaluator(
        {
            "answer_correct": False,
            "auto_grade_status": "graded",
            "auto_grade_reason": "numeric_mismatch",
        }
    )
    assert c14["contract_status"] == "pass"
    assert c14["downstream_task_effect"] == "answer_incorrect_or_ungraded"


def test_c02_quality_not_evaluated_and_c03_retrieval_separated():
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
    row = {
        "initial_plan": {"steps": [{"id": "s1", "type": "retrieve", "depends_on": []}]},
        "retrieval_funnels": [funnel],
    }
    c02 = eval_c02_filter(row)
    assert c02["contract_status"] == "pass"
    assert c02["stage_outcome_status"] == "not_evaluated"
    c03 = eval_c03_embedding(row)
    assert c03["contract_status"] == "pass"
    assert c03["stage_outcome_status"] == "not_evaluated"
    assert "Recall@K" in json.dumps(c03["detail"]) or "all155" in json.dumps(c03["detail"])


def test_disclosure_failure_taxonomy_offline():
    assert (
        classify_disclosure_stage_failure(
            stage="download",
            gateway_success=False,
            error="Download budget exhausted (max 1 announcement per adapter session).",
        )["class"]
        == "envelope_adapter_failure"
    )
    assert (
        classify_disclosure_stage_failure(
            stage="list",
            gateway_success=False,
            error="No announcements found for 00005",
            n_list_rows=0,
        )["class"]
        == "no_matching_result"
    )
    row = {
        "task_id": "fin_ht_016",
        "retrieval_funnels": [
            {
                "disclosure_composite": {
                    "trace": [
                        {
                            "stage": "list",
                            "success": False,
                            "gateway_success": False,
                            "n_list_rows": 0,
                            "error": "No announcements found for 00005",
                        }
                    ]
                }
            }
        ],
    }
    out = classify_task_disclosure_failures(row)
    assert out["primary_class"] == "no_matching_result"


def test_live_v1_raw_hash_immutability():
    assert LIVE_V1.exists()
    hashes = collect_raw_hashes(LIVE_V1)
    assert hashes["predictions.jsonl"]["sha256"] == RAW_SHA["predictions.jsonl"]
    assert hashes["run_metadata.json"]["sha256"] == RAW_SHA["run_metadata.json"]
    # Recompute direct
    assert (
        hashlib.sha256((LIVE_V1 / "predictions.jsonl").read_bytes()).hexdigest()
        == RAW_SHA["predictions.jsonl"]
    )


def test_offline_checkpoint_audit_replay_does_not_mutate_raw(tmp_path: Path):
    before = collect_raw_hashes(LIVE_V1)
    out = tmp_path / "final_checkpoint_audit_v1_replay"
    result = run_checkpoint_audit(run_dir=LIVE_V1, output_dir=out, expected_n=12)
    after = collect_raw_hashes(LIVE_V1)
    assert before == after
    assert result["raw_hashes_unchanged"] is True
    assert result["engineering_pass"] is True
    assert result["summary"]["explicitly_not_formal_accuracy"] is True
    # Dual-layer fields present
    line = (out / "per_task_checkpoints.jsonl").read_text(encoding="utf-8").splitlines()[0]
    sample = json.loads(line)
    c08 = sample["checkpoints"]["C08_normalize"]
    assert "contract_status" in c08 and "stage_outcome_status" in c08


def test_no_task_id_branching_and_no_gold_leakage_in_hardening_modules():
    blob = _src_blob().lower()
    forbidden_ids = [
        "fin_cf_011",
        "fin_cf_023",
        "xd_ht_014",
        "xd_dev_002",
        "mkt_dev_014",
        "mkt_dev_001",
        "fin_cf_020",
        "fin_ht_016",
        "fin_cf_024",
    ]
    for tid in forbidden_ids:
        assert tid not in blob, f"task-id branch leak: {tid}"
    # Gold numeric patches forbidden (ban-list string "reference_answer" may appear as a reject token).
    assert "1.21" not in blob
    assert "13123569" not in blob
    assert "if task_id" not in blob
    assert 'task_id ==' not in blob
    assert 'task_id==' not in blob


def test_compute_and_reason_source_has_no_eval_exec():
    calc = inspect.getsource(
        __import__(
            "hk_equity_mcp.pe_fin_lite.calculator", fromlist=["safe_eval_expression"]
        )
    )
    assert "eval(" not in calc.replace("safe_eval_expression", "")
    assert "exec(" not in calc
