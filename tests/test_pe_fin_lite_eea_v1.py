"""Focused offline tests for PE-Fin Lite v1.2-EEA (no network/API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_gates import hk_artifact_root, require_hk_artifact

from hk_equity_mcp.pe_fin_lite.answerability_eea import (
    classify_answerability,
    final_status_from_answerability,
)
from hk_equity_mcp.pe_fin_lite.compute_binding_eea import bind_compute_operands
from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import (
    apply_aliases_to_pool,
    forbid_question_target_as_observed,
    merge_primary_fallback_eea,
    required_outputs_satisfaction_eea,
    units_compatible_or_convertible,
)
from hk_equity_mcp.pe_fin_lite.evidence_provenance import make_field_provenance
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite.execution_error_handling import local_repair_arguments
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    RETRIEVAL_FORMULA_NAME,
    retrieval_formula_hash,
)
from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()
FORMAL = ARTIFACT_ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
FORMAL_SHA = {
    "predictions.jsonl": "4c5c4df3ababca99f9a91203605154ad8fc4a05cdfbc2ba2acb7cc11bc2a7d04",
    "run_metadata.json": "2a3dff284afa86db8ec93433965f2b451e7aef05c65a78edd53fce3f84b8d602",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_v12_raw_hashes_byte_identical():
    require_hk_artifact("outputs/pe_fin_lite_r5_hk155_live_v1/predictions.jsonl")
    assert FORMAL.exists()
    for name, expected in FORMAL_SHA.items():
        assert _sha(FORMAL / name) == expected


def test_missing_symbol_filled_from_taskspec_grounding():
    repaired = local_repair_arguments(
        arguments={"start_date": "2024-01-01", "end_date": "2024-01-31"},
        step={"scope_constraints": {"entity": "00700"}},
        grounding={"stock_code": "00700"},
        contract={
            "json_schema": {
                "properties": {"symbol": {}, "start_date": {}, "end_date": {}},
                "required": ["symbol", "start_date", "end_date"],
            }
        },
    )
    assert repaired["arguments"]["symbol"] == "00700"
    assert repaired["changed"] is True


def test_cannot_use_question_target_number_as_answer_bearing():
    assert forbid_question_target_as_observed(
        field="close", value=123.45, question="What was the close of 00700.HK? Was it 123.45?"
    )
    assert not forbid_question_target_as_observed(
        field="issuer_code", value="00700", question="Look up 00700.HK close"
    )


def test_alias_mapping_accepts_semantic_fields():
    pool = apply_aliases_to_pool({"HOLDER_PROFIT": 100.0, "OPERATE_INCOME": 200.0})
    assert pool["attributable_profit"] == 100.0
    assert pool["revenue"] == 200.0
    fmap = {
        "attributable_profit": make_field_provenance(
            value=100.0, provenance_class="observed", source="mcp"
        ),
        "currency": make_field_provenance(
            value="HKD", provenance_class="grounded_input", source="arg"
        ),
    }
    sat = required_outputs_satisfaction_eea(
        fmap, ["attributable_profit", "currency", "source_url"]
    )
    assert sat["ok"] is True
    assert "source_url" in sat["missing_optional"]
    assert "currency" not in sat["missing"]  # context may be grounded; not blocking


def test_optional_context_missing_does_not_auto_unavailable():
    fmap = {
        "close": make_field_provenance(
            value=10.0, provenance_class="observed", source="mcp"
        ),
    }
    sat = required_outputs_satisfaction_eea(fmap, ["close", "page_number", "source_url"])
    assert sat["ok"] is True
    assert set(sat["missing_optional"]) >= {"page_number", "source_url"}


def test_answer_bearing_missing_still_fail_closed():
    fmap = {
        "issuer_code": make_field_provenance(
            value="00700", provenance_class="grounded_input", source="arg"
        ),
    }
    sat = required_outputs_satisfaction_eea(fmap, ["close", "issuer_code"])
    assert sat["ok"] is False
    assert "close" in sat["missing_answer_bearing"]


def test_entity_and_period_mismatch_forbid_merge():
    p = {
        "close": make_field_provenance(
            value=1.0, provenance_class="observed", source="a"
        )
    }
    f = {
        "close": make_field_provenance(
            value=2.0, provenance_class="observed", source="b"
        )
    }
    out = merge_primary_fallback_eea(
        p,
        f,
        primary_scope={"entity": "00700", "period": "FY2024", "metric_basis": "", "trade_date": ""},
        fallback_scope={"entity": "00941", "period": "FY2024", "metric_basis": "", "trade_date": ""},
    )
    assert out["ok"] is False
    assert out["primary_fallback_merge_status"] == "rejected_scope"


def test_compatible_units_convert_incompatible_reject():
    ok, scale, _ = units_compatible_or_convertible("unit", "thousand")
    assert ok and scale == 1000.0
    ok2, scale2, _ = units_compatible_or_convertible("HKD", "USD")
    assert ok2 is False and scale2 is None


def test_compute_operand_binding_and_upstream_vs_calculator():
    store = EvidenceStore()
    store.values["s1.close"] = 10.0
    store.values["s2.close"] = 12.0
    store.provenance_by_key["s1.close"] = {"entity": "00700", "period": "FY2024"}
    store.provenance_by_key["s2.close"] = {"entity": "00700", "period": "FY2024"}
    ok = bind_compute_operands(
        spec={
            "expression": "(b - a) / a",
            "operand_refs": {"a": "s1.close", "b": "s2.close"},
            "output_name": "chg",
        },
        store=store,
    )
    assert ok["ok"] is True
    assert abs(ok["evaluated"]["result"] - 0.2) < 1e-9

    empty = EvidenceStore()
    empty.missing.append({"step_id": "s1", "missing_outputs": ["close"]})
    bad = bind_compute_operands(
        spec={
            "expression": "a + b",
            "operand_refs": {"a": "s1.close", "b": "s2.close"},
            "output_name": "x",
        },
        store=empty,
    )
    assert bad["ok"] is False
    assert bad["failure_class"] == "upstream_evidence_missing"

    calc = bind_compute_operands(
        spec={
            "expression": "a / b",
            "operand_refs": {"a": "s1.close", "b": "s2.close"},
            "output_name": "r",
        },
        store=store,
    )
    # b=12 ok; force zero
    store.values["s2.close"] = 0.0
    zero = bind_compute_operands(
        spec={
            "expression": "a / b",
            "operand_refs": {"a": "s1.close", "b": "s2.close"},
            "output_name": "r",
        },
        store=store,
    )
    assert zero["failure_class"] == "division_by_zero"


def test_partial_but_answerable_vs_infeasible():
    store = EvidenceStore()
    store.values["close"] = 10.0
    plan = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "required_outputs": ["close", "source_url"],
            }
        ]
    }
    ans = classify_answerability(
        plan=plan,
        store=store,
        step_statuses=[{"id": "s1", "type": "retrieve", "status": "completed"}],
    )
    assert ans["answerability_state"] in {"sufficient", "partial_but_answerable"}
    assert final_status_from_answerability(ans["answerability_state"]) == "answered"

    empty = EvidenceStore()
    inf = classify_answerability(
        plan=plan,
        store=empty,
        step_statuses=[
            {"id": "s1", "type": "retrieve", "status": "failed"},
        ],
        answerability_class="intentionally_infeasible",
    )
    assert inf["answerability_state"] == "infeasible"
    assert final_status_from_answerability("infeasible") == "unavailable"


def test_budgets_and_r5_unchanged():
    b = BudgetTracker({})
    assert b.limits["max_total_llm_calls_per_task"] == 8
    assert b.limits["max_total_mcp_invocations"] == 8
    assert b.limits["retrieval_top_k"] == 5
    assert b.limits["describe_top_n"] == 3
    assert RETRIEVAL_FORMULA_NAME == "R5_global_lexical@0.12"
    assert retrieval_formula_hash() == (
        "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"
    )


def test_diagnostic20_manifest_contract():
    path = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    assert man["role"] == "posthoc_engineering_diagnostic_only"
    assert man["explicitly_not_formal_accuracy"] is True
    assert man["cannot_replace_hk155"] is True
    assert len(man["ordered_task_ids"]) == 20
    assert man["selection_rule"]["no_gold_correctness_sort"] is True
    assert "fin_cf_022" not in json.dumps(man)  # no manual-grade leakage required


def test_disclosure_shortest_prefix_and_second_candidate_flag_exist():
    from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
        DISCLOSURE_CHAIN,
        needs_document_text,
        run_disclosure_composite,
    )

    assert DISCLOSURE_CHAIN[0] == "list"
    assert needs_document_text(["title", "release_date"]) is False
    assert needs_document_text(["snippet"]) is True
    # Signature accepts eea_enabled
    assert "eea_enabled" in run_disclosure_composite.__code__.co_varnames
