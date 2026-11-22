"""Final validator + budget stopping + clarify/refuse path tests."""

from __future__ import annotations

from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.plan_state import PlanState
from hk_equity_mcp.apv_fin.scheduler import BudgetTracker
from hk_equity_mcp.apv_fin.validator import (
    controlled_failure_payload,
    validate_final_state,
)


def test_validator_flags_unsupported_substitution_and_missing_evidence():
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "refuse1",
                    "step_type": "refuse",
                    "objective": "refuse temporal gap",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "refused",
                }
            ]
        }
    )
    ledger = EvidenceLedger()
    result = validate_final_state(
        plan_state=state,
        ledger=ledger,
        task_spec={
            "ambiguities": ["which period?"],
            "entities": ["IssuerX"],
            "periods": ["FY2025"],
            "answerability_hypothesis": "likely_unavailable",
            "answerability_reason": "Cutoff or disclosure gap likely blocks a supported answer.",
        },
        final_payload={
            "status": "answered",
            "final_answer": "42",
            "evidence_ids": [],
        },
    )
    assert result["ok"] is False
    assert "unsupported_substitution" in result["failed_checks"]
    assert "mandatory_plan_coverage" in result["failed_checks"]
    assert "source_evidence_support" in result["failed_checks"]


def test_max_budget_stopping():
    budget = BudgetTracker(
        {
            "max_plan_revisions": 2,
            "max_llm_monitor_rounds": 3,
            "max_total_mcp_invocations": 2,
        }
    )
    budget.monitor_rounds = 4
    assert budget.exhausted() == "max_llm_monitor_rounds"
    payload = controlled_failure_payload(reason="max_llm_monitor_rounds")
    assert payload["status"] == "unavailable"
    assert "max_llm_monitor_rounds" in payload["final_answer"]


def test_clarification_refusal_payloads():
    clarify = controlled_failure_payload(
        reason="ambiguous ticker", status="clarification_needed"
    )
    assert clarify["status"] == "clarification_needed"
    refuse = controlled_failure_payload(reason="capability gap")
    assert refuse["status"] == "unavailable"
