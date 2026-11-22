"""Disclosure prefix must survive capability compiler (APV-Fin v1.4.1)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.apv_fin.capability_contract import load_and_validate_capability_contracts
from hk_equity_mcp.apv_fin.dependency_expansion import (
    available_deps_from_ledger,
    disclosure_prefix_needed,
    rewrite_step_to_disclosure_list,
)
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.plan_state import PlanStep
from hk_equity_mcp.apv_fin.runtime_safety import disclosure_prefix_policy_hash
from hk_equity_mcp.apv_fin.scheduler import BudgetTracker, execute_retrieve_step


class _FakeResolver:
    def __init__(self, injected: Optional[List[str]] = None):
        self.injected = injected or [
            "hkexnews.list_hkexnews_announcements",
            "hkexnews.keyword_search",
            "hkexnews.get_announcement_pages",
        ]
        self.find_queries: List[str] = []

    def find_operations(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        self.find_queries.append(query)
        return {
            "candidates": [
                {"operation_id": oid, "score": 1.0 - i * 0.01}
                for i, oid in enumerate(self.injected)
            ]
        }

    def describe_operation(self, operation_id: str) -> Dict[str, Any]:
        requires = []
        provides = ["local_key", "title"]
        if "keyword" in operation_id or "page" in operation_id or "parse" in operation_id:
            requires = ["local_key"]
            provides = ["snippet", "local_key"]
        if "list" in operation_id:
            requires = []
            provides = ["local_key", "title", "release_date", "news_id"]
        return {
            "success": True,
            "contract": {
                "operation_id": operation_id,
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "stock_code": {"type": "string"},
                        "local_key": {"type": "string"},
                        "keywords": {"type": "string"},
                    },
                    "required": requires,
                },
                "requires_fields": requires,
                "provides_fields": provides,
                "capabilities": (
                    ["disclosure.list"]
                    if "list" in operation_id
                    else (
                        ["disclosure.keyword_search"]
                        if "keyword" in operation_id
                        else (
                            ["disclosure.page_read"]
                            if "page" in operation_id
                            else ["disclosure.parse"]
                        )
                    )
                ),
            },
        }

    def invoke_api(self, operation_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "operation_id": operation_id,
            "arguments": arguments,
            "result": {
                "success": True,
                "server_id": "hkexnews",
                "operation_id": operation_id,
                "arguments": arguments,
                "raw": {
                    "is_error": False,
                    "content": [
                        {
                            "json": {
                                "rows": [
                                    {
                                        "local_key": "lk_demo",
                                        "title": "FY2025 Results",
                                        "release_date": "2026-03-31",
                                        "news_id": "n1",
                                    }
                                ],
                                "total_rows": 1,
                                "returned_rows": 1,
                                "offset": 0,
                                "truncated": False,
                            }
                        }
                    ],
                },
            },
        }


def test_disclosure_prefix_policy_hash():
    assert disclosure_prefix_policy_hash()


def test_prefix_needed_when_ledger_lacks_identity():
    contracts = load_and_validate_capability_contracts()
    assert disclosure_prefix_needed(
        required_capability="financial.official_narrative_evidence",
        argument_slots={"ticker": "00700"},
        available_deps={},
        contracts=contracts,
    )
    assert not disclosure_prefix_needed(
        required_capability="financial.official_narrative_evidence",
        argument_slots={},
        available_deps={"local_key": "abc"},
        contracts=contracts,
    )
    assert not disclosure_prefix_needed(
        required_capability="financial.official_narrative_evidence",
        argument_slots={},
        available_deps={"news_id": "n1"},
        contracts=contracts,
    )


def test_available_deps_from_ledger_reads_local_key():
    ledger = EvidenceLedger()
    ledger.add(
        step_id="s0",
        claim="listed",
        source_type="mcp_invoke",
        raw_value={"local_key": "doc_1", "title": "FY"},
    )
    deps = available_deps_from_ledger(ledger)
    assert deps.get("local_key") == "doc_1"


def test_execute_retrieve_applies_prefix_after_compiler():
    """Narrative capability without ledger identity → disclosure.list before Binder."""
    contracts = load_and_validate_capability_contracts()
    step = PlanStep(
        step_id="s1_retrieve",
        step_type="retrieve",
        objective="official narrative non-IFRS profit",
        depends_on=[],
        mandatory=True,
        success_criteria="snippet",
        required_capability="financial.official_narrative_evidence",
        required_outputs=["snippet", "local_key"],
        argument_slots={"ticker": "00700", "period": "FY2025"},
        retrieval_query_hint="non-IFRS profit attributable",
        notes="Use disclosure keyword search",
    )
    budget = BudgetTracker(
        {
            "max_total_llm_calls_per_task": 10,
            "max_targeted_retrievals_per_step": 2,
            "max_total_mcp_invocations": 8,
            "max_exception_replan_calls_per_task": 2,
            "max_llm_monitor_rounds": 2,
            "max_plan_revisions": 2,
            "max_same_operation_repeats": 1,
            "max_final_repair_rounds": 1,
            "max_binding_repairs": 1,
            "max_idle_monitor_rounds": 1,
            "max_projected_observation_chars_per_step": 6000,
            "max_total_ledger_prompt_chars": 16000,
        }
    )
    resolver = _FakeResolver()
    # Offline binding will try to select; we only assert prefix rewrite + expansion count.
    out = execute_retrieve_step(
        step=step,
        task_spec={
            "goal": "non-IFRS",
            "entities": ["Tencent"],
            "periods": ["FY2025"],
            "metrics": ["non_IFRS_profit"],
        },
        resolver=resolver,  # type: ignore[arg-type]
        metadata_by_id={
            oid: {
                "operation_id": oid,
                "domain": "news",
                "runtime_availability": "runtime",
            }
            for oid in resolver.injected
        },
        runtime_operation_ids=resolver.injected,
        ledger=EvidenceLedger(),
        budget=budget,
        top_k=10,
        llm=None,
        question="Tencent FY2025 non-IFRS profit?",
        offline_tool_binding=True,
        contracts=contracts,
    )
    assert out.get("disclosure_prefix_applied") is True
    assert step.required_capability == "disclosure.list"
    assert budget.dependency_auto_expansion_count >= 1
    assert "disclosure_prefix_for:" in str(step.notes or "")
    # Binder pool should prefer list producers after rewrite.
    selected_cap = out.get("selected_capability") or step.required_capability
    assert selected_cap == "disclosure.list"
    # Must not require Binder to invent local_key for keyword/page as first hop.
    described = out.get("described_candidate_ids") or []
    assert any("list" in str(x) for x in described) or out.get("binding_mode") in {
        "deterministic",
        "expand_search",
        "unsupported",
        "skipped_empty_pool",
    }


def test_rewrite_helper_keeps_parent_target_in_notes():
    rewritten = rewrite_step_to_disclosure_list(
        {
            "required_capability": "financial.official_narrative_evidence",
            "required_outputs": ["snippet"],
            "argument_slots": {},
            "notes": "",
        }
    )
    assert rewritten["required_capability"] == "disclosure.list"
    assert "disclosure_prefix_for:financial.official_narrative_evidence" in rewritten[
        "notes"
    ]
