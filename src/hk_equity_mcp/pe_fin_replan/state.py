"""Plan–Execute–Replan runtime state (program-owned counters)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_replan.failed_invocation import (
    FailedInvocationGuard,
    failed_signature,
)
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS


def plan_hash(plan: Optional[Dict[str, Any]]) -> Optional[str]:
    if not plan:
        return None
    blob = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class PlanExecuteState:
    question: str
    original_plan: Optional[Dict[str, Any]] = None
    remaining_plan: List[Dict[str, Any]] = field(default_factory=list)
    past_steps: List[Dict[str, Any]] = field(default_factory=list)
    evidence_store: EvidenceStore = field(default_factory=EvidenceStore)
    invocation_guard: FailedInvocationGuard = field(
        default_factory=FailedInvocationGuard
    )
    failed_signatures: Set[str] = field(default_factory=set)
    final_response: Optional[Dict[str, Any]] = None

    executed_step_count: int = 0
    execution_round_count: int = 0
    replan_round_count: int = 0
    no_progress_round_count: int = 0
    premature_insufficient_blocked_count: int = 0
    forced_final_round: bool = False

    llm_calls: int = 0
    mcp_calls: int = 0
    embedding_queries: int = 0

    max_llm_calls: int = DEFAULT_BUDGETS["max_total_llm_calls_per_task"]
    max_mcp_calls: int = DEFAULT_BUDGETS["max_total_mcp_invocations"]
    max_execution_rounds: int = DEFAULT_BUDGETS["max_execution_rounds"]
    max_executed_steps: int = DEFAULT_BUDGETS["max_executed_steps"]
    max_replan_rounds: int = DEFAULT_BUDGETS["max_replan_rounds"]
    max_new_steps_per_replan: int = DEFAULT_BUDGETS["max_new_steps_per_replan"]
    max_no_progress_rounds: int = 2

    remaining_plan_history: List[Dict[str, Any]] = field(default_factory=list)
    replan_decisions: List[Dict[str, Any]] = field(default_factory=list)
    plan_diff_history: List[Dict[str, Any]] = field(default_factory=list)
    stopping_reason: Optional[str] = None
    resource_limit_failure: bool = False
    loop_prevention_trigger: Optional[str] = None
    final_response_source: Optional[str] = None
    answerability_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    completed_step_ids: Set[str] = field(default_factory=set)
    completed_objectives: Set[str] = field(default_factory=set)

    def original_plan_hash(self) -> Optional[str]:
        return plan_hash(self.original_plan)

    def compact_for_replanner(self, *, latest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compressed state for Replanner — no gold, no raw envelopes, no secrets."""
        store = self.evidence_store
        return {
            "question": self.question,
            "original_plan_summary": {
                "n_steps": len((self.original_plan or {}).get("steps") or []),
                "step_types": [
                    s.get("type") for s in ((self.original_plan or {}).get("steps") or [])
                ],
                "objectives": [
                    str(s.get("objective") or "")[:160]
                    for s in ((self.original_plan or {}).get("steps") or [])
                ],
            },
            "remaining_plan": deepcopy(self.remaining_plan),
            "past_executed_steps": [
                {
                    "step_id": p.get("step_id"),
                    "step_type": p.get("step_type"),
                    "status": p.get("status"),
                    "failure_reason": p.get("failure_reason"),
                    "missing_outputs": p.get("missing_outputs"),
                    "execution_failure_class": p.get("execution_failure_class"),
                    "values_keys": sorted((p.get("values") or {}).keys()),
                }
                for p in self.past_steps
            ],
            "latest_step_result": latest,
            "canonical_evidence": {
                "verified_values": dict(store.values),
                "missing": list(store.missing)[:20],
                "compute_records": [
                    {
                        "step_id": r.get("step_id"),
                        "output_name": r.get("output_name"),
                        "ok": r.get("ok"),
                        "result": r.get("result"),
                        "failure_class": r.get("failure_class"),
                    }
                    for r in store.compute_records
                ],
            },
            "failed_operation_signatures_n": len(self.invocation_guard.signatures)
            or len(self.failed_signatures),
            "signature_blocked_count": self.invocation_guard.signature_blocked_count,
            "blocked_invocation_keys_n": len(self.invocation_guard.blocked_keys),
            "answerability_latest": (
                self.answerability_snapshots[-1]
                if self.answerability_snapshots
                else None
            ),
            "budgets_remaining": {
                "llm_calls_remaining": max(0, self.max_llm_calls - self.llm_calls),
                "mcp_calls_remaining": max(0, self.max_mcp_calls - self.mcp_calls),
                "execution_rounds_remaining": max(
                    0, self.max_execution_rounds - self.execution_round_count
                ),
                "executed_steps_remaining": max(
                    0, self.max_executed_steps - self.executed_step_count
                ),
                "replan_rounds_remaining": max(
                    0, self.max_replan_rounds - self.replan_round_count
                ),
            },
            "execution_round_count": self.execution_round_count,
            "max_execution_rounds": self.max_execution_rounds,
        }

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "original_plan": self.original_plan,
            "original_plan_hash": self.original_plan_hash(),
            "remaining_plan": list(self.remaining_plan),
            "past_steps": list(self.past_steps),
            "executed_step_count": self.executed_step_count,
            "execution_round_count": self.execution_round_count,
            "max_execution_rounds": self.max_execution_rounds,
            "replan_round_count": self.replan_round_count,
            "max_replan_rounds": self.max_replan_rounds,
            "no_progress_round_count": self.no_progress_round_count,
            "premature_insufficient_blocked_count": self.premature_insufficient_blocked_count,
            "forced_final_round": self.forced_final_round,
            "llm_calls": self.llm_calls,
            "mcp_calls": self.mcp_calls,
            "embedding_queries": self.embedding_queries,
            "failed_signatures_n": len(self.invocation_guard.signatures)
            or len(self.failed_signatures),
            "signature_blocked_count": self.invocation_guard.signature_blocked_count,
            "blocked_invocation_keys_n": len(self.invocation_guard.blocked_keys),
            "remaining_plan_history": list(self.remaining_plan_history),
            "replan_decisions": list(self.replan_decisions),
            "plan_diff_history": list(self.plan_diff_history),
            "stopping_reason": self.stopping_reason,
            "resource_limit_failure": self.resource_limit_failure,
            "loop_prevention_trigger": self.loop_prevention_trigger,
            "final_response_source": self.final_response_source,
            "final_response": self.final_response,
            "answerability_snapshots": list(self.answerability_snapshots),
            "evidence_store": self.evidence_store.to_dict(),
        }
