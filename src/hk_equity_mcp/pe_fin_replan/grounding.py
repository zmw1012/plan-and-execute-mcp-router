"""Grounding + premature-insufficient checks for RePlan v1.1 responses."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import forbid_question_target_as_observed
from hk_equity_mcp.pe_fin_replan.plan_ops import select_one_ready_step

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

_EARLY_INSUFFICIENT_OK_MARKERS = (
    "cutoff",
    "capability unavailable",
    "capability",
    "exhausted",
    "no matching",
    "verified no matching",
    "primary/fallback",
    "primary and fallback",
)


def validate_replanner_response(
    *,
    response: Dict[str, Any],
    question: str,
    evidence_values: Dict[str, Any],
    answerability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return {ok, reasons} — fail closed for ungrounded answered."""
    completion = str(response.get("completion") or "")
    answer = str(response.get("final_answer") or "")
    reasons: List[str] = []

    if completion == "answered":
        nums = _NUM.findall(answer)
        evidence_nums = set()
        for v in (evidence_values or {}).values():
            if isinstance(v, (int, float)) or (
                isinstance(v, str) and _NUM.fullmatch(v.strip() or "")
            ):
                evidence_nums.add(str(v).strip())
        for n in nums:
            if n in question and forbid_question_target_as_observed(
                field="close",
                value=float(n) if "." in n else int(n),
                question=question,
            ):
                if n not in evidence_nums:
                    reasons.append(f"ungrounded_question_target_number:{n}")
            elif n not in evidence_nums and evidence_values:
                if evidence_nums:
                    reasons.append(f"answer_number_not_in_evidence:{n}")
        if not evidence_values and nums:
            reasons.append("answered_without_verified_evidence")
        if reasons:
            return {"ok": False, "reasons": reasons}
        return {"ok": True, "reasons": []}

    if completion == "insufficient_evidence":
        # Method-level simplified end state — no refusal taxonomy.
        return {"ok": True, "reasons": []}

    return {"ok": False, "reasons": [f"unknown_completion:{completion}"]}


def is_premature_insufficient(
    *,
    execution_round: int,
    max_execution_rounds: int,
    remaining_plan: Sequence[Dict[str, Any]],
    completed_ids: Set[str],
    answerability: Optional[Dict[str, Any]],
    decision_reason: str,
    evidence_values: Optional[Dict[str, Any]] = None,
) -> bool:
    """Block early insufficient_evidence when recoverable paths remain."""
    if execution_round >= max_execution_rounds:
        return False
    reason_l = f"{decision_reason or ''}".lower()
    if any(m in reason_l for m in _EARLY_INSUFFICIENT_OK_MARKERS):
        return False

    if select_one_ready_step(remaining_plan, completed_ids) is not None:
        return True
    if list(remaining_plan):
        return True

    ans_state = str((answerability or {}).get("answerability_state") or "")
    if ans_state == "infeasible":
        return False
    if ans_state in {"sufficient"} and evidence_values:
        # Evidence already sufficient — insufficient would be wrong, but not "premature path"
        return False
    # Core evidence still missing and rounds remain → must continue with Plan
    if ans_state in {
        "partial_but_answerable",
        "insufficient",
        "missing_core",
        "uncertain",
        "",
    }:
        return True
    if not evidence_values:
        return True
    return False
