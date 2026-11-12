"""Grounding validator for Replanner final responses."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import forbid_question_target_as_observed


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def validate_replanner_response(
    *,
    response: Dict[str, Any],
    question: str,
    evidence_values: Dict[str, Any],
    answerability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return {ok, reason, adjusted_response?} — fail closed for ungrounded answered."""
    status = str(response.get("status") or "")
    answer = str(response.get("final_answer") or "")
    reasons: List[str] = []

    if status == "answered":
        # Numeric tokens in answer must appear in verified evidence (or no numbers).
        nums = _NUM.findall(answer)
        evidence_nums = set()
        for v in (evidence_values or {}).values():
            if isinstance(v, (int, float)) or (
                isinstance(v, str) and _NUM.fullmatch(v.strip() or "")
            ):
                evidence_nums.add(str(v).strip())
        for n in nums:
            if n in question and forbid_question_target_as_observed(
                field="close", value=float(n) if "." in n else int(n), question=question
            ):
                # question target used as answer without evidence
                if n not in evidence_nums:
                    reasons.append(f"ungrounded_question_target_number:{n}")
            elif n not in evidence_nums and evidence_values:
                # Allow reason-only text without requiring every number if no evidence nums
                if evidence_nums:
                    reasons.append(f"answer_number_not_in_evidence:{n}")
        if not evidence_values and nums:
            reasons.append("answered_without_verified_evidence")
        if reasons:
            return {"ok": False, "reasons": reasons}
        return {"ok": True, "reasons": []}

    if status == "unavailable":
        brief = str(response.get("reasoning_brief") or "") + " " + str(
            response.get("refusal_reason") or ""
        )
        low = brief.lower()
        # Optional/context-only gaps must not justify unavailable.
        optional_only = (
            "optional" in low
            and ("page" in low or "source_url" in low or "context" in low)
            and not any(
                m in low
                for m in (
                    "cutoff",
                    "capability",
                    "exhausted",
                    "no matching",
                    "infeasible",
                    "conflict",
                    "resource",
                    "budget",
                    "primary",
                    "fallback",
                )
            )
        )
        markers = (
            "cutoff",
            "capability",
            "exhausted",
            "no matching",
            "missing",
            "infeasible",
            "conflict",
            "resource",
            "budget",
            "unavailable",
            "not provide",
            "cannot",
        )
        if optional_only:
            reasons.append("unavailable_for_optional_context_only")
            return {"ok": False, "reasons": reasons}
        if not any(m in low for m in markers) and not (
            answerability and answerability.get("answerability_state") == "infeasible"
        ):
            reasons.append("unavailable_without_infeasibility_evidence")
            return {"ok": False, "reasons": reasons}
        return {"ok": True, "reasons": []}

    if status == "clarification_needed":
        return {"ok": True, "reasons": []}

    return {"ok": False, "reasons": [f"unknown_status:{status}"]}
