"""Three-state (+contradictory) answerability for PE-Fin Lite v1.2-EEA."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

ANSWERABILITY_EEA_VERSION = "pe_fin_lite_answerability_eea_v1"

ANSWERABILITY_STATES = (
    "sufficient",
    "partial_but_answerable",
    "infeasible",
    "contradictory",
)


def answerability_eea_hash() -> str:
    blob = json.dumps(
        {
            "version": ANSWERABILITY_EEA_VERSION,
            "states": list(ANSWERABILITY_STATES),
            "unavailable_only_if_infeasible": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def classify_answerability(
    *,
    plan: Optional[Dict[str, Any]],
    store: Any,
    step_statuses: Sequence[Dict[str, Any]],
    merge_conflicts: Optional[Sequence[Dict[str, Any]]] = None,
    answerability_class: Optional[str] = None,
    temporal_blocked: bool = False,
    capability_gap: bool = False,
) -> Dict[str, Any]:
    """Deterministic answerability state before Final Synthesizer."""
    from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import field_role

    conflicts = list(merge_conflicts or [])
    for item in getattr(store, "items", []) or []:
        if isinstance(item, dict):
            conflicts.extend(list(item.get("merge_conflicts") or []))

    if conflicts:
        return {
            "answerability_state": "contradictory",
            "answerability_reason_code": "conflicting_evidence_same_scope",
            "core_outputs_satisfied": False,
            "optional_outputs_missing": [],
            "infeasibility_evidence": {"conflicts": conflicts[:10]},
        }

    required: List[str] = []
    for step in ((plan or {}).get("steps") or []):
        if step.get("type") == "retrieve":
            required.extend(str(x) for x in (step.get("required_outputs") or []))
    required = list(dict.fromkeys(required))

    values = dict(getattr(store, "values", None) or {})
    missing_store = list(getattr(store, "missing", None) or [])

    core_missing: List[str] = []
    optional_missing: List[str] = []
    context_missing: List[str] = []
    core_present: List[str] = []
    for req in required:
        role = field_role(req, required_outputs=required)
        # Prefer step-qualified then bare
        present = req in values or any(str(k).endswith("." + req) for k in values)
        if present:
            if role == "answer_bearing":
                core_present.append(req)
            continue
        if role == "answer_bearing":
            core_missing.append(req)
        elif role == "context":
            context_missing.append(req)
        else:
            optional_missing.append(req)

    # Also look at step missing_outputs tagged optional vs core
    for m in missing_store:
        for field in m.get("missing_outputs") or []:
            role = field_role(str(field), required_outputs=required)
            if role == "answer_bearing" and str(field) not in core_missing and str(field) not in core_present:
                core_missing.append(str(field))
            elif role == "optional_supporting" and str(field) not in optional_missing:
                optional_missing.append(str(field))
            elif role == "context" and str(field) not in context_missing:
                context_missing.append(str(field))

    statuses = [str(s.get("status") or "") for s in step_statuses or []]
    all_retrieve_failed = bool(step_statuses) and all(
        str(s.get("type")) != "retrieve"
        or str(s.get("status")) in {"failed", "unavailable", "empty", "skipped_due_to_dependency"}
        for s in step_statuses
        if str(s.get("type")) == "retrieve"
    )

    if temporal_blocked:
        return {
            "answerability_state": "infeasible",
            "answerability_reason_code": "temporal_cutoff_blocked",
            "core_outputs_satisfied": False,
            "optional_outputs_missing": optional_missing,
            "infeasibility_evidence": {"temporal_blocked": True},
        }
    if capability_gap or answerability_class in {"intentionally_infeasible", "capability_gap"}:
        if not core_present:
            return {
                "answerability_state": "infeasible",
                "answerability_reason_code": "capability_or_intentionally_infeasible",
                "core_outputs_satisfied": False,
                "optional_outputs_missing": optional_missing,
                "infeasibility_evidence": {"answerability_class": answerability_class},
            }

    if not core_missing and core_present:
        if optional_missing or context_missing:
            return {
                "answerability_state": "partial_but_answerable",
                "answerability_reason_code": "core_ok_optional_or_context_missing",
                "core_outputs_satisfied": True,
                "optional_outputs_missing": optional_missing + context_missing,
                "infeasibility_evidence": None,
            }
        return {
            "answerability_state": "sufficient",
            "answerability_reason_code": "all_core_outputs_present",
            "core_outputs_satisfied": True,
            "optional_outputs_missing": [],
            "infeasibility_evidence": None,
        }

    if core_present and core_missing:
        # Some core present — still may be partial_but_answerable if majority core ok
        return {
            "answerability_state": "partial_but_answerable",
            "answerability_reason_code": "partial_core_outputs_present",
            "core_outputs_satisfied": False,
            "optional_outputs_missing": optional_missing + context_missing,
            "infeasibility_evidence": {"core_missing": core_missing, "core_present": core_present},
        }

    if all_retrieve_failed and not values:
        return {
            "answerability_state": "infeasible",
            "answerability_reason_code": "all_primary_fallback_exhausted",
            "core_outputs_satisfied": False,
            "optional_outputs_missing": optional_missing,
            "infeasibility_evidence": {"statuses": statuses[:20]},
        }

    if answerability_class == "should_refuse_or_clarify" and not core_present:
        return {
            "answerability_state": "infeasible",
            "answerability_reason_code": "missing_user_information_or_refuse",
            "core_outputs_satisfied": False,
            "optional_outputs_missing": optional_missing,
            "infeasibility_evidence": {"answerability_class": answerability_class},
        }

    if not values:
        return {
            "answerability_state": "infeasible",
            "answerability_reason_code": "no_verified_evidence",
            "core_outputs_satisfied": False,
            "optional_outputs_missing": optional_missing,
            "infeasibility_evidence": {"missing_store": missing_store[:10]},
        }

    return {
        "answerability_state": "partial_but_answerable",
        "answerability_reason_code": "some_verified_evidence_present",
        "core_outputs_satisfied": bool(core_present) and not core_missing,
        "optional_outputs_missing": optional_missing + context_missing,
        "infeasibility_evidence": {"core_missing": core_missing},
    }


def final_status_from_answerability(state: str) -> str:
    """Map answerability → allowed Final status guidance (not an LLM call)."""
    if state == "infeasible":
        return "unavailable"
    if state == "contradictory":
        return "answered_with_conflict"  # Final must report conflict; not silent unavailable
    if state in {"sufficient", "partial_but_answerable"}:
        return "answered"
    return "unavailable"
