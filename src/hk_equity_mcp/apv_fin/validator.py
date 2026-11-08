"""Final APV-Fin validator (fail-closed gates before answer emission)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.plan_state import PlanState


def validate_final_state(
    *,
    plan_state: PlanState,
    ledger: EvidenceLedger,
    task_spec: Optional[Dict[str, Any]] = None,
    final_payload: Optional[Dict[str, Any]] = None,
    compute_traces: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    task_spec = task_spec or {}
    final_payload = final_payload or {}
    compute_traces = list(compute_traces or [])
    checks: List[Dict[str, Any]] = []

    coverage = plan_state.mandatory_coverage()
    checks.append(
        {
            "name": "mandatory_plan_coverage",
            "ok": len(coverage["mandatory_unresolved"]) == 0,
            "detail": coverage,
        }
    )

    ambiguities = list(task_spec.get("ambiguities") or [])
    status = str(final_payload.get("status") or "")
    unresolved_ambiguity = bool(ambiguities) and status == "answered"
    checks.append(
        {
            "name": "unresolved_ambiguity",
            "ok": not unresolved_ambiguity,
            "detail": {"ambiguities": ambiguities, "status": status},
        }
    )

    entities = list(task_spec.get("entities") or [])
    entity_ok = True
    if entities and status == "answered":
        # Require at least one evidence entity overlap or final entity field.
        ev_entities = {
            (i.entity or "").lower()
            for i in ledger.items
            if i.entity
        }
        final_entity = str(final_payload.get("entity") or "").lower()
        entity_ok = any(
            e.lower() in ev_entities or (final_entity and e.lower() in final_entity)
            for e in entities
        ) or any(e.lower() in str(final_payload.get("final_answer") or "").lower() for e in entities)
    checks.append(
        {
            "name": "company_ticker",
            "ok": entity_ok,
            "detail": {"entities": entities},
        }
    )

    periods = list(task_spec.get("periods") or [])
    period_ok = True
    if periods and status == "answered":
        blob = " ".join(
            [
                str(final_payload.get("period") or ""),
                str(final_payload.get("final_answer") or ""),
            ]
            + [str(i.period_or_date or "") for i in ledger.items]
        ).lower()
        period_ok = any(p.lower() in blob for p in periods) or any(
            "fy" in blob or "20" in blob for _ in [0]
        )
    checks.append(
        {
            "name": "period_cutoff",
            "ok": period_ok,
            "detail": {
                "periods": periods,
                "cutoff_date": task_spec.get("cutoff_date"),
            },
        }
    )

    basis_hints = list(task_spec.get("metric_basis_hints") or [])
    basis_ok = True
    if basis_hints and status == "answered":
        blob = " ".join(
            [
                str(final_payload.get("metric_basis") or ""),
                str(final_payload.get("final_answer") or ""),
                str(final_payload.get("reasoning_brief") or ""),
            ]
            + [str(i.metric_basis or "") for i in ledger.items]
        ).lower()
        keywords = ("ifrs", "non-ifrs", "non-gaap", "reported", "adjusted", "qfq", "hfq")
        basis_ok = any(h.lower() in blob for h in basis_hints) or any(
            k in blob for k in keywords
        )
    checks.append(
        {
            "name": "reported_adjusted_basis",
            "ok": basis_ok,
            "detail": {"metric_basis_hints": basis_hints},
        }
    )

    unit_hints = list(task_spec.get("currency_unit_hints") or [])
    unit_ok = True
    if unit_hints and status == "answered":
        blob = " ".join(
            [
                str(final_payload.get("currency") or ""),
                str(final_payload.get("unit") or ""),
                str(final_payload.get("final_answer") or ""),
            ]
            + [str(i.currency_or_unit or "") for i in ledger.items]
        ).lower()
        unit_ok = any(h.lower() in blob for h in unit_hints)
    checks.append(
        {
            "name": "currency_unit",
            "ok": unit_ok or not unit_hints,
            "detail": {"currency_unit_hints": unit_hints},
        }
    )

    calc_ok = True
    calc_detail: List[Dict[str, Any]] = []
    for tr in compute_traces:
        ok = bool(tr.get("ok")) and tr.get("result") is not None
        calc_detail.append(
            {
                "step_id": tr.get("step_id"),
                "ok": ok,
                "formula": tr.get("formula"),
                "operands": tr.get("operands"),
                "result": tr.get("result"),
            }
        )
        if not ok:
            calc_ok = False
    if any(s.step_type == "compute" and s.mandatory for s in plan_state.steps.values()):
        if not compute_traces:
            calc_ok = False
    checks.append(
        {
            "name": "calculation_operands",
            "ok": calc_ok,
            "detail": {"traces": calc_detail},
        }
    )

    evidence_ok = True
    if status == "answered":
        evidence_ok = len(ledger.items) > 0
        cited = list(final_payload.get("evidence_ids") or [])
        known = {i.evidence_id for i in ledger.items}
        if cited and any(c not in known for c in cited):
            evidence_ok = False
    checks.append(
        {
            "name": "source_evidence_support",
            "ok": evidence_ok,
            "detail": ledger.summary(),
        }
    )

    # Unsupported substitution: answered despite clarify/refuse mandatory steps unresolved.
    refuse_or_clarify = [
        s
        for s in plan_state.steps.values()
        if s.step_type in {"clarify", "refuse"} and s.mandatory
    ]
    substitution_ok = True
    if refuse_or_clarify and status == "answered":
        # If a mandatory refuse/clarify remains unresolved, answering is substitution.
        if any(s.status != "resolved" for s in refuse_or_clarify):
            substitution_ok = False
    if task_spec.get("answerability_hypothesis") in {
        "needs_clarification",
        "likely_unavailable",
    } and status == "answered":
        substitution_ok = False
    checks.append(
        {
            "name": "unsupported_substitution",
            "ok": substitution_ok,
            "detail": {
                "answerability_hypothesis": task_spec.get("answerability_hypothesis"),
                "status": status,
            },
        }
    )

    failed = [c for c in checks if not c["ok"]]
    return {
        "ok": len(failed) == 0,
        "failed_checks": [c["name"] for c in failed],
        "checks": checks,
        "coverage": coverage,
    }


def controlled_failure_payload(
    *,
    reason: str,
    status: str = "unavailable",
) -> Dict[str, Any]:
    return {
        "schema_version": "apv_fin_final_v1",
        "status": status,
        "final_answer": (
            "Unable to produce a supported answer under APV-Fin budgets/validators. "
            f"Reason: {reason}"
        ),
        "reasoning_brief": reason,
        "evidence_ids": [],
        "clarification_question": None,
        "refusal_reason": reason if status == "unavailable" else None,
        "currency": None,
        "unit": None,
        "metric_basis": None,
        "period": None,
        "entity": None,
    }
