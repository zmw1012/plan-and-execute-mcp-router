"""Offline disclosure failure taxonomy (passive; no network retry)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TAXONOMY_CLASSES = (
    "invalid_arguments",
    "no_matching_result",
    "scope_date_title_mismatch",
    "network_service_failure",
    "envelope_adapter_failure",
    "missing_dependency_output",
    "unknown_due_to_missing_observability",
)


def classify_disclosure_stage_failure(
    *,
    stage: Optional[str],
    gateway_success: Optional[bool],
    error: Optional[str],
    n_list_rows: Optional[int] = None,
    extracted_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Classify one disclosure chain stage from already-recorded observability."""
    err = str(error or "").strip()
    stage_s = str(stage or "").strip().lower()
    low = err.lower()

    if not err and gateway_success is None and n_list_rows is None:
        return {
            "class": "unknown_due_to_missing_observability",
            "stage": stage_s or None,
            "evidence": "insufficient_raw_observability",
        }

    if any(
        k in low
        for k in (
            "invalid argument",
            "missing required",
            "schema",
            "validation",
            "type error",
        )
    ):
        return {
            "class": "invalid_arguments",
            "stage": stage_s,
            "evidence": err,
        }

    if "budget exhausted" in low or "download budget" in low:
        # Adapter/session policy stop — not a network outage; not no_matching_result.
        return {
            "class": "envelope_adapter_failure",
            "stage": stage_s,
            "evidence": err,
            "subtype": "download_budget_exhausted",
        }

    if (
        "no announcements" in low
        or "no matching" in low
        or (stage_s == "list" and gateway_success is False and (n_list_rows or 0) == 0)
    ):
        return {
            "class": "no_matching_result",
            "stage": stage_s,
            "evidence": err or f"n_list_rows={n_list_rows}",
        }

    if any(k in low for k in ("timeout", "connection", "http", "503", "502", "network")):
        return {
            "class": "network_service_failure",
            "stage": stage_s,
            "evidence": err,
        }

    if any(k in low for k in ("scope", "date mismatch", "title", "period mismatch")):
        return {
            "class": "scope_date_title_mismatch",
            "stage": stage_s,
            "evidence": err,
        }

    if any(k in low for k in ("adapt", "envelope", "normalize", "projection")):
        return {
            "class": "envelope_adapter_failure",
            "stage": stage_s,
            "evidence": err,
        }

    if gateway_success is False and not err:
        return {
            "class": "unknown_due_to_missing_observability",
            "stage": stage_s,
            "evidence": "gateway_false_without_error_string",
        }

    if err:
        return {
            "class": "unknown_due_to_missing_observability",
            "stage": stage_s,
            "evidence": err,
            "note": "error_present_but_unmapped; no network retry performed",
        }

    return {
        "class": "unknown_due_to_missing_observability",
        "stage": stage_s or None,
        "evidence": "no_error_string",
    }


def classify_task_disclosure_failures(row: Dict[str, Any]) -> Dict[str, Any]:
    """Walk retrieval funnels' disclosure_composite traces (offline only)."""
    findings: List[Dict[str, Any]] = []
    for funnel in row.get("retrieval_funnels") or []:
        dc = funnel.get("disclosure_composite")
        if not isinstance(dc, dict) or not dc:
            continue
        for stage_rec in dc.get("trace") or []:
            if not isinstance(stage_rec, dict):
                continue
            if stage_rec.get("success") is True and stage_rec.get("gateway_success") is not False:
                continue
            findings.append(
                classify_disclosure_stage_failure(
                    stage=stage_rec.get("stage"),
                    gateway_success=stage_rec.get("gateway_success"),
                    error=stage_rec.get("error"),
                    n_list_rows=stage_rec.get("n_list_rows"),
                    extracted_fields=list(stage_rec.get("extracted_fields") or []),
                )
            )
        # Also honor step-level stage_failed:* if composite missing detail.
        fr = str((funnel.get("failure_reason") or dc.get("error") or ""))
        if fr.startswith("stage_failed:") and not findings:
            stage = fr.split(":", 1)[-1]
            findings.append(
                {
                    "class": "unknown_due_to_missing_observability",
                    "stage": stage,
                    "evidence": fr,
                }
            )
    return {
        "task_id": row.get("task_id") or row.get("id"),
        "findings": findings,
        "primary_class": findings[0]["class"] if findings else None,
    }
