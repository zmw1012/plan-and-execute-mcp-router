"""Generic argument-slot → schema parameter mapping (company/task-agnostic)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Slot name → acceptable target parameter names (schema-compatible only).
SLOT_PARAM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ticker": ("symbol", "stock", "stock_code", "code", "secu_code", "ticker"),
    "issuer_code": ("symbol", "stock", "stock_code", "code", "issuer_code"),
    "start_date": ("start_date", "from_date", "begin_date", "start"),
    "end_date": ("end_date", "to_date", "finish_date", "end"),
    "period_type": ("indicator", "period", "period_type", "report_type"),
    "adjustment": ("adjust", "adjustflag", "fq_type", "adjustment"),
    "adjust": ("adjust", "adjustflag", "fq_type", "adjustment"),
    "keywords": ("keywords", "title_keyword", "keyword", "query"),
    "local_key": ("local_key", "file_key", "doc_key"),
    "page_number": ("page_number", "start_page", "page", "page_no"),
    "news_id": ("news_id", "announcement_id", "doc_id"),
}


def _schema_props(json_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(json_schema, dict):
        return {}
    return dict(json_schema.get("properties") or {})


def _schema_required(json_schema: Dict[str, Any]) -> List[str]:
    if not isinstance(json_schema, dict):
        return []
    return [str(x) for x in (json_schema.get("required") or [])]


def map_slots_to_arguments(
    *,
    argument_slots: Dict[str, Any],
    json_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Map planner argument_slots onto schema parameters.

    Returns:
      {
        ok, arguments, argument_trace, missing_required, unused_slots
      }
    """
    props = _schema_props(json_schema)
    required = _schema_required(json_schema)
    prop_names = set(props.keys())
    arguments: Dict[str, Any] = {}
    argument_trace: List[Dict[str, Any]] = []
    unused_slots: List[str] = []

    slots = dict(argument_slots or {})
    for slot, value in slots.items():
        if value is None or value == "":
            # Empty adjust is often intentional for unadjusted prices.
            if str(slot) not in {"adjust", "adjustment"}:
                unused_slots.append(str(slot))
                continue
        aliases = SLOT_PARAM_ALIASES.get(str(slot), (str(slot),))
        matched = None
        for cand in aliases:
            if cand in prop_names:
                matched = cand
                break
        if matched is None:
            unused_slots.append(str(slot))
            continue
        # Enum / type soft checks.
        prop = props.get(matched) or {}
        valid = True
        reason = "ok"
        if "enum" in prop and value not in prop["enum"] and value != "":
            # allow empty string for adjust enums that include ""
            if "" in (prop.get("enum") or []) and value == "":
                valid = True
            else:
                valid = False
                reason = f"enum_mismatch:{value!r}"
        if valid:
            arguments[matched] = value
        argument_trace.append(
            {
                "source_slot": str(slot),
                "normalized_value": value,
                "target_parameter": matched,
                "validation_result": "valid" if valid else "invalid",
                "reason": reason,
            }
        )
        if not valid:
            # Do not bind invalid enum values.
            arguments.pop(matched, None)

    missing = [r for r in required if r not in arguments]
    has_invalid = any(t["validation_result"] == "invalid" for t in argument_trace)
    return {
        "ok": len(missing) == 0 and not has_invalid,
        "arguments": arguments,
        "argument_trace": argument_trace,
        "missing_required": missing,
        "unused_slots": unused_slots,
    }


def can_deterministic_bind(
    *,
    argument_slots: Dict[str, Any],
    json_schema: Dict[str, Any],
    candidate_count: int,
) -> Tuple[bool, Dict[str, Any]]:
    """True when exactly one candidate and all required args map cleanly."""
    mapped = map_slots_to_arguments(
        argument_slots=argument_slots, json_schema=json_schema
    )
    if candidate_count != 1:
        return False, mapped
    if mapped["missing_required"]:
        return False, mapped
    if any(t["validation_result"] == "invalid" for t in mapped["argument_trace"]):
        return False, mapped
    return True, mapped
