"""Execution Error-Aware Handling for PE-Fin Lite v1.2-EEA (local only).

No global replan. At most one local repair invocation + existing fallback path.
Never invents answer-bearing values from the question target number / gold.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXECUTION_ERROR_HANDLING_VERSION = "pe_fin_lite_execution_error_aware_v1"

EXECUTION_FAILURE_CLASSES = (
    "missing_required_argument",
    "invalid_argument_format",
    "argument_semantic_mismatch",
    "mcp_transport_or_protocol_failure",
    "empty_result",
    "no_matching_result",
    "gateway_envelope_failure",
    "adapter_failure",
    "scope_mismatch",
    "unsupported_operation",
    "disclosure_chain_stage_failure",
    "other",
)

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HK = re.compile(r"^\d{5}$")


def execution_error_handling_hash() -> str:
    blob = json.dumps(
        {
            "version": EXECUTION_ERROR_HANDLING_VERSION,
            "classes": list(EXECUTION_FAILURE_CLASSES),
            "max_local_repair_invocations": 1,
            "transport_retry_same_op": 1,
            "no_gold": True,
            "no_llm_guess_numbers": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def classify_execution_failure(
    *,
    invoke_result: Optional[Dict[str, Any]] = None,
    evaluated: Optional[Dict[str, Any]] = None,
    arguments: Optional[Dict[str, Any]] = None,
    required_schema_props: Optional[Sequence[str]] = None,
) -> str:
    """Mutual-exclusive primary failure class for one invoke attempt."""
    inv = invoke_result or {}
    ev = evaluated or {}
    args = arguments or {}
    err = str(inv.get("error") or ev.get("invoke_error") or ev.get("failure_reason") or "").lower()
    err_type = str(inv.get("error_type") or "").lower()

    if inv.get("success") is False or ev.get("gateway_success") is False or ev.get("mcp_transport_ok") is False:
        if any(x in err for x in ("timeout", "transport", "protocol", "connection", "http", "gateway")):
            return "mcp_transport_or_protocol_failure"
        if "unknown_operation" in err or err_type == "unknown_operation":
            return "unsupported_operation"
        if "envelope" in err or "normalization" in err:
            return "gateway_envelope_failure"
        if not (ev.get("values") or {}) and not err:
            return "empty_result"
        if "transport" in str(ev.get("failure_reason") or "").lower():
            return "mcp_transport_or_protocol_failure"
        # Fall through for argument-like gateway errors.

    req = list(required_schema_props or [])
    missing_args = [k for k in req if args.get(k) in (None, "")]
    if missing_args or "missing" in err and "argument" in err:
        return "missing_required_argument"
    if any(x in err for x in ("invalid argument", "invalid_arg", "type error", "schema")):
        return "invalid_argument_format"
    if "scope" in str(ev.get("failure_reason") or "").lower() or ev.get("scope_match") is False:
        if ev.get("gateway_success"):
            return "scope_mismatch"
    if "adapter" in err or (ev.get("normalization_success") is False and ev.get("gateway_success")):
        return "adapter_failure"
    if "no match" in err or "no_matching" in err or "not found" in err:
        return "no_matching_result"
    if ev.get("tool_result_valid") is False and not (ev.get("values") or {}):
        return "empty_result"
    if ev.get("gateway_success") is False:
        return "mcp_transport_or_protocol_failure"
    if "disclosure" in err:
        return "disclosure_chain_stage_failure"
    if "semantic" in err or "incompatible" in err:
        return "argument_semantic_mismatch"
    return "other"


def _period_to_window(period: Any) -> Tuple[Optional[str], Optional[str]]:
    text = str(period or "").strip().upper()
    m = re.fullmatch(r"FY(\d{4})", text)
    if m:
        y = m.group(1)
        return f"{y}-01-01", f"{y}-12-31"
    m = re.fullmatch(r"(\d{4})Q([1-4])", text)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        starts = {1: f"{y}-01-01", 2: f"{y}-04-01", 3: f"{y}-07-01", 4: f"{y}-10-01"}
        ends = {1: f"{y}-03-31", 2: f"{y}-06-30", 3: f"{y}-09-30", 4: f"{y}-12-31"}
        return starts[q], ends[q]
    return None, None


def local_repair_arguments(
    *,
    arguments: Dict[str, Any],
    step: Dict[str, Any],
    grounding: Optional[Dict[str, Any]] = None,
    contract: Optional[Dict[str, Any]] = None,
    evidence_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic argument repair from verified sources only.

    Sources: TaskSpec/step, question grounding, Evidence Store, schema defaults.
    Never copies gold/reference or invents news_id/local_key/page numbers.
    """
    from hk_equity_mcp.pe_fin_lite.issuer_code import canonicalize_hk_issuer_code

    args = deepcopy(arguments or {})
    g = grounding or {}
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    schema = (contract or {}).get("json_schema") or {}
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    evidence = evidence_values or {}
    repairs: List[str] = []

    def set_if_empty(key: str, value: Any, kind: str) -> None:
        if key not in props and key not in args and key not in required:
            # Still allow common aliases present in args or required
            if key not in required and key not in args:
                return
        if args.get(key) not in (None, "") or value in (None,):
            if args.get(key) not in (None, ""):
                return
        if value in (None,):
            return
        args[key] = value
        repairs.append(kind)

    # HK code canonicalization / fill
    for key in ("symbol", "stock", "stock_code", "code", "issuer_code"):
        raw = args.get(key)
        canon = canonicalize_hk_issuer_code(raw) if raw not in (None, "") else None
        if raw not in (None, "") and canon and str(raw) != canon:
            args[key] = canon
            repairs.append("hk_code_canonicalize")
        if args.get(key) in (None, ""):
            candidate = (
                g.get("stock_code")
                or evidence.get("issuer_code")
                or scope.get("entity")
                or step.get("entity")
            )
            if isinstance(candidate, dict):
                candidate = candidate.get("code") or candidate.get("stock_code")
            canon = canonicalize_hk_issuer_code(candidate)
            if canon:
                set_if_empty(key, canon, "fill_symbol_from_grounding")

    # ISO date normalization / fill
    for key in ("start_date", "from_date", "end_date", "to_date", "trade_date", "date"):
        raw = args.get(key)
        if isinstance(raw, str) and raw and not _ISO.match(raw.strip()):
            # Only accept already-ISO-looking substrings; no free-form parse invent.
            m = re.search(r"(20\d{2}-\d{2}-\d{2})", raw)
            if m:
                args[key] = m.group(1)
                repairs.append("iso_date_normalize")
        if args.get(key) in (None, ""):
            cand = g.get(key) or scope.get(key) or g.get("from_date" if "from" in key or key == "start_date" else "to_date")
            if key in {"trade_date", "date"}:
                cand = g.get("trade_date") or scope.get("trade_date") or step.get("date") or cand
            if cand and _ISO.match(str(cand).strip()):
                set_if_empty(key, str(cand).strip(), "fill_date_from_grounding")

    # Period → date window
    period = scope.get("period") or step.get("period")
    start, end = _period_to_window(period)
    if start and end:
        for key, val in (("start_date", start), ("from_date", start), ("end_date", end), ("to_date", end)):
            if key in props or key in args or key in required:
                set_if_empty(key, val, "period_to_date_window")

    # Enum / adjustment empty-string preserve
    if "adjust" in props or "adjustment" in props or "adjust" in args or "adjustment" in args:
        if "adjust" in args and args.get("adjust") is None:
            args["adjust"] = ""
            repairs.append("empty_adjustment_preserve")
        if "adjustment" in args and args.get("adjustment") is None:
            args["adjustment"] = ""
            repairs.append("empty_adjustment_preserve")
        if "adjust" in required and "adjust" not in args:
            args["adjust"] = scope.get("adjustment") if scope.get("adjustment") is not None else ""
            repairs.append("empty_adjustment_preserve")

    # Schema defaults (non-answer-bearing only)
    for key, prop in props.items():
        if key in args and args[key] not in (None, ""):
            continue
        if isinstance(prop, dict) and "default" in prop:
            default = prop["default"]
            # Refuse numeric defaults that look like answer-bearing metrics
            if key in {"close", "volume", "revenue", "attributable_profit", "basic_eps", "eps"}:
                continue
            args[key] = default
            repairs.append(f"schema_default:{key}")

    # Enum normalization: lowercase string enums when only case differs
    for key, prop in props.items():
        if not isinstance(prop, dict):
            continue
        enum = prop.get("enum")
        if not enum or key not in args:
            continue
        cur = args.get(key)
        if isinstance(cur, str):
            for opt in enum:
                if isinstance(opt, str) and opt.lower() == cur.lower() and opt != cur:
                    args[key] = opt
                    repairs.append("enum_normalize")
                    break

    return {
        "arguments": args,
        "changed": args != (arguments or {}),
        "repair_kinds": sorted(set(repairs)),
        "local_repair_attempted": True,
        "local_repair_kind": "+".join(sorted(set(repairs))) if repairs else "noop",
    }


def should_transport_retry(failure_class: str) -> bool:
    return failure_class == "mcp_transport_or_protocol_failure"


def should_argument_repair(failure_class: str) -> bool:
    return failure_class in {
        "missing_required_argument",
        "invalid_argument_format",
    }


def should_skip_primary_retry_use_fallback(failure_class: str) -> bool:
    """Semantic incompatibility: do not retry same tool; go to fallback if any."""
    return failure_class in {
        "argument_semantic_mismatch",
        "scope_mismatch",
        "unsupported_operation",
    }


def empty_or_no_result(failure_class: str) -> bool:
    return failure_class in {"empty_result", "no_matching_result"}
