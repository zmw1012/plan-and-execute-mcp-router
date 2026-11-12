"""Unified StepResult + deterministic success checks (no LLM Monitor)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence


STEP_STATUSES = (
    "completed",
    "empty",
    "unavailable",
    "failed",
    "skipped_due_to_dependency",
)


def make_step_result(
    *,
    execution_ok: bool = False,
    values: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    missing_outputs: Optional[Sequence[str]] = None,
    scope_match: bool = True,
    status: str = "failed",
    failure_reason: Optional[str] = None,
    mcp_transport_ok: Optional[bool] = None,
    tool_result_valid: Optional[bool] = None,
    required_output_satisfied: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    missing = list(missing_outputs or [])
    required_ok = required_output_satisfied
    if required_ok is None:
        required_ok = len(missing) == 0
    out = {
        "execution_ok": bool(execution_ok),
        "values": dict(values or {}),
        "provenance": dict(provenance or {}),
        "missing_outputs": missing,
        "scope_match": bool(scope_match),
        "status": status,
        "failure_reason": failure_reason,
        "mcp_transport_ok": mcp_transport_ok,
        "tool_result_valid": tool_result_valid,
        "required_output_satisfied": bool(required_ok),
    }
    if extra:
        out.update(extra)
    return out


def retrieve_step_succeeded(result: Dict[str, Any]) -> bool:
    """Retrieve success = execution_ok AND missing_outputs==[] AND scope_match."""
    if not result.get("execution_ok"):
        return False
    if list(result.get("missing_outputs") or []):
        return False
    if not result.get("scope_match", False):
        return False
    return str(result.get("status") or "") == "completed"


def finalize_retrieve_status(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize status from component flags without an LLM monitor."""
    out = dict(result)
    missing = list(out.get("missing_outputs") or [])
    transport_ok = out.get("mcp_transport_ok")
    valid = out.get("tool_result_valid")
    scope = bool(out.get("scope_match", True))
    if transport_ok is False:
        out["execution_ok"] = False
        out["status"] = "failed"
        out["failure_reason"] = out.get("failure_reason") or "mcp_transport_or_protocol_failure"
    elif valid is False:
        out["execution_ok"] = False
        out["status"] = "empty" if not out.get("values") else "failed"
        out["failure_reason"] = out.get("failure_reason") or "invalid_or_empty_result"
    elif missing:
        out["execution_ok"] = False
        out["required_output_satisfied"] = False
        out["status"] = "unavailable"
        out["failure_reason"] = out.get("failure_reason") or "required_outputs_missing"
    elif not scope:
        out["execution_ok"] = False
        out["status"] = "unavailable"
        out["failure_reason"] = out.get("failure_reason") or "scope_mismatch"
    else:
        out["execution_ok"] = True
        out["required_output_satisfied"] = True
        out["status"] = "completed"
        out["failure_reason"] = None
    return out


def extract_values_from_envelope(
    envelope: Dict[str, Any],
    required_outputs: Sequence[str],
) -> Dict[str, Any]:
    """Best-effort deterministic extraction from normalized gateway envelope / rows."""
    values: Dict[str, Any] = {}
    if not isinstance(envelope, dict):
        return values
    candidates: List[Dict[str, Any]] = []
    for key in ("normalized", "adapted", "projected", "values"):
        obj = envelope.get(key)
        if isinstance(obj, dict):
            candidates.append(obj)
    rows = []
    for obj in candidates + [envelope]:
        maybe_rows = obj.get("rows") if isinstance(obj, dict) else None
        if isinstance(maybe_rows, list):
            rows.extend([r for r in maybe_rows if isinstance(r, dict)])
        for k, v in (obj.items() if isinstance(obj, dict) else []):
            if k in required_outputs and k not in values and v is not None and v != "":
                values[k] = v
    for req in required_outputs:
        if req in values:
            continue
        for row in rows:
            if row.get(req) is not None and row.get(req) != "":
                values[req] = row.get(req)
                break
            # Light aliases
            aliases = {
                "open": ("open", "Open", "开盘"),
                "high": ("high", "High", "最高"),
                "low": ("low", "Low", "最低"),
                "close": ("close", "Close", "收盘"),
                "trade_date": ("trade_date", "date", "日期"),
                "local_key": ("local_key", "localKey"),
                "news_id": ("news_id", "newsId"),
                "snippet": ("snippet", "text", "content"),
                "title": ("title", "Title"),
                "page_number": ("page_number", "page"),
                "release_date": ("release_date", "publish_time", "date"),
            }
            for alt in aliases.get(req, ()):
                if row.get(alt) is not None and row.get(alt) != "":
                    values[req] = row.get(alt)
                    break
            if req in values:
                break
    return values


def check_scope_match(
    values: Dict[str, Any],
    *,
    entity: Optional[str] = None,
    period: Optional[str] = None,
    date: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> bool:
    """Conservative deterministic scope check. Unknown → True when no constraint."""
    args = arguments or {}
    if date:
        observed = (
            values.get("trade_date")
            or values.get("date")
            or values.get("release_date")
            or args.get("trade_date")
            or args.get("date")
        )
        if observed is not None and str(observed) and str(date) not in str(observed):
            # Allow YYYYMMDD vs YYYY-MM-DD soft match
            compact_obs = str(observed).replace("-", "")
            compact_req = str(date).replace("-", "")
            if compact_req not in compact_obs and compact_obs not in compact_req:
                return False
    if period:
        observed_period = values.get("period") or args.get("period") or args.get("year")
        if observed_period is not None and str(observed_period):
            if str(period) not in str(observed_period) and str(observed_period) not in str(
                period
            ):
                return False
    if entity:
        if isinstance(entity, dict):
            entity = (
                entity.get("code")
                or entity.get("stock_code")
                or entity.get("issuer_code")
                or entity.get("name")
                or entity.get("company_name")
            )
        observed_entity = (
            values.get("entity")
            or values.get("symbol")
            or values.get("ticker")
            or values.get("issuer_code")
            or args.get("symbol")
            or args.get("stock")
            or args.get("code")
            or args.get("stock_code")
        )
        if observed_entity is not None and str(observed_entity):
            ent = str(entity).lower().replace(".hk", "").strip()
            obs = str(observed_entity).lower().replace(".hk", "").strip()
            if ent and obs and ent not in obs and obs not in ent:
                # Digit ticker soft match
                ent_digits = "".join(ch for ch in ent if ch.isdigit())
                obs_digits = "".join(ch for ch in obs if ch.isdigit())
                if ent_digits and obs_digits and ent_digits.zfill(5) != obs_digits.zfill(5):
                    return False
    return True


def step_result_compact(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "execution_ok": result.get("execution_ok"),
        "status": result.get("status"),
        "missing_outputs": list(result.get("missing_outputs") or []),
        "scope_match": result.get("scope_match"),
        "values_keys": sorted((result.get("values") or {}).keys()),
        "provenance": result.get("provenance") or {},
        "failure_reason": result.get("failure_reason"),
    }


def dumps_step_result(result: Dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
