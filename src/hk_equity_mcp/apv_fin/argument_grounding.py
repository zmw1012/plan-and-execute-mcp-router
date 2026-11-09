"""Company-generic argument normalization helpers for Tool Binding.

Provides candidate values from task_spec / plan step / evidence only.
Does not invent missing facts or read gold/reference answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_DOT_TICKER = re.compile(r"^\s*(\d{1,5})\s*\.\s*HK\s*$", re.IGNORECASE)
_PURE_DIGITS = re.compile(r"^\s*(\d{1,5})\s*$")
_ISO_DATE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
_COMPACT_DATE = re.compile(r"^\s*(\d{4})(\d{2})(\d{2})\s*$")


def normalize_hk_symbol(raw: Any) -> Optional[str]:
    """Normalize HK equity codes to 5-digit zero-padded form when unambiguous.

    Accepts 0700.HK / 00700 / 700 / 0005.HK / 00005 → '00700' / '00005'.
    Returns None when the value is not a reliable HK numeric code.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = _DOT_TICKER.match(text)
    if m:
        return m.group(1).zfill(5)
    m = _PURE_DIGITS.match(text)
    if m:
        return m.group(1).zfill(5)
    # Reject names / tickers that need a resolver — do not guess.
    return None


def normalize_iso_date(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    m = _ISO_DATE.match(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _COMPACT_DATE.match(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def normalize_period_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    upper = text.upper().replace(" ", "")
    # Keep common annual/interim labels as-is when already compact.
    if re.match(r"^FY\d{4}$", upper) or re.match(r"^HY\d{4}$", upper):
        return upper
    if re.match(r"^20\d{2}$", upper):
        return f"FY{upper}"
    return text[:80]


def normalize_adjust_enum(raw: Any, allowed: Optional[Sequence[str]] = None) -> Optional[str]:
    if raw is None:
        return "" if allowed is not None and "" in list(allowed) else None
    text = str(raw).strip().lower()
    allowed_list = list(allowed) if allowed is not None else ["", "qfq", "hfq"]
    # Map common synonyms only when unambiguous.
    aliases = {
        "": "",
        "none": "",
        "raw": "",
        "unadjusted": "",
        "qfq": "qfq",
        "forward": "qfq",
        "hfq": "hfq",
        "backward": "hfq",
    }
    mapped = aliases.get(text, text)
    if mapped in allowed_list:
        return mapped
    return None


def coerce_schema_value(raw: Any, schema: Dict[str, Any]) -> Tuple[Any, bool]:
    """Coerce only when unambiguous for the declared JSON schema type/enum."""
    if raw is None:
        return None, False
    types = schema.get("type")
    allowed_types = types if isinstance(types, list) else [types]
    if "enum" in schema:
        if raw in schema["enum"]:
            return raw, True
        if isinstance(raw, str):
            for opt in schema["enum"]:
                if str(opt).lower() == raw.lower():
                    return opt, True
        return None, False
    if "string" in allowed_types and not isinstance(raw, (dict, list)):
        return str(raw), True
    if "integer" in allowed_types:
        if isinstance(raw, bool):
            return None, False
        if isinstance(raw, int):
            return raw, True
        if isinstance(raw, str) and re.fullmatch(r"-?\d+", raw.strip()):
            return int(raw.strip()), True
        return None, False
    if "number" in allowed_types:
        if isinstance(raw, bool):
            return None, False
        if isinstance(raw, (int, float)):
            return float(raw), True
        if isinstance(raw, str):
            try:
                return float(raw.strip()), True
            except ValueError:
                return None, False
        return None, False
    if "boolean" in allowed_types:
        if isinstance(raw, bool):
            return raw, True
        if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
            return raw.strip().lower() == "true", True
        return None, False
    if "array" in allowed_types:
        if isinstance(raw, list):
            return raw, True
        return None, False
    return raw, True


def _ledger_field_values(ledger_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not ledger_summary:
        return out
    for item in ledger_summary.get("items") or ledger_summary.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("validation_status") or "")
        if status not in {"valid", "verified"}:
            continue
        fields = item.get("fields") or {}
        if isinstance(fields, dict):
            for k, v in fields.items():
                out.setdefault(str(k), v)
        claim = item.get("claim")
        if isinstance(claim, dict):
            for k, v in claim.items():
                out.setdefault(str(k), v)
    return out


def collect_grounding_candidates(
    *,
    question: str,
    task_spec: Dict[str, Any],
    step: Dict[str, Any],
    ledger_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect normalized candidate values for binder (not a final arg map)."""
    entities = list(task_spec.get("entities") or [])
    periods = list(task_spec.get("periods") or [])
    metrics = list(task_spec.get("metrics") or [])
    ledger_vals = _ledger_field_values(ledger_summary)

    symbol_candidates: List[Dict[str, Any]] = []
    for src_name, values in (
        ("task_spec", entities),
        ("evidence_ledger", [ledger_vals.get("symbol"), ledger_vals.get("hk_code")]),
        ("plan_step", [step.get("notes"), step.get("retrieval_query_hint")]),
    ):
        for val in values:
            norm = normalize_hk_symbol(val)
            if norm:
                symbol_candidates.append(
                    {"value": norm, "source": src_name, "raw": val}
                )

    # Also scan question for unambiguous NNNN.HK / digit codes only.
    for match in re.finditer(r"\b(\d{1,5})\.HK\b", question or "", flags=re.IGNORECASE):
        norm = normalize_hk_symbol(match.group(0))
        if norm:
            symbol_candidates.append(
                {"value": norm, "source": "task_spec", "raw": match.group(0)}
            )

    date_candidates: List[Dict[str, Any]] = []
    for src_name, values in (
        ("task_spec", periods + [task_spec.get("cutoff_date")]),
        ("evidence_ledger", [ledger_vals.get("start_date"), ledger_vals.get("end_date")]),
    ):
        for val in values:
            norm = normalize_iso_date(val)
            if norm:
                date_candidates.append(
                    {"value": norm, "source": src_name, "raw": val}
                )

    period_candidates: List[Dict[str, Any]] = []
    for val in periods:
        norm = normalize_period_label(val)
        if norm:
            period_candidates.append(
                {"value": norm, "source": "task_spec", "raw": val}
            )

    return {
        "symbol_candidates": symbol_candidates,
        "date_candidates": date_candidates,
        "period_candidates": period_candidates,
        "metric_hints": [str(m) for m in metrics if m],
        "adjust_default": "",
        "cutoff_date": task_spec.get("cutoff_date") or "2026-08-12",
        "ledger_verified_fields": {
            k: v for k, v in ledger_vals.items() if v is not None
        },
    }


def propose_arguments_for_schema(
    *,
    json_schema: Dict[str, Any],
    grounding: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Deterministic best-effort fill for required fields from grounding.

    Returns (arguments, argument_sources, missing_required).
    Never invents issuer codes when no normalized candidate exists.
    """
    props = dict((json_schema or {}).get("properties") or {})
    required = list((json_schema or {}).get("required") or [])
    args: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    missing: List[str] = []

    symbol_opts = list(grounding.get("symbol_candidates") or [])
    date_opts = list(grounding.get("date_candidates") or [])
    period_opts = list(grounding.get("period_candidates") or [])

    for name in required:
        schema = props.get(name) or {"type": "string"}
        value = None
        source = None
        lname = name.lower()
        if lname in {"symbol", "code", "stock", "ticker", "secu_code", "stock_code"}:
            if symbol_opts:
                value = symbol_opts[0]["value"]
                source = symbol_opts[0]["source"]
                # Some HK APIs want 5-digit; others accept as-is after normalize.
                value, ok = coerce_schema_value(value, schema)
                if not ok:
                    value = None
        elif lname in {"start_date", "end_date", "date", "trade_date", "as_of"}:
            if date_opts:
                # Prefer earliest for start_date, latest for end_date when multiple.
                ordered = sorted(date_opts, key=lambda x: x["value"])
                pick = ordered[0] if "start" in lname else ordered[-1]
                value, ok = coerce_schema_value(pick["value"], schema)
                source = pick["source"] if ok else None
                if not ok:
                    value = None
        elif lname in {"period", "report_period", "fiscal_period"}:
            if period_opts:
                value, ok = coerce_schema_value(period_opts[0]["value"], schema)
                source = period_opts[0]["source"] if ok else None
                if not ok:
                    value = None
        elif lname in {"adjust", "adjust_type"}:
            allowed = schema.get("enum")
            adj = normalize_adjust_enum(
                grounding.get("adjust_default"), allowed=allowed
            )
            if adj is not None:
                value, ok = coerce_schema_value(adj, schema)
                source = "default" if ok else None
                if not ok:
                    value = None
        elif lname in {"limit"} and "default" in schema:
            value = schema.get("default")
            source = "default"

        if value is None and "default" in schema and name not in args:
            # Only use schema default for optional; required with default still fills.
            value = schema.get("default")
            source = "default"

        if value is None:
            missing.append(name)
        else:
            args[name] = value
            sources[name] = source or "task_spec"

    # Optional adjust with empty default often required-by-practice for hist APIs.
    for name, schema in props.items():
        if name in args:
            continue
        if name.lower() in {"adjust", "adjust_type"} and "default" in schema:
            args[name] = schema.get("default")
            sources[name] = "default"

    return args, sources, missing
