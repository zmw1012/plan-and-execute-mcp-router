"""Evidence Acceptance Contract EEA v1 (post-hoc exploratory).

Partitions required outputs into answer_bearing / context / optional_supporting.
Optional/context gaps do not auto-fail a step when answer-bearing is satisfied.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

EVIDENCE_ACCEPTANCE_EEA_VERSION = "pe_fin_lite_evidence_acceptance_eea_v1"

ANSWER_BEARING_FIELDS = frozenset(
    {
        "revenue",
        "attributable_profit",
        "basic_eps",
        "eps",
        "diluted_eps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "title",
        "release_date",
        "trade_date",
        "snippet",
        # factual values requested as the answer itself
        "news_id",  # when the question asks for id
    }
)

CONTEXT_FIELDS = frozenset(
    {
        "issuer_code",
        "company_name",
        "fiscal_period",
        "currency",
        "adjustment",
        "report_date",
        "security_code",
        "stock_code",
    }
)

OPTIONAL_SUPPORTING_FIELDS = frozenset(
    {
        "source_url",
        "local_key",
        "page_number",
        "snippet",  # optional when not the core asked metric; dual-listed handled below
    }
)

# Operation-level aliases (no task/company hardcoding).
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "eps": ("eps", "basic_eps", "BASIC_EPS"),
    "basic_eps": ("basic_eps", "eps", "BASIC_EPS"),
    "diluted_eps": ("diluted_eps", "DILUTED_EPS"),
    "revenue": ("revenue", "operate_income", "OPERATE_INCOME"),
    "attributable_profit": (
        "attributable_profit",
        "holder_profit",
        "HOLDER_PROFIT",
        "net_profit",
    ),
    "open": ("open", "Open", "开盘", "opening_price"),
    "high": ("high", "High", "最高", "high_price"),
    "low": ("low", "Low", "最低", "low_price"),
    "close": ("close", "closing_price", "latest_close", "Close", "收盘"),
    "volume": ("volume", "trading_volume", "Volume", "成交量"),
    "report_date": ("report_date", "REPORT_DATE", "date"),
    "fiscal_period": ("fiscal_period", "FISCAL_YEAR", "period"),
    "currency": ("currency", "CURRENCY"),
    "issuer_code": (
        "issuer_code",
        "security_code",
        "stock_code",
        "SECURITY_CODE",
        "SECUCODE",
        "symbol",
        "code",
    ),
    "release_date": ("release_date", "publish_time", "date"),
    "news_id": ("news_id", "newsId"),
    "local_key": ("local_key", "localKey", "file_key"),
    "title": ("title", "Title", "headline"),
    "snippet": ("snippet", "text", "content", "page_text"),
    "page_number": ("page_number", "page"),
    "company_name": ("company_name", "SECURITY_NAME_ABBR", "name"),
    "trade_date": ("trade_date", "date", "Date", "日期"),
    "adjustment": ("adjustment", "adjust", "ADJUST"),
    "source_url": ("source_url", "url", "link"),
}

# (family, scale_to_base). Magnitude families convert within family.
_UNIT_TO_BASE = {
    "yuan": ("CNY", 1.0),
    "cny": ("CNY", 1.0),
    "rmb": ("CNY", 1.0),
    "hkd": ("HKD", 1.0),
    "hk$": ("HKD", 1.0),
    "usd": ("USD", 1.0),
    "us$": ("USD", 1.0),
    # Dimensionless magnitude scales (apply across numeric values).
    "unit": ("MAG", 1.0),
    "ones": ("MAG", 1.0),
    "thousand": ("MAG", 1e3),
    "million": ("MAG", 1e6),
    "billion": ("MAG", 1e9),
}


def units_compatible_or_convertible(a: Any, b: Any) -> Tuple[bool, Optional[float], Optional[str]]:
    """Return (ok, scale_b_to_a, shared_unit). scale applied as b_value * scale → a units."""
    if a is None or b is None or str(a).strip() == "" or str(b).strip() == "":
        return True, 1.0, None
    sa = str(a).strip().lower()
    sb = str(b).strip().lower()
    if sa == sb:
        return True, 1.0, sa
    # Allow "thousand yuan" style: magnitude + currency family via space split
    def parse(u: str) -> Tuple[Optional[str], float]:
        parts = u.replace("_", " ").split()
        mag = 1.0
        fam = None
        for p in parts:
            if p in _UNIT_TO_BASE:
                f, s = _UNIT_TO_BASE[p]
                if f == "MAG":
                    mag *= s
                else:
                    fam = f
                    mag *= s
            else:
                return None, 1.0
        if fam is None and mag != 1.0:
            fam = "MAG"
        return fam, mag

    fa, ma = parse(sa)
    fb, mb = parse(sb)
    if fa is None or fb is None:
        # Fallback exact token map
        if sa not in _UNIT_TO_BASE or sb not in _UNIT_TO_BASE:
            return False, None, None
        ua, fa_scale = _UNIT_TO_BASE[sa]
        ub, fb_scale = _UNIT_TO_BASE[sb]
        if ua != ub:
            return False, None, None
        return True, float(fb_scale) / float(fa_scale), ua
    if fa != fb:
        # Magnitude-only may convert against magnitude-only
        if {fa, fb} <= {"MAG"}:
            return True, float(mb) / float(ma), "MAG"
        # Allow converting pure magnitude onto a currency when the other side is currency-scaled
        if fa == "MAG" or fb == "MAG":
            return True, float(mb) / float(ma), fa if fa != "MAG" else fb
        return False, None, None
    return True, float(mb) / float(ma), fa


def evidence_acceptance_eea_hash() -> str:
    blob = json.dumps(
        {
            "version": EVIDENCE_ACCEPTANCE_EEA_VERSION,
            "answer_bearing": sorted(ANSWER_BEARING_FIELDS),
            "context": sorted(CONTEXT_FIELDS),
            "optional": sorted(OPTIONAL_SUPPORTING_FIELDS),
            "aliases": {k: list(v) for k, v in sorted(FIELD_ALIASES.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def field_role(name: str, *, required_outputs: Optional[Sequence[str]] = None) -> str:
    n = str(name)
    req = set(str(x) for x in (required_outputs or []))
    # If snippet/news_id are explicitly required as the asked output, treat as answer-bearing.
    if n in ANSWER_BEARING_FIELDS:
        return "answer_bearing"
    if n in OPTIONAL_SUPPORTING_FIELDS and n not in req:
        return "optional_supporting"
    if n in OPTIONAL_SUPPORTING_FIELDS and n in req and n in {"source_url", "local_key", "page_number"}:
        return "optional_supporting"
    if n in CONTEXT_FIELDS:
        return "context"
    if n in OPTIONAL_SUPPORTING_FIELDS:
        return "optional_supporting"
    # Unknown required fields default to answer-bearing (fail closed).
    return "answer_bearing"


def apply_aliases_to_pool(pool: Dict[str, Any]) -> Dict[str, Any]:
    """Lift alias keys into canonical names without dropping raw keys."""
    out = dict(pool or {})
    for canon, aliases in FIELD_ALIASES.items():
        if out.get(canon) not in (None, ""):
            continue
        for a in aliases:
            if a == canon:
                continue
            if out.get(a) not in (None, ""):
                out[canon] = out[a]
                break
    return out


def make_evidence_field_record(
    *,
    canonical_name: str,
    raw_name: str,
    value: Any,
    provenance_class: str,
    operation_id: Optional[str] = None,
    entity: Any = None,
    period: Any = None,
    basis: Any = None,
    unit: Any = None,
    transformation: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "canonical_name": canonical_name,
        "raw_name": raw_name,
        "value": value,
        "provenance_class": provenance_class,
        "operation_id": operation_id,
        "entity": entity,
        "period": period,
        "basis": basis,
        "unit": unit,
        "transformation": transformation,
    }


def required_outputs_satisfaction_eea(
    field_map: Dict[str, Dict[str, Any]],
    required_outputs: Sequence[str],
) -> Dict[str, Any]:
    """EEA satisfaction: only answer-bearing gaps block step success."""
    present: List[str] = []
    missing_answer: List[str] = []
    missing_context: List[str] = []
    missing_optional: List[str] = []
    details: Dict[str, Any] = {}
    for req in required_outputs:
        role = field_role(str(req), required_outputs=required_outputs)
        entry = field_map.get(str(req))
        absent = not isinstance(entry, dict) or entry.get("value") is None
        if str(req) == "adjustment" and isinstance(entry, dict) and entry.get("value") == "":
            absent = False
        if absent:
            details[str(req)] = {"satisfied": False, "role": role, "reason": "absent"}
            if role == "answer_bearing":
                missing_answer.append(str(req))
            elif role == "context":
                missing_context.append(str(req))
            else:
                missing_optional.append(str(req))
            continue
        pclass = str(entry.get("provenance_class") or "")
        if role == "answer_bearing" and pclass == "grounded_input":
            missing_answer.append(str(req))
            details[str(req)] = {
                "satisfied": False,
                "role": role,
                "reason": "answer_bearing_cannot_be_grounded_input_alone",
                "provenance_class": pclass,
            }
            continue
        present.append(str(req))
        details[str(req)] = {
            "satisfied": True,
            "role": role,
            "provenance_class": pclass,
            "source": entry.get("source"),
        }
    return {
        "ok": len(missing_answer) == 0,
        "present": present,
        "missing": list(missing_answer),  # blocking only
        "missing_answer_bearing": missing_answer,
        "missing_context": missing_context,
        "missing_optional": missing_optional,
        "details": details,
        "contract_version": EVIDENCE_ACCEPTANCE_EEA_VERSION,
    }


def merge_primary_fallback_eea(
    primary_map: Dict[str, Dict[str, Any]],
    fallback_map: Dict[str, Dict[str, Any]],
    *,
    primary_scope: Dict[str, Any],
    fallback_scope: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge only when entity/period/basis/units compatible; never silent overwrite."""
    from hk_equity_mcp.pe_fin_lite.evidence_provenance import scopes_compatible

    conflicts: List[Dict[str, Any]] = []
    if primary_scope and fallback_scope and not scopes_compatible(primary_scope, fallback_scope):
        return {
            "merged": dict(fallback_map),
            "ok": False,
            "conflicts": [
                {
                    "reason": "entity_or_period_or_basis_mismatch",
                    "primary": primary_scope,
                    "fallback": fallback_scope,
                }
            ],
            "primary_fallback_merge_status": "rejected_scope",
        }

    merged = deepcopy(primary_map)
    for key, entry in (fallback_map or {}).items():
        if key not in merged:
            merged[key] = deepcopy(entry)
            continue
        prev = merged[key]
        # Unit conversion when both numeric + unit metadata present
        pu = prev.get("unit")
        fu = entry.get("unit")
        ok_u, scale, _ = units_compatible_or_convertible(pu, fu)
        if pu and fu and not ok_u:
            conflicts.append({"field": key, "reason": "unit_mismatch", "primary": prev, "fallback": entry})
            continue
        pv = prev.get("value")
        fv = entry.get("value")
        if scale is not None and scale != 1.0 and isinstance(fv, (int, float)):
            fv = float(fv) * scale
            entry = deepcopy(entry)
            entry["value"] = fv
            entry["transformation"] = f"unit_scale:{scale}"
        if pv == fv or (pv in (None, "") and fv not in (None, "")):
            if pv in (None, "") and fv not in (None, ""):
                merged[key] = entry
            continue
        if fv in (None, ""):
            continue
        conflicts.append({"field": key, "reason": "value_conflict", "primary": prev, "fallback": entry})

    return {
        "merged": merged,
        "ok": len(conflicts) == 0,
        "conflicts": conflicts,
        "primary_fallback_merge_status": "merged" if not conflicts else "conflict",
    }


def forbid_question_target_as_observed(
    *,
    field: str,
    value: Any,
    question: str,
) -> bool:
    """True if value looks like a target number copied from the question text."""
    if field_role(field) != "answer_bearing":
        return False
    if not isinstance(value, (int, float)) and not (
        isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip() or "")
    ):
        return False
    text = str(question or "")
    token = str(value).strip()
    if not token:
        return False
    # Crude but conservative: exact numeric token appears in question.
    return bool(re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", text))
