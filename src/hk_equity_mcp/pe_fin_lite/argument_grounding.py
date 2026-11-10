"""Generic entity / date / title grounding for PE-Fin Lite (no issuer whitelist).

Extracts explicit HK codes and dates from question + plan step + scope_constraints.
Never invents a stock code from a company name alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.apv_fin.argument_grounding import (
    normalize_hk_symbol,
    normalize_iso_date,
)

ARGUMENT_GROUNDING_VERSION = "pe_fin_lite_argument_grounding_v1"

_DOT_HK = re.compile(r"\b(\d{1,5})\s*\.\s*HK\b", re.IGNORECASE)
_PAREN_CODE = re.compile(r"\(\s*(\d{1,5})\s*\)")
_BARE_FIVE = re.compile(r"(?<![\d.])(\d{5})(?![\d.])")
_ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_RANGE = re.compile(
    r"(20\d{2}-\d{2}-\d{2})\s*(?:\.\.|to|–|-|—)\s*(20\d{2}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_TITLE_QUOTED = re.compile(r"['\"]([^'\"]{3,120})['\"]")


def argument_grounding_hash() -> str:
    blob = json.dumps(
        {
            "version": ARGUMENT_GROUNDING_VERSION,
            "extracts": [
                "explicit_hk_codes",
                "iso_dates",
                "date_ranges",
                "quoted_title_keywords",
            ],
            "no_company_whitelist": True,
            "no_code_invention": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _texts_from_scope(scope: Any) -> List[str]:
    out: List[str] = []
    if scope is None:
        return out
    if isinstance(scope, str):
        out.append(scope)
        return out
    if isinstance(scope, dict):
        for key in (
            "entity",
            "period",
            "trade_date",
            "from_date",
            "to_date",
            "metric_basis",
            "adjustment",
            "title_keyword",
            "name",
            "code",
            "stock_code",
            "issuer_code",
        ):
            val = scope.get(key)
            if isinstance(val, dict):
                out.extend(_texts_from_scope(val))
            elif val is not None and val != "":
                out.append(str(val))
    return out


def extract_explicit_hk_codes(*blobs: Any) -> List[str]:
    """Return unique normalized 5-digit HK codes found explicitly in text."""
    found: List[str] = []
    seen = set()
    for blob in blobs:
        if blob is None:
            continue
        texts: List[str] = []
        if isinstance(blob, dict):
            texts.extend(_texts_from_scope(blob))
            texts.append(json.dumps(blob, ensure_ascii=False))
        else:
            texts.append(str(blob))
        for text in texts:
            for rx in (_DOT_HK, _PAREN_CODE, _BARE_FIVE):
                for m in rx.finditer(text or ""):
                    norm = normalize_hk_symbol(m.group(0) if "." in m.group(0) else m.group(1))
                    if norm and norm not in seen:
                        seen.add(norm)
                        found.append(norm)
    return found


def extract_iso_dates(*blobs: Any) -> List[str]:
    found: List[str] = []
    seen = set()
    for blob in blobs:
        text = blob if isinstance(blob, str) else json.dumps(blob or "", ensure_ascii=False)
        for m in _ISO.finditer(text or ""):
            norm = normalize_iso_date(m.group(1))
            if norm and norm not in seen:
                seen.add(norm)
                found.append(norm)
    return found


def extract_date_range(*blobs: Any) -> Tuple[Optional[str], Optional[str]]:
    for blob in blobs:
        text = blob if isinstance(blob, str) else json.dumps(blob or "", ensure_ascii=False)
        m = _RANGE.search(text or "")
        if m:
            a = normalize_iso_date(m.group(1))
            b = normalize_iso_date(m.group(2))
            if a and b:
                return a, b
    dates = extract_iso_dates(*blobs)
    if len(dates) >= 2:
        ordered = sorted(dates)
        return ordered[0], ordered[-1]
    if len(dates) == 1:
        return dates[0], dates[0]
    return None, None


def extract_title_keyword(*blobs: Any) -> Optional[str]:
    for blob in blobs:
        if isinstance(blob, dict):
            sc = blob.get("scope_constraints") if "scope_constraints" in blob else blob
            if isinstance(sc, dict) and sc.get("title_keyword"):
                return str(sc.get("title_keyword")).strip() or None
        text = blob if isinstance(blob, str) else json.dumps(blob or "", ensure_ascii=False)
        m = _TITLE_QUOTED.search(text or "")
        if m:
            return m.group(1).strip()
    return None


def collect_step_grounding(
    *,
    question: str,
    step: Dict[str, Any],
) -> Dict[str, Any]:
    """Company-generic grounding bag for binder / disclosure stages."""
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    entity_field = step.get("entity")
    codes = extract_explicit_hk_codes(
        question,
        step.get("objective"),
        scope,
        entity_field,
        step.get("period"),
        step.get("date"),
    )
    # Prefer code already present on scope.entity.code / entity string.
    preferred = None
    if isinstance(scope.get("entity"), dict):
        preferred = normalize_hk_symbol(
            scope["entity"].get("code")
            or scope["entity"].get("stock_code")
            or scope["entity"].get("issuer_code")
        )
    if preferred is None:
        preferred = normalize_hk_symbol(scope.get("entity") if isinstance(scope.get("entity"), str) else None)
    if preferred is None:
        preferred = normalize_hk_symbol(entity_field)
    stock_code = preferred or (codes[0] if codes else None)

    from_date, to_date = extract_date_range(
        question,
        step.get("objective"),
        scope,
        scope.get("from_date"),
        scope.get("to_date"),
        scope.get("trade_date"),
        step.get("date"),
        step.get("period"),
    )
    trade_date = normalize_iso_date(scope.get("trade_date")) or (
        from_date if from_date and from_date == to_date else None
    )
    title_keyword = (
        (str(scope.get("title_keyword")).strip() if scope.get("title_keyword") else None)
        or extract_title_keyword(question, step.get("objective"), scope)
    )
    company_name = None
    if isinstance(scope.get("entity"), dict):
        company_name = scope["entity"].get("name") or scope["entity"].get("company_name")
    elif isinstance(scope.get("entity"), str) and not normalize_hk_symbol(scope.get("entity")):
        company_name = str(scope.get("entity")).strip() or None

    return {
        "version": ARGUMENT_GROUNDING_VERSION,
        "stock_code": stock_code,
        "all_codes": codes,
        "company_name": company_name,
        "from_date": from_date,
        "to_date": to_date,
        "trade_date": trade_date,
        "period": scope.get("period") or step.get("period"),
        "metric_basis": scope.get("metric_basis"),
        "adjustment": scope.get("adjustment") if "adjustment" in scope else None,
        "title_keyword": title_keyword,
        "code_available": stock_code is not None,
        "needs_lookup": stock_code is None and bool(company_name),
    }


def apply_grounding_to_step(step: Dict[str, Any], grounding: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied step with entity/date fields filled from grounding."""
    out = dict(step)
    if grounding.get("stock_code") and not out.get("entity"):
        out["entity"] = grounding["stock_code"]
    if grounding.get("trade_date") and not out.get("date"):
        out["date"] = grounding["trade_date"]
    scope = dict(out.get("scope_constraints") or {})
    if grounding.get("stock_code"):
        ent = scope.get("entity")
        if isinstance(ent, dict):
            ent = dict(ent)
            ent.setdefault("code", grounding["stock_code"])
            scope["entity"] = ent
        elif not ent:
            scope["entity"] = grounding["stock_code"]
    for key in ("from_date", "to_date", "trade_date", "title_keyword", "period", "metric_basis"):
        if grounding.get(key) is not None and key not in scope:
            scope[key] = grounding[key]
    if grounding.get("adjustment") is not None and "adjustment" not in scope:
        scope["adjustment"] = grounding["adjustment"]
    if scope:
        out["scope_constraints"] = scope
    return out
