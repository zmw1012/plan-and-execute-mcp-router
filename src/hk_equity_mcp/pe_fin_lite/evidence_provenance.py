"""Minimal evidence provenance classes for PE-Fin Lite (not an ontology).

Classes:
- observed: directly from MCP result values
- grounded_input: from question / validated invoke arguments (context fields only)
- derived: deterministic function of observed (+ optional grounded_input)
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROVENANCE_CONTRACT_VERSION = "pe_fin_lite_provenance_v1"
EVIDENCE_ACCEPTANCE_VERSION = "pe_fin_lite_evidence_acceptance_v1"

PROVENANCE_CLASSES = ("observed", "grounded_input", "derived")

# Answer-bearing: must be observed or derived-from-observed (never grounded_input alone).
ANSWER_BEARING_FIELDS = frozenset(
    {
        "revenue",
        "attributable_profit",
        "basic_eps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "news_id",
        "snippet",
        "trade_date",
        "title",
        "release_date",
        "local_key",
        "page_number",
        "source_url",
        "report_date",
        "currency",
    }
)

# Context/scope: may be satisfied by validated args / question grounding.
CONTEXT_FIELDS = frozenset(
    {
        "issuer_code",
        "company_name",
        "adjustment",
        "fiscal_period",
    }
)

_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def provenance_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": PROVENANCE_CONTRACT_VERSION,
            "classes": list(PROVENANCE_CLASSES),
            "answer_bearing": sorted(ANSWER_BEARING_FIELDS),
            "context": sorted(CONTEXT_FIELDS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def evidence_acceptance_hash() -> str:
    blob = json.dumps(
        {
            "version": EVIDENCE_ACCEPTANCE_VERSION,
            "answer_bearing_must_be_observed_or_derived": True,
            "context_may_be_grounded_input": True,
            "annual_report_date_to_fiscal_period": True,
            "market_symbol_adjust_grounding": True,
            "no_question_target_value_as_observed": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_field_provenance(
    *,
    value: Any,
    provenance_class: str,
    source: str,
    ref: Optional[str] = None,
    derivation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if provenance_class not in PROVENANCE_CLASSES:
        raise ValueError(f"unknown provenance_class: {provenance_class}")
    out: Dict[str, Any] = {
        "value": value,
        "provenance_class": provenance_class,
        "source": source,
        "ref": ref,
    }
    if derivation is not None:
        out["derivation"] = derivation
    return out


def values_from_field_map(field_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in (field_map or {}).items()
        if (v.get("value") if isinstance(v, dict) else v) not in (None, "")
        or (isinstance(v, dict) and v.get("value") == "")
    }


def _strip_iso(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    m = _ISO_DATE.match(text)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _is_annual_indicator(indicator: Any) -> bool:
    ind = str(indicator or "").strip().lower()
    return ind in {"年度", "annual", "year", "yearly", "fy", "y"}


def derive_fiscal_period_from_annual_report_date(
    *,
    report_date: Any,
    indicator: Any = None,
    period_hint: Any = None,
) -> Optional[Dict[str, Any]]:
    """Deterministic: annual indicator + report_date → FY{report_date.year}.

    No company whitelist. Requires an annual indicator (e.g. 年度) or an explicit
    FY* period_hint paired with annual indicator semantics.
    """
    iso = _strip_iso(report_date)
    if not iso:
        return None
    hint = str(period_hint or "").strip().upper()
    annual = _is_annual_indicator(indicator) or (
        bool(re.match(r"^FY\d{4}$", hint)) and _is_annual_indicator(indicator or "年度")
    )
    if not annual:
        return None
    fy = f"FY{iso[:4]}"
    return make_field_provenance(
        value=fy,
        provenance_class="derived",
        source="deterministic_derivation",
        ref="fiscal_period_from_annual_report_date",
        derivation={
            "rule": "annual_report_date_to_fy",
            "inputs": {
                "report_date": iso,
                "indicator": indicator,
                "period_hint": period_hint,
            },
        },
    )


def grounded_context_from_arguments(
    *,
    arguments: Optional[Dict[str, Any]],
    step: Optional[Dict[str, Any]] = None,
    question_grounding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build grounded_input field map for context/scope fields only.

    Never treats target answer numbers from the question as observed evidence.
    """
    args = dict(arguments or {})
    step = step or {}
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    g = question_grounding or {}
    out: Dict[str, Dict[str, Any]] = {}

    from hk_equity_mcp.pe_fin_lite.issuer_code import canonicalize_hk_issuer_code

    # Try candidates independently — do NOT short-circuit on statement labels
    # in `symbol` (e.g. report_em uses symbol="利润表" while stock="00005").
    code = None
    code_ref = None
    for ref, candidate in (
        ("stock", args.get("stock")),
        ("stock_code", args.get("stock_code")),
        ("code", args.get("code")),
        ("symbol", args.get("symbol")),
        ("question.stock_code", g.get("stock_code")),
    ):
        canon = canonicalize_hk_issuer_code(candidate)
        if canon is not None:
            code = canon
            code_ref = ref
            break
    if code is not None:
        out["issuer_code"] = make_field_provenance(
            value=code,
            provenance_class="grounded_input",
            source="validated_argument_or_question",
            ref=code_ref or "stock_code",
        )

    if "adjust" in args or "adjustment" in args or "adjustment" in scope:
        adj = args["adjust"] if "adjust" in args else (
            args["adjustment"] if "adjustment" in args else scope.get("adjustment")
        )
        # Empty string is a valid unadjusted marker.
        if adj is not None:
            out["adjustment"] = make_field_provenance(
                value="" if adj is None else adj,
                provenance_class="grounded_input",
                source="validated_argument_or_scope",
                ref="adjust",
            )

    name = None
    ent = scope.get("entity")
    if isinstance(ent, dict):
        name = ent.get("name") or ent.get("company_name")
    elif isinstance(ent, str) and not re.fullmatch(r"\d{1,5}(\.HK)?", ent, re.I):
        name = ent
    if name:
        out["company_name"] = make_field_provenance(
            value=name,
            provenance_class="grounded_input",
            source="scope_constraints",
            ref="entity.name",
        )
    return out


def observed_field_map(
    values: Dict[str, Any],
    *,
    source: str = "mcp",
    ref: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    from hk_equity_mcp.pe_fin_lite.issuer_code import canonicalize_hk_issuer_code

    out: Dict[str, Dict[str, Any]] = {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        k = str(key)
        v = value
        if k == "issuer_code":
            canon = canonicalize_hk_issuer_code(value)
            if canon is None:
                # Drop non-code pollutants (statement names, free text).
                continue
            v = canon
        out[k] = make_field_provenance(
            value=v,
            provenance_class="observed",
            source=source,
            ref=ref,
        )
    return out


def apply_deterministic_derivations(
    field_map: Dict[str, Dict[str, Any]],
    *,
    arguments: Optional[Dict[str, Any]] = None,
    step: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Add derived fields (e.g. fiscal_period) without overwriting observed."""
    out = deepcopy(field_map)
    args = arguments or {}
    step = step or {}
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    if "fiscal_period" not in out or out["fiscal_period"].get("value") in (None, ""):
        report = out["report_date"].get("value") if "report_date" in out else None
        derived = derive_fiscal_period_from_annual_report_date(
            report_date=report,
            indicator=args.get("indicator"),
            period_hint=scope.get("period") or step.get("period"),
        )
        if derived is not None:
            out["fiscal_period"] = derived
    return out


def merge_field_maps(
    *maps: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge field provenance maps; record value conflicts."""
    merged: Dict[str, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []
    for fmap in maps:
        for key, entry in (fmap or {}).items():
            if key not in merged:
                merged[key] = deepcopy(entry)
                continue
            prev = merged[key].get("value")
            cur = entry.get("value")
            equivalent = prev == cur
            if key == "issuer_code":
                from hk_equity_mcp.pe_fin_lite.issuer_code import (
                    canonicalize_hk_issuer_code,
                    issuer_codes_equivalent,
                )

                # Drop non-canonical existing pollution when incoming is a real code.
                prev_c = canonicalize_hk_issuer_code(prev)
                cur_c = canonicalize_hk_issuer_code(cur)
                if prev_c is None and cur_c is not None:
                    merged[key] = deepcopy(entry)
                    merged[key]["value"] = cur_c
                    continue
                if cur_c is None and prev_c is not None:
                    continue
                if prev_c is None and cur_c is None:
                    # Neither is a valid issuer_code — drop both from merge.
                    merged.pop(key, None)
                    continue
                equivalent = issuer_codes_equivalent(prev, cur)
                if equivalent:
                    # Canonicalize stored value.
                    preferred = deepcopy(entry)
                    preferred["value"] = cur_c or prev_c
                    rank = {"observed": 3, "derived": 2, "grounded_input": 1}
                    if rank.get(str(entry.get("provenance_class")), 0) >= rank.get(
                        str(merged[key].get("provenance_class")), 0
                    ):
                        merged[key] = preferred
                    else:
                        merged[key] = deepcopy(merged[key])
                        merged[key]["value"] = prev_c or cur_c
                    continue
            if equivalent:
                # Prefer observed > derived > grounded_input when equal value.
                rank = {"observed": 3, "derived": 2, "grounded_input": 1}
                if rank.get(str(entry.get("provenance_class")), 0) > rank.get(
                    str(merged[key].get("provenance_class")), 0
                ):
                    merged[key] = deepcopy(entry)
                continue
            # Empty vs non-empty: prefer non-empty.
            if prev in (None, "") and cur not in (None, ""):
                merged[key] = deepcopy(entry)
                continue
            if cur in (None, "") and prev not in (None, ""):
                continue
            conflicts.append(
                {
                    "field": key,
                    "existing": merged[key],
                    "incoming": entry,
                }
            )
    return merged, conflicts


def required_outputs_satisfaction(
    field_map: Dict[str, Dict[str, Any]],
    required_outputs: Sequence[str],
) -> Dict[str, Any]:
    """Evaluate which required outputs are satisfied under acceptance rules."""
    present: List[str] = []
    missing: List[str] = []
    details: Dict[str, Any] = {}
    for req in required_outputs:
        entry = field_map.get(str(req))
        if not isinstance(entry, dict) or entry.get("value") is None:
            # adjustment may legitimately be ""
            if str(req) == "adjustment" and isinstance(entry, dict) and entry.get("value") == "":
                pass
            else:
                missing.append(str(req))
                details[str(req)] = {"satisfied": False, "reason": "absent"}
                continue
        pclass = str(entry.get("provenance_class") or "")
        if str(req) in ANSWER_BEARING_FIELDS and pclass == "grounded_input":
            missing.append(str(req))
            details[str(req)] = {
                "satisfied": False,
                "reason": "answer_bearing_cannot_be_grounded_input_alone",
                "provenance_class": pclass,
            }
            continue
        if pclass not in PROVENANCE_CLASSES:
            missing.append(str(req))
            details[str(req)] = {"satisfied": False, "reason": "unknown_provenance"}
            continue
        present.append(str(req))
        details[str(req)] = {
            "satisfied": True,
            "provenance_class": pclass,
            "source": entry.get("source"),
        }
    return {
        "ok": len(missing) == 0,
        "present": present,
        "missing": missing,
        "details": details,
    }


def scope_fingerprint(step: Dict[str, Any], arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Comparable scope identity for partial merge gating."""
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    args = arguments or {}
    entity = scope.get("entity") or step.get("entity")
    if isinstance(entity, dict):
        entity = entity.get("code") or entity.get("stock_code") or entity.get("name")
    return {
        "entity": str(entity or args.get("symbol") or args.get("stock") or "").lower(),
        "period": str(scope.get("period") or step.get("period") or "").upper(),
        "metric_basis": str(scope.get("metric_basis") or "").lower(),
        "trade_date": str(scope.get("trade_date") or step.get("date") or ""),
        "adjustment": scope.get("adjustment")
        if "adjustment" in scope
        else args.get("adjust", None),
    }


def scopes_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    for key in ("entity", "period", "metric_basis", "trade_date"):
        av = str(a.get(key) or "").strip()
        bv = str(b.get(key) or "").strip()
        if av and bv and av != bv:
            # Soft entity digit match
            if key == "entity":
                ad = re.sub(r"\D", "", av)
                bd = re.sub(r"\D", "", bv)
                if ad and bd and ad.zfill(5) == bd.zfill(5):
                    continue
            return False
    if a.get("adjustment") is not None and b.get("adjustment") is not None:
        if a.get("adjustment") != b.get("adjustment"):
            return False
    return True
