"""GENERIC grounding validation for PE-Fin RePlan-ETOM (v2).

Version: ``pe_fin_replan_etom_grounding_v2``.

Contract (domain-agnostic; no HK ticker / date / market rules; no task-id branches):

Number classification in a final Replanner response
---------------------------------------------------
1. ``formatting_enumeration_ignored``
   Pure list / step markers such as ``1.``, ``2)``, ``Step 3:`` at enumeration
   positions. Never a global skip of all integers.

2. ``question_constraint_ok``
   Numbers that appear verbatim (or safely canonicalized) in the original
   question, or as ``grounded_input`` evidence recorded from the question.
   Allowed as constraints / parameters / ranges / task description echoes.
   Never labeled as ``numbers_supported`` (that would disguise a question-only
   target as observed execution evidence).

3. ``verified_route_identifier``
   Numbers that are components of a verified routed server/tool/operation
   identifier when that identifier string also appears in the response
   (exact metadata match). Uncited / unverified identifiers stay rejected.

4. ``numbers_supported``
   Answer-bearing numbers from ``observed`` or ``derived`` evidence only.

5. ``ungrounded_answer_value``
   Any remaining number — fail closed.

Identifiers / URLs / dates still require exact evidence or question scope echo.
Routing-only runs typically have empty observed business values; that is OK
when the response only uses the classes above.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

GROUNDING_CONTRACT_VERSION = "pe_fin_replan_etom_grounding_v2"

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_./:-])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![A-Za-z0-9_./:-])-?\d+(?:\.\d+)?%?"
)
_IDENT_RE = re.compile(
    r"\b(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9][A-Za-z0-9_\-]{3,}\b"
)
# List / step markers only at enumeration positions (not arbitrary integers).
_ENUM_RE = re.compile(
    r"(?m)(?:^|[\n\r]|;\s*)(?:(?:Step|步骤)\s*)?(\d{1,3})(?:[\.\)\:]|、)\s+"
)
_ENUM_STEP_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:Step|步骤)\s+(\d{1,3})(?=[\.\:\)]|\s)",
    re.IGNORECASE,
)


def grounding_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": GROUNDING_CONTRACT_VERSION,
            "classes": [
                "formatting_enumeration_ignored",
                "question_constraint_ok",
                "verified_route_identifier",
                "numbers_supported",
                "ungrounded_answer_value",
            ],
            "question_constraint": "exact_or_safe_canonical_match_never_observed_label",
            "enumeration": "position_aware_list_or_step_markers_only",
            "route_identifier": "verified_metadata_exact_match_in_response",
            "answer_numbers": "observed_or_derived_only",
            "identifiers_urls_dates": "exact_evidence_match_or_question_scope_echo",
            "generic": True,
            "no_hk_rules": True,
            "no_task_id_branches": True,
            "routing_only_compatible": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _to_float(token: str) -> Optional[float]:
    t = token.strip().replace(",", "")
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def _canonical_number_key(val: float) -> str:
    if float(val).is_integer():
        return str(int(val))
    return repr(float(val))


def _numbers_in(text: str, *, drop: Iterable[str] = ()) -> List[float]:
    scrubbed = text
    for chunk in drop:
        scrubbed = scrubbed.replace(chunk, " ")
    out: List[float] = []
    for m in _NUMBER_RE.finditer(scrubbed):
        val = _to_float(m.group(0))
        if val is not None:
            out.append(val)
    return out


def _number_matches_with_spans(
    text: str, *, drop_spans: Sequence[Tuple[int, int]] = ()
) -> List[Tuple[float, int, int, str]]:
    """Return (value, start, end, raw_token) for number matches outside drop spans."""
    out: List[Tuple[float, int, int, str]] = []
    for m in _NUMBER_RE.finditer(text or ""):
        start, end = m.start(), m.end()
        if any(start >= ds and end <= de for ds, de in drop_spans):
            continue
        raw = m.group(0)
        val = _to_float(raw)
        if val is not None:
            out.append((val, start, end, raw))
    return out


def extract_enumeration_spans(text: str) -> List[Tuple[int, int, float, str]]:
    """Spans of formatting list/step numbers only (not arbitrary integers)."""
    spans: List[Tuple[int, int, float, str]] = []
    seen: Set[Tuple[int, int]] = set()
    for m in list(_ENUM_RE.finditer(text or "")) + list(_ENUM_STEP_RE.finditer(text or "")):
        raw = m.group(1)
        val = _to_float(raw)
        if val is None:
            continue
        abs_start = m.start(1)
        abs_end = m.end(1)
        key = (abs_start, abs_end)
        if key in seen:
            continue
        seen.add(key)
        spans.append((abs_start, abs_end, float(val), raw))
    return spans


def extract_urls(text: str) -> List[str]:
    return [u.rstrip(".,;:") for u in _URL_RE.findall(text or "")]


def extract_dates(text: str) -> List[str]:
    return [f"{int(y):04d}-{int(m):02d}-{int(d):02d}" for y, m, d in _DATE_RE.findall(text or "")]


def extract_identifiers(text: str, *, drop: Iterable[str] = ()) -> List[str]:
    scrubbed = text or ""
    for chunk in drop:
        scrubbed = scrubbed.replace(chunk, " ")
    return [m.group(0) for m in _IDENT_RE.finditer(scrubbed)]


def _walk_numbers(value: Any, out: Set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return
        out.add(float(value))
    elif isinstance(value, str):
        for n in _numbers_in(value):
            out.add(n)
    elif isinstance(value, dict):
        for v in value.values():
            _walk_numbers(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_numbers(v, out)


def _evidence_strings(values: Iterable[Any]) -> str:
    chunks = []
    for v in values:
        chunks.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str))
    return "\n".join(chunks)


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _number_supported(x: float, pool: Set[float]) -> bool:
    for y in pool:
        if _close(x, y) or _close(x, y * 100.0) or _close(x / 100.0, y):
            return True
        if round(y, 2) == x or round(y, 1) == x or round(y, 0) == x:
            return True
    return False


def _collect_route_identifiers(route_records: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
    ids: List[str] = []
    for rec in route_records or []:
        if not isinstance(rec, dict):
            continue
        for key in ("operation_id", "server_name", "tool_name"):
            val = rec.get(key)
            if val is None:
                continue
            text = str(val).strip()
            if text and text not in ids:
                ids.append(text)
    return ids


def _number_in_verified_route(
    *,
    value: float,
    start: int,
    end: int,
    answer: str,
    route_ids: Sequence[str],
) -> bool:
    """True when the number is part of a verified route id that also appears in the answer."""
    raw_forms = {_canonical_number_key(value), str(value)}
    for rid in route_ids:
        if rid not in answer:
            continue
        # Number must be a component of this verified identifier string.
        if not any(form in rid for form in raw_forms if form):
            # also allow float forms like 10.0 inside ids rarely
            if not any(form.rstrip("0").rstrip(".") in rid for form in list(raw_forms)):
                continue
        # Prefer span overlap with the identifier occurrence in the answer.
        idx = 0
        while True:
            pos = answer.find(rid, idx)
            if pos < 0:
                break
            if start >= pos and end <= pos + len(rid):
                return True
            idx = pos + 1
        # If the response cites the identifier elsewhere, still allow the number
        # when it is clearly a component of that verified id (metadata echo).
        if any(form in rid for form in raw_forms if form):
            return True
    return False


def validate_replanner_response(
    *,
    response: Dict[str, Any],
    question: str,
    evidence_values: Dict[str, Any],
    provenance_class_by_key: Optional[Dict[str, str]] = None,
    route_records: Optional[Sequence[Dict[str, Any]]] = None,
    predicted_nodes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return ``{ok, reasons, classifications}``; fail closed on ungrounded claims."""
    completion = str(response.get("completion") or "")
    answer = str(response.get("final_answer") or "")
    classes = dict(provenance_class_by_key or {})
    # ``predicted_nodes`` is an alias used by callers/tests for route metadata.
    if route_records is None and predicted_nodes is not None:
        route_records = predicted_nodes

    obs_derived: Dict[str, Any] = {}
    grounded_input: Dict[str, Any] = {}
    for key, val in evidence_values.items():
        klass = classes.get(key, "observed" if not classes else None)
        if klass in ("observed", "derived"):
            obs_derived[key] = val
        elif klass == "grounded_input":
            grounded_input[key] = val
    all_evidence = dict(evidence_values)

    numeric_pool: Set[float] = set()
    for v in obs_derived.values():
        _walk_numbers(v, numeric_pool)

    question_constraint_pool: Set[float] = set(_numbers_in(question or ""))
    for v in grounded_input.values():
        _walk_numbers(v, question_constraint_pool)

    evidence_text = _evidence_strings(all_evidence.values())
    question_text = question or ""
    route_ids = _collect_route_identifiers(route_records)

    reasons: List[str] = []
    classifications: Dict[str, List[str]] = {
        "formatting_enumeration_ignored": [],
        "question_constraint_ok": [],
        "verified_route_identifier": [],
        "numbers_supported": [],
        "ungrounded_answer_value": [],
        # legacy aliases kept for older audits / traces
        "numbers_scope_echo": [],
        "numbers_ungrounded": [],
        "dates_ok": [],
        "dates_ungrounded": [],
        "urls_ok": [],
        "urls_ungrounded": [],
        "identifiers_ok": [],
        "identifiers_ungrounded": [],
    }

    enum_spans = extract_enumeration_spans(answer)
    for _s, _e, val, raw in enum_spans:
        classifications["formatting_enumeration_ignored"].append(repr(float(val)))

    urls = extract_urls(answer)
    dates = extract_dates(answer)
    ident_drop = list(urls)
    idents = extract_identifiers(answer, drop=ident_drop)

    # Drop URL/date/ident/enumeration spans from answer-bearing number scan.
    drop_spans: List[Tuple[int, int]] = [(s, e) for s, e, _v, _r in enum_spans]
    for m in _URL_RE.finditer(answer):
        drop_spans.append((m.start(), m.end()))
    for m in _DATE_RE.finditer(answer):
        drop_spans.append((m.start(), m.end()))
    for ident in idents:
        idx = 0
        while True:
            pos = answer.find(ident, idx)
            if pos < 0:
                break
            drop_spans.append((pos, pos + len(ident)))
            idx = pos + 1

    for url in urls:
        if url in evidence_text or url in question_text:
            classifications["urls_ok"].append(url)
        else:
            classifications["urls_ungrounded"].append(url)
            reasons.append(f"url_not_in_evidence:{url[:80]}")
    for d in dates:
        norm_ev = evidence_text.replace("/", "-")
        norm_q = question_text.replace("/", "-")
        variants = {d, d.replace("-0", "-")}
        if any(v in norm_ev or v in norm_q for v in variants):
            classifications["dates_ok"].append(d)
        else:
            classifications["dates_ungrounded"].append(d)
            reasons.append(f"date_not_in_evidence:{d}")
    for ident in idents:
        if ident in evidence_text or ident in question_text or ident in route_ids:
            classifications["identifiers_ok"].append(ident)
        else:
            classifications["identifiers_ungrounded"].append(ident)
            reasons.append(f"identifier_not_in_evidence:{ident[:60]}")

    for val, start, end, raw in _number_matches_with_spans(answer, drop_spans=drop_spans):
        # Skip if this match is exactly an enumeration marker already classified.
        if any(start == es and end == ee for es, ee, _v, _r in enum_spans):
            continue
        if _number_supported(val, numeric_pool):
            classifications["numbers_supported"].append(repr(float(val)))
            continue
        if _number_in_verified_route(
            value=val, start=start, end=end, answer=answer, route_ids=route_ids
        ):
            classifications["verified_route_identifier"].append(repr(float(val)))
            continue
        if _number_supported(val, question_constraint_pool):
            # Constraint / parameter echo only — never "observed".
            classifications["question_constraint_ok"].append(repr(float(val)))
            classifications["numbers_scope_echo"].append(repr(float(val)))
            continue
        classifications["ungrounded_answer_value"].append(repr(float(val)))
        classifications["numbers_ungrounded"].append(repr(float(val)))
        reasons.append(f"ungrounded_answer_value:{val}")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "classifications": classifications,
        "contract_version": GROUNDING_CONTRACT_VERSION,
        "completion": completion,
        "enumeration_count": len(enum_spans),
        "route_identifier_count": len(route_ids),
        # True when any question-constraint number was accepted and none were
        # labeled as observed/derived ``numbers_supported``.
        "question_constraint_not_observed": bool(
            classifications["question_constraint_ok"]
        )
        and not classifications["numbers_supported"],
    }


def is_premature_insufficient(
    *,
    execution_round: int,
    max_execution_rounds: int,
    remaining_plan: Sequence[Dict[str, Any]],
    completed_ids: Set[str],
    decision_reason: Optional[str] = None,
) -> bool:
    """insufficient_evidence is premature before the final round while a remaining
    step is still ready to execute."""
    if execution_round >= max_execution_rounds:
        return False
    for step in remaining_plan:
        sid = str(step.get("id"))
        if sid in completed_ids:
            continue
        deps = [str(d) for d in (step.get("depends_on") or [])]
        if all(d in completed_ids for d in deps):
            return True
    return False
