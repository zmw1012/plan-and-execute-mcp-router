"""GENERIC grounding validation for PE-Fin RePlan-ETOM (v1).

Version: ``pe_fin_replan_etom_grounding_v1``.

Contract (domain-agnostic; no HK ticker / date / market rules):

* NUMBERS in a final answer must match a verified ``observed`` or ``derived``
  evidence value (tolerance 1e-6 relative, or equality after percent scaling).
  The question is SCOPE ONLY and never grounds a number when
  ``completion == "answered"``. For ``insufficient_evidence`` answers a number
  may also echo a number that appears in the question (scope echo).
* IDENTIFIERS, URLS and DATES must exactly match (string equality after
  trivial normalization) some evidence value of ANY provenance class, or be an
  exact scope echo from the question.
* ``insufficient_evidence`` with an empty/short non-numeric answer always passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

GROUNDING_CONTRACT_VERSION = "pe_fin_replan_etom_grounding_v1"

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_./:-])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![A-Za-z0-9_./:-])-?\d+(?:\.\d+)?%?")
_IDENT_RE = re.compile(r"\b(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9][A-Za-z0-9_\-]{3,}\b")


def grounding_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": GROUNDING_CONTRACT_VERSION,
            "numbers": "observed_or_derived_only_question_scope_only",
            "identifiers_urls_dates": "exact_evidence_match_or_question_scope_echo",
            "insufficient_evidence_numbers": "evidence_or_question_echo",
            "generic": True,
            "no_hk_rules": True,
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


def _numbers_in(text: str, *, drop: Iterable[str] = ()) -> List[float]:
    scrubbed = text
    for chunk in drop:
        scrubbed = scrubbed.replace(chunk, " ")
    out = []
    for m in _NUMBER_RE.finditer(scrubbed):
        val = _to_float(m.group(0))
        if val is not None:
            out.append(val)
    return out


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
        # rounded display of a numeric evidence value (<= 2 dp)
        if round(y, 2) == x or round(y, 1) == x or round(y, 0) == x:
            return True
    return False


def validate_replanner_response(
    *,
    response: Dict[str, Any],
    question: str,
    evidence_values: Dict[str, Any],
    provenance_class_by_key: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return ``{ok, reasons, classifications}``; fail closed on ungrounded claims."""
    completion = str(response.get("completion") or "")
    answer = str(response.get("final_answer") or "")
    classes = dict(provenance_class_by_key or {})

    obs_derived: Dict[str, Any] = {}
    for key, val in evidence_values.items():
        # Unclassified values default to observed only if the caller gave no class map.
        klass = classes.get(key, "observed" if not classes else None)
        if klass in ("observed", "derived"):
            obs_derived[key] = val
    all_evidence = dict(evidence_values)

    numeric_pool: Set[float] = set()
    for v in obs_derived.values():
        _walk_numbers(v, numeric_pool)
    evidence_text = _evidence_strings(all_evidence.values())
    question_text = question or ""
    q_numbers: Set[float] = set(_numbers_in(question_text))

    reasons: List[str] = []
    classifications: Dict[str, List[str]] = {
        "numbers_supported": [],
        "numbers_scope_echo": [],
        "numbers_ungrounded": [],
        "dates_ok": [],
        "dates_ungrounded": [],
        "urls_ok": [],
        "urls_ungrounded": [],
        "identifiers_ok": [],
        "identifiers_ungrounded": [],
    }

    urls = extract_urls(answer)
    dates = extract_dates(answer)
    ident_drop = list(urls)
    idents = extract_identifiers(answer, drop=ident_drop)
    num_drop = list(urls) + [m.group(0) for m in _DATE_RE.finditer(answer)] + list(idents)
    numbers = _numbers_in(answer, drop=num_drop)

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
        if ident in evidence_text or ident in question_text:
            classifications["identifiers_ok"].append(ident)
        else:
            classifications["identifiers_ungrounded"].append(ident)
            reasons.append(f"identifier_not_in_evidence:{ident[:60]}")
    for n in numbers:
        if _number_supported(n, numeric_pool):
            classifications["numbers_supported"].append(repr(n))
        elif completion == "insufficient_evidence" and _number_supported(n, q_numbers):
            classifications["numbers_scope_echo"].append(repr(n))
        else:
            classifications["numbers_ungrounded"].append(repr(n))
            reasons.append(f"number_not_in_observed_or_derived:{n}")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "classifications": classifications,
        "contract_version": GROUNDING_CONTRACT_VERSION,
        "completion": completion,
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
