"""Deterministic answer grading for unified HK runs (evaluator grading v1.2).

Never uses LLM-as-judge. Uncertain / evidence-only items → manual_required.

v1.2 fixes (generic; no task_id special cases):
- Prefer structured gold_values / expected_status / tolerance / unit / metric.
- Do not score by comparing every number in reference_answer.
- Unicode minus, thousands separators, scientific notation, % vs decimal.
- Semantic units (decimal_return, percent, …) are not literal text requirements.
- Currency aliases (RMB↔CNY) and million/billion scale variants.
- Labeled multi-value gold (OHLC, unadjusted/qfq/hfq, MA+return, …).
- Extra intermediate numbers do not force a mismatch when the target matches.
- Refusal grading uses expected_status / answerability / refusal_spec; fabricated
  substitutes remain incorrect; capability-gap refusals without target values OK.
- answer_correct / evidence_correct / tool_selection_correct /
  overall_task_success are separate fields (never synthesized into one score).
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.unified_evaluator import REFUSAL_CLASSES, REFUSAL_STATUSES

GRADER_VERSION = "deterministic_v1.2.1"
GRADING_VERSION = "hk_unified_grading_v1.2.1"

# ASCII + common Unicode minus/dash signs normalized before numeric parse.
_MINUS_CHARS = str.maketrans(
    {
        "\u2212": "-",  # minus
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\uff0d": "-",  # fullwidth hyphen-minus
    }
)

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)(%?)"
)

# Label → value binding for multi-field gold dicts.
LABEL_PATTERNS: Dict[str, re.Pattern[str]] = {
    "open": re.compile(r"\bopen(?:ing)?(?:\s*price)?\b\s*[:=]?\s*", re.I),
    "high": re.compile(r"\bhigh(?:est)?(?:\s*price)?\b\s*[:=]?\s*", re.I),
    "low": re.compile(r"\blow(?:est)?(?:\s*price)?\b\s*[:=]?\s*", re.I),
    "close": re.compile(
        r"\b(?:close(?:ing)?|unadjusted\s+close)(?:\s*price)?\b\s*[:=]?\s*", re.I
    ),
    "unadjusted": re.compile(
        r"\b(?:unadjusted|adjust\s*=\s*[\"']{0,2}|raw)\b[^\d\-+]{0,40}", re.I
    ),
    "qfq": re.compile(r"\b(?:qfq|forward[-\s]?adjusted|前复权)\b[^\d\-+]{0,40}", re.I),
    "hfq": re.compile(r"\b(?:hfq|backward[-\s]?adjusted|后复权)\b[^\d\-+]{0,40}", re.I),
    "ma5": re.compile(r"\b(?:ma\s*5|ma5|5[-\s]?day\s+ma)\b[^\d\-+]{0,40}", re.I),
    "simple_return": re.compile(
        r"\b(?:simple\s+return|window\s+simple\s+return|return)\b[^\d\-+]{0,40}", re.I
    ),
    "reported": re.compile(r"\b(?:reported|ifrs|statutory)\b[^\d\-+]{0,40}", re.I),
    "adjusted": re.compile(
        r"\b(?:adjusted|non[-\s]?ifrs|non[-\s]?gaap)\b[^\d\-+]{0,40}", re.I
    ),
}

# Units that describe representation / metric type, not a literal currency token.
SEMANTIC_UNITS = frozenset(
    {
        "decimal",
        "decimal_return",
        "decimal_excess_return",
        "decimal_drawdown",
        "decimal_volatility",
        "ratio",
        "percent",
        "percentage",
        "percentage_points",
        "pp",
        "mixed",
        "identity",
        "index_points",
        "index points",
        "points",
    }
)

CURRENCY_ALIASES = {
    "cny": frozenset({"cny", "rmb", "¥", "￥", "人民币", "renminbi"}),
    "hkd": frozenset({"hkd", "hk$", "hk $", "港币", "港元"}),
    "usd": frozenset({"usd", "us$", "u.s. dollar", "美元"}),
}


def grading_identity() -> Dict[str, str]:
    return {
        "grader_version": GRADER_VERSION,
        "grading_version": GRADING_VERSION,
    }


def _answer_text(final_answer: Any) -> str:
    if final_answer is None:
        return ""
    if isinstance(final_answer, dict):
        parts = [
            str(final_answer.get("final_answer") or ""),
            str(final_answer.get("status") or ""),
            str(final_answer.get("reasoning_brief") or ""),
        ]
        return " ".join(parts)
    return str(final_answer)


def _normalize_text(text: str) -> str:
    return (text or "").translate(_MINUS_CHARS)


def _predicted_status(row: Dict[str, Any]) -> str:
    ps = row.get("predicted_status")
    if ps:
        return str(ps).strip().lower()
    fa = row.get("final_answer")
    if isinstance(fa, dict) and fa.get("status"):
        return str(fa.get("status")).strip().lower()
    return ""


def _parse_number_token(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", ""))
    except Exception:
        return None


def extract_number_spans(text: str) -> List[Dict[str, Any]]:
    """Extract numbers with optional trailing % and character spans."""
    norm = _normalize_text(text)
    out: List[Dict[str, Any]] = []
    for m in NUMBER_RE.finditer(norm):
        raw, pct = m.group(1), m.group(2)
        val = _parse_number_token(raw)
        if val is None:
            continue
        out.append(
            {
                "value": val,
                "raw": raw,
                "is_percent": bool(pct),
                "start": m.start(),
                "end": m.end(),
                "span": norm[m.start() : m.end()],
            }
        )
    return out


def extract_numbers(text: str) -> List[float]:
    return [span["value"] for span in extract_number_spans(text)]


def half_up(value: float, ndigits: int) -> float:
    """Round half-up to ndigits (banker's rounding avoided)."""
    if ndigits < 0:
        raise ValueError("ndigits must be >= 0")
    q = Decimal("1").scaleb(-ndigits)
    try:
        return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return round(value + (0.0 if value >= 0 else -0.0), ndigits)


def _normalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return ""
    u = str(unit).strip().lower()
    u = u.replace("_", " ").replace("-", " ")
    u = re.sub(r"\s+", " ", u)
    aliases = {
        "cny million": "cny million",
        "rmb million": "cny million",
        "cny_million": "cny million",
        "usd million": "usd million",
        "hkd": "hkd",
        "usd": "usd",
        "cny": "cny",
        "rmb": "cny",
        "shares": "shares",
        "percent": "percent",
        "percentage": "percent",
        "percentage points": "percentage_points",
        "pp": "percentage_points",
        "index points": "index_points",
        "index point": "index_points",
        "cny per share": "cny per share",
        "rmb per share": "cny per share",
        "usd per share": "usd per share",
        "hkd per share": "hkd per share",
    }
    return aliases.get(u, u)


def _is_semantic_unit(unit: Optional[str]) -> bool:
    norm = _normalize_unit(unit)
    if not norm:
        return True
    if norm in SEMANTIC_UNITS:
        return True
    # e.g. decimal_return already lowercased with spaces
    compact = norm.replace(" ", "_")
    return compact in SEMANTIC_UNITS or norm.startswith("decimal")


def _haystack(text: str, *, keep_underscore: bool = False) -> str:
    out = _normalize_text(text or "").lower()
    if not keep_underscore:
        out = out.replace("_", " ")
    return out


def _currency_mentioned(text: str, currency_key: str) -> bool:
    hay = _haystack(text)
    aliases = CURRENCY_ALIASES.get(currency_key, frozenset({currency_key}))
    return any(a in hay for a in aliases)


def _unit_present(text: str, unit: str) -> bool:
    """Whether answer expresses the gold unit (aliases + % + semantic)."""
    norm = _normalize_unit(unit)
    if not norm or _is_semantic_unit(norm):
        return True
    hay = _haystack(text)

    if norm == "percent":
        return ("%" in (text or "")) or ("percent" in hay) or ("pct" in hay)
    if norm == "percentage_points":
        return (
            ("percentage point" in hay)
            or ("pct point" in hay)
            or ("pp" in hay)
            or ("%" in (text or ""))
        )
    if norm == "index_points":
        return ("index" in hay and "point" in hay) or ("point" in hay) or True
    if norm == "shares":
        return "share" in hay

    if norm.endswith(" million"):
        cur = norm.split()[0]
        scale_ok = ("million" in hay) or ("million" in hay) or ("亿" in hay) or (
            "billion" in hay
        ) or ("trillion" in hay) or ("百万" in text or "")
        return _currency_mentioned(text, cur) and (
            scale_ok or _currency_mentioned(text, cur)
        )

    if norm.endswith(" per share"):
        cur = norm.split()[0]
        return _currency_mentioned(text, cur) and ("share" in hay or "eps" in hay)

    if norm in CURRENCY_ALIASES:
        return _currency_mentioned(text, norm)

    tokens = [t for t in norm.split() if t]
    if not tokens:
        return True
    # Currency-first token with aliases.
    if tokens[0] in CURRENCY_ALIASES:
        if not _currency_mentioned(text, tokens[0]):
            return False
        return all(t in hay for t in tokens[1:]) if len(tokens) > 1 else True
    return all(t in hay for t in tokens)


def _currency_present(text: str, currency: Optional[str]) -> bool:
    if not currency:
        return True
    norm = _normalize_unit(currency)
    if norm in CURRENCY_ALIASES:
        return _currency_mentioned(text, norm)
    return norm in _haystack(text)


def _values_close(
    observed: float,
    gold_value: float,
    *,
    tolerance_abs: Optional[float] = None,
    tolerance_rel: Optional[float] = None,
    tolerance_pp: Optional[float] = None,
) -> bool:
    diff = abs(observed - gold_value)
    if tolerance_pp is not None and diff <= float(tolerance_pp):
        return True
    if tolerance_abs is not None and diff <= float(tolerance_abs):
        return True
    if tolerance_rel is not None and gold_value != 0:
        if diff / abs(gold_value) <= float(tolerance_rel):
            return True
    if tolerance_abs is None and tolerance_rel is None and tolerance_pp is None:
        return math.isclose(observed, gold_value, rel_tol=0.0, abs_tol=1e-9)
    return False


def _scaled_tolerances(
    tag: str,
    *,
    tolerance_abs: Optional[float],
    tolerance_rel: Optional[float],
    tolerance_pp: Optional[float],
) -> Dict[str, Optional[float]]:
    """Rescale absolute/pp tolerances when comparing percent↔decimal forms."""
    tol_abs = tolerance_abs
    tol_pp = tolerance_pp
    if tag == "percent_as_decimal":
        if tol_abs is not None:
            tol_abs = float(tol_abs) / 100.0
        if tol_pp is not None:
            tol_pp = float(tol_pp) / 100.0
    elif tag == "decimal_as_percent":
        if tol_abs is not None:
            tol_abs = float(tol_abs) * 100.0
        if tol_pp is not None:
            tol_pp = float(tol_pp) * 100.0
    elif tag == "million_as_units":
        if tol_abs is not None:
            tol_abs = float(tol_abs) * 1_000_000.0
    elif tag == "million_as_billion":
        if tol_abs is not None:
            tol_abs = float(tol_abs) / 1_000.0
    elif tag == "million_as_trillion":
        if tol_abs is not None:
            tol_abs = float(tol_abs) / 1_000_000.0
    return {
        "tolerance_abs": tol_abs,
        "tolerance_rel": tolerance_rel,
        "tolerance_pp": tol_pp,
    }


def _candidate_numeric_forms(
    gold_value: float,
    *,
    unit: Optional[str],
    representation: Optional[str],
) -> List[Tuple[float, str]]:
    """Gold value plus equivalent percent/decimal/million-scale forms."""
    forms: List[Tuple[float, str]] = [(float(gold_value), "canonical")]
    rep = (representation or "").lower()
    norm = _normalize_unit(unit)

    if norm in {"percent", "percentage"} or "percentage_not_decimal" in rep:
        forms.append((float(gold_value) / 100.0, "percent_as_decimal"))
    if (
        norm.startswith("decimal")
        or "decimal_not_percent" in rep
        or norm in {"ratio", "decimal"}
    ):
        forms.append((float(gold_value) * 100.0, "decimal_as_percent"))

    if norm.endswith(" million"):
        forms.append((float(gold_value) * 1_000_000.0, "million_as_units"))
        forms.append((float(gold_value) / 1_000.0, "million_as_billion"))
        forms.append((float(gold_value) / 1_000_000.0, "million_as_trillion"))

    # Deduplicate by rounded key while preserving order.
    seen = set()
    out: List[Tuple[float, str]] = []
    for val, tag in forms:
        key = round(val, 12)
        if key in seen:
            continue
        seen.add(key)
        out.append((val, tag))
    return out


def _numeric_match_spans(
    spans: Sequence[Dict[str, Any]],
    gold_value: float,
    *,
    unit: Optional[str] = None,
    representation: Optional[str] = None,
    tolerance_abs: Optional[float] = None,
    tolerance_rel: Optional[float] = None,
    tolerance_pp: Optional[float] = None,
) -> bool:
    forms = _candidate_numeric_forms(
        gold_value, unit=unit, representation=representation
    )
    for span in spans:
        observed = float(span["value"])
        for target, tag in forms:
            tols = _scaled_tolerances(
                tag,
                tolerance_abs=tolerance_abs,
                tolerance_rel=tolerance_rel,
                tolerance_pp=tolerance_pp,
            )
            if _values_close(observed, target, **tols):
                return True
            # "10.46%" when gold is 0.1046 decimal: span value=10.46
            if span.get("is_percent"):
                as_decimal = observed / 100.0
                # Compare percent-token magnitude against canonical percent gold,
                # and decimal form against decimal gold, with scaled tolerances.
                if _values_close(
                    observed,
                    float(gold_value),
                    tolerance_abs=tolerance_abs,
                    tolerance_rel=tolerance_rel,
                    tolerance_pp=tolerance_pp,
                ):
                    return True
                dec_tols = _scaled_tolerances(
                    "percent_as_decimal",
                    tolerance_abs=tolerance_abs,
                    tolerance_rel=tolerance_rel,
                    tolerance_pp=tolerance_pp,
                )
                if abs(float(gold_value)) > 1.5:
                    # gold is percent-like; compare decimalized observation to gold/100
                    if _values_close(as_decimal, float(gold_value) / 100.0, **dec_tols):
                        return True
                else:
                    if _values_close(
                        as_decimal,
                        float(gold_value),
                        tolerance_abs=tolerance_abs,
                        tolerance_rel=tolerance_rel,
                        tolerance_pp=tolerance_pp,
                    ):
                        return True
    return False


def _is_year_token(raw: str, value: float) -> bool:
    token = raw.replace(",", "")
    return bool(re.fullmatch(r"\d{4}", token) and 1900 <= value <= 2100)


def _in_iso_date_context(window: str, match_start: int, raw: str) -> bool:
    """True when the number sits inside a YYYY-MM-DD / YYYY/MM/DD fragment."""
    token = raw.replace(",", "")
    # Only month/day-like integers participate in date fragments.
    if not re.fullmatch(r"\d{1,2}", token):
        return False
    left = window[max(0, match_start - 12) : match_start]
    right = window[match_start + len(token) : match_start + len(token) + 8]
    # YYYY-MM or YYYY-MM-DD boundaries.
    if re.search(r"\d{4}[-/]$", left):
        return True
    if re.search(r"\d{4}[-/]\d{1,2}[-/]$", left):
        return True
    # MM-DD continuing after a year-month prefix already consumed.
    if re.search(r"[-/]$", left) and re.match(r"^\d{1,2}\b", token + right):
        # Require a year somewhere nearby to the left.
        if re.search(r"\d{4}[-/]\d{1,2}[-/]$", left) or re.search(
            r"\d{4}[-/]$", left
        ):
            return True
    return False


def _extract_labeled_number(
    text: str, label: str, *, expected: Optional[float] = None, tol_abs: float = 0.05
) -> Optional[float]:
    pattern = LABEL_PATTERNS.get(label.lower())
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(label)}\b[^\d\-+]{{0,48}}", re.I)
    norm = _normalize_text(text)
    best: Optional[float] = None
    best_dist = float("inf")
    for m in pattern.finditer(norm):
        # Search a wider window; prefer '=' / ':' anchored numbers.
        window = norm[m.end() : m.end() + 220]
        # Stop early at next major clause marker when present.
        for stopper in ("(ii)", "(iii)", "(b)", "\n\n"):
            idx = window.find(stopper)
            if idx > 0:
                window = window[:idx]
                break
        candidates: List[float] = []
        for n in NUMBER_RE.finditer(window):
            raw = n.group(1)
            val = _parse_number_token(raw)
            if val is None:
                continue
            if _is_year_token(raw, val):
                continue
            if _in_iso_date_context(window, n.start(), raw):
                continue
            # Skip ticker-like zero-padded codes (00005).
            if re.fullmatch(r"0\d{4,}", raw.replace(",", "")):
                continue
            candidates.append(val)
            # Prefer first number after '=' or ':' near the match.
            prefix = window[: n.start()]
            if re.search(r"[:=]\s*$", prefix):
                if expected is None:
                    return val
                dist = abs(val - float(expected))
                if dist < best_dist:
                    best, best_dist = val, dist
        if expected is not None:
            for val in candidates:
                dist = abs(val - float(expected))
                if dist <= max(float(tol_abs), abs(float(expected)) * 1e-6 + 1e-9) and dist < best_dist:
                    best, best_dist = val, dist
            # If none within tolerance, still keep nearest for caller to reject.
            if best is None and candidates:
                nearest = min(candidates, key=lambda v: abs(v - float(expected)))
                return nearest
        elif candidates and best is None:
            return candidates[0]
    if best is not None:
        return best
    return None


def _gold_item_kind(item: Dict[str, Any]) -> str:
    if item.get("expected_behavior") or item.get("answerability"):
        return "refusal_spec"
    value = item.get("value")
    if isinstance(value, dict):
        # Boolean / identity payloads.
        if value and all(isinstance(v, bool) for v in value.values()):
            return "identity"
        if value and all(isinstance(v, str) for v in value.values()):
            return "identity"
        if any(isinstance(v, bool) for v in value.values()) and any(
            isinstance(v, str) for v in value.values()
        ):
            return "identity"
        if any(isinstance(v, (int, float)) for v in value.values()):
            return "labeled_multi"
        return "opaque"
    if "value" in item and isinstance(value, (int, float)):
        return "numeric"
    if "value" in item and isinstance(value, str):
        return "string"
    if item.get("field") and "value" in item:
        return "string"
    # Cross-domain fiscal window parts without a single value.
    if item.get("part") and (
        item.get("fy2026_start")
        or item.get("fy_start")
        or item.get("session_date")
        or item.get("note")
    ):
        if isinstance(item.get("value"), (int, float)):
            return "numeric"
        return "structured_part"
    return "opaque"


def classify_gold_values(gold_values: Any) -> str:
    if gold_values is None:
        return "missing"
    if isinstance(gold_values, list) and not gold_values:
        return "empty"
    if not isinstance(gold_values, list):
        return "opaque"
    kinds = {_gold_item_kind(x) for x in gold_values if isinstance(x, dict)}
    if not kinds:
        return "opaque"
    if kinds <= {"refusal_spec"}:
        return "refusal_spec"
    if kinds <= {"numeric"}:
        return "numeric"
    if kinds <= {"string"}:
        return "string"
    if kinds <= {"labeled_multi"}:
        return "labeled_multi"
    if kinds <= {"identity"}:
        return "identity"
    if kinds <= {"numeric", "string", "labeled_multi", "structured_part", "identity"}:
        return "mixed_auto"
    return "opaque"


def _tolerances_from_item(item: Dict[str, Any], key: Optional[str] = None) -> Dict[str, Optional[float]]:
    tol_abs = item.get("tolerance_abs")
    tol_rel = item.get("tolerance_rel")
    tol_pp = item.get("tolerance_pp")
    tolerances = item.get("tolerances")
    if key and isinstance(tolerances, dict):
        if f"{key}_abs" in tolerances:
            tol_abs = tolerances.get(f"{key}_abs")
        if f"{key}_rel" in tolerances:
            tol_rel = tolerances.get(f"{key}_rel")
        if f"{key}_pp" in tolerances:
            tol_pp = tolerances.get(f"{key}_pp")
        if key in tolerances and isinstance(tolerances[key], (int, float)):
            tol_abs = tolerances[key]
    top = item.get("tolerance")
    if isinstance(top, dict):
        tol_abs = top.get("abs", tol_abs)
        tol_rel = top.get("rel", tol_rel)
        tol_pp = top.get("pp", tol_pp)
    return {
        "tolerance_abs": float(tol_abs) if tol_abs is not None else None,
        "tolerance_rel": float(tol_rel) if tol_rel is not None else None,
        "tolerance_pp": float(tol_pp) if tol_pp is not None else None,
    }


def _grade_numeric_item(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    value = item.get("value")
    if not isinstance(value, (int, float)):
        return None, "non_numeric_value"
    spans = extract_number_spans(text)
    if not spans:
        return False, "no_numeric_in_answer"
    unit = item.get("unit")
    currency = item.get("currency")
    if unit and not _unit_present(text, str(unit)):
        return False, "unit_mismatch_or_missing"
    if currency and not _currency_present(text, str(currency)):
        return False, "currency_mismatch_or_missing"
    tols = _tolerances_from_item(item)
    ok = _numeric_match_spans(
        spans,
        float(value),
        unit=str(unit) if unit is not None else None,
        representation=item.get("representation"),
        **tols,
    )
    # Optional label preference: if metric/field/name present, try labeled first
    # but do not fail solely because label binding missed when global match works.
    label = item.get("metric") or item.get("field") or item.get("name") or item.get("price_field")
    if label and isinstance(label, str) and label.lower() not in {"close", "ohlc", "price"}:
        labeled = _extract_labeled_number(text, str(label))
        if labeled is not None:
            if _values_close(labeled, float(value), **tols) or _numeric_match_spans(
                [{"value": labeled, "is_percent": False}],
                float(value),
                unit=str(unit) if unit is not None else None,
                representation=item.get("representation"),
                **tols,
            ):
                return True, "numeric_match_labeled"
    return (True, "numeric_match") if ok else (False, "numeric_mismatch")


def _grade_string_item(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    value = item.get("value")
    if value is None:
        return None, "missing_string_value"
    needle = str(value).strip()
    if not needle:
        return None, "empty_string_value"
    hay = _haystack(text, keep_underscore=True)
    if needle.lower() in hay:
        return True, "string_match"
    # Also allow underscore/space equivalence for readable answers.
    if needle.lower().replace("_", " ") in _haystack(text):
        return True, "string_match"
    return False, "string_mismatch"


def _grade_labeled_multi(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    value = item.get("value")
    if not isinstance(value, dict) or not value:
        return None, "invalid_labeled_multi"
    unit = item.get("unit")
    if unit and not _is_semantic_unit(unit) and not _unit_present(text, str(unit)):
        return False, "unit_mismatch_or_missing"

    results: List[bool] = []
    reasons: List[str] = []
    for key, expected in value.items():
        if not isinstance(expected, (int, float)):
            # Non-numeric members handled by identity grader.
            continue
        tols = _tolerances_from_item(item, key=str(key))
        tol_for_bind = float(tols["tolerance_abs"] or 0.05)
        labeled = _extract_labeled_number(
            text, str(key), expected=float(expected), tol_abs=tol_for_bind
        )
        if labeled is None:
            # Fall back: any matching number (still target-focused per key).
            spans = extract_number_spans(text)
            ok = _numeric_match_spans(
                spans,
                float(expected),
                unit=str(unit) if unit is not None else None,
                representation=item.get("representation"),
                **tols,
            )
            results.append(ok)
            reasons.append(f"{key}:{'match' if ok else 'missing_label_or_mismatch'}")
            continue
        ok = _values_close(float(labeled), float(expected), **tols)
        if not ok:
            # percent/decimal variants for return-like keys
            ok = _numeric_match_spans(
                [{"value": float(labeled), "is_percent": False}],
                float(expected),
                unit=str(unit) if unit is not None else None,
                representation=item.get("representation"),
                **tols,
            )
        results.append(ok)
        reasons.append(f"{key}:{'match' if ok else 'mismatch'}")

    if not results:
        return None, "labeled_multi_no_numeric_fields"
    if all(results):
        return True, "labeled_multi_match:" + ",".join(reasons)
    return False, "labeled_multi_mismatch:" + ",".join(reasons)


def _grade_identity(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    value = item.get("value")
    if not isinstance(value, dict):
        return None, "invalid_identity"
    hay = _haystack(text)
    for key, expected in value.items():
        key_l = str(key).lower()
        if isinstance(expected, bool):
            if key_l in {"is_tencent", "tencent"}:
                ok = ("tencent" in hay or "腾讯" in text) if expected else (
                    "tencent" not in hay and "腾讯" not in text
                )
            elif key_l in {"hsi_is_index", "is_index"}:
                ok = (("index" in hay or "指数" in text) and ("hsi" in hay or "恒生" in text)) if expected else True
            else:
                # Generic boolean: require key token presence when True.
                token = key_l.replace("is_", "").replace("_", " ")
                ok = (token in hay) if expected else True
            if not ok:
                return False, f"identity_mismatch:{key}"
        elif isinstance(expected, str):
            if expected.lower() not in hay and expected not in (text or ""):
                return False, f"identity_mismatch:{key}"
        elif isinstance(expected, (int, float)):
            spans = extract_number_spans(text)
            tols = _tolerances_from_item(item, key=str(key))
            if not _numeric_match_spans(spans, float(expected), **tols):
                return False, f"identity_numeric_mismatch:{key}"
    return True, "identity_match"


def _grade_structured_part(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """Fiscal window / date parts: require start/end strings when present."""
    hay = _haystack(text)
    checks = []
    for key in (
        "fy2026_start",
        "fy2026_end",
        "fy_start",
        "fy_end",
        "session_date",
        "start",
        "end",
    ):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            checks.append(val.strip().lower() in hay or val.strip() in (text or ""))
    if item.get("value") is not None and isinstance(item.get("value"), (int, float)):
        ok, reason = _grade_numeric_item(text, item)
        return ok, reason
    if not checks:
        # Note-only part → cannot auto-grade from this item alone.
        return None, "structured_part_insufficient"
    return (True, "structured_part_match") if all(checks) else (False, "structured_part_mismatch")


_TICKER_IN_TEXT = re.compile(
    r"\b\d{1,5}\s*\.\s*HK\b|\(\s*\d{1,5}\s*\)|(?<![\d.])0\d{4}(?![\d.])",
    re.IGNORECASE,
)
_ISO_DATE_IN_TEXT = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def _grounded_numeric_raw_tokens(*blobs: str) -> set:
    """Numbers that appear as stock codes / dates in question or answer context."""
    tokens: set = set()
    for blob in blobs:
        text = blob or ""
        for m in _TICKER_IN_TEXT.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if digits:
                tokens.add(digits)
                tokens.add(digits.lstrip("0") or "0")
                tokens.add(str(int(digits)))
        for m in _ISO_DATE_IN_TEXT.finditer(text):
            raw = m.group(0)
            tokens.add(raw)
            tokens.add(raw.replace("-", ""))
            tokens.add(raw[:4])  # year
            tokens.add(raw[5:7])
            tokens.add(raw[8:10])
        # Bare question digits that look like HK codes (4–5 digits).
        for m in re.finditer(r"(?<![\d.])(\d{4,5})(?![\d.])", text):
            tokens.add(m.group(1))
            tokens.add(m.group(1).lstrip("0") or "0")
            tokens.add(str(int(m.group(1))))
    return tokens


def _answer_numeric_spans_excluding_grounded(
    text: str, *, question: str = ""
) -> List[Dict[str, Any]]:
    grounded = _grounded_numeric_raw_tokens(question, text)
    out: List[Dict[str, Any]] = []
    for span in extract_number_spans(text):
        raw = str(span.get("raw") or "").replace(",", "")
        compact = raw.replace(".", "")
        if raw in grounded or compact in grounded:
            continue
        # Skip ticker-like zero-padded codes and years.
        if re.fullmatch(r"0\d{3,}", raw):
            continue
        if re.fullmatch(r"20\d{2}", raw):
            continue
        # Skip number that is only part of NNNN.HK in the answer.
        start = int(span.get("start") or 0)
        end = int(span.get("end") or 0)
        window = (text or "")[max(0, start - 2) : end + 4]
        if re.search(r"\d{1,5}\s*\.\s*HK", window, flags=re.IGNORECASE):
            continue
        if re.search(r"\(\s*\d{1,5}\s*\)", window):
            continue
        out.append(span)
    return out


def _looks_like_fabricated_numeric(text: str, *, question: str = "") -> bool:
    """Heuristic: concrete money/share/price figures in a refusal answer.

    Stock codes, ISO dates, and other numbers already present in the question
    (or as ticker mentions in the answer) are not treated as fabricated values.
    """
    if not _answer_numeric_spans_excluding_grounded(text, question=question):
        return False
    hay = _haystack(text)
    cues = (
        "cny",
        "rmb",
        "hkd",
        "usd",
        "million",
        "revenue",
        "eps",
        "close",
        "shares",
        "price",
        "hk$",
        "¥",
    )
    return any(c in hay for c in cues)


def _has_capability_gap_language(text: str) -> bool:
    hay = _haystack(text)
    cues = (
        "unavailable",
        "cannot",
        "can't",
        "not available",
        "not reliably",
        "no access",
        "not supported",
        "blocked",
        "capability",
        "outside",
        "beyond the cutoff",
        "real-time",
        "realtime",
        "live quote",
        "not in this environment",
        "refuse",
        "unable to",
    )
    return any(c in hay for c in cues)


def grade_refusal(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    status = _predicted_status(row)
    text = _answer_text(row.get("final_answer"))
    question = str(row.get("question") or item.get("question") or "")
    expected = (
        item.get("expected_behavior")
        or item.get("expected_status")
        or row.get("gold_status")
        or ""
    )
    expected_l = str(expected).strip().lower()

    # Answering when a refusal/clarify was required.
    if status not in REFUSAL_STATUSES:
        # Soft language alone ("cannot") is insufficient if a substitute value is given.
        if _looks_like_fabricated_numeric(text, question=question):
            return {
                "auto_grade_status": "graded",
                "answer_correct": False,
                "reason": "refusal_required_but_answered_with_numeric_substitute",
            }
        return {
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": f"expected_refusal_got_status={status or 'missing'}",
        }

    if _looks_like_fabricated_numeric(text, question=question):
        return {
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": "refusal_with_fabricated_numeric",
        }

    # Status is a refusal class and no fabricated target value.
    if expected_l in REFUSAL_STATUSES or expected_l in {
        "refuse",
        "unanswerable",
        "should_refuse",
        "clarify",
    }:
        return {
            "auto_grade_status": "graded",
            "answer_correct": True,
            "reason": "refusal_status_ok",
        }

    if _has_capability_gap_language(text) or status in REFUSAL_STATUSES:
        return {
            "auto_grade_status": "graded",
            "answer_correct": True,
            "reason": "refusal_status_ok",
        }

    return {
        "auto_grade_status": "graded",
        "answer_correct": True,
        "reason": "refusal_status_ok",
    }


def _empty_component_fields() -> Dict[str, Any]:
    return {
        "evidence_correct": None,
        "tool_selection_correct": None,
        "overall_task_success": None,
    }


def grade_task(item: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic grading record for one task."""
    gold_values = item.get("gold_values")
    if gold_values is None:
        gold_values = row.get("gold_values")
    answerability = item.get("answerability_class") or row.get("answerability_class")
    gold_status = item.get("gold_status") or row.get("gold_status")
    task_id = row.get("task_id") or item.get("task_id") or row.get("id")

    base = {
        "task_id": task_id,
        "grader": GRADER_VERSION,
        **_empty_component_fields(),
    }

    # Hard fail: incomplete agent trajectories must not auto-grade correct.
    if str(row.get("status") or "").lower() == "failed":
        if "exact_tool_set" in row:
            base["tool_selection_correct"] = bool(row.get("exact_tool_set"))
        return {
            **base,
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": "status_failed",
            "overall_task_success": False,
        }

    # Optional tool-selection signal (separate from answer_correct).
    if "exact_tool_set" in row:
        base["tool_selection_correct"] = bool(row.get("exact_tool_set"))
    elif row.get("predicted_operations") is not None and row.get("gold_operations") is not None:
        base["tool_selection_correct"] = set(map(str, row.get("predicted_operations") or [])) == set(
            map(str, row.get("gold_operations") or [])
        )

    # Refusal / gap classes without reliable structured gold → status-based grading.
    if (answerability in REFUSAL_CLASSES) or (
        str(gold_status or "").lower()
        in {"unavailable", "unanswerable", "clarification_needed", "refuse"}
    ):
        if isinstance(gold_values, list) and gold_values and all(
            isinstance(g, dict) and _gold_item_kind(g) == "refusal_spec"
            for g in gold_values
        ):
            return {**base, **grade_refusal(row, gold_values[0])}
        if gold_values in (None, []) or classify_gold_values(gold_values) in {
            "empty",
            "missing",
            "refusal_spec",
        }:
            return {**base, **grade_refusal(row, {})}

    kind = classify_gold_values(gold_values)
    if kind in {"opaque", "missing", "empty"}:
        return {
            **base,
            "auto_grade_status": "manual_required",
            "answer_correct": None,
            "reason": f"gold_values_{kind}",
        }

    text = _answer_text(row.get("final_answer"))
    status = _predicted_status(row)
    if status in REFUSAL_STATUSES and kind in {
        "numeric",
        "string",
        "mixed_auto",
        "labeled_multi",
        "identity",
    }:
        return {
            **base,
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": "refused_answerable_item",
        }

    results: List[bool] = []
    reasons: List[str] = []
    for g in gold_values:
        if not isinstance(g, dict):
            return {
                **base,
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "reason": "non_dict_gold_item",
            }
        gkind = _gold_item_kind(g)
        if gkind == "numeric":
            ok, reason = _grade_numeric_item(text, g)
        elif gkind == "string":
            ok, reason = _grade_string_item(text, g)
        elif gkind == "labeled_multi":
            ok, reason = _grade_labeled_multi(text, g)
        elif gkind == "identity":
            ok, reason = _grade_identity(text, g)
        elif gkind == "structured_part":
            ok, reason = _grade_structured_part(text, g)
        else:
            return {
                **base,
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "reason": f"opaque_gold_item:{gkind}",
            }
        if ok is None:
            return {
                **base,
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "reason": reason,
            }
        results.append(bool(ok))
        reasons.append(reason)

    return {
        **base,
        "auto_grade_status": "graded",
        "answer_correct": all(results) if results else False,
        "reason": ";".join(reasons),
        "parts_correct": results,
        # overall_task_success intentionally left None — requires human / full rubric.
        "overall_task_success": None,
        "evidence_correct": None,
    }


def summarize_grading(grades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    auto = [g for g in grades if g.get("auto_grade_status") == "graded"]
    manual = [g for g in grades if g.get("auto_grade_status") == "manual_required"]
    correct = [g for g in auto if g.get("answer_correct") is True]
    return {
        "n": len(grades),
        "auto_gradable_count": len(auto),
        "auto_graded_correct": len(correct),
        "auto_graded_accuracy": (len(correct) / len(auto)) if auto else None,
        "manual_required_count": len(manual),
        "overall_accuracy": None,
        "overall_accuracy_note": (
            "overall accuracy only after manual grading fields are completed"
        ),
        "grader_version": GRADER_VERSION,
        "grading_version": GRADING_VERSION,
    }
