"""Conservative deterministic answer grading for unified HK runs.

Never uses LLM-as-judge. Uncertain / evidence-only items → manual_required.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.unified_evaluator import REFUSAL_CLASSES, REFUSAL_STATUSES

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


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


def _predicted_status(row: Dict[str, Any]) -> str:
    ps = row.get("predicted_status")
    if ps:
        return str(ps).strip().lower()
    fa = row.get("final_answer")
    if isinstance(fa, dict) and fa.get("status"):
        return str(fa.get("status")).strip().lower()
    return ""


def _parse_number(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", ""))
    except Exception:
        return None


def extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in NUMBER_RE.finditer(text or ""):
        val = _parse_number(m.group(0))
        if val is not None:
            out.append(val)
    return out


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
        "hkd": "hkd",
        "usd": "usd",
        "shares": "shares",
    }
    return aliases.get(u, u)


def _unit_present(text: str, unit: str) -> bool:
    norm = _normalize_unit(unit)
    if not norm:
        return True
    hay = (text or "").lower().replace("_", " ")
    # Require distinctive tokens (e.g. cny+million, hkd, usd, shares).
    tokens = [t for t in norm.split() if t]
    if not tokens:
        return True
    return all(t in hay for t in tokens)


def _currency_present(text: str, currency: Optional[str]) -> bool:
    if not currency:
        return True
    return _normalize_unit(currency) in (text or "").lower().replace("_", " ")


def _numeric_match(
    numbers: Sequence[float],
    gold_value: float,
    *,
    tolerance_abs: Optional[float] = None,
    tolerance_rel: Optional[float] = None,
) -> bool:
    for n in numbers:
        if tolerance_abs is not None and abs(n - gold_value) <= float(tolerance_abs):
            return True
        if tolerance_rel is not None and gold_value != 0:
            if abs(n - gold_value) / abs(gold_value) <= float(tolerance_rel):
                return True
        if tolerance_abs is None and tolerance_rel is None:
            if math.isclose(n, gold_value, rel_tol=0.0, abs_tol=1e-9):
                return True
    return False


def _gold_item_kind(item: Dict[str, Any]) -> str:
    if item.get("expected_behavior") or item.get("answerability"):
        return "refusal_spec"
    if "value" in item and isinstance(item.get("value"), (int, float)):
        return "numeric"
    if "value" in item and isinstance(item.get("value"), str):
        return "string"
    if item.get("field") and "value" in item:
        return "string"
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
    if kinds <= {"numeric", "string"}:
        return "mixed_auto"
    return "opaque"


def _grade_numeric_item(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    value = item.get("value")
    if not isinstance(value, (int, float)):
        return None, "non_numeric_value"
    numbers = extract_numbers(text)
    if not numbers:
        return False, "no_numeric_in_answer"
    unit = item.get("unit")
    currency = item.get("currency")
    if unit and not _unit_present(text, str(unit)):
        return False, "unit_mismatch_or_missing"
    if currency and not _currency_present(text, str(currency)):
        return False, "currency_mismatch_or_missing"
    ok = _numeric_match(
        numbers,
        float(value),
        tolerance_abs=item.get("tolerance_abs"),
        tolerance_rel=item.get("tolerance_rel"),
    )
    return (True, "numeric_match") if ok else (False, "numeric_mismatch")


def _grade_string_item(text: str, item: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    value = item.get("value")
    if value is None:
        return None, "missing_string_value"
    needle = str(value).strip()
    if not needle:
        return None, "empty_string_value"
    if needle.lower() in (text or "").lower():
        return True, "string_match"
    return False, "string_mismatch"


def _looks_like_fabricated_numeric(text: str) -> bool:
    """Heuristic: concrete money/share figures in a refusal answer."""
    if not extract_numbers(text):
        return False
    hay = (text or "").lower()
    cues = ("cny", "hkd", "usd", "million", "revenue", "eps", "close", "shares")
    return any(c in hay for c in cues)


def grade_refusal(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    status = _predicted_status(row)
    text = _answer_text(row.get("final_answer"))
    if status not in REFUSAL_STATUSES:
        return {
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": f"expected_refusal_got_status={status or 'missing'}",
        }
    if _looks_like_fabricated_numeric(text):
        return {
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": "refusal_with_fabricated_numeric",
        }
    return {
        "auto_grade_status": "graded",
        "answer_correct": True,
        "reason": "refusal_status_ok",
    }


def grade_task(item: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic grading record for one task."""
    gold_values = item.get("gold_values")
    if gold_values is None:
        gold_values = row.get("gold_values")
    answerability = item.get("answerability_class") or row.get("answerability_class")
    gold_status = item.get("gold_status") or row.get("gold_status")

    # Refusal / gap classes without reliable structured gold → status-based grading.
    if (answerability in REFUSAL_CLASSES) or (
        str(gold_status or "").lower()
        in {"unavailable", "unanswerable", "clarification_needed", "refuse"}
    ):
        # Prefer explicit refusal_spec gold when present.
        if isinstance(gold_values, list) and gold_values and all(
            isinstance(g, dict) and _gold_item_kind(g) == "refusal_spec"
            for g in gold_values
        ):
            return {
                "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
                **grade_refusal(row, gold_values[0]),
                "grader": "deterministic_v1",
            }
        if gold_values in (None, []) or classify_gold_values(gold_values) in {
            "empty",
            "missing",
            "refusal_spec",
        }:
            return {
                "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
                **grade_refusal(row, {}),
                "grader": "deterministic_v1",
            }

    kind = classify_gold_values(gold_values)
    if kind in {"opaque", "missing", "empty"}:
        return {
            "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
            "auto_grade_status": "manual_required",
            "answer_correct": None,
            "reason": f"gold_values_{kind}",
            "grader": "deterministic_v1",
        }

    text = _answer_text(row.get("final_answer"))
    status = _predicted_status(row)
    if status in REFUSAL_STATUSES and kind in {"numeric", "string", "mixed_auto"}:
        return {
            "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
            "auto_grade_status": "graded",
            "answer_correct": False,
            "reason": "refused_answerable_item",
            "grader": "deterministic_v1",
        }

    results: List[bool] = []
    reasons: List[str] = []
    for g in gold_values:
        if not isinstance(g, dict):
            return {
                "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "reason": "non_dict_gold_item",
                "grader": "deterministic_v1",
            }
        gkind = _gold_item_kind(g)
        if gkind == "numeric":
            ok, reason = _grade_numeric_item(text, g)
        elif gkind == "string":
            ok, reason = _grade_string_item(text, g)
        else:
            return {
                "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "reason": f"opaque_gold_item:{gkind}",
                "grader": "deterministic_v1",
            }
        if ok is None:
            return {
                "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "reason": reason,
                "grader": "deterministic_v1",
            }
        results.append(bool(ok))
        reasons.append(reason)

    return {
        "task_id": row.get("task_id") or item.get("task_id") or row.get("id"),
        "auto_grade_status": "graded",
        "answer_correct": all(results) if results else False,
        "reason": ";".join(reasons),
        "grader": "deterministic_v1",
        "parts_correct": results,
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
    }
