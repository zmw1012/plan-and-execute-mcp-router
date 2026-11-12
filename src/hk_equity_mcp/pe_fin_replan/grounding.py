"""Final-response grounding validator — pe_fin_replan_grounding_v2.

Tokenize before numeric checks. Use Evidence Store provenance. Allow
deterministic unit scaling / display rounding only when units are explicit
or provenance records a reliable unit. Fail closed for question-only targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import forbid_question_target_as_observed
from hk_equity_mcp.pe_fin_replan.plan_ops import select_one_ready_step

GROUNDING_CONTRACT_VERSION = "pe_fin_replan_grounding_v2"

# Half-up display rounding: relative tolerance for scaled comparisons.
_REL_TOL = 5e-4
_ABS_TOL_BASE = 0.51  # allow ±0.5 at display scale for half-up 0dp / 2dp

_ISO_DATE = re.compile(
    r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[Tt ][\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?(?!\d)"
)
_YEAR = re.compile(r"(?<![\d.])((?:19|20)\d{2})(?![\d.-])")
_FY = re.compile(r"(?i)\bFY\s*((?:19|20)\d{2})\b")
_HK_CODE = re.compile(r"(?<!\d)(\d{4,5})(?:\.HK)?(?!\d)")
_PERCENT = re.compile(
    r"(?<![\d.])([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*%"
)
# Monetary/quantity with explicit English or Chinese magnitude unit.
_UNIT_VALUE = re.compile(
    r"(?<![\d.])([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*"
    r"(thousand|million|billion|trillion|万亿|亿|万|千)s?\b",
    re.IGNORECASE,
)
# Comma-grouped or plain numbers (after dates/units stripped from working text).
# Trailing sentence punctuation must not block the match (use (?!\d), not (?![\d.])).
_PLAIN_NUM = re.compile(
    r"(?<![\d.])([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)(?!\d)"
)

_MAG_EN = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}
_MAG_ZH = {
    "千": 1e3,
    "万": 1e4,
    "亿": 1e8,
    "万亿": 1e12,
}

_EARLY_INSUFFICIENT_OK_MARKERS = (
    "cutoff",
    "capability unavailable",
    "capability",
    "exhausted",
    "no matching",
    "verified no matching",
    "primary/fallback",
    "primary and fallback",
)

_CONTEXT_FIELD_HINTS = (
    "trade_date",
    "report_date",
    "release_date",
    "fiscal_period",
    "issuer_code",
    "stock_code",
    "symbol",
)


def grounding_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": GROUNDING_CONTRACT_VERSION,
            "rel_tol": _REL_TOL,
            "abs_tol_base": _ABS_TOL_BASE,
            "classifications": [
                "exact_evidence_numeric",
                "derived_unit_conversion",
                "deterministic_rounding",
                "context_date_ok",
                "context_ticker_ok",
                "context_year_ok",
                "context_fiscal_ok",
                "ungrounded_answer_value",
                "unit_mismatch",
            ],
            "rules": [
                "tokenize_before_numeric",
                "iso_date_not_split",
                "context_date_year_ticker_from_scope_or_evidence",
                "compare_canonical_numeric",
                "unit_scale_only_if_explicit_or_provenance_unit",
                "half_up_display_rounding",
                "provenance_observed_derived_grounded_input",
                "reject_question_only_answer_targets",
                "no_arbitrary_approx_collision",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _parse_number_token(raw: str) -> Optional[float]:
    s = str(raw).strip().replace(",", "")
    if not s or s in {".", "+", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _half_up(value: float, places: int) -> float:
    q = Decimal("1").scaleb(-places) if places else Decimal("1")
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def _unit_scale(unit: str) -> Optional[float]:
    u = unit.strip().lower()
    if u in _MAG_EN:
        return float(_MAG_EN[u])
    # Chinese units are case-sensitive as written
    raw = unit.strip()
    if raw in _MAG_ZH:
        return float(_MAG_ZH[raw])
    # tolerate mixed case for Chinese-only tokens
    for k, v in _MAG_ZH.items():
        if k in raw:
            return float(v)
    return None


def _tokenize_answer(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Return (tokens, residual_text_with_matched_spans_blanked)."""
    tokens: List[Dict[str, Any]] = []
    work = str(text or "")

    def _blank(start: int, end: int) -> None:
        nonlocal work
        work = work[:start] + (" " * (end - start)) + work[end:]

    # ISO dates first — never split into year/month/day fragments.
    for m in list(_ISO_DATE.finditer(work)):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        tokens.append(
            {
                "kind": "iso_date",
                "raw": m.group(0),
                "value": f"{y}-{mo}-{d}",
                "year": int(y),
            }
        )
        _blank(m.start(), m.end())

    for m in list(_FY.finditer(work)):
        tokens.append(
            {
                "kind": "fiscal_period",
                "raw": m.group(0),
                "value": int(m.group(1)),
                "year": int(m.group(1)),
            }
        )
        _blank(m.start(), m.end())

    for m in list(_PERCENT.finditer(work)):
        num = _parse_number_token(m.group(1))
        if num is None:
            continue
        tokens.append(
            {
                "kind": "percentage",
                "raw": m.group(0),
                "value": num,
                "canonical": num / 100.0,
            }
        )
        _blank(m.start(), m.end())

    for m in list(_UNIT_VALUE.finditer(work)):
        num = _parse_number_token(m.group(1))
        scale = _unit_scale(m.group(2))
        if num is None or scale is None:
            continue
        tokens.append(
            {
                "kind": "monetary_quantity",
                "raw": m.group(0),
                "display": num,
                "unit": m.group(2),
                "scale": scale,
                "canonical": float(num) * float(scale),
                "unit_explicit": True,
            }
        )
        _blank(m.start(), m.end())

    # Tickers / HK codes before plain years to avoid double-counting 00700 as year.
    for m in list(_HK_CODE.finditer(work)):
        code = m.group(1)
        # Skip bare 4-digit years already handled; 19xx/20xx without .HK may be year.
        if re.fullmatch(r"(?:19|20)\d{2}", code) and ".HK" not in m.group(0).upper():
            continue
        tokens.append({"kind": "hk_stock_code", "raw": m.group(0), "value": code.zfill(5)})
        _blank(m.start(), m.end())

    for m in list(_YEAR.finditer(work)):
        tokens.append(
            {
                "kind": "year",
                "raw": m.group(0),
                "value": int(m.group(1)),
                "year": int(m.group(1)),
            }
        )
        _blank(m.start(), m.end())

    for m in list(_PLAIN_NUM.finditer(work)):
        num = _parse_number_token(m.group(1))
        if num is None:
            continue
        tokens.append(
            {
                "kind": "plain_number",
                "raw": m.group(0),
                "value": num,
                "canonical": num,
                "unit_explicit": False,
            }
        )
        _blank(m.start(), m.end())

    return tokens, work


def _field_provenance(
    provenance_by_key: Optional[Dict[str, Any]], key: str
) -> Dict[str, Any]:
    if not provenance_by_key:
        return {}
    prov = provenance_by_key.get(key) or {}
    if not isinstance(prov, dict):
        return {}
    fmap = prov.get("field_provenance")
    if isinstance(fmap, dict):
        # Prefer bare field name inside map
        bare = key.split(".", 1)[-1]
        fp = fmap.get(bare) or fmap.get(key)
        if isinstance(fp, dict):
            return fp
    return prov


def _provenance_class(fp: Dict[str, Any]) -> str:
    return str(fp.get("provenance_class") or fp.get("source") or "").strip().lower()


def _evidence_magnitude_unit(
    fp: Dict[str, Any], values: Dict[str, Any], key: str
) -> Optional[str]:
    """Return a magnitude unit (thousand/million/billion/万/亿), not currency alone."""
    del values, key  # currency is not a magnitude scale cue
    for cand in (
        fp.get("unit"),
        fp.get("output_unit"),
        (fp.get("derivation") or {}).get("unit")
        if isinstance(fp.get("derivation"), dict)
        else None,
        (fp.get("derivation") or {}).get("magnitude_unit")
        if isinstance(fp.get("derivation"), dict)
        else None,
    ):
        if cand is None or not str(cand).strip():
            continue
        raw = str(cand).strip()
        if _unit_scale(raw) is not None:
            return raw
        # phrases like "million yuan"
        for tok in re.split(r"[\s_/]+", raw.lower()):
            if _unit_scale(tok) is not None:
                return tok
    return None


_CONTEXT_ONLY_FIELDS = frozenset(
    {
        "issuer_code",
        "stock_code",
        "symbol",
        "company_name",
        "currency",
        "adjustment",
        "title",
        "news_id",
        "local_key",
        "file_link",
        "source_url",
        "snippet",
        "page_number",
    }
)


def _bare_field(key: str) -> str:
    return str(key).split(".", 1)[-1]


def _collect_evidence_numerics(
    evidence_values: Dict[str, Any],
    provenance_by_key: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, raw in (evidence_values or {}).items():
        bare = _bare_field(key)
        if bare in _CONTEXT_ONLY_FIELDS or bare.endswith("_url"):
            # Still record dates/fiscals/tickers via dedicated branches below when string.
            if bare in {"issuer_code", "stock_code", "symbol"} and raw is not None:
                digits = re.sub(r"\D", "", str(raw))
                if digits:
                    out.append(
                        {
                            "key": key,
                            "canonical": digits.zfill(5),
                            "raw": raw,
                            "provenance_class": "grounded_input",
                            "kind": "hk_stock_code",
                        }
                    )
                continue
            if bare not in {"report_date", "trade_date", "release_date", "fiscal_period"}:
                # currency/title/etc.: skip as answer-bearing numeric
                if bare != "fiscal_period" and not bare.endswith("_date"):
                    continue
        fp = _field_provenance(provenance_by_key, key)
        pclass = _provenance_class(fp)
        # Prefer field-level class when present
        if isinstance(fp.get("provenance_class"), str):
            pclass = str(fp["provenance_class"]).lower()

        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            if bare in _CONTEXT_ONLY_FIELDS:
                continue
            out.append(
                {
                    "key": key,
                    "canonical": float(raw),
                    "raw": raw,
                    "provenance_class": pclass or "observed",
                    "unit": _evidence_magnitude_unit(
                        fp, evidence_values, key
                    ),
                    "kind": "numeric",
                }
            )
            continue
        if isinstance(raw, str):
            s = raw.strip()
            # ISO / datetime strings as context
            dm = _ISO_DATE.search(s)
            if dm:
                out.append(
                    {
                        "key": key,
                        "canonical": f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}",
                        "raw": raw,
                        "provenance_class": pclass or "observed",
                        "kind": "iso_date",
                        "year": int(dm.group(1)),
                    }
                )
                continue
            fy = _FY.search(s)
            if fy:
                out.append(
                    {
                        "key": key,
                        "canonical": int(fy.group(1)),
                        "raw": raw,
                        "provenance_class": pclass or "derived",
                        "kind": "fiscal_period",
                        "year": int(fy.group(1)),
                    }
                )
                continue
            if bare in _CONTEXT_ONLY_FIELDS:
                continue
            compact = s.replace(",", "")
            num = _parse_number_token(compact)
            if num is not None and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
                out.append(
                    {
                        "key": key,
                        "canonical": float(num),
                        "raw": raw,
                        "provenance_class": pclass or "observed",
                        "unit": _evidence_magnitude_unit(
                            fp, evidence_values, key
                        ),
                        "kind": "numeric",
                    }
                )
    return out


def _context_pool(
    question: str,
    evidence_rows: List[Dict[str, Any]],
    evidence_values: Dict[str, Any],
) -> Dict[str, Set[Any]]:
    dates: Set[str] = set()
    years: Set[int] = set()
    tickers: Set[str] = set()
    fiscals: Set[int] = set()

    for m in _ISO_DATE.finditer(question or ""):
        dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        years.add(int(m.group(1)))
    for m in _FY.finditer(question or ""):
        fiscals.add(int(m.group(1)))
        years.add(int(m.group(1)))
    for m in _YEAR.finditer(question or ""):
        years.add(int(m.group(1)))
    for m in _HK_CODE.finditer(question or ""):
        code = m.group(1)
        if not re.fullmatch(r"(?:19|20)\d{2}", code):
            tickers.add(code.zfill(5))
            tickers.add(code)

    for row in evidence_rows:
        if row.get("kind") == "iso_date":
            dates.add(str(row["canonical"]))
            if row.get("year") is not None:
                years.add(int(row["year"]))
        elif row.get("kind") == "fiscal_period":
            fiscals.add(int(row["canonical"]))
            years.add(int(row["canonical"]))
        key = str(row.get("key") or "")
        bare = key.split(".", 1)[-1]
        if bare in {"issuer_code", "stock_code"} or "issuer" in bare:
            val = evidence_values.get(key)
            if val is not None:
                digits = re.sub(r"\D", "", str(val))
                if digits:
                    tickers.add(digits.zfill(5))
                    tickers.add(digits)

    # Explicit evidence string fields
    for k, v in (evidence_values or {}).items():
        bare = k.split(".", 1)[-1]
        if bare in _CONTEXT_FIELD_HINTS or bare.endswith("_date"):
            if isinstance(v, str):
                dm = _ISO_DATE.search(v)
                if dm:
                    dates.add(f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}")
                    years.add(int(dm.group(1)))
                fy = _FY.search(v)
                if fy:
                    fiscals.add(int(fy.group(1)))
                    years.add(int(fy.group(1)))
        if bare in {"issuer_code", "stock_code"}:
            digits = re.sub(r"\D", "", str(v))
            if digits:
                tickers.add(digits.zfill(5))
                tickers.add(digits)

    return {
        "dates": dates,
        "years": years,
        "tickers": tickers,
        "fiscals": fiscals,
    }


def _is_answer_bearing_provenance(pclass: str) -> bool:
    pc = (pclass or "").lower()
    if pc == "grounded_input":
        return False
    # observed / derived / empty / compute-like
    return True


def _close(a: float, b: float, *, abs_tol: float | None = None) -> bool:
    return math.isclose(
        float(a),
        float(b),
        rel_tol=_REL_TOL,
        abs_tol=_ABS_TOL_BASE if abs_tol is None else abs_tol,
    )


def _numeric_match(
    answer_canonical: float,
    *,
    evidence_rows: List[Dict[str, Any]],
    unit_explicit: bool,
    answer_scale: Optional[float],
    answer_display: Optional[float] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return (ok, classification, evidence_key).

    Compares canonical numeric values. Unit scaling is allowed only when the
    answer states an explicit magnitude unit, or evidence provenance carries a
    reliable unit cue that justifies the scale family.
    """
    candidates = [
        r
        for r in evidence_rows
        if r.get("kind") == "numeric" and _is_answer_bearing_provenance(
            str(r.get("provenance_class") or "")
        )
    ]

    for row in candidates:
        ev = float(row["canonical"])

        # Tight exact match on base units (display-rounding is not "exact").
        if math.isclose(answer_canonical, ev, rel_tol=0.0, abs_tol=_ABS_TOL_BASE):
            return True, "exact_evidence_numeric", row.get("key")

        for places in (0, 1, 2, 3):
            if math.isclose(
                answer_canonical,
                _half_up(ev, places),
                rel_tol=0.0,
                abs_tol=max(_ABS_TOL_BASE, 10 ** (-places) / 2 + 1e-9),
            ):
                return True, "deterministic_rounding", row.get("key")

        allow_scale = bool(unit_explicit) or bool(row.get("unit"))
        if not allow_scale:
            continue

        scales: List[float] = []
        if answer_scale:
            scales.append(float(answer_scale))
        if unit_explicit:
            for sc in (1e3, 1e4, 1e6, 1e8, 1e9, 1e12):
                if sc not in scales:
                    scales.append(float(sc))

        for sc in scales:
            disp = (
                float(answer_display)
                if answer_display is not None
                else (answer_canonical / sc if sc else answer_canonical)
            )
            if unit_explicit:
                for places in (0, 1, 2, 3, 4):
                    rounded = _half_up(ev / sc, places)
                    if math.isclose(
                        disp,
                        rounded,
                        rel_tol=_REL_TOL,
                        abs_tol=max(0.05, 10 ** (-places)),
                    ):
                        rebuilt = float(disp) * float(sc)
                        if math.isclose(
                            rebuilt,
                            ev,
                            rel_tol=_REL_TOL,
                            abs_tol=max(1.0, abs(ev) * _REL_TOL),
                        ):
                            if math.isclose(
                                disp, ev / sc, rel_tol=0.0, abs_tol=1e-12
                            ):
                                return (
                                    True,
                                    "derived_unit_conversion",
                                    row.get("key"),
                                )
                            return True, "deterministic_rounding", row.get("key")

            if math.isclose(
                answer_canonical,
                ev,
                rel_tol=_REL_TOL,
                abs_tol=max(1.0, abs(ev) * _REL_TOL),
            ):
                return True, "derived_unit_conversion", row.get("key")

    return False, None, None


def validate_replanner_response(
    *,
    response: Dict[str, Any],
    question: str,
    evidence_values: Dict[str, Any],
    answerability: Optional[Dict[str, Any]] = None,
    provenance_by_key: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return {ok, reasons, classifications} — fail closed for ungrounded answered."""
    del answerability  # reserved for future answerability coupling
    completion = str(response.get("completion") or "")
    answer = str(response.get("final_answer") or "")
    reasons: List[str] = []
    classifications: List[str] = []

    if completion == "insufficient_evidence":
        return {"ok": True, "reasons": [], "classifications": []}

    if completion != "answered":
        return {
            "ok": False,
            "reasons": [f"unknown_completion:{completion}"],
            "classifications": ["ungrounded_answer_value"],
        }

    tokens, _ = _tokenize_answer(answer)
    evidence_rows = _collect_evidence_numerics(evidence_values or {}, provenance_by_key)
    ctx = _context_pool(question or "", evidence_rows, evidence_values or {})

    answer_bearing_tokens = [
        t
        for t in tokens
        if t["kind"] in {"plain_number", "monetary_quantity", "percentage"}
    ]

    if not evidence_values and answer_bearing_tokens:
        return {
            "ok": False,
            "reasons": ["answered_without_verified_evidence"],
            "classifications": ["ungrounded_answer_value"],
        }

    for tok in tokens:
        kind = tok["kind"]
        if kind == "iso_date":
            if tok["value"] in ctx["dates"] or tok.get("year") in ctx["years"]:
                classifications.append("context_date_ok")
                continue
            reasons.append(f"ungrounded_context_date:{tok['raw']}")
            classifications.append("ungrounded_answer_value")
            continue
        if kind == "year":
            if tok["value"] in ctx["years"] or tok["value"] in ctx["fiscals"]:
                classifications.append("context_year_ok")
                continue
            # Year only in question and not in evidence → still context if question scope
            if str(tok["value"]) in (question or "") and tok["value"] in ctx["years"]:
                classifications.append("context_year_ok")
                continue
            reasons.append(f"ungrounded_context_year:{tok['raw']}")
            classifications.append("ungrounded_answer_value")
            continue
        if kind == "fiscal_period":
            if tok["value"] in ctx["fiscals"] or tok["value"] in ctx["years"]:
                classifications.append("context_fiscal_ok")
                continue
            reasons.append(f"ungrounded_context_fiscal:{tok['raw']}")
            classifications.append("ungrounded_answer_value")
            continue
        if kind == "hk_stock_code":
            if tok["value"] in ctx["tickers"] or tok["value"].lstrip("0") in {
                t.lstrip("0") for t in ctx["tickers"]
            }:
                classifications.append("context_ticker_ok")
                continue
            reasons.append(f"ungrounded_context_ticker:{tok['raw']}")
            classifications.append("ungrounded_answer_value")
            continue
        if kind == "percentage":
            ok, cls, _key = _numeric_match(
                float(tok["canonical"]),
                evidence_rows=evidence_rows,
                unit_explicit=True,
                answer_scale=None,
            )
            # Also try matching the display percent figure against evidence
            if not ok:
                ok, cls, _key = _numeric_match(
                    float(tok["value"]),
                    evidence_rows=evidence_rows,
                    unit_explicit=True,
                    answer_scale=None,
                )
            if ok and cls:
                classifications.append(cls)
            else:
                reasons.append(f"ungrounded_answer_value:{tok['raw']}")
                classifications.append("ungrounded_answer_value")
            continue
        if kind == "monetary_quantity":
            ok, cls, _key = _numeric_match(
                float(tok["canonical"]),
                evidence_rows=evidence_rows,
                unit_explicit=True,
                answer_scale=float(tok["scale"]),
                answer_display=float(tok["display"]),
            )
            if ok and cls:
                classifications.append(cls)
            else:
                # Unit present but no evidence match
                if evidence_rows:
                    reasons.append(f"unit_mismatch_or_ungrounded:{tok['raw']}")
                    classifications.append("unit_mismatch")
                else:
                    reasons.append(f"ungrounded_answer_value:{tok['raw']}")
                    classifications.append("ungrounded_answer_value")
            continue
        if kind == "plain_number":
            val = float(tok["canonical"])
            # Reject question-only target answers lacking evidence
            q_hit = str(tok["raw"]).replace(",", "") in (question or "").replace(",", "")
            if q_hit and forbid_question_target_as_observed(
                field="close",
                value=val if "." in str(tok["raw"]) else int(val) if val == int(val) else val,
                question=question,
            ):
                ok, cls, _key = _numeric_match(
                    val,
                    evidence_rows=evidence_rows,
                    unit_explicit=False,
                    answer_scale=None,
                    answer_display=val,
                )
                if not ok:
                    reasons.append(f"ungrounded_question_target_number:{tok['raw']}")
                    classifications.append("ungrounded_answer_value")
                    continue
                if cls:
                    classifications.append(cls)
                continue

            ok, cls, _key = _numeric_match(
                val,
                evidence_rows=evidence_rows,
                unit_explicit=False,
                answer_scale=None,
                answer_display=val,
            )
            if ok and cls:
                classifications.append(cls)
                continue
            # Without explicit unit, do not accept arbitrary scale collisions
            if evidence_rows:
                # Check if a scale would match but unit missing → unit_mismatch
                scaled_ok = False
                for row in evidence_rows:
                    if row.get("kind") != "numeric":
                        continue
                    if row.get("provenance_class") == "grounded_input":
                        continue
                    ev = float(row["canonical"])
                    for sc in (1e3, 1e4, 1e6, 1e8, 1e9, 1e12):
                        if math.isclose(
                            val * sc,
                            ev,
                            rel_tol=_REL_TOL,
                            abs_tol=max(1.0, abs(ev) * _REL_TOL),
                        ):
                            scaled_ok = True
                            break
                    if scaled_ok:
                        break
                if scaled_ok:
                    reasons.append(f"unit_mismatch:{tok['raw']}")
                    classifications.append("unit_mismatch")
                else:
                    reasons.append(f"ungrounded_answer_value:{tok['raw']}")
                    classifications.append("ungrounded_answer_value")
            else:
                reasons.append(f"ungrounded_answer_value:{tok['raw']}")
                classifications.append("ungrounded_answer_value")

    if reasons:
        return {"ok": False, "reasons": reasons, "classifications": classifications}
    return {"ok": True, "reasons": [], "classifications": classifications}


def is_premature_insufficient(
    *,
    execution_round: int,
    max_execution_rounds: int,
    remaining_plan: Sequence[Dict[str, Any]],
    completed_ids: Set[str],
    answerability: Optional[Dict[str, Any]],
    decision_reason: str,
    evidence_values: Optional[Dict[str, Any]] = None,
) -> bool:
    """Block early insufficient_evidence when recoverable paths remain."""
    if execution_round >= max_execution_rounds:
        return False
    reason_l = f"{decision_reason or ''}".lower()
    if any(m in reason_l for m in _EARLY_INSUFFICIENT_OK_MARKERS):
        return False

    if select_one_ready_step(remaining_plan, completed_ids) is not None:
        return True
    if list(remaining_plan):
        return True

    ans_state = str((answerability or {}).get("answerability_state") or "")
    if ans_state == "infeasible":
        return False
    if ans_state in {"sufficient"} and evidence_values:
        return False
    if ans_state in {
        "partial_but_answerable",
        "insufficient",
        "missing_core",
        "uncertain",
        "",
    }:
        return True
    if not evidence_values:
        return True
    return False
