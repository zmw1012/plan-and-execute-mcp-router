"""Temporal latest/current refusal policy (company-generic).

Benchmark cutoff: 2026-08-12 Asia/Hong_Kong (frozen market/news catalogs).
Questions asking for latest/current/today without an explicit target date must
not be answered with an earlier historical quote framed as current.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

DEFAULT_DATASET_CUTOFF = "2026-08-12"

_LATEST_RX = re.compile(
    r"\b(latest|current|today|now|most\s+recent|最新|今天|当前|現在|今日)\b",
    re.IGNORECASE,
)
_EXPLICIT_ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_EXPLICIT_FY = re.compile(r"\bFY\s*20\d{2}\b", re.IGNORECASE)


def question_requests_latest_current(question: str) -> bool:
    text = str(question or "")
    if not _LATEST_RX.search(text):
        return False
    # Explicit historical date / FY in the question → not a pure latest request.
    if _EXPLICIT_ISO.search(text) or _EXPLICIT_FY.search(text):
        return False
    return True


def assess_latest_current_refusal(
    *,
    question: str,
    evidence_store: Optional[Dict[str, Any]] = None,
    dataset_cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> Optional[Dict[str, Any]]:
    """Return refusal payload when latest/current cannot be honestly satisfied.

    Does not invent gold numbers. Fail closed to unavailable/clarification.
    """
    if not question_requests_latest_current(question):
        return None
    store = evidence_store or {}
    trade_dates = []
    closes = []
    for item in store.get("items") or []:
        vals = item.get("values") if isinstance(item, dict) else None
        if not isinstance(vals, dict):
            continue
        if vals.get("trade_date") is not None:
            trade_dates.append(str(vals.get("trade_date")))
        if vals.get("close") is not None:
            closes.append(vals.get("close"))
    # If we only have historical closes/trade_dates, refusing is required for
    # pure latest/current questions (cutoff policy: do not silently substitute).
    if closes or trade_dates:
        return {
            "should_refuse": True,
            "reason": "temporal_latest_current_unsatisfied_under_cutoff",
            "dataset_cutoff": dataset_cutoff,
            "observed_trade_dates": trade_dates[:8],
            "note": (
                "Question requests latest/current/today without an explicit "
                "historical target date; historical quotes must not silently "
                "substitute for current."
            ),
        }
    # No market quote evidence either — still refuse latest/current under cutoff.
    return {
        "should_refuse": True,
        "reason": "temporal_latest_current_unsatisfied_under_cutoff",
        "dataset_cutoff": dataset_cutoff,
        "observed_trade_dates": [],
        "note": "No verified current-session quote under dataset cutoff.",
    }
