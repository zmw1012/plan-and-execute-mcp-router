"""Benchmark cutoff enforcement for disclosure list/download paths.

HKEX MCP provider defaults missing dates to fromDate=19990401 and
toDate=today. This module ensures PE-Fin always prefers the frozen
benchmark cutoff over system date, LLM args, and provider defaults.

Company-generic. No task_id / gold branching.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF

BENCHMARK_CUTOFF_ENFORCEMENT_VERSION = "pe_fin_lite_benchmark_cutoff_enforcement_v1"

_ISO = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
_COMPACT = re.compile(r"^(20\d{2})(\d{2})(\d{2})$")
_ISO_IN_TEXT = re.compile(r"(20\d{2}-\d{2}-\d{2})")

_LIST_OP_MARKERS = (
    "list_hkexnews_announcements",
    "list_filings",
    "search_filings",
)
_DOWNLOAD_OP_MARKERS = (
    "download_hkexnews_announcement",
    "download_filing",
)

_END_KEYS = ("end_date", "to_date", "toDate")
_START_KEYS = ("start_date", "from_date", "fromDate")

_DISCLOSURE_VALUE_KEYS = (
    "release_date",
    "title",
    "news_id",
    "local_key",
    "file_link",
    "source_url",
    "snippet",
    "page_number",
    "company_name",
)


def benchmark_cutoff_enforcement_hash() -> str:
    blob = json.dumps(
        {
            "version": BENCHMARK_CUTOFF_ENFORCEMENT_VERSION,
            "default_cutoff": DEFAULT_DATASET_CUTOFF,
            "rules": [
                "effective_to_date_min_requested_cutoff",
                "missing_to_date_uses_cutoff_not_today",
                "missing_from_date_not_provider_1999_default",
                "filter_rows_release_date_le_cutoff",
                "block_download_without_dated_target",
                "block_download_post_cutoff_release",
                "reject_post_cutoff_evidence_values",
                "no_task_id_or_gold_branching",
            ],
            "list_op_markers": list(_LIST_OP_MARKERS),
            "download_op_markers": list(_DOWNLOAD_OP_MARKERS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalize_iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    if _ISO.fullmatch(text):
        return text
    if _COMPACT.fullmatch(text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    m = _ISO_IN_TEXT.search(text)
    if m:
        return m.group(1)
    return None


def is_disclosure_list_operation(operation_id: str) -> bool:
    oid = str(operation_id or "").lower()
    return any(m in oid for m in _LIST_OP_MARKERS)


def is_disclosure_download_operation(operation_id: str) -> bool:
    oid = str(operation_id or "").lower()
    return any(m in oid for m in _DOWNLOAD_OP_MARKERS)


def effective_to_date(
    requested_to_date: Any,
    *,
    cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> str:
    """effective_to_date = min(requested_to_date, benchmark_cutoff); missing → cutoff."""
    cut = normalize_iso_date(cutoff) or DEFAULT_DATASET_CUTOFF
    req = normalize_iso_date(requested_to_date)
    if not req:
        return cut
    return req if req <= cut else cut


def clamp_disclosure_date_window(
    arguments: Optional[Dict[str, Any]],
    *,
    cutoff: str = DEFAULT_DATASET_CUTOFF,
    grounding: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Clamp/fill list window dates. Never leaves end open to system today."""
    args = dict(arguments or {})
    notes: List[str] = []
    g = grounding or {}
    cut = normalize_iso_date(cutoff) or DEFAULT_DATASET_CUTOFF

    # Resolve current end candidate.
    end_raw = None
    end_key_used = None
    for key in _END_KEYS:
        if args.get(key) not in (None, ""):
            end_raw = args.get(key)
            end_key_used = key
            break
    if end_raw is None:
        for key in ("to_date", "end_date"):
            if g.get(key) not in (None, ""):
                end_raw = g.get(key)
                end_key_used = "end_date"
                notes.append("filled_end_date_from_grounding")
                break
    new_end = effective_to_date(end_raw, cutoff=cut)
    if end_raw is None:
        notes.append("filled_end_date_from_cutoff")
    elif normalize_iso_date(end_raw) and normalize_iso_date(end_raw) > cut:
        notes.append("clamped_end_date_to_cutoff")
    # Write back onto present keys; always ensure end_date exists for HKEX list.
    if end_key_used:
        args[end_key_used] = new_end
    args["end_date"] = new_end
    if "to_date" in args or g.get("to_date") is not None:
        args["to_date"] = new_end

    # Start / from: never leave null for list (provider would use 19990401).
    start_raw = None
    start_key_used = None
    for key in _START_KEYS:
        if args.get(key) not in (None, ""):
            start_raw = args.get(key)
            start_key_used = key
            break
    if start_raw is None:
        for key in ("from_date", "start_date"):
            if g.get(key) not in (None, ""):
                start_raw = g.get(key)
                start_key_used = "start_date"
                notes.append("filled_start_date_from_grounding")
                break
    start_n = normalize_iso_date(start_raw)
    if start_n is None:
        # Unscoped lower bound (explicit). Not broadening of an existing window.
        # End is still capped at cutoff so provider cannot use system today.
        start_n = "1999-04-01"
        start_key_used = start_key_used or "start_date"
        notes.append("filled_unscoped_start_date_explicit_19990401_end_capped")
    elif start_n > cut:
        start_n = cut
        notes.append("clamped_start_date_to_cutoff")
    if start_key_used:
        args[start_key_used] = start_n
    args["start_date"] = start_n
    # Only mirror into from_date when caller/grounding already used that key.
    if "from_date" in args:
        args["from_date"] = start_n
    elif normalize_iso_date(g.get("from_date")):
        args["from_date"] = start_n

    return args, notes


def _local_key_release_date(local_key: Any) -> Optional[str]:
    text = str(local_key or "")
    m = _ISO_IN_TEXT.search(text)
    return m.group(1) if m else None


def enforce_disclosure_invoke_arguments(
    operation_id: str,
    arguments: Optional[Dict[str, Any]],
    *,
    cutoff: str = DEFAULT_DATASET_CUTOFF,
    grounding: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """Return (args, notes, block_reason).

    block_reason set ⇒ caller must not invoke MCP (avoids provider today/1999 defaults).
    """
    oid = str(operation_id or "")
    args = dict(arguments or {})
    notes: List[str] = []
    cut = normalize_iso_date(cutoff) or DEFAULT_DATASET_CUTOFF
    g = grounding or {}

    if is_disclosure_list_operation(oid):
        args, clamp_notes = clamp_disclosure_date_window(
            args, cutoff=cut, grounding=g
        )
        notes.extend(clamp_notes)
        return args, notes, None

    if is_disclosure_download_operation(oid):
        # Prefer exact keys from a prior in-cutoff list selection.
        if args.get("local_key") not in (None, ""):
            lk_date = _local_key_release_date(args.get("local_key"))
            if lk_date and lk_date > cut:
                return args, notes, "download_local_key_beyond_benchmark_cutoff"
            rd = normalize_iso_date(args.get("release_date"))
            if rd and rd > cut:
                return args, notes, "download_release_date_beyond_benchmark_cutoff"
            return args, notes, None
        if args.get("news_id") not in (None, ""):
            rd = normalize_iso_date(args.get("release_date"))
            if rd and rd > cut:
                return args, notes, "download_release_date_beyond_benchmark_cutoff"
            if not rd:
                # news_id alone can still trigger provider list defaults in find().
                # Require an explicit in-cutoff release_date.
                return (
                    args,
                    notes,
                    "download_news_id_missing_release_date_under_cutoff_policy",
                )
            return args, notes, None

        rd = normalize_iso_date(args.get("release_date"))
        if not rd:
            return (
                args,
                notes,
                "download_missing_release_date_or_key_under_cutoff_policy",
            )
        if rd > cut:
            return args, notes, "download_release_date_beyond_benchmark_cutoff"
        return args, notes, None

    return args, notes, None


def filter_rows_by_benchmark_cutoff(
    rows: Sequence[Any],
    *,
    cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Split rows into (kept, rejected) by release_date <= cutoff."""
    cut = normalize_iso_date(cutoff) or DEFAULT_DATASET_CUTOFF
    kept: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = normalize_iso_date(
            row.get("release_date") or row.get("releaseDate") or row.get("date")
        )
        if rel and rel > cut:
            rejected.append(
                {
                    "title": str(row.get("title") or "")[:160],
                    "release_date": rel,
                    "reason": "beyond_benchmark_cutoff",
                }
            )
            continue
        kept.append(row)
    return kept, rejected


def reject_post_cutoff_evidence_values(
    values: Optional[Dict[str, Any]],
    *,
    cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Drop disclosure values when release_date exceeds cutoff."""
    vals = dict(values or {})
    cut = normalize_iso_date(cutoff) or DEFAULT_DATASET_CUTOFF
    rel = normalize_iso_date(vals.get("release_date"))
    if rel and rel > cut:
        for key in _DISCLOSURE_VALUE_KEYS:
            vals.pop(key, None)
        return vals, "beyond_benchmark_cutoff"
    # Path / local_key date markers
    for key in ("local_key", "file_link", "source_url"):
        marker = _local_key_release_date(vals.get(key))
        if marker and marker > cut:
            for drop in _DISCLOSURE_VALUE_KEYS:
                vals.pop(drop, None)
            return vals, "beyond_benchmark_cutoff_path_or_local_key"
    return vals, None


def blocked_invoke_response(*, operation_id: str, reason: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": f"benchmark_cutoff_blocked:{reason}:{operation_id}",
        "error_type": "benchmark_cutoff_blocked",
        "result": None,
        "_benchmark_cutoff_blocked": True,
        "_blocked_reason": reason,
    }
