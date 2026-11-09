"""Deterministic pagination for truncated Gateway table payloads (APV-Fin v1.4)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Set, Tuple

PAGINATION_VERSION = "apv_fin_deterministic_pagination_v1"

# Safety caps — do not raise MCP budget; only constrain pagination loops.
MAX_PAGINATION_PAGES_PER_STEP = 5
MAX_ROWS_SCANNED_PER_STEP = 500
DEFAULT_PAGE_MAX_ROWS = 50
FINANCIAL_BUNDLE_PAGE_MAX_ROWS = 100


def pagination_policy_hash() -> str:
    blob = json.dumps(
        {
            "version": PAGINATION_VERSION,
            "max_pages": MAX_PAGINATION_PAGES_PER_STEP,
            "max_rows_scanned": MAX_ROWS_SCANNED_PER_STEP,
            "default_page_max_rows": DEFAULT_PAGE_MAX_ROWS,
            "financial_bundle_page_max_rows": FINANCIAL_BUNDLE_PAGE_MAX_ROWS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def pagination_signature(
    *,
    operation_id: str,
    base_arguments: Dict[str, Any],
    offset: int,
    max_rows: int,
) -> str:
    # Exclude ephemeral keys; keep semantic args stable.
    args = {
        k: v
        for k, v in dict(base_arguments or {}).items()
        if k not in {"offset", "max_rows"}
    }
    blob = json.dumps(
        {"operation_id": operation_id, "arguments": args, "offset": offset, "max_rows": max_rows},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def next_page_arguments(
    *,
    current_arguments: Dict[str, Any],
    envelope: Dict[str, Any],
    capability: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build next-page arguments or None when pagination is exhausted/unsupported."""
    if not envelope.get("truncated"):
        return None
    returned = int(envelope.get("returned_rows") or len(envelope.get("rows") or []) or 0)
    offset = int(envelope.get("offset") or current_arguments.get("offset") or 0)
    next_offset = offset + max(returned, 0)
    total = envelope.get("total_rows")
    if total is not None:
        try:
            if next_offset >= int(total):
                return None
        except (TypeError, ValueError):
            pass
    if returned <= 0:
        return None
    max_rows = int(
        current_arguments.get("max_rows")
        or (
            FINANCIAL_BUNDLE_PAGE_MAX_ROWS
            if str(capability or "").startswith("financial.")
            else DEFAULT_PAGE_MAX_ROWS
        )
    )
    out = dict(current_arguments or {})
    out["offset"] = next_offset
    out["max_rows"] = max_rows
    return out


def should_paginate(
    *,
    mcp_success: bool,
    envelope: Optional[Dict[str, Any]],
    required_outputs_ok: bool,
    supports_offset: bool,
    pages_used: int,
    rows_scanned: int,
    mcp_budget_remaining: int,
) -> Tuple[bool, str]:
    if not mcp_success:
        return False, "mcp_not_successful"
    if required_outputs_ok:
        return False, "outputs_satisfied"
    if not supports_offset:
        return False, "operation_lacks_offset"
    if mcp_budget_remaining <= 0:
        return False, "mcp_budget_exhausted"
    if pages_used >= MAX_PAGINATION_PAGES_PER_STEP:
        return False, "page_cap_reached"
    if rows_scanned >= MAX_ROWS_SCANNED_PER_STEP:
        return False, "row_cap_reached"
    env = envelope or {}
    if not env.get("truncated"):
        return False, "not_truncated"
    return True, "paginate"


def operation_supports_offset(json_schema: Optional[Dict[str, Any]]) -> bool:
    props = (json_schema or {}).get("properties") or {}
    return "offset" in props or "max_rows" in props
