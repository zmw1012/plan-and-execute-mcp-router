"""Deterministic MCP result projection — keep prompts compact; raw stays in artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence


MAX_PROJECTED_OBSERVATION_CHARS_PER_STEP = 6000
MAX_TOTAL_LEDGER_PROMPT_CHARS = 16000
MAX_HKEX_LIST_ROWS = 5


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 20)] + "\n…[truncated]…", True


def _as_rows(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    if isinstance(result, dict):
        for key in ("rows", "data", "items", "records", "result"):
            val = result.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        return [result]
    return []


def project_financial_rows(
    raw: Any,
    *,
    requested_metrics: Sequence[str] = (),
    periods: Sequence[str] = (),
) -> Dict[str, Any]:
    rows = _as_rows(raw)
    metric_keys = {str(m).lower() for m in requested_metrics if m}
    keep_identity = {
        "symbol",
        "code",
        "name",
        "date",
        "report_date",
        "period",
        "currency",
        "unit",
        "indicator",
    }
    projected_rows: List[Dict[str, Any]] = []
    for row in rows[:50]:
        slim: Dict[str, Any] = {}
        for k, v in row.items():
            kl = str(k).lower()
            if kl in keep_identity:
                slim[k] = v
            elif metric_keys and any(m in kl for m in metric_keys):
                slim[k] = v
            elif not metric_keys and isinstance(v, (int, float, str)) and len(slim) < 12:
                slim[k] = v
        if periods:
            period_blob = " ".join(str(row.get(k, "")) for k in row).lower()
            if not any(str(p).lower() in period_blob for p in periods):
                # Keep a few unfiltered rows if period match fails entirely later.
                pass
        if slim:
            projected_rows.append(slim)
    if periods and projected_rows:
        filtered = []
        for row in projected_rows:
            blob = " ".join(str(v) for v in row.values()).lower()
            if any(str(p).lower() in blob for p in periods):
                filtered.append(row)
        if filtered:
            projected_rows = filtered
    return {
        "kind": "financial_structured",
        "n_raw_rows": len(rows),
        "n_projected_rows": len(projected_rows),
        "rows": projected_rows[:20],
        "requested_metrics": list(requested_metrics),
        "periods": list(periods),
    }


def project_market_ohlcv(
    raw: Any,
    *,
    target_dates: Sequence[str] = (),
) -> Dict[str, Any]:
    rows = _as_rows(raw)
    keep = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "adjust",
        "currency",
        "symbol",
    }
    projected = []
    for row in rows:
        slim = {k: row[k] for k in row if str(k).lower() in keep or k in keep}
        if target_dates:
            d = str(row.get("date") or row.get("Date") or "")
            if d and not any(td in d for td in target_dates):
                continue
        if slim:
            projected.append(slim)
    if not projected and rows:
        # Fall back to last few rows if date window missed.
        for row in rows[-5:]:
            slim = {k: row[k] for k in row if str(k).lower() in keep or k in keep}
            if slim:
                projected.append(slim)
    return {
        "kind": "market_ohlcv",
        "n_raw_rows": len(rows),
        "n_projected_rows": len(projected),
        "rows": projected[:30],
        "target_dates": list(target_dates),
    }


def project_hkex_list(raw: Any, *, max_rows: int = MAX_HKEX_LIST_ROWS) -> Dict[str, Any]:
    rows = _as_rows(raw)
    keep = {
        "title",
        "release_date",
        "news_id",
        "local_key",
        "file_link",
        "stock_code",
        "stock_name",
    }
    projected = []
    for row in rows[: max(max_rows * 3, max_rows)]:
        slim = {k: row.get(k) for k in keep if row.get(k) is not None}
        # Also accept alternate key casings.
        for k, v in row.items():
            if str(k).lower() in keep and k not in slim:
                slim[k] = v
        if slim:
            projected.append(slim)
        if len(projected) >= max_rows:
            break
    return {
        "kind": "hkex_announcement_list",
        "n_raw_rows": len(rows),
        "n_projected_rows": len(projected),
        "rows": projected,
        "document_identified": bool(projected),
        "local_keys": [
            str(r.get("local_key")) for r in projected if r.get("local_key")
        ],
    }


def project_pdf_pages(raw: Any) -> Dict[str, Any]:
    rows = _as_rows(raw)
    projected = []
    for row in rows[:20]:
        slim = {
            k: row.get(k)
            for k in (
                "page",
                "page_number",
                "snippet",
                "text",
                "local_key",
                "document_id",
                "keyword",
            )
            if row.get(k) is not None
        }
        if slim:
            projected.append(slim)
    return {
        "kind": "pdf_pages_or_keywords",
        "n_raw_rows": len(rows),
        "n_projected_rows": len(projected),
        "rows": projected,
    }


def project_mcp_result(
    *,
    operation_id: str,
    arguments: Optional[Dict[str, Any]],
    raw_result: Any,
    task_spec: Optional[Dict[str, Any]] = None,
    step_objective: str = "",
) -> Dict[str, Any]:
    """Project a raw MCP invoke result into a compact observation."""
    task_spec = task_spec or {}
    oid = str(operation_id or "")
    blob = (oid + " " + step_objective).lower()
    raw_hash = _stable_hash({"operation_id": oid, "arguments": arguments, "raw": raw_result})

    if "hkex" in oid or "announcement" in oid and "list" in oid:
        projected = project_hkex_list(raw_result)
    elif any(k in blob for k in ("hist", "daily", "ohlc", "close", "market")) and (
        "financial" not in oid or "hist" in oid or "daily" in oid
    ):
        projected = project_market_ohlcv(
            raw_result, target_dates=list(task_spec.get("periods") or [])
        )
    elif any(k in oid for k in ("page", "keyword", "parse", "pdf")):
        projected = project_pdf_pages(raw_result)
    else:
        projected = project_financial_rows(
            raw_result,
            requested_metrics=list(task_spec.get("metrics") or []),
            periods=list(task_spec.get("periods") or []),
        )

    text = json.dumps(projected, ensure_ascii=False, default=str)
    text, truncated = _truncate_text(text, MAX_PROJECTED_OBSERVATION_CHARS_PER_STEP)
    if truncated:
        projected = json.loads(text) if text.startswith("{") else {"truncated_text": text}

    return {
        "operation_id": oid,
        "arguments": dict(arguments or {}),
        "raw_artifact_hash": raw_hash,
        "projected": projected,
        "projected_fields": sorted(
            {
                k
                for row in (projected.get("rows") or [])
                if isinstance(row, dict)
                for k in row.keys()
            }
        ),
        "truncated": truncated,
        "projected_chars": len(text),
        "raw_chars": len(
            json.dumps(raw_result, ensure_ascii=False, default=str)[:2_000_000]
        ),
    }


def compact_ledger_for_prompt(
    items: Sequence[Dict[str, Any]],
    *,
    max_chars: int = MAX_TOTAL_LEDGER_PROMPT_CHARS,
) -> Dict[str, Any]:
    """Build a compact verified-ledger view for Final Answer / exceptional replan."""
    compact_items: List[Dict[str, Any]] = []
    for item in items:
        if str(item.get("validation_status") or "") not in {"valid", "verified"}:
            continue
        compact_items.append(
            {
                "evidence_id": item.get("evidence_id"),
                "step_id": item.get("step_id"),
                "operation_id": item.get("operation_id"),
                "claim": item.get("claim"),
                "normalized_value": item.get("normalized_value"),
                "projection": (item.get("extras") or {}).get("projection")
                or item.get("projection"),
                "validation_status": item.get("validation_status"),
            }
        )
    blob = json.dumps(compact_items, ensure_ascii=False, default=str)
    truncated = False
    if len(blob) > max_chars:
        truncated = True
        # Drop oldest first.
        while compact_items and len(json.dumps(compact_items, ensure_ascii=False, default=str)) > max_chars:
            compact_items.pop(0)
        blob = json.dumps(compact_items, ensure_ascii=False, default=str)
    return {
        "items": compact_items,
        "n_items": len(compact_items),
        "chars": len(blob),
        "truncated": truncated,
    }
