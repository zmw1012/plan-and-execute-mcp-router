"""Capability-scoped result adapters — project normalized envelopes to evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.apv_fin.financial_item_aliases import (
    WIDE_FIELD_ALIASES,
    map_long_table_item,
    map_wide_field,
)
from hk_equity_mcp.apv_fin.result_envelope import normalize_gateway_envelope

OUTPUT_ADAPTER_VERSION = "apv_fin_output_adapter_v1_4"

MARKET_COLUMN_ALIASES: Dict[str, str] = {
    "日期": "trade_date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "turnover",
    "date": "trade_date",
    "Date": "trade_date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "turnover": "turnover",
}


def output_adapter_hash() -> str:
    blob = json.dumps(
        {
            "version": OUTPUT_ADAPTER_VERSION,
            "market_aliases": MARKET_COLUMN_ALIASES,
            "wide_aliases": WIDE_FIELD_ALIASES,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
        for rk, rv in row.items():
            if str(rk).lower() == str(k).lower() and rv is not None:
                return rv
    return None


def _parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Keep YYYY-MM-DD when present.
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    for fmt in ("%Y/%m/%d", "%Y%m%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text[:10]


def _date_key(value: Any) -> str:
    return _parse_date(value) or ""


def _period_candidates(task_spec: Optional[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for p in list((task_spec or {}).get("periods") or []):
        d = _parse_date(p)
        if d:
            out.append(d)
        # Also keep raw year tokens.
        m = re.search(r"(20\d{2})", str(p))
        if m:
            out.append(m.group(1))
    return out


def _cutoff_date(task_spec: Optional[Dict[str, Any]], default: str = "2026-08-12") -> str:
    raw = (task_spec or {}).get("cutoff_date")
    return _parse_date(raw) or default


def _rows_from_raw(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Prefer envelope normalization; fall back only for already-flat row lists."""
    envelope: Dict[str, Any]
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
        envelope = {
            "rows": rows,
            "truncated": False,
            "offset": 0,
            "returned_rows": len(rows),
            "total_rows": len(rows),
            "execution_success": True,
            "protocol_success": True,
            "textual_content": [],
            "raw_provenance_paths": ["list_input"],
            "normalization_errors": [],
            "ok": True,
        }
        return rows, envelope
    try:
        envelope = normalize_gateway_envelope(raw)
        return list(envelope.get("rows") or []), envelope
    except Exception as exc:  # noqa: BLE001 — adapters stay fail-soft
        return [], {
            "rows": [],
            "truncated": False,
            "offset": 0,
            "returned_rows": 0,
            "total_rows": None,
            "execution_success": None,
            "protocol_success": None,
            "textual_content": [],
            "raw_provenance_paths": [],
            "normalization_errors": [f"{type(exc).__name__}:{exc}"],
            "ok": False,
        }


def _is_wide_financial_row(row: Dict[str, Any]) -> bool:
    keys = {str(k).upper() for k in row.keys()}
    return bool(keys & {"BASIC_EPS", "OPERATE_INCOME", "HOLDER_PROFIT", "DILUTED_EPS"})


def _is_long_financial_row(row: Dict[str, Any]) -> bool:
    keys = {str(k).upper() for k in row.keys()}
    return "STD_ITEM_NAME" in keys or "STD_ITEM_CODE" in keys or "AMOUNT" in keys


def _select_wide_row(
    rows: Sequence[Dict[str, Any]], task_spec: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], int]:
    if not rows:
        return None, -1
    periods = _period_candidates(task_spec)
    # Prefer exact report_date match, then year match, else latest by report_date.
    scored: List[Tuple[int, str, int, Dict[str, Any]]] = []
    for i, row in enumerate(rows):
        rd = _parse_date(_first_present(row, ("REPORT_DATE", "report_date")))
        score = 0
        if rd and rd in periods:
            score = 100
        elif rd and any(rd.startswith(p) or p in rd for p in periods if len(p) == 4):
            score = 50
        scored.append((score, rd or "", i, row))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    best = scored[0]
    if best[0] > 0:
        return best[3], best[2]
    # No period cue: choose latest report_date.
    by_date = sorted(
        (( _date_key(_first_present(r, ("REPORT_DATE", "report_date"))), i, r) for i, r in enumerate(rows)),
        key=lambda t: (t[0], t[1]),
        reverse=True,
    )
    return by_date[0][2], by_date[0][1]


def adapt_financial_reported_metric_bundle(
    raw: Any,
    *,
    field_aliases: Optional[Dict[str, str]] = None,
    task_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows, envelope = _rows_from_raw(raw)
    aliases = dict(WIDE_FIELD_ALIASES)
    aliases.update(dict(field_aliases or {}))

    metrics: Dict[str, Any] = {}
    provenance: List[Dict[str, Any]] = []
    entity = None
    report_date = None
    currency = None
    selected_index = -1
    pagination_needed = False

    if rows and _is_wide_financial_row(rows[0]):
        selected, selected_index = _select_wide_row(rows, task_spec)
        if selected:
            entity = _first_present(
                selected, ("SECURITY_NAME_ABBR", "entity", "symbol", "stock", "name")
            )
            report_date = _parse_date(
                _first_present(selected, ("REPORT_DATE", "report_date"))
            )
            currency = _first_present(selected, ("CURRENCY", "currency"))
            for raw_k, val in selected.items():
                canon = aliases.get(str(raw_k)) or aliases.get(str(raw_k).upper())
                canon = canon or map_wide_field(str(raw_k))
                if not canon or val is None:
                    continue
                if canon in {"report_date"}:
                    report_date = _parse_date(val) or report_date
                    continue
                if canon == "currency":
                    currency = val
                    continue
                if canon not in metrics:
                    metrics[canon] = val
                    provenance.append(
                        {
                            "canonical": canon,
                            "original_field": str(raw_k),
                            "item_code": None,
                            "raw_amount": val,
                            "source_row_index": selected_index,
                            "row_style": "wide",
                        }
                    )
    else:
        # Long-table / mixed.
        for i, row in enumerate(rows):
            entity = entity or _first_present(
                row, ("SECURITY_NAME_ABBR", "entity", "symbol", "stock", "name")
            )
            rd = _parse_date(_first_present(row, ("REPORT_DATE", "report_date")))
            if rd and (report_date is None or rd > report_date):
                # Keep a report_date anchor; period filter below.
                pass
            periods = _period_candidates(task_spec)
            if periods and rd:
                year_ok = any(rd.startswith(p) or p in rd for p in periods)
                exact_ok = rd in periods
                if periods and not (year_ok or exact_ok):
                    # Still allow if no year-like periods.
                    if any(len(p) >= 8 for p in periods) and not exact_ok:
                        continue
            item_name = _first_present(row, ("STD_ITEM_NAME", "item_name", "name"))
            item_code = _first_present(row, ("STD_ITEM_CODE", "item_code"))
            amount = _first_present(row, ("AMOUNT", "amount", "value"))
            canon = map_long_table_item(item_name=item_name, item_code=item_code)
            if canon and amount is not None and canon not in metrics:
                metrics[canon] = amount
                if rd:
                    report_date = report_date or rd
                provenance.append(
                    {
                        "canonical": canon,
                        "original_field": str(item_name) if item_name is not None else None,
                        "item_code": str(item_code) if item_code is not None else None,
                        "raw_amount": amount,
                        "source_row_index": i,
                        "row_style": "long",
                        "report_date": rd,
                    }
                )
            # Wide fields may also appear on mixed rows.
            for raw_k, val in row.items():
                mapped = map_wide_field(str(raw_k))
                if not mapped or val is None:
                    continue
                if mapped == "report_date":
                    report_date = report_date or _parse_date(val)
                    continue
                if mapped == "currency":
                    currency = currency or val
                    continue
                if mapped not in metrics:
                    metrics[mapped] = val
                    provenance.append(
                        {
                            "canonical": mapped,
                            "original_field": str(raw_k),
                            "item_code": None,
                            "raw_amount": val,
                            "source_row_index": i,
                            "row_style": "wide_in_mixed",
                        }
                    )
        if envelope.get("truncated") and (
            "attributable_profit" not in metrics or "revenue" not in metrics
        ):
            pagination_needed = True

    raw_field_map = {
        str(p["canonical"]): str(p["original_field"])
        for p in provenance
        if p.get("canonical") and p.get("original_field")
    }
    return {
        "capability": "financial.reported_metric_bundle",
        "entity": entity,
        "report_date": report_date,
        "currency": currency,
        "metrics": metrics,
        "basic_eps": metrics.get("basic_eps"),
        "revenue": metrics.get("revenue"),
        "attributable_profit": metrics.get("attributable_profit"),
        "field_provenance": provenance,
        "raw_field_map": raw_field_map,
        "selected_row_index": selected_index,
        "n_rows": len(rows),
        "envelope": {
            "truncated": envelope.get("truncated"),
            "offset": envelope.get("offset"),
            "returned_rows": envelope.get("returned_rows"),
            "total_rows": envelope.get("total_rows"),
            "provenance_paths": envelope.get("raw_provenance_paths"),
            "normalization_errors": envelope.get("normalization_errors"),
        },
        "pagination_needed": pagination_needed,
        "adapter_version": OUTPUT_ADAPTER_VERSION,
        "adapter_ok": bool(metrics) or bool(report_date or currency),
    }


def adapt_market_daily_ohlcv(
    raw: Any,
    *,
    field_aliases: Optional[Dict[str, str]] = None,
    ticker: Optional[str] = None,
    task_spec: Optional[Dict[str, Any]] = None,
    as_of: Optional[str] = None,
    prefer_latest: bool = True,
) -> Dict[str, Any]:
    rows, envelope = _rows_from_raw(raw)
    aliases = dict(MARKET_COLUMN_ALIASES)
    aliases.update(dict(field_aliases or {}))
    cutoff = _parse_date(as_of) or _cutoff_date(task_spec)

    projected: List[Dict[str, Any]] = []
    for row in rows:
        mapped = dict(row)
        for raw_k, canon in aliases.items():
            if raw_k in row and canon not in mapped:
                mapped[canon] = row[raw_k]
        item = {
            "ticker": ticker or _first_present(mapped, ("ticker", "symbol", "code")),
            "trade_date": _parse_date(
                _first_present(mapped, ("trade_date", "date", "Date", "日期"))
            ),
            "open": _first_present(mapped, ("open", "Open", "开盘")),
            "high": _first_present(mapped, ("high", "High", "最高")),
            "low": _first_present(mapped, ("low", "Low", "最低")),
            "close": _first_present(mapped, ("close", "Close", "收盘")),
            "volume": _first_present(mapped, ("volume", "Volume", "成交量")),
            "turnover": _first_present(mapped, ("turnover", "成交额")),
            "adjust": _first_present(mapped, ("adjust", "adjustflag")),
        }
        if item["trade_date"] is None and item["close"] is None:
            continue
        if prefer_latest and item["trade_date"] and item["trade_date"] > cutoff:
            continue
        projected.append(item)

    projected.sort(key=lambda r: r.get("trade_date") or "")
    latest = projected[-1] if projected else {}
    return {
        "capability": "market.daily_ohlcv",
        "ticker": latest.get("ticker") or ticker,
        "rows": projected[-60:],
        "trade_date": latest.get("trade_date"),
        "open": latest.get("open"),
        "high": latest.get("high"),
        "low": latest.get("low"),
        "close": latest.get("close"),
        "volume": latest.get("volume"),
        "turnover": latest.get("turnover"),
        "adjust": latest.get("adjust"),
        "currency": "HKD",
        "currency_source": "contract_derived",
        "cutoff_date": cutoff,
        "n_rows": len(projected),
        "envelope": {
            "truncated": envelope.get("truncated"),
            "offset": envelope.get("offset"),
            "returned_rows": envelope.get("returned_rows"),
            "total_rows": envelope.get("total_rows"),
            "provenance_paths": envelope.get("raw_provenance_paths"),
        },
        "adapter_version": OUTPUT_ADAPTER_VERSION,
        "adapter_ok": bool(latest.get("close") is not None or latest.get("volume") is not None),
    }


def adapt_disclosure_list(raw: Any) -> Dict[str, Any]:
    rows, envelope = _rows_from_raw(raw)
    # Some disclosure ops return textual JSON blobs.
    if not rows and envelope.get("textual_content"):
        # Best-effort: ignore free text for list identity.
        pass
    documents = []
    for row in rows[:40]:
        doc = {
            "title": _first_present(row, ("title", "Title")),
            "release_date": _first_present(
                row, ("release_date", "publish_time", "date", "NOTICE_DATE")
            ),
            "news_id": _first_present(row, ("news_id", "announcement_id", "id")),
            "local_key": _first_present(row, ("local_key", "file_key")),
            "stock_code": _first_present(row, ("stock_code", "symbol", "code")),
            "file_link": _first_present(row, ("file_link", "url", "link")),
        }
        if any(doc.values()):
            documents.append(doc)
    return {
        "capability": "disclosure.list",
        "documents": documents,
        "document_identified": bool(documents),
        "local_key": documents[0]["local_key"] if documents else None,
        "local_keys": [d["local_key"] for d in documents if d.get("local_key")],
        "news_id": documents[0]["news_id"] if documents else None,
        "release_date": documents[0]["release_date"] if documents else None,
        "stock_code": documents[0]["stock_code"] if documents else None,
        "title": documents[0]["title"] if documents else None,
        "n_documents": len(documents),
        "envelope": {
            "provenance_paths": envelope.get("raw_provenance_paths"),
            "normalization_errors": envelope.get("normalization_errors"),
        },
        "adapter_version": OUTPUT_ADAPTER_VERSION,
        "adapter_ok": bool(documents),
    }


def adapt_disclosure_text(raw: Any, *, capability: str) -> Dict[str, Any]:
    rows, envelope = _rows_from_raw(raw)
    snippets = []
    local_key = None
    page_number = None
    news_id = None
    release_date = None
    stock_code = None
    title = None
    for row in rows[:40]:
        local_key = local_key or _first_present(row, ("local_key",))
        page_number = page_number or _first_present(row, ("page_number", "page"))
        news_id = news_id or _first_present(row, ("news_id",))
        release_date = release_date or _first_present(row, ("release_date",))
        stock_code = stock_code or _first_present(row, ("stock_code",))
        title = title or _first_present(row, ("title",))
        snip = _first_present(row, ("snippet", "text", "content", "page_text"))
        if snip:
            snippets.append(
                {
                    "snippet": snip if isinstance(snip, str) else str(snip)[:800],
                    "page_number": _first_present(row, ("page_number", "page")),
                    "local_key": _first_present(row, ("local_key",)),
                }
            )
    for text in envelope.get("textual_content") or []:
        if text:
            snippets.append(
                {"snippet": str(text)[:800], "page_number": None, "local_key": local_key}
            )
    return {
        "capability": capability,
        "snippet": snippets[0]["snippet"] if snippets else None,
        "snippets": snippets,
        "local_key": local_key,
        "page_number": page_number,
        "news_id": news_id,
        "release_date": release_date,
        "stock_code": stock_code,
        "title": title,
        "textual_content": list(envelope.get("textual_content") or []),
        "n_snippets": len(snippets),
        "envelope": {
            "provenance_paths": envelope.get("raw_provenance_paths"),
            "normalization_errors": envelope.get("normalization_errors"),
        },
        "adapter_version": OUTPUT_ADAPTER_VERSION,
        "adapter_ok": bool(snippets or local_key),
    }


def adapt_mcp_result(
    *,
    capability: str,
    raw_result: Any,
    contract: Optional[Dict[str, Any]] = None,
    arguments: Optional[Dict[str, Any]] = None,
    task_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch adapter by required/selected capability using normalized envelopes."""
    contract = contract or {}
    aliases = dict(contract.get("field_aliases") or {})
    args = arguments or {}
    ticker = args.get("symbol") or args.get("stock") or args.get("stock_code")

    if capability in {
        "financial.reported_metric_bundle",
        "financial.reported_metric",
        "financial.ratio",
    }:
        adapted = adapt_financial_reported_metric_bundle(
            raw_result, field_aliases=aliases, task_spec=task_spec
        )
        if capability != "financial.reported_metric_bundle":
            adapted["capability"] = capability
        return adapted
    if capability in {"market.daily_ohlcv", "market.latest_close", "market.index_daily"}:
        adapted = adapt_market_daily_ohlcv(
            raw_result,
            field_aliases=aliases,
            ticker=str(ticker) if ticker else None,
            task_spec=task_spec,
            prefer_latest=True,
        )
        adapted["capability"] = capability
        return adapted
    if capability in {"disclosure.list", "news.official_announcement"}:
        return adapt_disclosure_list(raw_result)
    if capability in {
        "disclosure.download",
        "disclosure.parse",
        "disclosure.keyword_search",
        "disclosure.page_read",
        "financial.official_narrative_evidence",
    }:
        return adapt_disclosure_text(raw_result, capability=capability)
    if capability == "disclosure.company_lookup":
        rows, _env = _rows_from_raw(raw_result)
        code = None
        if rows:
            code = _first_present(rows[0], ("issuer_code", "stock_code", "symbol", "hk_code"))
        return {
            "capability": capability,
            "issuer_code": code,
            "entity": code,
            "adapter_ok": code is not None,
        }
    if capability == "news.media_latest":
        rows, _env = _rows_from_raw(raw_result)
        title = _first_present(rows[0], ("title", "headline")) if rows else None
        return {
            "capability": capability,
            "title": title,
            "n_rows": len(rows),
            "adapter_ok": title is not None,
        }
    if capability == "market.trading_calendar":
        rows, _env = _rows_from_raw(raw_result)
        return {
            "capability": capability,
            "trade_date": _first_present(rows[0], ("trade_date", "date")) if rows else None,
            "n_rows": len(rows),
            "adapter_ok": bool(rows),
        }
    if capability == "financial.company_profile":
        rows, _env = _rows_from_raw(raw_result)
        return {
            "capability": capability,
            "entity": _first_present(rows[0], ("entity", "name", "symbol")) if rows else None,
            "adapter_ok": bool(rows),
        }
    rows, envelope = _rows_from_raw(raw_result)
    return {
        "capability": capability,
        "raw_keys": list(rows[0].keys()) if rows else [],
        "n_rows": len(rows),
        "envelope": {
            "provenance_paths": envelope.get("raw_provenance_paths"),
            "normalization_errors": envelope.get("normalization_errors"),
        },
        "adapter_ok": bool(rows),
    }


def required_outputs_present(
    adapted: Dict[str, Any], required_outputs: Sequence[str]
) -> Dict[str, Any]:
    """Check required_outputs against adapted projection (including nested metrics)."""
    missing: List[str] = []
    present: List[str] = []
    metrics = adapted.get("metrics") if isinstance(adapted.get("metrics"), dict) else {}
    for key in required_outputs:
        found = False
        if key in adapted and adapted[key] is not None:
            found = True
        elif key in metrics and metrics[key] is not None:
            found = True
        elif key.startswith("metrics.") and metrics.get(key.split(".", 1)[1]) is not None:
            found = True
        elif key == "snippet" and adapted.get("snippets"):
            found = True
        if found:
            present.append(key)
        else:
            missing.append(key)
    return {
        "ok": len(missing) == 0,
        "present": present,
        "missing": missing,
    }


def flatten_projected_fields(adapted: Dict[str, Any]) -> Dict[str, Any]:
    """Flat map for operand refs like s1.metrics.revenue / s1.close."""
    out: Dict[str, Any] = {}
    for k, v in adapted.items():
        if k in {"envelope", "field_provenance", "rows", "snippets", "documents"}:
            continue
        if k == "metrics" and isinstance(v, dict):
            for mk, mv in v.items():
                out[f"metrics.{mk}"] = mv
                out[mk] = mv
        elif not isinstance(v, (dict, list)):
            out[k] = v
    rows = adapted.get("rows")
    if isinstance(rows, list) and rows:
        last = rows[-1]
        if isinstance(last, dict):
            for rk, rv in last.items():
                out.setdefault(rk, rv)
    return out
