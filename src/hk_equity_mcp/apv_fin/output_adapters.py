"""Capability-scoped result adapters — project raw MCP results to typed evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


def _as_rows(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        for key in ("rows", "data", "items", "records", "result", "documents"):
            val = raw.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        return [raw]
    return []


def _apply_aliases(row: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    out = dict(row)
    for raw_key, canon in (aliases or {}).items():
        if raw_key in row and canon not in out:
            out[canon] = row[raw_key]
        # case-insensitive
        for k, v in row.items():
            if str(k).lower() == str(raw_key).lower() and canon not in out:
                out[canon] = v
    return out


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
        for rk, rv in row.items():
            if str(rk).lower() == str(k).lower() and rv is not None:
                return rv
    return None


def adapt_financial_reported_metric_bundle(
    raw: Any, *, field_aliases: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    aliases = dict(field_aliases or {})
    # Default EM aliases.
    aliases.setdefault("OPERATE_INCOME", "revenue")
    aliases.setdefault("HOLDER_PROFIT", "attributable_profit")
    aliases.setdefault("BASIC_EPS", "basic_eps")
    aliases.setdefault("CURRENCY", "currency")
    aliases.setdefault("REPORT_DATE", "report_date")
    rows = [_apply_aliases(r, aliases) for r in _as_rows(raw)]
    metrics: Dict[str, Any] = {}
    raw_field_map: Dict[str, str] = {}
    entity = None
    report_date = None
    currency = None
    for row in rows:
        entity = entity or _first_present(row, ("entity", "symbol", "stock", "name", "SECURITY_NAME_ABBR"))
        report_date = report_date or _first_present(row, ("report_date", "REPORT_DATE", "date"))
        currency = currency or _first_present(row, ("currency", "CURRENCY"))
        for canon in ("revenue", "attributable_profit", "basic_eps", "holder_profit", "operating_income"):
            if canon in row and row[canon] is not None and canon not in metrics:
                metrics[canon] = row[canon]
        # Preserve attributable_profit alias from holder_profit.
        if "attributable_profit" not in metrics and "holder_profit" in metrics:
            metrics["attributable_profit"] = metrics["holder_profit"]
        if "revenue" not in metrics and "operating_income" in metrics:
            metrics["revenue"] = metrics["operating_income"]
        # Record original keys that mapped.
        for raw_k, canon in aliases.items():
            if raw_k in row or any(str(k) == raw_k for k in row):
                raw_field_map[canon] = raw_k
    return {
        "capability": "financial.reported_metric_bundle",
        "entity": entity,
        "report_date": report_date,
        "currency": currency,
        "metrics": metrics,
        "raw_field_map": raw_field_map,
        "n_rows": len(rows),
    }


def adapt_market_daily_ohlcv(
    raw: Any, *, field_aliases: Optional[Dict[str, str]] = None, ticker: Optional[str] = None
) -> Dict[str, Any]:
    aliases = dict(field_aliases or {})
    aliases.setdefault("日期", "trade_date")
    aliases.setdefault("收盘", "close")
    aliases.setdefault("开盘", "open")
    aliases.setdefault("最高", "high")
    aliases.setdefault("最低", "low")
    aliases.setdefault("成交量", "volume")
    aliases.setdefault("date", "trade_date")
    rows = [_apply_aliases(r, aliases) for r in _as_rows(raw)]
    projected = []
    for row in rows:
        item = {
            "ticker": ticker or _first_present(row, ("ticker", "symbol", "code")),
            "trade_date": _first_present(row, ("trade_date", "date", "Date")),
            "open": _first_present(row, ("open", "Open")),
            "high": _first_present(row, ("high", "High")),
            "low": _first_present(row, ("low", "Low")),
            "close": _first_present(row, ("close", "Close")),
            "volume": _first_present(row, ("volume", "Volume")),
            "adjust": _first_present(row, ("adjust", "adjustflag")),
        }
        if item["trade_date"] is not None or item["close"] is not None:
            projected.append(item)
    latest = projected[-1] if projected else {}
    return {
        "capability": "market.daily_ohlcv",
        "ticker": latest.get("ticker") or ticker,
        "rows": projected[:60],
        "trade_date": latest.get("trade_date"),
        "open": latest.get("open"),
        "high": latest.get("high"),
        "low": latest.get("low"),
        "close": latest.get("close"),
        "volume": latest.get("volume"),
        "adjust": latest.get("adjust"),
        "n_rows": len(projected),
    }


def adapt_disclosure_list(raw: Any) -> Dict[str, Any]:
    rows = _as_rows(raw)
    documents = []
    for row in rows[:20]:
        doc = {
            "title": _first_present(row, ("title", "Title")),
            "release_date": _first_present(row, ("release_date", "publish_time", "date")),
            "news_id": _first_present(row, ("news_id", "announcement_id", "id")),
            "local_key": _first_present(row, ("local_key", "file_key")),
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
        "title": documents[0]["title"] if documents else None,
        "n_documents": len(documents),
    }


def adapt_disclosure_text(raw: Any, *, capability: str) -> Dict[str, Any]:
    rows = _as_rows(raw)
    snippets = []
    local_key = None
    page_number = None
    for row in rows[:30]:
        local_key = local_key or _first_present(row, ("local_key",))
        page_number = page_number or _first_present(row, ("page_number", "page"))
        snip = _first_present(row, ("snippet", "text", "content", "page_text"))
        if snip:
            snippets.append(
                {
                    "snippet": snip if isinstance(snip, str) else str(snip)[:800],
                    "page_number": _first_present(row, ("page_number", "page")),
                    "local_key": _first_present(row, ("local_key",)),
                }
            )
    # Single blob result.
    if not snippets and isinstance(raw, dict):
        text = _first_present(raw, ("text", "parsed_text", "content", "snippet"))
        if text:
            snippets.append({"snippet": str(text)[:800], "page_number": None, "local_key": raw.get("local_key")})
            local_key = local_key or raw.get("local_key")
    return {
        "capability": capability,
        "snippet": snippets[0]["snippet"] if snippets else None,
        "snippets": snippets,
        "local_key": local_key,
        "page_number": page_number,
        "n_snippets": len(snippets),
    }


def adapt_mcp_result(
    *,
    capability: str,
    raw_result: Any,
    contract: Optional[Dict[str, Any]] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch adapter by required/selected capability."""
    contract = contract or {}
    aliases = dict(contract.get("field_aliases") or {})
    args = arguments or {}
    ticker = args.get("symbol") or args.get("stock") or args.get("stock_code")

    if capability in {
        "financial.reported_metric_bundle",
        "financial.reported_metric",
        "financial.ratio",
    }:
        adapted = adapt_financial_reported_metric_bundle(raw_result, field_aliases=aliases)
        if capability != "financial.reported_metric_bundle":
            adapted["capability"] = capability
        return adapted
    if capability in {"market.daily_ohlcv", "market.latest_close", "market.index_daily"}:
        adapted = adapt_market_daily_ohlcv(
            raw_result, field_aliases=aliases, ticker=str(ticker) if ticker else None
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
        rows = _as_rows(raw_result)
        code = None
        if rows:
            code = _first_present(rows[0], ("issuer_code", "stock_code", "symbol", "hk_code"))
        return {"capability": capability, "issuer_code": code, "entity": code}
    if capability == "news.media_latest":
        rows = _as_rows(raw_result)
        title = _first_present(rows[0], ("title", "headline")) if rows else None
        return {"capability": capability, "title": title, "n_rows": len(rows)}
    if capability == "market.trading_calendar":
        rows = _as_rows(raw_result)
        return {
            "capability": capability,
            "trade_date": _first_present(rows[0], ("trade_date", "date")) if rows else None,
            "n_rows": len(rows),
        }
    if capability == "financial.company_profile":
        rows = _as_rows(raw_result)
        return {
            "capability": capability,
            "entity": _first_present(rows[0], ("entity", "name", "symbol")) if rows else None,
        }
    # Generic fallback.
    return {"capability": capability, "raw_keys": list((_as_rows(raw_result)[0] or {}).keys()) if _as_rows(raw_result) else []}


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
        if k == "metrics" and isinstance(v, dict):
            for mk, mv in v.items():
                out[f"metrics.{mk}"] = mv
                out[mk] = mv
        elif k == "rows" and isinstance(v, list) and v:
            # expose latest row fields
            last = v[-1]
            if isinstance(last, dict):
                for rk, rv in last.items():
                    out.setdefault(rk, rv)
        elif not isinstance(v, (dict, list)):
            out[k] = v
    return out
