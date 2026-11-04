"""Market Real-MCP row normalization, cutoff filter, previous-valid-day."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

MARKET_CUTOFF = date(2026, 8, 12)
MARKET_TIMEZONE = "Asia/Hong_Kong"
CURRENCY_DEFAULT = "HKD"
CURRENCY_PROVENANCE = "exchange_convention"


def parse_session_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", "-")
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def to_ymd(d: date) -> str:
    return d.isoformat()


def to_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def row_date(row: Dict[str, Any]) -> Optional[date]:
    for key in ("date", "日期", "day", "trade_date", "时间", "datetime"):
        if key in row:
            d = parse_session_date(row.get(key))
            if d is not None:
                return d
    return None


def row_close(row: Dict[str, Any]) -> Optional[float]:
    for key in ("close", "收盘", "Close", "最新价", "close_price"):
        if key in row and row[key] is not None and str(row[key]) != "":
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def row_field(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def extract_rows(normalized: Any) -> List[Dict[str, Any]]:
    if normalized is None:
        return []
    if isinstance(normalized, list):
        return [r for r in normalized if isinstance(r, dict)]
    if isinstance(normalized, dict):
        for key in ("data", "rows", "items", "result"):
            val = normalized.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        # single row object
        if any(k in normalized for k in ("date", "日期", "close", "收盘")):
            return [normalized]
    return []


def filter_cutoff(
    rows: Sequence[Dict[str, Any]],
    *,
    cutoff: date = MARKET_CUTOFF,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in rows:
        d = row_date(row)
        if d is None:
            continue
        if d <= cutoff:
            kept.append(dict(row))
    return kept


def sort_by_date(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dated: List[Tuple[date, Dict[str, Any]]] = []
    for row in rows:
        d = row_date(row)
        if d is None:
            continue
        dated.append((d, row))
    dated.sort(key=lambda x: x[0])
    return [r for _, r in dated]


def previous_valid_session(
    rows: Sequence[Dict[str, Any]],
    requested: date,
    *,
    cutoff: date = MARKET_CUTOFF,
) -> Optional[Dict[str, Any]]:
    """max(available session date <= requested date and <= cutoff). No forward-fill."""
    if requested > cutoff:
        requested = cutoff
    candidates = [
        r
        for r in sort_by_date(filter_cutoff(rows, cutoff=cutoff))
        if (row_date(r) or date.min) <= requested
    ]
    return candidates[-1] if candidates else None


def normalize_bar(row: Dict[str, Any]) -> Dict[str, Any]:
    d = row_date(row)
    return {
        "date": to_ymd(d) if d else None,
        "open": _to_float(row_field(row, "open", "开盘", "Open")),
        "high": _to_float(row_field(row, "high", "最高", "High")),
        "low": _to_float(row_field(row, "low", "最低", "Low")),
        "close": _to_float(row_field(row, "close", "收盘", "Close")),
        "volume": _to_float(row_field(row, "volume", "成交量", "Volume")),
        "turnover": _to_float(
            row_field(row, "turnover", "成交额", "amount", "成交金额", "Amount")
        ),
        "raw": row,
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def market_result_schema(
    *,
    success: bool,
    source_server: str,
    source_operation: str,
    provider: Optional[str],
    symbol: Optional[str],
    requested_date_or_range: Any,
    effective_session_date: Optional[str],
    timezone: str = MARKET_TIMEZONE,
    currency_or_unit: Optional[str] = CURRENCY_DEFAULT,
    currency_provenance: Optional[str] = CURRENCY_PROVENANCE,
    adjustment_mode: Optional[str],
    data: Any,
    provenance: Dict[str, Any],
    error: Optional[str] = None,
    previous_valid_day_triggered: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "success": success,
        "source_server": source_server,
        "source_operation": source_operation,
        "provider": provider,
        "symbol": symbol,
        "requested_date_or_range": requested_date_or_range,
        "effective_session_date": effective_session_date,
        "timezone": timezone,
        "currency_or_unit": currency_or_unit,
        "currency_provenance": currency_provenance,
        "adjustment_mode": adjustment_mode,
        "data": data,
        "provenance": provenance,
        "previous_valid_day_triggered": previous_valid_day_triggered,
        "error": error,
    }
