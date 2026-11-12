"""Project UnifiedRealMcpGateway envelopes into required_outputs.

Reuses APV-Fin envelope normalizer + domain adapters (no Monitor/compiler).
Company-generic synonym map only — no task/issuer branching.

Handles both shapes seen in live/offline paths:
- OperationResolverServer wraps RealMcpResult.to_dict() as invoke_result['result']
- Recorded OV fixtures may be full outer {success, operation_id, result:{...}}
- Market Real-MCP gateway: outer.normalized.data / result.normalized.data bar object
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.apv_fin.output_adapters import (
    adapt_mcp_result,
    required_outputs_present,
)
from hk_equity_mcp.apv_fin.result_envelope import normalize_gateway_envelope
from hk_equity_mcp.pe_fin_lite.step_result import extract_values_from_envelope

MARKET_GATEWAY_PROJECTION_VERSION = "pe_fin_lite_market_gateway_projection_v2"


def market_gateway_projection_hash() -> str:
    blob = json.dumps(
        {
            "version": MARKET_GATEWAY_PROJECTION_VERSION,
            "shapes": [
                "outer.normalized.data",
                "result.normalized.data",
                "normalized.data",
                "result.data",
                "data",
            ],
            "fields": [
                "trade_date|date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjustment|adjustment_mode",
                "currency|currency_or_unit",
                "issuer_code|symbol",
            ],
            "adjustment_empty_string_legal": True,
            "none_adjustment_missing": True,
            "no_task_id_gold_branches": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Planner may emit short semantic names; adapters emit canonical fields.
REQUIRED_OUTPUT_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "issuer_code": (
        "issuer_code",
        "SECURITY_CODE",
        "SECUCODE",
        "stock_code",
        "hk_code",
        "symbol",
        "code",
        # entity last — adapters may put a display name there
        "entity",
    ),
    "company_name": (
        "company_name",
        "SECURITY_NAME_ABBR",
        "name",
        "entity_name",
        "entity",
    ),
    "report_date": ("report_date", "REPORT_DATE", "date"),
    "fiscal_period": (
        "fiscal_period",
        "FISCAL_YEAR",
        "period",
        "DATE_TYPE_CODE",
    ),
    "currency": ("currency", "CURRENCY", "currency_or_unit"),
    "eps": ("eps", "basic_eps", "BASIC_EPS"),
    "basic_eps": ("basic_eps", "eps", "BASIC_EPS"),
    "revenue": ("revenue", "OPERATE_INCOME", "operate_income"),
    "attributable_profit": (
        "attributable_profit",
        "HOLDER_PROFIT",
        "holder_profit",
        "net_profit",
    ),
    "open": ("open", "Open", "开盘"),
    "high": ("high", "High", "最高"),
    "low": ("low", "Low", "最低"),
    "close": ("close", "Close", "收盘"),
    "trade_date": ("trade_date", "date", "Date", "日期", "effective_session_date"),
    "volume": ("volume", "Volume", "成交量"),
    "adjustment": ("adjustment", "adjust", "ADJUST", "adjustment_mode"),
    "local_key": ("local_key", "localKey", "file_key"),
    "news_id": ("news_id", "newsId"),
    "snippet": ("snippet", "text", "content", "page_text"),
    "title": ("title", "Title", "headline"),
    "page_number": ("page_number", "page"),
    "release_date": ("release_date", "publish_time", "date"),
    "source_url": ("source_url", "url", "link"),
}


def _domain_capability(domain: Optional[str]) -> str:
    d = str(domain or "general").lower()
    if d == "market":
        return "market.daily_ohlcv"
    if d == "financial":
        return "financial.reported_metric_bundle"
    if d in {"disclosure", "news"}:
        return "disclosure.list"
    return "market.daily_ohlcv"


_ROW_CODE_KEYS = (
    "issuer_code",
    "SECURITY_CODE",
    "SECUCODE",
    "stock_code",
    "hk_code",
    "symbol",
    "code",
)
_ROW_NAME_KEYS = ("company_name", "SECURITY_NAME_ABBR", "name", "entity_name")


def _canonical_row_issuer_code(raw: Any) -> Optional[str]:
    from hk_equity_mcp.pe_fin_lite.issuer_code import canonicalize_hk_issuer_code

    return canonicalize_hk_issuer_code(raw)


def _collect_from_rows(rows: Any) -> Dict[str, Any]:
    """Lift common identity fields from the first non-empty row into the pool."""
    out: Dict[str, Any] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in _ROW_CODE_KEYS:
            if key not in out and row.get(key) not in (None, ""):
                # Never promote statement labels (e.g. 利润表) via symbol→issuer alias.
                if key in {"symbol", "code", "issuer_code", "SECURITY_CODE", "SECUCODE", "stock_code", "hk_code"}:
                    canon = _canonical_row_issuer_code(row.get(key))
                    if canon is None:
                        continue
                    out[key] = canon
                else:
                    out[key] = row.get(key)
        for key in _ROW_NAME_KEYS:
            if key not in out and row.get(key) not in (None, ""):
                out[key] = row.get(key)
        if out:
            break
    return out


def _collect_from_adapted(adapted: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    metrics = adapted.get("metrics") if isinstance(adapted.get("metrics"), dict) else {}
    for key, value in metrics.items():
        if value is not None and value != "":
            out[str(key)] = value
    for key, value in adapted.items():
        if key in {
            "metrics",
            "rows",
            "snippets",
            "documents",
            "envelope",
            "field_provenance",
            "raw_field_map",
        }:
            continue
        if value is not None and value != "" and not isinstance(value, (dict, list)):
            out[str(key)] = value
    if adapted.get("snippet") is None and adapted.get("snippets"):
        snips = adapted.get("snippets") or []
        if snips and isinstance(snips[0], dict) and snips[0].get("snippet"):
            out["snippet"] = snips[0]["snippet"]
    out.update(_collect_from_rows(adapted.get("rows")))
    return out


def _pool_has_value(pool: Dict[str, Any], syn: str, *, allow_empty: bool) -> bool:
    if syn not in pool:
        return False
    val = pool[syn]
    if val is None:
        return False
    if val == "" and not allow_empty:
        return False
    return True


def _resolve_required(
    pool: Dict[str, Any], required_outputs: Sequence[str]
) -> Tuple[Dict[str, Any], List[str]]:
    values: Dict[str, Any] = {}
    missing: List[str] = []
    for req in required_outputs:
        req_s = str(req)
        synonyms = REQUIRED_OUTPUT_SYNONYMS.get(req_s, (req_s,))
        allow_empty = req_s == "adjustment"
        found = None
        for syn in synonyms:
            if _pool_has_value(pool, syn, allow_empty=allow_empty):
                found = pool[syn]
                break
            for pk, pv in pool.items():
                if str(pk).lower() != str(syn).lower():
                    continue
                if pv is None:
                    continue
                if pv == "" and not allow_empty:
                    continue
                found = pv
                break
            if found is not None:
                break
        if found is None:
            missing.append(req_s)
            continue
        if req_s == "issuer_code":
            canon = _canonical_row_issuer_code(found)
            if canon is None:
                found = None
                for syn in synonyms:
                    candidates = []
                    if syn in pool and pool[syn] not in (None, ""):
                        candidates.append(pool[syn])
                    for pk, pv in pool.items():
                        if str(pk).lower() == str(syn).lower() and pv not in (None, ""):
                            candidates.append(pv)
                    for cand in candidates:
                        canon = _canonical_row_issuer_code(cand)
                        if canon is not None:
                            found = canon
                            break
                    if found is not None:
                        break
                if found is None:
                    missing.append(req_s)
                    continue
            else:
                found = canon
        if req_s == "adjustment":
            # None is never legal; only explicit "" (or non-empty adjust token).
            if found is None:
                missing.append(req_s)
                continue
        values[req_s] = found
    return values, missing


def _looks_like_market_bar(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    has_date = any(k in data for k in ("trade_date", "date", "日期", "Date"))
    has_ohlc = any(k in data for k in ("open", "high", "low", "close", "Open", "Close"))
    return bool(has_date or has_ohlc)


def _looks_like_market_normalized_layer(layer: Any) -> bool:
    if not isinstance(layer, dict):
        return False
    data = layer.get("data")
    if not _looks_like_market_bar(data):
        return False
    # Prefer layers that look like MarketRealMcpGateway normalized schema.
    if layer.get("source_operation") or layer.get("provider"):
        return True
    if "adjustment_mode" in layer or "currency_or_unit" in layer:
        return True
    if "effective_session_date" in layer:
        return True
    # Bare {data: bar} still accepted when bar shape is market-like.
    return True


def extract_market_gateway_normalized_data_v2(
    payload: Any,
) -> Optional[Dict[str, Any]]:
    """Locate Market Real-MCP normalized.data without task/gold branching."""
    if not isinstance(payload, dict):
        return None

    def _from_layer(layer: Any, path: str) -> Optional[Dict[str, Any]]:
        if not _looks_like_market_normalized_layer(layer):
            # Also accept direct bar object under "data" key only via layer check.
            if isinstance(layer, dict) and _looks_like_market_bar(layer) and path.endswith(".data"):
                return {"path": path, "normalized": {}, "data": layer}
            return None
        assert isinstance(layer, dict)
        return {"path": path, "normalized": layer, "data": layer.get("data")}

    result_layer = (
        payload.get("result") if isinstance(payload.get("result"), dict) else None
    )
    # Ordered shapes from the projection contract.
    candidates: List[Tuple[str, Any]] = []
    # If payload itself is already the normalized layer.
    if _looks_like_market_normalized_layer(payload):
        candidates.append(("normalized.data", payload))
    candidates.extend(
        [
            ("outer.normalized.data", payload.get("normalized")),
            (
                "result.normalized.data",
                (result_layer or {}).get("normalized") if result_layer else None,
            ),
            ("normalized.data", payload.get("normalized")),
            ("result.data", result_layer),
            ("data", payload),
        ]
    )

    for path, layer in candidates:
        if path == "result.data" and isinstance(layer, dict):
            if _looks_like_market_bar(layer.get("data")):
                hit = _from_layer(layer, path)
                if hit:
                    return hit
            continue
        if path == "data" and isinstance(layer, dict):
            if _looks_like_market_bar(layer.get("data")):
                hit = _from_layer(layer, "data")
                if hit:
                    return hit
            continue
        hit = _from_layer(layer, path)
        if hit:
            return hit
    return None


def project_market_gateway_bar_v2(
    *,
    normalized: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic field projection from market gateway normalized + data bar."""
    trade_date = data.get("trade_date")
    if trade_date in (None, ""):
        trade_date = data.get("date")
    if trade_date in (None, ""):
        trade_date = normalized.get("effective_session_date")

    adjustment = data.get("adjustment")
    if adjustment is None and "adjustment_mode" in normalized:
        adjustment = normalized.get("adjustment_mode")

    currency = data.get("currency")
    if currency in (None, ""):
        currency = normalized.get("currency_or_unit")

    issuer_raw = data.get("issuer_code")
    if issuer_raw in (None, ""):
        issuer_raw = normalized.get("symbol")
    issuer = _canonical_row_issuer_code(issuer_raw)

    values = {
        "trade_date": trade_date,
        "open": data.get("open"),
        "high": data.get("high"),
        "low": data.get("low"),
        "close": data.get("close"),
        "volume": data.get("volume"),
        "adjustment": adjustment,
        "currency": currency,
        "issuer_code": issuer,
    }
    provenance = {
        "trade_date": "observed",
        "open": "observed",
        "high": "observed",
        "low": "observed",
        "close": "observed",
        "volume": "observed",
        "adjustment": (
            "contract_derived"
            if adjustment == ""
            else ("observed" if adjustment is not None else None)
        ),
        "currency": "observed" if data.get("currency") not in (None, "") else "contract_derived",
        "issuer_code": "observed",
        "adjustment_mode": "unadjusted" if adjustment == "" else None,
        "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
        "market_gateway_projection_hash": market_gateway_projection_hash(),
    }
    return {
        "values": values,
        "field_provenance": provenance,
        "projection_ok": True,
        "projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
        "projection_hash": market_gateway_projection_hash(),
    }


def _candidate_payloads(invoke_result: Any) -> List[Any]:
    """Ordered unique payloads to try (live mid-layer first, then outer fixtures)."""
    out: List[Any] = []
    seen = set()

    def add(obj: Any) -> None:
        if obj is None:
            return
        key = id(obj)
        if key in seen:
            return
        seen.add(key)
        out.append(obj)

    if isinstance(invoke_result, dict):
        raw = invoke_result.get("result")
        add(raw)
        # If result is itself an outer invoke, also try its inner result (RealMcpResult).
        if isinstance(raw, dict) and isinstance(raw.get("result"), dict):
            add(raw.get("result"))
        add(invoke_result)
    else:
        add(invoke_result)
    return out


def _project_one(
    *,
    candidate: Any,
    required_outputs: Sequence[str],
    domain: Optional[str],
) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {}
    normalization_error: Optional[str] = None
    projection_ok = False
    projection_meta: Dict[str, Any] = {}
    try:
        if isinstance(candidate, dict):
            envelope = normalize_gateway_envelope(candidate)
            # Market gateway normalized.data is a supported path even when the
            # AKTools-table envelope normalizer reports no_supported_payload_path.
            if (
                isinstance(envelope.get("normalization_errors"), list)
                and "no_supported_payload_path" in envelope["normalization_errors"]
            ):
                hit = extract_market_gateway_normalized_data_v2(candidate)
                if hit is not None:
                    envelope = dict(envelope)
                    envelope["normalization_errors"] = [
                        e
                        for e in envelope["normalization_errors"]
                        if e != "no_supported_payload_path"
                    ]
                    envelope["market_gateway_data_path"] = hit["path"]
                    envelope["ok"] = True
                    normalization_error = None
        else:
            normalization_error = "non_object_result"
    except Exception as exc:  # noqa: BLE001
        normalization_error = f"{type(exc).__name__}: {exc}"
        envelope = {}

    pool = extract_values_from_envelope(envelope, list(required_outputs))
    if isinstance(envelope, dict):
        pool.update(_collect_from_rows(envelope.get("rows")))
        for nested_key in ("normalized", "raw", "result", "normalized_payloads"):
            nested = envelope.get(nested_key)
            if isinstance(nested, dict):
                pool.update(_collect_from_rows(nested.get("rows")))
                # Real HKEX list envelopes use announcements / data lists.
                pool.update(_collect_from_rows(nested.get("announcements")))
                data_obj = nested.get("data")
                if isinstance(data_obj, list):
                    pool.update(_collect_from_rows(data_obj))
                elif isinstance(data_obj, dict) and _looks_like_market_bar(data_obj):
                    # Single market bar object (not a row list).
                    for k, v in data_obj.items():
                        if k == "raw":
                            continue
                        if v is not None:
                            pool[str(k)] = v
            elif isinstance(nested, list):
                pool.update(_collect_from_rows(nested))

    # Versioned market gateway projection (primary path for Sina stock_hk_daily).
    market_hit = extract_market_gateway_normalized_data_v2(candidate)
    if market_hit is not None:
        projected_bar = project_market_gateway_bar_v2(
            normalized=dict(market_hit.get("normalized") or {}),
            data=dict(market_hit.get("data") or {}),
        )
        projection_ok = True
        projection_meta = {
            "market_gateway_data_path": market_hit.get("path"),
            "projection_version": projected_bar.get("projection_version"),
            "projection_hash": projected_bar.get("projection_hash"),
            "field_provenance": projected_bar.get("field_provenance"),
        }
        for key, value in (projected_bar.get("values") or {}).items():
            if value is None:
                continue
            # Always prefer observed market-bar values over empty adapter stubs.
            if key not in pool or pool[key] in (None, "") or key == "adjustment":
                pool[key] = value
            elif key in {"open", "high", "low", "close", "volume", "trade_date", "issuer_code", "currency"}:
                pool[key] = value

    # Disclosure-normalized announcement fields from raw candidate and envelope.
    # Gateway normalize may drop announcement rows into normalized_payloads only.
    try:
        from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
            _collect_rows as _disc_collect_rows,
            _row_to_values as _disc_row_to_values,
        )

        for src in (candidate, envelope):
            disc_rows = _disc_collect_rows(src)
            if disc_rows:
                pool.update(_disc_row_to_values(disc_rows[0]))
                pool.update(_collect_from_rows(disc_rows))
                break
    except Exception:  # noqa: BLE001
        pass
    adapted: Dict[str, Any] = {}
    try:
        adapted = adapt_mcp_result(
            capability=_domain_capability(domain),
            raw_result=candidate,
        )
    except Exception:  # noqa: BLE001
        adapted = {}
    if isinstance(adapted, dict):
        # Prefer row identity codes over display-name entity when both exist.
        adapted_pool = _collect_from_adapted(adapted)
        for key, value in adapted_pool.items():
            if key == "entity" and any(k in pool for k in _ROW_CODE_KEYS):
                continue
            # Do not let empty-string adapter fields wipe legal adjustment="".
            if key == "adjustment" and key in pool and pool[key] == "":
                continue
            if key not in pool or pool[key] in (None, ""):
                pool[key] = value
            elif key in _ROW_CODE_KEYS:
                pool[key] = value

    if str(domain or "").lower() in {"disclosure", "news"} and any(
        r in {"snippet", "local_key", "page_number"} for r in required_outputs
    ):
        try:
            text_adapted = adapt_mcp_result(
                capability="disclosure.page_read",
                raw_result=candidate,
            )
            if isinstance(text_adapted, dict):
                pool.update(_collect_from_adapted(text_adapted))
        except Exception:  # noqa: BLE001
            pass

    values, missing = _resolve_required(pool, required_outputs)
    present_check = required_outputs_present(
        {**pool, **values, "metrics": {k: v for k, v in pool.items()}},
        [REQUIRED_OUTPUT_SYNONYMS.get(r, (r,))[0] for r in required_outputs],
    )
    adapted_ok = bool(adapted.get("adapter_ok")) if isinstance(adapted, dict) else False
    if projection_ok and not missing:
        # Clear stale no_supported_payload_path once market data projected.
        if normalization_error == "no_supported_payload_path":
            normalization_error = None
        if isinstance(envelope, dict) and envelope.get("normalization_errors"):
            envelope = dict(envelope)
            envelope["normalization_errors"] = [
                e
                for e in (envelope.get("normalization_errors") or [])
                if e != "no_supported_payload_path"
            ]
    return {
        "values": values,
        "missing_outputs": missing,
        "envelope": envelope,
        "adapted_ok": adapted_ok or (projection_ok and not missing),
        "projection_ok": projection_ok and not missing,
        "normalization_error": normalization_error,
        "pool_keys": sorted(str(k) for k in pool.keys()),
        "adapter_present_check": present_check,
        "market_gateway_projection": projection_meta or None,
        "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
        "market_gateway_projection_hash": market_gateway_projection_hash(),
    }


def project_invoke_to_required_outputs(
    *,
    invoke_result: Dict[str, Any],
    required_outputs: Sequence[str],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize real Gateway nesting and project required_outputs deterministically."""
    best: Optional[Dict[str, Any]] = None
    for candidate in _candidate_payloads(invoke_result):
        projected = _project_one(
            candidate=candidate,
            required_outputs=required_outputs,
            domain=domain,
        )
        if best is None or len(projected["missing_outputs"]) < len(best["missing_outputs"]):
            best = projected
        if not projected["missing_outputs"]:
            break
    if best is None:
        return {
            "values": {},
            "missing_outputs": list(required_outputs),
            "envelope": {},
            "adapted_ok": False,
            "projection_ok": False,
            "normalization_error": "no_candidate",
            "pool_keys": [],
            "adapter_present_check": {
                "ok": False,
                "present": [],
                "missing": list(required_outputs),
            },
            "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
            "market_gateway_projection_hash": market_gateway_projection_hash(),
        }
    # Successful market projection clears stale envelope-normalizer errors.
    if not best.get("missing_outputs") and best.get("projection_ok"):
        best["normalization_error"] = None
    return best
