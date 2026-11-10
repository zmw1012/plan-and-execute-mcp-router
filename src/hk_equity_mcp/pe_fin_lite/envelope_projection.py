"""Project UnifiedRealMcpGateway envelopes into required_outputs.

Reuses APV-Fin envelope normalizer + domain adapters (no Monitor/compiler).
Company-generic synonym map only — no task/issuer branching.

Handles both shapes seen in live/offline paths:
- OperationResolverServer wraps RealMcpResult.to_dict() as invoke_result['result']
- Recorded OV fixtures may be full outer {success, operation_id, result:{...}}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.apv_fin.output_adapters import (
    adapt_mcp_result,
    required_outputs_present,
)
from hk_equity_mcp.apv_fin.result_envelope import normalize_gateway_envelope
from hk_equity_mcp.pe_fin_lite.step_result import extract_values_from_envelope

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
    "currency": ("currency", "CURRENCY"),
    "eps": ("eps", "basic_eps", "BASIC_EPS"),
    "basic_eps": ("basic_eps", "eps", "BASIC_EPS"),
    "revenue": ("revenue", "OPERATE_INCOME", "operate_income"),
    "attributable_profit": (
        "attributable_profit",
        "HOLDER_PROFIT",
        "holder_profit",
        "net_profit",
    ),
    "close": ("close", "Close", "收盘"),
    "trade_date": ("trade_date", "date", "Date", "日期"),
    "volume": ("volume", "Volume", "成交量"),
    "adjustment": ("adjustment", "adjust", "ADJUST"),
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


def _resolve_required(
    pool: Dict[str, Any], required_outputs: Sequence[str]
) -> Tuple[Dict[str, Any], List[str]]:
    values: Dict[str, Any] = {}
    missing: List[str] = []
    for req in required_outputs:
        synonyms = REQUIRED_OUTPUT_SYNONYMS.get(str(req), (str(req),))
        found = None
        for syn in synonyms:
            if syn in pool and pool[syn] not in (None, ""):
                found = pool[syn]
                break
            for pk, pv in pool.items():
                if str(pk).lower() == str(syn).lower() and pv not in (None, ""):
                    found = pv
                    break
            if found is not None:
                break
        if found is None:
            missing.append(str(req))
        else:
            values[str(req)] = found
    return values, missing


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
    try:
        if isinstance(candidate, dict):
            envelope = normalize_gateway_envelope(candidate)
        else:
            normalization_error = "non_object_result"
    except Exception as exc:  # noqa: BLE001
        normalization_error = f"{type(exc).__name__}: {exc}"
        envelope = {}

    pool = extract_values_from_envelope(envelope, list(required_outputs))
    if isinstance(envelope, dict):
        pool.update(_collect_from_rows(envelope.get("rows")))
        for nested_key in ("normalized", "raw", "result"):
            nested = envelope.get(nested_key)
            if isinstance(nested, dict):
                pool.update(_collect_from_rows(nested.get("rows")))
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
    return {
        "values": values,
        "missing_outputs": missing,
        "envelope": envelope,
        "adapted_ok": bool(adapted.get("adapter_ok")) if isinstance(adapted, dict) else False,
        "normalization_error": normalization_error,
        "pool_keys": sorted(pool.keys()),
        "adapter_present_check": present_check,
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
    return best or {
        "values": {},
        "missing_outputs": list(required_outputs),
        "envelope": {},
        "adapted_ok": False,
        "normalization_error": "no_candidate",
        "pool_keys": [],
        "adapter_present_check": {"ok": False, "present": [], "missing": list(required_outputs)},
    }
