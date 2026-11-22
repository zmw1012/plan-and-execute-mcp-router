#!/usr/bin/env python3
"""Direct Real-MCP gold provenance audit + Dev15/Held-out20 writers.

No LLM / embedding / Agent / Operation-Visible runs.
Fail-closed: refuses mock FinancialServer fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.real_mcp.market_calc import (  # noqa: E402
    annualized_volatility,
    maximum_drawdown,
    moving_average,
    percentage_return,
    round_half_up,
    simple_return,
)
from hk_equity_mcp.real_mcp.market_gateway import (  # noqa: E402
    DEFAULT_MARKET_CATALOG,
    MarketRealMcpGateway,
    require_market_real_mcp_ready,
)
from hk_equity_mcp.real_mcp.market_normalize import (  # noqa: E402
    MARKET_CUTOFF,
    parse_session_date,
    to_ymd,
    to_yyyymmdd,
)

OUT_AUDIT = ROOT / "docs" / "market_real_mcp_provenance_audit_v1.md"
OUT_AUDIT_JSONL = ROOT / "outputs" / "market_real_mcp_audit_v1" / "provenance_records.jsonl"
OUT_DEV = ROOT / "benchmarks" / "questions" / "market_development_15_v1.jsonl"
OUT_HELD = ROOT / "benchmarks" / "heldout" / "market_test_20_v1.jsonl"


def _unset_proxies() -> None:
    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k, None)


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _bar_at(gw: MarketRealMcpGateway, symbol: str, day: str, adjust: str = "") -> Dict[str, Any]:
    ymd = to_yyyymmdd(parse_session_date(day))
    # Fetch a short window ending at day (or cutoff) so previous-valid works.
    end = min(parse_session_date(day), MARKET_CUTOFF)
    start = end - timedelta(days=21)
    res = gw.invoke(
        "aktools.stock_hk_hist",
        {
            "symbol": symbol,
            "start_date": to_yyyymmdd(start),
            "end_date": to_yyyymmdd(end),
            "period": "daily",
            "adjust": adjust,
            "target_date": to_ymd(end) if day else None,
        },
    )
    if not res.success:
        raise RuntimeError(f"hist failed {symbol} {day} adjust={adjust!r}: {res.error}")
    return {
        "result": res,
        "bar": res.normalized["data"],
        "effective": res.normalized["effective_session_date"],
        "prev_triggered": res.normalized.get("previous_valid_day_triggered"),
        "args": res.arguments,
        "raw_hash": _sha(res.raw),
        "norm_hash": _sha(res.normalized),
        "server_version": res.server_version,
    }


def _series(
    gw: MarketRealMcpGateway,
    symbol: str,
    start: str,
    end: str,
    adjust: str = "",
) -> Dict[str, Any]:
    res = gw.invoke(
        "aktools.stock_hk_hist",
        {
            "symbol": symbol,
            "start_date": to_yyyymmdd(parse_session_date(start)),
            "end_date": to_yyyymmdd(parse_session_date(end)),
            "period": "daily",
            "adjust": adjust,
        },
    )
    if not res.success:
        raise RuntimeError(f"hist series failed {symbol} {start}:{end}: {res.error}")
    bars = res.normalized["data"]
    if not isinstance(bars, list) or not bars:
        raise RuntimeError(f"empty series {symbol} {start}:{end}")
    return {
        "result": res,
        "bars": bars,
        "closes": [float(b["close"]) for b in bars],
        "dates": [b["date"] for b in bars],
        "raw_hash": _sha(res.raw),
        "norm_hash": _sha(res.normalized),
        "server_version": res.server_version,
        "args": {
            "symbol": symbol,
            "start_date": to_yyyymmdd(parse_session_date(start)),
            "end_date": to_yyyymmdd(parse_session_date(end)),
            "adjust": adjust,
        },
    }


def _hsi_at(gw: MarketRealMcpGateway, day: str) -> Dict[str, Any]:
    res = gw.invoke(
        "aktools.stock_hk_index_daily_sina",
        {"symbol": "HSI", "offset": -500, "max_rows": 500, "target_date": day},
    )
    if not res.success:
        raise RuntimeError(f"HSI failed {day}: {res.error}")
    return {
        "result": res,
        "bar": res.normalized["data"],
        "effective": res.normalized["effective_session_date"],
        "prev_triggered": res.normalized.get("previous_valid_day_triggered"),
        "raw_hash": _sha(res.raw),
        "norm_hash": _sha(res.normalized),
        "server_version": res.server_version,
        "args": {"symbol": "HSI", "offset": -500, "max_rows": 500, "target_date": day},
    }


def _hsi_series(gw: MarketRealMcpGateway, start: str, end: str) -> Dict[str, Any]:
    res = gw.invoke(
        "aktools.stock_hk_index_daily_sina",
        {"symbol": "HSI", "offset": -500, "max_rows": 500},
    )
    if not res.success:
        raise RuntimeError(f"HSI series failed: {res.error}")
    bars = res.normalized["data"]
    if not isinstance(bars, list):
        raise RuntimeError("HSI series not list")
    s = parse_session_date(start)
    e = parse_session_date(end)
    filtered = [b for b in bars if s <= parse_session_date(b["date"]) <= e]
    if not filtered:
        raise RuntimeError(f"HSI empty window {start}:{end}")
    return {
        "result": res,
        "bars": filtered,
        "closes": [float(b["close"]) for b in filtered],
        "dates": [b["date"] for b in filtered],
        "raw_hash": _sha(res.raw),
        "norm_hash": _sha({"filtered": filtered, "full_hash": _sha(res.normalized)}),
        "server_version": res.server_version,
        "args": {"symbol": "HSI", "offset": -500, "max_rows": 500, "window": [start, end]},
    }


def _audit_rec(
    *,
    qid: str,
    operation: str,
    provider: str,
    args: Dict[str, Any],
    requested_date: Any,
    effective: Any,
    prev_triggered: Any,
    adjust: Any,
    value: Any,
    unit: str,
    formula: str,
    inputs: Any,
    rounded: Any,
    tolerance: Any,
    raw_hash: str,
    server_version: Any,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "question_id": qid,
        "mcp_server": "aktools",
        "package_command": os.environ.get(
            "AKTOOLS_MCP_COMMAND",
            str(ROOT / ".venv-aktools-discovery" / "bin" / "aktools-mcp"),
        ),
        "server_version": server_version,
        "provider": provider,
        "source_operation": operation,
        "raw_invocation_arguments": args,
        "requested_date_or_range": requested_date,
        "effective_session_date": effective,
        "previous_valid_day_triggered": prev_triggered,
        "adjustment_mode": adjust,
        "value": value,
        "unit": unit,
        "calculation_inputs": inputs,
        "calculation_formula": formula,
        "rounded_gold": rounded,
        "tolerance": tolerance,
        "raw_response_hash": raw_hash,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "mock_used": False,
        "llm_used": False,
    }


def _base_q(
    *,
    qid: str,
    company: str,
    ticker: str,
    etom: str,
    question: str,
    split: str,
    role: str,
    locked: bool,
    gold_ops: List[str],
    answerability: str,
    gold_status: str,
    reference: str,
    gold_values: List[Dict[str, Any]],
    task_type: str,
    capabilities: List[str],
    risks: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "id": qid,
        "question_id": qid,
        "domain": "market",
        "company": company,
        "ticker": ticker,
        "etom_level": etom,
        "task_type": task_type,
        "taxonomy": f"ETOM {etom} Market Real-MCP",
        "question": question,
        "dataset_split": split,
        "benchmark_role": role,
        "locked": locked,
        "frozen": locked,
        "verification_status": "verified" if gold_status == "answered" else "confirmed_infeasible",
        "formal_test_eligible": gold_status == "answered",
        "answerability_class": answerability,
        "gold_status": gold_status,
        "reference_answer": reference,
        "gold_values": gold_values,
        "gold_server_ids": ["aktools"],
        "gold_operations": gold_ops,
        "required_capabilities": capabilities,
        "known_data_risks": risks,
        "timezone": "Asia/Hong_Kong",
        "dataset_cutoff": "2026-08-12",
        "currency_provenance_default": "exchange_convention",
        "exclusion_reason": None if gold_status == "answered" else "infeasible_under_frozen_market_catalog",
    }
    if extra:
        row.update(extra)
    return row


def build_benchmarks(gw: MarketRealMcpGateway) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    audits: List[Dict[str, Any]] = []
    dev: List[Dict[str, Any]] = []
    held: List[Dict[str, Any]] = []

    # -------- Development L1 --------
    t_close = _bar_at(gw, "00700", "2026-07-31", "")
    v = float(t_close["bar"]["close"])
    vr = round_half_up(v, 2)
    audits.append(
        _audit_rec(
            qid="mkt_dev_001",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=t_close["args"],
            requested_date="2026-07-31",
            effective=t_close["effective"],
            prev_triggered=t_close["prev_triggered"],
            adjust="",
            value=v,
            unit="HKD",
            formula="unadjusted close field",
            inputs={"close": v},
            rounded=vr,
            tolerance={"abs": 0.05},
            raw_hash=t_close["raw_hash"],
            server_version=t_close["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_001",
            company="Tencent",
            ticker="0700.HK",
            etom="L1",
            question=(
                "What was Tencent (0700.HK) unadjusted closing price on 2026-07-31 "
                "(Asia/Hong_Kong)? Answer in HKD to 2 decimals. Use adjust=\"\" "
                "(unadjusted). Rounding: half-up to 2 decimals; tolerance ±0.05."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"HKD {vr:.2f} unadjusted close on {t_close['effective']}.",
            gold_values=[
                {
                    "value": vr,
                    "unit": "HKD",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "representation": "decimal_currency",
                    "rounding": "half_up_2dp",
                    "tolerance_abs": 0.05,
                    "requested_date": "2026-07-31",
                    "effective_session_date": t_close["effective"],
                    "currency_provenance": "exchange_convention",
                }
            ],
            task_type="direct_retrieval",
            capabilities=["fixed_date_close", "unadjusted"],
            risks=["HKD inferred by exchange convention"],
            extra={"price_field": "close", "adjustment_mode": ""},
        )
    )

    h_vol = _bar_at(gw, "00005", "2026-06-30", "")
    vol = float(h_vol["bar"]["volume"])
    audits.append(
        _audit_rec(
            qid="mkt_dev_002",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=h_vol["args"],
            requested_date="2026-06-30",
            effective=h_vol["effective"],
            prev_triggered=h_vol["prev_triggered"],
            adjust="",
            value=vol,
            unit="shares",
            formula="volume field",
            inputs={"volume": vol},
            rounded=vol,
            tolerance={"rel": 0.001},
            raw_hash=h_vol["raw_hash"],
            server_version=h_vol["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_002",
            company="HSBC",
            ticker="0005.HK",
            etom="L1",
            question=(
                "What was HSBC (0005.HK) trading volume on 2026-06-30? Use unadjusted "
                "daily history (adjust=\"\"). Report the raw volume field; relative "
                "tolerance 0.1%."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"Volume {vol} on {h_vol['effective']}.",
            gold_values=[
                {
                    "value": vol,
                    "unit": "shares",
                    "price_field": "volume",
                    "adjustment_mode": "",
                    "tolerance_rel": 0.001,
                    "requested_date": "2026-06-30",
                    "effective_session_date": h_vol["effective"],
                }
            ],
            task_type="direct_retrieval",
            capabilities=["volume"],
            risks=[],
            extra={"price_field": "volume", "adjustment_mode": ""},
        )
    )

    a_ohlc = _bar_at(gw, "09988", "2026-05-15", "")
    ohlc = {
        "open": round_half_up(float(a_ohlc["bar"]["open"]), 2),
        "high": round_half_up(float(a_ohlc["bar"]["high"]), 2),
        "low": round_half_up(float(a_ohlc["bar"]["low"]), 2),
        "close": round_half_up(float(a_ohlc["bar"]["close"]), 2),
    }
    audits.append(
        _audit_rec(
            qid="mkt_dev_003",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=a_ohlc["args"],
            requested_date="2026-05-15",
            effective=a_ohlc["effective"],
            prev_triggered=a_ohlc["prev_triggered"],
            adjust="",
            value=ohlc,
            unit="HKD",
            formula="OHLC fields",
            inputs=a_ohlc["bar"],
            rounded=ohlc,
            tolerance={"abs": 0.05},
            raw_hash=a_ohlc["raw_hash"],
            server_version=a_ohlc["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_003",
            company="Alibaba",
            ticker="9988.HK",
            etom="L1",
            question=(
                "For Alibaba (9988.HK) on 2026-05-15, report unadjusted open, high, low, "
                "and close in HKD (adjust=\"\", half-up 2dp, tolerance ±0.05 each)."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=(
                f"O={ohlc['open']:.2f}, H={ohlc['high']:.2f}, "
                f"L={ohlc['low']:.2f}, C={ohlc['close']:.2f} on {a_ohlc['effective']}."
            ),
            gold_values=[
                {
                    "value": ohlc,
                    "unit": "HKD",
                    "price_field": "ohlc",
                    "adjustment_mode": "",
                    "rounding": "half_up_2dp",
                    "tolerance_abs": 0.05,
                    "requested_date": "2026-05-15",
                    "effective_session_date": a_ohlc["effective"],
                    "currency_provenance": "exchange_convention",
                }
            ],
            task_type="direct_retrieval",
            capabilities=["ohlc"],
            risks=[],
            extra={"price_field": "ohlc", "adjustment_mode": ""},
        )
    )

    hsi = _hsi_at(gw, "2026-03-18")
    hsi_c = round_half_up(float(hsi["bar"]["close"]), 2)
    audits.append(
        _audit_rec(
            qid="mkt_dev_004",
            operation="aktools.stock_hk_index_daily_sina",
            provider="sina_via_akshare",
            args=hsi["args"],
            requested_date="2026-03-18",
            effective=hsi["effective"],
            prev_triggered=hsi["prev_triggered"],
            adjust=None,
            value=float(hsi["bar"]["close"]),
            unit="index_points",
            formula="HSI close after cutoff post-filter",
            inputs=hsi["bar"],
            rounded=hsi_c,
            tolerance={"abs": 0.5},
            raw_hash=hsi["raw_hash"],
            server_version=hsi["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_004",
            company="HSI",
            ticker="HSI",
            etom="L1",
            question=(
                "What was the Hang Seng Index (HSI) closing level on 2026-03-18? "
                "Use stock_hk_index_daily_sina with cutoff post-filter <=2026-08-12. "
                "Unit: index points; half-up 2dp; tolerance ±0.5. Not an equity HKD price."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_index_daily_sina"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"HSI {hsi_c:.2f} on {hsi['effective']}.",
            gold_values=[
                {
                    "value": hsi_c,
                    "unit": "index_points",
                    "price_field": "close",
                    "adjustment_mode": None,
                    "rounding": "half_up_2dp",
                    "tolerance_abs": 0.5,
                    "requested_date": "2026-03-18",
                    "effective_session_date": hsi["effective"],
                }
            ],
            task_type="direct_retrieval",
            capabilities=["hsi_level"],
            risks=["Sina HSI may differ slightly from other feeds"],
            extra={"price_field": "close"},
        )
    )

    # L2
    weekend = _bar_at(gw, "00700", "2026-08-09", "")
    wclose = round_half_up(float(weekend["bar"]["close"]), 2)
    audits.append(
        _audit_rec(
            qid="mkt_dev_005",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=weekend["args"],
            requested_date="2026-08-09",
            effective=weekend["effective"],
            prev_triggered=weekend["prev_triggered"],
            adjust="",
            value=float(weekend["bar"]["close"]),
            unit="HKD",
            formula="previous_valid_day max(session<=requested)",
            inputs=weekend["bar"],
            rounded=wclose,
            tolerance={"abs": 0.05},
            raw_hash=weekend["raw_hash"],
            server_version=weekend["server_version"],
            notes="Sunday request → prior session",
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_005",
            company="Tencent",
            ticker="0700.HK",
            etom="L2",
            question=(
                "Using previous-valid-day (max session date <= requested, no forward-fill), "
                "what was Tencent's unadjusted close for calendar date 2026-08-09 "
                "(Sunday)? Report HKD half-up 2dp, tolerance ±0.05, and the effective session date."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=(
                f"Effective {weekend['effective']}, close HKD {wclose:.2f} "
                f"(previous_valid_day_triggered={weekend['prev_triggered']})."
            ),
            gold_values=[
                {
                    "value": wclose,
                    "unit": "HKD",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "tolerance_abs": 0.05,
                    "requested_date": "2026-08-09",
                    "effective_session_date": weekend["effective"],
                    "previous_valid_day_triggered": weekend["prev_triggered"],
                }
            ],
            task_type="session_alignment",
            capabilities=["previous_valid_day"],
            risks=[],
            extra={"adjustment_mode": ""},
        )
    )

    hsbc_to = _bar_at(gw, "00005", "2026-07-15", "")
    turnover = float(hsbc_to["bar"]["turnover"]) if hsbc_to["bar"].get("turnover") is not None else None
    if turnover is None:
        raise RuntimeError("HSBC turnover missing on 2026-07-15")
    audits.append(
        _audit_rec(
            qid="mkt_dev_006",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=hsbc_to["args"],
            requested_date="2026-07-15",
            effective=hsbc_to["effective"],
            prev_triggered=hsbc_to["prev_triggered"],
            adjust="",
            value=turnover,
            unit="HKD",
            formula="turnover/成交额 field",
            inputs={"turnover": turnover},
            rounded=turnover,
            tolerance={"rel": 0.001},
            raw_hash=hsbc_to["raw_hash"],
            server_version=hsbc_to["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_006",
            company="HSBC",
            ticker="0005.HK",
            etom="L2",
            question=(
                "What was HSBC (0005.HK) turnover (成交额) on 2026-07-15 from unadjusted "
                "daily history? Relative tolerance 0.1%. Currency provenance is exchange_convention."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"Turnover {turnover} on {hsbc_to['effective']}.",
            gold_values=[
                {
                    "value": turnover,
                    "unit": "HKD",
                    "price_field": "turnover",
                    "adjustment_mode": "",
                    "tolerance_rel": 0.001,
                    "requested_date": "2026-07-15",
                    "effective_session_date": hsbc_to["effective"],
                    "currency_provenance": "exchange_convention",
                }
            ],
            task_type="direct_retrieval",
            capabilities=["turnover"],
            risks=[],
            extra={"price_field": "turnover"},
        )
    )

    a_u = _bar_at(gw, "09988", "2026-04-30", "")
    a_q = _bar_at(gw, "09988", "2026-04-30", "qfq")
    u_c = round_half_up(float(a_u["bar"]["close"]), 2)
    q_c = round_half_up(float(a_q["bar"]["close"]), 2)
    audits.append(
        _audit_rec(
            qid="mkt_dev_007",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args={"unadj": a_u["args"], "qfq": a_q["args"]},
            requested_date="2026-04-30",
            effective=a_u["effective"],
            prev_triggered=a_u["prev_triggered"],
            adjust="compare '' vs qfq",
            value={"unadjusted": u_c, "qfq": q_c},
            unit="HKD",
            formula="two explicit adjust modes; no auto-mix",
            inputs={"unadj_close": a_u["bar"]["close"], "qfq_close": a_q["bar"]["close"]},
            rounded={"unadjusted": u_c, "qfq": q_c},
            tolerance={"abs": 0.05},
            raw_hash=_sha([a_u["raw_hash"], a_q["raw_hash"]]),
            server_version=a_u["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_007",
            company="Alibaba",
            ticker="9988.HK",
            etom="L2",
            question=(
                "On 2026-04-30, what were Alibaba (9988.HK) closes under adjust=\"\" and "
                "adjust=\"qfq\"? Report both HKD values (half-up 2dp, ± ±0.05). "
                "Do not mix modes."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"unadjusted={u_c:.2f}, qfq={q_c:.2f} on {a_u['effective']}.",
            gold_values=[
                {
                    "value": {"unadjusted": u_c, "qfq": q_c},
                    "unit": "HKD",
                    "price_field": "close",
                    "adjustment_mode": ["", "qfq"],
                    "tolerance_abs": 0.05,
                    "requested_date": "2026-04-30",
                    "effective_session_date": a_u["effective"],
                }
            ],
            task_type="adjustment_compare",
            capabilities=["adjusted_vs_unadjusted"],
            risks=[],
            extra={"adjustment_mode": ["", "qfq"]},
        )
    )

    # L3
    t_ret = _series(gw, "00700", "2026-07-02", "2026-07-31", "")
    c0, c1 = t_ret["closes"][0], t_ret["closes"][-1]
    pret = percentage_return(c0, c1)
    pret_r = round_half_up(pret, 4)
    audits.append(
        _audit_rec(
            qid="mkt_dev_008",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=t_ret["args"],
            requested_date={"start": t_ret["dates"][0], "end": t_ret["dates"][-1]},
            effective=t_ret["dates"][-1],
            prev_triggered=False,
            adjust="",
            value=pret,
            unit="percent",
            formula="100*(close_end/close_start-1)",
            inputs={"close_start": c0, "close_end": c1, "dates": [t_ret["dates"][0], t_ret["dates"][-1]]},
            rounded=pret_r,
            tolerance={"abs": 0.01},
            raw_hash=t_ret["raw_hash"],
            server_version=t_ret["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_008",
            company="Tencent",
            ticker="0700.HK",
            etom="L3",
            question=(
                "Using unadjusted closes, compute Tencent percentage return from the first "
                "available session on/after 2026-07-02 through the last session on/before "
                "2026-07-31: 100*(close_end/close_start-1). Report percent with half-up 4dp; "
                "tolerance ±0.01 percentage points."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=(
                f"{pret_r}% using close_start={c0} ({t_ret['dates'][0]}) "
                f"close_end={c1} ({t_ret['dates'][-1]})."
            ),
            gold_values=[
                {
                    "value": pret_r,
                    "unit": "percent",
                    "representation": "percentage_not_decimal",
                    "formula": "100*(close_end/close_start-1)",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "rounding": "half_up_4dp",
                    "tolerance_abs": 0.01,
                    "date_window": [t_ret["dates"][0], t_ret["dates"][-1]],
                }
            ],
            task_type="calculation",
            capabilities=["simple_return", "percentage_return"],
            risks=[],
            extra={"formula": "100*(close_end/close_start-1)", "adjustment_mode": ""},
        )
    )

    hsbc_ma = _series(gw, "00005", "2026-05-20", "2026-06-30", "")
    if len(hsbc_ma["closes"]) < 20:
        raise RuntimeError("need 20 sessions for MA20")
    # MA20 including target 2026-06-30: last 20 closes in series ending that day
    # Ensure last date is <= 2026-06-30 and use last 20
    ma20 = moving_average(hsbc_ma["closes"], 20)
    ma20_r = round_half_up(ma20, 4)
    audits.append(
        _audit_rec(
            qid="mkt_dev_009",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=hsbc_ma["args"],
            requested_date="2026-06-30",
            effective=hsbc_ma["dates"][-1],
            prev_triggered=False,
            adjust="",
            value=ma20,
            unit="HKD",
            formula="mean of last 20 unadjusted closes including target session",
            inputs={"closes": hsbc_ma["closes"][-20:], "dates": hsbc_ma["dates"][-20:]},
            rounded=ma20_r,
            tolerance={"abs": 0.05},
            raw_hash=hsbc_ma["raw_hash"],
            server_version=hsbc_ma["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_009",
            company="HSBC",
            ticker="0005.HK",
            etom="L3",
            question=(
                "For HSBC on target session 2026-06-30, compute MA20 as the arithmetic mean "
                "of the most recent 20 valid unadjusted closes including that session. "
                "HKD half-up 4dp; tolerance ±0.05."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"MA20={ma20_r} ending {hsbc_ma['dates'][-1]}.",
            gold_values=[
                {
                    "value": ma20_r,
                    "unit": "HKD",
                    "formula": "mean(last_20_unadjusted_closes_incl_target)",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "rounding": "half_up_4dp",
                    "tolerance_abs": 0.05,
                    "date_window": [hsbc_ma["dates"][-20], hsbc_ma["dates"][-1]],
                }
            ],
            task_type="calculation",
            capabilities=["moving_average"],
            risks=[],
            extra={"formula": "MA20", "adjustment_mode": ""},
        )
    )

    hsi_d0 = _hsi_at(gw, "2026-08-11")
    hsi_d1 = _hsi_at(gw, "2026-08-12")
    dr = simple_return(float(hsi_d0["bar"]["close"]), float(hsi_d1["bar"]["close"]))
    dr_r = round_half_up(dr, 6)
    audits.append(
        _audit_rec(
            qid="mkt_dev_010",
            operation="aktools.stock_hk_index_daily_sina",
            provider="sina_via_akshare",
            args={"d0": hsi_d0["args"], "d1": hsi_d1["args"]},
            requested_date=["2026-08-11", "2026-08-12"],
            effective=hsi_d1["effective"],
            prev_triggered=False,
            adjust=None,
            value=dr,
            unit="decimal_return",
            formula="P_t/P_(t-1)-1",
            inputs={
                "P_t-1": hsi_d0["bar"]["close"],
                "P_t": hsi_d1["bar"]["close"],
                "dates": [hsi_d0["effective"], hsi_d1["effective"]],
            },
            rounded=dr_r,
            tolerance={"abs": 1e-5},
            raw_hash=_sha([hsi_d0["raw_hash"], hsi_d1["raw_hash"]]),
            server_version=hsi_d1["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_010",
            company="HSI",
            ticker="HSI",
            etom="L3",
            question=(
                "Compute HSI daily simple return from 2026-08-11 to 2026-08-12 as "
                "P_t/P_(t-1)-1 (decimal, not percent). Half-up 6dp; tolerance ±1e-5."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_index_daily_sina"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"daily_return={dr_r} (decimal).",
            gold_values=[
                {
                    "value": dr_r,
                    "unit": "decimal_return",
                    "representation": "decimal_not_percent",
                    "formula": "P_t/P_(t-1)-1",
                    "price_field": "close",
                    "rounding": "half_up_6dp",
                    "tolerance_abs": 1e-5,
                    "date_window": [hsi_d0["effective"], hsi_d1["effective"]],
                }
            ],
            task_type="calculation",
            capabilities=["daily_return", "hsi_level"],
            risks=[],
            extra={"formula": "P_t/P_(t-1)-1"},
        )
    )

    # L4
    t_vol = _series(gw, "00700", "2026-07-01", "2026-07-31", "")
    avol = annualized_volatility(t_vol["closes"])
    avol_r = round_half_up(avol, 6)
    audits.append(
        _audit_rec(
            qid="mkt_dev_011",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=t_vol["args"],
            requested_date=[t_vol["dates"][0], t_vol["dates"][-1]],
            effective=t_vol["dates"][-1],
            prev_triggered=False,
            adjust="",
            value=avol,
            unit="decimal_volatility",
            formula="sample_std(daily_simple_returns,ddof=1)*sqrt(252)",
            inputs={"closes": t_vol["closes"], "dates": t_vol["dates"]},
            rounded=avol_r,
            tolerance={"abs": 1e-4},
            raw_hash=t_vol["raw_hash"],
            server_version=t_vol["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_011",
            company="Tencent",
            ticker="0700.HK",
            etom="L4",
            question=(
                "Using unadjusted Tencent closes from 2026-07-01 through 2026-07-31, "
                "compute annualized volatility = sample_std(daily simple returns, ddof=1) "
                "* sqrt(252). Report as decimal (not percent), half-up 6dp, tolerance ±1e-4."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"ann_vol={avol_r} over {t_vol['dates'][0]}..{t_vol['dates'][-1]}.",
            gold_values=[
                {
                    "value": avol_r,
                    "unit": "decimal_volatility",
                    "representation": "decimal_not_percent",
                    "formula": "sample_std(daily_simple_returns,ddof=1)*sqrt(252)",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "rounding": "half_up_6dp",
                    "tolerance_abs": 1e-4,
                    "date_window": [t_vol["dates"][0], t_vol["dates"][-1]],
                }
            ],
            task_type="calculation",
            capabilities=["volatility"],
            risks=[],
            extra={"formula": "ann_vol", "adjustment_mode": ""},
        )
    )

    a_mdd = _series(gw, "09988", "2026-06-02", "2026-07-31", "")
    mdd = maximum_drawdown(a_mdd["closes"])
    mdd_r = round_half_up(mdd, 6)
    audits.append(
        _audit_rec(
            qid="mkt_dev_012",
            operation="aktools.stock_hk_hist",
            provider="eastmoney_via_akshare",
            args=a_mdd["args"],
            requested_date=[a_mdd["dates"][0], a_mdd["dates"][-1]],
            effective=a_mdd["dates"][-1],
            prev_triggered=False,
            adjust="",
            value=mdd,
            unit="decimal_drawdown",
            formula="min(P_t/running_max(P_1..P_t)-1)",
            inputs={"closes": a_mdd["closes"], "dates": a_mdd["dates"]},
            rounded=mdd_r,
            tolerance={"abs": 1e-4},
            raw_hash=a_mdd["raw_hash"],
            server_version=a_mdd["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_012",
            company="Alibaba",
            ticker="9988.HK",
            etom="L4",
            question=(
                "For Alibaba unadjusted closes from 2026-06-02 through 2026-07-31, compute "
                "maximum drawdown = min(P_t/running_max(P_1..P_t)-1) as a decimal "
                "(negative or zero). Half-up 6dp; tolerance ±1e-4."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"MDD={mdd_r} over {a_mdd['dates'][0]}..{a_mdd['dates'][-1]}.",
            gold_values=[
                {
                    "value": mdd_r,
                    "unit": "decimal_drawdown",
                    "representation": "decimal_not_percent",
                    "formula": "min(P_t/running_max-1)",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "rounding": "half_up_6dp",
                    "tolerance_abs": 1e-4,
                    "date_window": [a_mdd["dates"][0], a_mdd["dates"][-1]],
                }
            ],
            task_type="calculation",
            capabilities=["maximum_drawdown"],
            risks=[],
            extra={"formula": "MDD", "adjustment_mode": ""},
        )
    )

    eq = _series(gw, "00700", "2026-05-04", "2026-05-29", "")
    ix = _hsi_series(gw, eq["dates"][0], eq["dates"][-1])
    # Align by intersecting dates
    eq_map = dict(zip(eq["dates"], eq["closes"]))
    ix_map = dict(zip(ix["dates"], ix["closes"]))
    common = sorted(set(eq_map) & set(ix_map))
    if len(common) < 2:
        raise RuntimeError("insufficient overlap Tencent/HSI May window")
    d0, d1 = common[0], common[-1]
    eq_ret = simple_return(eq_map[d0], eq_map[d1])
    ix_ret = simple_return(ix_map[d0], ix_map[d1])
    rel = eq_ret - ix_ret
    rel_r = round_half_up(rel, 6)
    audits.append(
        _audit_rec(
            qid="mkt_dev_013",
            operation="aktools.stock_hk_hist+aktools.stock_hk_index_daily_sina",
            provider="eastmoney+sina",
            args={"equity": eq["args"], "index": ix["args"], "common": [d0, d1]},
            requested_date=[d0, d1],
            effective=d1,
            prev_triggered=False,
            adjust="",
            value=rel,
            unit="decimal_excess_return",
            formula="(P_e_end/P_e_start-1)-(P_i_end/P_i_start-1)",
            inputs={
                "equity": [eq_map[d0], eq_map[d1]],
                "hsi": [ix_map[d0], ix_map[d1]],
                "dates": [d0, d1],
            },
            rounded=rel_r,
            tolerance={"abs": 1e-4},
            raw_hash=_sha([eq["raw_hash"], ix["raw_hash"]]),
            server_version=eq["server_version"],
        )
    )
    dev.append(
        _base_q(
            qid="mkt_dev_013",
            company="Tencent",
            ticker="0700.HK",
            etom="L4",
            question=(
                "Over the overlapping session window spanning Tencent and HSI from early to "
                "late May 2026 (first/last common dates in 2026-05-04..2026-05-29), compute "
                "relative performance as equity simple return minus HSI simple return "
                "(decimals, not percent). Half-up 6dp; tolerance ±1e-4. Equity adjust=\"\"."
            ),
            split="development",
            role="method_development_and_error_analysis",
            locked=False,
            gold_ops=["aktools.stock_hk_hist", "aktools.stock_hk_index_daily_sina"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"relative={rel_r} on common window {d0}..{d1}.",
            gold_values=[
                {
                    "value": rel_r,
                    "unit": "decimal_excess_return",
                    "representation": "decimal_not_percent",
                    "formula": "simple_return(equity)-simple_return(HSI)",
                    "price_field": "close",
                    "adjustment_mode": "",
                    "rounding": "half_up_6dp",
                    "tolerance_abs": 1e-4,
                    "date_window": [d0, d1],
                }
            ],
            task_type="calculation",
            capabilities=["relative_performance", "equity_vs_hsi"],
            risks=["date alignment uses intersection of sessions"],
            extra={"formula": "excess_simple_return", "adjustment_mode": ""},
        )
    )

    # L5
    for qid, company, ticker, question, reason in [
        (
            "mkt_dev_014",
            "Tencent",
            "0700.HK",
            "What is Tencent's latest / current / today's closing price?",
            "Uses latest/current/today; forbidden as formal numeric gold under fixed cutoff.",
        ),
        (
            "mkt_dev_015",
            "Alibaba",
            "9988.HK",
            "Using FTShare hk_hist or stock_hk_hist_min_em, give Alibaba's 1-minute bar gold for 2026-08-12.",
            "FTShare blocked; minute hist deferred (empty-window). Not in active frozen catalog.",
        ),
    ]:
        audits.append(
            _audit_rec(
                qid=qid,
                operation="none",
                provider="n/a",
                args={},
                requested_date=None,
                effective=None,
                prev_triggered=None,
                adjust=None,
                value=None,
                unit="n/a",
                formula="infeasible",
                inputs={},
                rounded=None,
                tolerance=None,
                raw_hash="",
                server_version=None,
                notes=reason,
            )
        )
        dev.append(
            _base_q(
                qid=qid,
                company=company,
                ticker=ticker,
                etom="L5",
                question=question,
                split="development",
                role="method_development_and_error_analysis",
                locked=False,
                gold_ops=[],
                answerability="should_refuse_or_clarify",
                gold_status="unanswerable",
                reference="Refuse or clarify; no numeric gold under frozen Market Real-MCP v1.",
                gold_values=[],
                task_type="refusal_clarification",
                capabilities=["infeasible_under_catalog"],
                risks=[reason],
                extra={"confirmed_infeasible": True, "infeasibility_reason": reason},
            )
        )

    # -------- Held-out (distinct dates/windows/tasks) --------
    def add_held(row: Dict[str, Any], audit: Dict[str, Any]) -> None:
        row["dataset_split"] = "heldout"
        row["benchmark_role"] = "final_evaluation_only"
        row["locked"] = True
        row["frozen"] = True
        held.append(row)
        audits.append(audit)

    # L1 held
    b = _bar_at(gw, "00005", "2026-02-27", "")
    v = round_half_up(float(b["bar"]["close"]), 2)
    add_held(
        _base_q(
            qid="mkt_ht_001",
            company="HSBC",
            ticker="0005.HK",
            etom="L1",
            question=(
                "What was HSBC (0005.HK) unadjusted close on 2026-02-27 in HKD? "
                "adjust=\"\"; half-up 2dp; tolerance ±0.05."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"HKD {v:.2f} on {b['effective']}.",
            gold_values=[{
                "value": v, "unit": "HKD", "price_field": "close", "adjustment_mode": "",
                "rounding": "half_up_2dp", "tolerance_abs": 0.05,
                "requested_date": "2026-02-27", "effective_session_date": b["effective"],
                "currency_provenance": "exchange_convention",
            }],
            task_type="direct_retrieval",
            capabilities=["fixed_date_close"],
            risks=[],
            extra={"adjustment_mode": ""},
        ),
        _audit_rec(
            qid="mkt_ht_001", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=b["args"], requested_date="2026-02-27", effective=b["effective"],
            prev_triggered=b["prev_triggered"], adjust="", value=float(b["bar"]["close"]),
            unit="HKD", formula="close", inputs=b["bar"], rounded=v, tolerance={"abs": 0.05},
            raw_hash=b["raw_hash"], server_version=b["server_version"],
        ),
    )

    b = _bar_at(gw, "09988", "2026-01-15", "")
    vol = float(b["bar"]["volume"])
    add_held(
        _base_q(
            qid="mkt_ht_002",
            company="Alibaba",
            ticker="9988.HK",
            etom="L1",
            question=(
                "What was Alibaba (9988.HK) volume on 2026-01-15 from unadjusted daily history? "
                "Relative tolerance 0.1%."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"Volume {vol} on {b['effective']}.",
            gold_values=[{
                "value": vol, "unit": "shares", "price_field": "volume", "adjustment_mode": "",
                "tolerance_rel": 0.001, "requested_date": "2026-01-15",
                "effective_session_date": b["effective"],
            }],
            task_type="direct_retrieval",
            capabilities=["volume"],
            risks=[],
            extra={"price_field": "volume"},
        ),
        _audit_rec(
            qid="mkt_ht_002", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=b["args"], requested_date="2026-01-15", effective=b["effective"],
            prev_triggered=b["prev_triggered"], adjust="", value=vol, unit="shares",
            formula="volume", inputs={"volume": vol}, rounded=vol, tolerance={"rel": 0.001},
            raw_hash=b["raw_hash"], server_version=b["server_version"],
        ),
    )

    h = _hsi_at(gw, "2026-07-15")
    hv = round_half_up(float(h["bar"]["close"]), 2)
    add_held(
        _base_q(
            qid="mkt_ht_003",
            company="HSI",
            ticker="HSI",
            etom="L1",
            question=(
                "What was HSI close on 2026-07-15 (index points)? Use sina daily + cutoff "
                "post-filter; half-up 2dp; tolerance ±0.5."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_index_daily_sina"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"HSI {hv:.2f} on {h['effective']}.",
            gold_values=[{
                "value": hv, "unit": "index_points", "price_field": "close",
                "tolerance_abs": 0.5, "requested_date": "2026-07-15",
                "effective_session_date": h["effective"],
            }],
            task_type="direct_retrieval",
            capabilities=["hsi_level"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_003", operation="aktools.stock_hk_index_daily_sina",
            provider="sina_via_akshare", args=h["args"], requested_date="2026-07-15",
            effective=h["effective"], prev_triggered=h["prev_triggered"], adjust=None,
            value=float(h["bar"]["close"]), unit="index_points", formula="close",
            inputs=h["bar"], rounded=hv, tolerance={"abs": 0.5},
            raw_hash=h["raw_hash"], server_version=h["server_version"],
        ),
    )

    b = _bar_at(gw, "00700", "2026-04-02", "")
    ohlc = {
        "open": round_half_up(float(b["bar"]["open"]), 2),
        "high": round_half_up(float(b["bar"]["high"]), 2),
        "low": round_half_up(float(b["bar"]["low"]), 2),
        "close": round_half_up(float(b["bar"]["close"]), 2),
    }
    add_held(
        _base_q(
            qid="mkt_ht_004",
            company="Tencent",
            ticker="0700.HK",
            etom="L1",
            question=(
                "Report Tencent (0700.HK) unadjusted OHLC on 2026-04-02 in HKD "
                "(adjust=\"\", half-up 2dp, ± ±0.05)."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=(
                f"O={ohlc['open']:.2f}, H={ohlc['high']:.2f}, "
                f"L={ohlc['low']:.2f}, C={ohlc['close']:.2f} on {b['effective']}."
            ),
            gold_values=[{
                "value": ohlc, "unit": "HKD", "price_field": "ohlc", "adjustment_mode": "",
                "tolerance_abs": 0.05, "requested_date": "2026-04-02",
                "effective_session_date": b["effective"],
            }],
            task_type="direct_retrieval",
            capabilities=["ohlc"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_004", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=b["args"], requested_date="2026-04-02", effective=b["effective"],
            prev_triggered=b["prev_triggered"], adjust="", value=ohlc, unit="HKD",
            formula="ohlc", inputs=b["bar"], rounded=ohlc, tolerance={"abs": 0.05},
            raw_hash=b["raw_hash"], server_version=b["server_version"],
        ),
    )

    # L2 held
    b = _bar_at(gw, "00005", "2026-05-10", "")  # Sunday
    v = round_half_up(float(b["bar"]["close"]), 2)
    add_held(
        _base_q(
            qid="mkt_ht_005",
            company="HSBC",
            ticker="0005.HK",
            etom="L2",
            question=(
                "For calendar date 2026-05-10 (Sunday), apply previous-valid-day and report "
                "HSBC unadjusted close and effective session date."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"Effective {b['effective']}, close {v:.2f}.",
            gold_values=[{
                "value": v, "unit": "HKD", "price_field": "close", "adjustment_mode": "",
                "tolerance_abs": 0.05, "requested_date": "2026-05-10",
                "effective_session_date": b["effective"],
                "previous_valid_day_triggered": b["prev_triggered"],
            }],
            task_type="session_alignment",
            capabilities=["previous_valid_day"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_005", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=b["args"], requested_date="2026-05-10", effective=b["effective"],
            prev_triggered=b["prev_triggered"], adjust="", value=float(b["bar"]["close"]),
            unit="HKD", formula="previous_valid_day", inputs=b["bar"], rounded=v,
            tolerance={"abs": 0.05}, raw_hash=b["raw_hash"], server_version=b["server_version"],
        ),
    )

    u = _bar_at(gw, "00700", "2026-03-31", "")
    q = _bar_at(gw, "00700", "2026-03-31", "hfq")
    uv = round_half_up(float(u["bar"]["close"]), 2)
    qv = round_half_up(float(q["bar"]["close"]), 2)
    add_held(
        _base_q(
            qid="mkt_ht_006",
            company="Tencent",
            ticker="0700.HK",
            etom="L2",
            question=(
                "On 2026-03-31, compare Tencent closes under adjust=\"\" vs adjust=\"hfq\". "
                "Report both; do not mix modes."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"unadjusted={uv:.2f}, hfq={qv:.2f}.",
            gold_values=[{
                "value": {"unadjusted": uv, "hfq": qv}, "unit": "HKD", "price_field": "close",
                "adjustment_mode": ["", "hfq"], "tolerance_abs": 0.05,
                "requested_date": "2026-03-31", "effective_session_date": u["effective"],
            }],
            task_type="adjustment_compare",
            capabilities=["adjusted_vs_unadjusted"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_006", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args={"u": u["args"], "hfq": q["args"]}, requested_date="2026-03-31",
            effective=u["effective"], prev_triggered=u["prev_triggered"],
            adjust="'' vs hfq", value={"unadjusted": uv, "hfq": qv}, unit="HKD",
            formula="explicit dual adjust", inputs={"u": u["bar"]["close"], "hfq": q["bar"]["close"]},
            rounded={"unadjusted": uv, "hfq": qv}, tolerance={"abs": 0.05},
            raw_hash=_sha([u["raw_hash"], q["raw_hash"]]), server_version=u["server_version"],
        ),
    )

    b = _bar_at(gw, "09988", "2026-02-10", "")
    rng = round_half_up(float(b["bar"]["high"]) - float(b["bar"]["low"]), 4)
    add_held(
        _base_q(
            qid="mkt_ht_007",
            company="Alibaba",
            ticker="9988.HK",
            etom="L2",
            question=(
                "On 2026-02-10, what was Alibaba's unadjusted intraday range (high-low) in HKD? "
                "Half-up 4dp; tolerance ±0.05."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"high-low={rng} on {b['effective']}.",
            gold_values=[{
                "value": rng, "unit": "HKD", "price_field": "high-low", "adjustment_mode": "",
                "formula": "high-low", "tolerance_abs": 0.05, "requested_date": "2026-02-10",
                "effective_session_date": b["effective"],
            }],
            task_type="calculation",
            capabilities=["ohlc", "range"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_007", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=b["args"], requested_date="2026-02-10", effective=b["effective"],
            prev_triggered=b["prev_triggered"], adjust="", value=rng, unit="HKD",
            formula="high-low", inputs={"high": b["bar"]["high"], "low": b["bar"]["low"]},
            rounded=rng, tolerance={"abs": 0.05}, raw_hash=b["raw_hash"],
            server_version=b["server_version"],
        ),
    )

    # profile identity
    res = gw.invoke("aktools.stock_hk_security_profile_em", {"symbol": "00700"})
    if not res.success:
        raise RuntimeError(f"profile failed: {res.error}")
    add_held(
        _base_q(
            qid="mkt_ht_008",
            company="Tencent",
            ticker="0700.HK",
            etom="L2",
            question=(
                "Confirm ticker/index identity: does stock_hk_security_profile_em for symbol "
                "00700 refer to Tencent Holdings, and is HSI an index (not that equity)?"
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_security_profile_em"],
            answerability="structured_answerable",
            gold_status="answered",
            reference="00700 profile maps to Tencent; HSI is Hang Seng Index, not 0700.HK equity.",
            gold_values=[{
                "value": {"symbol": "00700", "is_tencent": True, "hsi_is_index": True},
                "unit": "identity",
                "tolerance_abs": 0,
            }],
            task_type="identity",
            capabilities=["ticker_index_identity"],
            risks=[],
            extra={"profile_snapshot_hash": _sha(res.normalized)},
        ),
        _audit_rec(
            qid="mkt_ht_008", operation="aktools.stock_hk_security_profile_em",
            provider="eastmoney_via_akshare", args={"symbol": "00700"},
            requested_date=None, effective=None, prev_triggered=None, adjust=None,
            value=res.normalized, unit="identity", formula="profile identity",
            inputs=res.normalized, rounded={"is_tencent": True, "hsi_is_index": True},
            tolerance={"abs": 0}, raw_hash=_sha(res.raw), server_version=res.server_version,
        ),
    )

    # L3 held
    s = _series(gw, "00005", "2026-05-04", "2026-05-29", "")
    pret = percentage_return(s["closes"][0], s["closes"][-1])
    pret_r = round_half_up(pret, 4)
    add_held(
        _base_q(
            qid="mkt_ht_009",
            company="HSBC",
            ticker="0005.HK",
            etom="L3",
            question=(
                "HSBC percentage return over first-to-last unadjusted closes in "
                "2026-05-04..2026-05-29: 100*(close_end/close_start-1). Half-up 4dp; "
                "tolerance ±0.01 pp."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"{pret_r}% on {s['dates'][0]}..{s['dates'][-1]}.",
            gold_values=[{
                "value": pret_r, "unit": "percent", "representation": "percentage_not_decimal",
                "formula": "100*(close_end/close_start-1)", "adjustment_mode": "",
                "rounding": "half_up_4dp", "tolerance_abs": 0.01,
                "date_window": [s["dates"][0], s["dates"][-1]],
            }],
            task_type="calculation",
            capabilities=["percentage_return"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_009", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=s["args"], requested_date=[s["dates"][0], s["dates"][-1]],
            effective=s["dates"][-1], prev_triggered=False, adjust="", value=pret,
            unit="percent", formula="100*(close_end/close_start-1)",
            inputs={"start": s["closes"][0], "end": s["closes"][-1]}, rounded=pret_r,
            tolerance={"abs": 0.01}, raw_hash=s["raw_hash"], server_version=s["server_version"],
        ),
    )

    s = _series(gw, "09988", "2026-03-01", "2026-03-31", "")
    if len(s["closes"]) < 10:
        raise RuntimeError("MA10 window too short")
    ma10 = moving_average(s["closes"], 10)
    ma10_r = round_half_up(ma10, 4)
    add_held(
        _base_q(
            qid="mkt_ht_010",
            company="Alibaba",
            ticker="9988.HK",
            etom="L3",
            question=(
                "For Alibaba target session on/before 2026-03-31, compute MA10 of unadjusted "
                "closes including that session. Half-up 4dp; tolerance ±0.05."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"MA10={ma10_r} ending {s['dates'][-1]}.",
            gold_values=[{
                "value": ma10_r, "unit": "HKD",
                "formula": "mean(last_10_unadjusted_closes_incl_target)",
                "adjustment_mode": "", "rounding": "half_up_4dp", "tolerance_abs": 0.05,
                "date_window": [s["dates"][-10], s["dates"][-1]],
            }],
            task_type="calculation",
            capabilities=["moving_average"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_010", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=s["args"], requested_date=s["dates"][-1], effective=s["dates"][-1],
            prev_triggered=False, adjust="", value=ma10, unit="HKD", formula="MA10",
            inputs={"closes": s["closes"][-10:]}, rounded=ma10_r, tolerance={"abs": 0.05},
            raw_hash=s["raw_hash"], server_version=s["server_version"],
        ),
    )

    h0 = _hsi_at(gw, "2026-06-03")
    h1 = _hsi_at(gw, "2026-06-30")
    sr = simple_return(float(h0["bar"]["close"]), float(h1["bar"]["close"]))
    sr_r = round_half_up(sr, 6)
    add_held(
        _base_q(
            qid="mkt_ht_011",
            company="HSI",
            ticker="HSI",
            etom="L3",
            question=(
                "HSI simple return from 2026-06-03 close to 2026-06-30 close as decimal "
                "close_end/close_start-1. Half-up 6dp; tolerance ±1e-5."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_index_daily_sina"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"simple_return={sr_r}.",
            gold_values=[{
                "value": sr_r, "unit": "decimal_return", "representation": "decimal_not_percent",
                "formula": "close_end/close_start-1", "rounding": "half_up_6dp",
                "tolerance_abs": 1e-5, "date_window": [h0["effective"], h1["effective"]],
            }],
            task_type="calculation",
            capabilities=["simple_return", "hsi_level"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_011", operation="aktools.stock_hk_index_daily_sina",
            provider="sina_via_akshare", args={"h0": h0["args"], "h1": h1["args"]},
            requested_date=[h0["effective"], h1["effective"]], effective=h1["effective"],
            prev_triggered=False, adjust=None, value=sr, unit="decimal_return",
            formula="close_end/close_start-1",
            inputs={"start": h0["bar"]["close"], "end": h1["bar"]["close"]},
            rounded=sr_r, tolerance={"abs": 1e-5},
            raw_hash=_sha([h0["raw_hash"], h1["raw_hash"]]), server_version=h1["server_version"],
        ),
    )

    s = _series(gw, "00700", "2026-02-03", "2026-02-27", "")
    to_sum = sum(float(b["turnover"] or 0) for b in s["bars"])
    add_held(
        _base_q(
            qid="mkt_ht_012",
            company="Tencent",
            ticker="0700.HK",
            etom="L3",
            question=(
                "Sum Tencent unadjusted daily turnover over sessions from 2026-02-03 through "
                "2026-02-27. Relative tolerance 0.1%. Currency provenance=exchange_convention."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"sum_turnover={to_sum} over {s['dates'][0]}..{s['dates'][-1]}.",
            gold_values=[{
                "value": to_sum, "unit": "HKD", "price_field": "turnover",
                "formula": "sum(turnover)", "adjustment_mode": "", "tolerance_rel": 0.001,
                "date_window": [s["dates"][0], s["dates"][-1]],
                "currency_provenance": "exchange_convention",
            }],
            task_type="calculation",
            capabilities=["turnover"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_012", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=s["args"], requested_date=[s["dates"][0], s["dates"][-1]],
            effective=s["dates"][-1], prev_triggered=False, adjust="", value=to_sum,
            unit="HKD", formula="sum(turnover)",
            inputs={"turnovers": [b.get("turnover") for b in s["bars"]]},
            rounded=to_sum, tolerance={"rel": 0.001}, raw_hash=s["raw_hash"],
            server_version=s["server_version"],
        ),
    )

    # L4 held
    s = _series(gw, "00005", "2026-04-01", "2026-04-30", "")
    av = annualized_volatility(s["closes"])
    av_r = round_half_up(av, 6)
    add_held(
        _base_q(
            qid="mkt_ht_013",
            company="HSBC",
            ticker="0005.HK",
            etom="L4",
            question=(
                "HSBC annualized volatility over unadjusted closes 2026-04-01..2026-04-30: "
                "sample_std(daily simple returns, ddof=1)*sqrt(252). Decimal; half-up 6dp; "
                "tolerance ±1e-4."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"ann_vol={av_r}.",
            gold_values=[{
                "value": av_r, "unit": "decimal_volatility", "representation": "decimal_not_percent",
                "formula": "sample_std(daily_simple_returns,ddof=1)*sqrt(252)",
                "adjustment_mode": "", "rounding": "half_up_6dp", "tolerance_abs": 1e-4,
                "date_window": [s["dates"][0], s["dates"][-1]],
            }],
            task_type="calculation",
            capabilities=["volatility"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_013", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=s["args"], requested_date=[s["dates"][0], s["dates"][-1]],
            effective=s["dates"][-1], prev_triggered=False, adjust="", value=av,
            unit="decimal_volatility", formula="ann_vol", inputs={"closes": s["closes"]},
            rounded=av_r, tolerance={"abs": 1e-4}, raw_hash=s["raw_hash"],
            server_version=s["server_version"],
        ),
    )

    s = _series(gw, "00700", "2026-01-05", "2026-02-27", "")
    mdd = maximum_drawdown(s["closes"])
    mdd_r = round_half_up(mdd, 6)
    add_held(
        _base_q(
            qid="mkt_ht_014",
            company="Tencent",
            ticker="0700.HK",
            etom="L4",
            question=(
                "Tencent MDD over unadjusted closes 2026-01-05..2026-02-27: "
                "min(P_t/running_max-1) as decimal. Half-up 6dp; tolerance ±1e-4."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"MDD={mdd_r}.",
            gold_values=[{
                "value": mdd_r, "unit": "decimal_drawdown", "representation": "decimal_not_percent",
                "formula": "min(P_t/running_max-1)", "adjustment_mode": "",
                "rounding": "half_up_6dp", "tolerance_abs": 1e-4,
                "date_window": [s["dates"][0], s["dates"][-1]],
            }],
            task_type="calculation",
            capabilities=["maximum_drawdown"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_014", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=s["args"], requested_date=[s["dates"][0], s["dates"][-1]],
            effective=s["dates"][-1], prev_triggered=False, adjust="", value=mdd,
            unit="decimal_drawdown", formula="MDD", inputs={"closes": s["closes"]},
            rounded=mdd_r, tolerance={"abs": 1e-4}, raw_hash=s["raw_hash"],
            server_version=s["server_version"],
        ),
    )

    eq = _series(gw, "09988", "2026-07-02", "2026-07-31", "")
    ix = _hsi_series(gw, eq["dates"][0], eq["dates"][-1])
    eq_map = dict(zip(eq["dates"], eq["closes"]))
    ix_map = dict(zip(ix["dates"], ix["closes"]))
    common = sorted(set(eq_map) & set(ix_map))
    d0, d1 = common[0], common[-1]
    rel = simple_return(eq_map[d0], eq_map[d1]) - simple_return(ix_map[d0], ix_map[d1])
    rel_r = round_half_up(rel, 6)
    add_held(
        _base_q(
            qid="mkt_ht_015",
            company="Alibaba",
            ticker="9988.HK",
            etom="L4",
            question=(
                "Alibaba vs HSI relative simple-return performance over common sessions in "
                "2026-07-02..2026-07-31 (equity adjust=\"\"): equity_return - HSI_return "
                "as decimal. Half-up 6dp; tolerance ±1e-4."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist", "aktools.stock_hk_index_daily_sina"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"relative={rel_r} on {d0}..{d1}.",
            gold_values=[{
                "value": rel_r, "unit": "decimal_excess_return",
                "representation": "decimal_not_percent",
                "formula": "simple_return(equity)-simple_return(HSI)",
                "adjustment_mode": "", "rounding": "half_up_6dp", "tolerance_abs": 1e-4,
                "date_window": [d0, d1],
            }],
            task_type="calculation",
            capabilities=["relative_performance", "equity_vs_hsi"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_015",
            operation="aktools.stock_hk_hist+aktools.stock_hk_index_daily_sina",
            provider="eastmoney+sina", args={"eq": eq["args"], "ix": ix["args"]},
            requested_date=[d0, d1], effective=d1, prev_triggered=False, adjust="",
            value=rel, unit="decimal_excess_return",
            formula="simple_return(equity)-simple_return(HSI)",
            inputs={"eq": [eq_map[d0], eq_map[d1]], "ix": [ix_map[d0], ix_map[d1]]},
            rounded=rel_r, tolerance={"abs": 1e-4},
            raw_hash=_sha([eq["raw_hash"], ix["raw_hash"]]), server_version=eq["server_version"],
        ),
    )

    s = _series(gw, "00005", "2026-03-02", "2026-03-31", "")
    ma5 = moving_average(s["closes"], 5)
    # combine MA + return as multi-step L4
    pret = simple_return(s["closes"][0], s["closes"][-1])
    combo = {"ma5": round_half_up(ma5, 4), "simple_return": round_half_up(pret, 6)}
    add_held(
        _base_q(
            qid="mkt_ht_016",
            company="HSBC",
            ticker="0005.HK",
            etom="L4",
            question=(
                "For HSBC unadjusted series 2026-03-02..2026-03-31, report (i) MA5 ending on "
                "the last session and (ii) window simple return close_end/close_start-1 "
                "(decimal). MA5 half-up 4dp ±0.05; return half-up 6dp ±1e-4."
            ),
            split="heldout",
            role="final_evaluation_only",
            locked=True,
            gold_ops=["aktools.stock_hk_hist"],
            answerability="structured_answerable",
            gold_status="answered",
            reference=f"ma5={combo['ma5']}, simple_return={combo['simple_return']}.",
            gold_values=[{
                "value": combo, "unit": "mixed", "adjustment_mode": "",
                "formulas": ["MA5", "close_end/close_start-1"],
                "date_window": [s["dates"][0], s["dates"][-1]],
                "tolerances": {"ma5_abs": 0.05, "simple_return_abs": 1e-4},
            }],
            task_type="calculation",
            capabilities=["moving_average", "simple_return"],
            risks=[],
        ),
        _audit_rec(
            qid="mkt_ht_016", operation="aktools.stock_hk_hist", provider="eastmoney_via_akshare",
            args=s["args"], requested_date=[s["dates"][0], s["dates"][-1]],
            effective=s["dates"][-1], prev_triggered=False, adjust="", value=combo,
            unit="mixed", formula="MA5+simple_return",
            inputs={"closes": s["closes"]}, rounded=combo,
            tolerance={"ma5_abs": 0.05, "simple_return_abs": 1e-4},
            raw_hash=s["raw_hash"], server_version=s["server_version"],
        ),
    )

    # L5 held
    for qid, company, ticker, question, reason in [
        (
            "mkt_ht_017",
            "HSBC",
            "0005.HK",
            "What is HSBC's current market price right now?",
            "current/right now forbidden as formal gold under fixed cutoff.",
        ),
        (
            "mkt_ht_018",
            "HSI",
            "HSI",
            "Using tool_trade_date_hist_sina as the official HKEX trading calendar, list HKEX sessions in August 2026.",
            "tool_trade_date_hist_sina is explicitly NOT HKEX official calendar.",
        ),
        (
            "mkt_ht_019",
            "Tencent",
            "0700.HK",
            "What was Tencent's unadjusted close on 2026-08-20 (after dataset cutoff 2026-08-12)?",
            "Requested date after frozen cutoff; post-cutoff gold forbidden.",
        ),
        (
            "mkt_ht_020",
            "Alibaba",
            "9988.HK",
            "Call FTShare stock_hk_daily and return Alibaba's official gold close for 2026-07-15.",
            "FTShare operations blocked; not in active registry; not gold.",
        ),
    ]:
        add_held(
            _base_q(
                qid=qid,
                company=company,
                ticker=ticker,
                etom="L5",
                question=question,
                split="heldout",
                role="final_evaluation_only",
                locked=True,
                gold_ops=[],
                answerability="should_refuse_or_clarify",
                gold_status="unanswerable",
                reference="Refuse/clarify; confirmed infeasible under frozen Market catalog.",
                gold_values=[],
                task_type="refusal_clarification",
                capabilities=["infeasible_under_catalog"],
                risks=[reason],
                extra={"confirmed_infeasible": True, "infeasibility_reason": reason},
            ),
            _audit_rec(
                qid=qid, operation="none", provider="n/a", args={}, requested_date=None,
                effective=None, prev_triggered=None, adjust=None, value=None, unit="n/a",
                formula="infeasible", inputs={}, rounded=None, tolerance=None,
                raw_hash="", server_version=None, notes=reason,
            ),
        )

    return dev, held, audits


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit_md(path: Path, audits: List[Dict[str, Any]]) -> None:
    lines = [
        "# Market Real-MCP Provenance Audit v1",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Catalog: `{DEFAULT_MARKET_CATALOG}`",
        f"- Cutoff: {MARKET_CUTOFF.isoformat()}",
        "- Method: direct Real-MCP invokes only (no Agent/LLM/embedding).",
        "- Mock FinancialServer: **not used**.",
        "",
        f"Records: {len(audits)}",
        "",
    ]
    for a in audits:
        lines.append(f"## {a['question_id']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(a, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _unset_proxies()
    require_market_real_mcp_ready(DEFAULT_MARKET_CATALOG)
    with MarketRealMcpGateway(catalog_path=DEFAULT_MARKET_CATALOG, persistent=True) as gw:
        dev, held, audits = build_benchmarks(gw)
    write_jsonl(OUT_DEV, dev)
    write_jsonl(OUT_HELD, held)
    OUT_AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_AUDIT_JSONL, audits)
    write_audit_md(OUT_AUDIT, audits)
    print(f"wrote {OUT_DEV} ({len(dev)})")
    print(f"wrote {OUT_HELD} ({len(held)})")
    print(f"wrote {OUT_AUDIT} + {OUT_AUDIT_JSONL} ({len(audits)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
