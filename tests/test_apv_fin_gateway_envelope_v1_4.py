"""APV-Fin v1.4 gateway envelope normalizer + adapter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hk_equity_mcp.apv_fin import APV_FIN_METHOD_VERSION
from hk_equity_mcp.apv_fin.identity import GATEWAY_ADAPTER_PROTOCOL_CHANGE
from hk_equity_mcp.apv_fin.output_adapters import (
    adapt_mcp_result,
    output_adapter_hash,
    required_outputs_present,
)
from hk_equity_mcp.apv_fin.result_envelope import (
    EnvelopeNormalizationError,
    normalize_gateway_envelope,
    result_envelope_hash,
)
from hk_equity_mcp.experiment.unified_apv_fin_runner import load_apv_fin_config

ROOT = Path(__file__).resolve().parents[1]


def test_method_version_v1_4():
    assert APV_FIN_METHOD_VERSION == "apv_fin_v1_4_gateway_adapter_and_progression"
    cfg = load_apv_fin_config()
    assert cfg["method_version"] == APV_FIN_METHOD_VERSION
    assert (
        GATEWAY_ADAPTER_PROTOCOL_CHANGE
        == "gateway_envelope_adapter_before_valid_method_evaluation"
    )
    assert result_envelope_hash()
    assert output_adapter_hash()


def test_nested_gateway_envelope_aktools_json_rows():
    outer = {
        "success": True,
        "operation_id": "aktools.stock_hk_hist",
        "arguments": {"symbol": "00700"},
        "result": {
            "success": True,
            "server_id": "aktools",
            "operation_id": "aktools.stock_hk_hist",
            "arguments": {},
            "raw": {
                "is_error": False,
                "content": [
                    {
                        "json": {
                            "name": "stock_hk_hist",
                            "total_rows": 2,
                            "offset": 0,
                            "returned_rows": 2,
                            "truncated": False,
                            "columns": ["日期", "收盘", "成交量"],
                            "rows": [
                                {"日期": "2026-08-11", "收盘": 10.0, "成交量": 1},
                                {"日期": "2026-08-12", "收盘": 11.0, "成交量": 2},
                            ],
                        }
                    }
                ],
            },
        },
    }
    env = normalize_gateway_envelope(outer)
    assert env["execution_success"] is True
    assert env["protocol_success"] is True
    assert len(env["rows"]) == 2
    assert "result.raw.content[0].json" in env["raw_provenance_paths"] or env["rows"]


def test_rejects_arbitrary_nested_rows_scavenge():
    # Unsupported shape: rows buried under unrelated key must not be accepted
    # unless it matches an explicit Gateway path.
    weird = {
        "success": True,
        "operation_id": "x",
        "arguments": {},
        "result": {
            "success": True,
            "raw": {"is_error": False, "content": [{"meta": {"rows": [{"a": 1}]}}]},
        },
    }
    env = normalize_gateway_envelope(weird)
    assert env["rows"] == []
    assert "no_supported_payload_path" in env["normalization_errors"]


def test_financial_wide_row_extraction():
    inv = {
        "success": True,
        "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
        "result": {
            "success": True,
            "raw": {
                "is_error": False,
                "content": [
                    {
                        "json": {
                            "truncated": False,
                            "offset": 0,
                            "returned_rows": 2,
                            "total_rows": 2,
                            "columns": [
                                "REPORT_DATE",
                                "BASIC_EPS",
                                "CURRENCY",
                                "OPERATE_INCOME",
                                "HOLDER_PROFIT",
                            ],
                            "rows": [
                                {
                                    "REPORT_DATE": "2024-12-31",
                                    "BASIC_EPS": 1.0,
                                    "CURRENCY": "HKD",
                                    "OPERATE_INCOME": 10,
                                    "HOLDER_PROFIT": 2,
                                },
                                {
                                    "REPORT_DATE": "2025-12-31",
                                    "BASIC_EPS": 8.5,
                                    "CURRENCY": "USD",
                                    "OPERATE_INCOME": 100,
                                    "HOLDER_PROFIT": 20,
                                },
                            ],
                        }
                    }
                ],
            },
        },
    }
    adapted = adapt_mcp_result(
        capability="financial.reported_metric_bundle",
        raw_result=inv,
        task_spec={"periods": ["2025-12-31", "FY2025"]},
    )
    assert adapted["basic_eps"] == 8.5
    assert adapted["currency"] == "USD"
    assert adapted["report_date"] == "2025-12-31"
    assert adapted["metrics"]["revenue"] == 100
    assert adapted["metrics"]["attributable_profit"] == 20
    check = required_outputs_present(
        adapted, ["basic_eps", "currency", "report_date"]
    )
    assert check["ok"] is True


def test_financial_long_table_and_pagination_needed():
    inv = {
        "success": True,
        "operation_id": "aktools.stock_financial_hk_report_em",
        "result": {
            "success": True,
            "raw": {
                "is_error": False,
                "content": [
                    {
                        "json": {
                            "truncated": True,
                            "offset": 0,
                            "returned_rows": 2,
                            "total_rows": 200,
                            "columns": [
                                "STD_ITEM_NAME",
                                "STD_ITEM_CODE",
                                "AMOUNT",
                                "REPORT_DATE",
                            ],
                            "rows": [
                                {
                                    "STD_ITEM_NAME": "营业额",
                                    "STD_ITEM_CODE": "R1",
                                    "AMOUNT": 1000,
                                    "REPORT_DATE": "2026-03-31",
                                },
                                {
                                    "STD_ITEM_NAME": "毛利",
                                    "STD_ITEM_CODE": "R2",
                                    "AMOUNT": 400,
                                    "REPORT_DATE": "2026-03-31",
                                },
                            ],
                        }
                    }
                ],
            },
        },
    }
    adapted = adapt_mcp_result(
        capability="financial.reported_metric_bundle",
        raw_result=inv,
        task_spec={"periods": ["2026-03-31"]},
    )
    assert adapted["metrics"]["revenue"] == 1000
    assert "attributable_profit" not in adapted["metrics"]
    assert adapted["pagination_needed"] is True
    assert adapted["field_provenance"][0]["original_field"] == "营业额"


def test_market_chinese_ohlcv_cutoff_selection():
    inv = {
        "success": True,
        "operation_id": "aktools.stock_hk_hist",
        "result": {
            "success": True,
            "raw": {
                "is_error": False,
                "content": [
                    {
                        "json": {
                            "truncated": False,
                            "rows": [
                                {"日期": "2026-08-13", "收盘": 999, "成交量": 1},
                                {"日期": "2026-08-12", "收盘": 461.6, "成交量": 27593015},
                                {"日期": "2026-08-11", "收盘": 450.0, "成交量": 100},
                            ],
                            "columns": ["日期", "收盘", "成交量"],
                        }
                    }
                ],
            },
        },
    }
    adapted = adapt_mcp_result(
        capability="market.latest_close",
        raw_result=inv,
        arguments={"symbol": "00700"},
        task_spec={"cutoff_date": "2026-08-12"},
    )
    assert adapted["trade_date"] == "2026-08-12"
    assert adapted["close"] == 461.6
    assert adapted["volume"] == 27593015
    assert adapted["currency"] == "HKD"
    assert adapted["currency_source"] == "contract_derived"


def test_no_task_company_gold_branching_in_adapters():
    src = (ROOT / "src/hk_equity_mcp/apv_fin/output_adapters.py").read_text(
        encoding="utf-8"
    )
    for banned in ("fin_cf_", "mkt_dev_", "gold_values", "Tencent", "HSBC", "Alibaba"):
        assert banned not in src
    env_src = (ROOT / "src/hk_equity_mcp/apv_fin/result_envelope.py").read_text(
        encoding="utf-8"
    )
    assert "if task_id" not in env_src
