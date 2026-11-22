"""Deterministic pagination policy tests (APV-Fin v1.4)."""

from __future__ import annotations

from hk_equity_mcp.apv_fin.pagination import (
    MAX_PAGINATION_PAGES_PER_STEP,
    next_page_arguments,
    operation_supports_offset,
    pagination_policy_hash,
    pagination_signature,
    should_paginate,
)


def test_pagination_policy_hash_stable():
    assert pagination_policy_hash()
    assert MAX_PAGINATION_PAGES_PER_STEP == 5


def test_next_page_advances_by_returned_rows():
    nxt = next_page_arguments(
        current_arguments={"symbol": "09988", "offset": 0, "max_rows": 10},
        envelope={
            "truncated": True,
            "offset": 0,
            "returned_rows": 10,
            "total_rows": 261,
            "rows": [{}] * 10,
        },
        capability="financial.reported_metric_bundle",
    )
    assert nxt is not None
    assert nxt["offset"] == 10
    assert nxt["max_rows"] >= 10


def test_should_paginate_gates():
    ok, reason = should_paginate(
        mcp_success=True,
        envelope={"truncated": True, "returned_rows": 10, "total_rows": 100, "offset": 0},
        required_outputs_ok=False,
        supports_offset=True,
        pages_used=0,
        rows_scanned=10,
        mcp_budget_remaining=3,
    )
    assert ok is True
    assert reason == "paginate"

    no, why = should_paginate(
        mcp_success=True,
        envelope={"truncated": True, "returned_rows": 10, "total_rows": 100, "offset": 0},
        required_outputs_ok=True,
        supports_offset=True,
        pages_used=0,
        rows_scanned=10,
        mcp_budget_remaining=3,
    )
    assert no is False
    assert why == "outputs_satisfied"


def test_duplicate_pagination_signature_stable():
    a = pagination_signature(
        operation_id="aktools.stock_financial_hk_report_em",
        base_arguments={"symbol": "09988", "indicator": "资产负债表"},
        offset=10,
        max_rows=100,
    )
    b = pagination_signature(
        operation_id="aktools.stock_financial_hk_report_em",
        base_arguments={"symbol": "09988", "indicator": "资产负债表", "offset": 999},
        offset=10,
        max_rows=100,
    )
    assert a == b


def test_operation_supports_offset():
    assert operation_supports_offset(
        {"properties": {"offset": {"type": "integer"}, "max_rows": {"type": "integer"}}}
    )
    assert not operation_supports_offset({"properties": {"symbol": {"type": "string"}}})
