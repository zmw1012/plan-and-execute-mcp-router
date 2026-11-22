"""Guard aktools.stock_financial_hk_report_em parameter semantics consistency."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
OID = "aktools.stock_financial_hk_report_em"
STATEMENT_TYPES = ("资产负债表", "利润表", "现金流量表")
FREQUENCIES = ("年度", "报告期")


def _load_report_em() -> dict:
    data = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    ops = data.get("operations") or []
    for op in ops:
        if isinstance(op, dict) and op.get("operation_id") == OID:
            return op
    raise AssertionError(f"{OID} missing from {CATALOG}")


def test_report_em_parameter_semantics_not_inverted():
    op = _load_report_em()
    params = {
        p["name"]: p
        for p in (op.get("parameters") or [])
        if isinstance(p, dict) and p.get("name")
    }
    assert set(params) >= {"stock", "symbol", "indicator"}

    stock_desc = str(params["stock"].get("description") or "")
    symbol_desc = str(params["symbol"].get("description") or "")
    indicator_desc = str(params["indicator"].get("description") or "")
    description = str(op.get("description") or "")

    # Parameter schema semantics.
    assert "00700" in stock_desc or "5-digit" in stock_desc.lower()
    for label in STATEMENT_TYPES:
        assert label in symbol_desc
        assert label not in stock_desc
        assert label not in indicator_desc
    for freq in FREQUENCIES:
        assert freq in indicator_desc
        assert freq not in symbol_desc

    # Top-level description must not invert symbol↔stock or indicator↔statement.
    low = description.lower()
    assert "use symbol like 00700" not in low
    assert "symbol like 00700" not in low
    assert "stock" in low and "00700" in description
    assert "symbol" in low
    assert all(label in description for label in STATEMENT_TYPES)
    assert all(freq in description for freq in FREQUENCIES)
    # Statement types must not be attributed to indicator in free text.
    assert "indicator" in low
    # After "indicator", only frequency words should appear as the allowed set.
    after_ind = description.split("indicator", 1)[1]
    assert "年度" in after_ind and "报告期" in after_ind
    assert "利润表" not in after_ind.split("Bank", 1)[0]
