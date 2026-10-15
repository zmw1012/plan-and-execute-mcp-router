from hk_equity_mcp.common.schema import validate_response
from hk_equity_mcp.mcp_servers.financial import FinancialServer


def test_get_financial_metric_success():
    server = FinancialServer()
    result = server.get_financial_metric("0700.HK", "revenue")
    assert validate_response(result)
    assert result["success"] is True
    assert result["source"] == "mock_financial"
    assert result["data"]["value"] == 609015_000_000
    assert result["evidence"]


def test_get_filing_summary_success():
    server = FinancialServer()
    result = server.get_filing_summary("0005.HK")
    assert result["success"] is True
    assert "net interest income" in result["data"]["summary"]


def test_unknown_ticker():
    server = FinancialServer()
    result = server.get_financial_metric("9999.HK", "revenue")
    assert result["success"] is False
    assert result["error"]
