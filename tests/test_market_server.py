from hk_equity_mcp.common.schema import validate_response
from hk_equity_mcp.mcp_servers.market import MarketServer


def test_get_price_series():
    server = MarketServer()
    result = server.get_price_series("0700.HK", start_date="2024-12-27")
    assert validate_response(result)
    assert result["success"] is True
    assert len(result["data"]["series"]) == 3


def test_calculate_return():
    server = MarketServer()
    result = server.calculate_return("0005.HK", "2024-12-27", "2024-12-31")
    assert result["success"] is True
    assert result["data"]["return"] == 0.017931
