from hk_equity_mcp.common.schema import validate_response
from hk_equity_mcp.mcp_servers.news import NewsServer


def test_search_announcements():
    server = NewsServer()
    result = server.search_announcements("9988.HK", query="logistics")
    assert validate_response(result)
    assert result["success"] is True
    assert result["data"]["announcements"]
    assert "logistics" in result["data"]["announcements"][0]["headline"].lower()


def test_get_news_summary():
    server = NewsServer()
    result = server.get_news_summary("0700.HK")
    assert result["success"] is True
    assert "AI" in result["data"]["summary"]
