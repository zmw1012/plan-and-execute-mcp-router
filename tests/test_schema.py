from hk_equity_mcp.common.schema import REQUIRED_FIELDS, make_response, validate_response


def test_make_response_has_required_fields():
    payload = make_response(
        success=True,
        source="mock",
        ticker="0700.HK",
        as_of_date="2024-12-31",
        data={"x": 1},
        evidence=["e1"],
        error=None,
    )
    for key in REQUIRED_FIELDS:
        assert key in payload
    assert validate_response(payload)


def test_validate_response_rejects_missing_field():
    assert not validate_response({"success": True})
