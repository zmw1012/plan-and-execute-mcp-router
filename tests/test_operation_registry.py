"""Tests for deterministic operation registry."""

from hk_equity_mcp.operation_resolver.registry import (
    build_default_registry,
    build_financial_operations,
    registry_fingerprint,
)


def test_stable_unique_operation_ids():
    ops = build_financial_operations()
    ids = [op.operation_id for op in ops]
    assert ids == [
        "financial.get_financial_metric",
        "financial.get_filing_summary",
    ]
    registry = build_default_registry()
    assert set(registry) == set(ids)
    # Second build is identical (deterministic).
    assert list(build_default_registry()) == list(registry)


def test_index_document_contains_paper_fields():
    op = build_default_registry()["financial.get_financial_metric"]
    doc = op.index_document()
    for needle in (
        "summary:",
        "service: financial",
        "operation: get_financial_metric",
        "action_terms:",
        "required_parameters:",
        "optional_parameters:",
        "constraints:",
        "ticker:string",
        "metric:string",
    ):
        assert needle in doc


def test_registry_fingerprint_stable():
    a = registry_fingerprint(build_default_registry())
    b = registry_fingerprint(build_default_registry())
    assert a == b
    assert len(a) == 64


def test_contract_schema_shapes():
    op = build_default_registry()["financial.get_financial_metric"]
    contract = op.contract()
    assert contract["service_id"] == "financial"
    assert contract["name"] == "get_financial_metric"
    assert contract["read_only"] is True
    req_names = {p["name"] for p in contract["required_parameters"]}
    opt_names = {p["name"] for p in contract["optional_parameters"]}
    assert req_names == {"ticker", "metric"}
    assert "period" in opt_names
    schema = contract["json_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"ticker", "metric"}
    assert schema["properties"]["metric"]["enum"] == ["revenue", "net_profit", "eps"]
    assert schema["additionalProperties"] is False
