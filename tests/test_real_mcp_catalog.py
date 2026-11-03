"""Offline tests for frozen Financial Real-MCP catalog + resolver wiring."""

from __future__ import annotations

from pathlib import Path

from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.catalog import (
    DEFAULT_CATALOG_PATH,
    load_real_mcp_catalog,
    load_real_mcp_registry,
)
from hk_equity_mcp.real_mcp.types import RealMcpResult


def test_catalog_file_exists_and_counts():
    assert DEFAULT_CATALOG_PATH.exists()
    catalog = load_real_mcp_catalog()
    assert catalog["catalog_id"] == "financial_real_mcp_v1"
    assert set(catalog["servers"]) == {"aktools", "mcp_hkexnews"}
    ops = catalog["operations"]
    assert len(ops) == 21  # 3 meta + 12 curated + 6 hkexnews
    ids = [o["operation_id"] for o in ops]
    assert len(ids) == len(set(ids))
    assert "aktools.stock_financial_hk_analysis_indicator_em" in ids
    assert "hkexnews.keyword_search" in ids
    assert all(o.get("read_only", True) for o in ops)
    # Forbidden servers not present in active server map / operation server_ids.
    assert "ftshare" not in catalog["servers"]
    assert "ah_disclosure" not in catalog["servers"]
    assert {o["server_id"] for o in ops} == {"aktools", "mcp_hkexnews"}


def test_registry_loads_operation_specs():
    registry = load_real_mcp_registry()
    assert len(registry) == 21
    op = registry["aktools.stock_financial_hk_report_em"]
    assert op.service_id == "aktools"
    assert op.role == "core_gold"
    assert op.exposed_mcp_tool == "get_data"
    assert op.underlying_aktools_interface == "stock_financial_hk_report_em"
    assert op.execution.get("kind") == "aktools_get_data"


class _FakeGateway:
    def invoke(self, operation_id, arguments=None):
        return RealMcpResult(
            success=True,
            server_id="aktools",
            operation_id=operation_id,
            arguments=arguments or {},
            normalized={"rows": [{"OPERATE_INCOME": 751766000000}]},
            real_mcp=True,
            protocol_success=True,
            execution_success=True,
            transport="stdio",
        )


def test_resolver_invoke_uses_real_gateway_not_financial_server(tmp_path: Path):
    registry = load_real_mcp_registry()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=registry,
        cache_dir=tmp_path / "emb",
        real_mcp_gateway=_FakeGateway(),
    )
    assert resolver.financial is None
    resp = resolver.invoke_api(
        "aktools.stock_financial_hk_analysis_indicator_em",
        {"symbol": "00700", "indicator": "年度"},
    )
    assert resp["success"] is True
    assert resp["result"]["real_mcp"] is True
    assert resp["result"]["normalized"]["rows"][0]["OPERATE_INCOME"] == 751766000000


def test_coverage_jsonl_counts():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "questions"
        / "financial_real_mcp_coverage_v1.jsonl"
    )
    rows = [
        __import__("json").loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 30
    classes = {r["answerability_class"] for r in rows}
    assert classes == {
        "structured_answerable",
        "evidence_answerable",
        "intentionally_infeasible",
    }
    assert sum(1 for r in rows if r["answerability_class"] == "structured_answerable") == 10
    assert sum(1 for r in rows if r["answerability_class"] == "evidence_answerable") == 15
    assert sum(1 for r in rows if r["answerability_class"] == "intentionally_infeasible") == 5
    assert sum(1 for r in rows if r["answerability_class"] == "unsupported_invalid") == 0
    # HSBC absolute revenue not structured.
    hsbc_rev = next(r for r in rows if r["id"] == "fin_cf_010")
    assert hsbc_rev["answerability_class"] == "evidence_answerable"
    assert hsbc_rev["gold_server_ids"] == ["mcp_hkexnews"]
