"""Deterministic local operation registry for HK-equity tools."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec


def build_financial_operations() -> List[OperationSpec]:
    """Register existing Financial MCP tools as Operation-Resolver operations.

    Does not invent new financial data or remote APIs.
    """
    return [
        OperationSpec(
            operation_id="financial.get_financial_metric",
            service_id="financial",
            name="get_financial_metric",
            summary=(
                "Retrieve a reported or adjusted financial metric for a HK-listed "
                "ticker from a selected annual or interim report period."
            ),
            description=(
                "Looks up revenue, net_profit, or eps for Tencent (0700.HK), "
                "HSBC (0005.HK), or Alibaba (9988.HK). Period may be a "
                "fiscal_year_label (FY2025, 1H2026) or period_end (YYYY-MM-DD). "
                "Default without period is latest annual. Ambiguous calendar years "
                "without report_type return a structured error."
            ),
            action_terms=[
                "get",
                "retrieve",
                "lookup",
                "metric",
                "revenue",
                "profit",
                "eps",
                "financial",
                "annual",
                "interim",
            ],
            parameters=[
                ParameterSpec(
                    name="ticker",
                    type="string",
                    required=True,
                    description="HK ticker, e.g. 0700.HK or 0700",
                ),
                ParameterSpec(
                    name="metric",
                    type="string",
                    required=True,
                    description="Metric key: revenue | net_profit | eps",
                    enum=["revenue", "net_profit", "eps"],
                ),
                ParameterSpec(
                    name="period",
                    type="string",
                    required=False,
                    description="fiscal_year_label or period_end date",
                ),
                ParameterSpec(
                    name="report_type",
                    type="string",
                    required=False,
                    description="annual or interim",
                    enum=["annual", "interim"],
                ),
                ParameterSpec(
                    name="metric_type",
                    type="string",
                    required=False,
                    description="reported or adjusted",
                    enum=["reported", "adjusted"],
                    default="reported",
                ),
            ],
            read_only=True,
            constraints=[
                "Do not mix currencies across tickers.",
                "Ambiguous period like '2026' without report_type is rejected.",
                "Alibaba FY2026 period_end is 2026-03-31, not calendar year-end.",
            ],
            invocation_guidance=(
                "Prefer explicit period + report_type. Use metric_type=adjusted "
                "only when the user asks for non-IFRS / non-GAAP / constant-currency "
                "excl. notable items."
            ),
            underlying_tool="get_financial_metric",
        ),
        OperationSpec(
            operation_id="financial.get_filing_summary",
            service_id="financial",
            name="get_filing_summary",
            summary=(
                "Return a structured factual filing summary for a selected "
                "annual or interim report, including key reported metrics and sources."
            ),
            description=(
                "Builds a non-LLM structured summary for a verified report period: "
                "selected report metadata, key reported metrics, currency/unit, "
                "period, and source evidence. Does not invent subjective conclusions."
            ),
            action_terms=[
                "summarize",
                "filing",
                "report",
                "annual",
                "interim",
                "highlights",
                "overview",
            ],
            parameters=[
                ParameterSpec(
                    name="ticker",
                    type="string",
                    required=True,
                    description="HK ticker, e.g. 0005.HK",
                ),
                ParameterSpec(
                    name="period",
                    type="string",
                    required=False,
                    description="fiscal_year_label or period_end date",
                ),
                ParameterSpec(
                    name="report_type",
                    type="string",
                    required=False,
                    description="annual or interim",
                    enum=["annual", "interim"],
                ),
            ],
            read_only=True,
            constraints=[
                "Summary is factual only; no subjective investment advice.",
                "Unknown period/report_type returns a structured error.",
            ],
            invocation_guidance=(
                "Use when the user asks for filing highlights or a report overview. "
                "Still require a concrete period when the question is period-specific."
            ),
            underlying_tool="get_filing_summary",
        ),
    ]


def build_default_registry() -> Dict[str, OperationSpec]:
    ops = build_financial_operations()
    registry = {op.operation_id: op for op in ops}
    # Stable IDs are deterministic and unique.
    if len(registry) != len(ops):
        raise ValueError("Duplicate operation_id in registry")
    return registry


def registry_fingerprint(registry: Dict[str, OperationSpec]) -> str:
    payload = {
        op_id: {
            "service_id": op.service_id,
            "name": op.name,
            "document": op.index_document(),
        }
        for op_id, op in sorted(registry.items())
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
