"""Deterministic tests for Market Real-MCP candidate catalog (discovery v1).

No LLM/embedding calls. Live network smoke is optional/separate.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "market_real_mcp_candidate_v1.yaml"
FIN_CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
DOCS = [
    ROOT / "docs" / "market_mcp_candidate_audit.md",
    ROOT / "docs" / "market_mcp_capability_matrix.md",
    ROOT / "docs" / "market_operation_metadata_for_retrieval.md",
    ROOT / "docs" / "market_benchmark_blueprint.md",
]
SMOKE = ROOT / "scripts" / "market_mcp_discovery_smoke.py"
REQUIRED_OP_FIELDS = {
    "operation_id",
    "server_id",
    "exposed_mcp_tool",
    "name",
    "summary",
    "description",
    "action_terms",
    "parameters",
    "output_fields",
    "asset_type",
    "ticker_format",
    "supported_exchanges",
    "frequency",
    "historical_date_coverage",
    "timezone",
    "currency",
    "adjustment_mode",
    "data_source_provider",
    "read_only",
    "known_risks",
    "audit_status",
    "live_evidence",
    "decision",
    "classification",
}


def _load_catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def test_artifacts_exist():
    assert CATALOG.is_file()
    assert SMOKE.is_file()
    for doc in DOCS:
        assert doc.is_file(), doc


def test_catalog_is_candidate_and_separate_from_financial():
    cat = _load_catalog()
    assert cat["catalog_id"] == "market_real_mcp_candidate_v1"
    assert cat["catalog_status"] == "candidate_v1"
    assert cat.get("freeze_status") == "not_frozen"
    assert cat["domain"] == "market"
    assert FIN_CATALOG.is_file()
    fin = yaml.safe_load(FIN_CATALOG.read_text(encoding="utf-8"))
    assert fin["catalog_id"] == "financial_real_mcp_v1"
    # Market file must not reuse Financial catalog_id
    assert cat["catalog_id"] != fin["catalog_id"]


def test_unique_operation_ids_and_required_metadata():
    cat = _load_catalog()
    ops = cat["operations"]
    ids = [op["operation_id"] for op in ops]
    assert len(ids) == len(set(ids))
    assert len(ops) == cat["counts"]["logical_operations_in_this_candidate_file"]
    for op in ops:
        missing = REQUIRED_OP_FIELDS - set(op)
        assert not missing, f"{op.get('operation_id')}: missing {sorted(missing)}"
        assert op["server_id"] in cat["servers"]
        assert op["decision"] in {
            "primary",
            "supporting",
            "distractor",
            "rejected",
            "blocked",
        }
        assert op["classification"] == op["decision"]
        assert op["read_only"] is True


def test_decision_counts_and_statuses():
    cat = _load_catalog()
    decisions = Counter(op["decision"] for op in cat["operations"])
    # Catalog schema may include zero-count decision keys; Counter omits zeros.
    expected = Counter(
        {k: v for k, v in cat["counts"]["by_decision"].items() if int(v) != 0}
    )
    assert decisions == expected
    assert decisions["primary"] >= 2
    assert decisions["distractor"] >= 3
    assert decisions["blocked"] >= 1
    for op in cat["operations"]:
        if op["decision"] in {"primary", "supporting", "distractor"}:
            assert op["server_id"] == "aktools"
            assert "live" in op["audit_status"] or op["audit_status"].startswith(
                "described"
            )
            assert op.get("live_evidence")
        if op["decision"] == "blocked":
            assert op["server_id"] == "ftshare"
            assert "blocked" in op["audit_status"]


def test_primary_hist_covers_three_companies_and_adjustment_explicit():
    cat = _load_catalog()
    hist = next(op for op in cat["operations"] if op["operation_id"] == "aktools.stock_hk_hist")
    assert hist["decision"] == "primary"
    assert set(hist["covers_companies"]) == {"Tencent", "HSBC", "Alibaba"}
    assert "qfq" in hist["adjustment_mode"]
    assert hist["timezone"] == "Asia/Hong_Kong"
    assert "2026-08-12" in cat["temporal_policy"]["max_data_date"]
    assert "previous_valid_trading_day_rule" in cat["temporal_policy"]


def test_index_primary_and_server_raw_tool_counts():
    cat = _load_catalog()
    hsi = next(
        op
        for op in cat["operations"]
        if op["operation_id"] == "aktools.stock_hk_index_daily_sina"
    )
    assert hsi["decision"] == "primary"
    assert hsi["asset_type"] == "index"
    assert cat["servers"]["aktools"]["raw_mcp_tool_count"] == 3
    assert cat["counts"]["raw_mcp_tools_by_server"]["aktools"] == 3
    # Logical ops must exceed raw tools (interfaces behind get_data)
    assert cat["counts"]["logical_operations_in_this_candidate_file"] > 3


def test_valid_aktools_tool_mappings():
    cat = _load_catalog()
    raw = set(cat["servers"]["aktools"]["exposed_mcp_tools"])
    assert raw == {"search_interfaces", "describe_interface", "get_data"}
    for op in cat["operations"]:
        if op["server_id"] != "aktools":
            continue
        assert op["exposed_mcp_tool"] in raw
        if op["exposed_mcp_tool"] == "get_data":
            assert op.get("underlying_interface"), op["operation_id"]


def test_financial_files_unchanged_in_git_diff():
    # Against HEAD: Market discovery branch should not modify Financial freeze paths.
    paths = [
        "configs/operations/financial_real_mcp.yaml",
        "benchmarks/heldout/financial_test_30_v1.jsonl",
        "benchmarks/heldout/financial_test_30_v1_1.jsonl",
        "benchmarks/questions/financial_mcp_free_30_v1.jsonl",
        "docs/financial_heldout_real_mcp_provenance_audit_v1_1.md",
    ]
    proc = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "", proc.stdout


def test_blueprint_has_no_final_heldout_gold_block():
    text = (ROOT / "docs" / "market_benchmark_blueprint.md").read_text(encoding="utf-8")
    assert "Does not contain" in text or "does not contain" in text.lower()
    assert "Development" in text and "15" in text
    assert "Held-out" in text and "20" in text
    # Guard against accidental pasted JSONL gold dumps
    assert "fin_ht_" not in text
    assert "gold_values" not in text.lower() or "Required gold fields" in text
