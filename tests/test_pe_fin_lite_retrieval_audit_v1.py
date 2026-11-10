"""Unit tests for PE-Fin Lite HK-155 Retrieval Audit (offline only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.catalog_audit import (
    analyze_catalog_coverage,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    assert_frozen_evaluation_allowed,
    compare_formulas,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.gold_reference import (
    acceptable_ops_union,
    build_evaluation_reference,
    equivalents_by_node,
    is_disclosure_composite_task,
    is_no_tool_task,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    load_hk155_locked,
    question_level_cache_identity,
    step_retrieval_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.metrics import (
    complete_route_at_k,
    hit_at_k,
    mrr_from_rank,
    rank4_to5_not_described,
    recall_at_k,
    score_multi_step_route,
    score_ranked_list,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    FORBIDDEN_PLANNER_KEYS,
    extract_retrieve_steps,
    mock_plan_for_question,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    rank_operations,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (
    RetrievalAuditRunner,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hk155_identity_and_order():
    data = load_hk155_locked(ROOT)
    assert data["identities"]["n_tasks"] == 155
    assert len(data["task_ids"]) == 155
    assert len(set(data["task_ids"])) == 155
    assert data["task_ids"] == list(data["run_order"]["task_ids"])
    assert data["identities"]["not_for_unseen_final_claims"] is True


def test_split_70_85():
    data = load_hk155_locked(ROOT)
    assert len(data["splits"]["development70"]) == 70
    assert len(data["splits"]["previously_reserved85"]) == 85
    assert len(data["splits"]["all155"]) == 155
    assert set(data["splits"]["development70"]).isdisjoint(
        data["splits"]["previously_reserved85"]
    )


def test_no_gold_in_mock_planner_or_queries():
    task = {
        "task_id": "fin_cf_001",
        "domain": "financial",
        "question": "What is revenue?",
        "gold_operations": ["aktools.stock_financial_hk_report_em"],
        "acceptable_tool_sets": [["aktools.stock_financial_hk_report_em"]],
        "reference_answer": "secret",
    }
    plan = mock_plan_for_question(task)
    blob = json.dumps(plan, ensure_ascii=False)
    for key in FORBIDDEN_PLANNER_KEYS:
        assert key not in blob
    assert "aktools.stock_financial_hk_report_em" not in blob
    steps = extract_retrieve_steps(plan)
    assert steps
    assert "aktools." not in steps[0]["search_query"]


def test_hit_recall_mrr_and_complete_route():
    eq = [{"a", "a2"}, {"b"}]
    ranked = ["x", "a2", "b", "y", "z"]
    assert hit_at_k(ranked, {"a", "a2", "b"}, 1) == 0.0
    assert hit_at_k(ranked, {"a", "a2", "b"}, 2) == 1.0
    assert recall_at_k(ranked, eq, 2) == 0.5
    assert recall_at_k(ranked, eq, 3) == 1.0
    assert complete_route_at_k(ranked, eq, 3) == 1.0
    assert mrr_from_rank(2) == 0.5
    metrics = score_ranked_list(
        ranked_ids=ranked,
        equivalents_by_node=eq,
        acceptable_union={"a", "a2", "b"},
    )
    assert metrics["complete_route"]["@3"] == 1.0


def test_efs_and_multi_step_complete_route():
    task = {
        "id": "t1",
        "gold_operations": ["op.a", "op.b"],
        "acceptable_tool_sets": [["op.a", "op.b"], ["op.a2", "op.b2"]],
        "answerability_class": "structured_answerable",
        "domain": "financial",
    }
    eq = equivalents_by_node(task)
    assert eq[0] == {"op.a", "op.a2"}
    assert eq[1] == {"op.b", "op.b2"}
    multi = score_multi_step_route(
        step_ranked_ids=[["op.a2", "x"], ["y", "op.b2"]],
        equivalents_by_node=eq,
        acceptable_union=acceptable_ops_union(task),
    )
    assert multi["complete_route"]["@3"] == 1.0


def test_cross_domain_and_refusal_exclusion_and_disclosure_separation():
    refusal = {
        "id": "mkt_dev_014",
        "answerability_class": "should_refuse_or_clarify",
        "gold_operations": ["aktools.stock_hk_hist"],
        "domain": "market",
    }
    assert is_no_tool_task(refusal) is True
    ref = build_evaluation_reference(refusal)
    assert ref["include_in_tool_recall_denominator"] is False

    disc = {
        "id": "news_dev_006",
        "answerability_class": "evidence_answerable",
        "domain": "news",
        "gold_operations": [
            "hkexnews.list_hkexnews_announcements",
            "hkexnews.download_hkexnews_announcement",
            "hkexnews.parse_hkexnews_announcement",
            "hkexnews.keyword_search",
            "hkexnews.get_announcement_pages",
        ],
    }
    assert is_disclosure_composite_task(disc) is True
    dref = build_evaluation_reference(disc)
    assert dref["include_in_tool_recall_denominator"] is False
    assert dref["is_disclosure_composite"] is True

    xd = {
        "id": "xd_dev_001",
        "domain": "cross_domain",
        "answerability_class": "structured_answerable",
        "gold_operations": [
            "aktools.stock_financial_hk_analysis_indicator_em",
            "aktools.stock_hk_hist",
        ],
        "gold_domains": ["financial", "market"],
    }
    xref = build_evaluation_reference(xd)
    assert "financial" in xref["domain_branches"]
    assert "market" in xref["domain_branches"]


def test_filter_miss_and_distractor_blocked_leakage():
    docs = {
        "op.gold": {
            "document_text": "financial revenue report indicator",
            "document_sha256": "1",
        },
        "op.dist": {
            "document_text": "unrelated weather",
            "document_sha256": "2",
        },
        "op.blocked": {
            "document_text": "blocked tool",
            "document_sha256": "3",
        },
    }
    meta = {
        "op.gold": {
            "domain": "financial",
            "provides": ["revenue"],
            "availability": "runtime",
            "runtime_role": "runtime",
        },
        "op.dist": {
            "domain": "market",
            "provides": [],
            "availability": "runtime",
            "runtime_role": "distractor",
            "is_distractor": True,
        },
        "op.blocked": {
            "domain": "financial",
            "provides": [],
            "availability": "blocked",
            "runtime_role": "blocked",
            "blocked": True,
        },
    }
    index = OfflineEmbeddingIndex(
        documents=docs,
        metadata_by_id=meta,
        runtime_operation_ids=["op.gold", "op.dist"],
    )
    # Domain filter: market step should drop financial gold → filter_miss.
    ranked = rank_operations(
        index=index,
        query="financial revenue report indicator",
        variant="R2",
        step_domain="market",
        required_outputs=["revenue"],
        top_k=5,
        gold_ops_for_filter_diag=["op.gold"],
    )
    assert ranked["filter"]["filter_miss"] is True
    assert "op.blocked" not in ranked["ranked_ids"]

    cov = analyze_catalog_coverage(
        formal_gold_ids=["op.gold"],
        distractor_ids=["op.dist"],
        excluded_ids=["op.blocked"],
        per_query_rankings=[
            {"ranked_ids": ["op.dist", "op.gold"]},
            {"ranked_ids": ["op.gold"]},
            {"ranked_ids": ["op.blocked", "op.gold"]},
        ],
    )
    assert cov["distractors"]["intrusion@1"] == 1
    assert cov["excluded"]["candidate_leakage_events"] == 1
    assert cov["excluded"]["ok"] is False


def test_rank4_to5_not_described():
    ranked = ["x", "y", "z", "gold", "w"]
    assert rank4_to5_not_described(ranked, {"gold"}, describe_n=3, top_k=5) is True
    assert rank4_to5_not_described(["gold", "x"], {"gold"}, describe_n=3) is False


def test_formula_selection_only_uses_70_and_frozen_85_cannot_retune():
    # Higher complete-route@3 on simpler R2 should beat complex R3.
    comparison = compare_formulas(
        development_metrics_by_formula={
            "R3@0.7": {
                "complete_route@3": 0.50,
                "recall@3": 0.90,
                "min_domain_recall@3": 0.80,
                "mrr": 0.70,
            },
            "R2": {
                "complete_route@3": 0.50,
                "recall@3": 0.90,
                "min_domain_recall@3": 0.80,
                "mrr": 0.70,
            },
            "R3@0.9": {
                "complete_route@3": 0.40,
                "recall@3": 0.95,
                "min_domain_recall@3": 0.90,
                "mrr": 0.80,
            },
        }
    )
    assert comparison["selected"]["formula_key"] == "R2"
    lock = {
        "selected_variant": "R2",
        "alpha": None,
        "formula_key": "R2",
    }
    assert_frozen_evaluation_allowed(
        lock=lock, requested_variant="R2", requested_alpha=None
    )
    with pytest.raises(RuntimeError, match="cannot retune"):
        assert_frozen_evaluation_allowed(
            lock=lock, requested_variant="R3", requested_alpha=0.8
        )


def test_cache_identity_fail_closed_and_deterministic_rerun(tmp_path: Path):
    data = load_hk155_locked(ROOT)
    a = question_level_cache_identity(
        dataset_identities=data["identities"],
        operation_documents_sha256="doc",
        catalog_fingerprint="cat",
        embedding_model="mock-hash-embedding-v1",
        embedding_provider_host="local_hash",
    )
    b = question_level_cache_identity(
        dataset_identities=data["identities"],
        operation_documents_sha256="doc",
        catalog_fingerprint="cat",
        embedding_model="mock-hash-embedding-v1",
        embedding_provider_host="local_hash",
    )
    assert a["identity_sha256"] == b["identity_sha256"]
    c = step_retrieval_cache_identity(
        planner_cache_sha256="p",
        operation_documents_sha256="doc",
        catalog_fingerprint="cat",
        ranking_variant="R3",
        alpha=0.8,
    )
    d = step_retrieval_cache_identity(
        planner_cache_sha256="p",
        operation_documents_sha256="doc",
        catalog_fingerprint="cat",
        ranking_variant="R3",
        alpha=0.7,
    )
    assert c["identity_sha256"] != d["identity_sha256"]


def test_offline_runner_mock_pipeline_no_network(tmp_path: Path):
    out = tmp_path / "retrieval_audit"
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=out, mock=True, dry_run=False
    )
    # Catalog stats sanity (42 / 39 / 18 / 21 / 3 expected band).
    stats = runner.catalog["stats"]
    assert stats["unique_operation_ids"] == 42
    assert stats["runtime_exposed_unique"] == 39
    assert stats["formal_gold_eligible_unique"] == 18
    assert stats["distractor_runtime_unique"] == 21
    assert stats["blocked_or_rejected_unique"] == 3

    runner.run_question_level()
    runner.run_planner_only()
    runner.run_rank_from_planner_cache()
    runner.run_catalog_coverage()
    runner.run_select_formula()
    runner.run_evaluate_frozen_formula()
    runner.run_report_all()

    assert (out / "question_level" / "per_query_rankings.jsonl").exists()
    assert (out / "planner_cache" / "planner_outputs.jsonl").exists()
    assert (out / "ranking_variants" / "per_step_rankings.jsonl").exists()
    assert (out / "ranking_variants" / "formula_comparison.json").exists()
    assert (out / "formula_selection" / "run_metadata.json").exists()
    assert (out / "all155_report" / "all155_report.json").exists()
    assert (out / "run_metadata.json").exists()
    # Mock cannot write formal lock.
    assert not (out / "formula_selection" / "formula_lock.json").exists()

    # Deterministic rerun: same identity hashes.
    q1 = json.loads((out / "question_level" / "cache_identity.json").read_text())
    runner2 = RetrievalAuditRunner(
        root=ROOT, output_dir=out, mock=True, dry_run=False
    )
    runner2.run_question_level()
    q2 = json.loads((out / "question_level" / "cache_identity.json").read_text())
    assert q1["identity_sha256"] == q2["identity_sha256"]


def test_dry_run_and_ambiguous_network_refused(tmp_path: Path):
    from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import (
        LivePathError,
    )

    with pytest.raises(LivePathError, match="allow_network"):
        RetrievalAuditRunner(
            root=ROOT, output_dir=tmp_path / "x", allow_network=True
        )
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=tmp_path / "dry", dry_run=True
    )
    meta = runner.run_question_level()
    assert meta["dry_run"] is True
