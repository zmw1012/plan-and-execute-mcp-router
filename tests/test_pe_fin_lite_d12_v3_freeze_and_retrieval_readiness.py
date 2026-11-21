"""Offline freeze + retrieval readiness invariants (no network / no method mutation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.operation_documents import (
    build_all_operation_documents,
)
from hk_equity_mcp.experiment.unified_runner import catalog_fingerprint
from hk_equity_mcp.pe_fin_lite import PE_FIN_LITE_METHOD_VERSION
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    build_index_identity,
    identity_digest,
    index_status,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "pe_fin_lite_diagnostic12_checkpoint_live_v3"
AUDIT = RUN / "final_checkpoint_audit_v1"
FREEZE = ROOT / "docs" / "pe_fin_lite_diagnostic12_checkpoint_live_v3_final"
READY = ROOT / "docs" / "pe_fin_lite_retrieval_all155_readiness_v1"
CACHE = ROOT / ".cache" / "pe_fin_lite_operation_embeddings_v1"

EXPECTED_RAW = {
    "predictions.jsonl": "e1420f3c4d4bbb8832cc1bdcdc5e52bb2e80193e35f8fb1d9e40edbdb7ad573c",
    "run_metadata.json": "b024cc2b35e50c8c1405e011857546d7b4cf7c58b0088a0de92c918215e0d6ba",
    "error_summary.json": "5d7955c3c010ca2561531c8e85438683493b07bb44727a5561b4e4bbe17fe3d8",
    "cost_summary.json": "00f72ac9966ce3a273a7a644805f02f8f9a6038f0ee1a20bfa712751525b5aee",
    "grading_summary.json": "bb4b3d33cea466f9b54c5143f02c67a69c31ce885a91428e3045ebbcc9a09c7f",
    "per_task_metrics.csv": "85179fe73da6051f322b1a9e60c6df74cf8d24d32fe6724f8c0fc10f9e3feef1",
}


@pytest.mark.skipif(not RUN.exists(), reason="D12 v3 run outputs not present")
def test_d12_v3_raw_hashes_unchanged_and_freeze_docs_present():
    assert PE_FIN_LITE_METHOD_VERSION == "pe_fin_lite_v1.2"
    meta = json.loads((RUN / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["run_status"] == "completed"
    assert meta["method_version"] == "pe_fin_lite_v1.2"
    assert meta["git_commit"].startswith("baf5724")
    assert meta["git_dirty"] is False
    assert meta["full_count"] == 5
    assert meta["partial_count"] == 5
    assert meta["failed_count"] == 2

    preds = [
        json.loads(line)
        for line in (RUN / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds) == 12
    assert len({p.get("task_id") or p.get("id") for p in preds}) == 12

    for name, digest in EXPECTED_RAW.items():
        blob = (RUN / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest

    audit_hashes = json.loads((AUDIT / "raw_hashes.json").read_text(encoding="utf-8"))
    assert audit_hashes["unchanged"] is True
    summary = json.loads((AUDIT / "checkpoint_summary.json").read_text(encoding="utf-8"))
    assert summary["engineering_pass"] is True
    assert summary["explicitly_not_formal_accuracy"] is True
    assert summary["raw_hashes_unchanged"] is True

    assert (FREEZE / "README.md").exists()
    assert (FREEZE / "method_freeze_declaration.md").exists()
    assert (FREEZE / "v1_v2_v3_comparison.md").exists()
    ver = json.loads(
        (FREEZE / "raw_hashes_freeze_verification.json").read_text(encoding="utf-8")
    )
    assert ver["raw_hashes_unchanged"] is True
    assert ver["raw_not_mutated"] is True


@pytest.mark.skipif(not RUN.exists(), reason="D12 v3 run outputs not present")
def test_xd_dev_002_reason_limitation_retained():
    preds = [
        json.loads(line)
        for line in (RUN / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = next(p for p in preds if (p.get("task_id") or p.get("id")) == "xd_dev_002")
    art = (row.get("reason_artifacts") or [None])[0]
    assert isinstance(art, dict)
    assert art.get("reason_parse_failed") is True
    assert art.get("runtime_recovered") is False
    assert art.get("initial_parse_error") == "$: unexpected property 'basic_eps'"
    retained = json.loads(
        (FREEZE / "xd_dev_002_retained_limitations.json").read_text(encoding="utf-8")
    )
    assert retained["reason_stage"]["initial_parse_error"].startswith("$: unexpected property")


def test_retrieval_readiness_docs_and_current_index_identity():
    assert (READY / "README.md").exists()
    assert (READY / "future_commands.md").exists()
    inv = json.loads((READY / "readiness_inventory.json").read_text(encoding="utf-8"))
    assert inv["zero_network"] is True
    assert inv["current"]["operation_documents_sha256"].startswith("5989196c")
    assert inv["caches"]["planner_only_all155_live_v1"]["reusable"] is False
    assert inv["caches"]["diagnostic10_planner_v3"]["cannot_resume_all155"] is True
    assert inv["caches"]["step_query_embeddings"]["exists"] is False
    assert inv["caches"]["formula_lock_formal"]["exists"] is False
    assert not (ROOT / "outputs" / "pe_fin_lite_retrieval_all155_planner_v2").exists()

    bundle = build_unified_catalog(root=ROOT)
    docs = build_all_operation_documents(bundle)
    assert docs["operation_documents_sha256"].startswith("5989196c")
    cat_fp = catalog_fingerprint(bundle.stats, bundle.source_catalogs)
    ident = build_index_identity(
        registry=bundle.registry,
        catalog_fingerprint=cat_fp,
        embedding_model="text-embedding-3-small",
        embedding_provider_host="api.zhizengzeng.com",
    )
    digest = identity_digest(ident)
    assert digest == inv["current"]["embedding_identity_sha256"]
    assert digest.startswith("ac4c109b")
    if CACHE.exists():
        st = index_status(registry=bundle.registry, identity=ident, cache_dir=CACHE)
        # Partial/unrelated caches may exist in a worktree; only assert completeness
        # when a matching complete index is present.
        if st.identity_sha256 != digest or not st.complete:
            pytest.skip(
                "Local embedding cache missing a complete index for current identity "
                f"(identity={st.identity_sha256}, complete={st.complete}, "
                f"want={digest}). Set MCP_ROUTER_HK_ARTIFACT_ROOT to a tree with "
                "the frozen pe_fin_lite_operation_embeddings_v1 index."
            )
        assert st.complete is True
        assert st.identity_sha256 == digest
