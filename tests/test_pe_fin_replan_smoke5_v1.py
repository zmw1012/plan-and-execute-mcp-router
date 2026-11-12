"""Offline tests for PE-Fin RePlan engineering Smoke-5 (no live API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
    UnifiedHkPeFinLiteRunner,
    assert_output_dir_safe,
    load_replan_smoke5_manifest,
    load_replan_smoke5_tasks,
)
from hk_equity_mcp.pe_fin_replan import PE_FIN_REPLAN_METHOD_VERSION, PE_FIN_REPLAN_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_replan_smoke5_v1.json"
DIAG20 = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"
EXPECTED_IDS = [
    "mkt_dev_003",
    "news_dev_004",
    "fin_cf_019",
    "xd_dev_008",
    "xd_dev_004",
]
DIAG20_FILE_SHA = "7076a39ffc6b896ddc2bd2153c8c8d4f9a6dca268e805e3bf239867077cad1a9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replan_smoke5_manifest_contract_and_hashes():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert man["manifest_id"] == "hk_pe_fin_replan_smoke5_v1"
    assert man["role"] == "live_engineering_smoke_only"
    assert man["explicitly_not_accuracy"] is True
    assert man["source_manifest"] == "hk_pe_fin_lite_eea_diagnostic20_v1"
    assert man["cannot_replace_diagnostic20"] is True
    assert man["cannot_replace_hk155"] is True
    assert man["posthoc_exploratory"] is True
    assert man["ordered_task_ids"] == EXPECTED_IDS
    oid_hash = hashlib.sha256(
        json.dumps(EXPECTED_IDS, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert man["ordered_task_ids_hash"] == oid_hash
    core = {k: v for k, v in man.items() if k != "manifest_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert man["manifest_sha256"] == expected_sha
    # All IDs must be subset of frozen Diagnostic-20
    d20 = json.loads(DIAG20.read_text(encoding="utf-8"))
    assert set(EXPECTED_IDS).issubset(set(d20["ordered_task_ids"]))
    assert _sha(DIAG20) == DIAG20_FILE_SHA


def test_loader_order_and_development_origin():
    man = load_replan_smoke5_manifest(ROOT)
    assert man["ordered_task_ids"] == EXPECTED_IDS
    tasks = load_replan_smoke5_tasks(ROOT)
    assert [t["task_id"] for t in tasks] == EXPECTED_IDS
    assert all(t.get("source_split") == "development" for t in tasks)


def test_runner_dry_run_identity_replan_smoke5(tmp_path: Path):
    out = tmp_path / "smoke5_dry"
    runner = UnifiedHkPeFinLiteRunner(
        output_dir=out,
        split="exploratory_all",
        replan_smoke5=True,
        replan_enabled=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta = runner.run()
    assert meta["subset_kind"] == "replan_smoke5"
    assert meta["replan_smoke5"] is True
    assert meta["question_count"] == 5
    assert meta["method_version"] == PE_FIN_REPLAN_METHOD_VERSION
    assert meta["protocol_version"] == PE_FIN_REPLAN_PROTOCOL_VERSION
    assert meta["condition"] == "pe_fin_replan"
    assert meta["dry_run"] is True
    assert meta["mock"] is False
    assert meta["replan_smoke5_manifest_hash"]
    assert meta["subset_ordered_id_hash"] == json.loads(MANIFEST.read_text())[
        "ordered_task_ids_hash"
    ]


def test_resume_refuses_smoke5_into_diagnostic20(tmp_path: Path):
    out = tmp_path / "mixed"
    out.mkdir()
    # Fake existing Diagnostic-20 replan metadata
    existing = {
        "condition": "pe_fin_replan",
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
        "architecture": "plan_execute_replan",
        "subset_kind": "diagnostic20_replan",
        "mock": False,
        "dry_run": False,
    }
    (out / "run_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(existing)
    current["subset_kind"] = "replan_smoke5"
    with pytest.raises(RuntimeError, match="subset_kind mismatch"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta=current,
        )


def test_resume_refuses_mock_into_live(tmp_path: Path):
    out = tmp_path / "mock_to_live"
    out.mkdir()
    existing = {
        "condition": "pe_fin_replan",
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
        "architecture": "plan_execute_replan",
        "subset_kind": "replan_smoke5",
        "mock": True,
        "dry_run": False,
    }
    (out / "run_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(existing)
    current["mock"] = False
    with pytest.raises(RuntimeError, match="mock"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta=current,
        )


def test_resume_refuses_v1_2_method_into_v1_3(tmp_path: Path):
    out = tmp_path / "v12_to_v13"
    out.mkdir()
    existing = {
        "condition": "pe_fin_replan",
        "method_version": "pe_fin_replan_v1_2",
        "protocol_version": "planner_executor_replanner_contract_v2",
        "architecture": "plan_execute_replan",
        "subset_kind": "replan_smoke5",
        "mock": False,
        "dry_run": False,
        "execution_backend": "real_mcp",
        "generation_provider": "x",
        "generation_base_url_host": "h",
        "generation_model": "m",
        "embedding_provider": "x",
        "embedding_base_url_host": "h",
        "embedding_model": "e",
        "embedding_dimension": 1,
        "system_prompt_hash": "a",
        "schema_hash": "b",
        "capability_profile_hash": "c",
        "operation_metadata_hash": "d",
        "catalog_fingerprint": "e",
        "embedding_index_identity_sha256": "f",
        "operation_documents_sha256": "g",
        "retrieval_formula_name": "R5",
        "retrieval_formula_version": "1",
        "retrieval_formula_hash": "h",
        "lexical_weight": 0.12,
        "domain_hard_filter": False,
        "candidate_lock_sha256": "i",
        "dataset_split": "exploratory_all",
        "subset_manifest_hash": "j",
        "subset_ordered_id_hash": "k",
        "diagnostic_manifest_hash": None,
        "smoke5_manifest_hash": None,
        "replan_smoke5_manifest_hash": "l",
        "temperature": 0.0,
        "max_tokens": 1,
        "planner_max_tokens": 1,
        "json_mode": True,
        "limits": {},
        "max_total_llm_calls_per_task": 16,
        "checkpoint_trace_contract_version": "v",
    }
    (out / "run_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(existing)
    current["method_version"] = PE_FIN_REPLAN_METHOD_VERSION
    current["protocol_version"] = PE_FIN_REPLAN_PROTOCOL_VERSION
    with pytest.raises(RuntimeError, match="identity mismatch|method_version"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta=current,
        )


def test_cannot_combine_replan_smoke5_and_diagnostic20():
    with pytest.raises(RuntimeError, match="Cannot combine"):
        UnifiedHkPeFinLiteRunner(
            output_dir=ROOT / "outputs" / "_tmp_combo",
            split="exploratory_all",
            diagnostic20_replan=True,
            replan_smoke5=True,
            dry_run=True,
            allow_dirty=True,
            root=ROOT,
        )
