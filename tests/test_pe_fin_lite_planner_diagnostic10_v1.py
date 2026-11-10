"""Tests for Planner protocol Diagnostic-10 (offline / fake only)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    PLANNER_ONLY_PROTOCOL_VERSION,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    assert_identity_match,
    load_hk155_locked,
    planner_only_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import LivePathError
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_diagnostic10 import (
    DIAGNOSTIC10_CACHE_KIND,
    DIAGNOSTIC10_ROLE,
    EXPECTED_N,
    SOURCE_PLANNER_V1_CACHE_SHA256,
    assert_not_cross_resume,
    build_planner_protocol_diagnostic,
    load_planner_diagnostic10_manifest,
    planner_diagnostic10_cache_identity,
    select_diagnostic10_tasks,
    strip_task_for_diagnostic_runtime,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    FORBIDDEN_PLANNER_KEYS,
    run_planner_only,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (
    RetrievalAuditRunner,
)
from hk_equity_mcp.pe_fin_lite.identity import combined_schema_hash
from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import plan_canonicalizer_hash
from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_identity_order_and_hk155_membership():
    man = load_planner_diagnostic10_manifest(ROOT)
    assert man["role"] == DIAGNOSTIC10_ROLE
    assert len(man["ordered_task_ids"]) == EXPECTED_N
    assert man["payload"]["source_planner_v1_cache_sha256"] == SOURCE_PLANNER_V1_CACHE_SHA256
    expected = [
        "news_dev_003",
        "xd_ht_003",
        "xd_dev_003",
        "mkt_dev_007",
        "mkt_ht_007",
        "news_ht_011",
        "xd_ht_006",
        "xd_dev_002",
        "fin_ht_024",
        "fin_cf_029",
    ]
    assert man["ordered_task_ids"] == expected
    data = load_hk155_locked(ROOT)
    hk_ids = {str(t.get("task_id") or t.get("id")) for t in data["tasks"]}
    assert set(expected).issubset(hk_ids)
    blob = json.dumps(man["payload"], ensure_ascii=False)
    for key in FORBIDDEN_PLANNER_KEYS:
        assert key not in blob


def test_no_gold_in_runtime_strip():
    dirty = {
        "task_id": "news_dev_003",
        "question": "q?",
        "domain": "news",
        "gold_operations": ["secret.op"],
        "reference_answer": "42",
        "gold_values": {"x": 1},
    }
    clean = strip_task_for_diagnostic_runtime(dirty)
    assert "gold_operations" not in clean
    assert "reference_answer" not in clean
    assert clean["question"] == "q?"


def test_mock_writes_10_rows_and_resume(tmp_path: Path):
    out = tmp_path / "diag10_mock"
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=out, mock=True, planner_diagnostic10=True
    )
    meta = runner.run_planner_only()
    path = out / "planner_cache" / "planner_outputs.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows) == 10
    assert len({r["task_id"] for r in rows}) == 10
    identity = json.loads(
        (out / "planner_cache" / "planner_cache_identity.json").read_text()
    )
    assert identity["kind"] == DIAGNOSTIC10_CACHE_KIND
    assert identity["planner_only_protocol_version"] == PLANNER_ONLY_PROTOCOL_VERSION
    diag = json.loads(
        (out / "planner_cache" / "planner_protocol_diagnostic.json").read_text()
    )
    assert diag["expected"] == 10
    assert diag["processed"] == 10
    assert diag["unique"] == 10
    assert diag["explicitly_not_formal_accuracy"] is True
    assert diag["complete_for_diagnostic10"] is True
    # Mock must not fabricate a live engineering_pass.
    assert diag["engineering_pass"] is False
    assert meta["planner_protocol_diagnostic"]["ok"] == 10

    # Resume should not duplicate.
    meta2 = runner.run_planner_only()
    rows2 = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows2) == 10
    assert meta2["planner_only_summary"]["newly_written"] == 0


def test_diagnostic_and_all155_cannot_cross_resume(tmp_path: Path):
    data = load_hk155_locked(ROOT)
    man = load_planner_diagnostic10_manifest(ROOT)
    diag_id = planner_diagnostic10_cache_identity(
        diagnostic_manifest=man,
        planner_prompt_hash=pe_fin_lite_prompt_hash(),
        planner_schema_hash=combined_schema_hash(),
        plan_canonicalizer_hash=plan_canonicalizer_hash(),
        capability_profile_hash="x",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock_plan_v1",
        planner_only_protocol_version=PLANNER_ONLY_PROTOCOL_VERSION,
    )
    all_id = planner_only_cache_identity(
        dataset_identities=data["identities"],
        capability_profile_hash="x",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock_plan_v1",
    )
    with pytest.raises(RuntimeError, match="kind mismatch"):
        assert_not_cross_resume(diag_id, all_id)
    with pytest.raises(RuntimeError, match="kind mismatch"):
        assert_not_cross_resume(all_id, diag_id)

    # Also via run_planner_only resume path.
    out = tmp_path / "cross"
    out.mkdir(parents=True)
    (out / "planner_cache_identity.json").write_text(
        json.dumps(all_id, indent=2) + "\n", encoding="utf-8"
    )
    (out / "planner_outputs.jsonl").write_text("", encoding="utf-8")
    selected = select_diagnostic10_tasks(root=ROOT, all_tasks=data["tasks"])
    with pytest.raises(RuntimeError, match="kind mismatch"):
        run_planner_only(
            root=ROOT,
            tasks=selected["tasks"],
            output_dir=out,
            cache_identity=diag_id,
            mock=True,
            resume=True,
            capability_profile={},
        )


def test_identity_drift_rejected(tmp_path: Path):
    man = load_planner_diagnostic10_manifest(ROOT)
    a = planner_diagnostic10_cache_identity(
        diagnostic_manifest=man,
        planner_prompt_hash="aaa",
        planner_schema_hash=combined_schema_hash(),
        plan_canonicalizer_hash=plan_canonicalizer_hash(),
        capability_profile_hash="p",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock_plan_v1",
        planner_only_protocol_version=PLANNER_ONLY_PROTOCOL_VERSION,
    )
    b = planner_diagnostic10_cache_identity(
        diagnostic_manifest=man,
        planner_prompt_hash="bbb",
        planner_schema_hash=combined_schema_hash(),
        plan_canonicalizer_hash=plan_canonicalizer_hash(),
        capability_profile_hash="p",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock_plan_v1",
        planner_only_protocol_version=PLANNER_ONLY_PROTOCOL_VERSION,
    )
    with pytest.raises(RuntimeError, match="Identity mismatch"):
        assert_identity_match(a, b, label="planner_only_cache")


def test_output_isolation_default_prefix():
    runner = RetrievalAuditRunner(
        root=ROOT, mock=True, planner_diagnostic10=True, output_dir=None
    )
    assert "diagnostic10" in str(runner.output_dir)


def test_diagnostic_cannot_formal_lock(tmp_path: Path):
    out = tmp_path / "diag_formal"
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=out, mock=True, planner_diagnostic10=True
    )
    runner.run_planner_only()
    with pytest.raises(LivePathError, match="Diagnostic-10"):
        runner.run_select_formula(formal=True)
    with pytest.raises(LivePathError, match="Diagnostic-10"):
        runner._require_planner_for_step_stage(stage="rank", formal=False)


def test_failed_task_still_writes_structured_row(tmp_path: Path):
    class FailLLM:
        def complete(self, messages, **kwargs):
            return {"content": "", "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}}

    data = load_hk155_locked(ROOT)
    selected = select_diagnostic10_tasks(root=ROOT, all_tasks=data["tasks"])
    man = selected["manifest"]
    identity = planner_diagnostic10_cache_identity(
        diagnostic_manifest=man,
        planner_prompt_hash=pe_fin_lite_prompt_hash(),
        planner_schema_hash=combined_schema_hash(),
        plan_canonicalizer_hash=plan_canonicalizer_hash(),
        capability_profile_hash="x",
        generation_provider="openai_compatible",
        generation_host="api.zhizengzeng.com",
        generation_model="deepseek-v4.1-flash",
        planner_only_protocol_version=PLANNER_ONLY_PROTOCOL_VERSION,
    )
    out = tmp_path / "fail_rows"
    # Only first task to keep test fast.
    summary = run_planner_only(
        root=ROOT,
        tasks=selected["tasks"][:1],
        output_dir=out,
        cache_identity=identity,
        llm=FailLLM(),
        live=True,
        mock=False,
        resume=False,
        capability_profile={},
    )
    rows = [
        json.loads(l)
        for l in (out / "planner_outputs.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["ok"] is False
    assert rows[0]["error"]
    assert summary["n_errors"] == 1
    diag = build_planner_protocol_diagnostic(
        rows=rows,
        manifest=man,
        manifest_task_meta_by_id=selected["manifest_task_meta_by_id"],
        cache_identity=identity,
    )
    assert diag["failed"] == 1
    assert diag["engineering_pass"] is False


def test_cli_diagnostic10_mock_and_rejects_combo(tmp_path: Path):
    script = ROOT / "scripts" / "run_pe_fin_lite_retrieval_audit.py"
    spec = importlib.util.spec_from_file_location("diag10_cli", script)
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    rc = cli.main(
        [
            "--root",
            str(ROOT),
            "--output-dir",
            str(tmp_path / "cli_mock"),
            "--mock",
            "--planner-only",
            "--planner-diagnostic10",
        ]
    )
    assert rc == 0
    assert (
        tmp_path / "cli_mock" / "planner_cache" / "planner_protocol_diagnostic.json"
    ).exists()

    rc2 = cli.main(
        [
            "--root",
            str(ROOT),
            "--output-dir",
            str(tmp_path / "cli_bad"),
            "--mock",
            "--planner-diagnostic10",
            "--question-level",
        ]
    )
    assert rc2 == 2


def test_dry_run_diagnostic10(tmp_path: Path):
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=tmp_path / "dry", dry_run=True, planner_diagnostic10=True
    )
    meta = runner.run_planner_only()
    assert meta["dry_run"] is True
    rows = [
        json.loads(l)
        for l in (tmp_path / "dry" / "planner_cache" / "planner_outputs.jsonl")
        .read_text()
        .splitlines()
        if l.strip()
    ]
    assert len(rows) == 10
    assert all(r.get("dry_run") for r in rows)
