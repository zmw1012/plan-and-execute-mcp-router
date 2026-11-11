"""Failed-row Planner cache retry tests (fake LLM; zero network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    assert_formal_formula_lock_allowed,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
    IncompletePlannerCacheError,
    assert_planner_cache_for_step_pipeline,
    audit_planner_cache_zero_network,
    summarize_planner_rows,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_retry import (
    PlannerCacheRetryError,
    aggregate_row_costs,
    derive_failed_task_ids,
    file_sha256,
    freeze_attempt_snapshot,
    merge_canonical_rows,
    retry_failed_planner_cache,
    rows_by_task_id,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    mock_plan_for_question,
    extract_retrieve_steps,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_OUT = ROOT / "outputs" / "pe_fin_lite_retrieval_all155_planner_v3"
REAL_PLANNER = REAL_OUT / "planner_cache" / "planner_outputs.jsonl"
EXPECTED_REAL_SHA = (
    "b740c07924d4a46e79fdb2a27c1e65fd3dd5ac929649f0f7d310e717ad0234f6"
)


def _row(
    tid: str,
    *,
    ok: bool = True,
    llm_calls: int = 1,
    tokens_in: int = 10,
    tokens_out: int = 5,
    error: str | None = None,
) -> Dict[str, Any]:
    task = {"task_id": tid, "domain": "market", "question": f"Q for {tid}"}
    plan = mock_plan_for_question(task) if ok else None
    return {
        "task_id": tid,
        "ok": ok,
        "mock": True,
        "plan": plan,
        "retrieve_steps": extract_retrieve_steps(plan or {}),
        "error": error,
        "error_type": None if ok else "empty_content",
        "repair_count": 0 if ok else 1,
        "llm_calls": llm_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "usage": {
            "calls": llm_calls,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
        "planner_only_protocol_version": "pe_fin_lite_planner_only_v3",
        "no_embedding": True,
        "no_mcp": True,
        "no_binder": True,
    }


def _identity(**overrides: Any) -> Dict[str, Any]:
    base = {
        "kind": "planner_only_cache",
        "manifest_sha256": "m",
        "ordered_task_ids_hash": "o",
        "question_identity_hash": "q",
        "planner_prompt_hash": "prompt",
        "planner_schema_hash": "schema",
        "capability_profile_hash": "cap",
        "plan_canonicalizer_hash": "canon",
        "scope_contract_hash": "scope",
        "generation_provider": "openai_compatible",
        "generation_host": "api.zhizengzeng.com",
        "generation_model": "deepseek-v4.1-flash",
        "temperature": 0.0,
        "max_tokens": 4096,
        "planner_max_tokens": 4096,
        "planner_only_protocol_version": "pe_fin_lite_planner_only_v3",
        "no_binder": True,
        "no_mcp": True,
        "no_embedding_in_planner_stage": True,
        "no_gold_in_planner": True,
    }
    base.update(overrides)
    blob = json.dumps(base, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    base["identity_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return base


def _write_cache(
    out: Path,
    *,
    ordered_ids: List[str],
    failed_id: str,
    identity: Dict[str, Any],
    fail_calls: int = 2,
) -> str:
    cache = out / "planner_cache"
    cache.mkdir(parents=True)
    rows = []
    for tid in ordered_ids:
        if tid == failed_id:
            rows.append(
                _row(tid, ok=False, llm_calls=fail_calls, tokens_in=20, tokens_out=0, error="empty content")
            )
        else:
            rows.append(_row(tid, ok=True, llm_calls=1))
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (cache / "planner_outputs.jsonl").write_text(text, encoding="utf-8")
    (cache / "planner_cache_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    costs = aggregate_row_costs(rows)
    (cache / "planner_only_summary.json").write_text(
        json.dumps(
            {
                "n_tasks": len(ordered_ids),
                "n_rows": len(rows),
                "n_errors": 1,
                "planner_cache_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "identity": identity,
                "errors": [{"task_id": failed_id, "error": "empty content"}],
                "original_costs_seed": costs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cache / "error_summary.json").write_text(
        json.dumps({"errors": [{"task_id": failed_id, "error": "empty content"}]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(text.encode()).hexdigest()


def test_derive_failed_ids_no_hardcode():
    rows = [_row("a"), _row("b", ok=False, error="x"), _row("c", ok=False, error="y")]
    assert derive_failed_task_ids(rows) == ["b", "c"]
    assert derive_failed_task_ids(rows, ordered_task_ids=["c", "a", "b"]) == ["c", "b"]
    # Ensure source does not mention a specific production task id constant.
    import hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_retry as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "mkt_dev_012" not in src


def test_duplicate_ids_refused():
    rows = [_row("a"), _row("a")]
    with pytest.raises(PlannerCacheRetryError, match="duplicate"):
        rows_by_task_id(rows)


def test_retry_success_promotes_155_and_preserves_ok_rows(tmp_path: Path):
    ordered = [f"t{i:03d}" for i in range(155)]
    failed = "t042"
    identity = _identity()
    src_sha = _write_cache(
        tmp_path, ordered_ids=ordered, failed_id=failed, identity=identity, fail_calls=2
    )
    tasks_by_id = {
        tid: {"task_id": tid, "domain": "market", "question": f"Q {tid}"}
        for tid in ordered
    }
    before_rows = [
        json.loads(l)
        for l in (tmp_path / "planner_cache/planner_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]
    before_by = rows_by_task_id(before_rows)
    original_costs = aggregate_row_costs(before_rows)
    assert original_costs["llm_calls"] == 154 * 1 + 2

    calls = {"n": 0}

    def fake_runner(*, task: Dict[str, Any]) -> Dict[str, Any]:
        calls["n"] += 1
        tid = task["task_id"]
        assert tid == failed  # only failed task invoked
        row = _row(tid, ok=True, llm_calls=1, tokens_in=3, tokens_out=4)
        row["retry"] = True
        return row

    report = retry_failed_planner_cache(
        output_dir=tmp_path,
        tasks_by_id=tasks_by_id,
        ordered_task_ids=ordered,
        expected_identity=identity,
        task_runner=fake_runner,
    )
    assert calls["n"] == 1
    assert report["promoted"] is True
    assert report["complete_155_ok"] is True
    assert report["formal_step_pipeline_allowed"] is True
    assert report["combined_costs"]["llm_calls"] == original_costs["llm_calls"] + 1

    # attempt snapshot preserved with original sha
    snap = tmp_path / "planner_cache_attempt1" / "planner_outputs.jsonl"
    assert file_sha256(snap) == src_sha
    assert (tmp_path / "planner_cache_attempt1" / "raw_hashes.json").exists()

    after_rows = [
        json.loads(l)
        for l in (tmp_path / "planner_cache/planner_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]
    assert len(after_rows) == 155
    assert len({r["task_id"] for r in after_rows}) == 155
    assert all(r.get("ok") for r in after_rows)
    after_by = rows_by_task_id(after_rows)
    for tid in ordered:
        if tid == failed:
            assert after_by[tid]["ok"] is True
            assert after_by[tid].get("retry") is True
            continue
        assert after_by[tid] == before_by[tid]

    summary = summarize_planner_rows(after_rows)
    assert summary["complete"] is True
    assert_planner_cache_for_step_pipeline(
        rows=after_rows, allow_incomplete=False, formal=True, stage="embed"
    )
    audit = audit_planner_cache_zero_network(
        output_dir=tmp_path, expected_task_ids=ordered
    )
    assert audit["complete_155_ok"] is True
    assert audit["formal_step_pipeline_allowed"] is True
    assert audit["failed"] == 0


def test_retry_failure_leaves_canonical_untouched(tmp_path: Path):
    ordered = [f"t{i:03d}" for i in range(155)]
    failed = "t007"
    identity = _identity()
    src_sha = _write_cache(
        tmp_path, ordered_ids=ordered, failed_id=failed, identity=identity
    )

    def fail_runner(*, task: Dict[str, Any]) -> Dict[str, Any]:
        return _row(task["task_id"], ok=False, llm_calls=2, error="empty content again")

    with pytest.raises(IncompletePlannerCacheError, match="still_failed"):
        retry_failed_planner_cache(
            output_dir=tmp_path,
            tasks_by_id={
                tid: {"task_id": tid, "domain": "market", "question": f"Q {tid}"}
                for tid in ordered
            },
            ordered_task_ids=ordered,
            expected_identity=identity,
            task_runner=fail_runner,
        )
    assert (
        file_sha256(tmp_path / "planner_cache" / "planner_outputs.jsonl") == src_sha
    )
    assert file_sha256(tmp_path / "planner_cache_attempt1" / "planner_outputs.jsonl") == src_sha
    # retry artifacts exist
    assert (tmp_path / "planner_cache_retry" / "latest" / "retry_outputs.jsonl").exists()


def test_identity_and_prompt_mismatch_refused(tmp_path: Path):
    ordered = [f"t{i:03d}" for i in range(155)]
    identity = _identity()
    _write_cache(tmp_path, ordered_ids=ordered, failed_id="t001", identity=identity)
    bad = _identity(planner_prompt_hash="OTHER")
    with pytest.raises((PlannerCacheRetryError, RuntimeError), match="mismatch|Identity"):
        retry_failed_planner_cache(
            output_dir=tmp_path,
            tasks_by_id={
                tid: {"task_id": tid, "domain": "market", "question": "q"}
                for tid in ordered
            },
            ordered_task_ids=ordered,
            expected_identity=bad,
            dry_run=True,
        )


def test_merge_refuses_duplicates_and_incomplete():
    ordered = [f"t{i:03d}" for i in range(155)]
    by = {tid: _row(tid) for tid in ordered}
    by["t000"]["ok"] = False
    with pytest.raises(PlannerCacheRetryError, match="incomplete|failed"):
        merge_canonical_rows(
            ordered_task_ids=ordered,
            original_rows_by_id=by,
            retry_success_by_id={},  # no successful retry
        )
    # duplicate ordered ids
    with pytest.raises(PlannerCacheRetryError, match="duplicate"):
        merge_canonical_rows(
            ordered_task_ids=ordered + ["t000"],
            original_rows_by_id=by,
            retry_success_by_id={"t000": _row("t000")},
        )


def test_incomplete_still_refuses_formal_lock_gate():
    rows = [_row(f"t{i:03d}") for i in range(154)] + [
        _row("t154", ok=False, error="x")
    ]
    # pad to wrong count on purpose
    with pytest.raises(IncompletePlannerCacheError):
        assert_planner_cache_for_step_pipeline(
            rows=rows, allow_incomplete=False, formal=True, stage="select"
        )
    with pytest.raises(RuntimeError, match="mock"):
        assert_formal_formula_lock_allowed(
            embedding_backend="real_text_embedding_3_small", mock=True
        )


def test_atomic_promotion_and_snapshot_idempotent(tmp_path: Path):
    ordered = [f"t{i:03d}" for i in range(155)]
    identity = _identity()
    src_sha = _write_cache(
        tmp_path, ordered_ids=ordered, failed_id="t010", identity=identity
    )
    snap1 = freeze_attempt_snapshot(
        planner_cache_dir=tmp_path / "planner_cache",
        attempt_dir=tmp_path / "planner_cache_attempt1",
        expected_outputs_sha256=src_sha,
    )
    snap2 = freeze_attempt_snapshot(
        planner_cache_dir=tmp_path / "planner_cache",
        attempt_dir=tmp_path / "planner_cache_attempt1",
        expected_outputs_sha256=src_sha,
    )
    assert snap1["planner_outputs_sha256"] == snap2["planner_outputs_sha256"] == src_sha


def test_no_embedding_mcp_flags_on_retry_rows(tmp_path: Path):
    ordered = [f"t{i:03d}" for i in range(155)]
    identity = _identity()
    _write_cache(tmp_path, ordered_ids=ordered, failed_id="t003", identity=identity)

    def runner(*, task: Dict[str, Any]) -> Dict[str, Any]:
        row = _row(task["task_id"], ok=True)
        row["no_embedding"] = True
        row["no_mcp"] = True
        row["no_binder"] = True
        return row

    report = retry_failed_planner_cache(
        output_dir=tmp_path,
        tasks_by_id={
            tid: {"task_id": tid, "domain": "market", "question": "q"}
            for tid in ordered
        },
        ordered_task_ids=ordered,
        expected_identity=identity,
        task_runner=runner,
    )
    assert report["no_embedding"] is True
    assert report["no_mcp"] is True
    assert report["no_gold_use"] is True
    assert report["no_task_id_hardcode"] is True


@pytest.mark.skipif(not REAL_PLANNER.exists(), reason="real planner cache missing")
def test_real_planner_outputs_sha_unchanged_by_this_prep_round():
    assert file_sha256(REAL_PLANNER) == EXPECTED_REAL_SHA
    # Prep must not have created a mutating rewrite; attempt1 may or may not exist.
    # If attempt1 exists, its outputs must equal the frozen sha.
    attempt = REAL_OUT / "planner_cache_attempt1" / "planner_outputs.jsonl"
    if attempt.exists():
        assert file_sha256(attempt) == EXPECTED_REAL_SHA
