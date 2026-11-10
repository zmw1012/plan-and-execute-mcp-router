"""Planner-only stage: generate/freeze plans + step queries (no Binder/MCP/embed)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    PLANNER_ONLY_PROTOCOL_VERSION,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    assert_identity_match,
    sha256_text,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import build_step_query
from hk_equity_mcp.pe_fin_lite.capability_profile import load_capability_profile
from hk_equity_mcp.pe_fin_lite.planner import run_planner


FORBIDDEN_PLANNER_KEYS = (
    "gold_operations",
    "acceptable_tool_sets",
    "reference_answer",
    "gold_values",
)


def _assert_no_gold_leakage(payload: Dict[str, Any]) -> None:
    blob = json.dumps(payload, ensure_ascii=False)
    for key in FORBIDDEN_PLANNER_KEYS:
        if f'"{key}"' in blob:
            raise RuntimeError(f"Gold/reference key {key!r} must not enter planner cache")


def mock_plan_for_question(task: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic offline mock plan (no gold ops; domain-generic retrieve)."""
    domain = str(task.get("domain") or "general")
    question = str(task.get("question") or "")
    # Map cross_domain to dual retrieve steps without naming gold tools.
    if domain == "cross_domain":
        steps = [
            {
                "id": "step1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "retrieve financial evidence relevant to the question",
                "required_outputs": ["basic_eps", "currency", "report_date"],
                "depends_on": [],
            },
            {
                "id": "step2",
                "type": "retrieve",
                "domain": "market",
                "objective": "retrieve market evidence relevant to the question",
                "required_outputs": ["close", "trade_date", "volume"],
                "depends_on": [],
            },
        ]
    elif domain in {"news", "disclosure"}:
        steps = [
            {
                "id": "step1",
                "type": "retrieve",
                "domain": "disclosure",
                "objective": "retrieve official disclosure evidence",
                "required_outputs": ["news_id", "title", "release_date", "local_key"],
                "depends_on": [],
                "workflow": "retrieve_official_disclosure_evidence",
            }
        ]
    elif domain == "market":
        steps = [
            {
                "id": "step1",
                "type": "retrieve",
                "domain": "market",
                "objective": "retrieve market series for the asked scope",
                "required_outputs": ["close", "trade_date", "volume"],
                "depends_on": [],
            }
        ]
    else:
        steps = [
            {
                "id": "step1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "retrieve financial indicators for the asked scope",
                "required_outputs": ["revenue", "basic_eps", "report_date", "currency"],
                "depends_on": [],
            }
        ]
    # Keep objective grounded in question text only (no gold).
    if question and steps:
        steps[0]["objective"] = (steps[0]["objective"] + " | " + question[:180]).strip()
    return {
        "schema_version": "pe_fin_lite_plan_v2",
        "task_summary": question[:240],
        "steps": steps,
        "final_output_spec": {
            "answer_type": "structured",
            "units": None,
            "rounding": None,
            "presentation": "concise",
        },
    }


def extract_retrieve_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return []
    out = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("type") or "") != "retrieve":
            continue
        row = dict(step)
        row["search_query"] = build_step_query(row)
        out.append(row)
    return out


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_completed_ids(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = str(row.get("task_id") or "")
        if tid:
            out[tid] = row
    return out


def run_planner_only(
    *,
    root: Path,
    tasks: Sequence[Dict[str, Any]],
    output_dir: Path,
    cache_identity: Dict[str, Any],
    llm: Any = None,
    mock: bool = False,
    dry_run: bool = False,
    live: bool = False,
    resume: bool = True,
    capability_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write planner_outputs.jsonl with checkpoint/resume support.

    Live mode requires a real ``llm`` and never falls back to mock plans.
    Does not call embedding, MCP, Binder, Reason, or Final.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "planner_outputs.jsonl"
    meta_path = output_dir / "planner_cache_identity.json"

    if live:
        if mock or dry_run:
            raise RuntimeError("planner-only live cannot combine mock/dry-run")
        if llm is None:
            raise RuntimeError(
                "planner-only live requires a real llm client (llm=None refused; "
                "no mock_plan_for_question fallback)"
            )

    if meta_path.exists() and resume:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        # Refuse diagnostic10 ↔ all155 cross-resume even before hash compare.
        from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_diagnostic10 import (
            assert_not_cross_resume,
        )

        assert_not_cross_resume(cache_identity, prev)
        assert_identity_match(cache_identity, prev, label="planner_only_cache")
    else:
        with meta_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(cache_identity, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    completed = _load_completed_ids(out_path) if resume else {}
    profile = capability_profile
    if profile is None and not dry_run:
        profile = load_capability_profile(root)

    errors: List[Dict[str, Any]] = []
    written = 0
    for task in tasks:
        tid = str(task.get("task_id") or task.get("id"))
        if tid in completed:
            continue
        if dry_run:
            row = {
                "task_id": tid,
                "ok": True,
                "dry_run": True,
                "plan": None,
                "retrieve_steps": [],
                "error": None,
                "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION,
            }
            _append_jsonl(out_path, row)
            written += 1
            continue

        if mock:
            if live:
                raise RuntimeError("unreachable: live+mock")
            plan = mock_plan_for_question(task)
            retrieve_steps = extract_retrieve_steps(plan)
            row = {
                "task_id": tid,
                "ok": True,
                "mock": True,
                "plan": plan,
                "retrieve_steps": retrieve_steps,
                "raw_planner_response": json.dumps(plan, ensure_ascii=False),
                "validated_plan": plan,
                "error": None,
                "repair_count": 0,
                "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION,
            }
            _assert_no_gold_leakage(row)
            _append_jsonl(out_path, row)
            written += 1
            continue

        # Live planner path — llm already validated non-None.
        t0 = time.perf_counter()
        result = run_planner(
            llm=llm,
            question=str(task.get("question") or ""),
            capability_profile=profile or {},
            limits={
                "max_total_llm_calls_per_task": 8,
                "max_total_mcp_invocations": 8,
            },
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        plan = result.get("plan") if result.get("ok") else None
        retrieve_steps = extract_retrieve_steps(plan or {})
        usage = dict(result.get("usage") or {})
        row = {
            "task_id": tid,
            "ok": bool(result.get("ok")),
            "mock": False,
            "live": True,
            "plan": plan,
            "retrieve_steps": retrieve_steps,
            "raw_planner_response": result.get("raw_content"),
            "repair_raw_content": result.get("repair_raw_content"),
            "validated_plan": plan,
            "error": result.get("error"),
            "error_type": result.get("error_type"),
            "repair_count": result.get("repair_count"),
            "usage": usage,
            "latency_ms": latency_ms,
            "llm_calls": int(usage.get("calls") or usage.get("llm_calls") or (1 + int(result.get("repair_count") or 0))),
            "tokens_in": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "tokens_out": usage.get("output_tokens") or usage.get("completion_tokens"),
            "tokens_total": usage.get("total_tokens"),
            "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION,
            "no_embedding": True,
            "no_mcp": True,
            "no_binder": True,
        }
        _assert_no_gold_leakage(
            {
                "task_id": tid,
                "plan": plan,
                "retrieve_steps": retrieve_steps,
                "raw": result.get("raw_content"),
            }
        )
        _append_jsonl(out_path, row)
        written += 1
        if not row["ok"]:
            errors.append({"task_id": tid, "error": row.get("error")})

    # Finalize cache sha over outputs.
    final_rows = list(_load_completed_ids(out_path).values())
    cache_sha = sha256_text(
        "\n".join(
            json.dumps(final_rows[i], sort_keys=True, ensure_ascii=False)
            for i in range(len(final_rows))
        )
        if False
        else out_path.read_text(encoding="utf-8")
        if out_path.exists()
        else ""
    )
    summary = {
        "n_tasks": len(tasks),
        "n_rows": len(final_rows),
        "newly_written": written,
        "n_errors": len(errors),
        "planner_cache_sha256": cache_sha,
        "identity": cache_identity,
        "output_path": str(out_path),
        "errors": errors[:50],
    }
    (output_dir / "planner_only_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
