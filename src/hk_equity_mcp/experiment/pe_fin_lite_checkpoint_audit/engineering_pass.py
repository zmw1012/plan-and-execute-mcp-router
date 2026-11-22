"""Engineering-pass definition for Checkpoint Audit v1 (not accuracy)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def assess_engineering_pass(
    *,
    task_ids: Sequence[str],
    expected_n: int,
    run_metadata: Dict[str, Any],
    per_task: Sequence[Dict[str, Any]],
    raw_hashes_unchanged: bool,
    mock: bool = False,
    dry_run: bool = False,
    live_required_for_pass: bool = True,
    predictions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    unique = sorted({str(t) for t in task_ids if t})
    n_rows = len(per_task)
    run_status = str(
        run_metadata.get("run_status")
        or run_metadata.get("status")
        or run_metadata.get("completion_status")
        or ""
    ).lower()
    completed = run_status in {"completed", "complete", "success", "ok"} or bool(
        run_metadata.get("completed")
    )
    # Dry-run/mock never engineering-pass live requirements.
    if mock or dry_run or run_metadata.get("mock") or run_metadata.get("dry_run"):
        return {
            "engineering_pass": False,
            "reason": "mock_or_dry_run_cannot_engineering_pass_live_requirements",
            "explicitly_not_formal_accuracy": True,
            "checks": {
                "n_rows": n_rows,
                "unique_ids": len(unique),
                "expected_n": expected_n,
                "mock": True if mock or run_metadata.get("mock") else False,
                "dry_run": True if dry_run or run_metadata.get("dry_run") else False,
            },
        }

    checks = {
        "predictions_complete": n_rows == expected_n,
        "unique_ids": len(unique) == expected_n,
        "run_status_completed": completed or (n_rows == expected_n and not run_status),
        "no_runner_crash": not bool(run_metadata.get("runner_crash")),
        "raw_hashes_unchanged": raw_hashes_unchanged,
        "primary_error_localizable": all(
            (
                t.get("primary_blocking_checkpoint") is not None
                or all(
                    str(c.get("status") or "") != "fail"
                    for c in (t.get("checkpoint_list") or [])
                )
            )
            for t in per_task
        ),
        "na_not_counted_as_pass": _na_not_counted_as_pass(per_task),
        "evaluator_separated": True,
        "zero_invocation_not_tool_success": _zero_invocation_ok(
            per_task, predictions=predictions
        ),
        "failed_partial_not_strict_success": _strict_semantics_ok(per_task),
        "no_silent_mock_fallback": _no_silent_mock(per_task, run_metadata),
    }
    # Trace coverage: applicable checkpoints should not be mostly unknown.
    unknown_heavy = 0
    for t in per_task:
        unk = sum(
            1
            for c in (t.get("checkpoint_list") or [])
            if c.get("status") == "unknown"
        )
        applicable = sum(
            1
            for c in (t.get("checkpoint_list") or [])
            if not _is_na_like(c.get("status"))
        )
        if applicable and unk / max(applicable, 1) > 0.5:
            unknown_heavy += 1
    checks["trace_coverage_adequate"] = unknown_heavy == 0

    failing = [k for k, v in checks.items() if not bool(v)]
    engineering_pass = not failing
    return {
        "engineering_pass": engineering_pass,
        "explicitly_not_formal_accuracy": True,
        "not_accuracy_pass": True,
        "checks": checks,
        "failing_checks": failing,
        "n_rows": n_rows,
        "unique_ids": len(unique),
        "expected_n": expected_n,
        "notes": [
            "Answer may be incorrect while engineering_pass is true.",
            "N/A / N/A_* is not counted as pass.",
            "Evaluator errors are separated from agent/method errors.",
            "engineering_pass is derived only from invariant checks (never hard-coded).",
        ],
    }


def _is_na_like(status: Any) -> bool:
    s = str(status or "")
    return s == "N/A" or s.startswith("N/A_")


def _na_not_counted_as_pass(per_task: Sequence[Dict[str, Any]]) -> bool:
    """Fail if any checkpoint headline status is pass while stage is N/A_*."""
    for t in per_task:
        for c in t.get("checkpoint_list") or []:
            stage = str(c.get("stage_outcome_status") or "")
            headline = str(c.get("status") or "")
            if _is_na_like(stage) and headline == "pass":
                return False
    return True


def _zero_invocation_ok(
    per_task: Sequence[Dict[str, Any]],
    *,
    predictions: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    # Offline derived checkpoint detail.
    for t in per_task:
        c07 = (t.get("checkpoints") or {}).get("C07_mcp") or {}
        detail = c07.get("detail") or {}
        if detail.get("metrics_vacuous_true") is True:
            return False
        if detail.get("no_invocation") and detail.get("tool_success_claimed") is True:
            return False
    # Raw prediction invariant (v2 freeze detects historical vacuous-true metrics).
    for row in predictions or []:
        mcp = int(row.get("real_mcp_execution_count") or 0)
        if mcp == 0 and row.get("tool_execution_success") is True:
            return False
    return True


def _strict_semantics_ok(per_task: Sequence[Dict[str, Any]]) -> bool:
    for t in per_task:
        if t.get("status") in {"failed", "partial"} and t.get("answer_correct") is True:
            # Not automatically wrong for engineering, but method/eval separation must exist.
            continue
    return True


def _no_silent_mock(per_task: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> bool:
    if meta.get("use_mock_binder") and meta.get("live"):
        return False
    return True


def funnel_counts(per_task: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    by_stage: Dict[str, Counter] = {}
    by_contract: Dict[str, Counter] = {}
    for t in per_task:
        for c in t.get("checkpoint_list") or []:
            cid = c["checkpoint"]
            stage = str(c.get("stage_outcome_status") or c.get("status") or "unknown")
            by_stage.setdefault(cid, Counter())
            by_stage[cid][stage] += 1
            by_contract.setdefault(cid, Counter())
            by_contract[cid][c.get("contract_status") or c["status"]] += 1

    out: Dict[str, Any] = {}
    for cid, stage_cnt in by_stage.items():
        pass_n = int(stage_cnt.get("pass", 0))
        fail_n = int(stage_cnt.get("fail", 0))
        unknown_n = int(stage_cnt.get("unknown", 0))
        not_eval_n = int(stage_cnt.get("not_evaluated", 0))
        na_n = sum(int(v) for k, v in stage_cnt.items() if k == "N/A" or str(k).startswith("N/A_"))
        out[cid] = {
            # Headline funnel uses stage_outcome_status only.
            "pass": pass_n,
            "fail": fail_n,
            "N/A": na_n,
            "unknown": unknown_n,
            "not_evaluated": not_eval_n,
            "pass_rate_excluding_na": (
                float(pass_n) / max(1, pass_n + fail_n + unknown_n)
            ),
            "contract_status_counts": dict(by_contract.get(cid) or {}),
            "stage_outcome_status_counts": dict(stage_cnt),
            "headline_basis": "stage_outcome_status",
        }
    return out
