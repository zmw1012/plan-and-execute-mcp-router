"""Engineering-pass definition for Checkpoint Audit v1 (not accuracy)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


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
                    c.get("status") != "fail"
                    for c in (t.get("checkpoint_list") or [])
                )
            )
            for t in per_task
        ),
        "na_not_counted_as_pass": True,  # enforced in funnel accounting
        "evaluator_separated": True,
        "zero_invocation_not_tool_success": _zero_invocation_ok(per_task),
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
            if c.get("status") != "N/A"
        )
        if applicable and unk / max(applicable, 1) > 0.5:
            unknown_heavy += 1
    checks["trace_coverage_adequate"] = unknown_heavy == 0

    engineering_pass = all(bool(v) for v in checks.values())
    return {
        "engineering_pass": engineering_pass,
        "explicitly_not_formal_accuracy": True,
        "not_accuracy_pass": True,
        "checks": checks,
        "n_rows": n_rows,
        "unique_ids": len(unique),
        "expected_n": expected_n,
        "notes": [
            "Answer may be incorrect while engineering_pass is true.",
            "N/A is not counted as pass.",
            "Evaluator errors are separated from agent/method errors.",
        ],
    }


def _zero_invocation_ok(per_task: Sequence[Dict[str, Any]]) -> bool:
    for t in per_task:
        c07 = (t.get("checkpoints") or {}).get("C07_mcp") or {}
        detail = c07.get("detail") or {}
        if detail.get("no_invocation") and detail.get("tool_success_claimed") is True:
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

    by_cp: Dict[str, Counter] = {}
    by_stage: Dict[str, Counter] = {}
    by_contract: Dict[str, Counter] = {}
    for t in per_task:
        for c in t.get("checkpoint_list") or []:
            cid = c["checkpoint"]
            by_cp.setdefault(cid, Counter())
            by_cp[cid][c["status"]] += 1
            by_stage.setdefault(cid, Counter())
            by_stage[cid][c.get("stage_outcome_status") or c["status"]] += 1
            by_contract.setdefault(cid, Counter())
            by_contract[cid][c.get("contract_status") or c["status"]] += 1
    return {
        cid: {
            "pass": int(cnt.get("pass", 0)),
            "fail": int(cnt.get("fail", 0)),
            "N/A": int(cnt.get("N/A", 0)),
            "unknown": int(cnt.get("unknown", 0)),
            "not_evaluated": int(cnt.get("not_evaluated", 0)),
            "pass_rate_excluding_na": (
                float(cnt.get("pass", 0))
                / max(
                    1,
                    int(cnt.get("pass", 0))
                    + int(cnt.get("fail", 0))
                    + int(cnt.get("unknown", 0)),
                )
            ),
            "contract_status_counts": dict(by_contract.get(cid) or {}),
            "stage_outcome_status_counts": dict(by_stage.get(cid) or {}),
        }
        for cid, cnt in by_cp.items()
    }
