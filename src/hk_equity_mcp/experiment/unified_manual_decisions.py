"""Load and validate human manual-review decisions (fail-closed).

Decisions live outside the run directory so audit reruns cannot overwrite them.
Never silently substitute suggested_* fields for human_confirmed results.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


class ManualDecisionError(ValueError):
    """Fail-closed validation error for manual decision files."""


def _parse_01(raw: Any, *, field: str, tid: str, allow_blank: bool = False) -> Optional[int]:
    if raw is None:
        text = ""
    else:
        text = str(raw).strip()
    if text == "":
        if allow_blank:
            return None
        raise ManualDecisionError(f"{tid}: {field} must be 0 or 1 (blank not allowed)")
    if text in {"0", "1"}:
        return int(text)
    # Reject True/False/yes/suggested values — only explicit 0/1.
    raise ManualDecisionError(
        f"{tid}: {field} must be exactly '0' or '1' (got {raw!r})"
    )


def load_manual_decisions(
    path: Path,
    *,
    allowed_task_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Load decisions CSV and validate against the review-queue allow-list.

    Required columns:
      task_id, answer_correct, evidence_correct, overall_task_success
    Optional:
      notes

    Rules:
      - every task_id must be in allowed_task_ids
      - task_id unique
      - answer_correct in {0,1}
      - overall_task_success in {0,1}
      - evidence_correct in {0,1,\"\"} where blank means N/A
      - unknown / duplicate IDs → fail closed
      - missing required queue coverage is allowed only if caller checks separately;
        this loader requires all rows to be valid members of the queue.
    """
    path = Path(path)
    if not path.exists():
        raise ManualDecisionError(f"manual decisions file not found: {path}")

    allowed: Set[str] = {str(x) for x in allowed_task_ids}
    if not allowed:
        raise ManualDecisionError("allowed_task_ids is empty")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManualDecisionError("manual decisions CSV has no header")
        required = {
            "task_id",
            "answer_correct",
            "evidence_correct",
            "overall_task_success",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ManualDecisionError(
                f"manual decisions missing columns: {sorted(missing)}"
            )
        # Reject accidental suggested_* authoritative usage.
        forbidden_as_source = {
            "suggested_answer_correct",
            "suggested_evidence_correct",
            "suggested_overall_task_success",
            "human_confirmed_answer_correct",
            "human_confirmed_evidence_correct",
            "human_confirmed_overall_task_success",
        }
        # Presence of these columns is OK only if we never read them as the
        # decision source. We deliberately ignore them.
        _ = forbidden_as_source

        out: Dict[str, Dict[str, Any]] = {}
        for line_no, row in enumerate(reader, start=2):
            tid = str(row.get("task_id") or "").strip()
            if not tid:
                raise ManualDecisionError(f"line {line_no}: empty task_id")
            if tid not in allowed:
                raise ManualDecisionError(
                    f"unknown task_id {tid!r} not in review queue "
                    f"(allowed={sorted(allowed)})"
                )
            if tid in out:
                raise ManualDecisionError(f"duplicate task_id {tid!r}")

            answer = _parse_01(row.get("answer_correct"), field="answer_correct", tid=tid)
            overall = _parse_01(
                row.get("overall_task_success"),
                field="overall_task_success",
                tid=tid,
            )
            evidence = _parse_01(
                row.get("evidence_correct"),
                field="evidence_correct",
                tid=tid,
                allow_blank=True,
            )
            out[tid] = {
                "task_id": tid,
                "answer_correct": bool(answer),
                "answer_correct_01": int(answer),  # type: ignore[arg-type]
                "evidence_correct": (
                    None if evidence is None else bool(evidence)
                ),
                "evidence_correct_01": evidence,
                "evidence_na": evidence is None,
                "overall_task_success": bool(overall),
                "overall_task_success_01": int(overall),  # type: ignore[arg-type]
                "notes": str(row.get("notes") or ""),
                "source_path": str(path),
            }

    return out


def require_full_queue_coverage(
    decisions: Dict[str, Dict[str, Any]],
    allowed_task_ids: Sequence[str],
) -> None:
    """Fail closed if any review-queue task lacks a human decision."""
    allowed = {str(x) for x in allowed_task_ids}
    missing = sorted(allowed - set(decisions))
    if missing:
        raise ManualDecisionError(
            f"manual decisions incomplete; missing queue tasks: {missing}"
        )
    extra = sorted(set(decisions) - allowed)
    if extra:
        raise ManualDecisionError(
            f"manual decisions contain non-queue tasks: {extra}"
        )


def compute_strict_task_success(
    row: Dict[str, Any],
    *,
    answer_correct: Optional[bool],
    evidence_correct: Optional[bool],
    evidence_na: bool = False,
) -> Dict[str, Any]:
    """Strict overall task success — independent of human overall_task_success notes.

    Required (all must hold):
      - answer correct
      - exact required tool set
      - all required tool invocations successful
      - dependency/order correct when applicable
      - refusal correct when applicable
      - evidence correct OR evidence N/A
    """
    checks: Dict[str, Optional[bool]] = {}
    checks["answer_correct"] = bool(answer_correct) if answer_correct is not None else False

    checks["exact_tool_set"] = bool(row.get("exact_tool_set"))

    tool_ok = row.get("all_tool_invocations_successful")
    if tool_ok is None:
        tool_ok = row.get("tool_execution_success")
    checks["all_tool_invocations_successful"] = bool(tool_ok) if tool_ok is not None else False

    # Dependency applicable when a numeric graph accuracy is present.
    dep = row.get("graph_dependency_accuracy")
    if isinstance(dep, (int, float)):
        checks["dependency_order"] = float(dep) >= 1.0 - 1e-12
    else:
        checks["dependency_order"] = None  # N/A

    refusal = row.get("refusal_correctness")
    if isinstance(refusal, (int, float)):
        checks["refusal_correct"] = float(refusal) >= 1.0 - 1e-12
    else:
        checks["refusal_correct"] = None  # N/A

    if evidence_na or evidence_correct is None:
        checks["evidence_correct_or_na"] = True
    else:
        checks["evidence_correct_or_na"] = bool(evidence_correct)

    required_failed = [
        name
        for name, val in checks.items()
        if val is False
    ]
    strict = len(required_failed) == 0 and checks["answer_correct"] is True
    return {
        "strict_task_success": bool(strict),
        "strict_task_success_01": 1 if strict else 0,
        "strict_checks": checks,
        "strict_failed_checks": required_failed,
    }


def merge_final_answer_correct(
    auto_answer_correct: Optional[bool],
    auto_grade_status: Optional[str],
    decision: Optional[Dict[str, Any]],
) -> Optional[bool]:
    """Human decision overrides only for queued tasks; never use suggested_*."""
    if decision is not None:
        return bool(decision["answer_correct"])
    if auto_grade_status == "manual_required":
        # No human decision → still unresolved (caller should have failed closed
        # when decisions were required).
        return None
    return auto_answer_correct if isinstance(auto_answer_correct, bool) else None


def summarize_final_metrics(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Separate A/B/C metrics — never conflate them."""
    n = len(rows)
    answer_vals = [
        1 if r.get("final_answer_correct") is True else 0
        for r in rows
        if r.get("final_answer_correct") is not None
    ]
    # Final answer accuracy requires every task resolved.
    if len(answer_vals) != n:
        final_answer_accuracy = None
        final_answer_correct_count = sum(answer_vals)
        unresolved = n - len(answer_vals)
    else:
        final_answer_correct_count = sum(answer_vals)
        final_answer_accuracy = final_answer_correct_count / n if n else None
        unresolved = 0

    exact_count = sum(1 for r in rows if r.get("exact_tool_set"))
    strict_count = sum(1 for r in rows if r.get("strict_task_success"))

    return {
        "n": n,
        "final_answer_correct_count": final_answer_correct_count,
        "final_answer_accuracy": final_answer_accuracy,
        "final_answer_unresolved_count": unresolved,
        "exact_tool_set_success_count": exact_count,
        "exact_tool_set_success_rate": (exact_count / n) if n else None,
        "strict_task_success_count": strict_count,
        "strict_task_success_rate": (strict_count / n) if n else None,
        "metric_definitions": {
            "A_final_answer_accuracy": (
                "Share of tasks with correct final answer "
                "(auto v1.2 or human decision). Independent of tool-set."
            ),
            "B_exact_tool_set_success": (
                "Share of tasks where predicted invoke set exactly equals gold set."
            ),
            "C_strict_overall_task_success": (
                "answer correct AND exact tool set AND all invocations successful "
                "AND dependency OK when applicable AND refusal OK when applicable "
                "AND evidence correct or N/A."
            ),
        },
        "note": (
            "A, B, and C are separate metrics and must not be substituted for each other."
        ),
    }


def summarize_final_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    base = summarize_final_metrics(rows)
    return {
        "task_count": len(rows),
        **base,
        "mean_operation_f1": _avg(
            [float(r["operation_f1"]) for r in rows if isinstance(r.get("operation_f1"), (int, float))]
        ),
        "mean_operation_precision": _avg(
            [
                float(r["operation_precision"])
                for r in rows
                if isinstance(r.get("operation_precision"), (int, float))
            ]
        ),
        "mean_operation_recall": _avg(
            [
                float(r["operation_recall"])
                for r in rows
                if isinstance(r.get("operation_recall"), (int, float))
            ]
        ),
        "all_tool_invocations_successful_rate": (
            sum(1 for r in rows if r.get("all_tool_invocations_successful")) / len(rows)
            if rows
            else None
        ),
        "refusal_accuracy": _avg(
            [
                float(r["refusal_correctness"])
                for r in rows
                if isinstance(r.get("refusal_correctness"), (int, float))
            ]
        ),
    }


def _avg(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_final_grouped_aggregates(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import defaultdict

    out: Dict[str, Any] = {"overall": summarize_final_group(rows)}
    for key in (
        "domain",
        "project_difficulty",
        "answerability_class",
        "provider_segment",
    ):
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key) or "null")].append(row)
        out[key] = {
            name: summarize_final_group(group)
            for name, group in sorted(buckets.items())
        }
    return out
