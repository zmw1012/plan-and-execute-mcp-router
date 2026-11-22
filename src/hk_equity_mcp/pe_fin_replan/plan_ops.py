"""Remaining-plan selection, diff, and validation for RePlan loop."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.pe_fin_lite.planner import topological_ready_order
from hk_equity_mcp.pe_fin_lite.schemas import PlanValidationError


def ready_steps(remaining: Sequence[Dict[str, Any]], completed_ids: Set[str]) -> List[Dict[str, Any]]:
    """Return currently executable steps (all depends_on in completed_ids)."""
    ready: List[Dict[str, Any]] = []
    for step in remaining:
        sid = str(step.get("id"))
        if sid in completed_ids:
            continue
        deps = [str(d) for d in (step.get("depends_on") or [])]
        if all(d in completed_ids for d in deps):
            ready.append(step)
    return ready


def select_one_ready_step(
    remaining: Sequence[Dict[str, Any]], completed_ids: Set[str]
) -> Optional[Dict[str, Any]]:
    ready = ready_steps(remaining, completed_ids)
    return deepcopy(ready[0]) if ready else None


def remove_step(remaining: List[Dict[str, Any]], step_id: str) -> List[Dict[str, Any]]:
    return [s for s in remaining if str(s.get("id")) != str(step_id)]


def step_fingerprint(step: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(step.get("type") or ""),
            str(step.get("objective") or "").strip().lower()[:200],
            ",".join(str(x) for x in (step.get("required_outputs") or [])),
            str((step.get("scope_constraints") or {}).get("period") or ""),
            str((step.get("scope_constraints") or {}).get("metric_basis") or ""),
        ]
    )


def diff_plans(
    before: Sequence[Dict[str, Any]],
    after: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    before_ids = {str(s.get("id")): s for s in before}
    after_ids = {str(s.get("id")): s for s in after}
    added = [sid for sid in after_ids if sid not in before_ids]
    removed = [sid for sid in before_ids if sid not in after_ids]
    modified = []
    for sid in after_ids:
        if sid in before_ids and step_fingerprint(before_ids[sid]) != step_fingerprint(
            after_ids[sid]
        ):
            modified.append(sid)
    return {
        "added_step_ids": added,
        "removed_step_ids": removed,
        "modified_step_ids": modified,
        "n_added": len(added),
        "n_removed": len(removed),
        "n_modified": len(modified),
        "changed": bool(added or removed or modified),
    }


def validate_updated_remaining_plan(
    *,
    proposed: Sequence[Dict[str, Any]],
    previous_remaining: Sequence[Dict[str, Any]],
    completed_ids: Set[str],
    completed_objectives: Set[str],
    max_new_steps: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Validate Replanner plan update; raise PlanValidationError on hard violations."""
    steps = [deepcopy(s) for s in proposed]
    # Ensure ids unique
    ids = [str(s.get("id")) for s in steps]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("duplicate step ids in remaining_steps")
    # Cap new steps relative to previous remaining
    prev_ids = {str(s.get("id")) for s in previous_remaining}
    new_ids = [i for i in ids if i not in prev_ids]
    if len(new_ids) > max_new_steps:
        raise PlanValidationError(
            f"too_many_new_steps:{len(new_ids)}>max={max_new_steps}"
        )
    # Completed step ids must not reappear unless fingerprint/objective changed
    for s in steps:
        sid = str(s.get("id"))
        if sid in completed_ids:
            raise PlanValidationError(f"completed_step_readded:{sid}")
        obj = str(s.get("objective") or "").strip().lower()
        if obj and obj in completed_objectives:
            # Allow only if scope clearly differs — require period or metric_basis present & different marker
            scope = s.get("scope_constraints") if isinstance(s.get("scope_constraints"), dict) else {}
            if not (scope.get("period") or scope.get("metric_basis") or scope.get("trade_date")):
                raise PlanValidationError(f"completed_objective_repeat_without_scope_change:{obj[:80]}")
    # DAG orderable check when non-empty
    if steps:
        try:
            topological_ready_order(steps)
        except PlanValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PlanValidationError(f"remaining_plan_not_orderable:{exc}") from exc
    diff = diff_plans(previous_remaining, steps)
    return steps, diff
