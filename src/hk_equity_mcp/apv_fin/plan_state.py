"""Dynamic APV-Fin plan state: DAG validation, readiness, and patch application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from hk_equity_mcp.apv_fin.schemas import STEP_STATUSES, STEP_TYPES


class PlanStateError(ValueError):
    pass


@dataclass
class PlanStep:
    step_id: str
    step_type: str
    objective: str
    depends_on: List[str] = field(default_factory=list)
    mandatory: bool = True
    success_criteria: str = ""
    retrieval_query_hint: Optional[str] = None
    notes: Optional[str] = None
    status: str = "pending"
    observation_ids: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PlanStep":
        step_type = str(raw.get("step_type") or "")
        if step_type not in STEP_TYPES:
            raise PlanStateError(f"invalid step_type: {step_type!r}")
        status = str(raw.get("status") or "pending")
        if status not in STEP_STATUSES:
            raise PlanStateError(f"invalid step status: {status!r}")
        step_id = str(raw.get("step_id") or "").strip()
        if not step_id:
            raise PlanStateError("step_id must be non-empty")
        depends = list(raw.get("depends_on") or [])
        if not all(isinstance(d, str) and d for d in depends):
            raise PlanStateError(f"invalid depends_on for {step_id}")
        return cls(
            step_id=step_id,
            step_type=step_type,
            objective=str(raw.get("objective") or ""),
            depends_on=depends,
            mandatory=bool(raw.get("mandatory", True)),
            success_criteria=str(raw.get("success_criteria") or ""),
            retrieval_query_hint=raw.get("retrieval_query_hint"),
            notes=raw.get("notes"),
            status=status,
            observation_ids=list(raw.get("observation_ids") or []),
            failure_reason=raw.get("failure_reason"),
        )


@dataclass
class PlanState:
    steps: Dict[str, PlanStep] = field(default_factory=dict)
    revision: int = 0
    revision_log: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_initial_plan(cls, initial_plan: Dict[str, Any]) -> "PlanState":
        raw_steps = list((initial_plan or {}).get("steps") or [])
        state = cls()
        state._replace_steps(raw_steps, revision_note="initial_plan")
        return state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "steps": {sid: step.to_dict() for sid, step in self.steps.items()},
            "revision_log": deepcopy(self.revision_log),
        }

    def step_ids(self) -> List[str]:
        return list(self.steps.keys())

    def mandatory_step_ids(self) -> List[str]:
        return [sid for sid, step in self.steps.items() if step.mandatory]

    def _replace_steps(
        self, raw_steps: Sequence[Dict[str, Any]], *, revision_note: str
    ) -> None:
        parsed = [PlanStep.from_dict(raw) for raw in raw_steps]
        ids = [s.step_id for s in parsed]
        if len(ids) != len(set(ids)):
            raise PlanStateError("duplicate step_id in plan")
        by_id = {s.step_id: s for s in parsed}
        self._validate_dependencies(by_id)
        self._detect_cycles(by_id)
        self.steps = by_id
        self._recompute_ready()
        self.revision_log.append(
            {
                "revision": self.revision,
                "note": revision_note,
                "step_ids": list(self.steps.keys()),
            }
        )

    @staticmethod
    def _validate_dependencies(steps: Dict[str, PlanStep]) -> None:
        for sid, step in steps.items():
            for dep in step.depends_on:
                if dep not in steps:
                    raise PlanStateError(
                        f"missing dependency {dep!r} referenced by {sid!r}"
                    )
                if dep == sid:
                    raise PlanStateError(f"step {sid!r} depends on itself")

    @staticmethod
    def _detect_cycles(steps: Dict[str, PlanStep]) -> None:
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def dfs(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise PlanStateError(f"cyclic plan involving {node!r}")
            visiting.add(node)
            for dep in steps[node].depends_on:
                dfs(dep)
            visiting.remove(node)
            visited.add(node)

        for sid in steps:
            dfs(sid)

    def _recompute_ready(self) -> None:
        for sid, step in self.steps.items():
            if step.status in {"resolved", "failed", "skipped", "running"}:
                continue
            deps_ok = all(
                self.steps[d].status == "resolved" for d in step.depends_on
            )
            step.status = "ready" if deps_ok else "pending"

    def ready_steps(self) -> List[PlanStep]:
        return [s for s in self.steps.values() if s.status == "ready"]

    def mark_running(self, step_id: str) -> None:
        step = self._require(step_id)
        if step.status not in {"ready", "pending"}:
            raise PlanStateError(
                f"cannot mark {step_id} running from status={step.status}"
            )
        step.status = "running"

    def mark_resolved(
        self, step_id: str, *, observation_ids: Optional[Sequence[str]] = None
    ) -> None:
        step = self._require(step_id)
        step.status = "resolved"
        if observation_ids:
            for oid in observation_ids:
                if oid not in step.observation_ids:
                    step.observation_ids.append(oid)
        self._recompute_ready()

    def mark_failed(self, step_id: str, reason: str) -> None:
        step = self._require(step_id)
        step.status = "failed"
        step.failure_reason = reason
        self._recompute_ready()

    def mark_skipped(self, step_id: str, reason: str) -> None:
        step = self._require(step_id)
        step.status = "skipped"
        step.failure_reason = reason
        self._recompute_ready()

    def resolve_steps_from_observation(
        self,
        step_ids: Sequence[str],
        observation_id: str,
    ) -> List[str]:
        """One observation may resolve multiple plan steps."""
        resolved: List[str] = []
        for sid in step_ids:
            if sid not in self.steps:
                raise PlanStateError(f"unknown step_id in observation resolve: {sid}")
            self.mark_resolved(sid, observation_ids=[observation_id])
            resolved.append(sid)
        return resolved

    def apply_monitor_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply incremental patch from monitor (completed/updated/new/removed)."""
        patch = patch or {}
        completed = list(patch.get("completed_steps") or [])
        updated = list(patch.get("updated_steps") or [])
        new_steps = list(patch.get("new_steps") or [])
        removed = list(patch.get("removed_steps") or [])

        for sid in completed:
            if sid in self.steps and self.steps[sid].status != "resolved":
                self.mark_resolved(sid)

        for rem in removed:
            if rem not in self.steps:
                continue
            # Do not remove already resolved mandatory evidence silently —
            # mark skipped so coverage checks remain auditable.
            if self.steps[rem].status == "resolved":
                continue
            self.mark_skipped(rem, "removed_by_replan")
            del self.steps[rem]

        for upd in updated:
            sid = str(upd.get("step_id") or "")
            if sid not in self.steps:
                raise PlanStateError(f"updated_steps references unknown step {sid!r}")
            step = self.steps[sid]
            if upd.get("objective") is not None:
                step.objective = str(upd["objective"])
            if upd.get("depends_on") is not None:
                step.depends_on = list(upd["depends_on"])
            if upd.get("mandatory") is not None:
                step.mandatory = bool(upd["mandatory"])
            if upd.get("success_criteria") is not None:
                step.success_criteria = str(upd["success_criteria"])
            if upd.get("step_type") is not None:
                st = str(upd["step_type"])
                if st not in STEP_TYPES:
                    raise PlanStateError(f"invalid updated step_type: {st}")
                step.step_type = st
            if "retrieval_query_hint" in upd:
                step.retrieval_query_hint = upd.get("retrieval_query_hint")
            if "notes" in upd:
                step.notes = upd.get("notes")

        for raw in new_steps:
            step = PlanStep.from_dict(raw)
            if step.step_id in self.steps:
                raise PlanStateError(
                    f"new_steps duplicates existing step_id {step.step_id!r}"
                )
            self.steps[step.step_id] = step

        self._validate_dependencies(self.steps)
        self._detect_cycles(self.steps)
        self._recompute_ready()
        self.revision += 1
        summary = {
            "revision": self.revision,
            "completed_steps": completed,
            "updated_steps": [u.get("step_id") for u in updated],
            "new_steps": [s.get("step_id") for s in new_steps],
            "removed_steps": removed,
        }
        self.revision_log.append(summary)
        return summary

    def mandatory_coverage(self) -> Dict[str, Any]:
        mandatory = self.mandatory_step_ids()
        resolved = [
            sid for sid in mandatory if self.steps[sid].status == "resolved"
        ]
        unresolved = [
            sid
            for sid in mandatory
            if self.steps[sid].status not in {"resolved", "skipped"}
        ]
        return {
            "mandatory_total": len(mandatory),
            "mandatory_resolved": len(resolved),
            "mandatory_unresolved": unresolved,
            "coverage_rate": (
                (len(resolved) / len(mandatory)) if mandatory else 1.0
            ),
        }

    def is_complete(self) -> bool:
        cov = self.mandatory_coverage()
        return len(cov["mandatory_unresolved"]) == 0

    def _require(self, step_id: str) -> PlanStep:
        if step_id not in self.steps:
            raise PlanStateError(f"unknown step_id: {step_id}")
        return self.steps[step_id]


def topological_ready_order(steps: Iterable[PlanStep]) -> List[PlanStep]:
    return sorted(steps, key=lambda s: s.step_id)
