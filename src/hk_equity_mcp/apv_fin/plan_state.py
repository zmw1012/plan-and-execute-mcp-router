"""Dynamic APV-Fin plan state: model-owned plan fields vs runtime-only state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from hk_equity_mcp.apv_fin.schemas import STEP_STATUSES, STEP_TYPES

# Fields the model may emit / patch. Runtime owns everything else.
MODEL_OWNED_STEP_FIELDS = (
    "step_id",
    "step_type",
    "objective",
    "depends_on",
    "mandatory",
    "success_criteria",
    "retrieval_query_hint",
    "compute_spec",
    "notes",
    # v1.2/v1.3 retrieve contract (planner-owned; never operation_id)
    "required_capability",
    "required_outputs",
    "argument_slots",
    "source_preference",
    "fallback_capabilities",
    "semantic_output_requirements",
    "semantic_to_canonical_output_map",
)

RUNTIME_ONLY_STEP_FIELDS = (
    "status",
    "observation_ids",
    "attempt_count",
    "failure_reason",
    "selected_operation",
    "invoked_operation",
    "execution_result",
    "resolved_at",
)


class PlanStateError(ValueError):
    pass


def normalize_compute_spec(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PlanStateError("compute_spec must be an object or null")
    expression = str(raw.get("expression") or "").strip()
    if not expression:
        raise PlanStateError("compute_spec.expression must be non-empty")
    operand_refs = raw.get("operand_refs") or {}
    if not isinstance(operand_refs, dict):
        raise PlanStateError("compute_spec.operand_refs must be an object")
    refs = {str(k): str(v) for k, v in operand_refs.items()}
    rounding = raw.get("rounding")
    if rounding is not None and not isinstance(rounding, int):
        raise PlanStateError("compute_spec.rounding must be int or null")
    return {
        "expression": expression,
        "operand_refs": refs,
        "output_name": str(raw.get("output_name") or "result"),
        "output_unit": str(raw.get("output_unit") or ""),
        "rounding": rounding,
    }


@dataclass
class PlanStep:
    """Mixed view used by the scheduler: model fields + runtime fields.

    Model JSON / Monitor patches may only set MODEL_OWNED_STEP_FIELDS.
    """

    step_id: str
    step_type: str
    objective: str
    depends_on: List[str] = field(default_factory=list)
    mandatory: bool = True
    success_criteria: str = ""
    retrieval_query_hint: Optional[str] = None
    compute_spec: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    required_capability: Optional[str] = None
    required_outputs: List[str] = field(default_factory=list)
    argument_slots: Dict[str, Any] = field(default_factory=dict)
    source_preference: List[str] = field(default_factory=list)
    fallback_capabilities: List[str] = field(default_factory=list)
    semantic_output_requirements: List[str] = field(default_factory=list)
    semantic_to_canonical_output_map: Dict[str, str] = field(default_factory=dict)
    # Runtime-only:
    status: str = "pending"
    observation_ids: List[str] = field(default_factory=list)
    attempt_count: int = 0
    failure_reason: Optional[str] = None
    selected_operation: Optional[str] = None
    invoked_operation: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    resolved_at: Optional[str] = None

    def model_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "objective": self.objective,
            "depends_on": list(self.depends_on),
            "mandatory": self.mandatory,
            "success_criteria": self.success_criteria,
            "retrieval_query_hint": self.retrieval_query_hint,
            "compute_spec": deepcopy(self.compute_spec),
            "notes": self.notes,
            "required_capability": self.required_capability,
            "required_outputs": list(self.required_outputs),
            "argument_slots": deepcopy(self.argument_slots),
            "source_preference": list(self.source_preference),
            "fallback_capabilities": list(self.fallback_capabilities),
            "semantic_output_requirements": list(self.semantic_output_requirements),
            "semantic_to_canonical_output_map": dict(
                self.semantic_to_canonical_output_map
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, allow_runtime: bool = True) -> "PlanStep":
        step_type = str(raw.get("step_type") or "")
        if step_type not in STEP_TYPES:
            raise PlanStateError(f"invalid step_type: {step_type!r}")
        step_id = str(raw.get("step_id") or "").strip()
        if not step_id:
            raise PlanStateError("step_id must be non-empty")
        depends = list(raw.get("depends_on") or [])
        if not all(isinstance(d, str) and d for d in depends):
            raise PlanStateError(f"invalid depends_on for {step_id}")

        # Reject runtime fields when loading model-owned plan JSON.
        if not allow_runtime:
            leaked = [k for k in RUNTIME_ONLY_STEP_FIELDS if k in raw]
            if leaked:
                raise PlanStateError(
                    f"model plan must not include runtime fields: {sorted(set(leaked))}"
                )

        compute_spec = None
        if "compute_spec" in raw:
            compute_spec = normalize_compute_spec(raw.get("compute_spec"))
        if step_type == "compute" and compute_spec is None:
            raise PlanStateError(f"compute step {step_id!r} requires compute_spec")
        if step_type != "compute" and compute_spec is not None:
            raise PlanStateError(
                f"non-compute step {step_id!r} must have compute_spec=null"
            )

        required_capability = raw.get("required_capability")
        required_outputs = list(raw.get("required_outputs") or [])
        argument_slots = dict(raw.get("argument_slots") or {})
        source_preference = list(raw.get("source_preference") or [])
        fallback_capabilities = list(raw.get("fallback_capabilities") or [])
        semantic_output_requirements = list(
            raw.get("semantic_output_requirements") or []
        )
        semantic_to_canonical_output_map = {
            str(k): str(v)
            for k, v in dict(raw.get("semantic_to_canonical_output_map") or {}).items()
        }
        if step_type == "retrieve":
            if not required_capability or not str(required_capability).strip():
                raise PlanStateError(
                    f"retrieve step {step_id!r} requires required_capability"
                )
            required_capability = str(required_capability).strip()
            # Fail closed on unknown / informal capability that escaped compile.
            from hk_equity_mcp.apv_fin.capability_ontology import CAPABILITY_IDS

            if required_capability not in CAPABILITY_IDS:
                raise PlanStateError(
                    f"retrieve step {step_id!r} required_capability "
                    f"{required_capability!r} is not a formal CAPABILITY_IDS value"
                )
        else:
            required_capability = (
                str(required_capability).strip() if required_capability else None
            )

        status = str(raw.get("status") or "pending") if allow_runtime else "pending"
        if status not in STEP_STATUSES:
            raise PlanStateError(f"invalid step status: {status!r}")

        return cls(
            step_id=step_id,
            step_type=step_type,
            objective=str(raw.get("objective") or ""),
            depends_on=depends,
            mandatory=bool(raw.get("mandatory", True)),
            success_criteria=str(raw.get("success_criteria") or ""),
            retrieval_query_hint=raw.get("retrieval_query_hint"),
            compute_spec=compute_spec,
            notes=raw.get("notes"),
            required_capability=required_capability,
            required_outputs=[str(x) for x in required_outputs],
            argument_slots=argument_slots,
            source_preference=[str(x) for x in source_preference],
            fallback_capabilities=[str(x) for x in fallback_capabilities],
            semantic_output_requirements=[str(x) for x in semantic_output_requirements],
            semantic_to_canonical_output_map=semantic_to_canonical_output_map,
            status=status,
            observation_ids=list(raw.get("observation_ids") or []) if allow_runtime else [],
            attempt_count=int(raw.get("attempt_count") or 0) if allow_runtime else 0,
            failure_reason=raw.get("failure_reason") if allow_runtime else None,
            selected_operation=raw.get("selected_operation") if allow_runtime else None,
            invoked_operation=raw.get("invoked_operation") if allow_runtime else None,
            execution_result=raw.get("execution_result") if allow_runtime else None,
            resolved_at=raw.get("resolved_at") if allow_runtime else None,
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
        state._replace_steps(raw_steps, revision_note="initial_plan", from_model=True)
        return state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "steps": {sid: step.to_dict() for sid, step in self.steps.items()},
            "revision_log": deepcopy(self.revision_log),
        }

    def model_plan_dict(self) -> Dict[str, Any]:
        """Public plan view for LLM prompts — no runtime fields."""
        return {
            "revision": self.revision,
            "steps": [step.model_dict() for step in self.steps.values()],
        }

    def step_ids(self) -> List[str]:
        return list(self.steps.keys())

    def mandatory_step_ids(self) -> List[str]:
        return [sid for sid, step in self.steps.items() if step.mandatory]

    def _replace_steps(
        self,
        raw_steps: Sequence[Dict[str, Any]],
        *,
        revision_note: str,
        from_model: bool = False,
    ) -> None:
        parsed = [
            PlanStep.from_dict(raw, allow_runtime=not from_model) for raw in raw_steps
        ]
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
            deps_ok = all(self.steps[d].status == "resolved" for d in step.depends_on)
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
        step.attempt_count = int(step.attempt_count or 0) + 1

    def mark_resolved(
        self, step_id: str, *, observation_ids: Optional[Sequence[str]] = None
    ) -> None:
        from datetime import datetime, timezone

        step = self._require(step_id)
        step.status = "resolved"
        step.failure_reason = None
        step.resolved_at = datetime.now(timezone.utc).isoformat()
        if observation_ids:
            for oid in observation_ids:
                if oid not in step.observation_ids:
                    step.observation_ids.append(oid)
        self._recompute_ready()

    def resolve_steps_from_observation(
        self, step_ids: Sequence[str], observation_id: str
    ) -> List[str]:
        """Runtime helper: resolve multiple steps from one observation id."""
        resolved: List[str] = []
        for sid in step_ids:
            if sid not in self.steps:
                continue
            if self.steps[sid].status in {"resolved", "skipped"}:
                continue
            self.mark_resolved(sid, observation_ids=[observation_id])
            resolved.append(sid)
        return resolved

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

    def apply_monitor_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply exceptional replan patch (model-owned fields only)."""
        patch = patch or {}
        # Accept both v1 names and v1.1 names.
        completed = list(
            patch.get("completed_plan_step_ids")
            or patch.get("completed_steps")
            or []
        )
        updated = list(
            patch.get("updated_plan_steps") or patch.get("updated_steps") or []
        )
        new_steps = list(
            patch.get("new_plan_steps") or patch.get("new_steps") or []
        )
        removed = list(
            patch.get("removed_plan_step_ids")
            or patch.get("removed_steps")
            or []
        )

        for sid in completed:
            # Suggestion only — runtime must already have evidence; here we only
            # record the suggestion in the summary. Do NOT auto-resolve.
            if sid not in self.steps:
                raise PlanStateError(f"completed_plan_step_ids unknown: {sid!r}")

        for rem in removed:
            if rem not in self.steps:
                continue
            if self.steps[rem].status == "resolved":
                continue
            self.mark_skipped(rem, "removed_by_replan")
            del self.steps[rem]

        for upd in updated:
            if not isinstance(upd, dict):
                raise PlanStateError("updated_plan_steps entries must be objects")
            leaked = [k for k in RUNTIME_ONLY_STEP_FIELDS if k in upd]
            if leaked:
                raise PlanStateError(
                    f"updated_plan_steps must not include runtime fields: {leaked}"
                )
            sid = str(upd.get("step_id") or "")
            if sid not in self.steps:
                raise PlanStateError(f"updated_plan_steps references unknown step {sid!r}")
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
            if "compute_spec" in upd:
                step.compute_spec = normalize_compute_spec(upd.get("compute_spec"))
            if "required_capability" in upd:
                step.required_capability = upd.get("required_capability")
            if "required_outputs" in upd:
                step.required_outputs = list(upd.get("required_outputs") or [])
            if "argument_slots" in upd:
                step.argument_slots = dict(upd.get("argument_slots") or {})
            if "source_preference" in upd:
                step.source_preference = list(upd.get("source_preference") or [])
            if "fallback_capabilities" in upd:
                step.fallback_capabilities = list(upd.get("fallback_capabilities") or [])
            if "semantic_output_requirements" in upd:
                step.semantic_output_requirements = list(
                    upd.get("semantic_output_requirements") or []
                )
            if "semantic_to_canonical_output_map" in upd:
                step.semantic_to_canonical_output_map = {
                    str(k): str(v)
                    for k, v in dict(
                        upd.get("semantic_to_canonical_output_map") or {}
                    ).items()
                }
            if step.status in {"failed", "skipped"} and (
                "retrieval_query_hint" in upd
                or upd.get("objective") is not None
                or upd.get("notes") is not None
                or "compute_spec" in upd
            ):
                step.status = "pending"
                step.failure_reason = None

        for raw in new_steps:
            if not isinstance(raw, dict):
                raise PlanStateError("new_plan_steps entries must be objects")
            leaked = [k for k in RUNTIME_ONLY_STEP_FIELDS if k in raw]
            if leaked:
                raise PlanStateError(
                    f"new_plan_steps must not include runtime fields: {leaked}"
                )
            step = PlanStep.from_dict(raw, allow_runtime=False)
            if step.step_id in self.steps:
                raise PlanStateError(
                    f"new_plan_steps duplicates existing step_id {step.step_id!r}"
                )
            self.steps[step.step_id] = step

        self._validate_dependencies(self.steps)
        self._detect_cycles(self.steps)
        self._recompute_ready()
        self.revision += 1
        summary = {
            "revision": self.revision,
            "completed_plan_step_ids": completed,
            "updated_plan_steps": [u.get("step_id") for u in updated],
            "new_plan_steps": [
                s.get("step_id") if isinstance(s, dict) else None for s in new_steps
            ],
            "removed_plan_step_ids": removed,
        }
        self.revision_log.append(summary)
        return summary

    def mandatory_coverage(self) -> Dict[str, Any]:
        mandatory = self.mandatory_step_ids()
        resolved = [sid for sid in mandatory if self.steps[sid].status == "resolved"]
        unresolved = [
            sid
            for sid in mandatory
            if self.steps[sid].status not in {"resolved", "skipped"}
        ]
        return {
            "mandatory_total": len(mandatory),
            "mandatory_resolved": len(resolved),
            "mandatory_unresolved": unresolved,
            "coverage_rate": (len(resolved) / len(mandatory)) if mandatory else 1.0,
        }

    def is_complete(self) -> bool:
        cov = self.mandatory_coverage()
        return len(cov["mandatory_unresolved"]) == 0

    def non_answer_mandatory_resolved(self) -> bool:
        for sid, step in self.steps.items():
            if not step.mandatory:
                continue
            if step.step_type == "answer":
                continue
            if step.status != "resolved":
                return False
        return True

    def _require(self, step_id: str) -> PlanStep:
        if step_id not in self.steps:
            raise PlanStateError(f"unknown step_id: {step_id}")
        return self.steps[step_id]


def topological_ready_order(steps: Iterable[PlanStep]) -> List[PlanStep]:
    return sorted(steps, key=lambda s: s.step_id)
