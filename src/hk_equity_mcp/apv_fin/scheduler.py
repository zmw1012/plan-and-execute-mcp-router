"""Step scheduler: only retrieve steps enter the MCP operation resolver."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.calculator import (
    CalculatorError,
    compute_trace_record,
    evaluate_compute_payload,
)
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.plan_state import PlanStep, PlanState
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_REASON_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
)
from hk_equity_mcp.apv_fin.reranker import RetrievalContext, rerank_candidates
from hk_equity_mcp.apv_fin.tool_binding import (
    annotate_selection_flags,
    build_deterministic_binding_proposal,
    grounding_from_context,
    run_tool_binder,
    select_candidates_to_describe,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer


def build_retrieval_query(step: PlanStep, task_spec: Optional[Dict[str, Any]] = None) -> str:
    task_spec = task_spec or {}
    parts = [step.objective]
    if step.retrieval_query_hint:
        parts.append(str(step.retrieval_query_hint))
    for key in ("entities", "metrics", "periods"):
        vals = task_spec.get(key) or []
        if vals:
            parts.append(f"{key}: " + ", ".join(str(v) for v in vals))
    return " | ".join(p for p in parts if p).strip()


def _intent_blob(
    task_spec: Optional[Dict[str, Any]] = None,
    step: Optional[PlanStep] = None,
    question: str = "",
) -> str:
    task_spec = task_spec or {}
    parts = [
        str(task_spec.get("goal") or ""),
        " ".join(str(v) for v in (task_spec.get("metrics") or [])),
        " ".join(str(v) for v in (task_spec.get("constraints") or [])),
        str(question or ""),
    ]
    if step is not None:
        parts.extend(
            [
                str(step.objective or ""),
                str(step.retrieval_query_hint or ""),
                str(step.success_criteria or ""),
                str(step.notes or ""),
            ]
        )
    return " ".join(parts).lower()


def context_from_task_spec(
    task_spec: Optional[Dict[str, Any]] = None,
    step: Optional[PlanStep] = None,
    question: str = "",
) -> RetrievalContext:
    """Build retrieval context from question + task_spec + retrieve step text.

    Step/question text is required so market-close retrieves do not inherit empty
    domain hints when the planner used a generic metric token.
    """
    task_spec = task_spec or {}
    domain_hints: List[str] = []
    required_provides: List[str] = []
    source_prefs: List[str] = [
        "official_disclosure",
        "exchange_feed",
        "structured_vendor",
    ]
    market_keys = (
        "price",
        "close",
        "ohlc",
        "ohlcv",
        "volume",
        "turnover",
        "return",
        "index",
        "quote",
        "market",
        "hist",
        "history",
        "open",
        "high",
        "low",
    )
    financial_keys = (
        "revenue",
        "eps",
        "ifrs",
        "profit",
        "net income",
        "net_income",
        "filing",
        "financial",
        "balance sheet",
        "income statement",
        "indicator",
        "ratio",
    )
    news_keys = ("news", "media", "headline", "press")
    disclosure_keys = (
        "announcement",
        "disclosure",
        "hkex",
        "filing pdf",
        "annual report",
        "interim report",
    )

    def _domains_from(text: str) -> List[str]:
        found: List[str] = []
        if any(k in text for k in market_keys):
            found.append("market")
        if any(k in text for k in financial_keys):
            found.append("financial")
        if any(k in text for k in news_keys):
            found.append("news")
        if any(k in text for k in disclosure_keys):
            found.append("financial")
        return list(dict.fromkeys(found))

    # Prefer question + task_spec domain signals. Step retrieval hints must not
    # broaden a clear market/financial intent (e.g. stale "financial indicator" hint
    # on a close-price question).
    primary_blob = _intent_blob(task_spec, step=None, question=question)
    step_blob = _intent_blob({}, step, question="")
    blob = _intent_blob(task_spec, step, question=question)
    primary_domains = _domains_from(primary_blob)
    if primary_domains:
        domain_hints.extend(primary_domains)
    else:
        domain_hints.extend(_domains_from(blob))

    if "market" in domain_hints:
        required_provides.extend(["close", "ohlcv", "price"])
    if "financial" in domain_hints:
        required_provides.extend(
            ["eps", "revenue", "net_income", "financial_indicators", "financial_report_rows"]
        )
    if "news" in domain_hints:
        pass
    if any(k in primary_blob or k in step_blob for k in disclosure_keys):
        if "financial" not in domain_hints:
            domain_hints.append("financial")
        source_prefs = [
            "official_disclosure",
            "structured_vendor",
            "exchange_feed",
        ]
        required_provides.extend(
            ["announcement_list", "announcement_pdf", "filing_text", "disclosure"]
        )
    # Deduplicate while preserving order.
    domain_hints = list(dict.fromkeys(domain_hints))
    required_provides = list(dict.fromkeys(required_provides))
    temporal: List[str] = []
    if task_spec.get("periods") or any(
        k in blob for k in ("fy", "date", "period", "day", "as of", "as-of")
    ):
        temporal.extend(["fiscal_period", "date_range", "point_in_time"])
    return RetrievalContext(
        domain_hints=domain_hints,
        market_scope="hk",
        entity_scope_hints=["hk_listed_issuer", "hk_equity", "index"],
        identifier_types=["hk_code_5digit", "hk_code_4digit", "hk_ticker_dot", "index_symbol"],
        temporal_needs=temporal,
        source_type_prefs=source_prefs,
        metric_basis_hints=list(task_spec.get("metric_basis_hints") or []),
        required_provides=required_provides,
        cutoff_date=task_spec.get("cutoff_date") or "2026-08-12",
        disallow_post_cutoff_spot=True,
    )


class BudgetTracker:
    def __init__(self, limits: Dict[str, Any]):
        self.limits = {
            "max_plan_revisions": int(limits.get("max_plan_revisions", 2)),
            "max_targeted_retrievals_per_step": int(
                limits.get("max_targeted_retrievals_per_step", 2)
            ),
            "max_total_mcp_invocations": int(limits.get("max_total_mcp_invocations", 8)),
            # Exceptional replan Monitor only (normal progress is deterministic).
            "max_exception_replan_calls_per_task": int(
                limits.get("max_exception_replan_calls_per_task", 2)
            ),
            "max_llm_monitor_rounds": int(
                limits.get(
                    "max_llm_monitor_rounds",
                    limits.get("max_exception_replan_calls_per_task", 2),
                )
            ),
            "max_same_operation_repeats": int(limits.get("max_same_operation_repeats", 1)),
            "max_final_repair_rounds": int(
                limits.get("max_final_repair_rounds", limits.get("max_final_repairs", 1))
            ),
            "max_binding_repairs": int(limits.get("max_binding_repairs", 1)),
            "max_idle_monitor_rounds": int(limits.get("max_idle_monitor_rounds", 1)),
            # Global per-task generation cap (planner/binding/exceptional replan/final).
            "max_total_llm_calls_per_task": int(
                limits.get("max_total_llm_calls_per_task", 10)
            ),
            "max_projected_observation_chars_per_step": int(
                limits.get("max_projected_observation_chars_per_step", 6000)
            ),
            "max_total_ledger_prompt_chars": int(
                limits.get("max_total_ledger_prompt_chars", 16000)
            ),
        }
        self.plan_revisions = 0
        self.monitor_rounds = 0
        self.exception_replan_calls = 0
        self.idle_monitor_rounds = 0
        self.mcp_invocations = 0
        self.retrievals_per_step: Dict[str, int] = {}
        self.operation_invoke_counts: Dict[str, int] = {}
        self.last_failed_invoke_signatures: set = set()
        self.successful_invoke_signatures: set = set()
        self.repeated_signature_block_count = 0
        self.final_repair_rounds = 0
        self.llm_calls = 0
        self.calls_by_stage: Dict[str, int] = {}
        self.stopping_reason: Optional[str] = None
        self.deterministic_step_transitions = 0
        self.dependency_auto_expansion_count = 0
        self.schema_description_expansion_count = 0
        self.raw_observation_chars = 0
        self.projected_observation_chars = 0
        self.exception_trigger: Optional[str] = None

    @property
    def llm_call_budget(self) -> int:
        return int(self.limits["max_total_llm_calls_per_task"])

    @property
    def llm_calls_remaining(self) -> int:
        return max(0, self.llm_call_budget - self.llm_calls)

    @property
    def llm_budget_exhausted(self) -> bool:
        return self.llm_calls >= self.llm_call_budget

    def check_llm_budget_or_raise(self, *, stage: str) -> None:
        """Call before issuing the next generation request."""
        from hk_equity_mcp.apv_fin.llm_budget import LlmCallBudgetExhausted

        if self.llm_calls >= self.llm_call_budget:
            self.stopping_reason = "resource_limit_failure"
            raise LlmCallBudgetExhausted(
                used=self.llm_calls, limit=self.llm_call_budget, stage=stage
            )

    def record_llm_call(self, *, stage: str) -> None:
        """Record one generation call after budget check passes (before/at request)."""
        self.llm_calls += 1
        key = str(stage or "unspecified")
        self.calls_by_stage[key] = int(self.calls_by_stage.get(key) or 0) + 1

    def llm_budget_fields(self) -> Dict[str, Any]:
        return {
            "llm_call_budget": self.llm_call_budget,
            "llm_calls_used": self.llm_calls,
            "llm_calls_remaining": self.llm_calls_remaining,
            "llm_budget_exhausted": self.llm_budget_exhausted,
            "calls_by_stage": dict(self.calls_by_stage),
            "exception_replan_calls": self.exception_replan_calls,
            "max_exception_replan_calls_per_task": self.limits[
                "max_exception_replan_calls_per_task"
            ],
        }

    def snapshot(self) -> Dict[str, Any]:
        raw = self.raw_observation_chars
        proj = self.projected_observation_chars
        return {
            "limits": dict(self.limits),
            "plan_revisions": self.plan_revisions,
            "monitor_rounds": self.monitor_rounds,
            "exception_replan_calls": self.exception_replan_calls,
            "idle_monitor_rounds": self.idle_monitor_rounds,
            "mcp_invocations": self.mcp_invocations,
            "retrievals_per_step": dict(self.retrievals_per_step),
            "operation_invoke_counts": dict(self.operation_invoke_counts),
            "final_repair_rounds": self.final_repair_rounds,
            "llm_calls": self.llm_calls,
            "calls_by_stage": dict(self.calls_by_stage),
            "stopping_reason": self.stopping_reason,
            "exception_trigger": self.exception_trigger,
            "deterministic_step_transitions": self.deterministic_step_transitions,
            "dependency_auto_expansion_count": self.dependency_auto_expansion_count,
            "schema_description_expansion_count": self.schema_description_expansion_count,
            "repeated_signature_block_count": self.repeated_signature_block_count,
            "raw_observation_chars": raw,
            "projected_observation_chars": proj,
            "compression_ratio": (proj / raw) if raw else None,
            **self.llm_budget_fields(),
        }

    def exhausted(self) -> Optional[str]:
        # LLM call cap is enforced at request time by BudgetedLLMClient
        # (non-LLM steps may still proceed when the generation budget is spent).
        if self.plan_revisions > self.limits["max_plan_revisions"]:
            return "max_plan_revisions"
        if self.exception_replan_calls > self.limits[
            "max_exception_replan_calls_per_task"
        ]:
            return "max_exception_replan_calls_per_task"
        if self.monitor_rounds > self.limits["max_llm_monitor_rounds"]:
            return "max_llm_monitor_rounds"
        if self.mcp_invocations > self.limits["max_total_mcp_invocations"]:
            return "max_total_mcp_invocations"
        if self.idle_monitor_rounds > self.limits["max_idle_monitor_rounds"]:
            return "max_idle_monitor_rounds"
        return None


def execute_retrieve_step(
    *,
    step: PlanStep,
    task_spec: Dict[str, Any],
    resolver: OperationResolverServer,
    metadata_by_id: Dict[str, Dict[str, Any]],
    runtime_operation_ids: Sequence[str],
    ledger: EvidenceLedger,
    budget: BudgetTracker,
    top_k: int = 20,
    llm: Optional[LLMClient] = None,
    question: str = "",
    invoke_arguments: Optional[Dict[str, Any]] = None,
    select_operation_id: Optional[str] = None,
    allow_forced_select: bool = False,
    offline_tool_binding: bool = False,
    _reformulation_depth: int = 0,
) -> Dict[str, Any]:
    """Retrieve: find → hard-filter/rerank → describe Top-N → Tool Binding → validate → invoke.

    Empty-argument invoke is forbidden. Forced select is live-forbidden and must not
    bypass Tool Binding validation when used in offline unit tests.
    """
    if select_operation_id and not allow_forced_select:
        return {
            "ok": False,
            "error": "forced_select_forbidden_in_live_mode",
            "step_id": step.step_id,
            "mcp_called": False,
            "binding_decision": None,
        }
    count = budget.retrievals_per_step.get(step.step_id, 0)
    if count >= budget.limits["max_targeted_retrievals_per_step"]:
        return {
            "ok": False,
            "error": "max_targeted_retrievals_per_step",
            "step_id": step.step_id,
            "mcp_called": False,
            "retrieval_reformulation_count": _reformulation_depth,
        }
    budget.retrievals_per_step[step.step_id] = count + 1

    query = build_retrieval_query(step, task_spec)
    find_out = resolver.find_operations(query, top_k=top_k)
    candidates = list(find_out.get("candidates") or [])
    ctx = context_from_task_spec(task_spec, step, question=question)
    ranked = rerank_candidates(
        candidates,
        metadata_by_id,
        ctx,
        runtime_operation_ids=runtime_operation_ids,
    )
    reranked = list(ranked["reranked"] or [])

    # Unit-test escape hatch: inject a candidate so binding can select it, still
    # requiring describe + binding validation (never empty-arg invoke).
    forced_select_used = False
    if select_operation_id and allow_forced_select:
        if not any(r["operation_id"] == select_operation_id for r in reranked):
            forced_select_used = True
            reranked = [
                {
                    "operation_id": select_operation_id,
                    "original_rank": None,
                    "base_score": 0.0,
                    "compatibility_tier": "A",
                    "compatibility_reasons": {
                        "supports": ["forced_select_override"],
                        "partial": [],
                        "conflicts": [],
                    },
                    "hard_filtered": False,
                    "final_rank": 0,
                    "candidate": {"operation_id": select_operation_id, "score": 0.0},
                }
            ] + reranked
            ranked = dict(ranked)
            ranked["reranked"] = reranked

    to_describe = select_candidates_to_describe(
        reranked, required_provides=list(ctx.required_provides or [])
    )
    described_ids: List[str] = []
    described_contracts: Dict[str, Dict[str, Any]] = {}
    describe_outputs: List[Dict[str, Any]] = []
    for row in to_describe:
        oid = row["operation_id"]
        dout = resolver.describe_operation(oid)
        describe_outputs.append(dout)
        if dout.get("success") and dout.get("contract"):
            described_ids.append(oid)
            described_contracts[oid] = dout["contract"]

    grounding = grounding_from_context(
        question=question, task_spec=task_spec, step=step, ledger=ledger
    )
    tier_by_id = {r["operation_id"]: r.get("compatibility_tier") for r in reranked}
    allowed_ids = [r["operation_id"] for r in reranked]
    blocked_ids = [
        r["operation_id"]
        for r in (ranked.get("hard_filtered") or [])
        if "blocked" in str(r.get("hard_filter_reason") or "")
        or "unavailable" in str(r.get("hard_filter_reason") or "")
    ]
    candidate_summaries = [
        {
            "operation_id": r["operation_id"],
            "final_rank": r.get("final_rank"),
            "original_rank": r.get("original_rank"),
            "base_score": r.get("base_score"),
            "compatibility_tier": r.get("compatibility_tier"),
            "compatibility_reasons": r.get("compatibility_reasons"),
        }
        for r in reranked[:10]
    ]

    # Offline / mock: build a synthetic binding that still passes the same validator.
    # Use a one-shot ScriptedLLM under BudgetedLLMClient so tool_binding counts toward
    # the per-task cap without consuming the main trajectory script queue.
    use_offline_binding = bool(offline_tool_binding) or llm is None
    offline_proposal = None
    if use_offline_binding:
        offline_proposal = build_deterministic_binding_proposal(
            described_rows=to_describe,
            described_contracts=described_contracts,
            grounding=grounding,
        )
        if (
            select_operation_id
            and allow_forced_select
            and select_operation_id in described_contracts
        ):
            from hk_equity_mcp.apv_fin.argument_grounding import (
                propose_arguments_for_schema,
            )

            args, sources, missing = propose_arguments_for_schema(
                json_schema=(
                    described_contracts[select_operation_id].get("json_schema") or {}
                ),
                grounding=grounding,
            )
            if invoke_arguments:
                args.update(dict(invoke_arguments))
                for k in invoke_arguments:
                    sources.setdefault(k, "plan_step")
            schema_req = list(
                (
                    described_contracts[select_operation_id].get("json_schema") or {}
                ).get("required")
                or []
            )
            if all(k in args for k in schema_req):
                offline_proposal = {
                    "schema_version": "apv_fin_tool_binding_v1",
                    "decision": "select",
                    "selected_operation_id": select_operation_id,
                    "arguments": args,
                    "argument_sources": sources,
                    "expected_output": "rows",
                    "decision_reason": (
                        "Offline forced-select binding with grounded/supplied args."
                    ),
                    "reformulated_query": None,
                }

    if use_offline_binding:
        from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
        from hk_equity_mcp.apv_fin.llm_budget import BudgetedLLMClient

        one_shot = ScriptedLLMClient(
            [json.dumps(offline_proposal, ensure_ascii=False)]
        )
        binder_llm: LLMClient = BudgetedLLMClient(one_shot, budget)
    else:
        binder_llm = llm  # type: ignore[assignment]

    binding = run_tool_binder(
        binder_llm,
        question=question or str(task_spec.get("goal") or step.objective or ""),
        task_spec=task_spec,
        step={
            "step_id": step.step_id,
            "objective": step.objective,
            "retrieval_query_hint": step.retrieval_query_hint,
            "success_criteria": step.success_criteria,
            "notes": step.notes,
        },
        reranked_summaries=candidate_summaries,
        described_contracts=described_contracts,
        grounding=grounding,
        budget_snapshot=budget.snapshot(),
        allowed_operation_ids=allowed_ids,
        blocked_operation_ids=blocked_ids,
        tier_by_id=tier_by_id,
        offline_proposal=None,
    )

    annotated = annotate_selection_flags(
        reranked,
        described_ids=described_ids,
        selected_id=(binding.get("selected_operation_id") if binding.get("ok") else None),
    )
    ranked = dict(ranked)
    ranked["reranked"] = annotated

    base_out: Dict[str, Any] = {
        "step_id": step.step_id,
        "query": query,
        "find": find_out,
        "rerank": ranked,
        "described_candidate_ids": described_ids,
        "describe_outputs": describe_outputs,
        "describe": describe_outputs[0] if describe_outputs else None,
        "binding": binding,
        "binding_decision": binding.get("decision"),
        "binding_selected_operation": binding.get("selected_operation_id"),
        "grounded_arguments": binding.get("arguments"),
        "argument_sources": binding.get("argument_sources"),
        "argument_validation": binding.get("argument_validation"),
        "binding_repair_count": binding.get("repair_count"),
        "retrieval_reformulation_count": _reformulation_depth,
        "forced_select_used": forced_select_used,
        "mcp_called": False,
        "chosen_operation_id": None,
        "invoke": None,
        "validation_passed": False,
        "invoked": False,
        "successful": False,
    }

    if not binding.get("ok"):
        base_out.update(
            {
                "ok": False,
                "error": binding.get("error") or "tool_binding_fail_closed",
            }
        )
        return base_out

    decision = binding.get("decision")
    # One-shot schema-description expansion: describe a capability-compatible
    # undescibed candidate without re-running embedding retrieval.
    if (
        decision in {"expand_search", "unsupported"}
        and int(getattr(budget, "schema_description_expansion_count", 0) or 0) < 1
    ):
        undescribed = [
            r
            for r in reranked
            if r["operation_id"] not in described_ids
            and r.get("compatibility_tier") in {"A", "B", "C"}
        ]
        # Prefer provides-overlap / hist|daily when close required.
        undescribed = sorted(
            undescribed,
            key=lambda r: (
                0
                if (
                    "hist" in str(r.get("operation_id") or "").lower()
                    or "daily" in str(r.get("operation_id") or "").lower()
                    or "provides_overlap"
                    in ((r.get("compatibility_reasons") or {}).get("supports") or [])
                )
                else 1,
                int(r.get("final_rank") or 10**6),
            ),
        )
        if undescribed:
            expand_row = undescribed[0]
            eoid = expand_row["operation_id"]
            dout = resolver.describe_operation(eoid)
            describe_outputs.append(dout)
            budget.schema_description_expansion_count = (
                int(getattr(budget, "schema_description_expansion_count", 0) or 0) + 1
            )
            if dout.get("success") and dout.get("contract"):
                described_ids.append(eoid)
                described_contracts[eoid] = dout["contract"]
                to_describe = list(to_describe) + [expand_row]
                if use_offline_binding:
                    from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
                    from hk_equity_mcp.apv_fin.llm_budget import BudgetedLLMClient

                    offline_proposal = build_deterministic_binding_proposal(
                        described_rows=to_describe,
                        described_contracts=described_contracts,
                        grounding=grounding,
                    )
                    one_shot = ScriptedLLMClient(
                        [json.dumps(offline_proposal, ensure_ascii=False)]
                    )
                    binder_llm = BudgetedLLMClient(one_shot, budget)
                binding = run_tool_binder(
                    binder_llm,
                    question=question
                    or str(task_spec.get("goal") or step.objective or ""),
                    task_spec=task_spec,
                    step={
                        "step_id": step.step_id,
                        "objective": step.objective,
                        "retrieval_query_hint": step.retrieval_query_hint,
                        "success_criteria": step.success_criteria,
                        "notes": step.notes,
                    },
                    reranked_summaries=candidate_summaries,
                    described_contracts=described_contracts,
                    grounding=grounding,
                    budget_snapshot=budget.snapshot(),
                    allowed_operation_ids=allowed_ids,
                    blocked_operation_ids=blocked_ids,
                    tier_by_id=tier_by_id,
                    offline_proposal=None,
                )
                base_out["binding"] = binding
                base_out["binding_decision"] = binding.get("decision")
                base_out["binding_selected_operation"] = binding.get(
                    "selected_operation_id"
                )
                base_out["grounded_arguments"] = binding.get("arguments")
                base_out["argument_sources"] = binding.get("argument_sources")
                base_out["argument_validation"] = binding.get("argument_validation")
                base_out["described_candidate_ids"] = described_ids
                base_out["schema_description_expansion"] = {
                    "operation_id": eoid,
                    "described_rank": expand_row.get("final_rank"),
                    "source": "schema_description_expansion",
                }
                decision = binding.get("decision") if binding.get("ok") else decision

    if decision == "expand_search":
        reform = (binding.get("payload") or {}).get("reformulated_query")
        if (
            reform
            and _reformulation_depth + 1
            < budget.limits["max_targeted_retrievals_per_step"]
        ):
            step.retrieval_query_hint = str(reform)
            return execute_retrieve_step(
                step=step,
                task_spec=task_spec,
                resolver=resolver,
                metadata_by_id=metadata_by_id,
                runtime_operation_ids=runtime_operation_ids,
                ledger=ledger,
                budget=budget,
                top_k=top_k,
                llm=llm,
                question=question,
                invoke_arguments=invoke_arguments,
                select_operation_id=select_operation_id,
                allow_forced_select=allow_forced_select,
                offline_tool_binding=offline_tool_binding,
                _reformulation_depth=_reformulation_depth + 1,
            )
        base_out.update(
            {
                "ok": False,
                "error": "binding_expand_search_exhausted",
            }
        )
        return base_out

    if decision != "select":
        base_out.update({"ok": False, "error": f"binding_decision_{decision}"})
        return base_out

    oid = binding.get("selected_operation_id")
    args = dict(binding.get("arguments") or {})
    # Never allow empty required invoke — validator already enforces, belt-and-suspenders:
    if args == {} and described_contracts.get(oid or "", {}).get("json_schema", {}).get(
        "required"
    ):
        base_out.update(
            {
                "ok": False,
                "error": "refusing_empty_arguments_invoke",
                "chosen_operation_id": oid,
            }
        )
        return base_out

    # Identical failed OR successful invocation guard (no empty re-list loops).
    invoke_sig = json.dumps(
        {"operation_id": oid, "arguments": args}, sort_keys=True, ensure_ascii=False
    )
    prior_fail = getattr(budget, "last_failed_invoke_signatures", None)
    if prior_fail is None:
        budget.last_failed_invoke_signatures = set()
        prior_fail = budget.last_failed_invoke_signatures
    prior_ok = getattr(budget, "successful_invoke_signatures", None)
    if prior_ok is None:
        budget.successful_invoke_signatures = set()
        prior_ok = budget.successful_invoke_signatures
    if invoke_sig in prior_fail or invoke_sig in prior_ok:
        budget.repeated_signature_block_count = (
            int(getattr(budget, "repeated_signature_block_count", 0) or 0) + 1
        )
        base_out.update(
            {
                "ok": False,
                "error": "identical_successful_or_failed_invocation_not_repeated",
                "chosen_operation_id": oid,
                "grounded_arguments": args,
                "repeated_signature_blocked": True,
            }
        )
        return base_out

    repeats = budget.operation_invoke_counts.get(oid, 0)
    if repeats > budget.limits["max_same_operation_repeats"]:
        base_out.update(
            {
                "ok": False,
                "error": "max_same_operation_repeats",
                "chosen_operation_id": oid,
            }
        )
        return base_out
    if budget.mcp_invocations >= budget.limits["max_total_mcp_invocations"]:
        base_out.update(
            {
                "ok": False,
                "error": "max_total_mcp_invocations",
                "chosen_operation_id": oid,
            }
        )
        return base_out

    base_out["validation_passed"] = True
    invoke_out = resolver.invoke_api(oid, args)
    budget.mcp_invocations += 1
    budget.operation_invoke_counts[oid] = repeats + 1
    success = bool(invoke_out.get("success"))
    if not success:
        prior_fail.add(invoke_sig)
    else:
        prior_ok.add(invoke_sig)

    from hk_equity_mcp.apv_fin.projection import project_mcp_result

    projection = None
    if success:
        projection = project_mcp_result(
            operation_id=str(oid),
            arguments=args,
            raw_result=(invoke_out or {}).get("result"),
            task_spec=task_spec,
            step_objective=str(step.objective or ""),
        )
        budget.raw_observation_chars += int(projection.get("raw_chars") or 0)
        budget.projected_observation_chars += int(
            projection.get("projected_chars") or 0
        )

    ev = ledger.add(
        step_id=step.step_id,
        claim=f"retrieve/invoke {oid}",
        source_type="mcp_invoke",
        operation_id=oid,
        raw_value=invoke_out,
        normalized_value=(
            (projection or {}).get("projected")
            if projection
            else (invoke_out or {}).get("result")
        ),
        provenance="UnifiedRealMcpGateway via OperationResolverServer",
        validation_status="valid" if success else "invalid",
        **({"projection": projection} if projection else {}),
    )
    base_out.update(
        {
            "ok": success,
            "chosen_operation_id": oid,
            "invoke": invoke_out,
            "projection": projection,
            "evidence_id": ev.evidence_id,
            "mcp_called": True,
            "invoked": True,
            "successful": success,
            "error": None if success else invoke_out.get("error"),
        }
    )
    # Compact observation for subsequent LLM prompts (raw stays in ledger/artifact).
    if projection:
        base_out["compact_observation"] = {
            "step_id": step.step_id,
            "operation_id": oid,
            "arguments": args,
            "projected": projection.get("projected"),
            "raw_artifact_hash": projection.get("raw_artifact_hash"),
            "truncated": projection.get("truncated"),
        }
    return base_out


def _resolve_operand_refs(
    operand_refs: Dict[str, str], ledger: EvidenceLedger
) -> tuple[Dict[str, float], List[str]]:
    """Bind compute_spec.operand_refs from verified ledger values.

    Ref formats: "s1.net_income", "evidence_id.field", or plain evidence field name.
    """
    operands: Dict[str, float] = {}
    missing: List[str] = []
    items = list(ledger.items)

    def _numeric_from(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for k in ("value", "result", "amount", "close", "normalized_value"):
                if isinstance(value.get(k), (int, float)) and not isinstance(
                    value.get(k), bool
                ):
                    return float(value[k])
        return None

    for name, ref in operand_refs.items():
        ref_s = str(ref or "").strip()
        found: Optional[float] = None
        step_prefix = None
        field = ref_s
        if "." in ref_s:
            step_prefix, field = ref_s.split(".", 1)
        for item in reversed(items):
            if str(item.validation_status or "") not in {"valid", "verified"}:
                continue
            if step_prefix and item.step_id != step_prefix and item.evidence_id != step_prefix:
                continue
            candidates = [
                item.normalized_value,
                item.raw_value,
                (item.extras or {}).get("projection"),
            ]
            for cand in candidates:
                if cand is None:
                    continue
                if isinstance(cand, dict):
                    # Search rows / flat keys.
                    if field in cand and _numeric_from(cand[field]) is not None:
                        found = _numeric_from(cand[field])
                        break
                    rows = cand.get("rows") if isinstance(cand.get("rows"), list) else None
                    if rows is None and isinstance(cand.get("projected"), dict):
                        rows = cand["projected"].get("rows")
                    if isinstance(rows, list):
                        for row in rows:
                            if isinstance(row, dict) and field in row:
                                found = _numeric_from(row[field])
                                if found is not None:
                                    break
                    if found is not None:
                        break
                    # Case-insensitive key match.
                    for k, v in cand.items():
                        if str(k).lower() == field.lower():
                            found = _numeric_from(v)
                            if found is not None:
                                break
                else:
                    if not step_prefix or field in {"value", "result", "amount"}:
                        found = _numeric_from(cand)
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            missing.append(name)
        else:
            operands[name] = found
    return operands, missing


def _extract_compute_payload_from_step(
    step: PlanStep,
    ledger: EvidenceLedger,
    scripted_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve compute_spec → expression/operands without any LLM call / eval()."""
    if scripted_payload is not None:
        return {"ok": True, "payload": dict(scripted_payload), "missing": []}

    spec = step.compute_spec
    if isinstance(spec, dict) and spec.get("expression"):
        refs = dict(spec.get("operand_refs") or {})
        operands, missing = _resolve_operand_refs(refs, ledger)
        if missing:
            return {
                "ok": False,
                "error": "missing_compute_operands",
                "missing": missing,
                "payload": {
                    "schema_version": "apv_fin_compute_v1",
                    "expression": spec["expression"],
                    "operands": operands,
                    "unit": spec.get("output_unit"),
                    "notes": spec.get("output_name"),
                    "rounding": spec.get("rounding"),
                },
            }
        payload = {
            "schema_version": "apv_fin_compute_v1",
            "expression": spec["expression"],
            "operands": operands,
            "unit": spec.get("output_unit"),
            "notes": spec.get("output_name"),
            "rounding": spec.get("rounding"),
        }
        return {"ok": True, "payload": payload, "missing": []}

    # Legacy fallback: structured JSON in notes (offline fixtures).
    for raw in (step.notes, step.objective):
        if not raw or not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text.startswith("{"):
            continue
        try:
            obj = json.loads(text)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict) and obj.get("expression") and obj.get("operands") is not None:
            payload = {
                "schema_version": obj.get("schema_version") or "apv_fin_compute_v1",
                "expression": obj["expression"],
                "operands": obj["operands"],
                "unit": obj.get("unit"),
                "notes": obj.get("notes"),
            }
            return {"ok": True, "payload": payload, "missing": []}
    return {
        "ok": False,
        "error": "compute_missing_expression_operands",
        "missing": [],
        "payload": None,
    }


def execute_compute_step(
    *,
    llm: Optional[LLMClient] = None,  # retained for call-site compat; never used
    question: str = "",
    step: PlanStep,
    ledger: EvidenceLedger,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    scripted_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic compute via compute_spec. Never calls LLM/MCP/eval()."""
    del llm, question, temperature, max_tokens  # unused; keep signature stable
    extracted = _extract_compute_payload_from_step(step, ledger, scripted_payload)
    if not extracted.get("ok"):
        err = extracted.get("error") or "compute_missing_expression_operands"
        payload = extracted.get("payload") or {}
        return {
            "ok": False,
            "mcp_called": False,
            "llm_called": False,
            "error": err,
            "missing_operands": extracted.get("missing") or [],
            "unresolved": True,
            "repair_count": 0,
            "llm": None,
            "payload": payload or None,
            "compute_trace": compute_trace_record(
                step_id=step.step_id,
                payload=payload or {},
                error=err,
            ),
        }

    payload = extracted["payload"]
    try:
        evaluated = evaluate_compute_payload(payload)
        rounding = payload.get("rounding")
        if rounding is not None and isinstance(evaluated.get("result"), (int, float)):
            evaluated = dict(evaluated)
            evaluated["result"] = round(float(evaluated["result"]), int(rounding))
        trace = compute_trace_record(
            step_id=step.step_id, payload=payload, evaluated=evaluated
        )
        ev = ledger.add(
            step_id=step.step_id,
            claim=f"compute {payload.get('expression')}",
            source_type="compute",
            raw_value=payload,
            normalized_value=evaluated.get("result"),
            currency_or_unit=payload.get("unit"),
            provenance="deterministic_safe_arithmetic",
            validation_status="valid",
            formula=payload.get("expression"),
            operands=payload.get("operands"),
        )
        return {
            "ok": True,
            "mcp_called": False,
            "llm_called": False,
            "error": None,
            "unresolved": False,
            "repair_count": 0,
            "llm": None,
            "payload": payload,
            "evaluated": evaluated,
            "compute_trace": trace,
            "evidence_id": ev.evidence_id,
            "value": evaluated.get("result"),
        }
    except CalculatorError as exc:
        trace = compute_trace_record(
            step_id=step.step_id, payload=payload, error=str(exc)
        )
        return {
            "ok": False,
            "mcp_called": False,
            "llm_called": False,
            "error": str(exc),
            "unresolved": True,
            "repair_count": 0,
            "llm": None,
            "payload": payload,
            "compute_trace": trace,
        }


def execute_reason_step(
    *,
    llm: LLMClient,
    question: str,
    step: PlanStep,
    ledger: EvidenceLedger,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    scripted_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reason/verify steps never call MCP; store concise summary only."""
    if scripted_payload is not None:
        parsed = scripted_payload
        llm_meta = None
    else:
        messages = [
            {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": APV_FIN_REASON_USER_TEMPLATE.format(
                    question=question,
                    step_id=step.step_id,
                    step_type=step.step_type,
                    objective=step.objective,
                    evidence_json=json.dumps(ledger.to_list(), ensure_ascii=False)[:8000],
                ),
            },
        ]
        stage = "verify" if step.step_type == "verify" else "reason"
        resp = llm.complete(
            messages, temperature=temperature, max_tokens=max_tokens, stage=stage
        )
        llm_meta = resp
        try:
            parsed = json.loads(resp["content"])
        except Exception:  # noqa: BLE001
            parsed = {
                "summary": str(resp["content"])[:500],
                "decision": "continue",
                "resolved": True,
                "claims": [],
            }

    summary = str(parsed.get("summary") or parsed.get("decision_reason") or "")
    ev = ledger.add(
        step_id=step.step_id,
        claim=summary or f"{step.step_type} summary",
        source_type="reason" if step.step_type == "reason" else "verify",
        raw_value=parsed,
        normalized_value=summary,
        provenance="concise_reasoning_summary",
        validation_status="unchecked",
    )
    return {
        "ok": bool(parsed.get("resolved", True)),
        "mcp_called": False,
        "error": None,
        "llm": llm_meta,
        "payload": parsed,
        "evidence_id": ev.evidence_id,
    }


def select_next_step(
    plan_state: PlanState, preferred_ids: Optional[Sequence[str]] = None
) -> Optional[PlanStep]:
    ready = plan_state.ready_steps()
    if not ready:
        return None
    if preferred_ids:
        for sid in preferred_ids:
            for step in ready:
                if step.step_id == sid:
                    return step
    # Prefer retrieve → compute → reason/verify → answer/clarify/refuse
    priority = {
        "retrieve": 0,
        "compute": 1,
        "reason": 2,
        "verify": 3,
        "answer": 4,
        "clarify": 5,
        "refuse": 6,
    }
    ready = sorted(
        ready, key=lambda s: (priority.get(s.step_type, 9), s.step_id)
    )
    return ready[0]
