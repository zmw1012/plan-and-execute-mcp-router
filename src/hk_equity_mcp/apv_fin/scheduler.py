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
            "max_llm_monitor_rounds": int(limits.get("max_llm_monitor_rounds", 6)),
            "max_same_operation_repeats": int(limits.get("max_same_operation_repeats", 1)),
            "max_final_repair_rounds": int(limits.get("max_final_repair_rounds", 1)),
            "max_idle_monitor_rounds": int(limits.get("max_idle_monitor_rounds", 1)),
            # Global per-task generation cap (planner/monitor/reason/verify/final + repairs).
            "max_total_llm_calls_per_task": int(
                limits.get("max_total_llm_calls_per_task", 14)
            ),
        }
        self.plan_revisions = 0
        self.monitor_rounds = 0
        self.idle_monitor_rounds = 0
        self.mcp_invocations = 0
        self.retrievals_per_step: Dict[str, int] = {}
        self.operation_invoke_counts: Dict[str, int] = {}
        self.last_failed_invoke_signatures: set = set()
        self.final_repair_rounds = 0
        self.llm_calls = 0
        self.calls_by_stage: Dict[str, int] = {}
        self.stopping_reason: Optional[str] = None

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
            self.stopping_reason = "llm_call_budget_exhausted"
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
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "limits": dict(self.limits),
            "plan_revisions": self.plan_revisions,
            "monitor_rounds": self.monitor_rounds,
            "idle_monitor_rounds": self.idle_monitor_rounds,
            "mcp_invocations": self.mcp_invocations,
            "retrievals_per_step": dict(self.retrievals_per_step),
            "operation_invoke_counts": dict(self.operation_invoke_counts),
            "final_repair_rounds": self.final_repair_rounds,
            "llm_calls": self.llm_calls,
            "calls_by_stage": dict(self.calls_by_stage),
            "stopping_reason": self.stopping_reason,
            **self.llm_budget_fields(),
        }

    def exhausted(self) -> Optional[str]:
        # LLM call cap is enforced at request time by BudgetedLLMClient
        # (non-LLM steps may still proceed when the generation budget is spent).
        if self.plan_revisions > self.limits["max_plan_revisions"]:
            return "max_plan_revisions"
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

    to_describe = select_candidates_to_describe(reranked)
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

    # Identical failed invocation guard.
    invoke_sig = json.dumps(
        {"operation_id": oid, "arguments": args}, sort_keys=True, ensure_ascii=False
    )
    prior = getattr(budget, "last_failed_invoke_signatures", None)
    if prior is None:
        budget.last_failed_invoke_signatures = set()
        prior = budget.last_failed_invoke_signatures
    if invoke_sig in prior:
        base_out.update(
            {
                "ok": False,
                "error": "identical_failed_invocation_not_repeated",
                "chosen_operation_id": oid,
                "grounded_arguments": args,
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
        prior.add(invoke_sig)
    ev = ledger.add(
        step_id=step.step_id,
        claim=f"retrieve/invoke {oid}",
        source_type="mcp_invoke",
        operation_id=oid,
        raw_value=invoke_out,
        normalized_value=(invoke_out or {}).get("result"),
        provenance="UnifiedRealMcpGateway via OperationResolverServer",
        validation_status="valid" if success else "invalid",
    )
    base_out.update(
        {
            "ok": success,
            "chosen_operation_id": oid,
            "invoke": invoke_out,
            "evidence_id": ev.evidence_id,
            "mcp_called": True,
            "invoked": True,
            "successful": success,
            "error": None if success else invoke_out.get("error"),
        }
    )
    return base_out


def _extract_compute_payload_from_step(
    step: PlanStep, scripted_payload: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Resolve compute expression/operands without any LLM call."""
    if scripted_payload is not None:
        return dict(scripted_payload)
    # Prefer structured JSON in notes; otherwise accept notes already shaped as payload.
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
            return payload
    return None


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
    """Deterministic compute: safe arithmetic only. Never calls LLM or MCP.

    If expression/operands are missing, mark unresolved for monitor/replan
    (no hidden compute LLM call).
    """
    del llm, question, temperature, max_tokens  # unused; keep signature stable
    payload = _extract_compute_payload_from_step(step, scripted_payload)
    if payload is None:
        return {
            "ok": False,
            "mcp_called": False,
            "llm_called": False,
            "error": "compute_missing_expression_operands",
            "unresolved": True,
            "repair_count": 0,
            "llm": None,
            "payload": None,
            "compute_trace": compute_trace_record(
                step_id=step.step_id,
                payload={},
                error="compute_missing_expression_operands",
            ),
        }

    try:
        evaluated = evaluate_compute_payload(payload)
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
