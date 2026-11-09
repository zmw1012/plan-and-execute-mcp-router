"""Tool Binding stage: select operation + ground arguments before MCP invoke."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set  # noqa: F401 — Optional used

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.argument_grounding import (
    collect_grounding_candidates,
    propose_arguments_for_schema,
)
from hk_equity_mcp.apv_fin.planner import _merge_usage
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_SYSTEM_PROMPT,
    APV_FIN_TOOL_BINDING_USER_TEMPLATE,
)
from hk_equity_mcp.apv_fin.schemas import SchemaValidationError, validate_against_schema
from hk_equity_mcp.apv_fin.schemas import APV_FIN_TOOL_BINDING_V1
from hk_equity_mcp.apv_fin.structured_parse import parse_structured, repair_user_message

DESCRIBE_TOP_N = 3


class ToolBindingValidationError(ValueError):
    pass


def _capability_score(row: Dict[str, Any], required_provides: Sequence[str]) -> int:
    if not required_provides:
        return 0
    reasons = row.get("compatibility_reasons") or {}
    supports = set(reasons.get("supports") or [])
    score = 0
    if "provides_overlap" in supports:
        score += 5
    if "domain:market" in supports and any(
        p in {"close", "ohlcv", "price"} for p in required_provides
    ):
        score += 3
    if "domain:financial" in supports and any(
        p in {"eps", "revenue", "net_income", "financial_indicators"}
        for p in required_provides
    ):
        score += 3
    # Prefer market hist/daily ids when close is required.
    oid = str(row.get("operation_id") or "").lower()
    if any(p in {"close", "ohlcv", "price"} for p in required_provides):
        if "hist" in oid or "daily" in oid:
            score += 4
        if "financial" in oid or "indicator" in oid:
            score -= 2
    return score


def select_candidates_to_describe(
    reranked: Sequence[Dict[str, Any]],
    *,
    top_n: int = DESCRIBE_TOP_N,
    required_provides: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Describe up to top_n candidates from final compatibility order.

    Prefer Tier A then B; C only when no A/B. Within a tier, prefer capability-
    compatible ops (provides/domain) so market-close does not describe only
    financial Tier-C heads when hist/daily sit lower in the same tier.
    """
    if not reranked:
        return []
    required = list(required_provides or [])
    a = [r for r in reranked if r.get("compatibility_tier") == "A"]
    b = [r for r in reranked if r.get("compatibility_tier") == "B"]
    c = [r for r in reranked if r.get("compatibility_tier") == "C"]
    pool = a + b
    if not pool:
        pool = list(c)

    def sort_key(row: Dict[str, Any]) -> tuple:
        return (
            -_capability_score(row, required),
            int(row.get("final_rank") or 10**6),
            -float(row.get("base_score") or 0.0),
        )

    pool = sorted(pool, key=sort_key)
    # Guarantee: if required close/price and a provides-overlap market op exists
    # in the same pool, it must appear in the described set when possible.
    selected = list(pool[: max(1, int(top_n))])
    if required and any(p in {"close", "ohlcv", "price"} for p in required):
        marketish = [
            r
            for r in pool
            if _capability_score(r, required) > 0
            and (
                "hist" in str(r.get("operation_id") or "").lower()
                or "daily" in str(r.get("operation_id") or "").lower()
                or "provides_overlap"
                in ((r.get("compatibility_reasons") or {}).get("supports") or [])
            )
        ]
        if marketish and not any(
            r["operation_id"] == marketish[0]["operation_id"] for r in selected
        ):
            selected = selected[:-1] + [marketish[0]] if selected else [marketish[0]]
            # re-unique preserve order
            seen = set()
            uniq = []
            for r in selected:
                oid = r["operation_id"]
                if oid in seen:
                    continue
                seen.add(oid)
                uniq.append(r)
            selected = uniq[: max(1, int(top_n))]
    return selected


def annotate_selection_flags(
    reranked: Sequence[Dict[str, Any]],
    *,
    described_ids: Sequence[str],
    selected_id: Optional[str],
) -> List[Dict[str, Any]]:
    described_set = set(described_ids)
    out: List[Dict[str, Any]] = []
    for row in reranked:
        item = dict(row)
        oid = item.get("operation_id")
        item["described"] = oid in described_set
        item["selected"] = bool(selected_id) and oid == selected_id
        out.append(item)
    return out


def validate_binding_payload(
    payload: Dict[str, Any],
    *,
    allowed_operation_ids: Sequence[str],
    described_contracts: Dict[str, Dict[str, Any]],
    blocked_operation_ids: Optional[Sequence[str]] = None,
    tier_by_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fail-closed semantic validation after JSON schema parse."""
    validate_against_schema(payload, APV_FIN_TOOL_BINDING_V1)
    decision = payload.get("decision")
    selected = payload.get("selected_operation_id")
    arguments = payload.get("arguments") or {}
    sources = payload.get("argument_sources") or {}
    allowed: Set[str] = set(allowed_operation_ids)
    blocked = set(blocked_operation_ids or [])
    tier_by_id = tier_by_id or {}

    if decision == "select":
        if not selected or not isinstance(selected, str):
            raise ToolBindingValidationError(
                "select decision requires selected_operation_id from described candidates"
            )
        if selected in blocked:
            raise ToolBindingValidationError(
                f"blocked/unavailable operation cannot be selected: {selected}"
            )
        if selected not in allowed:
            raise ToolBindingValidationError(
                f"selected_operation_id {selected!r} not in current unfiltered candidates"
            )
        # Tier C only when no A/B among allowed described set.
        selected_tier = tier_by_id.get(selected)
        has_ab = any(tier_by_id.get(oid) in {"A", "B"} for oid in allowed)
        if selected_tier == "C" and has_ab:
            raise ToolBindingValidationError(
                "Tier C operation cannot be selected while Tier A/B candidates exist"
            )
        contract = described_contracts.get(selected) or {}
        json_schema = contract.get("json_schema") or {}
        props = (json_schema.get("properties") or {}) if json_schema else {}
        required = list(json_schema.get("required") or [])
        # Arguments may only contain schema-allowed fields.
        for key in arguments:
            if props and key not in props:
                raise ToolBindingValidationError(
                    f"argument {key!r} not allowed by operation schema"
                )
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ToolBindingValidationError(
                f"missing required arguments: {missing}"
            )
        if required and arguments == {}:
            raise ToolBindingValidationError(
                "refusing empty arguments when schema has required fields"
            )
        for name in arguments:
            if name not in sources:
                raise ToolBindingValidationError(
                    f"argument_sources missing coverage for {name!r}"
                )
        # Light type check via schema enum/type when present.
        for name, value in arguments.items():
            schema = props.get(name) or {}
            if "enum" in schema and value not in schema["enum"]:
                raise ToolBindingValidationError(
                    f"argument {name!r} value {value!r} not in enum {schema['enum']}"
                )
            types = schema.get("type")
            if types is None:
                continue
            allowed_types = types if isinstance(types, list) else [types]
            ok = False
            if value is None and "null" in allowed_types:
                ok = True
            elif isinstance(value, bool) and "boolean" in allowed_types:
                ok = True
            elif isinstance(value, int) and not isinstance(value, bool) and (
                "integer" in allowed_types or "number" in allowed_types
            ):
                ok = True
            elif isinstance(value, float) and "number" in allowed_types:
                ok = True
            elif isinstance(value, str) and "string" in allowed_types:
                ok = True
            elif isinstance(value, list) and "array" in allowed_types:
                ok = True
            elif isinstance(value, dict) and "object" in allowed_types:
                ok = True
            if not ok:
                raise ToolBindingValidationError(
                    f"argument {name!r} type mismatch for schema type {allowed_types}"
                )
    else:
        if selected is not None:
            raise ToolBindingValidationError(
                f"{decision} decision requires selected_operation_id=null"
            )
        if arguments:
            raise ToolBindingValidationError(
                f"{decision} decision must not include invoke arguments"
            )

    return {
        "ok": True,
        "payload": payload,
        "decision": decision,
        "selected_operation_id": selected,
        "arguments": dict(arguments) if decision == "select" else {},
        "argument_sources": dict(sources) if decision == "select" else {},
    }


def build_deterministic_binding_proposal(
    *,
    described_rows: Sequence[Dict[str, Any]],
    described_contracts: Dict[str, Dict[str, Any]],
    grounding: Dict[str, Any],
) -> Dict[str, Any]:
    """Synthetic binding for offline mock — still must pass validate_binding_payload."""
    if not described_rows:
        return {
            "schema_version": "apv_fin_tool_binding_v1",
            "decision": "unsupported",
            "selected_operation_id": None,
            "arguments": {},
            "argument_sources": {},
            "expected_output": "none",
            "decision_reason": "No compatible candidates available to bind.",
            "reformulated_query": None,
        }
    # Prefer first Tier A/B (already ordered).
    for row in described_rows:
        oid = row["operation_id"]
        contract = described_contracts.get(oid) or {}
        json_schema = contract.get("json_schema") or {}
        args, sources, missing = propose_arguments_for_schema(
            json_schema=json_schema, grounding=grounding
        )
        if missing:
            continue
        return {
            "schema_version": "apv_fin_tool_binding_v1",
            "decision": "select",
            "selected_operation_id": oid,
            "arguments": args,
            "argument_sources": sources,
            "expected_output": "rows supporting the retrieve objective",
            "decision_reason": "Selected top compatible candidate with grounded required args.",
            "reformulated_query": None,
        }
    # Cannot ground required args → expand_search once if possible.
    return {
        "schema_version": "apv_fin_tool_binding_v1",
        "decision": "expand_search",
        "selected_operation_id": None,
        "arguments": {},
        "argument_sources": {},
        "expected_output": "better candidate match",
        "decision_reason": "Required arguments could not be reliably grounded for described candidates.",
        "reformulated_query": None,
    }


def run_tool_binder(
    llm: LLMClient,
    *,
    question: str,
    task_spec: Dict[str, Any],
    step: Dict[str, Any],
    reranked_summaries: Sequence[Dict[str, Any]],
    described_contracts: Dict[str, Dict[str, Any]],
    grounding: Dict[str, Any],
    budget_snapshot: Dict[str, Any],
    allowed_operation_ids: Sequence[str],
    blocked_operation_ids: Optional[Sequence[str]] = None,
    tier_by_id: Optional[Dict[str, str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    offline_proposal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One LLM tool-binding call (+ optional one repair). Counts via BudgetedLLMClient stage."""
    described_brief = []
    for oid, contract in described_contracts.items():
        described_brief.append(
            {
                "operation_id": oid,
                "summary": contract.get("summary") or contract.get("description"),
                "json_schema": contract.get("json_schema"),
                "required_parameters": contract.get("required_parameters"),
                "compatibility_tier": (tier_by_id or {}).get(oid),
            }
        )
    user = APV_FIN_TOOL_BINDING_USER_TEMPLATE.format(
        question=question,
        task_spec_json=json.dumps(task_spec, ensure_ascii=False),
        step_json=json.dumps(step, ensure_ascii=False),
        grounding_json=json.dumps(grounding, ensure_ascii=False),
        candidates_json=json.dumps(list(reranked_summaries)[:10], ensure_ascii=False),
        described_json=json.dumps(described_brief, ensure_ascii=False),
        budget_json=json.dumps(budget_snapshot, ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    # Offline mock path: inject proposal as the LLM content via a single call that
    # still goes through BudgetedLLMClient when wrapped (stage=tool_binding).
    if offline_proposal is not None:
        # Still perform the LLM call so stage counters increment; scripted clients
        # may ignore messages and return queue content — prefer forcing content
        # by calling complete then overriding parse input with proposal when the
        # scripted content is the sentinel or when factory-provided.
        resp = llm.complete(
            messages, temperature=temperature, max_tokens=max_tokens, stage="tool_binding"
        )
        # Prefer explicit offline proposal (validated) over whatever script returned.
        content = json.dumps(offline_proposal, ensure_ascii=False)
        parsed = parse_structured(content, "apv_fin_tool_binding_v1")
        raw_contents = [resp.get("content"), content]
        repair_count = 0
    else:
        resp = llm.complete(
            messages, temperature=temperature, max_tokens=max_tokens, stage="tool_binding"
        )
        parsed = parse_structured(resp.get("content"), "apv_fin_tool_binding_v1")
        raw_contents = [parsed.get("raw_content")]
        repair_count = 0
        if not parsed["ok"]:
            messages.append(
                {"role": "assistant", "content": str(resp.get("content") or "")}
            )
            messages.append(
                {
                    "role": "user",
                    "content": repair_user_message(
                        "apv_fin_tool_binding_v1", parsed.get("error") or "parse_failed"
                    ),
                }
            )
            repair_resp = llm.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stage="tool_binding_repair",
            )
            parsed = parse_structured(
                resp.get("content"),
                "apv_fin_tool_binding_v1",
                repair_content=repair_resp.get("content"),
            )
            raw_contents.append(parsed.get("repair_raw_content"))
            repair_count = int(parsed.get("repair_count") or 1)
            resp = {
                "content": repair_resp.get("content"),
                "raw_content": repair_resp.get("raw_content", repair_resp.get("content")),
                "usage": _merge_usage(resp.get("usage"), repair_resp.get("usage")),
                "latency_ms": float(resp.get("latency_ms") or 0)
                + float(repair_resp.get("latency_ms") or 0),
                "provider": repair_resp.get("provider"),
                "model": repair_resp.get("model"),
                "repair": True,
            }

    if not parsed["ok"]:
        return {
            "ok": False,
            "fail_closed": True,
            "error": parsed.get("error") or "tool_binding_parse_fail_closed",
            "repair_count": repair_count,
            "llm": resp,
            "payload": None,
            "raw_contents": raw_contents,
            "argument_validation": {"ok": False, "error": parsed.get("error")},
        }

    payload = parsed["payload"]
    try:
        validated = validate_binding_payload(
            payload,
            allowed_operation_ids=allowed_operation_ids,
            described_contracts=described_contracts,
            blocked_operation_ids=blocked_operation_ids,
            tier_by_id=tier_by_id,
        )
    except (ToolBindingValidationError, SchemaValidationError) as exc:
        # One semantic repair attempt when not using offline proposal.
        if offline_proposal is None and repair_count == 0:
            messages.append(
                {"role": "assistant", "content": str(resp.get("content") or "")}
            )
            messages.append(
                {
                    "role": "user",
                    "content": repair_user_message(
                        "apv_fin_tool_binding_v1", f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            repair_resp = llm.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stage="tool_binding_repair",
            )
            parsed2 = parse_structured(
                repair_resp.get("content"), "apv_fin_tool_binding_v1"
            )
            repair_count = 1
            raw_contents.append(parsed2.get("raw_content"))
            resp = {
                "content": repair_resp.get("content"),
                "usage": _merge_usage(resp.get("usage"), repair_resp.get("usage")),
                "latency_ms": float(resp.get("latency_ms") or 0)
                + float(repair_resp.get("latency_ms") or 0),
                "repair": True,
            }
            if parsed2["ok"]:
                try:
                    validated = validate_binding_payload(
                        parsed2["payload"],
                        allowed_operation_ids=allowed_operation_ids,
                        described_contracts=described_contracts,
                        blocked_operation_ids=blocked_operation_ids,
                        tier_by_id=tier_by_id,
                    )
                    return {
                        "ok": True,
                        "fail_closed": False,
                        "error": None,
                        "repair_count": repair_count,
                        "llm": resp,
                        "payload": validated["payload"],
                        "decision": validated["decision"],
                        "selected_operation_id": validated["selected_operation_id"],
                        "arguments": validated["arguments"],
                        "argument_sources": validated["argument_sources"],
                        "raw_contents": raw_contents,
                        "argument_validation": {"ok": True, "error": None},
                    }
                except (ToolBindingValidationError, SchemaValidationError) as exc2:
                    exc = exc2
        return {
            "ok": False,
            "fail_closed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "repair_count": repair_count,
            "llm": resp,
            "payload": payload,
            "raw_contents": raw_contents,
            "argument_validation": {"ok": False, "error": str(exc)},
        }

    return {
        "ok": True,
        "fail_closed": False,
        "error": None,
        "repair_count": repair_count,
        "llm": resp,
        "payload": validated["payload"],
        "decision": validated["decision"],
        "selected_operation_id": validated["selected_operation_id"],
        "arguments": validated["arguments"],
        "argument_sources": validated["argument_sources"],
        "raw_contents": raw_contents,
        "argument_validation": {"ok": True, "error": None},
    }


def grounding_from_context(
    *,
    question: str,
    task_spec: Dict[str, Any],
    step: Any,
    ledger: Any = None,
) -> Dict[str, Any]:
    step_dict = (
        step
        if isinstance(step, dict)
        else {
            "step_id": getattr(step, "step_id", None),
            "objective": getattr(step, "objective", None),
            "retrieval_query_hint": getattr(step, "retrieval_query_hint", None),
            "notes": getattr(step, "notes", None),
            "success_criteria": getattr(step, "success_criteria", None),
        }
    )
    ledger_summary = None
    if ledger is not None:
        if hasattr(ledger, "to_list"):
            ledger_summary = {"items": ledger.to_list()}
        elif hasattr(ledger, "summary"):
            ledger_summary = ledger.summary()
        elif isinstance(ledger, dict):
            ledger_summary = ledger
    return collect_grounding_candidates(
        question=question,
        task_spec=task_spec,
        step=step_dict,
        ledger_summary=ledger_summary,
    )
