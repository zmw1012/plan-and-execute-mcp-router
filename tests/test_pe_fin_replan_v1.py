"""Offline tests for PE-Fin RePlan v1 (no network / no paid API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_TOP_K,
    retrieval_formula_hash,
)
from hk_equity_mcp.pe_fin_lite.schemas import PlanValidationError
from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
)
from hk_equity_mcp.pe_fin_replan.grounding import validate_replanner_response
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS, replan_protocol_identity
from hk_equity_mcp.pe_fin_replan.plan_ops import (
    select_one_ready_step,
    validate_updated_remaining_plan,
)
from hk_equity_mcp.pe_fin_replan.schemas import (
    LOOP_POLICY,
    parse_replanner_output,
    validate_replanner_payload,
)
from hk_equity_mcp.pe_fin_replan.state import PlanExecuteState, failed_signature

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
EEA_ANALYSIS = ROOT / "outputs" / "pe_fin_lite_eea_dev_failure_analysis_v1"
DIAG20 = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"

FORMAL_SHA = {
    "predictions.jsonl": "4c5c4df3ababca99f9a91203605154ad8fc4a05cdfbc2ba2acb7cc11bc2a7d04",
    "run_metadata.json": "2a3dff284afa86db8ec93433965f2b451e7aef05c65a78edd53fce3f84b8d602",
}
DIAG20_SHA = "7076a39ffc6b896ddc2bd2153c8c8d4f9a6dca268e805e3bf239867077cad1a9"
R5_HASH = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step(
    sid: str,
    stype: str = "retrieve",
    *,
    objective: str = "get close",
    depends_on: Optional[List[str]] = None,
    required_outputs: Optional[List[str]] = None,
    domain: str = "market",
) -> Dict[str, Any]:
    return {
        "id": sid,
        "type": stype,
        "domain": domain,
        "objective": objective,
        "required_outputs": required_outputs or (["close"] if stype == "retrieve" else []),
        "depends_on": depends_on or [],
        "entity": None,
        "period": None,
        "date": None,
        "compute_spec": None,
        "workflow": None,
    }


# ---------------------------------------------------------------------------
# Freeze / identity
# ---------------------------------------------------------------------------


def test_formal_v12_raw_hashes_unchanged():
    for name, expected in FORMAL_SHA.items():
        assert _sha(FORMAL / name) == expected


def test_eea_analysis_artifacts_exist_unchanged_presence():
    assert EEA_ANALYSIS.is_dir()
    assert (EEA_ANALYSIS / "dev_failure_matrix.csv").exists()
    assert (EEA_ANALYSIS / "c07_failure_taxonomy.json").exists()


def test_diagnostic20_manifest_unchanged():
    assert _sha(DIAG20) == DIAG20_SHA
    man = json.loads(DIAG20.read_text(encoding="utf-8"))
    assert len(man["ordered_task_ids"]) == 20
    assert man["explicitly_not_formal_accuracy"] is True


def test_r5_topk_describe_n_unchanged():
    assert RETRIEVAL_FORMULA_NAME == "R5_global_lexical@0.12"
    assert retrieval_formula_hash() == R5_HASH
    assert RETRIEVAL_TOP_K == 5
    assert DESCRIBE_N == 3


def test_replan_identity_and_budgets():
    assert PE_FIN_REPLAN_CONDITION == "pe_fin_replan"
    assert PE_FIN_REPLAN_METHOD_VERSION == "pe_fin_replan_v1"
    assert PE_FIN_REPLAN_PROTOCOL_VERSION == "planner_executor_replanner_loop_v1"
    assert PE_FIN_REPLAN_ARCHITECTURE == "plan_execute_replan"
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 8
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
    assert DEFAULT_BUDGETS["max_executed_steps"] == 6
    assert DEFAULT_BUDGETS["max_replan_rounds"] == 6
    ident = replan_protocol_identity()
    assert ident["retrieval_formula_hash"] == R5_HASH
    assert ident["flags"]["posthoc_exploratory"] is True
    assert ident["flags"]["formal_primary_result"] is False
    assert LOOP_POLICY["no_executor_to_executor_edge"] is True


# ---------------------------------------------------------------------------
# Plan ops / loop protections
# ---------------------------------------------------------------------------


def test_executor_selects_exactly_one_ready_step():
    remaining = [
        _step("s1"),
        _step("s2", depends_on=["s1"], objective="second"),
    ]
    one = select_one_ready_step(remaining, set())
    assert one is not None
    assert one["id"] == "s1"
    # After s1 done, only s2
    one2 = select_one_ready_step(remaining, {"s1"})
    assert one2["id"] == "s2"


def test_replanner_can_remove_completed_and_add_at_most_two():
    prev = [_step("s2", objective="keep")]
    proposed = [
        _step("s2", objective="keep"),
        _step("r1", objective="alt retrieve A"),
        _step("r2", objective="alt retrieve B"),
    ]
    steps, diff = validate_updated_remaining_plan(
        proposed=proposed,
        previous_remaining=prev,
        completed_ids={"s1"},
        completed_objectives={"get close"},
        max_new_steps=2,
    )
    assert len(steps) == 3
    assert diff["n_added"] == 2

    with pytest.raises(PlanValidationError, match="too_many_new_steps"):
        validate_updated_remaining_plan(
            proposed=proposed + [_step("r3", objective="third new")],
            previous_remaining=prev,
            completed_ids={"s1"},
            completed_objectives=set(),
            max_new_steps=2,
        )


def test_completed_step_cannot_reappear_without_scope():
    with pytest.raises(PlanValidationError, match="completed_step_readded"):
        validate_updated_remaining_plan(
            proposed=[_step("s1")],
            previous_remaining=[],
            completed_ids={"s1"},
            completed_objectives=set(),
            max_new_steps=2,
        )


def test_completed_objective_repeat_without_scope_fails():
    with pytest.raises(PlanValidationError, match="completed_objective_repeat"):
        validate_updated_remaining_plan(
            proposed=[_step("r9", objective="get close")],
            previous_remaining=[],
            completed_ids={"s1"},
            completed_objectives={"get close"},
            max_new_steps=2,
        )


def test_failed_signature_stable():
    a = failed_signature("op.x", {"b": 2, "a": 1})
    b = failed_signature("op.x", {"a": 1, "b": 2})
    c = failed_signature("op.x", {"a": 1, "b": 3})
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Replanner schema
# ---------------------------------------------------------------------------


def test_replanner_schema_plan_and_response():
    plan = {
        "schema_version": "pe_fin_replan_v1",
        "action": "plan",
        "decision_reason": "Need alternate retrieve",
        "remaining_steps": [_step("r1", objective="alternate retrieve")],
    }
    validate_replanner_payload(plan)
    resp = {
        "schema_version": "pe_fin_replan_v1",
        "action": "response",
        "decision_reason": "Sufficient evidence",
        "response": {
            "status": "answered",
            "final_answer": "ok",
            "reasoning_brief": "evidence grounded",
        },
    }
    validate_replanner_payload(resp)


def test_invalid_replanner_json_one_repair_then_fail():
    bad = "not json"
    obj, _, err = parse_replanner_output(bad)
    assert obj is None and err
    still_bad = "{}"
    obj2, _, err2 = parse_replanner_output(bad, repair_content=still_bad)
    assert obj2 is None and err2
    good = json.dumps(
        {
            "schema_version": "pe_fin_replan_v1",
            "action": "response",
            "decision_reason": "done",
            "response": {
                "status": "unavailable",
                "final_answer": "",
                "reasoning_brief": "capability unavailable after exhausted retrieve",
                "refusal_reason": "capability unavailable",
            },
        }
    )
    obj3, _, err3 = parse_replanner_output(bad, repair_content=good)
    assert obj3 is not None and err3 is None


# ---------------------------------------------------------------------------
# Grounding / answerability interaction
# ---------------------------------------------------------------------------


def test_answered_grounding_requires_evidence_for_numbers():
    bad = validate_replanner_response(
        response={
            "status": "answered",
            "final_answer": "close was 123.45",
            "reasoning_brief": "from question",
        },
        question="Was the close 123.45?",
        evidence_values={},
    )
    assert bad["ok"] is False

    ok = validate_replanner_response(
        response={
            "status": "answered",
            "final_answer": "close was 10.5",
            "reasoning_brief": "from evidence",
        },
        question="What was the close?",
        evidence_values={"close": 10.5},
    )
    assert ok["ok"] is True


def test_unavailable_needs_infeasibility_marker():
    bad = validate_replanner_response(
        response={
            "status": "unavailable",
            "final_answer": "",
            "reasoning_brief": "optional page missing",
        },
        question="q",
        evidence_values={"close": 1},
    )
    assert bad["ok"] is False

    ok = validate_replanner_response(
        response={
            "status": "unavailable",
            "final_answer": "",
            "reasoning_brief": "primary/fallback exhausted; capability unavailable",
            "refusal_reason": "capability unavailable",
        },
        question="q",
        evidence_values={},
    )
    assert ok["ok"] is True


def test_optional_context_missing_can_still_answer():
    ok = validate_replanner_response(
        response={
            "status": "answered",
            "final_answer": "close observed in evidence",
            "reasoning_brief": "optional source_url missing but core present",
        },
        question="What is close?",
        evidence_values={"close": 9.0},
    )
    assert ok["ok"] is True


# ---------------------------------------------------------------------------
# Architecture fidelity via instrumented agent loop (fake LLM)
# ---------------------------------------------------------------------------


class _SeqLLM:
    provider = "mock"
    model = "seq"

    def __init__(self, by_stage: Dict[str, List[str]]):
        self.by_stage = {k: list(v) for k, v in by_stage.items()}
        self.calls: List[str] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None, **kw):
        st = str(stage or "unknown")
        self.calls.append(st)
        key = st
        if key not in self.by_stage or not self.by_stage[key]:
            # fallback strip _repair
            base = key.replace("_repair", "")
            key = base if self.by_stage.get(base) else key
        queue = self.by_stage.get(key) or self.by_stage.get("default") or []
        if not queue:
            raise RuntimeError(f"no scripted content for stage={st}")
        content = queue.pop(0)
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "input_tokens": 1,
                "output_tokens": 1,
            },
            "latency_ms": 0.1,
            "model": self.model,
            "provider": self.provider,
        }


def _plan_json(steps: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {"schema_version": "pe_fin_lite_plan_v1", "steps": steps},
        ensure_ascii=False,
    )


def _replan_response(*, status="answered", answer="ok", brief="done") -> str:
    return json.dumps(
        {
            "schema_version": "pe_fin_replan_v1",
            "action": "response",
            "decision_reason": brief,
            "response": {
                "status": status,
                "final_answer": answer,
                "reasoning_brief": brief,
                "clarification_question": None,
                "refusal_reason": None if status == "answered" else brief,
                "limitations": None,
            },
        },
        ensure_ascii=False,
    )


def _replan_plan(steps: List[Dict[str, Any]], reason: str = "revise") -> str:
    return json.dumps(
        {
            "schema_version": "pe_fin_replan_v1",
            "action": "plan",
            "decision_reason": reason,
            "remaining_steps": steps,
        },
        ensure_ascii=False,
    )


def test_graph_edges_planner_once_executor_one_then_replanner(monkeypatch):
    """Architecture: Planner×1, no Executor→Executor, final from Replanner."""
    from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
    from hk_equity_mcp.pe_fin_replan.agent import PeFinReplanAgent

    plan_steps = [_step("s1", objective="retrieve close price")]
    llm = _SeqLLM(
        {
            "planner": [_plan_json(plan_steps)],
            "replanner": [
                _replan_response(
                    status="answered",
                    answer="close observed in evidence",
                    brief="sufficient verified evidence",
                )
            ],
        }
    )

    class _FakeResolver:
        def list_underlying_operation_ids(self):
            return ["market.get_daily"]

        def find_operations(self, *a, **k):
            return [{"operation_id": "market.get_daily", "score": 1.0}]

        def describe_operation(self, *a, **k):
            return {"operation_id": "market.get_daily", "schema": {}}

    # Patch execute_retrieve_step to avoid full binder/MCP stack
    def _fake_retrieve(**kwargs):
        store: EvidenceStore = kwargs["store"]
        store.values["close"] = 10.0
        store.values["s1.close"] = 10.0
        return {
            "step_result": {
                "status": "completed",
                "execution_ok": True,
                "failure_reason": None,
                "missing_outputs": [],
                "values": {"close": 10.0},
            },
            "usage": {},
            "funnel": {
                "search_query": "q",
                "invoked_operations": ["market.get_daily"],
                "invoke_gateway_success": {"market.get_daily": True},
                "described_candidates": [],
                "primary": {"operation_id": "market.get_daily", "arguments": {}},
            },
        }

    monkeypatch.setattr(
        "hk_equity_mcp.pe_fin_replan.agent.execute_retrieve_step", _fake_retrieve
    )

    agent = PeFinReplanAgent(
        llm=llm,
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        capability_profile={
            "profile_id": "test",
            "profile_version": "v1",
            "source_types": {},
            "composite_workflows": [],
            "unsupported_capabilities": [],
            "execution_limits_hint": {},
        },
        config={
            "limits": dict(DEFAULT_BUDGETS),
            "use_mock_binder": True,
            "llm": {"temperature": 0.0, "max_tokens": 512},
        },
        live_mode=False,
    )
    # Bypass assert_prompts_clean already done; profile_for_planner needs real-ish profile
    from hk_equity_mcp.pe_fin_lite.capability_profile import load_capability_profile

    agent.capability_profile = load_capability_profile(ROOT)

    out = agent.run("What is the close?")
    stages = [t.get("stage") for t in out["trace"]]
    assert stages.count("planner") == 1
    assert "executor" in stages
    assert "replanner" in stages
    # No back-to-back executor without replanner between
    for i in range(len(stages) - 1):
        if stages[i] == "executor":
            assert stages[i + 1] == "replanner"
    assert out["final_response_source"] == "replanner"
    assert out["planner_calls"] >= 1
    assert out["replan_calls"] >= 1
    assert out["llm_calls"] <= 8
    assert "Executor→Replanner" in out["graph_edges"]
    assert out["monitor_decisions"] == []


def test_retrieve_fail_replan_alternate_then_answer(monkeypatch):
    from hk_equity_mcp.pe_fin_lite.capability_profile import load_capability_profile
    from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
    from hk_equity_mcp.pe_fin_replan.agent import PeFinReplanAgent

    calls = {"n": 0}

    def _fake_retrieve(**kwargs):
        calls["n"] += 1
        store: EvidenceStore = kwargs["store"]
        step = kwargs["step"]
        if step["id"] == "s1":
            return {
                "step_result": {
                    "status": "failed",
                    "execution_ok": False,
                    "failure_reason": "no_match",
                    "missing_outputs": ["close"],
                    "values": {},
                    "execution_failure_class": "C07_no_match",
                },
                "usage": {},
                "funnel": {
                    "search_query": "bad",
                    "invoked_operations": ["op.bad"],
                    "invoke_gateway_success": {"op.bad": False},
                    "described_candidates": [],
                    "primary": {"operation_id": "op.bad", "arguments": {"x": 1}},
                },
            }
        store.values["close"] = 42.0
        return {
            "step_result": {
                "status": "completed",
                "execution_ok": True,
                "failure_reason": None,
                "missing_outputs": [],
                "values": {"close": 42.0},
            },
            "usage": {},
            "funnel": {
                "search_query": "alt",
                "invoked_operations": ["op.good"],
                "invoke_gateway_success": {"op.good": True},
                "described_candidates": [],
                "primary": {"operation_id": "op.good", "arguments": {"x": 2}},
            },
        }

    monkeypatch.setattr(
        "hk_equity_mcp.pe_fin_replan.agent.execute_retrieve_step", _fake_retrieve
    )

    llm = _SeqLLM(
        {
            "planner": [_plan_json([_step("s1", objective="primary retrieve close")])],
            "replanner": [
                _replan_plan(
                    [_step("r1", objective="alternate retrieve close price")],
                    reason="primary retrieve failed; try alternate objective",
                ),
                _replan_response(
                    status="answered",
                    answer="close was 42.0",
                    brief="recovered via revised retrieve",
                ),
            ],
        }
    )

    class _FakeResolver:
        def list_underlying_operation_ids(self):
            return ["op.bad", "op.good"]

    agent = PeFinReplanAgent(
        llm=llm,
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        capability_profile=load_capability_profile(ROOT),
        config={
            "limits": dict(DEFAULT_BUDGETS),
            "use_mock_binder": True,
            "llm": {"temperature": 0.0, "max_tokens": 512},
        },
        live_mode=False,
    )
    out = agent.run("What is the close?")
    assert calls["n"] == 2
    assert out["final_response_source"] == "replanner"
    assert out["steps_added"] >= 1
    assert out["recovery_attempts"] >= 1
    assert "42" in str(out.get("final_answer") or "")
    assert out["llm_calls"] <= 8


def test_loop_budget_caps_and_no_progress_fields():
    state = PlanExecuteState(question="q", max_executed_steps=6, max_replan_rounds=6)
    assert state.max_llm_calls == 8
    assert state.max_no_progress_rounds == 2
    assert state.max_mcp_calls == 8


def test_ungrounded_response_is_controlled_failure_not_normal_unavailable(monkeypatch):
    from hk_equity_mcp.pe_fin_lite.capability_profile import load_capability_profile
    from hk_equity_mcp.pe_fin_replan.agent import PeFinReplanAgent

    def _fake_retrieve(**kwargs):
        return {
            "step_result": {
                "status": "completed",
                "execution_ok": True,
                "failure_reason": None,
                "missing_outputs": [],
                "values": {},
            },
            "usage": {},
            "funnel": {
                "search_query": "q",
                "invoked_operations": [],
                "invoke_gateway_success": {},
                "described_candidates": [],
            },
        }

    monkeypatch.setattr(
        "hk_equity_mcp.pe_fin_replan.agent.execute_retrieve_step", _fake_retrieve
    )
    llm = _SeqLLM(
        {
            "planner": [_plan_json([_step("s1")])],
            "replanner": [
                _replan_response(
                    status="answered",
                    answer="close was 999",
                    brief="invented",
                )
            ],
        }
    )

    class _FakeResolver:
        def list_underlying_operation_ids(self):
            return []

    agent = PeFinReplanAgent(
        llm=llm,
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        capability_profile=load_capability_profile(ROOT),
        config={
            "limits": dict(DEFAULT_BUDGETS),
            "use_mock_binder": True,
            "llm": {"temperature": 0.0, "max_tokens": 512},
        },
        live_mode=False,
    )
    out = agent.run("Was close 999?")
    assert out["final_response_source"] == "controlled_failure"
    assert out["stopping_reason"] == "ungrounded_replanner_response"
    assert out["status"] == "failed"
