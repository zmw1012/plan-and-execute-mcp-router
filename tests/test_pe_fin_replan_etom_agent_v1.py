"""Behavioural (agent-loop) offline tests for PE-Fin RePlan-ETOM v1.

Covers the loop-level checks of the 20-check list with a REAL ``PeFinReplanEtomAgent``
over a tiny fake registry (``tests/fixtures/pe_fin_replan_etom/mini_registry.json``):

 11 one-step Executor -> Replanner edge        12 Round-7 response-only
 13 no eighth round                            14 LLM <= 16 / MCP <= 8
 10 failed-signature blocking (live loop)      16 grounding blocks ungrounded answers
 17 task isolation                             19 aborted finalization
 20 mock / dry-run cannot forge a live pass    +  no gold / operation ids in Planner input

No network, no real LLM / embedding / MCP: ``ScriptedEtomPeFinLLM`` / queue LLMs and
``HashEmbeddingClient`` only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from hk_equity_mcp.etom.adapter import EtomRoutingResolver
from hk_equity_mcp.etom.embeddings import CatalogIndex, EmbeddingUsage, HashEmbeddingClient
from hk_equity_mcp.pe_fin_replan_etom.agent import PeFinReplanEtomAgent
from hk_equity_mcp.pe_fin_replan_etom.capability_profile import load_capability_profile
from hk_equity_mcp.pe_fin_replan_etom.mock_llm import (
    ScriptedEtomPeFinLLM,
    _payload_after_first_line,
)
from hk_equity_mcp.pe_fin_replan_etom.operation_documents import (
    documents_for_registry,
    registry_from_payload,
)
from hk_equity_mcp.pe_fin_replan_etom.prompts import (
    REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "pe_fin_replan_etom" / "mini_registry.json"
SCHEMA_V = "pe_fin_replan_etom_v1"


# --------------------------------------------------------------------------- helpers
class StageQueueLLM:
    """Stage-scripted LLM. Items may be strings or ``callable(messages) -> str``."""

    provider = "mock"
    model = "mock-queue"

    def __init__(self, scripts: Optional[Dict[str, List[Any]]] = None, fallback: Any = None):
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.fallback = fallback or ScriptedEtomPeFinLLM()
        self.log: List[Dict[str, Any]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None):
        base = ScriptedEtomPeFinLLM._base_stage(stage)
        self.log.append({"stage": stage, "base": base, "messages": list(messages)})
        queue = self.scripts.get(base)
        if queue:
            item = queue.pop(0)
            content = item(messages) if callable(item) else item
            return {
                "content": content,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        return self.fallback.complete(
            messages, temperature=temperature, max_tokens=max_tokens, stage=stage
        )

    def stages(self) -> List[str]:
        return [row["base"] for row in self.log]


def _step(sid: str, objective: Optional[str] = None, deps: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "id": sid,
        "type": "retrieve",
        "domain": "search",
        "objective": objective or f"look up item {sid} on the web",
        "required_outputs": ["result summary"],
        "depends_on": deps or [],
        "scope_constraints": None,
        "compute_spec": None,
    }


def _plan(*steps: Dict[str, Any]) -> str:
    return json.dumps({"steps": list(steps)})


def _replan_plan(remaining: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {"schema_version": SCHEMA_V, "action": "plan", "decision_reason": None, "remaining_steps": remaining}
    )


def _replan_response(answer: str = "The request was routed to the selected tool.", completion: str = "answered") -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_V,
            "action": "response",
            "decision_reason": None,
            "response": {"completion": completion, "final_answer": answer, "reasoning_brief": "ok"},
        }
    )


def keep_plan_then_respond(extra_step_on_round: Optional[Dict[int, Dict[str, Any]]] = None) -> Callable:
    """Replanner policy: keep remaining steps (optionally add one), respond at the final round/when empty."""
    extra = dict(extra_step_on_round or {})

    def _fn(messages) -> str:
        payload = _payload_after_first_line(messages[-1]["content"])
        state = payload["state"]
        remaining = list(state["remaining_plan"])
        rnd = int(payload["execution_round"])
        if payload.get("forced_final_round") or not remaining:
            return _replan_response()
        if rnd in extra:
            remaining = remaining + [extra[rnd]]
        return _replan_plan(remaining)

    return _fn


class Env:
    def __init__(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.registry, self.official = registry_from_payload(payload)
        self.embedder = HashEmbeddingClient(model_name="mock-hash-embedding-v1")
        docs = documents_for_registry(self.registry)
        vectors_list, _ = self.embedder.embed([doc for _o, doc, _h in docs])
        self.vectors = {oid: vec for (oid, _d, _h), vec in zip(docs, vectors_list)}
        self.index = CatalogIndex(
            model_name=self.embedder.model_name,
            registry_sha256="fixture",
            vectors=self.vectors,
            build_usage=EmbeddingUsage(),
            cache_reused=False,
        )
        self.resolver = EtomRoutingResolver(self.registry, self.official, self.index, self.embedder)
        self.profile = load_capability_profile(ROOT)

    def agent(self, llm: Any, limits: Optional[Dict[str, Any]] = None) -> PeFinReplanEtomAgent:
        return PeFinReplanEtomAgent(
            llm,
            self.resolver,
            capability_profile=self.profile,
            registry=self.registry,
            official=self.official,
            vectors=self.vectors,
            embedder=self.embedder,
            limits=limits,
        )


@pytest.fixture()
def env() -> Env:
    return Env()


def _stage_sequence(out: Dict[str, Any]) -> List[str]:
    return [t["stage"] for t in out["trace"] if t.get("stage") in ("planner", "executor", "replanner")]


QUESTION = "Search the web for the latest release notes."


# --------------------------------------------------------------------------- tests
def test_11_one_step_executor_to_replanner_edge(env: Env):
    llm = StageQueueLLM(scripts={"planner": [_plan(_step("s1"), _step("s2"), _step("s3"))]})
    out = env.agent(llm).run(QUESTION)
    assert out["status"] in ("full", "partial")
    # Planner exactly once, then strictly alternating Executor -> Replanner.
    assert _stage_sequence(out) == ["planner", "executor", "replanner"] * 1 + [
        "executor",
        "replanner",
        "executor",
        "replanner",
    ]
    assert llm.stages().count("planner") == 1
    assert out["planner_calls"] == 1
    assert out["executed_step_count"] == 3 == out["execution_round_count"]
    assert out["executor_to_executor_edge"] is False
    assert len(out["predicted_tools"]) == 3 and out["mcp_invocations"] == 3
    assert out["real_tool_executions"] == 0
    assert "Executor->Executor" not in "|".join(out["graph_edges"])
    # final response comes from the Replanner only
    assert out["final_response_source"] == "replanner"


def test_12_13_round7_response_only_and_no_eighth_round(env: Env):
    s7 = _step("s7")
    steps = [_step(f"s{i}") for i in range(1, 7)]
    llm = StageQueueLLM(
        scripts={
            "planner": [_plan(*steps)],
            "replanner": [keep_plan_then_respond({1: s7})] * 7,
        }
    )
    out = env.agent(llm).run(QUESTION)
    assert out["execution_round_count"] == 7 and out["executed_step_count"] == 7
    assert out["no_eighth_round"] is True and out["round7_response_only"] is True
    final_decision = out["replan_decisions"][-1]
    assert final_decision["round"] == 7 and final_decision["response_only"] is True
    assert final_decision["action"] == "response"
    assert out["final_response_source"] == "replanner"
    # rounds 1..6 are NOT forced-final; the final replanner call used the final-round prompt
    replanner_calls = [row for row in llm.log if row["base"] == "replanner"]
    assert len(replanner_calls) == 7
    assert replanner_calls[-1]["messages"][0]["content"] == REPLANNER_FINAL_ROUND_SYSTEM_PROMPT
    assert all(c["messages"][0]["content"] == REPLANNER_SYSTEM_PROMPT for c in replanner_calls[:-1])
    # budgets hold even on the longest path
    assert out["llm_calls"] <= 16 and out["mcp_invocations"] <= 8


def test_13_final_round_plan_is_a_controlled_failure_not_an_extra_round(env: Env):
    always_plan = lambda m: _replan_plan([_step("s3")])  # noqa: E731  (plan on a response-only round)
    llm = StageQueueLLM(
        scripts={
            "planner": [_plan(*[_step(f"s{i}") for i in range(1, 6)])],
            "replanner": [keep_plan_then_respond(), always_plan, always_plan],
        }
    )
    out = env.agent(llm, limits={"max_execution_rounds": 2, "max_executed_steps": 2, "max_replan_rounds": 2}).run(
        QUESTION
    )
    assert out["execution_round_count"] <= 2
    assert out["final_response_source"] == "controlled_failure"
    assert out["predicted_status"] == "unavailable"
    assert out["error_type"] == "replanner_schema_or_semantics"
    assert "final_round" in (out["error"] or "") or "invalid_replanner_output" in (
        out["stopping_reason"] or ""
    )
    assert out["executed_step_count"] <= 2


def test_14_llm_cap_16_and_mcp_cap_8(env: Env):
    # LLM cap: with a cap of 2 the third call (replanner) / second call (binder) is refused.
    out = env.agent(ScriptedEtomPeFinLLM(), limits={"max_total_llm_calls_per_task": 2}).run(QUESTION)
    assert out["llm_calls"] <= 2
    assert out["error_type"] == "resource_limit_failure" or out["stopping_reason"] in (
        "max_total_llm_calls",
        "resource_limit_failure",
    )
    assert out["final_response_source"] == "controlled_failure"
    # Garbage everywhere can never exceed the frozen ceiling of 16.
    junk = StageQueueLLM(
        scripts={"planner": ["not json", "still not json"]}, fallback=ScriptedEtomPeFinLLM()
    )
    out2 = env.agent(junk).run(QUESTION)
    assert out2["llm_calls"] == 2 and out2["final_response_source"] == "controlled_failure"
    assert out2["executed_step_count"] == 0 and out2["mcp_invocations"] == 0
    # MCP cap: one allowed invocation, three steps -> loop stops before a 2nd route node.
    llm = StageQueueLLM(scripts={"planner": [_plan(_step("s1"), _step("s2"), _step("s3"))]})
    out3 = env.agent(llm, limits={"max_total_mcp_invocations": 1}).run(QUESTION)
    assert out3["mcp_invocations"] == 1 and len(out3["predicted_tools"]) == 1
    assert out3["stopping_reason"] == "max_total_mcp_calls"
    assert out3["final_response_source"] == "controlled_failure"
    assert out3["budget"]["limits"]["max_total_mcp_invocations"] == 1


def test_10_failed_signature_blocking_in_live_loop(env: Env):
    calls = {"n": 0}
    real_invoke = env.resolver.invoke_api

    def flaky(operation_id, arguments=None, depends_on=None):
        calls["n"] += 1
        return {
            "success": False,
            "executed": False,
            "execution_mode": "etom_routing_only",
            "operation_id": operation_id,
            "business_result": None,
            "error": "ConnectionResetError: connection reset by peer",
            "error_type": "transport_failure",
        }

    env.resolver.invoke_api = flaky  # type: ignore[assignment]
    try:
        same = "look up the same thing"
        llm = StageQueueLLM(
            scripts={
                "planner": [_plan(_step("s1", same), _step("s2", same))],
                "replanner": [keep_plan_then_respond()] * 3,
            }
        )
        out = env.agent(llm).run(QUESTION)
    finally:
        env.resolver.invoke_api = real_invoke  # type: ignore[assignment]
    assert calls["n"] == 1, "second identical (operation, arguments) must be refused, not re-invoked"
    assert out["failed_signature_blocks"] >= 1
    assert out["signature_blocked_count"] >= 1
    assert out["mcp_invocations"] == 1  # blocked attempt is not an MCP invocation
    assert out["predicted_tools"] == []  # no route recorded for failed/blocked calls


def test_16_grounding_blocks_ungrounded_final_answer(env: Env):
    llm = StageQueueLLM(
        scripts={
            "planner": [_plan(_step("s1"))],
            "replanner": [_replan_response("The total is 123456 units.")],
        }
    )
    out = env.agent(llm).run(QUESTION)
    assert out["stopping_reason"] == "ungrounded_replanner_response"
    assert out["error_type"] == "replanner_grounding_failure"
    assert out["final_response_source"] == "controlled_failure"
    assert out["predicted_status"] == "unavailable"


def test_17_task_isolation_between_runs(env: Env):
    agent = env.agent(ScriptedEtomPeFinLLM())
    a = agent.run("Search the web for alpha.")
    budget_a, gateway_a = agent.budget, agent.gateway
    b = agent.run("Search the web for beta.")
    assert agent.budget is not budget_a and agent.gateway is not gateway_a
    assert a["llm_calls"] == b["llm_calls"] == 3
    assert len(a["predicted_tools"]) == len(b["predicted_tools"]) == 1
    assert b["predicted_nodes"][0]["order"] == 0  # route restarts per task
    assert a["evidence_store"] is not b["evidence_store"]
    assert b["signature_blocked_count"] == 0


def test_planner_input_has_no_gold_or_operation_ids(env: Env):
    llm = StageQueueLLM(scripts={"planner": [_plan(_step("s1"))]})
    out = env.agent(llm).run(QUESTION)
    planner_msgs = "\n".join(m["content"] for m in llm.log[0]["messages"])
    for oid in env.registry:
        assert oid not in planner_msgs
    for marker in ("ground_truth", "gold_operations", "accepted_tools", "L1-q"):
        assert marker not in planner_msgs
    replanner_msgs = "\n".join(
        m["content"] for row in llm.log if row["base"] == "replanner" for m in row["messages"]
    )
    # The Replanner may see the operation it ITSELF routed (runtime evidence), nothing else.
    selected = {n["operation_id"] for n in out["predicted_nodes"]}
    for oid in env.registry:
        if oid not in selected:
            assert oid not in replanner_msgs
    for marker in ("ground_truth", "gold_operations", "accepted_tools", "L1-q"):
        assert marker not in replanner_msgs


def test_mock_llm_refuses_gold_context():
    with pytest.raises(RuntimeError):
        ScriptedEtomPeFinLLM().complete(
            [{"role": "user", "content": "ground_truth leak"}], stage="planner"
        )


# ------------------------------------------------------------------ runner-level checks
def _raw_available() -> bool:
    return (ROOT / "data" / "etom" / "raw" / "data" / "mcp_registry.json").exists()


@pytest.mark.skipif(not _raw_available(), reason="pinned ETOM raw checkout not present")
def test_19_20_mock_runner_runs_agent_cannot_forge_live_and_aborted_finalizes(tmp_path):
    from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
        UnifiedPeFinReplanEtomRunner,
    )

    out = tmp_path / "mock"
    runner = UnifiedPeFinReplanEtomRunner(ROOT, out, mode="mock", formal100=True, max_tasks=3)
    meta = runner.execute()
    assert meta["live_pass_claimed"] is False and meta["mode"] == "mock"
    rows = [json.loads(x) for x in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["mode"] == "mock" and row["live_pass_claimed"] is False
        assert row["executed_etom_mcp_tool"] is False and row["real_tool_executions"] == 0
        assert row["llm_calls"] <= 16 and row["mcp_invocations"] <= 8
        assert row["predicted_tools"], "mock agent must route through the gateway"
    summary = json.loads((out / "mock_summary.json").read_text())
    assert summary["live_pass_claimed"] is False

    # resume skips completed rows (no duplicates)
    resumed = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", formal100=True, max_tasks=3, resume=True
    )
    resumed.execute()
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 3

    # aborted finalization keeps rows and never claims live
    runner.run_status = "aborted"
    runner._finalize_metadata(extra={"abort_reason": "test"})
    final_meta = json.loads((out / "run_metadata.json").read_text())
    assert final_meta["run_status"] == "aborted" and final_meta["live_pass_claimed"] is False
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 3


def test_live_mode_is_gated(tmp_path):
    from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
        EtomPeFinReplanRunnerError,
        UnifiedPeFinReplanEtomRunner,
    )

    with pytest.raises(EtomPeFinReplanRunnerError):
        UnifiedPeFinReplanEtomRunner(ROOT, tmp_path / "live", mode="live").execute()
