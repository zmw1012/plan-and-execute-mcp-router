"""Offline tests for RePlan-4 round-depth ablation (no network / no live)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from artifact_gates import hk_artifact_root, require_hk_artifact

from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
    REPLAN_FLAGS,
)
from hk_equity_mcp.pe_fin_replan.identity import (
    DEFAULT_BUDGETS,
    replan_protocol_identity,
)
from hk_equity_mcp.pe_fin_replan.round_policy import (
    DEFAULT_REPLAN_ROUND_CAP,
    PE_FIN_REPLAN_R4_CONDITION,
    PE_FIN_REPLAN_R4_METHOD_VERSION,
    PE_FIN_REPLAN_R4_PROTOCOL_VERSION,
    apply_round_cap_to_limits,
    budgets_for_round_cap,
    identity_labels_for_round_cap,
    loop_policy_for_round_cap,
    replan_flags_for_round_cap,
    round_policy_hash,
    validate_replan_round_cap,
)
from hk_equity_mcp.pe_fin_replan.schemas import LOOP_POLICY
from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
    PE_RESUME_IDENTITY_KEYS,
    UnifiedHkPeFinLiteRunner,
    assert_output_dir_safe,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()
RUN3_PRED = (
    ARTIFACT_ROOT
    / "outputs"
    / "pe_fin_replan_diagnostic20_live_v1_4_1_run3"
    / "predictions.jsonl"
)
RUN3_PRED_SHA = (
    "5264af94db3451eedc10517027420b0b500cb63ba9016404c511b99f4bad810d"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_round_cap_remains_seven_and_r7_identity_unchanged():
    assert DEFAULT_REPLAN_ROUND_CAP == 7
    assert DEFAULT_BUDGETS["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_replan_rounds"] == 7
    assert DEFAULT_BUDGETS["max_executed_steps"] == 7
    assert LOOP_POLICY["max_execution_rounds"] == 7
    assert LOOP_POLICY["round7_response_only"] is True
    assert PE_FIN_REPLAN_METHOD_VERSION == "pe_fin_replan_v1_4_1"
    assert PE_FIN_REPLAN_PROTOCOL_VERSION == (
        "planner_executor_replanner_contract_v4_1"
    )
    assert REPLAN_FLAGS["max_execution_rounds"] == 7
    ident7 = replan_protocol_identity()
    assert ident7["condition"] == PE_FIN_REPLAN_CONDITION
    assert ident7["method_version"] == PE_FIN_REPLAN_METHOD_VERSION
    assert ident7["protocol_version"] == PE_FIN_REPLAN_PROTOCOL_VERSION
    assert "replan_round_cap" not in ident7  # historical shape
    assert ident7["flags"] == REPLAN_FLAGS
    assert ident7["budgets"]["max_execution_rounds"] == 7

    runner = UnifiedHkPeFinLiteRunner(
        output_dir=ROOT / "outputs" / "_replan_r7_default_tmp",
        split="development",
        replan_enabled=True,
        dry_run=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    assert runner.replan_round_cap == 7
    assert runner.method_version == PE_FIN_REPLAN_METHOD_VERSION
    assert runner.protocol_version == PE_FIN_REPLAN_PROTOCOL_VERSION
    assert runner.condition == PE_FIN_REPLAN_CONDITION
    assert runner.limits["max_execution_rounds"] == 7
    assert runner.limits["max_total_llm_calls_per_task"] == 16
    assert runner.limits["max_total_mcp_invocations"] == 8


def test_round4_identity_is_isolated():
    labels = identity_labels_for_round_cap(4)
    assert labels["condition"] == PE_FIN_REPLAN_R4_CONDITION
    assert labels["method_version"] == PE_FIN_REPLAN_R4_METHOD_VERSION
    assert labels["protocol_version"] == PE_FIN_REPLAN_R4_PROTOCOL_VERSION
    assert labels["base_method"] == "pe_fin_replan_v1_4_1"
    assert labels["replan_round_cap"] == 4
    ident4 = replan_protocol_identity(round_cap=4)
    ident7 = replan_protocol_identity(round_cap=7)
    assert ident4["method_version"] != ident7["method_version"]
    assert ident4["condition"] != ident7["condition"]
    assert ident4["round_policy_hash"] == round_policy_hash(4)
    assert round_policy_hash(4) != round_policy_hash(7)
    assert ident4["budgets"]["max_execution_rounds"] == 4
    assert ident4["budgets"]["max_total_llm_calls_per_task"] == 16
    flags4 = replan_flags_for_round_cap(4)
    assert flags4["round4_response_only"] is True
    assert flags4["round7_response_only"] is False
    assert flags4["round_depth_ablation"] is True

    runner = UnifiedHkPeFinLiteRunner(
        output_dir=ROOT / "outputs" / "_replan_r4_identity_tmp",
        split="development",
        replan_enabled=True,
        replan_round_cap=4,
        dry_run=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    assert runner.replan_round_cap == 4
    assert runner.condition == PE_FIN_REPLAN_R4_CONDITION
    assert runner.method_version == PE_FIN_REPLAN_R4_METHOD_VERSION
    assert runner.protocol_version == PE_FIN_REPLAN_R4_PROTOCOL_VERSION
    assert runner.limits["max_execution_rounds"] == 4
    assert runner.limits["max_executed_steps"] == 4
    assert runner.limits["max_replan_rounds"] == 4
    assert runner.limits["max_total_llm_calls_per_task"] == 16
    assert runner.limits["max_total_mcp_invocations"] == 8


def test_round_cap_cli_gate_without_replan():
    with pytest.raises(RuntimeError, match="replan-round-cap requires"):
        UnifiedHkPeFinLiteRunner(
            output_dir=ROOT / "outputs" / "_replan_r4_bad_tmp",
            split="development",
            replan_enabled=False,
            replan_round_cap=4,
            dry_run=True,
            mock=True,
            allow_dirty=True,
            root=ROOT,
        )


def test_invalid_round_cap_rejected():
    with pytest.raises(ValueError):
        validate_replan_round_cap(5)
    with pytest.raises(ValueError):
        validate_replan_round_cap(3)


def test_loop_policy_round4_response_only_contract():
    p4 = loop_policy_for_round_cap(4)
    assert p4["max_execution_rounds"] == 4
    assert p4["response_only_round"] == 4
    assert p4["round4_response_only"] is True
    assert p4["round7_response_only"] is False
    assert p4["final_round_response_only"] is True
    # Frozen default LOOP_POLICY remains 7.
    assert LOOP_POLICY["max_execution_rounds"] == 7
    assert LOOP_POLICY["round7_response_only"] is True


class _Scripted:
    def __init__(self, contents: List[str], model: str = "mock"):
        self._q = list(contents)
        self.model = model
        self.calls: List[Dict[str, Any]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None, **kw):
        content = self._q.pop(0) if self._q else "{}"
        self.calls.append({"stage": stage, "content": content})
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "latency_ms": 1.0,
            "model": self.model,
            "provider": "mock",
            "finish_reason": "stop",
        }


def _plan_json(steps: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {"schema_version": "pe_fin_lite_plan_v1", "steps": steps},
        ensure_ascii=False,
    )


def _replan_plan(steps: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "schema_version": "pe_fin_replan_v1_4",
            "action": "plan",
            "remaining_steps": steps,
        },
        ensure_ascii=False,
    )


def _replan_response() -> str:
    return json.dumps(
        {
            "schema_version": "pe_fin_replan_v1_4",
            "action": "response",
            "response": {
                "completion": "answered",
                "final_answer": "ok",
                "reasoning_brief": "grounded mock",
            },
        },
        ensure_ascii=False,
    )


def test_round4_forces_response_only_and_blocks_fifth_round():
    from hk_equity_mcp.pe_fin_replan.replanner import run_replanner

    plan_payload = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "remaining_steps": [],
    }
    # First call returns plan under response_only → triggers one repair;
    # second still plan → controlled failure.
    llm = _Scripted(
        [json.dumps(plan_payload), json.dumps(plan_payload)]
    )
    out = run_replanner(
        llm=llm,  # type: ignore[arg-type]
        compact_state={"question": "q", "past_steps": [], "remaining_steps": []},
        response_only=True,
        execution_round=4,
        max_execution_rounds=4,
    )
    assert out["ok"] is False
    assert out["response_only"] is True
    assert out["repair_count"] == 1
    # Schema path rejects plan under response_only; one full repair; still plan → fail.
    assert "action_response" in str(out.get("error") or "") or out.get(
        "error"
    ) == "forced_final_round_plan_not_allowed"
    assert out["stopping_reason"] in {
        "invalid_replanner_output_after_repair",
        "forced_final_round_plan_after_repair",
        "round7_plan_after_repair",
    }


def test_round4_response_only_repair_then_success():
    from hk_equity_mcp.pe_fin_replan.replanner import run_replanner

    bad = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "remaining_steps": [],
    }
    good = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "response",
        "response": {
            "completion": "insufficient_evidence",
            "final_answer": "",
            "reasoning_brief": "budgeted final round",
        },
    }
    llm = _Scripted([json.dumps(bad), json.dumps(good)])
    out = run_replanner(
        llm=llm,  # type: ignore[arg-type]
        compact_state={"question": "q", "past_steps": [], "remaining_steps": []},
        response_only=True,
        execution_round=4,
        max_execution_rounds=4,
    )
    assert out["ok"] is True
    assert out["repair_count"] == 1
    assert out["payload"]["action"] == "response"


def test_rounds_1_to_3_allow_plan():
    from hk_equity_mcp.pe_fin_replan.replanner import run_replanner

    plan = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "remaining_steps": [
            {
                "id": "s9",
                "type": "reason",
                "domain": "general",
                "objective": "summarize",
                "required_outputs": [],
                "depends_on": [],
            }
        ],
    }
    llm = _Scripted([json.dumps(plan)])
    out = run_replanner(
        llm=llm,  # type: ignore[arg-type]
        compact_state={"question": "q", "past_steps": [], "remaining_steps": []},
        response_only=False,
        execution_round=2,
        max_execution_rounds=4,
    )
    assert out["ok"] is True
    assert out["payload"]["action"] == "plan"


def test_agent_round_cap_four_no_fifth_round(monkeypatch):
    """Agent loop with cap=4 never reaches round 5."""
    from hk_equity_mcp.pe_fin_replan.agent import PeFinReplanAgent

    step = {
        "id": "s1",
        "type": "reason",
        "domain": "general",
        "objective": "noop reason",
        "required_outputs": [],
        "depends_on": [],
    }
    planner_plan = _plan_json([step])
    contents = [planner_plan]
    for i in range(3):
        contents.append(json.dumps({"summary": "x", "claims": [], "gaps": []}))
        contents.append(
            _replan_plan(
                [
                    {
                        "id": f"r{i+2}",
                        "type": "reason",
                        "domain": "general",
                        "objective": f"continue {i}",
                        "required_outputs": [],
                        "depends_on": [],
                    }
                ]
            )
        )
    contents.append(json.dumps({"summary": "x", "claims": [], "gaps": []}))
    contents.append(_replan_response())

    class _LLM:
        def __init__(self):
            self._q = list(contents)
            self.model = "mock"

        def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None, **kw):
            if not self._q:
                raise AssertionError(f"unexpected LLM call stage={stage}")
            content = self._q.pop(0)
            return {
                "content": content,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "latency_ms": 1.0,
                "model": self.model,
                "provider": "mock",
                "finish_reason": "stop",
            }

    class _FakeResolver:
        def list_underlying_operation_ids(self):
            return []

    limits = apply_round_cap_to_limits(budgets_for_round_cap(4), cap=4)
    agent = PeFinReplanAgent(
        llm=_LLM(),  # type: ignore[arg-type]
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        capability_profile={"profile_id": "test"},
        config={"limits": limits, "use_mock_binder": True},
        runtime_operation_ids=[],
        live_mode=False,
        allow_hkex_download=False,
    )

    def _fake_execute_one_step(*, state, step, question):
        sid = str(step.get("id"))
        past = {
            "step_id": sid,
            "step_type": step.get("type"),
            "status": "ok",
            "execution_ok": True,
            "failure_reason": None,
            "missing_outputs": [],
            "values": {"note": "mock"},
            "ok": True,
            "funnel": {},
        }
        state.evidence_store.values["note"] = "mock"
        return {"past": past, "result": past, "usage": {}, "funnel": {}, "ok": True}

    monkeypatch.setattr(agent, "_execute_one_step", _fake_execute_one_step)
    monkeypatch.setattr(
        "hk_equity_mcp.pe_fin_replan.agent.validate_replanner_response",
        lambda **kwargs: {"ok": True, "reasons": [], "classifications": []},
    )
    monkeypatch.setattr(
        "hk_equity_mcp.pe_fin_replan.agent.is_premature_insufficient",
        lambda **kwargs: False,
    )

    out = agent.run("What is the mock answer?")
    assert out["max_execution_rounds"] == 4
    assert out["execution_round_count"] <= 4
    assert out["replan_round_count"] <= 4
    assert out["replan_round_cap"] == 4
    assert out["response_only_round"] == 4
    assert out["round4_response_only"] is True
    assert out["round7_response_only"] is False
    assert out["method_version"] == PE_FIN_REPLAN_R4_METHOD_VERSION
    stages = [t for t in out["trace"] if t.get("stage") == "replanner"]
    assert all(t.get("round", 0) <= 4 for t in stages)
    assert not any(t.get("round") == 5 for t in out["trace"])


def test_resume_refuses_r7_into_r4(tmp_path: Path):
    out = tmp_path / "r7_to_r4"
    out.mkdir()
    existing = {
        "condition": PE_FIN_REPLAN_CONDITION,
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
        "architecture": "plan_execute_replan",
        "subset_kind": "replan_smoke5",
        "mock": True,
        "dry_run": False,
        "replan_round_cap": 7,
        "max_execution_rounds": 7,
        "max_replan_rounds": 7,
        "response_only_round": 7,
        "round_policy_hash": round_policy_hash(7),
    }
    (out / "run_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(existing)
    current.update(
        {
            "condition": PE_FIN_REPLAN_R4_CONDITION,
            "method_version": PE_FIN_REPLAN_R4_METHOD_VERSION,
            "protocol_version": PE_FIN_REPLAN_R4_PROTOCOL_VERSION,
            "replan_round_cap": 4,
            "max_execution_rounds": 4,
            "max_replan_rounds": 4,
            "response_only_round": 4,
            "round_policy_hash": round_policy_hash(4),
        }
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta=current,
        )


def test_resume_refuses_r4_into_r7(tmp_path: Path):
    out = tmp_path / "r4_to_r7"
    out.mkdir()
    existing = {
        "condition": PE_FIN_REPLAN_R4_CONDITION,
        "method_version": PE_FIN_REPLAN_R4_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_R4_PROTOCOL_VERSION,
        "architecture": "plan_execute_replan",
        "subset_kind": "replan_smoke5",
        "mock": True,
        "dry_run": False,
        "replan_round_cap": 4,
        "max_execution_rounds": 4,
        "max_replan_rounds": 4,
        "response_only_round": 4,
        "round_policy_hash": round_policy_hash(4),
    }
    (out / "run_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(existing)
    current.update(
        {
            "condition": PE_FIN_REPLAN_CONDITION,
            "method_version": PE_FIN_REPLAN_METHOD_VERSION,
            "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
            "replan_round_cap": 7,
            "max_execution_rounds": 7,
            "max_replan_rounds": 7,
            "response_only_round": 7,
            "round_policy_hash": round_policy_hash(7),
        }
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta=current,
        )


def test_resume_keys_include_round_policy():
    for key in (
        "replan_round_cap",
        "max_execution_rounds",
        "max_replan_rounds",
        "response_only_round",
        "round_policy_hash",
        "condition",
        "method_version",
        "protocol_version",
    ):
        assert key in PE_RESUME_IDENTITY_KEYS


def test_mock_trajectory_retrieve_retrieve_compute_response(monkeypatch):
    """Scripted R4 path: retrieve → retrieve → reason → response."""
    from hk_equity_mcp.pe_fin_replan.agent import PeFinReplanAgent

    def _step(sid: str, stype: str, objective: str, outputs=None):
        return {
            "id": sid,
            "type": stype,
            "domain": "market" if stype == "retrieve" else "general",
            "objective": objective,
            "required_outputs": list(outputs or []),
            "depends_on": [],
            "entity": None,
            "period": None,
            "date": None,
            "compute_spec": None,
            "workflow": None,
        }

    s1 = _step("s1", "retrieve", "retrieve first", ["close"])
    s2 = _step("s2", "retrieve", "retrieve second", ["close"])
    s3 = _step("s3", "reason", "reason over retrieved evidence")
    # Modify reason objective so plan-diff counts as progress (avoids no-progress stop).
    s3b = _step("s3", "reason", "reason over both retrieves")
    plan_s = _plan_json([s1, s2, s3])
    keep2 = _replan_plan([s2, s3b])
    keep3 = _replan_plan([_step("s3c", "reason", "final reason synthesis")])
    final = _replan_response()
    reason = json.dumps({"summary": "ok", "claims": [], "gaps": []})

    class _FakeResolver:
        def list_underlying_operation_ids(self):
            return []

    class _LLM:
        def __init__(self):
            self.model = "mock"
            self.replan_calls = 0

        def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None, **kw):
            st = str(stage or "")
            if st.startswith("planner"):
                content = plan_s
            elif st.startswith("reason"):
                content = reason
            elif st.startswith("replanner"):
                self.replan_calls += 1
                if self.replan_calls == 1:
                    content = keep2
                elif self.replan_calls == 2:
                    content = keep3
                else:
                    content = final
            else:
                content = final
            return {
                "content": content,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "latency_ms": 1.0,
                "model": self.model,
                "provider": "mock",
                "finish_reason": "stop",
            }

    agent = PeFinReplanAgent(
        llm=_LLM(),  # type: ignore[arg-type]
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        capability_profile={"profile_id": "test"},
        config={
            "limits": apply_round_cap_to_limits(budgets_for_round_cap(4), cap=4),
            "use_mock_binder": True,
        },
        runtime_operation_ids=[],
        live_mode=False,
        allow_hkex_download=False,
    )

    def _fake_execute_one_step(*, state, step, question):
        sid = str(step.get("id"))
        stype = str(step.get("type"))
        values = {"close": 1.0} if stype == "retrieve" else {"note": "ok"}
        state.evidence_store.values.update(values)
        past = {
            "step_id": sid,
            "step_type": stype,
            "status": "ok",
            "execution_ok": True,
            "failure_reason": None,
            "missing_outputs": [],
            "values": values,
            "ok": True,
            "funnel": {},
        }
        return {"past": past, "result": past, "usage": {}, "funnel": {}, "ok": True}

    import hk_equity_mcp.pe_fin_replan.agent as agent_mod

    monkeypatch.setattr(
        agent_mod,
        "validate_replanner_response",
        lambda **kwargs: {"ok": True, "reasons": [], "classifications": []},
    )
    monkeypatch.setattr(agent_mod, "is_premature_insufficient", lambda **kwargs: False)
    monkeypatch.setattr(agent, "_execute_one_step", _fake_execute_one_step)
    out = agent.run("mock trajectory")
    assert out.get("error") is None, out.get("error")
    assert out["execution_round_count"] == 3
    assert out["execution_round_count"] <= 4
    types = [p.get("step_type") for p in out["past_steps"]]
    assert types == ["retrieve", "retrieve", "reason"]
    assert out["final_payload"]["completion"] == "answered"
    assert out["llm_calls"] <= 16
    assert out["budget"]["mcp_invocations"] <= 8


def test_frozen_run3_raw_sha_unchanged():
    require_hk_artifact("outputs/pe_fin_replan_diagnostic20_live_v1_4_1_run3/predictions.jsonl")
    assert RUN3_PRED.exists()
    assert _sha(RUN3_PRED) == RUN3_PRED_SHA
