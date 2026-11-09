"""APV-Fin live-path audit spies, JSON mode, idle monitor, resume guards (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.agent import ApvFinAgent
from hk_equity_mcp.apv_fin.identity import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_GENERATION_HOST,
    EXPECTED_GENERATION_MODEL,
    EXPECTED_GENERATION_PROVIDER,
    assert_live_provider_identity,
    worst_case_budget_estimate,
)
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.plan_state import PlanState
from hk_equity_mcp.apv_fin.scheduler import (
    BudgetTracker,
    build_retrieval_query,
    execute_retrieve_step,
    select_next_step,
)
from hk_equity_mcp.apv_fin.structured_parse import normalize_llm_content, parse_structured
from hk_equity_mcp.apv_fin.validator import validate_final_state
from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    UnifiedHkApvFinRunner,
    assert_output_dir_safe,
    load_apv_fin_config,
    load_smoke4_manifest,
    load_smoke4_tasks,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan_payload(steps: Optional[List[Dict[str, Any]]] = None) -> dict:
    return {
        "schema_version": "apv_fin_plan_v1",
        "task_spec": {
            "goal": "answer from evidence",
            "entities": [],
            "periods": [],
            "metrics": ["metric"],
            "constraints": ["cutoff_2026-08-12"],
            "ambiguities": [],
            "answerability_hypothesis": "answerable",
            "answerability_reason": "Required evidence is available from supported sources.",
            "cutoff_date": "2026-08-12",
            "currency_unit_hints": [],
            "metric_basis_hints": [],
        },
        "initial_plan": {
            "steps": (
                steps
                if steps is not None
                else [
                    {
                        "step_id": "s1_retrieve",
                        "step_type": "retrieve",
                        "objective": "retrieve evidence",
                        "depends_on": [],
                        "mandatory": True,
                        "success_criteria": "mcp evidence",
                        "retrieval_query_hint": "hong kong equity",
                        "notes": None,
                    },
                    {
                        "step_id": "s2_reason",
                        "step_type": "reason",
                        "objective": "summarize",
                        "depends_on": ["s1_retrieve"],
                        "mandatory": True,
                        "success_criteria": "summary",
                        "retrieval_query_hint": None,
                        "notes": None,
                    },
                    {
                        "step_id": "s3_answer",
                        "step_type": "answer",
                        "objective": "emit answer",
                        "depends_on": ["s2_reason"],
                        "mandatory": True,
                        "success_criteria": "final",
                        "retrieval_query_hint": None,
                        "notes": None,
                    },
                ]
            ),
            "planning_notes": "spy plan",
        },
    }


def _monitor(decision: str = "continue", **extra) -> dict:
    base = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": decision,
        "decision_reason": f"spy {decision}",
        "ready_step_ids": [],
        "patch": {
            "completed_steps": [],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
        "clarify_question": None,
        "refuse_reason": None,
        "fail_reason": None,
    }
    base.update(extra)
    return base


def _final(status: str = "answered") -> dict:
    return {
        "schema_version": "apv_fin_final_v1",
        "status": status,
        "final_answer": "spy answer",
        "reasoning_brief": "brief",
        "evidence_ids": [],
        "clarification_question": None,
        "refusal_reason": None,
        "currency": None,
        "unit": None,
        "metric_basis": None,
        "period": None,
        "entity": None,
    }


def _script(*objs: dict) -> ScriptedLLMClient:
    return ScriptedLLMClient([json.dumps(o, ensure_ascii=False) for o in objs])


# ---------------------------------------------------------------------------
# Provider / identity
# ---------------------------------------------------------------------------


def test_live_provider_identity_fail_closed():
    assert_live_provider_identity(
        generation_provider=EXPECTED_GENERATION_PROVIDER,
        generation_host=EXPECTED_GENERATION_HOST,
        generation_model=EXPECTED_GENERATION_MODEL,
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_host="api.example.com",
        embedding_provider="openai_compatible",
    )
    with pytest.raises(RuntimeError, match="provider mismatch"):
        assert_live_provider_identity(
            generation_provider="aliyun",
            generation_host=EXPECTED_GENERATION_HOST,
            generation_model=EXPECTED_GENERATION_MODEL,
            embedding_model=EXPECTED_EMBEDDING_MODEL,
        )
    with pytest.raises(RuntimeError, match="host mismatch"):
        assert_live_provider_identity(
            generation_provider=EXPECTED_GENERATION_PROVIDER,
            generation_host="dashscope.aliyuncs.com",
            generation_model=EXPECTED_GENERATION_MODEL,
            embedding_model=EXPECTED_EMBEDDING_MODEL,
        )
    with pytest.raises(RuntimeError, match="model mismatch"):
        assert_live_provider_identity(
            generation_provider=EXPECTED_GENERATION_PROVIDER,
            generation_host=EXPECTED_GENERATION_HOST,
            generation_model="wrong-model",
            embedding_model=EXPECTED_EMBEDDING_MODEL,
        )
    with pytest.raises(RuntimeError, match="embedding model mismatch"):
        assert_live_provider_identity(
            generation_provider=EXPECTED_GENERATION_PROVIDER,
            generation_host=EXPECTED_GENERATION_HOST,
            generation_model=EXPECTED_GENERATION_MODEL,
            embedding_model="text-embedding-ada-002",
        )


def test_config_locks_formal_identity():
    cfg = load_apv_fin_config(root=ROOT)
    assert cfg["generation_provider"] == EXPECTED_GENERATION_PROVIDER
    assert cfg["generation_host"] == EXPECTED_GENERATION_HOST
    assert cfg["generation_model"] == EXPECTED_GENERATION_MODEL
    assert cfg["embedding_model"] == EXPECTED_EMBEDDING_MODEL


def test_worst_case_budget_does_not_retune_limits():
    cfg = load_apv_fin_config(root=ROOT)
    est = worst_case_budget_estimate(cfg["limits"])
    assert est["limits_snapshot"] == cfg["limits"]
    assert est["diagnostic12_worst_case_llm_turns"] == 12 * est["per_task_max_llm_turns"]
    assert est["per_task_max_mcp_invocations"] == cfg["limits"]["max_total_mcp_invocations"]


# ---------------------------------------------------------------------------
# Structured JSON / null / repair
# ---------------------------------------------------------------------------


def test_null_content_fails_closed():
    text, meta = normalize_llm_content(None)
    assert text == ""
    assert meta["content_was_null"] is True
    out = parse_structured(None, "apv_fin_plan_v1")
    assert out["ok"] is False
    assert out["fail_closed"] is True
    assert out["raw_content"] == ""


def test_fenced_and_truncated_json():
    plan = _plan_payload()
    fenced = "```json\n" + json.dumps(plan) + "\n```"
    out = parse_structured(fenced, "apv_fin_plan_v1")
    assert out["ok"] is True
    truncated = json.dumps(plan)[:-20]
    out2 = parse_structured(truncated, "apv_fin_plan_v1")
    assert out2["ok"] is False
    repaired = parse_structured(
        truncated, "apv_fin_plan_v1", repair_content=json.dumps(plan)
    )
    assert repaired["ok"] is True
    assert repaired["repair_count"] == 1
    second = parse_structured(
        truncated, "apv_fin_plan_v1", repair_content="{still broken"
    )
    assert second["ok"] is False
    assert second["fail_closed"] is True


def test_empty_object_and_empty_steps_fail():
    empty = parse_structured("{}", "apv_fin_final_v1")
    assert empty["ok"] is False
    # Explicit empty-object guard on plan schema path.
    empty_plan = parse_structured("{}", "apv_fin_plan_v1")
    assert empty_plan["ok"] is False or empty_plan.get("fail_closed") is True
    bad = _plan_payload(steps=[])
    out = parse_structured(json.dumps(bad), "apv_fin_plan_v1")
    assert out["ok"] is False


def test_invalid_dependency_rejected_by_plan_state():
    plan = _plan_payload(
        steps=[
            {
                "step_id": "a",
                "step_type": "retrieve",
                "objective": "a",
                "depends_on": ["missing"],
                "mandatory": True,
                "success_criteria": "x",
                "retrieval_query_hint": None,
                "notes": None,
            }
        ]
    )
    with pytest.raises(Exception):
        PlanState.from_initial_plan(plan["initial_plan"])


def test_invalid_monitor_patch_fails():
    state = PlanState.from_initial_plan(_plan_payload()["initial_plan"])
    with pytest.raises(Exception):
        state.apply_monitor_patch(
            {
                "completed_steps": [],
                "updated_steps": [{"step_id": "does_not_exist", "objective": "x"}],
                "new_steps": [],
                "removed_steps": [],
            }
        )


# ---------------------------------------------------------------------------
# Execution order / idle monitor / dependencies
# ---------------------------------------------------------------------------


def test_ready_dependencies_and_multi_resolve():
    state = PlanState.from_initial_plan(_plan_payload()["initial_plan"])
    ready = [s.step_id for s in state.ready_steps()]
    assert ready == ["s1_retrieve"]
    assert select_next_step(state).step_id == "s1_retrieve"
    state.mark_running("s1_retrieve")
    state.mark_resolved("s1_retrieve", observation_ids=["obs1"])
    assert [s.step_id for s in state.ready_steps()] == ["s2_reason"]
    # One observation can resolve multiple steps.
    state2 = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "a",
                    "step_type": "retrieve",
                    "objective": "a",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "x",
                },
                {
                    "step_id": "b",
                    "step_type": "reason",
                    "objective": "b",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "y",
                },
            ]
        }
    )
    resolved = state2.resolve_steps_from_observation(["a", "b"], "shared_obs")
    assert resolved == ["a", "b"]
    assert state2.steps["a"].observation_ids == ["shared_obs"]
    assert state2.steps["b"].observation_ids == ["shared_obs"]


def test_same_operation_repeat_budget(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb",
    )
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
    from hk_equity_mcp.apv_fin.plan_state import PlanStep

    ledger = EvidenceLedger()
    budget = BudgetTracker({"max_same_operation_repeats": 0})
    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="hist",
        depends_on=[],
        mandatory=True,
        success_criteria="rows",
        retrieval_query_hint="stock_hk_hist",
    )
    out1 = execute_retrieve_step(
        step=step,
        task_spec={"goal": "close", "metrics": ["close"], "periods": ["2026-07"]},
        resolver=resolver,
        metadata_by_id=meta,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
        ledger=ledger,
        budget=budget,
        invoke_arguments={
            "symbol": "0700",
            "adjust": "",
            "period": "daily",
            "start_date": "20260701",
            "end_date": "20260731",
        },
        select_operation_id="aktools.stock_hk_hist",
        allow_forced_select=True,
    )
    assert out1.get("ok") is True
    out2 = execute_retrieve_step(
        step=step,
        task_spec={"goal": "close", "metrics": ["close"], "periods": ["2026-07"]},
        resolver=resolver,
        metadata_by_id=meta,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
        ledger=ledger,
        budget=budget,
        invoke_arguments={
            "symbol": "0700",
            "adjust": "",
            "period": "daily",
            "start_date": "20260701",
            "end_date": "20260731",
        },
        select_operation_id="aktools.stock_hk_hist",
        allow_forced_select=True,
    )
    assert out2.get("ok") is False
    assert out2.get("error") in {
        "max_same_operation_repeats",
        "identical_successful_or_failed_invocation_not_repeated",
    }
    gateway.close()


def test_forced_select_forbidden_without_allow(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb",
    )
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
    from hk_equity_mcp.apv_fin.plan_state import PlanStep

    out = execute_retrieve_step(
        step=PlanStep(
            step_id="r1",
            step_type="retrieve",
            objective="x",
            depends_on=[],
            mandatory=True,
            success_criteria="y",
        ),
        task_spec={"goal": "x"},
        resolver=resolver,
        metadata_by_id=meta,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
        ledger=EvidenceLedger(),
        budget=BudgetTracker({}),
        select_operation_id="aktools.stock_hk_hist",
        allow_forced_select=False,
    )
    assert out["ok"] is False
    assert out["error"] == "forced_select_forbidden_in_live_mode"
    assert transport.calls == []
    gateway.close()


def test_live_mode_rejects_mock_hooks():
    with pytest.raises(RuntimeError, match="allow_mock_hooks"):
        ApvFinAgent(
            llm=_script(_plan_payload()),
            resolver=MagicMock(),
            config={"allow_mock_hooks": True},
            live_mode=True,
        )
    with pytest.raises(RuntimeError, match="mock hooks"):
        ApvFinAgent(
            llm=_script(_plan_payload()),
            resolver=MagicMock(),
            config={
                "allow_mock_hooks": False,
                "mock_select_operation_id": "aktools.stock_hk_hist",
            },
            live_mode=True,
        )


def test_execution_order_planner_then_execute_then_progressor(tmp_path: Path):
    """Spy: planner → retrieve → deterministic progress (no Monitor on success)."""
    bundle = build_unified_catalog(root=ROOT)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb",
    )
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    llm = _script(
        _plan_payload(),
        {"summary": "ok", "decision": "answer", "resolved": True, "claims": ["c"]},
        _final(),
    )
    agent = ApvFinAgent(
        llm=llm,
        resolver=resolver,
        config={
            "allow_mock_hooks": True,
            "mock_retrieve_arguments": {"*": {"symbol": "0700"}},
            "mock_select_operation_id": (
                "aktools.stock_financial_hk_analysis_indicator_em"
            ),
            "limits": {
                "max_exception_replan_calls_per_task": 2,
                "max_idle_monitor_rounds": 1,
                "max_total_mcp_invocations": 8,
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(bundle.stats.runtime_operation_ids),
        live_mode=False,
    )
    result = agent.run("What is a sample HK equity indicator?")
    types = [t.get("type") for t in result["trace"]]
    assert types[0] == "planner"
    assert any(t in {"scheduler", "step_execution"} for t in types)
    # Normal success path must not require Monitor LLM.
    assert "monitor" not in types or "exceptional_monitor" in types
    assert result["execution_order"][0] == "planner"
    assert "deterministic_progressor" in result["execution_order"]
    assert transport.calls, "retrieve must hit UnifiedRealMcpGateway"
    assert result.get("selected_operations")
    assert result.get("invoked_operations")
    gateway.close()


def test_idle_monitor_stops_without_infinite_loop():
    """Unit: idle counter trips when no ready work and no new observations."""
    budget = BudgetTracker({"max_idle_monitor_rounds": 1, "max_llm_monitor_rounds": 6})
    # Simulate agent idle guard: no execute, no observations.
    for _ in range(5):
        budget.idle_monitor_rounds += 1
        if budget.idle_monitor_rounds > budget.limits["max_idle_monitor_rounds"]:
            budget.stopping_reason = "max_idle_monitor_rounds"
            break
    assert budget.stopping_reason == "max_idle_monitor_rounds"
    assert budget.idle_monitor_rounds == 2
    assert budget.exhausted() == "max_idle_monitor_rounds"

    # Plan with failed gate keeps dependent pending forever (no re-exec of same state).
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "gate",
                    "step_type": "retrieve",
                    "objective": "gate",
                    "depends_on": [],
                    "mandatory": False,
                    "success_criteria": "x",
                },
                {
                    "step_id": "stuck",
                    "step_type": "reason",
                    "objective": "stuck",
                    "depends_on": ["gate"],
                    "mandatory": True,
                    "success_criteria": "y",
                },
            ]
        }
    )
    state.mark_running("gate")
    state.mark_failed("gate", "simulated")
    assert select_next_step(state) is None
    assert [s.step_id for s in state.ready_steps()] == []
    # Same plan state must not re-run gate.
    assert state.steps["gate"].status == "failed"


def test_answer_ready_but_validator_fails_controlled():
    from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger

    plan = PlanState.from_initial_plan(_plan_payload()["initial_plan"])
    # Leave mandatory steps unresolved → validator fails.
    ledger = EvidenceLedger()
    final = _final("answered")
    vr = validate_final_state(
        plan_state=plan,
        ledger=ledger,
        task_spec=_plan_payload()["task_spec"],
        final_payload=final,
        compute_traces=[],
    )
    assert vr["ok"] is False
    assert vr.get("failed_checks")


def test_retrieval_query_from_task_spec():
    from hk_equity_mcp.apv_fin.plan_state import PlanStep

    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="get eps",
        depends_on=[],
        mandatory=True,
        success_criteria="value",
        retrieval_query_hint="earnings per share",
    )
    q = build_retrieval_query(
        step,
        {
            "entities": ["hk_listed_issuer"],
            "metrics": ["eps"],
            "periods": ["FY2025"],
        },
    )
    assert "get eps" in q
    assert "earnings per share" in q
    assert "eps" in q
    assert "FY2025" in q
    # No gold / reference / acceptable ops in query construction surface.
    assert "gold" not in q.lower()
    assert "reference" not in q.lower()


# ---------------------------------------------------------------------------
# Output dir / resume / Smoke-4
# ---------------------------------------------------------------------------


def test_output_dir_refuse_dry_to_live_resume(tmp_path: Path):
    out = tmp_path / "dir"
    out.mkdir()
    meta = {
        "condition": "apv_fin",
        "method_version": "apv_fin_v1",
        "dry_run": True,
        "mock": False,
        "subset_kind": "diagnostic12",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dry-run"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta={**meta, "dry_run": False},
        )


def test_output_dir_refuse_mock_to_live_and_subset_mix(tmp_path: Path):
    out = tmp_path / "dir2"
    out.mkdir()
    meta = {
        "condition": "apv_fin",
        "method_version": "apv_fin_v1",
        "dry_run": False,
        "mock": True,
        "subset_kind": "smoke4",
        "limits": {},
    }
    (out / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mock"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta={**meta, "mock": False},
        )
    with pytest.raises(RuntimeError, match="subset_kind"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta={**meta, "subset_kind": "diagnostic12"},
        )


def test_output_dir_refuse_overwrite_without_resume(tmp_path: Path):
    out = tmp_path / "dir3"
    out.mkdir()
    (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=False,
            run_meta={"subset_kind": "diagnostic12"},
        )


def test_smoke4_manifest_and_dry_run(tmp_path: Path):
    man = load_smoke4_manifest(ROOT)
    assert man["ordered_task_ids"] == [
        "fin_cf_003",
        "fin_cf_023",
        "xd_dev_002",
        "mkt_dev_014",
    ]
    assert len(man["ordered_task_ids"]) == 4
    tasks = load_smoke4_tasks(ROOT)
    assert [t["task_id"] for t in tasks] == man["ordered_task_ids"]
    out = tmp_path / "smoke4_dry"
    meta = UnifiedHkApvFinRunner(
        output_dir=out,
        split="exploratory_all",
        smoke4=True,
        dry_run=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        root=ROOT,
    ).run()
    assert meta["smoke4"] is True
    assert meta["subset_kind"] == "smoke4"
    assert meta["not_formal_accuracy_result"] is True
    assert meta["smoke4_manifest_hash"] == man["manifest_hash"]
    assert meta["question_count"] == 4
    assert meta["generation_provider"] == EXPECTED_GENERATION_PROVIDER
    assert meta["generation_model"] == EXPECTED_GENERATION_MODEL


def test_metadata_no_company_whitelist():
    meta = load_and_validate_metadata(root=ROOT)
    blob = json.dumps(meta, ensure_ascii=False)
    for name in ("Tencent", "HSBC", "Alibaba", "Pop Mart", "Xiaomi"):
        assert name not in blob
    # No 5-digit issuer allowlist arrays in metadata.
    for row in meta.values():
        for key, val in row.items():
            if "issuer" in str(key).lower() and isinstance(val, list):
                assert not all(
                    isinstance(x, str) and x.isdigit() and len(x) in {4, 5} for x in val
                ), f"issuer code whitelist in {row.get('operation_id')} key={key}"
