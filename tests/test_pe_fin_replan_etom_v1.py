"""Offline PE-Fin RePlan ETOM-100 transfer tests (no live LLM/embedding/MCP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.etom.adapter import EtomRoutingResolver
from hk_equity_mcp.etom.embeddings import HashEmbeddingClient
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker
from hk_equity_mcp.pe_fin_lite.calculator import safe_eval_expression
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    RETRIEVAL_TOP_K,
    r5_final_score,
    r5_score_from_vectors,
)
from hk_equity_mcp.pe_fin_replan_etom.capability_profile import (
    load_capability_profile,
    validate_capability_profile,
)
from hk_equity_mcp.pe_fin_replan_etom.dataset import assert_frozen_etom100_identity
from hk_equity_mcp.pe_fin_replan_etom.envelope import normalize_invoke_result, normalize_tool_result
from hk_equity_mcp.pe_fin_replan_etom.evidence import (
    compress_evidence,
    make_evidence_store,
    record_derived_evidence,
    record_tool_evidence,
)
from hk_equity_mcp.pe_fin_replan_etom.gateway import EtomGateway
from hk_equity_mcp.pe_fin_replan_etom.grounding import validate_replanner_response
from hk_equity_mcp.pe_fin_replan_etom.identity import (
    DEFAULT_BUDGETS,
    ENGINEERING_SMOKE5_EXPECTED_IDS,
    FROZEN_ORDERED_TASK_IDS_SHA256,
    FROZEN_SAMPLE_SHA256,
    engineering_smoke5_task_ids,
    resume_identity_hash,
)
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    BINDER_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    REPLAN_SCHEMA_VERSION,
    validate_plan_payload,
    validate_replanner_payload,
)
from hk_equity_mcp.pe_fin_replan_etom.tool_binding import (
    ToolBindingError,
    validate_binding_payload,
)
from hk_equity_mcp.pe_fin_replan_etom.tool_retrieval import (
    build_step_search_query,
    rank_r5,
    select_describe_candidates,
)
from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
    UnifiedPeFinReplanEtomRunner,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "pe_fin_replan_etom" / "mini_official.json"


def _official() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _registry(official: dict) -> dict:
    out = {}
    for oid, meta in official.items():
        schema = meta.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        params = [
            ParameterSpec(
                name=str(n),
                type=str((p or {}).get("type") or "string"),
                required=n in required,
                description=str((p or {}).get("description") or ""),
            )
            for n, p in props.items()
        ]
        out[oid] = OperationSpec(
            operation_id=oid,
            service_id=str(meta.get("server_name") or ""),
            name=str(meta.get("tool_name") or ""),
            summary=f"{meta.get('tool_name')} on {meta.get('server_name')}",
            description=str(meta.get("description") or ""),
            action_terms=[str(meta.get("tool_name") or "").replace("_", " ")],
            parameters=params,
            underlying_tool=str(meta.get("tool_name") or ""),
        )
    return out


def _embed_one(embedder, text: str):
    vectors, _usage = embedder.embed([text])
    return vectors[0]


def test_01_manifest_identity_freeze():
    ident = assert_frozen_etom100_identity(ROOT)
    assert ident["sample_sha256"] == FROZEN_SAMPLE_SHA256
    assert ident["ordered_task_ids_sha256"] == FROZEN_ORDERED_TASK_IDS_SHA256
    assert ident["task_count"] == 100


def test_02_capability_profile_no_gold_task_leakage():
    profile = load_capability_profile(ROOT)
    validate_capability_profile(profile)
    blob = json.dumps(profile)
    assert "L1-q" not in blob
    assert "gold_operations" not in blob.lower()
    assert "operation_id" not in blob
    assert profile.get("flags", {}).get("contains_gold") is False
    assert profile["profile_id"] == "etom100_capability_profile_v1"


def test_03_planner_schema():
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "search",
                "objective": "find echo tool",
                "required_outputs": ["text"],
                "depends_on": [],
            }
        ],
    }
    validated = validate_plan_payload(plan)
    assert validated["steps"][0]["id"] == "s1"
    with pytest.raises(Exception):
        validate_plan_payload(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "steps": [
                    {
                        "id": "s1",
                        "type": "retrieve",
                        "domain": "search",
                        "objective": "x",
                        "required_outputs": ["text"],
                        "depends_on": [],
                        "operation_id": "Demo Server::echo",
                    }
                ],
            }
        )


def test_04_05_retrieval_and_r5():
    official = _official()
    registry = _registry(official)
    embedder = HashEmbeddingClient(model_name="mock-hash-embedding-v1")
    vectors = {
        oid: _embed_one(embedder, spec.index_document()) for oid, spec in registry.items()
    }
    query = "echo text query"
    qvec = _embed_one(embedder, query)
    ranked = rank_r5(
        query=query,
        query_vector=qvec,
        registry=registry,
        vectors=vectors,
        top_k=RETRIEVAL_TOP_K,
    )
    assert len(ranked) <= RETRIEVAL_TOP_K
    described = select_describe_candidates(ranked, describe_n=DESCRIBE_N)
    assert len(described) <= DESCRIBE_N
    oid = next(iter(registry))
    metrics = r5_score_from_vectors(
        query=query,
        query_vector=qvec,
        doc_vector=vectors[oid],
        operation=registry[oid],
    )
    assert (
        abs(
            metrics["score"]
            - r5_final_score(
                cosine=metrics["cosine_similarity"],
                query=query,
                operation=registry[oid],
            )
        )
        < 1e-12
    )


def test_06_07_binder():
    payload = {
        "schema_version": BINDER_SCHEMA_VERSION,
        "primary": {"operation_id": "Demo Server::echo", "arguments": {"query": "hi"}},
        "fallback": None,
        "selection_reason": "top described",
        "notes": None,
    }
    # ETOM_BINDER_V1 may not include selection_reason — strip if needed after failure
    contracts = {
        "Demo Server::echo": {
            "json_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            }
        }
    }
    try:
        out = validate_binding_payload(
            payload,
            described_operation_ids=["Demo Server::echo"],
            described_contracts=contracts,
        )
    except Exception:
        payload.pop("selection_reason", None)
        out = validate_binding_payload(
            payload,
            described_operation_ids=["Demo Server::echo"],
            described_contracts=contracts,
        )
    assert out["primary"]["operation_id"] == "Demo Server::echo"

    with pytest.raises(ToolBindingError):
        validate_binding_payload(
            {
                "schema_version": BINDER_SCHEMA_VERSION,
                "primary": {
                    "operation_id": "Not Described::x",
                    "arguments": {"query": "hi"},
                },
                "fallback": None,
            },
            described_operation_ids=["Demo Server::echo"],
            described_contracts=contracts,
        )

    with pytest.raises(ToolBindingError):
        validate_binding_payload(
            {
                "schema_version": BINDER_SCHEMA_VERSION,
                "primary": {"operation_id": "Demo Server::echo", "arguments": {}},
                "fallback": None,
            },
            described_operation_ids=["Demo Server::echo"],
            described_contracts=contracts,
        )


def test_08_09_envelope_evidence():
    env = normalize_invoke_result(
        {
            "success": True,
            "executed": False,
            "execution_mode": "etom_routing_only",
            "operation_id": "Demo Server::echo",
            "business_result": None,
        }
    )
    assert env["kind"] == "routing_only"
    text_env = normalize_tool_result({"ok": True, "value": 12, "id": "abc"})
    assert text_env["raw_result_ref"]
    assert text_env["content_sha256"]
    store = make_evidence_store()
    record_tool_evidence(
        store,
        projection={
            "success": True,
            "provenance": "observed",
            "observed_values": {"value": 12},
            "operation_id": "Demo Server::echo",
            "server_name": "Demo Server",
            "tool_name": "echo",
            "arguments": {"query": "hi"},
            "raw_result_ref": {"sha256": text_env["content_sha256"]},
            "normalized_result": text_env["value"],
        },
        step_id="s1",
    )
    record_derived_evidence(
        store, step_id="s2", output_name="x", result=3.0, formula="1+2", ok=True
    )
    rows = compress_evidence(store)
    assert rows


def test_10_failed_signature_blocking():
    official = _official()
    registry = _registry(official)

    class FakeIndex:
        def rank(self, query, registry, embedder, top_k=5, lexical_weight=0.12):
            return [
                {"operation_id": oid, "score": 1.0 - i * 0.1}
                for i, oid in enumerate(list(registry)[:top_k])
            ]

    resolver = EtomRoutingResolver(
        registry=registry,
        official=official,
        index=FakeIndex(),
        embedder=HashEmbeddingClient(model_name="mock-hash-embedding-v1"),
    )
    budget = BudgetTracker(DEFAULT_BUDGETS)
    gw = EtomGateway(resolver, budget=budget)
    first = gw.invoke("Demo Server::echo", {"query": "hi"})
    assert first.get("success") is True
    gw.guard.record_failure(
        operation_id="Demo Server::echo",
        arguments={"query": "bad"},
        failure_class="protocol_failure",
        error="boom",
    )
    blocked = gw.invoke("Demo Server::echo", {"query": "bad"})
    assert blocked.get("signature_blocked") is True
    ok2 = gw.invoke("Demo Server::echo", {"query": "other"})
    assert ok2.get("signature_blocked") is not True


def test_11_12_13_14_round_and_budgets():
    validate_replanner_payload(
        {
            "schema_version": REPLAN_SCHEMA_VERSION,
            "action": "response",
            "response": {
                "completion": "insufficient_evidence",
                "final_answer": "no evidence",
                "reasoning_brief": "empty",
            },
        },
        response_only=True,
    )
    validate_replanner_payload(
        {
            "schema_version": REPLAN_SCHEMA_VERSION,
            "action": "plan",
            "remaining_steps": [],
        }
    )
    assert DEFAULT_BUDGETS["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8


def test_15_safe_calculator():
    assert safe_eval_expression("1 + 2 * 3", {}) == 7
    with pytest.raises(Exception):
        safe_eval_expression("__import__('os').system('x')", {})


def test_16_grounding():
    ok = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "value is 42",
            "reasoning_brief": "from evidence",
        },
        question="what is the value?",
        evidence_values={"x": 42},
        provenance_class_by_key={"x": "observed"},
    )
    assert ok.get("ok") is True
    bad = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "value is 99",
            "reasoning_brief": "guess",
        },
        question="what is the value?",
        evidence_values={"x": 42},
        provenance_class_by_key={"x": "observed"},
    )
    assert bad.get("ok") is False


def test_17_task_isolation():
    a = BudgetTracker(DEFAULT_BUDGETS)
    b = BudgetTracker(DEFAULT_BUDGETS)
    a.record_llm_call(stage="planner")
    assert a.llm_calls == 1
    assert b.llm_calls == 0


def test_18_resume_identity():
    meta = {
        "condition": "pe_fin_replan_etom",
        "method_version": "pe_fin_replan_etom_v1",
        "protocol_version": "planner_executor_replanner_etom_seven_round_v1",
        "dataset_sample_sha256": FROZEN_SAMPLE_SHA256,
        "dataset_manifest_sha256": "x",
        "ordered_task_ids_sha256": FROZEN_ORDERED_TASK_IDS_SHA256,
        "capability_profile_hash": "p",
        "prompt_hash": "h",
        "schema_set_hash": "s",
        "replanner_schema_hash": "r",
        "grounding_contract_hash": "g",
        "catalog_registry_sha256": "c",
        "operation_documents_sha256": "d",
        "operation_index_identity": {
            "namespace": "pe_fin_replan_etom_operation_embeddings_v1"
        },
        "embedding_cache_namespace": "pe_fin_replan_etom_operation_embeddings_v1",
        "generation_provider": "aliyun",
        "generation_model": "deepseek-v4.1-flash",
        "generation_host": "x",
        "embedding_model": "text-embedding-3-small",
        "embedding_host": "y",
        "budgets": DEFAULT_BUDGETS,
        "evaluator_identity": {"module": "hk_equity_mcp.etom.evaluator"},
        "run_mode": "mock",
        "run_role": "mock",
        "generation_provider_profile_id": None,
        "generation_provider_profile_hash": None,
        "git_commit": "abc",
        "git_dirty": False,
        "fixture_run": False,
    }
    h1 = resume_identity_hash(meta)
    meta2 = dict(meta)
    meta2["max_tasks"] = 3
    assert resume_identity_hash(meta2) == h1
    meta3 = dict(meta)
    meta3["method_version"] = "other"
    assert resume_identity_hash(meta3) != h1
    meta4 = dict(meta)
    meta4["run_role"] = "engineering_live_path_smoke_only"
    assert resume_identity_hash(meta4) != h1


def test_19_20_mock_dry_aborted(tmp_path):
    out = tmp_path / "out"
    runner = UnifiedPeFinReplanEtomRunner(ROOT, out, mode="dry-run", formal100=True)
    result = runner.execute()
    assert result.get("live_pass_claimed") is False
    assert result.get("mode") == "dry-run"

    mock_out = tmp_path / "mock"
    mock_runner = UnifiedPeFinReplanEtomRunner(
        ROOT, mock_out, mode="mock", formal100=True, max_tasks=2
    )
    mock_result = mock_runner.execute()
    assert mock_result.get("live_pass_claimed") is False
    preds = (mock_out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds) == 2
    for line in preds:
        row = json.loads(line)
        assert row["live_pass_claimed"] is False
        assert row["mode"] == "mock"

    mock_runner.run_status = "aborted"
    mock_runner._finalize_metadata(extra={"abort_reason": "test"})
    final_meta = json.loads((mock_out / "run_metadata.json").read_text(encoding="utf-8"))
    assert final_meta["run_status"] == "aborted"
    assert final_meta["live_pass_claimed"] is False


def test_live_mode_gated(tmp_path):
    runner = UnifiedPeFinReplanEtomRunner(ROOT, tmp_path / "live", mode="live")
    with pytest.raises(Exception):
        runner.execute()


def test_step_query_helper():
    q = build_step_search_query(
        {
            "domain": "search",
            "objective": "echo",
            "required_outputs": ["text"],
        }
    )
    assert "echo" in q


def test_engineering_smoke5_ids_first_per_level_no_gold():
    identity = assert_frozen_etom100_identity(ROOT)
    ids = engineering_smoke5_task_ids(identity["ordered_task_ids"])
    assert ids == list(ENGINEERING_SMOKE5_EXPECTED_IDS)
    assert [tid.split("-")[0] for tid in ids] == ["L1", "L2", "L3", "L4", "L5"]
    # Prefix max_tasks=5 is L1-only and must NOT be treated as Smoke-5.
    assert identity["ordered_task_ids"][:5] != ids


def test_engineering_smoke5_runner_isolation(tmp_path):
    with pytest.raises(Exception):
        UnifiedPeFinReplanEtomRunner(
            ROOT, tmp_path / "bad", mode="mock", formal100=True, engineering_smoke5=True
        )
    with pytest.raises(Exception):
        UnifiedPeFinReplanEtomRunner(
            ROOT,
            tmp_path / "bad2",
            mode="mock",
            build_index_only=True,
            require_complete_index=True,
            mock_embeddings=True,
        )
    out = tmp_path / "smoke"
    runner = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", engineering_smoke5=True, formal100=False
    )
    assert runner.run_role == "engineering_live_path_smoke_only"
    assert runner.formal100 is False
    assert runner.max_tasks == 5
    result = runner.execute()
    assert result.get("live_pass_claimed") is False
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["run_role"] == "engineering_live_path_smoke_only"
    assert meta["explicitly_not_formal_accuracy"] is True
    assert meta["formal100"] is False
    assert meta["task_ids"] == list(ENGINEERING_SMOKE5_EXPECTED_IDS)
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [p["task_id"] for p in preds] == list(ENGINEERING_SMOKE5_EXPECTED_IDS)
