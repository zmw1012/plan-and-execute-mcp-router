"""ETOM routing-only cost pilot. No network and no real MCP execution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hk_equity_mcp.etom import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_LLM_MODEL,
    EXPECTED_LLM_PROVIDER,
    FIDELITY_NAME,
)
from hk_equity_mcp.etom.adapter import EtomRoutingResolver
from hk_equity_mcp.etom.agent import EtomRoutingAgent
from hk_equity_mcp.etom.convert import convert_registry
from hk_equity_mcp.etom.cost import summarize_costs
from hk_equity_mcp.etom.embeddings import HashEmbeddingClient, load_or_build_catalog_index
from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.llm import FailClosedChatClient, ScriptedChatClient
from hk_equity_mcp.etom.loader import EtomTask, load_official_registry, load_official_tasks
from hk_equity_mcp.etom.prompts import ETOM_ROUTING_SYSTEM_PROMPT, assert_prompt_has_no_gold
from hk_equity_mcp.etom.runner import PilotLimitError, assert_pilot_task_count
from hk_equity_mcp.etom.sampler import (
    assert_lock_matches,
    assert_not_excluded,
    sample_cost_pilot,
)
from hk_equity_mcp.etom.source import file_sha256, raw_root, repo_root_from_here, verify_source
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

ROOT = repo_root_from_here()
RAW = raw_root(ROOT)


def _task(level, query_id, servers, **flags):
    return EtomTask(
        task_id=f"L{level}-q{query_id}",
        level=level,
        query_id=query_id,
        query=f"query {level}-{query_id}",
        ground_truth_tools=[],
        source_file=f"data/queries/level_{level}.json",
        server_names=set(servers),
        step_count=flags.get("step_count", 1),
        has_dependency=flags.get("has_dependency", False),
        single_server_sequential=flags.get("single_server_sequential", False),
        cross_server_dependency=flags.get("cross_server_dependency", False),
        infeasible=flags.get("infeasible", False),
        query_sha256="q",
        ground_truth_sha256="g",
    )


def _synthetic_pool():
    tasks = []
    for level, names in (
        (1, ["alpha", "beta", "gamma", "delta"]),
        (2, ["epsilon", "zeta", "eta", "theta"]),
    ):
        for offset, name in enumerate(names, start=1):
            tasks.append(_task(level, offset, [name]))
    tasks.append(
        _task(3, 1, ["seq"], single_server_sequential=True, has_dependency=True, step_count=2)
    )
    tasks.append(_task(3, 2, ["other"]))
    tasks.append(_task(3, 3, ["third"]))
    tasks.append(
        _task(
            4,
            1,
            ["cross-a", "cross-b"],
            cross_server_dependency=True,
            has_dependency=True,
            step_count=2,
        )
    )
    tasks.append(_task(4, 2, ["cross-c", "cross-d"], cross_server_dependency=True, has_dependency=True))
    tasks.append(_task(4, 3, ["cross-e", "cross-f"], cross_server_dependency=True, has_dependency=True))
    tasks.append(_task(5, 1, [], infeasible=True, step_count=0))
    tasks.append(_task(5, 2, [], infeasible=True, step_count=0))
    tasks.append(_task(5, 3, [], infeasible=True, step_count=0))
    return tasks


def _tiny_registry():
    spec = OperationSpec(
        operation_id="demo-server::echo_tool",
        service_id="demo-server",
        name="echo_tool",
        summary="Echo a short message",
        description="Echo a short message",
        action_terms=["echo", "tool"],
        parameters=[ParameterSpec(name="text", type="string", required=True)],
        invocation_guidance="Input fields: text",
        underlying_tool="echo_tool",
    )
    official = {
        spec.operation_id: {
            "operation_id": spec.operation_id,
            "server_name": "demo-server",
            "tool_name": "echo_tool",
            "description": "Echo a short message",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "pseudo_output_metadata": {"available": False, "reason": "not in pinned registry"},
        }
    }
    return {spec.operation_id: spec}, official


def test_prompt_names_the_experiment_and_hides_gold():
    assert FIDELITY_NAME in ETOM_ROUTING_SYSTEM_PROMPT
    assert_prompt_has_no_gold()
    assert "no business data" in ETOM_ROUTING_SYSTEM_PROMPT
    assert "cannot execute MCP tools" in ETOM_ROUTING_SYSTEM_PROMPT


def test_sampler_is_deterministic_and_covers_required_patterns():
    pool = _synthetic_pool()
    first = sample_cost_pilot(pool, seed=5018)
    second = sample_cost_pilot(pool, seed=5018)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert len(first) == 10
    counts = {level: 0 for level in range(1, 6)}
    for task in first:
        counts[task.level] += 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert any(task.single_server_sequential for task in first if task.level == 3)
    assert any(task.cross_server_dependency for task in first if task.level == 4)
    assert all(task.infeasible and not task.server_names for task in first if task.level == 5)
    assert len({name for task in first for name in task.server_names}) >= 4


def test_pilot_refuses_an_eleventh_task():
    with pytest.raises(PilotLimitError):
        assert_pilot_task_count(11)
    assert_pilot_task_count(10)


def test_model_mismatch_fails_before_network():
    with pytest.raises(Exception):
        FailClosedChatClient(provider="other", model=EXPECTED_LLM_MODEL, base_url="http://example.invalid", api_key="x")
    with pytest.raises(Exception):
        FailClosedChatClient(
            provider=EXPECTED_LLM_PROVIDER,
            model="some-other-model",
            base_url="http://example.invalid",
            api_key="x",
        )


def test_invoke_records_a_route_and_does_not_execute(tmp_path: Path):
    specs, official = _tiny_registry()
    embedder = HashEmbeddingClient()
    index = load_or_build_catalog_index(
        registry=specs,
        embedder=embedder,
        registry_sha256="tiny",
        cache_dir=tmp_path,
    )
    resolver = EtomRoutingResolver(specs, official, index, embedder)
    found = resolver.find_operations("echo a message", top_k=5)
    assert found["success"] is True
    assert found["catalog_size"] == 1
    described = resolver.describe_operation("demo-server::echo_tool")
    assert described["name"] == "echo_tool"
    assert described["description"] == "Echo a short message"
    assert described["input_schema"]["properties"]["text"]["type"] == "string"
    assert described["pseudo_output_metadata"]["available"] is False
    invoked = resolver.invoke_api(
        "demo-server::echo_tool",
        arguments={"text": "hello"},
        depends_on=[],
    )
    assert invoked["executed"] is False
    assert invoked["business_result"] is None
    assert invoked["execution_mode"] == "etom_routing_only"
    assert "hello world payload" not in json.dumps(invoked)
    assert resolver.real_tool_executions == 0
    assert resolver.predicted_nodes[0]["server_name"] == "demo-server"
    assert resolver.predicted_nodes[0]["tool_name"] == "echo_tool"


def test_gold_graph_does_not_enter_the_mock_context(tmp_path: Path):
    specs, official = _tiny_registry()
    embedder = HashEmbeddingClient()
    index = load_or_build_catalog_index(
        registry=specs,
        embedder=embedder,
        registry_sha256="tiny-gold",
        cache_dir=tmp_path,
    )
    resolver = EtomRoutingResolver(specs, official, index, embedder)
    agent = EtomRoutingAgent(
        ScriptedChatClient(),
        resolver,
        temperature=0.0,
        max_tokens=256,
        max_llm_turns=6,
    )
    gold = [
        {
            "tool_id": "GOLD_SENTINEL_TOOL_ZZZ",
            "server_name": "hidden-server",
            "tool_name": "GOLD_SENTINEL_TOOL_ZZZ",
            "description": "This gold description must stay out of the prompt.",
            "dependencies": [0],
        }
    ]
    result = agent.run("Please echo a short message for the user.", gold)
    assert result["real_tool_executions"] == 0
    blob = json.dumps(result["trace"])
    assert "GOLD_SENTINEL_TOOL_ZZZ" not in blob
    assert "ground_truth" not in blob
    assert result["predicted_tools"][0]["tool_name"] == "echo_tool"
    assert all(node.get("business_result", None) is None for node in result["predicted_nodes"])


def test_evaluator_matches_official_graph_rules():
    pytest.importorskip("dataclasses")
    path = RAW / "eval" / "graph_evaluator.py"
    if not path.exists():
        pytest.skip("Pinned ETOM checkout is not present.")
    spec = importlib.util.spec_from_file_location("etom_official_graph_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ground_truth = [
        [{"server_name": "s1", "tool_name": "tool_a", "dependencies": []}],
        [{"server_name": "s1", "tool_name": "tool_b", "dependencies": [0]}],
    ]
    prediction = [
        json.dumps({"server_name": "s1", "tool_name": "tool_a"}),
        json.dumps({"server_name": "s1", "tool_name": "tool_b"}),
    ]
    official = module.GraphEvaluator(level=3).evaluate(ground_truth, prediction)
    ours = evaluate_prediction(
        3,
        [
            {"server_name": "s1", "tool_name": "tool_a", "dependencies": []},
            {"server_name": "s1", "tool_name": "tool_b", "dependencies": [0]},
        ],
        [
            {"server_name": "s1", "tool_name": "tool_a"},
            {"server_name": "s1", "tool_name": "tool_b"},
        ],
    )
    assert ours["precision"] == official.precision
    assert ours["recall"] == official.recall
    assert ours["f1"] == official.f1_score
    assert ours["exact_match"] is True
    swapped = evaluate_prediction(
        3,
        [
            {"server_name": "s1", "tool_name": "tool_a", "dependencies": []},
            {"server_name": "s1", "tool_name": "tool_b", "dependencies": [0]},
        ],
        [
            {"server_name": "s1", "tool_name": "tool_b"},
            {"server_name": "s1", "tool_name": "tool_a"},
        ],
    )
    assert swapped["exact_match"] is False
    assert swapped["recall"] == 1.0
    assert swapped["precision"] < 1.0


def test_level5_empty_route_is_exact_and_a_tool_is_not():
    empty = evaluate_prediction(5, [], [])
    assert empty["exact_match"] is True
    assert empty["f1"] == 1.0
    wrong = evaluate_prediction(5, [], [{"server_name": "s", "tool_name": "t"}])
    assert wrong["exact_match"] is False
    accepted = evaluate_prediction(5, [], [{"server_name": "No", "tool_name": "No"}])
    assert accepted["exact_match"] is True


def test_level2_accepts_one_equivalent_tool():
    ground_truth = [[
        {"server_name": "s1", "tool_name": "a", "dependencies": []},
        {"server_name": "s2", "tool_name": "b", "dependencies": []},
    ]]
    hit = evaluate_prediction(2, ground_truth, [{"server_name": "s2", "tool_name": "b"}])
    assert hit["exact_match"] is True
    miss = evaluate_prediction(2, ground_truth, [{"server_name": "s9", "tool_name": "z"}])
    assert miss["exact_match"] is False


def test_cost_summary_separates_catalog_and_task_cost():
    rows = []
    for level in range(1, 6):
        for _ in range(2):
            rows.append(
                {
                    "task_id": f"L{level}-q{_}",
                    "level": level,
                    "status": "completed",
                    "exact_match": True,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "llm_calls": 4,
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "query_embedding_calls": 1,
                    "query_embedding_tokens": 50,
                    "latency_ms": 10.0 + level,
                    "http_retries": 0,
                    "parse_repairs": 0,
                    "timeout": False,
                    "variable_cost_usd": 0.001,
                    "real_tool_executions": 0,
                }
            )
    summary = summarize_costs(
        rows,
        pricing={
            "llm_input_usd_per_million": 0.3,
            "llm_output_usd_per_million": 1.2,
            "embedding_usd_per_million": 0.02,
            "source": "test",
        },
        catalog={"calls": 38, "retries": 0, "tokens": 200000, "token_source": "provider", "cache_reused": False},
        probe={"llm_calls": 1, "input_tokens": 10, "output_tokens": 5, "embedding_calls": 1, "embedding_tokens": 4, "usd": 0.01},
    )
    assert summary["tasks_total"] == 10
    assert summary["one_time_catalog_index"]["tokens"] == 200000
    assert summary["one_time_catalog_index"]["usd"] == pytest.approx(0.004)
    assert summary["query_embedding_tokens"] == 500
    assert summary["projected_cost_100_tasks"]["point_usd"] > summary["one_time_catalog_index"]["usd"]
    assert summary["projected_cost_200_tasks"]["with_buffer_usd"] >= summary["projected_cost_200_tasks"]["point_usd"]
    assert summary["real_tool_executions"] == 0
    assert summary["results_by_level"]["L5"]["tasks"] == 2


@pytest.mark.skipif(not (RAW / "data" / "mcp_registry.json").exists(), reason="ETOM raw checkout missing")
def test_pinned_source_is_unmodified_and_conversion_is_exact():
    before = {
        rel: file_sha256(RAW / rel)
        for rel in (
            "data/mcp_registry.json",
            "data/queries/level_1.json",
            "data/queries/level_2.json",
            "data/queries/level_3.json",
            "data/queries/level_4.json",
            "data/queries/level_5.json",
        )
    }
    verified = verify_source(ROOT)
    registry = load_official_registry(ROOT, verify=False)
    specs, official = convert_registry(registry)
    assert len(specs) == 2375
    for server in registry["servers"]:
        for tool in server["tools"]:
            op_id = f"{server['server_name']}::{tool['tool_name']}"
            assert official[op_id]["description"] == tool["tool_description"]
            assert official[op_id]["input_schema"] == tool["input_schema"]
            assert official[op_id]["pseudo_output_metadata"]["available"] is False
    after = {rel: file_sha256(RAW / rel) for rel in before}
    assert before == after
    assert verified["pin"]["commit_sha"] == "790f28727f567b88918420d163c524d46644a694"
    tasks = load_official_tasks(ROOT, verify=False)
    locked = assert_lock_matches(ROOT, tasks)
    assert locked["exclude_from_final_etom_sample"] is True
    assert locked["dataset_split"] == "cost_pilot"
    chosen = sample_cost_pilot(tasks)
    assert [task.task_id for task in chosen] == locked["task_ids"]
    assert any(task.single_server_sequential for task in chosen if task.level == 3)
    assert any(task.cross_server_dependency for task in chosen if task.level == 4)
    assert all(task.infeasible for task in chosen if task.level == 5)
    with pytest.raises(Exception):
        assert_not_excluded(ROOT, locked["task_ids"][:1])


def test_adapter_module_does_not_call_the_network():
    source = Path(ROOT / "src" / "hk_equity_mcp" / "etom" / "adapter.py").read_text(encoding="utf-8")
    assert "urlopen" not in source
    assert "import urllib" not in source
    assert "mcp.client" not in source
