"""Official ETOM metrics + L4 separation (offline)."""

from __future__ import annotations

from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.formal_metrics import dependency_accuracy
from hk_equity_mcp.mcp_zero_etom.l4_metrics import evaluate_l4_separate, strict_level_rates


def test_l1_exact_match():
    gold = [{"server_name": "A", "tool_name": "t1"}]
    pred = [{"server_name": "A", "tool_name": "t1"}]
    row = evaluate_prediction(1, gold, pred)
    assert row["exact_match"] is True


def test_l2_equal_function_set():
    gold = [[
        {"server_name": "A", "tool_name": "t1"},
        {"server_name": "B", "tool_name": "t2"},
    ]]
    pred = [{"server_name": "B", "tool_name": "t2"}]
    row = evaluate_prediction(2, gold, pred)
    assert row["exact_match"] is True


def test_l3_dependency_order():
    gold = [
        {"server_name": "S", "tool_name": "a", "dependencies": []},
        {"server_name": "S", "tool_name": "b", "dependencies": [0]},
    ]
    pred_nodes = [
        {"server_name": "S", "tool_name": "a", "depends_on": []},
        {"server_name": "S", "tool_name": "b", "depends_on": [0]},
    ]
    dep = dependency_accuracy(level=3, ground_truth_tools=gold, predicted_nodes=pred_nodes)
    assert dep["dependency_edge_accuracy"] == 1.0
    pred = [{"server_name": "S", "tool_name": "a"}, {"server_name": "S", "tool_name": "b"}]
    row = evaluate_prediction(3, gold, pred)
    assert row["f1"] == 1.0


def test_l4_four_metrics_separate():
    gold = [
        {"server_name": "A", "tool_name": "t1", "dependencies": []},
        {"server_name": "B", "tool_name": "t2", "dependencies": [0]},
    ]
    pred_tools = [
        {"server_name": "A", "tool_name": "t1"},
        {"server_name": "B", "tool_name": "t2"},
    ]
    pred_nodes = [
        {"server_name": "A", "tool_name": "t1", "depends_on": []},
        {"server_name": "B", "tool_name": "t2", "depends_on": [0]},
    ]
    m = evaluate_l4_separate(
        ground_truth_tools=gold,
        predicted_tools=pred_tools,
        predicted_nodes=pred_nodes,
        status="completed",
    )
    assert m["official_union_node_set_em"] is True
    assert m["efs_covering_node_set"] is True
    assert m["cross_server_dependency_correctness"] is True
    assert m["complete_route_success"] is True
    # failure stays in strict denom with 0
    failed = evaluate_l4_separate(
        ground_truth_tools=gold,
        predicted_tools=[],
        predicted_nodes=[],
        status="failed",
    )
    assert failed["strict_score"] == 0.0
    assert failed["complete_route_success"] is False
    rows = [
        {"task_id": f"L4-q{i}", "level": 4, "status": "completed", "complete_route_success": True}
        for i in range(19)
    ] + [{"task_id": "L4-q99", "level": 4, "status": "failed", "complete_route_success": False}]
    rates = strict_level_rates(rows, level=4, key="complete_route_success", expected_n=20)
    assert rates["strict_denominator"] == 20
    assert rates["successes"] == 19
    assert rates["failed_scored_zero"] == 1


def test_l5_empty_route():
    row = evaluate_prediction(5, [], [])
    assert row["exact_match"] is True
    bad = evaluate_prediction(5, [], [{"server_name": "A", "tool_name": "t"}])
    assert bad["exact_match"] is False
