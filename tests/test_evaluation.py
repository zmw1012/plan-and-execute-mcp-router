from pathlib import Path

from hk_equity_mcp.evaluation.runner import EvaluationRunner


def test_evaluation_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    questions = root / "benchmarks" / "questions" / "sample_questions.jsonl"
    output = tmp_path / "eval_results.csv"
    runner = EvaluationRunner(
        questions_path=str(questions),
        output_csv=str(output),
        mode="flat",
    )
    summary = runner.run()
    assert output.exists()
    assert summary["execution_success_rate"] == 1.0
    assert summary["tool_selection_accuracy"] == 1.0
    assert summary["strict_task_success_rate"] == 1.0
