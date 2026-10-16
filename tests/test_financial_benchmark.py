import json
from pathlib import Path

from hk_equity_mcp.evaluation.runner import EvaluationRunner


ROOT = Path(__file__).resolve().parents[1]
FIN_V1 = ROOT / "benchmarks" / "questions" / "financial_questions_v1.jsonl"
FIN_PENDING = ROOT / "benchmarks" / "questions" / "financial_questions_draft_pending.jsonl"


def test_financial_v1_has_twelve_questions():
    lines = [ln for ln in FIN_V1.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 12
    for ln in lines:
        item = json.loads(ln)
        assert item["required_domain"] == "financial"
        assert item["verification_status"] == "verified"


def test_pending_benchmark_not_scored(tmp_path: Path):
    # Mix one pending row with one verified row in a temp file.
    verified = json.loads(FIN_V1.read_text(encoding="utf-8").splitlines()[0])
    pending = json.loads(FIN_PENDING.read_text(encoding="utf-8").splitlines()[0])
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        json.dumps(verified, ensure_ascii=False)
        + "\n"
        + json.dumps(pending, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.csv"
    runner = EvaluationRunner(
        questions_path=str(mixed), output_csv=str(output), mode="flat"
    )
    rows = runner.load_questions()
    assert len(rows) == 1
    assert rows[0]["id"] == verified["id"]
    summary = runner.run()
    assert summary["execution_success_rate"] == 1.0


def test_financial_v1_eval_smoke(tmp_path: Path):
    output = tmp_path / "fin_eval.csv"
    runner = EvaluationRunner(
        questions_path=str(FIN_V1), output_csv=str(output), mode="flat"
    )
    summary = runner.run()
    assert output.exists()
    assert summary["strict_task_success_rate"] == 1.0
    assert summary["execution_success_rate"] == 1.0
