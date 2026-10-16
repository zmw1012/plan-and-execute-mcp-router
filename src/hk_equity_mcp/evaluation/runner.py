"""Minimal evaluation runner: JSONL in, CSV out."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hk_equity_mcp.agent.router import AgentRouter
from hk_equity_mcp.evaluation.metrics import (
    MetricCalculator,
    answer_contains_reference,
    args_match,
    evidence_overlap,
)


class EvaluationRunner:
    def __init__(
        self,
        questions_path: str,
        output_csv: str,
        mode: str = "flat",
        agent_config: Optional[str] = None,
    ):
        self.questions_path = Path(questions_path)
        self.output_csv = Path(output_csv)
        self.mode = mode
        self.agent = AgentRouter(config_path=agent_config, mode=mode)
        self.metrics = MetricCalculator()

    def load_questions(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with self.questions_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                for key in (
                    "id",
                    "question",
                    "required_domain",
                    "gold_tools",
                    "gold_arguments",
                    "reference_answer",
                    "supporting_evidence",
                ):
                    if key not in item:
                        raise ValueError(
                            f"Missing field '{key}' in {self.questions_path}:{line_no}"
                        )
                # Pending items stay in JSONL for draft review but are not scored.
                if str(item.get("verification_status", "verified")).lower() == "pending":
                    continue
                rows.append(item)
        return rows

    def evaluate_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        gold_tools = list(item["gold_tools"])
        gold_tool = gold_tools[0] if gold_tools else None
        gold_args = dict(item["gold_arguments"])

        result = self.agent.run(
            question=item["question"],
            tool_name=gold_tool,
            arguments=gold_args,
        )

        predicted_tool = result["selected_tool"]
        predicted_args = result.get("arguments") or {}
        tool_result = result["tool_result"]
        expect_error = bool(item.get("expect_error"))
        tool_correct = predicted_tool in gold_tools
        args_correct = args_match(predicted_args, gold_args)

        if expect_error:
            # Structured failure is the gold outcome for ambiguity / mismatch items.
            execution_success = tool_result.get("success") is False
            answer_correct = answer_contains_reference(
                tool_result.get("error"), item["reference_answer"]
            )
            evidence_correct = evidence_overlap(
                [tool_result.get("error") or ""],
                item.get("supporting_evidence") or [],
            )
        else:
            execution_success = bool(tool_result.get("success"))
            answer_correct = answer_contains_reference(
                result.get("answer"), item["reference_answer"]
            )
            evidence_correct = evidence_overlap(
                result.get("evidence") or [], item.get("supporting_evidence") or []
            )

        wrong_or_unnecessary = not tool_correct
        strict_success = (
            tool_correct
            and args_correct
            and execution_success
            and answer_correct
            and evidence_correct
        )

        return {
            "id": item["id"],
            "question": item["question"],
            "required_domain": item["required_domain"],
            "gold_tools": gold_tools,
            "gold_arguments": gold_args,
            "reference_answer": item["reference_answer"],
            "supporting_evidence": item.get("supporting_evidence") or [],
            "predicted_tool": predicted_tool,
            "predicted_tools": [predicted_tool] if predicted_tool else [],
            "predicted_arguments": predicted_args,
            "predicted_answer": result.get("answer"),
            "predicted_evidence": result.get("evidence") or [],
            "mode": result.get("mode"),
            "tool_correct": tool_correct,
            "args_correct": args_correct,
            "execution_success": execution_success,
            "answer_correct": answer_correct,
            "evidence_correct": evidence_correct,
            "wrong_or_unnecessary_tool": wrong_or_unnecessary,
            "strict_success": strict_success,
            "tool_calls": 1,
            "tokens": result.get("tokens", 0),
            "latency_ms": result.get("latency_ms", 0),
        }

    def run(self) -> Dict[str, float]:
        questions = self.load_questions()
        rows = [self.evaluate_one(item) for item in questions]
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "id",
            "question",
            "required_domain",
            "gold_tools",
            "gold_arguments",
            "reference_answer",
            "supporting_evidence",
            "predicted_tool",
            "predicted_arguments",
            "predicted_answer",
            "predicted_evidence",
            "mode",
            "tool_correct",
            "args_correct",
            "execution_success",
            "answer_correct",
            "evidence_correct",
            "wrong_or_unnecessary_tool",
            "strict_success",
            "tool_calls",
            "tokens",
            "latency_ms",
        ]
        with self.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                for key in (
                    "gold_tools",
                    "gold_arguments",
                    "supporting_evidence",
                    "predicted_arguments",
                    "predicted_evidence",
                ):
                    csv_row[key] = json.dumps(csv_row.get(key), ensure_ascii=False)
                writer.writerow(csv_row)

        summary = self.metrics.summarize(rows)
        summary_path = self.output_csv.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary


def _load_eval_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        default = Path(__file__).resolve().parents[3] / "configs" / "evaluation.yaml"
        path = str(default)
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run mock HK equity evaluation")
    parser.add_argument("--config", default=None, help="Path to evaluation.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", default=None)
    args = parser.parse_args(argv)

    cfg = _load_eval_config(args.config)
    root = Path(__file__).resolve().parents[3]
    questions = args.questions or cfg.get(
        "questions_path", "benchmarks/questions/sample_questions.jsonl"
    )
    output = args.output or cfg.get("output_csv", "outputs/eval_results.csv")
    mode = args.mode or cfg.get("experiment_mode", "flat")

    questions_path = Path(questions)
    if not questions_path.is_absolute():
        questions_path = root / questions_path
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path

    runner = EvaluationRunner(
        questions_path=str(questions_path),
        output_csv=str(output_path),
        mode=mode,
    )
    summary = runner.run()
    print(f"Wrote results to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
