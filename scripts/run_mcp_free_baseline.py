#!/usr/bin/env python3
"""Run a closed-book (MCP-free) financial QA baseline.

Only the question is sent to the model. Gold answers and source metadata stay local.
The script deliberately produces a manual-grading CSV because 25 examples are small
and human grading is more defensible than using the tested model as its own judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """You are taking a closed-book financial question-answering benchmark.
You have no MCP tools, no web search, no retrieval system, no uploaded reports, and no
external memory. Answer only from your model knowledge and the information explicitly
included in the question. Do not fabricate exact figures.

Return one JSON object with exactly these fields:
- status: one of answered, clarification_needed, unavailable
- final_answer: a concise answer; include currency, unit, accounting basis, and period when relevant
- reasoning_brief: at most two sentences

If the question is ambiguous, use clarification_needed and state what must be clarified.
If the requested information is not reliably available, use unavailable rather than guessing.
Do not include markdown fences around the JSON."""


def load_env_file(path: Path) -> None:
    """Load a small KEY=VALUE env file without adding a dependency.

    Existing shell variables take precedence. Values are never printed.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 25-question MCP-free pilot")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-flash",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def call_chat_completions(
    *, api_key: str, base_url: str, model: str, question: str, temperature: float, max_tokens: int
) -> tuple[dict, float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    return body, elapsed_ms


def extract_prediction(body: dict) -> tuple[dict, str]:
    raw = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"status": "parse_error", "final_answer": raw, "reasoning_brief": ""}
    return parsed, raw


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = Path(os.getenv("ENV_FILE", str(repo_root / ".env"))).expanduser()
    load_env_file(env_path)
    args = parse_args()
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Set LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY in local .env. Never commit it."
        )

    questions_path = Path(args.questions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    grading_path = output_dir / "manual_grading.csv"
    run_meta_path = output_dir / "run_metadata.json"

    items = load_jsonl(questions_path)
    prediction_rows: list[dict] = []

    with predictions_path.open("w", encoding="utf-8") as prediction_file:
        for index, item in enumerate(items, start=1):
            print(f"[{index:02d}/{len(items)}] {item['id']}", flush=True)
            body, latency_ms = call_chat_completions(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                question=item["question"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            prediction, raw = extract_prediction(body)
            row = {
                "id": item["id"],
                "question": item["question"],
                "prediction": prediction,
                "raw_response": raw,
                "latency_ms": round(latency_ms, 2),
                "usage": body.get("usage", {}),
                "model_returned": body.get("model"),
            }
            prediction_rows.append(row)
            prediction_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            prediction_file.flush()

    with grading_path.open("w", encoding="utf-8", newline="") as grading_file:
        fieldnames = [
            "id", "company", "task_type", "question", "gold_status", "reference_answer",
            "predicted_status", "predicted_answer", "correct_0_or_1", "error_type", "notes",
        ]
        writer = csv.DictWriter(grading_file, fieldnames=fieldnames)
        writer.writeheader()
        by_id = {row["id"]: row for row in prediction_rows}
        for item in items:
            prediction = by_id[item["id"]]["prediction"]
            writer.writerow({
                "id": item["id"],
                "company": item["company"],
                "task_type": item["task_type"],
                "question": item["question"],
                "gold_status": item["gold_status"],
                "reference_answer": item["reference_answer"],
                "predicted_status": prediction.get("status", ""),
                "predicted_answer": prediction.get("final_answer", ""),
                "correct_0_or_1": "",
                "error_type": "",
                "notes": "",
            })

    run_meta_path.write_text(json.dumps({
        "condition": "MCP-Free closed-book",
        "model_requested": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "question_count": len(items),
        "system_prompt": SYSTEM_PROMPT,
        "questions_file": str(questions_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Predictions: {predictions_path}")
    print(f"Manual grading sheet: {grading_path}")
    print(
        "Fill correct_0_or_1 with 1 or 0. "
        f"Accuracy = sum(correct_0_or_1) / {len(items)}."
    )


if __name__ == "__main__":
    main()
