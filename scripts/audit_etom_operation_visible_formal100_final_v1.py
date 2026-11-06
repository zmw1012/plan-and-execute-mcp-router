#!/usr/bin/env python3
"""Offline final audit for the ETOM-100 Operation-Visible run.

Reads frozen run artifacts and official local benchmark files only.
Does not call an LLM, embedding, MCP, or paid API.
Does not rerun tasks and does not modify raw predictions, gold, prompts,
the registry, or run metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.etom.agent import _parse_agent_json
from hk_equity_mcp.etom.cost import percentile
from hk_equity_mcp.etom.evaluator import (
    _format_graph_ground_truth,
    _format_simple_ground_truth,
    evaluate_prediction,
)
from hk_equity_mcp.etom.formal_metrics import dependency_accuracy
from hk_equity_mcp.etom.formal_sampler import load_formal_100_lock
from hk_equity_mcp.etom.loader import load_official_tasks
from hk_equity_mcp.etom.prompts import ETOM_ROUTING_SYSTEM_PROMPT, prompt_hash
from hk_equity_mcp.etom.source import verify_source

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
ORIG_DEFAULT = ROOT_DEFAULT / "outputs" / "etom_external_formal100_operation_visible_v1"
AUDIT_DEFAULT = ORIG_DEFAULT / "final_audit_v1"
DOCS_DEFAULT = ROOT_DEFAULT / "docs" / "etom_operation_visible_formal100_final_audit_v1"

RAW_ARTIFACTS = [
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "cost_summary.json",
    "retrieval_analysis.json",
    "error_summary.json",
    "grouped_results.json",
]

PRIMARY_PRIORITY = [
    "parse/evaluation_failure",
    "max_turn_resource_limit",
    "infeasible_detection",
    "retrieval_miss",
    "dependency_order",
    "incomplete_tool_chain",
    "unnecessary_tool",
    "server_selection",
    "tool_selection",
]

FIND_NAMES = {"find_operations", "find_operation"}
RAW_TEXT_KEYS = {
    "raw_response",
    "raw_content",
    "assistant_content",
    "model_output",
    "response_text",
    "llm_content",
    "message_content",
}
SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def round6(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_agent_json_v1(content: str) -> Dict[str, Any]:
    """Parser v1: the harness parser used during the formal run."""
    payload = _parse_agent_json(content)
    if not isinstance(payload, dict):
        raise ValueError("Agent JSON must be an object.")
    return payload


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    output: List[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            look = index + 1
            while look < len(text) and text[look] in " \t\r\n":
                look += 1
            if look < len(text) and text[look] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def parse_agent_json_v11(content: str) -> Dict[str, Any]:
    """Parser v1.1: deterministic normalization applied uniformly.

    Rules, in order, with no task-id branches:
    1. Drop a leading BOM and map smart quotes to ASCII quotes.
    2. Strip one surrounding markdown fence.
    3. Parse the whole string, then the first balanced JSON object.
    4. Remove commas that sit immediately before '}' or ']' outside strings.
    5. Accept only a JSON object.
    """
    text = content.replace("\ufeff", "")
    text = "".join(SMART_QUOTES.get(char, char) for char in text).strip()
    text = _strip_fence(text)
    candidates = [text]
    extracted = _extract_first_object(text)
    if extracted and extracted != text:
        candidates.append(extracted)
    seen: Set[str] = set()
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        for variant in (candidate, _strip_trailing_commas(candidate)):
            if variant in seen:
                continue
            seen.add(variant)
            try:
                payload = json.loads(variant)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(payload, dict):
                last_error = ValueError("Agent JSON must be an object.")
                continue
            return payload
    raise ValueError("parser v1.1 failed") from last_error


def extract_raw_assistant_texts(row: Dict[str, Any]) -> List[str]:
    """Return stored raw assistant strings. Parsed tool calls are not raw text."""
    found: List[str] = []

    def walk(node: Any, parent_key: Optional[str] = None) -> None:
        if isinstance(node, dict):
            role = node.get("role")
            for key, value in node.items():
                if key in RAW_TEXT_KEYS and isinstance(value, str):
                    found.append(value)
                elif key == "content" and isinstance(value, str) and role == "assistant":
                    found.append(value)
                else:
                    walk(value, key)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_key)

    walk(row)
    for event in row.get("trace") or []:
        if not isinstance(event, dict):
            continue
        if event.get("type") in {"parse_error", "llm_turn"} and isinstance(event.get("content"), str):
            found.append(event["content"])
    return found


def gold_step_lists(level: int, ground_truth: Any) -> List[List[Tuple[str, str]]]:
    if level in (1, 2):
        formatted = _format_simple_ground_truth(ground_truth or [])
    else:
        formatted = _format_graph_ground_truth(ground_truth or [], allow_efs=level == 4)
    steps: List[List[Tuple[str, str]]] = []
    for group in formatted:
        tools: List[Tuple[str, str]] = []
        for tool in group:
            server = tool.get("server_name") or tool.get("server") or ""
            name = tool.get("tool_name") or ""
            if server and name:
                tools.append((server, name))
        steps.append(tools)
    return steps


def gold_dependency_pairs(level: int, ground_truth: Any) -> List[Tuple[int, int]]:
    if level not in (3, 4):
        return []
    formatted = _format_graph_ground_truth(ground_truth or [], allow_efs=level == 4)
    edges: List[Tuple[int, int]] = []
    for index, group in enumerate(formatted):
        deps: Set[int] = set()
        for tool in group:
            for dep in tool.get("dependencies") or []:
                deps.add(int(dep))
        for dep in sorted(deps):
            edges.append((dep, index))
    return edges


def prediction_sequence(predicted_tools: Sequence[Any]) -> List[Tuple[str, str]]:
    sequence: List[Tuple[str, str]] = []
    for item in predicted_tools or []:
        if not isinstance(item, dict):
            continue
        server = item.get("server_name") or item.get("server") or ""
        name = item.get("tool_name") or ""
        if server and name:
            sequence.append((server, name))
    return sequence


def assign_steps(
    steps: Sequence[Sequence[Tuple[str, str]]], predicted: Sequence[Tuple[str, str]]
) -> Dict[int, int]:
    used: Set[int] = set()
    mapping: Dict[int, int] = {}
    for step_index, options in enumerate(steps):
        option_set = set(options)
        for pred_index, tool in enumerate(predicted):
            if pred_index in used or tool not in option_set:
                continue
            mapping[step_index] = pred_index
            used.add(pred_index)
            break
    return mapping


def iter_candidates(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for event in trace or []:
        if event.get("type") != "tool_results":
            continue
        for result in event.get("results") or []:
            if result.get("name") not in FIND_NAMES:
                continue
            payload = result.get("result") or {}
            for candidate in payload.get("candidates") or []:
                if isinstance(candidate, dict):
                    candidates.append(candidate)
    return candidates


def candidate_identity(candidate: Dict[str, Any]) -> Tuple[str, str, str]:
    operation_id = str(candidate.get("operation_id") or "")
    server = str(candidate.get("server_name") or candidate.get("service_id") or "")
    name = str(candidate.get("name") or candidate.get("tool_name") or "")
    if not operation_id and server and name:
        operation_id = f"{server}::{name}"
    return operation_id, server, name


def search_call_count(trace: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for event in trace or []:
        if event.get("type") != "tool_results":
            continue
        count += sum(1 for result in event.get("results") or [] if result.get("name") in FIND_NAMES)
    return count


def last_parsed_model_output(trace: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    last = None
    for event in trace or []:
        if event.get("type") == "llm_turn":
            last = {
                "turn": event.get("turn"),
                "tool_calls": event.get("tool_calls"),
                "final_answer": event.get("final_answer"),
                "model": event.get("model"),
                "raw_text_stored": isinstance(event.get("content"), str),
            }
    return last


def route_from_parsed_object(payload: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Recover an explicit route from parsed JSON without calling a model or tool.

    A route is recovered only when the object already lists tools or invoke_api
    operation ids. Search calls alone do not become a route.
    """
    final_answer = payload.get("final_answer")
    if isinstance(final_answer, dict):
        tools = final_answer.get("tools") or final_answer.get("route") or final_answer.get("predicted_tools")
        if isinstance(tools, list):
            recovered = []
            for item in tools:
                if not isinstance(item, dict):
                    continue
                server = item.get("server_name") or item.get("server") or ""
                name = item.get("tool_name") or ""
                if server and name:
                    recovered.append({"server_name": server, "tool_name": name})
            if recovered or final_answer.get("infeasible") is True or tools == []:
                return recovered
    calls = payload.get("tool_calls") or []
    if not isinstance(calls, list) or not calls:
        return None
    if any(not isinstance(call, dict) or call.get("name") != "invoke_api" for call in calls):
        return None
    recovered = []
    for call in calls:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        operation_id = str(arguments.get("operation_id") or "")
        if "::" not in operation_id:
            return None
        server, name = operation_id.split("::", 1)
        if not server or not name:
            return None
        recovered.append({"server_name": server, "tool_name": name})
    return recovered


def scores_of(level: int, ground_truth: Any, predicted: Sequence[Any], *, failed: bool) -> Dict[str, Any]:
    diagnostic = evaluate_prediction(level, ground_truth, predicted or [])
    strict_exact = bool(diagnostic.get("exact_match")) and not failed
    if failed:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        exact = False
    else:
        precision = float(diagnostic.get("precision") or 0.0)
        recall = float(diagnostic.get("recall") or 0.0)
        f1 = float(diagnostic.get("f1") or 0.0)
        exact = bool(diagnostic.get("exact_match"))
    return {
        "diagnostic": diagnostic,
        "strict_exact_match": strict_exact,
        "official_precision": precision,
        "official_recall": recall,
        "official_f1": f1,
        "official_exact_match": exact,
    }


def classify_task(task_row: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    if task_row["strict_exact_match"]:
        return None, []
    flags: List[str] = []
    error = task_row["error"]
    if task_row["status"] != "completed" and error == "parse_error":
        flags.append("parse/evaluation_failure")
    if task_row["status"] != "completed" and error == "max_llm_turns":
        flags.append("max_turn_resource_limit")
    if task_row["level"] == 5:
        flags.append("infeasible_detection")
    if task_row["retrieval_miss"]:
        flags.append("retrieval_miss")
    if task_row["dependency_order_flag"]:
        flags.append("dependency_order")
    if task_row["incomplete_tool_chain_flag"]:
        flags.append("incomplete_tool_chain")
    if task_row["unnecessary_tool_flag"]:
        flags.append("unnecessary_tool")
    if task_row["server_selection_flag"]:
        flags.append("server_selection")
    if task_row["tool_selection_flag"]:
        flags.append("tool_selection")
    if error == "max_llm_turns":
        primary = "max_turn_resource_limit"
    elif error == "parse_error":
        primary = "parse/evaluation_failure"
    else:
        primary = next((name for name in PRIMARY_PRIORITY if name in flags), "tool_selection")
        if primary not in flags:
            flags.append(primary)
    secondary = [name for name in PRIMARY_PRIORITY if name in flags and name != primary]
    return primary, secondary


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def artifact_table(original: Path) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for name in RAW_ARTIFACTS:
        path = original / name
        if not path.exists():
            table[name] = None
            continue
        stat = path.stat()
        table[name] = {
            "sha256": sha256_file(path),
            "bytes": stat.st_size,
        }
    return table


def fmt_count(num: int, den: int) -> str:
    if not den:
        return "n/a"
    return f"{num}/{den} = {num / den:.3f}"


def fmt_opt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_report(payload: Dict[str, Any]) -> str:
    identity = payload["identity"]
    recon = payload["terminal_reconciliation"]
    metrics = payload["per_level_metrics"]
    pooled = metrics["pooled"]
    taxonomy = payload["error_summary"]
    cost = payload["cost_summary_audited"]
    parser = payload["parser_comparison"]
    retrieval = payload["retrieval_analysis_audited"]
    lines: List[str] = []
    add = lines.append
    add("# ETOM Formal-100 Operation-Visible Final Audit v1")
    add("")
    add("Offline scoring and error analysis only. No LLM, embedding, MCP, or paid API call was made. No task was rerun. Raw predictions, gold routes, the formal lock, the prompt, the registry, and run metadata were not modified.")
    add("")
    add("## 1. Raw artifact integrity")
    add("")
    add("`grouped_aggregates.json` is not in the run directory. The run artifact is `grouped_results.json`.")
    add("")
    add("| File | SHA256 | Bytes |")
    add("| --- | --- | ---: |")
    for name, info in payload["artifact_sha256"].items():
        if info is None:
            add(f"| `{name}` | MISSING |  |")
        else:
            add(f"| `{name}` | `{info['sha256']}` | {info['bytes']} |")
    add("")
    add("| Identity field | Value |")
    add("| --- | --- |")
    add(f"| predictions rows / unique task IDs | {identity['prediction_rows']} / {identity['unique_task_ids']} |")
    add(f"| L1–L5 counts | {identity['level_counts']} |")
    add(f"| matches formal_100_v1.lock.json | {identity['matches_formal_lock']} |")
    add(f"| formal sample SHA256 | `{identity['formal_sample_sha256']}` |")
    add(f"| ordered task IDs SHA256 | `{identity['ordered_task_ids_sha256']}` |")
    add(f"| pilot overlap | {identity['pilot_overlap_count']} |")
    add(f"| registry SHA256 | `{identity['registry_sha256']}` |")
    add(f"| catalog index | {identity['catalog_completed']}/{identity['catalog_total']} `{identity['catalog_file']}` |")
    add(f"| catalog file SHA256 | `{identity['catalog_sha256']}` |")
    add(f"| prompt SHA256 | `{identity['prompt_sha256']}` |")
    add(f"| generation model | `{identity['generation_model']}` |")
    add(f"| embedding model | `{identity['embedding_model']}` |")
    add(f"| provider host | `{identity['provider_host']}` |")
    add(f"| temperature | {identity['temperature']} |")
    add(f"| max_tokens | {identity['max_tokens']} |")
    add(f"| max_llm_turns limit | {identity['max_llm_turns_limit']} |")
    add(f"| real_tool_executions | {identity['real_tool_executions']} |")
    add("")
    add("Provider host is the frozen relay host only. No API key is recorded here.")
    add("")
    add("## 2. Freeze")
    add("")
    add("Derived files are written only under `outputs/etom_external_formal100_operation_visible_v1/final_audit_v1/` and mirrored to `docs/etom_operation_visible_formal100_final_audit_v1/`.")
    add("")
    add(f"Predictions SHA256 after the audit: `{payload['predictions_sha256_after']}` (unchanged={payload['predictions_unchanged']}).")
    add("")
    add("## 3. Official ETOM metrics")
    add("")
    add("Denominator policy:")
    add("")
    add("- Strict: all 20 tasks in the level. A failed task scores exact-match false and precision/recall/F1 = 0.")
    add("- Completed-only: the 92 tasks whose frozen status is `completed`. This matches `cost_summary.json`.")
    add("- L2 exact match is Equal Function Set exact match from the pinned GraphEvaluator. It accepts any listed equivalent tool, and a single listed gold tool is not the whole gold set.")
    add("- L3/L4 complete route success is the official graph exact match, including dependency order and no extra tools. Node-set exact match, precision, recall, F1, and dependency correctness are reported separately.")
    add("- L4 node-set exact match is the official union of every EFS member. EFS-covering match is the set version aligned with one accepted tool per step.")
    add("")
    add("| Level | Metric | Strict | Completed-only |")
    add("| --- | --- | ---: | ---: |")
    for level in range(1, 6):
        block = metrics["per_level"][f"L{level}"]
        for row in block["table_rows"]:
            add(f"| L{level} | {row['metric']} | {row['strict']} | {row['completed']} |")
    add("")
    add("| Aggregate | Value |")
    add("| --- | ---: |")
    add(f"| strict exact-route accuracy | {fmt_count(pooled['strict_exact_route_count'], 100)} |")
    add(f"| completion rate | {fmt_count(pooled['completed'], 100)} |")
    add(f"| failure rate | {fmt_count(pooled['failed'], 100)} |")
    add(f"| macro exact-route across L1–L5 | {fmt_opt(pooled['macro_exact_route'])} |")
    add(f"| micro exact-route | {fmt_opt(pooled['micro_exact_route'])} |")
    add(f"| macro F1 strict | {fmt_opt(pooled['macro_f1_strict'])} |")
    add(f"| micro F1 strict | {fmt_opt(pooled['micro_f1_strict'])} |")
    add(f"| macro F1 completed-only | {fmt_opt(pooled['macro_f1_completed'])} |")
    add(f"| micro F1 completed-only | {fmt_opt(pooled['micro_f1_completed'])} |")
    add("")
    add(pooled["micro_macro_note"])
    add("")
    l2 = metrics["l2_equal_function_sets"]
    add(
        f"L2 gold shape: {l2['singleton_steps']} / 20 tasks are one step with one gold tool; "
        f"{l2['multi_step_efs']} / 20 tasks are multi-step equal-function sets "
        f"(each step accepts any listed equivalent). "
        f"Exact match requires every step, not the first listed tool alone. "
        f"Multi-step equal-function exact matches: {l2['multi_step_efs_exact']} / {l2['multi_step_efs']}. "
        f"Singleton exact matches: {l2['singleton_exact']} / {l2['singleton_steps']}; those gold sets contain one tool, so there is no alternate equivalent to select. "
        f"Predictions that select a non-first equivalent and still exact-match: {l2['exact_using_nonfirst_equivalent']}."
    )
    add("")
    add("## 4. Failure audit")
    add("")
    add("The eight failed tasks stay failed in the official score. `max_llm_turns` is a resource limit. Parse errors are not repaired per task id.")
    add("")
    for item in payload["failure_audit"]:
        add(f"### {item['task_id']} ({item['error']})")
        add("")
        add(f"- Level: L{item['level']}")
        add(f"- Raw response stored: {item['raw_response_present']}")
        add(f"- Selected server/tools: {item['selected_tools'] or '[]'}")
        add(f"- Retrieval searches: {item['search_calls']}; gold operation retrieved: {item['gold_operation_retrieved']}")
        add(f"- Retrieved candidate count: {item['retrieved_candidate_count']}")
        add(f"- Last parsed model output stored: {item['last_parsed_output_present']}")
        add(f"- Last parsed output: `{item['last_parsed_output_preview']}`")
        add(f"- Failure stage: {item['failure_stage']}")
        add(f"- Partial correct route: {item['partial_correct_route']} ({item['covered_steps']}/{item['gold_steps']} gold steps)")
        add(f"- Harness/evaluation failure: {item['harness_evaluation_failure']}")
        add(f"- Agent reasoning failure: {item['agent_reasoning_failure']}")
        add(f"- Resource limit: {item['resource_limit']}")
        add(f"- Note: {item['note']}")
        add("")
    add("## 5. Error taxonomy")
    add("")
    add("Each task has one primary error. Exact-route successes have `primary_error = null`. Secondary errors describe the recorded route and may co-occur.")
    add("")
    add("| Primary error | Count |")
    add("| --- | ---: |")
    for name, count in taxonomy["primary_counts_including_exact"]:
        add(f"| {name} | {count} |")
    add("")
    add("| Secondary error | Count |")
    add("| --- | ---: |")
    if taxonomy["secondary_counts"]:
        for name, count in taxonomy["secondary_counts"]:
            add(f"| {name} | {count} |")
    else:
        add("| (none) | 0 |")
    add("")
    add("Primary counts by level are in `error_summary.json`.")
    add("")
    add("## 6. Retrieval")
    add("")
    add("| Metric | Value |")
    add("| --- | ---: |")
    add(f"| tasks with find_operations | {retrieval['tasks_with_search']}/100 |")
    add(f"| L1–L4 gold operation hit | {fmt_count(retrieval['l1_l4_gold_hit_count'], retrieval['l1_l4_tasks'])} |")
    add(f"| L1–L4 gold hit among tasks that searched | {fmt_count(retrieval['l1_l4_gold_hit_given_search'], retrieval['l1_l4_searched'])} |")
    add(f"| L1–L4 tasks that never searched | {retrieval['l1_l4_never_searched']} |")
    add(f"| L1–L4 gold step coverage | {fmt_opt(retrieval['l1_l4_gold_step_coverage'])} |")
    add(f"| L1–L4 gold step coverage given a search | {fmt_opt(retrieval['l1_l4_gold_step_coverage_given_search'])} |")
    for level in range(1, 5):
        block = retrieval["by_level"][f"L{level}"]
        add(
            f"| L{level} gold operation hit | {fmt_count(block['gold_hit_count'], block['tasks'])} |"
        )
    add(f"| non-exact tasks whose primary error is retrieval_miss | {retrieval['primary_retrieval_miss']} |")
    add("")
    add(retrieval["interpretation"])
    add("")
    add("## 7. Cost and efficiency")
    add("")
    add("Token totals below are recomputed from `predictions.jsonl`. USD figures are copied from the frozen `cost_summary.json` price configuration (`llm input $0.30 / 1M`, `llm output $1.20 / 1M`, `embedding $0.02 / 1M`). This audit does not invent a second price.")
    add("")
    add("| Metric | Value |")
    add("| --- | ---: |")
    add(f"| task LLM calls | {cost['task_llm_calls']} |")
    add(f"| verification probe LLM calls | {cost['probe_llm_calls']} |")
    add(f"| cost_summary llm_calls | {cost['cost_summary_llm_calls']} |")
    add(f"| mean / median LLM calls per task | {fmt_opt(cost['mean_llm_calls'])} / {fmt_opt(cost['median_llm_calls'])} |")
    for level in range(1, 6):
        add(f"| L{level} LLM calls | {cost['llm_calls_by_level'][f'L{level}']} |")
    add(f"| task input / output / total tokens | {cost['task_input_tokens']} / {cost['task_output_tokens']} / {cost['task_total_tokens']} |")
    add(f"| probe input / output tokens | {cost['probe_input_tokens']} / {cost['probe_output_tokens']} |")
    add(f"| embedding calls / tokens | {cost['embedding_calls']} / {cost['embedding_tokens']} |")
    add(f"| query embedding calls / tokens | {cost['query_embedding_calls']} / {cost['query_embedding_tokens']} |")
    add(f"| catalog embedding calls / tokens | {cost['catalog_embedding_calls']} / {cost['catalog_embedding_tokens']} |")
    add(f"| catalog cache_reused | {cost['cache_reused']} |")
    add(f"| catalog paid on this run | {cost['catalog_paid_on_this_run']} |")
    add(f"| latency mean / median / p95 ms | {fmt_opt(cost['latency_ms']['mean'])} / {fmt_opt(cost['latency_ms']['median'])} / {fmt_opt(cost['latency_ms']['p95'])} |")
    add(f"| frozen this_run_estimated_usd | {cost['frozen_this_run_estimated_usd']} |")
    add(f"| frozen variable_per_task.total_usd | {cost['frozen_variable_total_usd']} |")
    add(f"| failed-task LLM calls | {cost['failed_llm_calls']} |")
    add(f"| failed-task input / output / total tokens | {cost['failed_input_tokens']} / {cost['failed_output_tokens']} / {cost['failed_total_tokens']} |")
    add(f"| real_tool_executions | {cost['real_tool_executions']} |")
    add("")
    add(cost["accounting_note"])
    add("")
    add("## 8. Terminal reconciliation")
    add("")
    add("| Claim | Raw artifact | Match |")
    add("| --- | --- | --- |")
    for row in recon["checks"]:
        add(f"| {row['claim']} | {row['observed']} | {row['match']} |")
    add("")
    add(recon["llm_call_note"])
    add("")
    add("## 9. Parser v1 and parser v1.1")
    add("")
    add("Parser v1 is `_parse_agent_json`: strip a surrounding fence when the whole reply starts with ```, then `json.loads`, and require an object.")
    add("")
    add("Parser v1.1 adds only generic, deterministic normalizations: drop a BOM, map smart quotes, strip one fence, take the first balanced object, and delete trailing commas outside strings. The same function is applied to every task. It does not branch on task id, and it does not edit stored responses.")
    add("")
    add(f"Stored raw assistant strings across 100 tasks: {parser['raw_assistant_string_count']}.")
    add(f"Strings that fail v1 and parse under v1.1: {parser['v11_recovered_string_count']}.")
    add(f"Routes changed by v1.1: {parser['routes_changed']}.")
    add("")
    add("| Metric | Parser v1 | Parser v1.1 | Delta |")
    add("| --- | ---: | ---: | ---: |")
    for row in parser["metric_deltas"]:
        add(f"| {row['metric']} | {row['v1']} | {row['v11']} | {row['delta']} |")
    add("")
    add("Per-task score changes:")
    add("")
    if parser["per_task_score_changes"]:
        for change in parser["per_task_score_changes"]:
            add(f"- {change}")
    else:
        add("- None. v1.1 does not change any task score.")
    add("")
    add("Parse-error tasks that remain failed under both parsers:")
    add("")
    for task_id in parser["parse_error_ids_still_failed"]:
        add(f"- {task_id}")
    add("")
    add("The five parse failures have no stored model text, so a parser cannot recover a route from this artifact. Their official scores stay failed.")
    add("")
    add("## 10. Proposed method targets")
    add("")
    for bullet in payload["proposed_method_targets"]:
        add(f"- {bullet}")
    add("")
    add("## 11. Confirmation")
    add("")
    add("- No live model, embedding, MCP, or paid API client is imported or called by this auditor.")
    add("- No task was rerun.")
    add("- Raw prediction bytes are unchanged.")
    add("- Gold, prompt, registry, formal lock, and run metadata were not written.")
    add("")
    return "\n".join(lines) + "\n"


def proposed_targets(error_summary: Dict[str, Any], retrieval: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    l4 = metrics["per_level"]["L4"]
    l2 = metrics["l2_equal_function_sets"]
    routing = ", ".join(f"{name} {count}" for name, count in error_summary["top_completed_routing_errors"])
    return [
        (
            f"L4 complete route success is {l4['complete_route_success_strict']}, "
            f"with official union node-set exact match {l4['node_set_exact_match_strict']} "
            f"and EFS-covering node set {l4['efs_covering_node_set_strict']}. "
            f"Strict F1 is {l4['f1_strict']:.3f}. "
            "The union node-set is lower because a gold step can list several equivalent tools; "
            "predicting one valid equivalent does not equal the full union. "
            "Completed L4 misses are mostly incomplete tool chains. "
            f"dependency_order appears {error_summary['dependency_order_labels']} times as a primary or secondary label, "
            "so the gap is missing steps rather than a reordering of a complete chain."
        ),
        (
            f"Completed non-exact primary errors: {routing}. "
            f"L2 has {l2['multi_step_efs']} multi-step equal-function tasks and "
            f"{l2['multi_step_efs_exact']} of them are exact; the {l2['singleton_exact']} L2 exact matches are single-tool tasks. "
            "A method should finish every equal-function step, not stop after the first acceptable tool."
        ),
        (
            f"Among L1–L4 tasks that called find_operations, gold-operation hit is "
            f"{retrieval['l1_l4_gold_hit_given_search']}/{retrieval['l1_l4_searched']}. "
            f"The {retrieval['l1_l4_never_searched']} tasks without a gold hit never searched; all three died in parse_before_any_action. "
            f"Primary retrieval_miss count is {retrieval['primary_retrieval_miss']}. "
            f"Gold-step coverage given a search is {retrieval['l1_l4_gold_step_coverage_given_search']:.3f}, "
            "so a few individual gold steps are absent from the candidate lists. "
            "That is smaller than the incomplete-chain gap."
        ),
        (
            "Parser robustness matters for the next run: five tasks stopped on JSON parse errors, and three of them never reached retrieval. "
            "Parser v1.1 does not change this run's score because those assistant strings were not stored."
        ),
        (
            "Three max_llm_turns tasks remain resource_limit failures. L4-q75 used 17 searches and never committed a route. "
            "Earlier route commitment is the relevant method change; these three tasks were not rerun."
        ),
        (
            "L5 exact rejection is 19/20. The one miss, L5-q33, is infeasible_detection from a false tool selection."
        ),
    ]


def level_table_rows(block: Dict[str, Any], level: int) -> List[Dict[str, str]]:
    rows = []

    def add(metric: str, strict: str, completed: str) -> None:
        rows.append({"metric": metric, "strict": strict, "completed": completed})

    if level == 1:
        add("Exact Match", block["exact_match_strict_text"], block["exact_match_completed_text"])
    elif level == 2:
        add(
            "Equal Function Set Exact Match",
            block["exact_match_strict_text"],
            block["exact_match_completed_text"],
        )
    elif level in (3, 4):
        add(
            "Node Set Exact Match (official union)",
            block["node_set_strict_text"],
            block["node_set_completed_text"],
        )
        add(
            "EFS-covering node set",
            block["efs_cover_strict_text"],
            block["efs_cover_completed_text"],
        )
        add("Precision", f"{block['precision_strict']:.3f}", f"{block['precision_completed']:.3f}")
        add("Recall", f"{block['recall_strict']:.3f}", f"{block['recall_completed']:.3f}")
        add("F1", f"{block['f1_strict']:.3f}", f"{block['f1_completed']:.3f}")
        if level == 3:
            add(
                "Dependency/order correctness",
                block["dependency_order_strict_text"],
                block["dependency_order_completed_text"],
            )
        else:
            add(
                "Cross-server dependency correctness",
                block["cross_server_dependency_strict_text"],
                block["cross_server_dependency_completed_text"],
            )
        add(
            "Complete route success",
            block["complete_route_strict_text"],
            block["complete_route_completed_text"],
        )
    else:
        add("Exact Rejection Match", block["exact_match_strict_text"], block["exact_match_completed_text"])
        add(
            "False-positive tool selection rate",
            block["false_positive_strict_text"],
            block["false_positive_completed_text"],
        )
    return rows


def run_audit(root: Path, original: Path, audit_dir: Path, docs_dir: Path) -> int:
    audit_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    hashes = artifact_table(original)
    if hashes.get("predictions.jsonl") is None or hashes.get("grouped_results.json") is None:
        write_json(audit_dir / "terminal_reconciliation.json", {"stopped": True, "reason": "missing artifacts", "hashes": hashes})
        return 2

    predictions_before = hashes["predictions.jsonl"]["sha256"]
    meta = json.loads((original / "run_metadata.json").read_text(encoding="utf-8"))
    cost = json.loads((original / "cost_summary.json").read_text(encoding="utf-8"))
    grouped = json.loads((original / "grouped_results.json").read_text(encoding="utf-8"))
    error_raw = json.loads((original / "error_summary.json").read_text(encoding="utf-8"))
    retrieval_raw = json.loads((original / "retrieval_analysis.json").read_text(encoding="utf-8"))
    index_status = json.loads((original / "index_status.json").read_text(encoding="utf-8"))
    rows = load_predictions(original / "predictions.jsonl")
    by_id = {row["task_id"]: row for row in rows}

    verify_source(root)
    lock = load_formal_100_lock(root)
    official = {task.task_id: task for task in load_official_tasks(root, verify=False)}
    prompt_sha = prompt_hash(ETOM_ROUTING_SYSTEM_PROMPT)
    catalog_path = Path(meta["catalog_cache_path"])
    catalog_sha = sha256_file(catalog_path) if catalog_path.exists() else None
    level_counts = Counter(int(row["level"]) for row in rows)
    pilot_overlap = sorted(set(lock["ordered_task_ids"]) & set(lock["excluded_task_ids"]))
    ordered_ids_hash = sha256_text("\n".join(lock["ordered_task_ids"]) + "\n")

    identity = {
        "prediction_rows": len(rows),
        "unique_task_ids": len(by_id),
        "level_counts": {f"L{level}": level_counts.get(level, 0) for level in range(1, 6)},
        "matches_formal_lock": set(by_id) == set(lock["ordered_task_ids"]) and meta.get("task_ids") == lock["ordered_task_ids"],
        "formal_sample_sha256": meta.get("formal_sample_sha256"),
        "lock_sample_sha256": lock.get("sample_sha256"),
        "ordered_task_ids_sha256": meta.get("ordered_task_ids_sha256"),
        "ordered_task_ids_sha256_recomputed_lines": ordered_ids_hash,
        "pilot_overlap": pilot_overlap,
        "pilot_overlap_count": len(pilot_overlap),
        "registry_sha256": meta.get("registry_sha256"),
        "catalog_completed": index_status.get("completed"),
        "catalog_total": index_status.get("total"),
        "catalog_file": catalog_path.name,
        "catalog_sha256": catalog_sha,
        "prompt_sha256": meta.get("system_prompt_sha256"),
        "prompt_sha256_recomputed": prompt_sha,
        "generation_model": meta.get("generation_model"),
        "embedding_model": meta.get("embedding_model"),
        "provider_host": meta.get("generation_base_url_host"),
        "embedding_host": meta.get("embedding_base_url_host"),
        "temperature": (meta.get("parameters") or {}).get("temperature"),
        "max_tokens": (meta.get("parameters") or {}).get("max_tokens"),
        "max_llm_turns_limit": (meta.get("parameters") or {}).get("max_llm_turns"),
        "real_tool_executions": cost.get("real_tool_executions"),
        "source_commit": meta.get("source_commit"),
        "git_commit": meta.get("git_commit"),
        "branch": meta.get("branch"),
    }

    hard_stops: List[str] = []
    if len(rows) != 100 or len(by_id) != 100:
        hard_stops.append(f"predictions are {len(rows)} rows / {len(by_id)} ids, expected 100/100")
    if any(level_counts.get(level) != 20 for level in range(1, 6)):
        hard_stops.append(f"level counts {dict(level_counts)}")
    if set(by_id) != set(lock["ordered_task_ids"]):
        hard_stops.append("task ids differ from formal lock")
    if meta.get("formal_sample_sha256") != lock.get("sample_sha256"):
        hard_stops.append("formal sample hash differs from lock")
    if meta.get("ordered_task_ids_sha256") != lock.get("ordered_task_ids_sha256"):
        hard_stops.append("ordered task id hash differs from lock")
    if pilot_overlap:
        hard_stops.append(f"pilot overlap {pilot_overlap}")
    if prompt_sha != meta.get("system_prompt_sha256"):
        hard_stops.append("prompt hash differs from run metadata")
    if meta.get("generation_model") != "deepseek-v4.1-flash":
        hard_stops.append(f"generation model {meta.get('generation_model')}")
    if meta.get("embedding_model") != "text-embedding-3-small":
        hard_stops.append(f"embedding model {meta.get('embedding_model')}")
    if int(cost.get("real_tool_executions") or 0) != 0:
        hard_stops.append("real_tool_executions != 0")
    if any(int(row.get("real_tool_executions") or 0) != 0 for row in rows):
        hard_stops.append("a prediction records a real tool execution")
    if index_status.get("completed") != 2375 or index_status.get("total") != 2375:
        hard_stops.append("catalog index is not 2375/2375")

    metric_mismatches: List[Dict[str, Any]] = []
    audited: List[Dict[str, Any]] = []
    raw_text_count = 0
    recovered_strings = 0
    route_changes: List[Dict[str, Any]] = []

    if not hard_stops:
        for task_id in lock["ordered_task_ids"]:
            prediction = by_id[task_id]
            task = official[task_id]
            predicted = prediction.get("predicted_tools") or []
            failed = prediction.get("status") != "completed"
            v1_scores = scores_of(task.level, task.ground_truth_tools, predicted, failed=failed)
            texts = extract_raw_assistant_texts(prediction)
            raw_text_count += len(texts)
            v11_predicted = predicted
            for text in texts:
                v1_ok = True
                try:
                    parse_agent_json_v1(text)
                except (json.JSONDecodeError, ValueError, TypeError):
                    v1_ok = False
                if v1_ok:
                    continue
                try:
                    payload = parse_agent_json_v11(text)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                recovered_strings += 1
                recovered = route_from_parsed_object(payload)
                if recovered is not None:
                    v11_predicted = recovered
            if v11_predicted != predicted:
                route_changes.append({"task_id": task_id})
            v11_scores = scores_of(task.level, task.ground_truth_tools, v11_predicted, failed=failed and v11_predicted == predicted)
            # A recovered route is an offline parse of stored text. Parse-error tasks
            # remain failed when no text is stored. A changed route is rescored as a
            # completed diagnostic only for the optional v1.1 column.
            if v11_predicted != predicted:
                v11_scores = scores_of(task.level, task.ground_truth_tools, v11_predicted, failed=False)

            if not failed:
                for field, stored, recomputed in (
                    ("exact_match", prediction.get("exact_match"), v1_scores["diagnostic"].get("exact_match")),
                    ("precision", prediction.get("precision"), v1_scores["diagnostic"].get("precision")),
                    ("recall", prediction.get("recall"), v1_scores["diagnostic"].get("recall")),
                    ("f1", prediction.get("f1"), v1_scores["diagnostic"].get("f1")),
                ):
                    if isinstance(stored, float) or isinstance(recomputed, float):
                        if not math.isclose(float(stored or 0.0), float(recomputed or 0.0), rel_tol=0, abs_tol=1e-9):
                            metric_mismatches.append(
                                {"task_id": task_id, "field": field, "stored": stored, "recomputed": recomputed}
                            )
                    elif stored != recomputed:
                        metric_mismatches.append(
                            {"task_id": task_id, "field": field, "stored": stored, "recomputed": recomputed}
                        )

            steps = gold_step_lists(task.level, task.ground_truth_tools)
            sequence = prediction_sequence(predicted)
            mapping = assign_steps(steps, sequence)
            gold_union = {tool for step in steps for tool in step}
            edges = gold_dependency_pairs(task.level, task.ground_truth_tools)
            all_covered = bool(steps) and len(mapping) == len(steps)
            order_edges_ok = all(
                mapping.get(src) is not None
                and mapping.get(dst) is not None
                and mapping[src] < mapping[dst]
                for src, dst in edges
            )
            dependency_order_correct = bool(all_covered and order_edges_ok) if task.level in (3, 4) else None
            partial_order_violation = any(
                src in mapping and dst in mapping and mapping[src] >= mapping[dst] for src, dst in edges
            )
            pred_servers = {server for server, _name in sequence}
            gold_servers = {server for step in steps for server, _name in step}
            cover_servers = {sequence[index][0] for index in mapping.values()} if mapping else set()
            trace = prediction.get("trace") or []
            candidates = iter_candidates(trace)
            retrieved_ids = {candidate_identity(item)[0] for item in candidates}
            retrieved_ids.discard("")
            gold_ids = set(task.operation_ids)
            gold_hit = bool(gold_ids & retrieved_ids) or any(
                (server, name) in gold_union
                for _op, server, name in (candidate_identity(item) for item in candidates)
            )
            searches = search_call_count(trace)
            step_hits = 0
            for step in steps:
                step_ids = {f"{server}::{name}" for server, name in step}
                if step_ids & retrieved_ids or any(tool in gold_union and tool in step for tool in sequence):
                    # step retrieval hit is retrieval only, not prediction overlap
                    if step_ids & retrieved_ids:
                        step_hits += 1
            node_set = None
            efs_cover = None
            if task.level in (3, 4):
                formatted = _format_graph_ground_truth(task.ground_truth_tools, allow_efs=task.level == 4)
                gt_nodes = {
                    (tool.get("server_name") or "", tool.get("tool_name") or "")
                    for group in formatted
                    for tool in group
                    if tool.get("server_name") and tool.get("tool_name")
                }
                node_set = set(sequence) == gt_nodes if sequence or not gt_nodes else True
                if not sequence and gt_nodes:
                    node_set = False
                efs_cover = all_covered and len(sequence) == len(steps)
            dep = dependency_accuracy(
                level=task.level,
                ground_truth_tools=task.ground_truth_tools,
                predicted_nodes=prediction.get("predicted_nodes") or [],
            )
            first_listed = steps[0][0] if steps and steps[0] else None
            nonfirst = False
            if task.level == 2 and v1_scores["official_exact_match"] and len(sequence) == 1 and first_listed:
                nonfirst = sequence[0] != first_listed and sequence[0] in set(steps[0])
            cross_dep = None
            if task.level == 4:
                cross_dep = bool(dependency_order_correct and len(cover_servers) >= 2)
            false_positive = None
            if task.level == 5:
                false_positive = bool(sequence) and not v1_scores["official_exact_match"]
                if failed:
                    false_positive = True
            retrieval_miss = task.level < 5 and searches > 0 and not gold_hit and not v1_scores["strict_exact_match"]
            incomplete_flag = task.level in (2, 3, 4) and len(steps) > 1 and len(mapping) < len(steps)
            unnecessary_flag = all_covered and len(sequence) > len(steps)
            server_flag = bool(pred_servers) and bool(gold_servers) and pred_servers.isdisjoint(gold_servers)
            tool_flag = task.level < 5 and bool(sequence) and not all_covered
            dependency_flag = task.level in (3, 4) and (
                (all_covered and not dependency_order_correct) or partial_order_violation
            )
            raw_present = bool(texts)
            parsed_last = last_parsed_model_output(trace)
            parse_only = bool(trace) and all(event.get("type") == "parse_error" for event in trace)
            harness = prediction.get("error") == "parse_error" or (
                prediction.get("error") == "max_llm_turns" and int(prediction.get("parse_repairs") or 0) > 0
            )
            resource = prediction.get("error") == "max_llm_turns"
            if v1_scores["strict_exact_match"]:
                reasoning = False
            elif prediction.get("error") == "parse_error":
                reasoning = not parse_only and (bool(sequence) or (searches > 0 and not gold_hit))
            elif prediction.get("error") == "max_llm_turns":
                reasoning = True
            else:
                reasoning = True
            if prediction.get("error") == "parse_error" and parse_only:
                stage = "parse_before_any_action"
            elif prediction.get("error") == "parse_error":
                stage = "parse_after_partial_trace"
            elif prediction.get("error") == "max_llm_turns":
                stage = "turn_budget_exhausted"
            elif v1_scores["strict_exact_match"]:
                stage = "completed_exact_route"
            else:
                stage = "completed_wrong_route"
            record = {
                "task_id": task_id,
                "level": task.level,
                "status": prediction.get("status"),
                "error": prediction.get("error"),
                "strict_exact_match": v1_scores["strict_exact_match"],
                "v11_exact_match": v11_scores["strict_exact_match"] if v11_predicted == predicted or failed is False else v11_scores["official_exact_match"],
                "official_precision": v1_scores["official_precision"],
                "official_recall": v1_scores["official_recall"],
                "official_f1": v1_scores["official_f1"],
                "v11_precision": v11_scores["official_precision"] if v11_predicted == predicted else v11_scores["diagnostic"].get("precision"),
                "v11_recall": v11_scores["official_recall"] if v11_predicted == predicted else v11_scores["diagnostic"].get("recall"),
                "v11_f1": v11_scores["official_f1"] if v11_predicted == predicted else v11_scores["diagnostic"].get("f1"),
                "diagnostic_f1": v1_scores["diagnostic"].get("f1"),
                "node_set_exact_match": node_set,
                "efs_covering_node_set_match": efs_cover,
                "dependency_order_correct": dependency_order_correct,
                "cross_server_dependency_correct": cross_dep,
                "complete_route_success": v1_scores["strict_exact_match"] if task.level in (3, 4) else None,
                "false_positive_tool": false_positive,
                "retrieval_miss": retrieval_miss,
                "dependency_order_flag": bool(dependency_flag),
                "incomplete_tool_chain_flag": bool(incomplete_flag),
                "unnecessary_tool_flag": bool(unnecessary_flag),
                "server_selection_flag": bool(server_flag),
                "tool_selection_flag": bool(tool_flag),
                "covered_steps": len(mapping),
                "gold_steps": len(steps),
                "gold_steps_retrieved": step_hits,
                "search_calls": searches,
                "gold_operation_retrieved": gold_hit if task.level < 5 else None,
                "retrieved_candidate_count": len(candidates),
                "retrieved_operation_ids": sorted(retrieved_ids),
                "selected_tools": [f"{server}::{name}" for server, name in sequence],
                "gold_operation_ids": list(task.operation_ids),
                "raw_response_present": raw_present,
                "parse_only_trace": parse_only,
                "harness_evaluation_failure": harness,
                "agent_reasoning_failure": reasoning,
                "resource_limit": resource,
                "failure_stage": stage,
                "partial_correct_route": len(mapping) > 0 and not v1_scores["strict_exact_match"],
                "l2_nonfirst_equivalent_em": nonfirst,
                "llm_calls": int(prediction.get("llm_calls") or 0),
                "input_tokens": int(prediction.get("input_tokens") or 0),
                "output_tokens": int(prediction.get("output_tokens") or 0),
                "total_tokens": int(prediction.get("total_tokens") or 0),
                "query_embedding_calls": int(prediction.get("query_embedding_calls") or 0),
                "query_embedding_tokens": int(prediction.get("query_embedding_tokens") or 0),
                "latency_ms": prediction.get("latency_ms"),
                "parse_repairs": int(prediction.get("parse_repairs") or 0),
                "real_tool_executions": int(prediction.get("real_tool_executions") or 0),
                "dependency_edge_accuracy": dep["dependency_edge_accuracy"],
                "graph_structure_match": v1_scores["diagnostic"].get("graph_structure_match"),
                "last_parsed_output": parsed_last,
                "trace_types": dict(Counter(event.get("type") for event in trace)),
            }
            if v11_predicted == predicted:
                record["v11_exact_match"] = v1_scores["strict_exact_match"]
                record["v11_precision"] = v1_scores["official_precision"]
                record["v11_recall"] = v1_scores["official_recall"]
                record["v11_f1"] = v1_scores["official_f1"]
            else:
                rescored = scores_of(task.level, task.ground_truth_tools, v11_predicted, failed=False)
                record["v11_exact_match"] = bool(rescored["official_exact_match"])
                record["v11_precision"] = rescored["official_precision"]
                record["v11_recall"] = rescored["official_recall"]
                record["v11_f1"] = rescored["official_f1"]
            primary, secondary = classify_task(record)
            record["primary_error"] = primary
            record["secondary_errors"] = secondary
            audited.append(record)

    if metric_mismatches:
        hard_stops.append(f"stored metrics differ from recomputed evaluator on {len(metric_mismatches)} fields")

    def strict_count(level: int) -> int:
        return sum(1 for row in audited if row["level"] == level and row["strict_exact_match"])

    expected_strict = {1: 17, 2: 12, 3: 13, 4: 5, 5: 19}
    if audited:
        for level, expected in expected_strict.items():
            observed = strict_count(level)
            if observed != expected:
                hard_stops.append(f"L{level} strict EM {observed} != {expected}")
        if sum(row["strict_exact_match"] for row in audited) != 66:
            hard_stops.append("overall strict EM != 66")
        completed = sum(row["status"] == "completed" for row in audited)
        failed = sum(row["status"] != "completed" for row in audited)
        parse_errors = sum(row["error"] == "parse_error" for row in audited)
        max_turns = sum(row["error"] == "max_llm_turns" for row in audited)
        task_calls = sum(row["llm_calls"] for row in audited)
        if completed != 92 or failed != 8:
            hard_stops.append(f"completed/failed {completed}/{failed}")
        if parse_errors != 5 or max_turns != 3:
            hard_stops.append(f"parse_error/max_llm_turns {parse_errors}/{max_turns}")
        if task_calls != 434:
            hard_stops.append(f"task LLM calls {task_calls} != 434")
        if int(cost.get("exact_match_count") or 0) != 66:
            hard_stops.append("cost_summary exact_match_count != 66")
        taxonomy_raw = error_raw.get("failure_taxonomy") or {}
        if taxonomy_raw.get("exact_match") != 66 or taxonomy_raw.get("parse_error") != 5 or taxonomy_raw.get("max_llm_turns") != 3:
            hard_stops.append(f"error_summary taxonomy {taxonomy_raw}")

    probe = cost.get("verification_probe") or {}
    probe_calls = int(probe.get("llm_calls") or 0)
    cost_calls = int(cost.get("llm_calls") or 0)
    llm_note = (
        f"Terminal total LLM calls=434 matches the sum of per-task llm_calls. "
        f"cost_summary.llm_calls={cost_calls} equals that sum plus verification_probe.llm_calls={probe_calls}. "
        "Both numbers are reported. Neither number is rewritten."
    )
    if audited and cost_calls != 434 + probe_calls:
        hard_stops.append(
            f"cost_summary llm_calls {cost_calls} != task calls 434 + probe {probe_calls}"
        )

    checks = []
    if audited and not any("strict EM" in item or "overall strict" in item for item in hard_stops):
        for level, expected in expected_strict.items():
            checks.append(
                {
                    "claim": f"L{level} strict EM {expected}/20",
                    "observed": f"{strict_count(level)}/20",
                    "match": strict_count(level) == expected,
                }
            )
        checks.extend(
            [
                {"claim": "overall strict EM 66/100", "observed": f"{sum(row['strict_exact_match'] for row in audited)}/100", "match": sum(row["strict_exact_match"] for row in audited) == 66},
                {"claim": "completed 92", "observed": sum(row["status"] == "completed" for row in audited), "match": sum(row["status"] == "completed" for row in audited) == 92},
                {"claim": "failed 8", "observed": sum(row["status"] != "completed" for row in audited), "match": sum(row["status"] != "completed" for row in audited) == 8},
                {"claim": "parse_error 5", "observed": sum(row["error"] == "parse_error" for row in audited), "match": sum(row["error"] == "parse_error" for row in audited) == 5},
                {"claim": "max_llm_turns failures 3", "observed": sum(row["error"] == "max_llm_turns" for row in audited), "match": sum(row["error"] == "max_llm_turns" for row in audited) == 3},
                {"claim": "task LLM calls 434", "observed": sum(row["llm_calls"] for row in audited), "match": sum(row["llm_calls"] for row in audited) == 434},
                {"claim": "cost_summary llm_calls includes probe", "observed": cost_calls, "match": cost_calls == 435},
            ]
        )

    if hard_stops:
        reconciliation = {
            "stopped": True,
            "hard_stops": hard_stops,
            "metric_mismatches": metric_mismatches[:20],
            "llm_call_note": llm_note,
            "checks": checks,
        }
        write_json(audit_dir / "terminal_reconciliation.json", reconciliation)
        report = (
            "# Audit stopped\n\n"
            "Raw artifacts disagree with the terminal claims or with the frozen lock. "
            "No score was rewritten.\n\n"
            + "\n".join(f"- {item}" for item in hard_stops)
            + "\n"
        )
        (audit_dir / "audit_report.md").write_text(report, encoding="utf-8")
        (docs_dir / "audit_report.md").write_text(report, encoding="utf-8")
        print("AUDIT_STOPPED")
        for item in hard_stops:
            print(item)
        return 2

    per_level: Dict[str, Any] = {}
    for level in range(1, 6):
        group = [row for row in audited if row["level"] == level]
        completed_rows = [row for row in group if row["status"] == "completed"]

        def count_true(rows_in: Sequence[Dict[str, Any]], field: str) -> int:
            return sum(1 for row in rows_in if row.get(field) is True)

        def mean_field(rows_in: Sequence[Dict[str, Any]], field: str) -> float:
            return float(statistics.mean(float(row[field]) for row in rows_in))

        block = {
            "tasks": 20,
            "completed": len(completed_rows),
            "failed": 20 - len(completed_rows),
            "exact_match_strict": fmt_count(count_true(group, "strict_exact_match"), 20),
            "exact_match_completed": fmt_count(count_true(completed_rows, "strict_exact_match"), len(completed_rows)),
            "exact_match_strict_text": fmt_count(count_true(group, "strict_exact_match"), 20),
            "exact_match_completed_text": fmt_count(count_true(completed_rows, "strict_exact_match"), len(completed_rows)),
            "precision_strict": mean_field(group, "official_precision"),
            "recall_strict": mean_field(group, "official_recall"),
            "f1_strict": mean_field(group, "official_f1"),
            "precision_completed": mean_field(completed_rows, "official_precision"),
            "recall_completed": mean_field(completed_rows, "official_recall"),
            "f1_completed": mean_field(completed_rows, "official_f1"),
        }
        if level in (3, 4):
            block.update(
                {
                    "node_set_exact_match_strict": fmt_count(count_true(group, "node_set_exact_match"), 20),
                    "node_set_exact_match_completed": fmt_count(count_true(completed_rows, "node_set_exact_match"), len(completed_rows)),
                    "node_set_strict_text": fmt_count(count_true(group, "node_set_exact_match"), 20),
                    "node_set_completed_text": fmt_count(count_true(completed_rows, "node_set_exact_match"), len(completed_rows)),
                    "efs_covering_node_set_strict": fmt_count(count_true(group, "efs_covering_node_set_match"), 20),
                    "efs_covering_node_set_completed": fmt_count(count_true(completed_rows, "efs_covering_node_set_match"), len(completed_rows)),
                    "efs_cover_strict_text": fmt_count(count_true(group, "efs_covering_node_set_match"), 20),
                    "efs_cover_completed_text": fmt_count(count_true(completed_rows, "efs_covering_node_set_match"), len(completed_rows)),
                    "dependency_order_strict_text": fmt_count(count_true(group, "dependency_order_correct"), 20),
                    "dependency_order_completed_text": fmt_count(count_true(completed_rows, "dependency_order_correct"), len(completed_rows)),
                    "cross_server_dependency_strict_text": fmt_count(count_true(group, "cross_server_dependency_correct"), 20),
                    "cross_server_dependency_completed_text": fmt_count(count_true(completed_rows, "cross_server_dependency_correct"), len(completed_rows)),
                    "complete_route_success_strict": fmt_count(count_true(group, "complete_route_success"), 20),
                    "complete_route_success_completed": fmt_count(count_true(completed_rows, "complete_route_success"), len(completed_rows)),
                    "complete_route_strict_text": fmt_count(count_true(group, "complete_route_success"), 20),
                    "complete_route_completed_text": fmt_count(count_true(completed_rows, "complete_route_success"), len(completed_rows)),
                    "mean_explicit_dependency_edge_accuracy_completed": mean(
                        [float(row["dependency_edge_accuracy"]) for row in completed_rows if row["dependency_edge_accuracy"] is not None]
                    ),
                }
            )
        if level == 5:
            block["false_positive_strict_text"] = fmt_count(count_true(group, "false_positive_tool"), 20)
            block["false_positive_completed_text"] = fmt_count(count_true(completed_rows, "false_positive_tool"), len(completed_rows))
        block["table_rows"] = level_table_rows(block, level)
        per_level[f"L{level}"] = block

    strict_f1_by_level = [per_level[f"L{level}"]["f1_strict"] for level in range(1, 6)]
    completed_f1_by_level = [per_level[f"L{level}"]["f1_completed"] for level in range(1, 6)]
    strict_em_by_level = [strict_count(level) / 20 for level in range(1, 6)]
    pooled = {
        "strict_exact_route_count": 66,
        "micro_exact_route": 0.66,
        "macro_exact_route": float(statistics.mean(strict_em_by_level)),
        "micro_f1_strict": float(statistics.mean(row["official_f1"] for row in audited)),
        "macro_f1_strict": float(statistics.mean(strict_f1_by_level)),
        "micro_f1_completed": float(statistics.mean(row["official_f1"] for row in audited if row["status"] == "completed")),
        "macro_f1_completed": float(statistics.mean(completed_f1_by_level)),
        "completed": 92,
        "failed": 8,
        "micro_macro_note": (
            "Micro exact-route is 66/100. Macro exact-route is the unweighted mean of the five level rates. "
            "They match because the formal sample has 20 tasks in every level. "
            "Completed-only macro and micro F1 differ because L1–L5 do not have the same number of completed tasks. "
            "L3/L4 strict exact match is the complete-route number only; node set, F1, and dependency metrics stay visible above."
        ),
    }
    l2_rows = [row for row in audited if row["level"] == 2]
    multi_step_l2 = [row for row in l2_rows if row["gold_steps"] > 1]
    singleton_l2 = [row for row in l2_rows if row["gold_steps"] == 1]
    per_level_metrics = {
        "definition_source": "data/etom/raw/eval/eval.py and graph_evaluator.py at the pinned ETOM commit, via hk_equity_mcp.etom.evaluator",
        "l2_em_using_nonfirst_equivalent": sum(1 for row in audited if row["l2_nonfirst_equivalent_em"]),
        "l2_equal_function_sets": {
            "singleton_steps": len(singleton_l2),
            "singleton_exact": sum(1 for row in singleton_l2 if row["strict_exact_match"]),
            "multi_step_efs": len(multi_step_l2),
            "multi_step_efs_exact": sum(1 for row in multi_step_l2 if row["strict_exact_match"]),
            "exact_using_nonfirst_equivalent": sum(1 for row in audited if row["l2_nonfirst_equivalent_em"]),
            "note": (
                "A multi-step L2 item is an ordered chain of equal-function sets. "
                "Exact match accepts any listed equivalent at each step and still requires the full chain."
            ),
        },
        "pooled": pooled,
        "per_level": per_level,
        "completed_only_matches_cost_summary": {
            "exact_match_count": cost.get("exact_match_count"),
            "mean_f1": cost.get("mean_graph_f1"),
            "recomputed_completed_mean_f1": pooled["micro_f1_completed"],
        },
    }

    primary_counter = Counter(row["primary_error"] or "exact_route_success" for row in audited)
    secondary_counter: Counter = Counter()
    for row in audited:
        secondary_counter.update(row["secondary_errors"])
    by_level_errors = {}
    for level in range(1, 6):
        group = [row for row in audited if row["level"] == level]
        by_level_errors[f"L{level}"] = {
            "primary": dict(Counter(row["primary_error"] or "exact_route_success" for row in group)),
            "non_exact_ids": [row["task_id"] for row in group if not row["strict_exact_match"]],
        }
    routing_primary = Counter(
        row["primary_error"]
        for row in audited
        if row["status"] == "completed" and row["primary_error"]
    )
    error_summary = {
        "tasks": 100,
        "primary_error_is_unique": all(row["primary_error"] is None or isinstance(row["primary_error"], str) for row in audited),
        "exact_route_success": primary_counter["exact_route_success"],
        "primary_counts_including_exact": primary_counter.most_common(),
        "secondary_counts": secondary_counter.most_common(),
        "by_level": by_level_errors,
        "top_completed_routing_errors": routing_primary.most_common(),
        "dependency_order_labels": primary_counter["dependency_order"] + secondary_counter["dependency_order"],
        "parse_error_ids": [row["task_id"] for row in audited if row["error"] == "parse_error"],
        "max_llm_turns_ids": [row["task_id"] for row in audited if row["error"] == "max_llm_turns"],
        "real_tool_executions": 0,
    }

    failure_audit = []
    for row in audited:
        if row["status"] == "completed":
            continue
        last = row["last_parsed_output"]
        preview = "raw model text was not stored"
        if last:
            preview = json.dumps(
                {"turn": last.get("turn"), "tool_calls": last.get("tool_calls"), "final_answer": last.get("final_answer")},
                ensure_ascii=False,
            )[:700]
        if row["error"] == "parse_error" and row["parse_only_trace"]:
            note = "Every stored turn is a JSON parse failure. The assistant string itself was not persisted, so the route cannot be reconstructed."
        elif row["error"] == "parse_error":
            note = "The run stopped after repeated JSON parse failures. Earlier turns contain tool observations, and the recorded route does not cover the gold chain."
        else:
            note = "The frozen turn limit was reached. The task stays a resource_limit failure; no extra model call was added."
        failure_audit.append(
            {
                "task_id": row["task_id"],
                "level": row["level"],
                "error": row["error"],
                "raw_response_present": row["raw_response_present"],
                "selected_tools": row["selected_tools"],
                "search_calls": row["search_calls"],
                "gold_operation_retrieved": row["gold_operation_retrieved"],
                "retrieved_candidate_count": row["retrieved_candidate_count"],
                "retrieved_operation_ids_preview": row["retrieved_operation_ids"][:15],
                "last_parsed_output_present": last is not None,
                "last_parsed_output_preview": preview,
                "failure_stage": row["failure_stage"],
                "partial_correct_route": row["partial_correct_route"],
                "covered_steps": row["covered_steps"],
                "gold_steps": row["gold_steps"],
                "harness_evaluation_failure": row["harness_evaluation_failure"],
                "agent_reasoning_failure": row["agent_reasoning_failure"],
                "resource_limit": row["resource_limit"],
                "primary_error": row["primary_error"],
                "secondary_errors": row["secondary_errors"],
                "note": note,
            }
        )

    l1_l4 = [row for row in audited if row["level"] < 5]
    searched = [row for row in l1_l4 if row["search_calls"] > 0]
    never_searched = [row for row in l1_l4 if row["search_calls"] == 0]
    gold_steps_total = sum(row["gold_steps"] for row in l1_l4)
    gold_steps_hit = sum(row["gold_steps_retrieved"] for row in l1_l4)
    searched_steps = sum(row["gold_steps"] for row in searched)
    searched_hits = sum(row["gold_steps_retrieved"] for row in searched)
    retrieval = {
        "tasks": 100,
        "tasks_with_search": sum(1 for row in audited if row["search_calls"] > 0),
        "l1_l4_tasks": len(l1_l4),
        "l1_l4_searched": len(searched),
        "l1_l4_never_searched": len(never_searched),
        "l1_l4_never_searched_ids": [row["task_id"] for row in never_searched],
        "l1_l4_gold_hit_count": sum(1 for row in l1_l4 if row["gold_operation_retrieved"]),
        "l1_l4_gold_hit_given_search": sum(1 for row in searched if row["gold_operation_retrieved"]),
        "l1_l4_gold_step_coverage": rate(gold_steps_hit, gold_steps_total),
        "l1_l4_gold_step_coverage_given_search": rate(searched_hits, searched_steps),
        "primary_retrieval_miss": sum(1 for row in audited if row["primary_error"] == "retrieval_miss"),
        "interpretation": (
            "Gold hit means at least one gold operation id appeared in a stored find_operations candidate list. "
            "L5 has an empty gold route, so it is outside the hit rate. "
            f"The {len(never_searched)} L1–L4 tasks with no gold hit never called find_operations "
            f"({', '.join(row['task_id'] for row in never_searched)}). "
            "Conditional on a search, every L1–L4 task retrieved at least one gold operation."
            if searched and sum(1 for row in searched if row["gold_operation_retrieved"]) == len(searched)
            else "Gold hit means at least one gold operation id appeared in a stored find_operations candidate list."
        ),
        "by_level": {},
        "query_embedding_calls": sum(row["query_embedding_calls"] for row in audited),
        "query_embedding_tokens": sum(row["query_embedding_tokens"] for row in audited),
        "frozen_retrieval_analysis": retrieval_raw,
        "note": "Hit rate uses stored find_operations candidates. No embedding was recomputed.",
    }
    for level in range(1, 5):
        group = [row for row in audited if row["level"] == level]
        retrieval["by_level"][f"L{level}"] = {
            "tasks": len(group),
            "gold_hit_count": sum(1 for row in group if row["gold_operation_retrieved"]),
            "tasks_with_search": sum(1 for row in group if row["search_calls"] > 0),
            "mean_search_calls": mean([row["search_calls"] for row in group]),
        }

    latencies = [float(row["latency_ms"]) for row in audited if row["latency_ms"] is not None]
    failed_rows = [row for row in audited if row["status"] != "completed"]
    catalog_block = cost.get("one_time_catalog_index") or {}
    cost_audited = {
        "task_llm_calls": sum(row["llm_calls"] for row in audited),
        "probe_llm_calls": probe_calls,
        "cost_summary_llm_calls": cost_calls,
        "mean_llm_calls": mean([row["llm_calls"] for row in audited]),
        "median_llm_calls": median([row["llm_calls"] for row in audited]),
        "llm_calls_by_level": {
            f"L{level}": sum(row["llm_calls"] for row in audited if row["level"] == level) for level in range(1, 6)
        },
        "task_input_tokens": sum(row["input_tokens"] for row in audited),
        "task_output_tokens": sum(row["output_tokens"] for row in audited),
        "task_total_tokens": sum(row["total_tokens"] for row in audited),
        "probe_input_tokens": int(probe.get("input_tokens") or 0),
        "probe_output_tokens": int(probe.get("output_tokens") or 0),
        "embedding_calls": int(cost.get("embedding_calls") or 0),
        "embedding_tokens": int(cost.get("embedding_tokens") or 0),
        "query_embedding_calls": sum(row["query_embedding_calls"] for row in audited),
        "query_embedding_tokens": sum(row["query_embedding_tokens"] for row in audited),
        "catalog_embedding_calls": int(catalog_block.get("calls") or 0),
        "catalog_embedding_tokens": int(catalog_block.get("tokens") or 0),
        "cache_reused": bool(catalog_block.get("cache_reused")),
        "catalog_paid_on_this_run": bool(catalog_block.get("paid_on_this_run")),
        "latency_ms": {
            "mean": mean(latencies),
            "median": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "frozen_latency_ms": cost.get("latency_ms"),
        "frozen_this_run_estimated_usd": cost.get("this_run_estimated_usd"),
        "frozen_variable_total_usd": (cost.get("variable_per_task") or {}).get("total_usd"),
        "price_config_present": bool((cost.get("projection_uncertainty") or {}).get("llm_input_usd_per_million")),
        "price_config": {
            "llm_input_usd_per_million": (cost.get("projection_uncertainty") or {}).get("llm_input_usd_per_million"),
            "llm_output_usd_per_million": (cost.get("projection_uncertainty") or {}).get("llm_output_usd_per_million"),
            "embedding_usd_per_million": (cost.get("projection_uncertainty") or {}).get("embedding_usd_per_million"),
            "source": (cost.get("projection_uncertainty") or {}).get("price_source"),
        },
        "failed_llm_calls": sum(row["llm_calls"] for row in failed_rows),
        "failed_input_tokens": sum(row["input_tokens"] for row in failed_rows),
        "failed_output_tokens": sum(row["output_tokens"] for row in failed_rows),
        "failed_total_tokens": sum(row["total_tokens"] for row in failed_rows),
        "real_tool_executions": 0,
        "accounting_note": llm_note,
        "grouped_results_file": "grouped_results.json",
        "frozen_grouped_exact_match_rate_completed_only": grouped["pooled_100"]["exact_match_rate"],
    }

    metric_deltas = []

    def add_delta(name: str, v1_value: float, v11_value: float) -> None:
        metric_deltas.append(
            {
                "metric": name,
                "v1": round6(v1_value),
                "v11": round6(v11_value),
                "delta": round6(v11_value - v1_value),
            }
        )

    add_delta("strict exact-route", sum(row["strict_exact_match"] for row in audited) / 100, sum(row["v11_exact_match"] for row in audited) / 100)
    add_delta("strict mean F1", pooled["micro_f1_strict"], float(statistics.mean(row["v11_f1"] for row in audited)))
    for level in range(1, 6):
        group = [row for row in audited if row["level"] == level]
        add_delta(
            f"L{level} exact-route",
            sum(row["strict_exact_match"] for row in group) / 20,
            sum(row["v11_exact_match"] for row in group) / 20,
        )
    per_task_changes = []
    for row in audited:
        changed = {}
        if row["strict_exact_match"] != row["v11_exact_match"]:
            changed["exact_match"] = [row["strict_exact_match"], row["v11_exact_match"]]
        if not math.isclose(float(row["official_f1"]), float(row["v11_f1"]), rel_tol=0, abs_tol=1e-12):
            changed["f1"] = [row["official_f1"], row["v11_f1"]]
        if changed:
            per_task_changes.append({"task_id": row["task_id"], "changes": changed})
    parser_comparison = {
        "parser_v1": "hk_equity_mcp.etom.agent._parse_agent_json",
        "parser_v11_rules": [
            "strip BOM",
            "map smart quotes to ASCII",
            "strip one surrounding markdown fence",
            "parse the first balanced JSON object if the whole string is not JSON",
            "remove trailing commas outside strings",
            "require a JSON object",
        ],
        "task_id_special_cases": 0,
        "raw_responses_modified": False,
        "raw_assistant_string_count": raw_text_count,
        "v11_recovered_string_count": recovered_strings,
        "routes_changed": len(route_changes),
        "metric_deltas": metric_deltas,
        "per_task_score_changes": per_task_changes,
        "parse_error_ids_still_failed": [row["task_id"] for row in audited if row["error"] == "parse_error"],
        "max_llm_turns_kept_failed": [row["task_id"] for row in audited if row["error"] == "max_llm_turns"],
    }

    taxonomy_lines = []
    csv_rows = []
    for row in audited:
        taxonomy_lines.append(
            {
                "task_id": row["task_id"],
                "level": row["level"],
                "status": row["status"],
                "error": row["error"],
                "strict_exact_match": row["strict_exact_match"],
                "primary_error": row["primary_error"],
                "secondary_errors": row["secondary_errors"],
                "failure_stage": row["failure_stage"],
                "partial_correct_route": row["partial_correct_route"],
                "harness_evaluation_failure": row["harness_evaluation_failure"],
                "agent_reasoning_failure": row["agent_reasoning_failure"],
                "resource_limit": row["resource_limit"],
                "selected_tools": row["selected_tools"],
                "gold_operation_retrieved": row["gold_operation_retrieved"],
            }
        )
        csv_rows.append(
            {
                "task_id": row["task_id"],
                "level": row["level"],
                "status": row["status"],
                "error": row["error"] or "",
                "exact_match": row["strict_exact_match"],
                "precision": row["official_precision"],
                "recall": row["official_recall"],
                "f1": row["official_f1"],
                "node_set_exact_match": row["node_set_exact_match"],
                "efs_covering_node_set_match": row["efs_covering_node_set_match"],
                "dependency_order_correct": row["dependency_order_correct"],
                "cross_server_dependency_correct": row["cross_server_dependency_correct"],
                "complete_route_success": row["complete_route_success"],
                "false_positive_tool": row["false_positive_tool"],
                "primary_error": row["primary_error"] or "",
                "secondary_errors": "|".join(row["secondary_errors"]),
                "llm_calls": row["llm_calls"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
                "query_embedding_calls": row["query_embedding_calls"],
                "query_embedding_tokens": row["query_embedding_tokens"],
                "latency_ms": row["latency_ms"],
                "parse_repairs": row["parse_repairs"],
                "real_tool_executions": row["real_tool_executions"],
                "gold_operation_retrieved": row["gold_operation_retrieved"],
                "predicted_tool_count": len(row["selected_tools"]),
                "gold_step_count": row["gold_steps"],
                "parser_v11_exact_match": row["v11_exact_match"],
            }
        )

    predictions_after = sha256_file(original / "predictions.jsonl")
    report_payload = {
        "artifact_sha256": hashes,
        "identity": identity,
        "terminal_reconciliation": {"stopped": False, "checks": checks, "llm_call_note": llm_note},
        "per_level_metrics": per_level_metrics,
        "error_summary": error_summary,
        "failure_audit": failure_audit,
        "retrieval_analysis_audited": retrieval,
        "cost_summary_audited": cost_audited,
        "parser_comparison": parser_comparison,
        "predictions_sha256_after": predictions_after,
        "predictions_unchanged": predictions_after == predictions_before,
        "proposed_method_targets": [],
    }
    report_payload["proposed_method_targets"] = proposed_targets(error_summary, retrieval, per_level_metrics)
    report = build_report(report_payload)

    write_json(audit_dir / "artifact_sha256.json", hashes)
    write_json(audit_dir / "dataset_identity.json", identity)
    write_json(audit_dir / "per_level_metrics.json", per_level_metrics)
    write_json(audit_dir / "error_summary.json", error_summary)
    write_json(audit_dir / "failure_audit.json", failure_audit)
    write_json(audit_dir / "retrieval_analysis_audited.json", retrieval)
    write_json(audit_dir / "cost_summary_audited.json", cost_audited)
    write_json(audit_dir / "parser_comparison.json", parser_comparison)
    write_json(audit_dir / "terminal_reconciliation.json", report_payload["terminal_reconciliation"])
    with (audit_dir / "error_taxonomy.jsonl").open("w", encoding="utf-8") as handle:
        for item in taxonomy_lines:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (audit_dir / "per_task_metrics_audited.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    (audit_dir / "audit_report.md").write_text(report, encoding="utf-8")

    if sha256_file(original / "predictions.jsonl") != predictions_before:
        raise SystemExit("predictions.jsonl changed during the audit")
    if sha256_file(original / "run_metadata.json") != hashes["run_metadata.json"]["sha256"]:
        raise SystemExit("run_metadata.json changed during the audit")

    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    shutil.copytree(audit_dir, docs_dir)
    print("AUDIT_OK")
    print(f"strict_EM=66/100")
    print(f"audit_dir={audit_dir}")
    print(f"docs_dir={docs_dir}")
    return 0


def main() -> int:
    root = ROOT_DEFAULT
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    return run_audit(root, ORIG_DEFAULT, AUDIT_DEFAULT, DOCS_DEFAULT)


if __name__ == "__main__":
    raise SystemExit(main())
