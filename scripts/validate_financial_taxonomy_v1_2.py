#!/usr/bin/env python3
"""Semantic validator for Financial taxonomy v1.2 (no LLM / no live MCP)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_2.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_2.jsonl"
DEV_V1 = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
HELD_V11 = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_1.jsonl"

DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_question(text: str, *, scrub_entities: bool = False) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9.\-/% ]+", "", t)
    if scrub_entities:
        for tok in (
            "tencent",
            "hsbc",
            "alibaba",
            "0700",
            "0005",
            "9988",
            "00700",
            "09988",
        ):
            t = t.replace(tok, "ENTITY")
        t = DATE_RE.sub("DATE", t)
    return t


def token_set(text: str) -> Set[str]:
    return {
        w
        for w in normalize_question(text, scrub_entities=True).split()
        if len(w) > 2
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gold_tuple(row: Dict[str, Any]) -> Any:
    return (
        row.get("id"),
        row.get("question"),
        row.get("reference_answer"),
        json.dumps(row.get("gold_values"), sort_keys=True, ensure_ascii=False),
        row.get("gold_status"),
    )


def cross_server_edges(graph: List[Dict[str, str]]) -> int:
    n = 0
    for e in graph or []:
        a = str(e.get("from", "")).split(".", 1)[0]
        b = str(e.get("to", "")).split(".", 1)[0]
        if a and b and a != b:
            n += 1
    return n


def main() -> int:
    errors: List[str] = []
    dev = load_jsonl(DEV)
    held = load_jsonl(HELD)
    if len(dev) != 30 or len(held) != 30:
        errors.append(f"sizes dev={len(dev)} held={len(held)}")

    # gold equality vs prior active files
    for oldp, new_rows, label in (
        (DEV_V1, dev, "dev"),
        (HELD_V11, held, "held"),
    ):
        old = load_jsonl(oldp)
        for a, b in zip(old, new_rows):
            if gold_tuple(a) != gold_tuple(b):
                errors.append(f"{label} gold mismatch {a.get('id')}")

    ids = [r["id"] for r in dev + held]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")

    d_dev = Counter(r["project_difficulty_level"] for r in dev)
    d_held = Counter(r["project_difficulty_level"] for r in held)
    if dict(d_dev) != {"D1": 10, "D2": 6, "D3": 4, "D4": 5, "D5": 5}:
        errors.append(f"dev project difficulty {dict(d_dev)}")
    if dict(d_held) != {"D1": 6, "D2": 6, "D3": 6, "D4": 6, "D5": 6}:
        errors.append(f"held project difficulty {dict(d_held)}")

    for r in held:
        if r.get("dataset_split") not in {"heldout_test", "heldout"}:
            errors.append(f"{r['id']} bad dataset_split={r.get('dataset_split')}")
        if not r.get("locked"):
            errors.append(f"{r['id']} held-out not locked")
        if r.get("prior_file_status") != "superseded_before_any_agent_run":
            errors.append(f"{r['id']} held prior_file_status")
        if r.get("supersession_status") != "active":
            errors.append(f"{r['id']} not active")

    for r in dev:
        if r.get("prior_file_status") != "taxonomy_corrected_after_nonformal_pilot":
            errors.append(f"{r['id']} dev prior_file_status")

    for r in dev + held:
        qid = r["id"]
        et = r.get("etom_level")
        align = r.get("etom_alignment")
        oc = int(r.get("operation_count") or 0)
        sc = int(r.get("server_count") or 0)
        dep = bool(r.get("has_output_input_dependency"))
        graph = r.get("dependency_graph") or []
        tools = r.get("acceptable_tool_sets") or []
        rob = r.get("robustness_subtype")
        ops = r.get("gold_operations") or []

        required = [
            "project_difficulty_level",
            "orchestration_scope",
            "operation_count",
            "server_count",
            "has_output_input_dependency",
            "dependency_graph",
            "acceptable_tool_sets",
            "etom_alignment",
            "etom_alignment_reason",
            "robustness_subtype",
        ]
        for k in required:
            if k not in r:
                errors.append(f"{qid} missing {k}")

        if align == "exact" and et == "L2":
            if not tools:
                errors.append(f"{qid} exact L2 requires acceptable_tool_sets")

        if align == "exact" and et == "L3":
            if sc != 1 or oc < 2 or not dep or len(graph) < 1:
                errors.append(
                    f"{qid} exact L3 requires server_count=1, operation_count>=2, "
                    f"dependency edge>=1 (sc={sc} oc={oc} dep={dep} edges={len(graph)})"
                )
            if cross_server_edges(graph) > 0:
                errors.append(f"{qid} exact L3 cannot have cross-server edges")

        if align == "exact" and et == "L4":
            if sc < 2 or oc < 2 or not dep or cross_server_edges(graph) < 1:
                errors.append(
                    f"{qid} exact L4 requires server_count>=2, operation_count>=2, "
                    f"cross-server dependency edge (sc={sc} oc={oc} dep={dep} "
                    f"xedges={cross_server_edges(graph)})"
                )

        if align == "exact" and et == "L5":
            if rob != "capability_gap":
                errors.append(f"{qid} exact L5 must be capability_gap")
            if ops:
                errors.append(f"{qid} exact L5 must have empty gold_operations")

        # single op + local calc cannot be exact L3/L4
        if align == "exact" and et in {"L3", "L4"} and oc < 2:
            errors.append(f"{qid} single-operation cannot be exact {et}")

        # parallel independent retrieval cannot claim sequential dependency
        if dep and not graph:
            errors.append(f"{qid} has_output_input_dependency true but empty graph")
        if (not dep) and graph:
            errors.append(f"{qid} non-empty graph but has_output_input_dependency false")

    # Exact duplicates keep company/ticker identity (parallel company templates are intentional).
    norms = [(r["id"], normalize_question(r["question"])) for r in dev + held]
    bucket: Dict[str, List[str]] = {}
    for qid, n in norms:
        bucket.setdefault(n, []).append(qid)
    for n, qids in bucket.items():
        if len(qids) > 1:
            errors.append(f"exact normalized duplicate {qids}")
    # Near-dup check only flags near-identical wording for the SAME company across splits.
    for a in dev:
        for b in held:
            if (a.get("company") or "").lower() != (b.get("company") or "").lower():
                continue
            ja = jaccard(token_set(a["question"]), token_set(b["question"]))
            if ja >= 0.92:
                errors.append(f"near-dup {a['id']}~{b['id']}={ja:.2f}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    rows = dev + held
    print("VALIDATION OK")
    print("project_difficulty_dev=", dict(Counter(r["project_difficulty_level"] for r in dev)))
    print("project_difficulty_held=", dict(Counter(r["project_difficulty_level"] for r in held)))
    print(
        "exact_etom=",
        dict(
            Counter(
                r["etom_level"]
                for r in rows
                if r.get("etom_alignment") == "exact" and r.get("etom_level")
            )
        ),
    )
    print("align=", dict(Counter(r["etom_alignment"] for r in rows)))
    print("robustness=", dict(Counter(r["robustness_subtype"] for r in rows)))
    print(
        "true_L3=",
        sorted(
            r["id"]
            for r in rows
            if r.get("etom_level") == "L3" and r.get("etom_alignment") == "exact"
        ),
    )
    print(
        "true_L4=",
        sorted(
            r["id"]
            for r in rows
            if r.get("etom_level") == "L4" and r.get("etom_alignment") == "exact"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
