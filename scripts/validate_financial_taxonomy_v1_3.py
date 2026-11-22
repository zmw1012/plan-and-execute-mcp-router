#!/usr/bin/env python3
"""Schema-aware Financial taxonomy v1.3 validator (no LLM)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_3.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_3.jsonl"
DEV_V12 = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_2.jsonl"
HELD_V12 = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_2.jsonl"
CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
SMOKE_SCHEMAS = (
    ROOT / "docs" / "financial_hkexnews_dependency_smoke_v1" / "tool_input_schemas.json"
)
SMOKE_EDGES = (
    ROOT
    / "docs"
    / "financial_hkexnews_dependency_smoke_v1"
    / "observed_dependency_edges.json"
)

CONCEPTUAL_FIELDS = {
    "announcement_id",
    "document",
    "document_bytes",
    "parsed_text",
    "selected_announcement",
    "hit_locator",
    "year_end_constrains_period_interpretation",
    "issuer_identity_gates_evidence",
}

DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gold_tuple(row: Dict[str, Any]) -> Any:
    return (
        row.get("id"),
        row.get("question"),
        row.get("reference_answer"),
        json.dumps(row.get("gold_values"), sort_keys=True, ensure_ascii=False),
        row.get("gold_status"),
    )


def normalize_question(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"[^a-z0-9.\-/% ]+", "", t)


def token_set(text: str) -> Set[str]:
    t = normalize_question(text)
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
    return {w for w in t.split() if len(w) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def catalog_param_map(catalog: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for op in catalog.get("operations") or []:
        oid = op["operation_id"]
        params = {p["name"] for p in (op.get("parameters") or [])}
        out[oid] = params
    return out


def smoke_param_map(path: Path) -> Dict[str, Set[str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Set[str]] = {}
    for tool, schema in raw.items():
        props = (schema or {}).get("properties") or {}
        out[f"hkexnews.{tool}"] = set(props.keys())
    return out


def server_of(op: str) -> str:
    if op.startswith("hkexnews."):
        return "mcp_hkexnews"
    if op.startswith("aktools."):
        return "aktools"
    return op.split(".", 1)[0]


def cross_server_field_edges(graph: List[Dict[str, Any]]) -> int:
    n = 0
    for e in graph or []:
        if server_of(str(e.get("producer_operation", ""))) != server_of(
            str(e.get("consumer_operation", ""))
        ):
            n += 1
    return n


def main() -> int:
    errors: List[str] = []
    if not SMOKE_EDGES.exists():
        errors.append(f"missing smoke edges evidence: {SMOKE_EDGES}")

    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    cat_params = catalog_param_map(catalog)
    smoke_params = smoke_param_map(SMOKE_SCHEMAS)
    # Prefer live MCP schemas for hkexnews when available.
    params = dict(cat_params)
    params.update(smoke_params)
    active_ops = set(cat_params)

    dev = load_jsonl(DEV)
    held = load_jsonl(HELD)
    if len(dev) != 30 or len(held) != 30:
        errors.append(f"sizes {len(dev)}/{len(held)}")

    for oldp, new_rows, label in ((DEV_V12, dev, "dev"), (HELD_V12, held, "held")):
        old = load_jsonl(oldp)
        for a, b in zip(old, new_rows):
            if gold_tuple(a) != gold_tuple(b):
                errors.append(f"{label} gold mismatch {a['id']}")

    d_dev = Counter(r["project_difficulty_level"] for r in dev)
    d_held = Counter(r["project_difficulty_level"] for r in held)
    if dict(d_dev) != {"D1": 10, "D2": 6, "D3": 4, "D4": 5, "D5": 5}:
        errors.append(f"dev difficulty {dict(d_dev)}")
    if dict(d_held) != {"D1": 6, "D2": 6, "D3": 6, "D4": 6, "D5": 6}:
        errors.append(f"held difficulty {dict(d_held)}")

    for r in held:
        if not r.get("locked"):
            errors.append(f"{r['id']} not locked")
        if r.get("supersession_status") != "active":
            errors.append(f"{r['id']} not active")
        if r.get("prior_file_status") != "superseded_before_any_agent_run":
            errors.append(f"{r['id']} prior_file_status")
        if r.get("heldout_agent_run_status") != "never_agent_run":
            errors.append(f"{r['id']} heldout must remain never_agent_run")

    for r in dev:
        # v1.2 never Agent/LLM-run; Development pilot history is separate.
        if r.get("prior_file_status") != "superseded_before_any_agent_run":
            errors.append(f"{r['id']} prior_file_status")
        if "nonformal" not in str(r.get("development_exposure_history") or ""):
            errors.append(f"{r['id']} missing development_exposure_history")
        if r.get("supersession_status") != "active":
            errors.append(f"{r['id']} not active")

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

        for op in ops:
            if op not in active_ops:
                errors.append(f"{qid} gold op not in frozen catalog: {op}")

        for e in graph:
            if not isinstance(e, dict):
                errors.append(f"{qid} edge not object")
                continue
            for req in (
                "producer_operation",
                "producer_output_field",
                "consumer_operation",
                "consumer_input_field",
            ):
                if req not in e:
                    errors.append(f"{qid} edge missing {req}")
            pout = str(e.get("producer_output_field") or "")
            cin = str(e.get("consumer_input_field") or "")
            if pout in CONCEPTUAL_FIELDS or cin in CONCEPTUAL_FIELDS:
                errors.append(f"{qid} conceptual field edge {pout}->{cin}")
            # also reject legacy via-only conceptual edges
            if e.get("via") in CONCEPTUAL_FIELDS:
                errors.append(f"{qid} legacy conceptual via={e.get('via')}")
            prod = str(e.get("producer_operation") or "")
            cons = str(e.get("consumer_operation") or "")
            if prod not in active_ops:
                errors.append(f"{qid} producer not in catalog: {prod}")
            if cons not in active_ops:
                errors.append(f"{qid} consumer not in catalog: {cons}")
            if cons in params and cin not in params[cons]:
                errors.append(
                    f"{qid} consumer_input_field {cin} not in schema of {cons} "
                    f"(have {sorted(params.get(cons) or [])})"
                )
            if align == "exact" and et in {"L3", "L4"}:
                if not e.get("live_verified_dependency"):
                    errors.append(f"{qid} exact edge missing live_verified_dependency")

        if align == "exact" and et == "L2" and not tools:
            errors.append(f"{qid} exact L2 needs acceptable_tool_sets")

        if align == "exact" and et == "L3":
            if sc != 1 or oc < 2 or not dep or len(graph) < 1:
                errors.append(f"{qid} exact L3 topology invalid")
            if cross_server_field_edges(graph) > 0:
                errors.append(f"{qid} exact L3 has cross-server edges")
            if oc < 2:
                errors.append(f"{qid} single-op cannot be exact L3")

        if align == "exact" and et == "L4":
            if sc < 2 or oc < 2 or not dep or cross_server_field_edges(graph) < 1:
                errors.append(
                    f"{qid} exact L4 requires cross-server field edge "
                    f"(sc={sc} oc={oc} dep={dep} x={cross_server_field_edges(graph)})"
                )

        if align == "exact" and et == "L5":
            if rob != "capability_gap" or ops:
                errors.append(f"{qid} exact L5 invalid")

        if dep and not graph:
            errors.append(f"{qid} dep true but empty graph")
        if (not dep) and graph and align == "exact":
            errors.append(f"{qid} exact with graph but dep false")

    # exact duplicates keep company names
    bucket: Dict[str, List[str]] = {}
    for r in dev + held:
        bucket.setdefault(normalize_question(r["question"]), []).append(r["id"])
    for _, qids in bucket.items():
        if len(qids) > 1:
            errors.append(f"exact dup {qids}")
    for a in dev:
        for b in held:
            if (a.get("company") or "").lower() != (b.get("company") or "").lower():
                continue
            if jaccard(token_set(a["question"]), token_set(b["question"])) >= 0.92:
                errors.append(f"near-dup {a['id']}~{b['id']}")

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
    print(
        "adapted_L4=",
        sorted(
            r["id"]
            for r in rows
            if r.get("etom_level") == "L4" and r.get("etom_alignment") == "adapted"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
