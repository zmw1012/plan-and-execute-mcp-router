#!/usr/bin/env python3
"""Validate frozen News Real-MCP catalog + Dev15/Held20 benchmarks (no LLM)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "news_real_mcp_v1.yaml"
CANDIDATE = ROOT / "configs" / "operations" / "news_real_mcp_candidate_v1.yaml"
DEV = ROOT / "benchmarks" / "questions" / "news_development_15_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "news_test_20_v1.jsonl"
CUTOFF = "2026-08-12"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_q(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"[^a-z0-9.\-/% ]+", "", t)


def tokens(text: str) -> Set[str]:
    t = normalize_q(text)
    for tok in ("tencent", "hsbc", "alibaba", "0700", "0005", "9988", "00700", "00005", "09988"):
        t = t.replace(tok, "ENTITY")
    t = re.sub(r"20\d{2}-\d{2}-\d{2}", "DATE", t)
    return {w for w in t.split() if len(w) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bucket(r: Dict[str, Any]) -> str:
    if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L5":
        return "exact_L5"
    if r.get("etom_alignment") == "adapted":
        return "adapted_robustness"
    if r.get("project_reasoning"):
        return "project_reasoning"
    if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L3":
        return "exact_L3"
    if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L1":
        return "exact_L1"
    return "other"


def main() -> int:
    errors: List[str] = []
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    if cat.get("catalog_status") != "frozen_v1" or cat.get("freeze_status") != "frozen":
        errors.append("catalog not frozen_v1")
    if (cat.get("temporal_policy") or {}).get("max_data_date") != CUTOFF:
        errors.append("cutoff mismatch")

    # checksum recompute excluding checksum field
    tmp = json.loads(json.dumps(cat))  # deep copy via json for yaml objs
    # better: reload and pop
    cat2 = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    stored = cat2.pop("catalog_checksum_sha256", None)
    raw = yaml.safe_dump(cat2, sort_keys=False, allow_unicode=True)
    calc = hashlib.sha256(raw.encode()).hexdigest()
    if stored != calc:
        errors.append(f"catalog checksum mismatch stored={stored} calc={calc}")

    ops = {o["operation_id"]: o for o in cat.get("operations") or []}
    runtime_ops = {
        oid for oid, o in ops.items() if o.get("runtime_registry") and o.get("decision") not in {"blocked", "rejected"}
    }
    for oid, o in ops.items():
        if o.get("decision") in {"blocked", "rejected"}:
            if o.get("runtime_registry") or o.get("exposed_to_agent"):
                errors.append(f"{oid} blocked/rejected must not be runtime/exposed")
        if oid == "aktools.stock_news_em" and o.get("formal_gold_eligible") is not False:
            errors.append("stock_news_em must not be formal gold")

    if not CANDIDATE.exists():
        errors.append("candidate_v1 missing as provenance trail")

    dev = load_jsonl(DEV)
    held = load_jsonl(HELD)
    if len(dev) != 15:
        errors.append(f"dev size {len(dev)}")
    if len(held) != 20:
        errors.append(f"held size {len(held)}")

    # distributions
    if Counter(bucket(r) for r in dev) != {
        "exact_L1": 5,
        "exact_L3": 4,
        "exact_L5": 2,
        "adapted_robustness": 2,
        "project_reasoning": 2,
    }:
        errors.append(f"dev bucket {dict(Counter(bucket(r) for r in dev))}")
    if Counter(bucket(r) for r in held) != {
        "exact_L1": 6,
        "exact_L3": 5,
        "exact_L5": 3,
        "adapted_robustness": 3,
        "project_reasoning": 3,
    }:
        errors.append(f"held bucket {dict(Counter(bucket(r) for r in held))}")

    # exact L2/L4 must be zero
    for r in dev + held:
        if r.get("etom_alignment") == "exact" and r.get("etom_level") in {"L2", "L4"}:
            errors.append(f"{r['id']} forbids exact {r['etom_level']}")

    # company coverage
    if Counter(r["company"] for r in dev) != {"Tencent": 5, "HSBC": 5, "Alibaba": 5}:
        errors.append(f"dev companies {dict(Counter(r['company'] for r in dev))}")
    if Counter(r["company"] for r in held) != {"Tencent": 7, "HSBC": 7, "Alibaba": 6}:
        errors.append(f"held companies {dict(Counter(r['company'] for r in held))}")

    # gold ops rules
    for r in dev + held:
        qid = r["id"]
        for op in r.get("gold_operations") or []:
            if op not in ops:
                errors.append(f"{qid} gold op missing from catalog: {op}")
                continue
            meta = ops[op]
            if meta.get("decision") in {"blocked", "rejected"}:
                errors.append(f"{qid} gold uses blocked/rejected {op}")
            if meta.get("formal_gold_eligible") is False and r.get("etom_level") != "L5":
                # L5 has empty gold ops; non-L5 must not use ineligible ops as gold
                errors.append(f"{qid} gold uses formal_gold_ineligible {op}")
            if not meta.get("runtime_registry", True) and r.get("gold_operations"):
                errors.append(f"{qid} gold op not in runtime registry {op}")
        if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L3":
            if not r.get("has_output_input_dependency"):
                errors.append(f"{qid} L3 missing dependency flag")
            graph = r.get("dependency_graph") or []
            if len(graph) < 1:
                errors.append(f"{qid} L3 empty dependency_graph")
            for e in graph:
                for k in (
                    "producer_operation",
                    "producer_output_field",
                    "consumer_operation",
                    "consumer_input_field",
                ):
                    if k not in e:
                        errors.append(f"{qid} edge missing {k}")
                if not e.get("live_verified_dependency"):
                    errors.append(f"{qid} edge not live_verified")
                if e.get("consumer_operation") not in ops:
                    errors.append(f"{qid} unknown consumer {e.get('consumer_operation')}")
            # require page_number -> start_page or local_key chain presence
            fields = {(e.get("producer_output_field"), e.get("consumer_input_field")) for e in graph}
            if ("page_number", "start_page") not in fields and ("local_key", "local_key") not in fields:
                errors.append(f"{qid} L3 missing core field edges")
        if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L5":
            if r.get("robustness_subtype") != "capability_gap":
                errors.append(f"{qid} exact L5 must be capability_gap")
            if r.get("gold_operations"):
                errors.append(f"{qid} exact L5 must have empty gold_operations")
        if r.get("etom_alignment") == "adapted":
            if r.get("robustness_subtype") not in {
                "invalid_symbol",
                "temporal_policy",
                "ambiguous_request",
            }:
                errors.append(f"{qid} adapted subtype unexpected {r.get('robustness_subtype')}")

        # Answer gold dates must be <= cutoff. Robustness probes may mention
        # post-cutoff requested_date as the rejected input — allow only there.
        if r.get("robustness_subtype") != "temporal_policy":
            blob = json.dumps(r.get("gold_values") or [], ensure_ascii=False)
            for m in re.findall(r"20\d{2}-\d{2}-\d{2}", blob):
                if m > CUTOFF:
                    errors.append(f"{qid} gold date after cutoff {m}")
        else:
            for gv in r.get("gold_values") or []:
                if not isinstance(gv, dict):
                    continue
                for k, v in gv.items():
                    if k in {"requested_date", "cutoff"}:
                        continue
                    if isinstance(v, str) and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", v) and v > CUTOFF:
                        errors.append(f"{qid} non-request gold date after cutoff {v}")

        if r.get("dataset_split") == "heldout":
            if not r.get("locked"):
                errors.append(f"{qid} held not locked")
            if r.get("heldout_agent_run_status") != "never_agent_run":
                errors.append(f"{qid} held agent status")
            if r.get("benchmark_role") != "final_evaluation":
                errors.append(f"{qid} held role")

    # duplicates
    norms = Counter(normalize_q(r["question"]) for r in dev + held)
    for n, c in norms.items():
        if c > 1:
            errors.append(f"exact duplicate question norm count={c}")
    for a in dev:
        for b in held:
            if a["company"] != b["company"]:
                continue
            if jaccard(tokens(a["question"]), tokens(b["question"])) >= 0.92:
                errors.append(f"near-dup {a['id']}~{b['id']}")

    # heldout must not appear in predictions outputs (scan workspace lightly)
    pred_hits = []
    for path in ROOT.glob("outputs/**/predictions.jsonl"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "news_ht_" in text:
            pred_hits.append(str(path.relative_to(ROOT)))
    if pred_hits:
        errors.append(f"heldout ids found in predictions: {pred_hits[:5]}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print("VALIDATION OK")
    print("catalog_ops", len(ops))
    print(
        "exposure",
        {
            "exposed_true": sum(1 for o in ops.values() if o.get("exposed_to_agent")),
            "runtime_true": sum(1 for o in ops.values() if o.get("runtime_registry")),
            "formal_gold_true": sum(1 for o in ops.values() if o.get("formal_gold_eligible")),
        },
    )
    print("dev_buckets", dict(Counter(bucket(r) for r in dev)))
    print("held_buckets", dict(Counter(bucket(r) for r in held)))
    print("dev_companies", dict(Counter(r["company"] for r in dev)))
    print("held_companies", dict(Counter(r["company"] for r in held)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
