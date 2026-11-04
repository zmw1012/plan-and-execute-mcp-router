#!/usr/bin/env python3
"""Validate Cross-domain benchmark v1 (no LLM / no network)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "operations" / "cross_domain_real_mcp_v1.yaml"
DEV = ROOT / "benchmarks" / "questions" / "cross_domain_development_10_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "cross_domain_test_15_v1.jsonl"
FROZEN = [
    ROOT / "configs" / "operations" / "financial_real_mcp.yaml",
    ROOT / "configs" / "operations" / "market_real_mcp_v1_1.yaml",
    ROOT / "configs" / "operations" / "news_real_mcp_v1.yaml",
    ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_3.jsonl",
    ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_3.jsonl",
    ROOT / "benchmarks" / "questions" / "market_development_15_v1_1.jsonl",
    ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl",
    ROOT / "benchmarks" / "questions" / "news_development_15_v1.jsonl",
    ROOT / "benchmarks" / "heldout" / "news_test_20_v1.jsonl",
]
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
    for tok in (
        "tencent",
        "hsbc",
        "alibaba",
        "0700",
        "0005",
        "9988",
        "00700",
        "00005",
        "09988",
    ):
        t = t.replace(tok, "ENTITY")
    t = re.sub(r"20\d{2}-\d{2}-\d{2}", "DATE", t)
    return {w for w in t.split() if len(w) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    errors: List[str] = []
    man = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if man.get("catalog_id") != "cross_domain_real_mcp_v1":
        errors.append("bad catalog_id")
    refs = {r["path"] for r in man.get("referenced_catalogs") or []}
    for req in (
        "configs/operations/financial_real_mcp.yaml",
        "configs/operations/market_real_mcp_v1_1.yaml",
        "configs/operations/news_real_mcp_v1.yaml",
    ):
        if req not in refs:
            errors.append(f"manifest missing ref {req}")
        if not (ROOT / req).exists():
            errors.append(f"missing frozen file {req}")

    # frozen files must exist (read-only presence)
    for p in FROZEN:
        if not p.exists():
            errors.append(f"missing frozen artifact {p.relative_to(ROOT)}")

    allowed_ops = {o["operation_id"] for o in man.get("operations") or []}
    formal_map = {
        o["operation_id"]: o.get("formal_gold_eligible")
        for o in man.get("operations") or []
    }

    dev = load_jsonl(DEV)
    held = load_jsonl(HELD)
    if len(dev) != 10:
        errors.append(f"dev size {len(dev)}")
    if len(held) != 15:
        errors.append(f"held size {len(held)}")

    want_dev = {
        "financial+market": 3,
        "financial+news": 3,
        "market+news": 2,
        "financial+market+news": 2,
    }
    want_held = {
        "financial+market": 4,
        "financial+news": 4,
        "market+news": 4,
        "financial+market+news": 3,
    }
    if dict(Counter(r["domain_combo"] for r in dev)) != want_dev:
        errors.append(f"dev combo {dict(Counter(r['domain_combo'] for r in dev))}")
    if dict(Counter(r["domain_combo"] for r in held)) != want_held:
        errors.append(f"held combo {dict(Counter(r['domain_combo'] for r in held))}")

    for r in dev + held:
        qid = r["id"]
        domains = r.get("domains") or []
        if len(domains) < 2:
            errors.append(f"{qid} needs >=2 domains")
        if len(set(domains)) != len(domains):
            errors.append(f"{qid} duplicate domain entries")
        for op in r.get("gold_operations") or []:
            if op not in allowed_ops:
                errors.append(f"{qid} gold op not in manifest: {op}")
            # if any namespaced entry marks false for all domains, reject
            # allow if at least one domain entry is formal-gold eligible
            elig = [
                o.get("formal_gold_eligible")
                for o in man["operations"]
                if o["operation_id"] == op
            ]
            if elig and not any(elig):
                errors.append(f"{qid} gold op formal ineligible everywhere: {op}")
        dep = r.get("tool_dependency_type")
        if dep not in {
            "independent_parallel",
            "sequential_field_dependency",
            "mixed",
        }:
            errors.append(f"{qid} bad dependency type {dep}")
        if r.get("exact_etom_alignment") is True:
            if r.get("exact_etom_level") != "L4":
                errors.append(f"{qid} exact alignment requires exact_etom_level=L4")
            graph = r.get("dependency_graph") or []
            if not graph:
                errors.append(f"{qid} exact L4 needs dependency_graph")
            cross = 0
            for e in graph:
                for k in (
                    "producer_operation",
                    "producer_output_field",
                    "consumer_operation",
                    "consumer_input_field",
                ):
                    if not e.get(k):
                        errors.append(f"{qid} edge missing {k}")
                if not e.get("live_verified_dependency"):
                    errors.append(f"{qid} edge not live verified")
                # naive server prefix check
                p = str(e.get("producer_operation"))
                c = str(e.get("consumer_operation"))
                if p.split(".", 1)[0] != c.split(".", 1)[0] or (
                    "list_hkexnews" in p and "stock_hk_hist" in c
                ):
                    cross += 1
            if cross < 1 and not any(
                "list_hkexnews" in str(e.get("producer_operation"))
                and "stock_hk_hist" in str(e.get("consumer_operation"))
                for e in graph
            ):
                errors.append(f"{qid} exact L4 needs cross-domain field edge")
        else:
            # must not claim exact alignment true
            if r.get("exact_etom_alignment") not in {False, None}:
                pass
        if dep == "independent_parallel" and r.get("exact_etom_alignment") is True:
            errors.append(f"{qid} independent_parallel cannot be exact L4")
        if r.get("dataset_split") == "heldout":
            if not r.get("locked"):
                errors.append(f"{qid} held not locked")
            if r.get("heldout_agent_run_status") != "never_agent_run":
                errors.append(f"{qid} held agent status")

        # answer gold dates <= cutoff except robustness requested dates
        if r.get("robustness_subtype") not in {"temporal_policy"}:
            blob = json.dumps(r.get("gold_values") or [], ensure_ascii=False)
            for m in re.findall(r"20\d{2}-\d{2}-\d{2}", blob):
                if m > CUTOFF:
                    errors.append(f"{qid} gold date after cutoff {m}")

        if r.get("live_audit_status") != "passed":
            errors.append(f"{qid} live_audit_status")

    # duplicates within cross-domain
    norms = Counter(normalize_q(r["question"]) for r in dev + held)
    for n, c in norms.items():
        if c > 1:
            errors.append("exact duplicate within cross-domain")

    # near-dup vs single-domain benches (same company)
    singles: List[Dict[str, Any]] = []
    for p in (
        ROOT / "benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl",
        ROOT / "benchmarks/questions/market_development_15_v1_1.jsonl",
        ROOT / "benchmarks/questions/news_development_15_v1.jsonl",
        ROOT / "benchmarks/heldout/financial_test_30_v1_3.jsonl",
        ROOT / "benchmarks/heldout/market_test_20_v1_1.jsonl",
        ROOT / "benchmarks/heldout/news_test_20_v1.jsonl",
    ):
        singles.extend(load_jsonl(p))
    for a in dev + held:
        for b in singles:
            if (a.get("company") or "").lower() != (b.get("company") or "").lower():
                continue
            if jaccard(tokens(a["question"]), tokens(b.get("question") or "")) >= 0.90:
                errors.append(f"near-dup {a['id']}~{b.get('id')}")

    # held ids not in predictions
    for path in ROOT.glob("outputs/**/predictions.jsonl"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "xd_ht_" in text:
            errors.append(f"held id in predictions {path}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print("VALIDATION OK")
    print("manifest_counts", man.get("counts"))
    print("dev_combo", dict(Counter(r["domain_combo"] for r in dev)))
    print("held_combo", dict(Counter(r["domain_combo"] for r in held)))
    print("companies_dev", dict(Counter(r["company"] for r in dev)))
    print("companies_held", dict(Counter(r["company"] for r in held)))
    print("dependency", dict(Counter(r["tool_dependency_type"] for r in dev + held)))
    print(
        "exact_L4",
        sum(1 for r in dev + held if r.get("exact_etom_alignment") is True),
    )
    print(
        "parallel_project",
        sum(
            1
            for r in dev + held
            if r.get("project_taxonomy") == "cross_domain_parallel_reasoning"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
