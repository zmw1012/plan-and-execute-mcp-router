#!/usr/bin/env python3
"""Deterministic validator for Financial held-out Test 30 v1.

No LLM, embedding, MCP live calls, or experiment execution.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELDOUT = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1.jsonl"
DEFAULT_DEV = ROOT / "benchmarks" / "questions" / "financial_mcp_free_30_v1.jsonl"
DEFAULT_DEV_COV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
DEFAULT_CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"

REQUIRED_FIELDS = {
    "id",
    "question_id",
    "question",
    "domain",
    "company",
    "ticker",
    "etom_level",
    "task_type",
    "answerability_class",
    "required_capabilities",
    "reference_answer",
    "gold_server_ids",
    "gold_operations",
    "formal_test_eligible",
    "dataset_split",
    "benchmark_role",
    "locked",
}

ANSWERABLE = {"structured_answerable", "evidence_answerable"}
L5_BEHAVIOR_KEYS = ("expected_behavior", "clarification_or_refusal")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{i}: {exc}") from exc
    return rows


def norm_q(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return set(norm_q(text).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate(
    heldout_path: Path,
    dev_path: Path,
    dev_cov_path: Path,
    catalog_path: Path,
    near_dup_threshold: float = 0.85,
) -> list[str]:
    errors: list[str] = []
    rows = load_jsonl(heldout_path)

    if len(rows) != 30:
        errors.append(f"expected 30 rows, got {len(rows)}")

    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")

    for r in rows:
        missing = REQUIRED_FIELDS - set(r)
        if missing:
            errors.append(f"{r.get('id')}: missing fields {sorted(missing)}")
        if r.get("id") != r.get("question_id"):
            errors.append(f"{r.get('id')}: id != question_id")
        if r.get("dataset_split") != "heldout_test":
            errors.append(f"{r.get('id')}: dataset_split must be heldout_test")
        if r.get("benchmark_role") != "final_evaluation_only":
            errors.append(f"{r.get('id')}: benchmark_role must be final_evaluation_only")
        if r.get("locked") is not True:
            errors.append(f"{r.get('id')}: locked must be true")
        if r.get("domain") != "financial":
            errors.append(f"{r.get('id')}: domain must be financial")

    etom = Counter(r.get("etom_level") for r in rows)
    for level in ("L1", "L2", "L3", "L4", "L5"):
        if etom.get(level) != 6:
            errors.append(f"ETOM {level} expected 6, got {etom.get(level)}")

    company = Counter(r.get("company") for r in rows)
    for name in ("Tencent", "HSBC", "Alibaba"):
        if company.get(name) != 10:
            errors.append(f"company {name} expected 10, got {company.get(name)}")

    ans = Counter(r.get("answerability_class") for r in rows)
    if ans.get("intentionally_infeasible") != 6:
        errors.append(
            f"intentionally_infeasible expected 6, got {ans.get('intentionally_infeasible')}"
        )
    if ans.get("structured_answerable") != 10:
        errors.append(
            f"structured_answerable expected 10, got {ans.get('structured_answerable')}"
        )
    if ans.get("evidence_answerable") != 14:
        errors.append(
            f"evidence_answerable expected 14, got {ans.get('evidence_answerable')}"
        )

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    op_ids = {op["operation_id"] for op in catalog.get("operations", [])}

    for r in rows:
        level = r.get("etom_level")
        aclass = r.get("answerability_class")
        ops = r.get("gold_operations") or []
        for op in ops:
            if op not in op_ids:
                errors.append(f"{r.get('id')}: unknown gold operation {op}")

        if level in {"L1", "L2", "L3", "L4"}:
            if aclass not in ANSWERABLE:
                errors.append(f"{r.get('id')}: L1-L4 must be answerable, got {aclass}")
            if not r.get("source_document"):
                errors.append(f"{r.get('id')}: L1-L4 missing source_document")
            if aclass in ANSWERABLE and not ops:
                errors.append(f"{r.get('id')}: answerable item missing gold_operations")
            if not r.get("reference_answer"):
                errors.append(f"{r.get('id')}: missing reference_answer")
            # Period sanity: forbid invented Tencent full-year FY2026 label as answerable period
            periods = r.get("periods") or ([r["period"]] if r.get("period") else [])
            if r.get("company") == "Tencent" and "FY2026" in periods and aclass in ANSWERABLE:
                errors.append(f"{r.get('id')}: Tencent full-year FY2026 is not an answerable period")

        if level == "L5":
            if aclass != "intentionally_infeasible":
                errors.append(f"{r.get('id')}: L5 must be intentionally_infeasible")
            if not any(r.get(k) for k in L5_BEHAVIOR_KEYS):
                errors.append(f"{r.get('id')}: L5 missing expected refusal/clarification behavior")
            if r.get("gold_status") not in {"unavailable", "clarification_needed"}:
                errors.append(f"{r.get('id')}: L5 gold_status must be unavailable or clarification_needed")

        if level == "L4":
            servers = set(r.get("gold_server_ids") or [])
            if len(servers) < 2:
                errors.append(f"{r.get('id')}: L4 should name >=2 gold servers")

    # Duplicate checks vs Development
    dev_rows = load_jsonl(dev_path)
    if dev_cov_path.exists():
        dev_rows = dev_rows + load_jsonl(dev_cov_path)
    dev_norm = {norm_q(d["question"]): d.get("id") for d in dev_rows if d.get("question")}
    for r in rows:
        n = norm_q(r["question"])
        if n in dev_norm:
            errors.append(f"{r.get('id')}: exact duplicate of Development {dev_norm[n]}")
        ht = tokens(r["question"])
        for d in dev_rows:
            if not d.get("question"):
                continue
            score = jaccard(ht, tokens(d["question"]))
            if score >= near_dup_threshold:
                errors.append(
                    f"{r.get('id')}: near-duplicate of {d.get('id')} (jaccard={score:.3f})"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--dev-coverage", type=Path, default=DEFAULT_DEV_COV)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    errors = validate(
        args.heldout,
        args.dev,
        args.dev_coverage,
        args.catalog,
        near_dup_threshold=args.near_dup_threshold,
    )
    if errors:
        print(f"FAIL ({len(errors)} issues):")
        for e in errors:
            print(f" - {e}")
        return 1
    print("PASS: financial held-out Test 30 v1 validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
