#!/usr/bin/env python3
"""Deterministic validator for Financial held-out Test 30 v1.1.

No LLM, embedding, agent, or experiment execution.
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
DEFAULT_HELDOUT = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_1.jsonl"
DEFAULT_V1 = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1.jsonl"
DEFAULT_DEV = ROOT / "benchmarks" / "questions" / "financial_mcp_free_30_v1.jsonl"
DEFAULT_DEV_COV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
DEFAULT_CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
DEFAULT_README = ROOT / "benchmarks" / "heldout" / "README.md"

ALLOWED_EVIDENCE_TYPES = {
    "annual_results_announcement",
    "interim_results_announcement",
}
FORMAL_REPORT_MARKERS = (
    "annual report",
    "interim report",
    "annual report and accounts",
)
REQUIRED = {
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
    "benchmark_version",
    "catalog_version",
    "provenance_status",
    "gold_path_verification",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL {path}:{i}: {exc}") from exc
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


def looks_like_formal_report(title: str | None) -> bool:
    t = (title or "").lower()
    if not t:
        return False
    if "results" in t and "announcement" in t:
        return False
    if t.strip() in {"2025 results", "2026 interim results"}:
        return False
    return any(m in t for m in FORMAL_REPORT_MARKERS)


def validate(
    heldout_path: Path,
    v1_path: Path,
    dev_path: Path,
    dev_cov_path: Path,
    catalog_path: Path,
    readme_path: Path,
    near_dup_threshold: float = 0.85,
) -> list[str]:
    errors: list[str] = []
    rows = load_jsonl(heldout_path)

    if not v1_path.is_file():
        errors.append(f"missing superseded v1 file: {v1_path}")
    if not readme_path.is_file():
        errors.append(f"missing heldout README: {readme_path}")
    else:
        readme = readme_path.read_text(encoding="utf-8")
        if "superseded_before_any_agent_run" not in readme:
            errors.append("README must mark v1 as superseded_before_any_agent_run")
        if "financial_test_30_v1_1.jsonl" not in readme or "active" not in readme.lower():
            errors.append("README must mark v1.1 as active")

    if len(rows) != 30:
        errors.append(f"expected 30 rows, got {len(rows)}")
    if len({r.get("id") for r in rows}) != len(rows):
        errors.append("duplicate ids")

    etom = Counter(r.get("etom_level") for r in rows)
    for level in ("L1", "L2", "L3", "L4", "L5"):
        if etom.get(level) != 6:
            errors.append(f"ETOM {level} expected 6, got {etom.get(level)}")
    company = Counter(r.get("company") for r in rows)
    for name in ("Tencent", "HSBC", "Alibaba"):
        if company.get(name) != 10:
            errors.append(f"company {name} expected 10, got {company.get(name)}")
    ans = Counter(r.get("answerability_class") for r in rows)
    if ans.get("structured_answerable") != 10:
        errors.append(f"structured expected 10, got {ans.get('structured_answerable')}")
    if ans.get("evidence_answerable") != 14:
        errors.append(f"evidence expected 14, got {ans.get('evidence_answerable')}")
    if ans.get("intentionally_infeasible") != 6:
        errors.append(
            f"infeasible expected 6, got {ans.get('intentionally_infeasible')}"
        )

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    op_ids = {op["operation_id"] for op in catalog.get("operations", [])}

    for r in rows:
        missing = REQUIRED - set(r)
        if missing:
            errors.append(f"{r.get('id')}: missing {sorted(missing)}")
        if r.get("id") != r.get("question_id"):
            errors.append(f"{r.get('id')}: id != question_id")
        if r.get("dataset_split") != "heldout_test":
            errors.append(f"{r.get('id')}: bad dataset_split")
        if r.get("benchmark_role") != "final_evaluation_only":
            errors.append(f"{r.get('id')}: bad benchmark_role")
        if r.get("locked") is not True:
            errors.append(f"{r.get('id')}: locked must be true")
        if r.get("benchmark_version") != "financial_test_30_v1_1":
            errors.append(f"{r.get('id')}: benchmark_version must be financial_test_30_v1_1")
        if r.get("catalog_version") != "financial_real_mcp_v1":
            errors.append(f"{r.get('id')}: catalog_version must be financial_real_mcp_v1")

        for op in r.get("gold_operations") or []:
            if op not in op_ids:
                errors.append(f"{r.get('id')}: unknown gold op {op}")

        aclass = r.get("answerability_class")
        level = r.get("etom_level")
        status = str(r.get("provenance_status") or "")

        if aclass == "structured_answerable":
            if not status.startswith("live_verified"):
                errors.append(f"{r.get('id')}: structured not live-verified ({status})")
            gpv = r.get("gold_path_verification") or {}
            if not gpv:
                errors.append(f"{r.get('id')}: structured missing gold_path_verification")

        if aclass == "evidence_answerable":
            dtype = r.get("evidence_document_type")
            if dtype not in ALLOWED_EVIDENCE_TYPES:
                errors.append(
                    f"{r.get('id')}: evidence_document_type {dtype} incompatible with mcp-hkexnews v0.1"
                )
            if looks_like_formal_report(r.get("source_document")):
                errors.append(
                    f"{r.get('id')}: source_document looks like formal Annual/Interim Report "
                    f"but gold path is mcp-hkexnews: {r.get('source_document')}"
                )
            if r.get("evidence_retrieved_via") != "mcp_hkexnews":
                errors.append(f"{r.get('id')}: evidence_retrieved_via must be mcp_hkexnews")
            if not r.get("source_url") or "hkexnews.hk" not in str(r.get("source_url")):
                errors.append(f"{r.get('id')}: missing/invalid hkexnews source_url")
            if r.get("source_page_pdf") in (None, ""):
                errors.append(f"{r.get('id')}: missing source_page_pdf")
            if not r.get("evidence_snippet"):
                errors.append(f"{r.get('id')}: missing evidence_snippet")
            if not r.get("evidence_release_date"):
                errors.append(f"{r.get('id')}: missing evidence_release_date")
            if "live_verified" not in status:
                errors.append(f"{r.get('id')}: evidence not live-verified ({status})")

        if level == "L4":
            servers = set(r.get("gold_server_ids") or [])
            if len(servers) < 2:
                errors.append(f"{r.get('id')}: L4 needs >=2 gold servers")
            if r.get("l4_semantic_compatible") is not True:
                errors.append(f"{r.get('id')}: L4 semantic compatibility not confirmed")

        if level == "L5" or aclass == "intentionally_infeasible":
            if not r.get("infeasibility_kind") or not r.get("infeasibility_reason"):
                errors.append(f"{r.get('id')}: L5 missing infeasibility kind/reason")
            if status != "l5_infeasibility_confirmed":
                errors.append(f"{r.get('id')}: L5 provenance_status incorrect")

    # Duplicate checks vs Development (+ coverage annotation)
    dev_rows = load_jsonl(dev_path)
    if dev_cov_path.exists():
        dev_rows = dev_rows + load_jsonl(dev_cov_path)
    for r in rows:
        n = norm_q(r["question"])
        ht = tokens(r["question"])
        for d in dev_rows:
            if not d.get("question"):
                continue
            if norm_q(d["question"]) == n:
                errors.append(f"{r.get('id')}: exact duplicate of Development {d.get('id')}")
            if jaccard(ht, tokens(d["question"])) >= near_dup_threshold:
                errors.append(
                    f"{r.get('id')}: near-duplicate of {d.get('id')} "
                    f"(jaccard>={near_dup_threshold})"
                )

    # Guard: no agent result artifacts required/produced by this validator path
    agent_globs = list((ROOT / "outputs").glob("**/financial_*agent*")) if (ROOT / "outputs").exists() else []
    # Do not fail on unrelated outputs; only note if an obvious held-out agent run dir exists
    for p in agent_globs:
        if "heldout" in str(p).lower() and "agent" in str(p).lower():
            errors.append(f"unexpected held-out agent artifact present: {p}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--dev-coverage", type=Path, default=DEFAULT_DEV_COV)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    args = parser.parse_args()
    errors = validate(
        args.heldout,
        args.v1,
        args.dev,
        args.dev_coverage,
        args.catalog,
        args.readme,
        near_dup_threshold=args.near_dup_threshold,
    )
    if errors:
        print(f"FAIL ({len(errors)} issues):")
        for e in errors:
            print(f" - {e}")
        return 1
    print("PASS: financial held-out Test 30 v1.1 validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
