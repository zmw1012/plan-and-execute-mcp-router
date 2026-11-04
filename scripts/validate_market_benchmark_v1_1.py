#!/usr/bin/env python3
"""Validate Market Development 15 / Held-out 20 v1.1 taxonomy + schema (no LLM)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from hk_equity_mcp.real_mcp.market_gateway import BLOCKED_MARKET_OPS  # noqa: E402
from hk_equity_mcp.real_mcp.market_normalize import (  # noqa: E402
    MARKET_CUTOFF,
    parse_session_date,
)

CATALOG = ROOT / "configs" / "operations" / "market_real_mcp_v1_1.yaml"
CANDIDATE = ROOT / "configs" / "operations" / "market_real_mcp_candidate_v1.yaml"
DEV = ROOT / "benchmarks" / "questions" / "market_development_15_v1_1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl"
DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")

EXPECTED_BLOCKED = {
    "ftshare.ft_v2_hk_hk_candlesticks",
    "ftshare.ft_v1_hk_hsi_daily_weight",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_question(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9.\-/% ]+", "", t)
    for tok in (
        "tencent",
        "hsbc",
        "alibaba",
        "hsi",
        "hang seng",
        "0700",
        "0005",
        "9988",
        "00700",
        "09988",
    ):
        t = t.replace(tok, "ENTITY")
    return DATE_RE.sub("DATE", t)


def token_set(text: str) -> Set[str]:
    return {w for w in normalize_question(text).split() if len(w) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pip_show_aktools_version() -> str:
    # Prefer discovery venv used for Market Real-MCP.
    candidates = [
        ROOT / ".venv-aktools-discovery" / "bin" / "pip",
        Path("pip"),
    ]
    for pip in candidates:
        cmd = [str(pip), "show", "aktools"]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for line in out.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    return ""


def extract_dates(obj: Any) -> List[date]:
    found: List[date] = []
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(extract_dates(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(extract_dates(v))
    elif isinstance(obj, str):
        for m in DATE_RE.findall(obj):
            d = parse_session_date(m)
            if d:
                found.append(d)
    return found


def main() -> int:
    errors: List[str] = []
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    active = {o["operation_id"] for o in catalog["operations"]}
    if len(active) != 13:
        errors.append(f"active ops {len(active)} != 13")

    # package metadata
    servers = catalog.get("servers") or []
    if isinstance(servers, dict):
        ak = servers.get("aktools") or {}
    else:
        ak = next((s for s in servers if s.get("server_id") == "aktools"), {})
    if ak.get("package") != "aktools":
        errors.append(f"package must be aktools, got {ak.get('package')}")
    if str(ak.get("package_version")) != "0.1.4":
        errors.append(f"package_version must be 0.1.4, got {ak.get('package_version')}")
    live = pip_show_aktools_version()
    if live and live != "0.1.4":
        errors.append(f"pip show aktools version {live} != catalog 0.1.4")

    blocked_ids = {b["operation_id"] for b in (catalog.get("blocked_operations") or [])}
    if blocked_ids != EXPECTED_BLOCKED:
        errors.append(f"blocked IDs {blocked_ids} != candidate exact {EXPECTED_BLOCKED}")
    cand_blocked = {
        o["operation_id"]
        for o in candidate.get("operations", [])
        if o.get("decision") == "blocked" or o.get("role") == "blocked"
    }
    if not EXPECTED_BLOCKED.issubset(cand_blocked):
        errors.append("blocked IDs not present in candidate audit operations")

    deferred = {d["operation_id"] for d in (catalog.get("deferred_operations") or [])}
    if "aktools.stock_hk_hist_min_em" not in deferred:
        errors.append("minute hist must remain deferred")
    if active & (EXPECTED_BLOCKED | deferred | BLOCKED_MARKET_OPS):
        errors.append("blocked/deferred leaked into active registry")

    roles = Counter(o["role"] for o in catalog["operations"])
    if dict(roles) != {"primary": 2, "supporting": 6, "distractor": 5}:
        errors.append(f"role counts {dict(roles)}")

    dev = load_jsonl(DEV)
    held = load_jsonl(HELD)
    if len(dev) != 15:
        errors.append(f"dev size {len(dev)}")
    if len(held) != 20:
        errors.append(f"held size {len(held)}")

    ids = [r["id"] for r in dev + held]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")

    # Project difficulty distributions (NOT called ETOM distribution)
    d_dev = Counter(r["project_difficulty_level"] for r in dev)
    d_held = Counter(r["project_difficulty_level"] for r in held)
    if dict(d_dev) != {"D1": 4, "D2": 3, "D3": 3, "D4": 3, "D5": 2}:
        errors.append(f"dev project difficulty {dict(d_dev)}")
    if dict(d_held) != {"D1": 4, "D2": 4, "D3": 4, "D4": 4, "D5": 4}:
        errors.append(f"held project difficulty {dict(d_held)}")

    for r in held:
        if not r.get("locked") or r.get("dataset_split") != "heldout":
            errors.append(f"{r['id']} held-out lock/split")
        if r.get("benchmark_role") != "final_evaluation_only":
            errors.append(f"{r['id']} role")

    for r in dev + held:
        qid = r["id"]
        if "± ±" in r["question"]:
            errors.append(f"{qid} still has ± ± typo")
        et = r.get("etom_level")
        align = r.get("etom_alignment")
        ops = r.get("gold_operations") or []
        oc = int(r.get("operation_count") or 0)
        sc = int(r.get("server_count") or 0)
        dep = bool(r.get("has_output_input_dependency"))
        rob = r.get("robustness_subtype")
        adj = r.get("adjustment_mode")
        reason = r.get("adjustment_mode_reason")

        if et == "L3":
            if sc != 1 or oc < 2 or not dep:
                errors.append(
                    f"{qid} etom L3 requires server_count==1, operation_count>=2, "
                    f"has_output_input_dependency==true (got sc={sc} oc={oc} dep={dep})"
                )
        if et == "L4":
            if sc < 2 or oc < 2 or not dep:
                errors.append(
                    f"{qid} etom L4 requires server_count>=2, operation_count>=2, "
                    f"has_output_input_dependency==true (got sc={sc} oc={oc} dep={dep})"
                )
        if et == "L5" and align == "exact":
            if rob != "capability_gap":
                errors.append(f"{qid} exact L5 must be capability_gap")
            if ops:
                errors.append(f"{qid} exact L5 must have empty gold_operations")
            if not r.get("confirmed_infeasible"):
                errors.append(f"{qid} exact L5 must be confirmed_infeasible")

        # adjustment schema
        is_hsi = r.get("company") == "HSI" or r.get("ticker") == "HSI"
        is_refusal = r.get("gold_status") in {"unanswerable", "unavailable"} or rob in {
            "capability_gap",
            "temporal_policy",
            "unavailable_provider",
        }
        if is_refusal:
            if adj is not None and reason is None:
                errors.append(f"{qid} refusal adj without reason")
            if adj is None and not reason:
                errors.append(f"{qid} refusal needs adjustment_mode_reason")
        elif is_hsi:
            if adj != "not_applicable_index_level":
                errors.append(f"{qid} HSI adjustment_mode must be not_applicable_index_level")
        elif r.get("task_type") == "identity":
            if adj is not None and not reason:
                errors.append(f"{qid} identity adj schema")
        else:
            # equity price-like
            if adj is None:
                errors.append(f"{qid} price question adjustment_mode must not be null")
            if adj == []:
                errors.append(f"{qid} empty adjustment_mode list invalid")

        for op in ops:
            if op not in active:
                errors.append(f"{qid} gold op not active: {op}")
            if op in EXPECTED_BLOCKED or op == "aktools.stock_hk_hist_min_em":
                errors.append(f"{qid} gold uses blocked/deferred {op}")

        for d in extract_dates(r):
            if d > MARKET_CUTOFF and rob != "temporal_policy" and et != "L5":
                # post-cutoff may appear only in temporal_policy / L5 wording
                if "2026-08-20" in r["question"] and rob == "temporal_policy":
                    continue
                if d > MARKET_CUTOFF and rob not in {"temporal_policy"}:
                    errors.append(f"{qid} date {d} > cutoff")

        required = [
            "project_difficulty_level",
            "orchestration_scope",
            "operation_count",
            "server_count",
            "has_output_input_dependency",
            "etom_alignment",
            "etom_alignment_reason",
            "robustness_subtype",
            "adjustment_mode_reason",
        ]
        for k in required:
            if k not in r:
                errors.append(f"{qid} missing {k}")

    # duplicates
    norms = [(r["id"], normalize_question(r["question"])) for r in dev + held]
    seen: Dict[str, List[str]] = {}
    for qid, n in norms:
        seen.setdefault(n, []).append(qid)
    for n, qids in seen.items():
        if len(qids) > 1:
            errors.append(f"exact dup {qids}")
    for a in dev:
        for b in held:
            ja = jaccard(token_set(a["question"]), token_set(b["question"]))
            if ja >= 0.85:
                errors.append(f"near-dup {a['id']}~{b['id']}={ja:.2f}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    etom_exact = Counter(
        r["etom_level"]
        for r in dev + held
        if r.get("etom_alignment") == "exact" and r.get("etom_level")
    )
    print("VALIDATION OK")
    print(f"active_operations={len(active)} roles={dict(roles)}")
    print(f"package=aktools=={ak.get('package_version')} pip_show={live or 'n/a'}")
    print(f"blocked={sorted(blocked_ids)}")
    print(f"project_difficulty_dev={dict(d_dev)}")
    print(f"project_difficulty_held={dict(d_held)}")
    print(f"exact_etom_counts={dict(etom_exact)}")
    print(
        "align_counts=",
        dict(Counter(r["etom_alignment"] for r in dev + held)),
    )
    print(
        "robustness_counts=",
        dict(Counter(r["robustness_subtype"] for r in dev + held)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
