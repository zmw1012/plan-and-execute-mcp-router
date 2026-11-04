#!/usr/bin/env python3
"""Validate Market Development 15 + Held-out 20 against frozen catalog (no LLM)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from hk_equity_mcp.real_mcp.market_gateway import BLOCKED_MARKET_OPS  # noqa: E402
from hk_equity_mcp.real_mcp.market_normalize import (  # noqa: E402
    MARKET_CUTOFF,
    parse_session_date,
)

CATALOG = ROOT / "configs" / "operations" / "market_real_mcp_v1.yaml"
DEV = ROOT / "benchmarks" / "questions" / "market_development_15_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "market_test_20_v1.jsonl"

DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_question(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9.\-/% ]+", "", t)
    # scrub tickers/dates/company names for near-dup check
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
    t = DATE_RE.sub("DATE", t)
    return t


def token_set(text: str) -> Set[str]:
    return {w for w in normalize_question(text).split() if len(w) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
        if len(obj) == 8 and obj.isdigit():
            d = parse_session_date(obj)
            if d:
                found.append(d)
    return found


def main() -> int:
    errors: List[str] = []
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    active = {o["operation_id"] for o in catalog["operations"]}
    if active & BLOCKED_MARKET_OPS:
        errors.append(f"blocked ops in active catalog: {active & BLOCKED_MARKET_OPS}")

    roles = Counter(o["role"] for o in catalog["operations"])
    if dict(roles) != {"primary": 2, "supporting": 6, "distractor": 5}:
        errors.append(f"unexpected role counts: {dict(roles)}")

    dev = load_jsonl(DEV)
    held = load_jsonl(HELD)
    if len(dev) != 15:
        errors.append(f"dev size {len(dev)} != 15")
    if len(held) != 20:
        errors.append(f"held size {len(held)} != 20")

    ids = [r["id"] for r in dev + held]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")

    def check_etom(rows: List[Dict[str, Any]], expected: Dict[str, int], label: str) -> None:
        c = Counter(r["etom_level"] for r in rows)
        if dict(c) != expected:
            errors.append(f"{label} ETOM {dict(c)} != {expected}")

    check_etom(dev, {"L1": 4, "L2": 3, "L3": 3, "L4": 3, "L5": 2}, "dev")
    check_etom(held, {"L1": 4, "L2": 4, "L3": 4, "L4": 4, "L5": 4}, "held")

    for r in held:
        if not r.get("locked") or not r.get("frozen"):
            errors.append(f"{r['id']} held-out not locked/frozen")
        if r.get("dataset_split") != "heldout":
            errors.append(f"{r['id']} bad split")
        if r.get("benchmark_role") != "final_evaluation_only":
            errors.append(f"{r['id']} bad benchmark_role")

    companies = {r.get("company") for r in dev}
    for need in ("Tencent", "HSBC", "Alibaba", "HSI"):
        if need not in companies:
            errors.append(f"dev missing company/index {need}")

    for r in dev + held:
        qid = r["id"]
        for d in extract_dates(r):
            if d > MARKET_CUTOFF and r.get("etom_level") != "L5":
                # L5 may mention post-cutoff as the refusal target
                errors.append(f"{qid} date {d} > cutoff")
        if r["etom_level"] in {"L1", "L2", "L3", "L4"}:
            ops = r.get("gold_operations") or []
            if not ops:
                errors.append(f"{qid} L1-L4 missing gold_operations")
            for op in ops:
                if op not in active:
                    errors.append(f"{qid} gold op not active: {op}")
                if op in BLOCKED_MARKET_OPS:
                    errors.append(f"{qid} gold uses blocked op: {op}")
            if r.get("gold_status") != "answered":
                errors.append(f"{qid} L1-L4 not answered")
            # field completeness
            gvs = r.get("gold_values") or []
            if not gvs:
                errors.append(f"{qid} missing gold_values")
            for gv in gvs:
                if "tolerance_abs" not in gv and "tolerance_rel" not in gv and "tolerances" not in gv:
                    errors.append(f"{qid} missing tolerance")
        else:
            if r.get("gold_operations"):
                errors.append(f"{qid} L5 should not have gold_operations")
            if not r.get("confirmed_infeasible"):
                errors.append(f"{qid} L5 not confirmed_infeasible")

    # duplicate checks
    norms = [(r["id"], normalize_question(r["question"])) for r in dev + held]
    norm_map: Dict[str, List[str]] = {}
    for qid, n in norms:
        norm_map.setdefault(n, []).append(qid)
    for n, qids in norm_map.items():
        if len(qids) > 1:
            errors.append(f"exact normalized duplicate: {qids}")

    # near-dup across splits
    for a in dev:
        for b in held:
            ja = jaccard(token_set(a["question"]), token_set(b["question"]))
            if ja >= 0.85:
                errors.append(
                    f"high-overlap near duplicate {a['id']}~{b['id']} jaccard={ja:.2f}"
                )

    # coverage cues
    blob = " ".join(r["question"].lower() for r in dev)
    for cue in (
        "close",
        "volume",
        "turnover",
        "qfq",
        "return",
        "volatility",
        "drawdown",
        "ma20",
        "hsi",
        "relative",
    ):
        if cue not in blob and cue != "ma20":
            # ma20 may appear as MA20
            if cue == "ma20" and "ma20" not in blob.replace(" ", ""):
                errors.append(f"dev missing coverage cue {cue}")
        elif cue != "ma20" and cue not in blob:
            errors.append(f"dev missing coverage cue {cue}")

    if "ma20" not in blob and "ma 20" not in blob:
        errors.append("dev missing MA20 coverage")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("VALIDATION OK")
    print(f"active_operations={len(active)} roles={dict(roles)}")
    print(f"dev={len(dev)} held={len(held)}")
    print(
        "live_audited_L1_L4="
        f"{sum(1 for r in dev+held if r['etom_level'] in {'L1','L2','L3','L4'})}"
    )
    print(
        "confirmed_infeasible_L5="
        f"{sum(1 for r in dev+held if r['etom_level']=='L5')}"
    )
    print("exact_or_near_duplicates=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
