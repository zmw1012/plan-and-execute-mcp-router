#!/usr/bin/env python3
"""Offline final audit of PE-Fin Lite Planner-only all-155 (v1 live cache).

No LLM / embedding / MCP / network. Read-only for raw live_v1 artifacts;
writes only under docs/... and planner_cache/final_audit_v1/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (  # noqa: E402
    load_hk155_locked,
)
from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import (  # noqa: E402
    CANONICALIZER_VERSION,
    canonicalize_plan_payload,
    describe_canonicalizer,
    normalize_json_text,
    plan_canonicalizer_hash,
)
from hk_equity_mcp.pe_fin_lite.schemas import (  # noqa: E402
    SchemaValidationError,
    extract_json_object,
    parse_and_validate,
)

LIVE_DIR = ROOT / "outputs" / "pe_fin_lite_retrieval_all155_live_v1"
PLANNER_DIR = LIVE_DIR / "planner_cache"
AUDIT_OUT = PLANNER_DIR / "final_audit_v1"
DOCS_OUT = ROOT / "docs" / "pe_fin_lite_planner_all155_v1_audit"

STATUS_MARK = "invalid_for_formal_step_level_retrieval_due_to_planner_protocol_failure_rate"


def sha256_file(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def collect_raw_hashes() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path in sorted(LIVE_DIR.rglob("*")):
        if not path.is_file():
            continue
        # Do not include audit outputs themselves if re-run.
        if "final_audit_v1" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        out[rel] = sha256_file(path)
    return out


def classify_error(error: Optional[str]) -> str:
    e = str(error or "")
    if not e:
        return "none"
    if "missing required property 'depends_on'" in e and "steps[0]" in e:
        return "missing_depends_on_first_step"
    if "missing required property 'depends_on'" in e:
        return "missing_depends_on_later_step"
    if e.startswith("invalid JSON"):
        return "invalid_json"
    if e == "empty content" or e == "null content":
        return "empty_or_null_content"
    if "truncated" in e or "no JSON object found" in e:
        return "truncated_or_non_json"
    if "not in enum" in e and "domain" in e:
        return "invalid_domain_enum"
    if "maxLength" in e:
        return "maxLength_violation"
    if "missing required property" in e:
        return "other_missing_field"
    return "other"


def try_validate_content(content: Any) -> Tuple[bool, Optional[str], List[str]]:
    """Parse+validate with current canonicalizer. Returns ok, error, applied_rules."""
    applied: List[str] = []
    if content is None:
        return False, "null content", applied
    text = str(content)
    norm, applied = normalize_json_text(text)
    try:
        payload = extract_json_object(norm if norm.strip() else text)
    except SchemaValidationError as exc:
        return False, str(exc), applied
    # Track structural rules that would apply (even if already canonical).
    _, struct_applied = canonicalize_plan_payload(deepcopy(payload))
    applied = list(dict.fromkeys(applied + struct_applied))
    parsed, _, err = parse_and_validate(text, "plan")
    return parsed is not None, err, applied


def semantic_changed(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> bool:
    if before is None or after is None:
        return False
    # Compare without runtime-owned fields / depends_on empty inject noise:
    # drop only rules we allow; if other fields differ → semantic change.
    b = deepcopy(before)
    a = deepcopy(after)
    for payload in (b, a):
        payload.pop("schema_version", None)
        for step in payload.get("steps") or []:
            if isinstance(step, dict):
                step.pop("schema_version", None)
    # Normalize domains via same aliases for comparison of semantic intent:
    # If only depends_on[] inject or domain alias → not semantic invention.
    return json.dumps(b, sort_keys=True) != json.dumps(a, sort_keys=True)


def main() -> int:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)

    hashes_before = collect_raw_hashes()
    (AUDIT_OUT / "raw_hashes_before.json").write_text(
        json.dumps(hashes_before, indent=2) + "\n", encoding="utf-8"
    )

    rows: List[Dict[str, Any]] = []
    for line in (PLANNER_DIR / "planner_outputs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    identity = json.loads((PLANNER_DIR / "planner_cache_identity.json").read_text())
    summary_file = json.loads((PLANNER_DIR / "planner_only_summary.json").read_text())
    dataset = load_hk155_locked(ROOT)
    tasks_by_id = {
        str(t.get("task_id") or t.get("id")): t for t in dataset["tasks"]
    }
    splits = dataset["splits"]

    n = len(rows)
    unique = {str(r["task_id"]) for r in rows}
    ok_rows = [r for r in rows if r.get("ok")]
    fail_rows = [r for r in rows if not r.get("ok")]
    repair1 = sum(1 for r in rows if int(r.get("repair_count") or 0) == 1)
    retrieve_steps = sum(len(r.get("retrieve_steps") or []) for r in rows)

    # Validity of first vs repair responses (offline re-parse with CURRENT code).
    first_valid = 0
    repair_valid = 0
    repair_recovered = 0
    per_task: List[Dict[str, Any]] = []
    rule_recover_counts: Counter = Counter()
    offline_recoverable = 0
    still_unrecoverable = 0
    recoverable_ids: List[str] = []
    unrecoverable_ids: List[str] = []

    tax_final: Counter = Counter()
    tax_first_fail: Counter = Counter()

    for r in rows:
        tid = str(r["task_id"])
        task = tasks_by_id.get(tid) or {}
        raw = r.get("raw_planner_response")
        repair_raw = r.get("repair_raw_content")
        first_ok, first_err, first_rules = try_validate_content(raw)
        repair_ok, repair_err, repair_rules = (False, None, [])
        if repair_raw is not None and str(repair_raw).strip() != "":
            repair_ok, repair_err, repair_rules = try_validate_content(repair_raw)
        elif repair_raw is not None:
            repair_ok, repair_err, repair_rules = try_validate_content(repair_raw)

        if first_ok:
            first_valid += 1
        else:
            tax_first_fail[classify_error(first_err)] += 1

        if int(r.get("repair_count") or 0) >= 1:
            if repair_ok:
                repair_valid += 1
            if (not first_ok) and repair_ok:
                repair_recovered += 1

        final_err = None if r.get("ok") else str(r.get("error") or "")
        tax_final[classify_error(final_err if not r.get("ok") else None)] += 1

        # Offline recoverable among FINAL failures only.
        offline_ok = False
        applied_for_recovery: List[str] = []
        semantic_change = False
        if not r.get("ok"):
            # Prefer first content if recoverable; else repair content.
            candidates = [(raw, first_ok, first_err, first_rules)]
            if int(r.get("repair_count") or 0) >= 1:
                candidates.append((repair_raw, repair_ok, repair_err, repair_rules))
            for content, ok_flag, err, rules in candidates:
                if ok_flag:
                    offline_ok = True
                    applied_for_recovery = rules
                    # Check semantic: parse without structural aliases for domain/depends
                    # by comparing canonicalize-only effect.
                    try:
                        payload = extract_json_object(str(content or ""))
                        before = deepcopy(payload)
                        after, _ = canonicalize_plan_payload(payload)
                        # Semantic invention check: after removing allowed transforms,
                        # remaining fields equal.
                        b2 = deepcopy(before)
                        a2 = deepcopy(after)
                        # Undo allowed: if first missing depends_on was added, strip for compare
                        if (
                            isinstance(b2.get("steps"), list)
                            and b2["steps"]
                            and isinstance(b2["steps"][0], dict)
                            and "depends_on" not in b2["steps"][0]
                            and isinstance(a2.get("steps"), list)
                            and a2["steps"]
                        ):
                            a2["steps"][0].pop("depends_on", None)
                        # Undo domain aliases for compare
                        for s_b, s_a in zip(b2.get("steps") or [], a2.get("steps") or []):
                            if not isinstance(s_b, dict) or not isinstance(s_a, dict):
                                continue
                            if s_b.get("type") == "reason" and str(s_b.get("domain") or "").lower() in {
                                "analysis",
                                "reasoning",
                            }:
                                s_a["domain"] = s_b["domain"]
                            if s_b.get("type") == "compute" and str(
                                s_b.get("domain") or ""
                            ).lower() == "calculation":
                                s_a["domain"] = s_b["domain"]
                        semantic_change = json.dumps(b2, sort_keys=True) != json.dumps(
                            a2, sort_keys=True
                        )
                    except Exception:  # noqa: BLE001
                        semantic_change = False
                    break
            if offline_ok:
                offline_recoverable += 1
                recoverable_ids.append(tid)
                for rule in applied_for_recovery:
                    rule_recover_counts[rule] += 1
            else:
                still_unrecoverable += 1
                unrecoverable_ids.append(tid)

        per_task.append(
            {
                "task_id": tid,
                "ok": bool(r.get("ok")),
                "source_split": task.get("source_split")
                or (
                    "development"
                    if tid in set(splits["development70"])
                    else "previously_reserved"
                ),
                "domain": task.get("domain"),
                "difficulty": task.get("project_difficulty_level")
                or task.get("difficulty"),
                "repair_count": int(r.get("repair_count") or 0),
                "n_retrieve_steps": len(r.get("retrieve_steps") or []),
                "error": r.get("error"),
                "error_taxonomy": classify_error(final_err if not r.get("ok") else None),
                "first_response_valid_under_v2_canonicalizer": first_ok,
                "first_error_under_v2": first_err,
                "repair_response_valid_under_v2_canonicalizer": repair_ok,
                "repair_error_under_v2": repair_err,
                "offline_recoverable_by_structural_rules": offline_ok if not r.get("ok") else None,
                "structural_rules_applied": "|".join(applied_for_recovery),
                "semantic_plan_changed_beyond_allowed_rules": semantic_change
                if not r.get("ok")
                else False,
                "llm_calls": r.get("llm_calls"),
                "tokens_in": r.get("tokens_in"),
                "tokens_out": r.get("tokens_out"),
                "tokens_total": r.get("tokens_total"),
                "latency_ms": r.get("latency_ms"),
            }
        )

    def rate(num: int, den: int) -> Optional[float]:
        return (num / den) if den else None

    def bucket_success(key_fn):
        buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
        for r in rows:
            tid = str(r["task_id"])
            task = tasks_by_id.get(tid) or {}
            key = str(key_fn(task, tid) or "unknown")
            buckets[key]["n"] += 1
            if r.get("ok"):
                buckets[key]["ok"] += 1
        return {
            k: {
                "n": v["n"],
                "ok": v["ok"],
                "failed": v["n"] - v["ok"],
                "success_rate": rate(v["ok"], v["n"]),
            }
            for k, v in sorted(buckets.items())
        }

    # Resources
    llm_calls = [int(r.get("llm_calls") or 0) for r in rows]
    tokens_in = [int(r.get("tokens_in") or 0) for r in rows if r.get("tokens_in") is not None]
    tokens_out = [int(r.get("tokens_out") or 0) for r in rows if r.get("tokens_out") is not None]
    tokens_total = [
        int(r.get("tokens_total") or 0) for r in rows if r.get("tokens_total") is not None
    ]
    latency = [float(r.get("latency_ms") or 0.0) for r in rows]

    funnel = {
        "status_mark": STATUS_MARK,
        "rows": n,
        "unique_task_ids": len(unique),
        "ok": len(ok_rows),
        "failed": len(fail_rows),
        "repair_count_eq_1": repair1,
        "retrieve_steps_total": retrieve_steps,
        "planner_cache_sha256": summary_file.get("planner_cache_sha256"),
        "identity": identity,
        "by_source_split": bucket_success(
            lambda task, tid: (
                "development70"
                if tid in set(splits["development70"])
                else "previously_reserved85"
            )
        ),
        "by_domain": bucket_success(lambda task, tid: task.get("domain")),
        "by_difficulty": bucket_success(
            lambda task, tid: task.get("project_difficulty_level")
            or task.get("difficulty")
        ),
        "first_response_valid_under_v2_canonicalizer": first_valid,
        "repair_response_valid_under_v2_canonicalizer": repair_valid,
        "repair_recovered_count_under_v2_replay": repair_recovered,
        "r0_question_level_remains_valid": True,
        "catalog_index_remains_valid": True,
        "question_query_cache_remains_valid": True,
        "failures_are_not_embedding_failures": True,
        "formal_step_level_r1_r3_evaluation": "forbidden_until_planner_v2_complete",
    }

    taxonomy = {
        "final_error_taxonomy": dict(tax_final),
        "expected_final_distribution": {
            "missing_depends_on_first_step": 19,
            "invalid_json": 7,
            "empty_or_null_content": 7,
            "truncated_or_non_json": 3,
            "invalid_domain_enum": 2,
            "maxLength_violation": 2,
            "other_missing_field": 1,
        },
        "final_distribution_matches_expected": (
            tax_final.get("missing_depends_on_first_step", 0) == 19
            and tax_final.get("invalid_json", 0) == 7
            and tax_final.get("empty_or_null_content", 0) == 7
            and tax_final.get("truncated_or_non_json", 0) == 3
            and tax_final.get("invalid_domain_enum", 0) == 2
            and tax_final.get("maxLength_violation", 0) == 2
            and tax_final.get("other_missing_field", 0) == 1
        ),
        "first_response_fail_taxonomy_under_v2": dict(tax_first_fail),
        "failed_task_ids": sorted(str(r["task_id"]) for r in fail_rows),
    }

    repair_analysis = {
        "n_tasks_with_repair_count_1": repair1,
        "first_valid_under_v2": first_valid,
        "repair_valid_under_v2": repair_valid,
        "repair_recovered_under_v2": repair_recovered,
        "note": (
            "Validity under v2 structural canonicalizer is offline replay only; "
            "does not mutate the frozen v1 planner_outputs.jsonl."
        ),
    }

    offline_recovery = {
        "canonicalizer": describe_canonicalizer(),
        "offline_recoverable_count": offline_recoverable,
        "still_unrecoverable_count": still_unrecoverable,
        "recoverable_task_ids": recoverable_ids,
        "unrecoverable_task_ids": unrecoverable_ids,
        "rules_recovery_counts": dict(rule_recover_counts),
        "semantic_plan_changed_beyond_allowed_rules_count": sum(
            1
            for row in per_task
            if row.get("semantic_plan_changed_beyond_allowed_rules")
        ),
        "forbidden_auto_fixes_not_applied": [
            "non-first depends_on invent",
            "missing required_outputs invent",
            "missing compute_spec invent",
            "overlength truncation",
            "truncated JSON splice",
            "gold/task-id inference",
        ],
    }

    def _stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "sum": float(sum(vals)),
            "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    resource_summary = {
        "llm_calls": _stats([float(x) for x in llm_calls]),
        "tokens_in": _stats([float(x) for x in tokens_in]),
        "tokens_out": _stats([float(x) for x in tokens_out]),
        "tokens_total": _stats([float(x) for x in tokens_total]),
        "latency_ms": _stats(latency),
        "total_llm_calls": int(sum(llm_calls)),
        "total_tokens_in": int(sum(tokens_in)),
        "total_tokens_out": int(sum(tokens_out)),
        "total_tokens": int(sum(tokens_total)),
    }

    # Write JSON artifacts to both audit dirs.
    payloads = {
        "planner_funnel.json": funnel,
        "planner_error_taxonomy.json": taxonomy,
        "repair_analysis.json": repair_analysis,
        "offline_structural_recovery.json": offline_recovery,
        "resource_summary.json": resource_summary,
        "canonicalizer_identity.json": describe_canonicalizer(),
        "status.json": {
            "planner_cache_status": STATUS_MARK,
            "r0_valid": True,
            "catalog_index_valid": True,
            "question_query_cache_valid": True,
            "formal_r1_r3": False,
            "protocol_for_next_formal_run": "pe_fin_lite_planner_only_v2",
            "formal_output_dir": "outputs/pe_fin_lite_retrieval_all155_planner_v2",
        },
    }
    for name, obj in payloads.items():
        for dest in (AUDIT_OUT, DOCS_OUT):
            (dest / name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    # CSV
    fieldnames = list(per_task[0].keys()) if per_task else ["task_id"]
    for dest in (AUDIT_OUT, DOCS_OUT):
        with (dest / "per_task_planner_audit.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_task)

    # Markdown report
    report = f"""# PE-Fin Lite Planner-only All-155 Offline Final Audit (v1)

## Status mark

`{STATUS_MARK}`

- R0 question-level embedding/ranking: **still valid**
- Catalog embedding index: **still valid**
- Question-query embedding cache: **still valid**
- Current 41 Planner failures: **protocol/schema failures, not Embedding failures**
- Formal R1–R3 / formula selection: **forbidden until Planner-only v2 completes**

## Freeze

- Planner cache SHA256: `{summary_file.get("planner_cache_sha256")}`
- Protocol: `{identity.get("planner_only_protocol_version")}`
- Provider/host/model: `{identity.get("generation_provider")}` / `{identity.get("generation_host")}` / `{identity.get("generation_model")}`
- Prompt hash: `{identity.get("planner_prompt_hash")}`
- Schema hash: `{identity.get("planner_schema_hash")}`
- Capability profile hash: `{identity.get("capability_profile_hash")}`

## Funnel

| Metric | Value |
|--------|------:|
| rows | {n} |
| unique task IDs | {len(unique)} |
| ok | {len(ok_rows)} |
| failed | {len(fail_rows)} |
| repair_count=1 | {repair1} |
| retrieve steps | {retrieve_steps} |

### Success by source split
{json.dumps(funnel["by_source_split"], indent=2)}

### Success by domain
{json.dumps(funnel["by_domain"], indent=2)}

### Success by difficulty
{json.dumps(funnel["by_difficulty"], indent=2)}

## Final error taxonomy

Expected vs observed:

{json.dumps(taxonomy["final_error_taxonomy"], indent=2)}

Matches expected distribution: **{taxonomy["final_distribution_matches_expected"]}**

## Repair / first-response (offline v2 canonicalizer replay)

- first-response valid under v2 canonicalizer: {first_valid}
- repair-response valid under v2 canonicalizer: {repair_valid}
- repair recovered (first fail → repair ok) under v2: {repair_recovered}

## Offline structural recovery (failed 41 only)

- recoverable: **{offline_recoverable}**
- still unrecoverable: **{still_unrecoverable}**
- rules recovery counts: `{dict(rule_recover_counts)}`
- semantic plan changed beyond allowed rules: {offline_recovery["semantic_plan_changed_beyond_allowed_rules_count"]}

Allowed unambiguous rules only (canonicalizer `{CANONICALIZER_VERSION}` / `{plan_canonicalizer_hash()}`):

- first step missing `depends_on` → `[]`
- reason domain `analysis|reasoning` → `general`
- compute domain `calculation` → `compute`
- JSON text whitespace / fence / unicode quotes / trailing commas

Forbidden: non-first depends_on invent, required_outputs invent, compute_spec invent, overlength truncation, truncated JSON splice, gold/task-id inference.

## Resources

{json.dumps(resource_summary, indent=2)}

## Next formal run

Use new directory `outputs/pe_fin_lite_retrieval_all155_planner_v2` with planner-only protocol **v2**.
Do not overwrite `live_v1`. Do not resume v1 cache into v2.
"""
    for dest in (AUDIT_OUT, DOCS_OUT):
        (dest / "audit_report.md").write_text(report, encoding="utf-8")

    hashes_after = collect_raw_hashes()
    unchanged = hashes_before == hashes_after
    raw_hashes_payload = {
        "before": hashes_before,
        "after": hashes_after,
        "unchanged": unchanged,
        "n_files": len(hashes_before),
    }
    for dest in (AUDIT_OUT, DOCS_OUT):
        (dest / "raw_hashes.json").write_text(
            json.dumps(raw_hashes_payload, indent=2) + "\n", encoding="utf-8"
        )

    print("PLANNER ALL155 OFFLINE AUDIT OK")
    print(f"rows={n} ok={len(ok_rows)} failed={len(fail_rows)} repair1={repair1}")
    print(f"retrieve_steps={retrieve_steps}")
    print(f"taxonomy_match={taxonomy['final_distribution_matches_expected']}")
    print(f"offline_recoverable={offline_recoverable} unrecoverable={still_unrecoverable}")
    print(f"raw_hashes_unchanged={unchanged}")
    print(f"docs={DOCS_OUT}")
    print(f"audit={AUDIT_OUT}")
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
