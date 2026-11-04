#!/usr/bin/env python3
"""Build Financial taxonomy v1.3 from v1.2 with executable field-level dependency graphs.

No LLM. Numerical gold unchanged. Free HKEXnews MCP smoke evidence required for live edges.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEV_SRC = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_2.jsonl"
HELD_SRC = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_2.jsonl"
DEV_OUT = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_3.jsonl"
HELD_OUT = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_3.jsonl"
SMOKE = "docs/financial_hkexnews_dependency_smoke_v1/observed_dependency_edges.json"

LIST = "hkexnews.list_hkexnews_announcements"
DL = "hkexnews.download_hkexnews_announcement"
PARSE = "hkexnews.parse_hkexnews_announcement"
KW = "hkexnews.keyword_search"
PAGES = "hkexnews.get_announcement_pages"

HKEX_CORE_OPS = [LIST, DL, PARSE, KW]
HKEX_PAGES_OPS = [LIST, DL, PARSE, KW, PAGES]
L3_CORE = {
    "fin_cf_003",
    "fin_cf_010",
    "fin_cf_011",
    "fin_cf_013",
    "fin_cf_016",
    "fin_cf_020",
    "fin_ht_013",
}
L3_PAGES = {"fin_ht_011", "fin_ht_012", "fin_ht_014", "fin_ht_015", "fin_ht_016"}
L4_ADAPTED = {"fin_ht_017", "fin_ht_018"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def edge(prod: str, pout: str, cons: str, cin: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "producer_operation": prod,
        "producer_output_field": pout,
        "consumer_operation": cons,
        "consumer_input_field": cin,
        "live_verified_dependency": True,
        "smoke_evidence": SMOKE,
    }
    d.update(extra)
    return d


CORE_EDGES = [
    edge(LIST, "local_key", DL, "local_key"),
    edge(LIST, "news_id", DL, "news_id"),
    edge(LIST, "release_date", DL, "release_date"),
    edge(
        LIST,
        "title",
        DL,
        "title_keyword",
        note="list.title used to form download.title_keyword",
    ),
    edge(LIST, "stock_code", DL, "stock_code"),
    edge(
        DL,
        "local_key",
        PARSE,
        "local_key",
        note="download hint requires parse before search",
    ),
    edge(
        PARSE,
        "local_key",
        KW,
        "local_key",
        note="keyword_search fails with Parsed file not found unless parse ran",
    ),
]

PAGE_EDGES = CORE_EDGES + [
    edge(PARSE, "local_key", PAGES, "local_key"),
    edge(
        KW,
        "page_number",
        PAGES,
        "start_page",
        note="keyword_search matches[].page_number → get_announcement_pages.start_page",
    ),
]


def gold_core(row: Dict[str, Any]) -> Any:
    return (
        row.get("id"),
        row.get("question"),
        row.get("reference_answer"),
        json.dumps(row.get("gold_values"), sort_keys=True, ensure_ascii=False),
        row.get("gold_status"),
    )


def dedupe(ops: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for o in ops:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def transform(
    row: Dict[str, Any],
    *,
    prior_status: str,
    supersedes: str,
    development_exposure_history: Optional[str] = None,
) -> Dict[str, Any]:
    out = deepcopy(row)
    qid = out["id"]
    out["benchmark_schema_version"] = "financial_v1_3"
    out["taxonomy_version"] = "financial_taxonomy_v1_3"
    out["supersession_status"] = "active"
    out["supersedes"] = supersedes
    # v1.2 itself had no Agent/LLM experiment; both splits use this prior label.
    out["prior_file_status"] = prior_status
    if development_exposure_history is not None:
        out["development_exposure_history"] = development_exposure_history
    out["heldout_agent_run_status"] = (
        "never_agent_run" if qid.startswith("fin_ht_") else "not_heldout"
    )
    out["dependency_schema_version"] = "field_level_v1"
    out["hkexnews_dependency_smoke"] = "docs/financial_hkexnews_dependency_smoke_v1"

    if qid in L3_CORE:
        out["gold_operations"] = list(HKEX_CORE_OPS)
        out["gold_server_ids"] = ["mcp_hkexnews"]
        out["operation_count"] = len(HKEX_CORE_OPS)
        out["server_count"] = 1
        out["has_output_input_dependency"] = True
        out["dependency_graph"] = CORE_EDGES
        out["orchestration_scope"] = "multi_operation_same_server"
        out["etom_level"] = "L3"
        out["etom_alignment"] = "exact"
        out["etom_alignment_reason"] = (
            "Intra-server executable chain list→download→parse→keyword_search with "
            "live-verified field edges."
        )
        out["acceptable_tool_sets"] = [list(HKEX_CORE_OPS)]
        out["robustness_subtype"] = "not_applicable"
    elif qid in L3_PAGES:
        out["gold_operations"] = list(HKEX_PAGES_OPS)
        out["gold_server_ids"] = ["mcp_hkexnews"]
        out["operation_count"] = len(HKEX_PAGES_OPS)
        out["server_count"] = 1
        out["has_output_input_dependency"] = True
        out["dependency_graph"] = PAGE_EDGES
        out["orchestration_scope"] = "multi_operation_same_server"
        out["etom_level"] = "L3"
        out["etom_alignment"] = "exact"
        out["etom_alignment_reason"] = (
            "Intra-server executable chain including page_number→start_page "
            "(live-verified)."
        )
        out["acceptable_tool_sets"] = [list(HKEX_PAGES_OPS)]
        out["robustness_subtype"] = "not_applicable"
    elif qid in L4_ADAPTED:
        profile_ops = [
            o for o in (out.get("gold_operations") or []) if o.startswith("aktools.")
        ]
        ops = dedupe(
            profile_ops
            + (HKEX_PAGES_OPS if qid == "fin_ht_018" else HKEX_CORE_OPS)
        )
        out["gold_operations"] = ops
        out["gold_server_ids"] = ["aktools", "mcp_hkexnews"]
        out["operation_count"] = len(ops)
        out["server_count"] = 2
        out["has_output_input_dependency"] = False
        out["dependency_graph"] = []
        out["orchestration_scope"] = "cross_server"
        out["etom_level"] = "L4"
        out["etom_alignment"] = "adapted"
        out["etom_alignment_reason"] = (
            "AKTools profile and HKEXnews evidence answer distinct sub-questions; "
            "question already names HSBC and period. No AKTools output field is passed "
            "into an HKEXnews input parameter. Not exact ETOM L4."
        )
        out["acceptable_tool_sets"] = [ops]
        out["robustness_subtype"] = "not_applicable"
    else:
        graph = out.get("dependency_graph") or []
        if any(
            isinstance(e, dict) and "producer_output_field" not in e for e in graph
        ):
            out["dependency_graph"] = []
            out["has_output_input_dependency"] = False

    out["taxonomy"] = (
        f"Financial Real-MCP project_difficulty={out.get('project_difficulty_level')}; "
        f"etom_level={out.get('etom_level')}; etom_alignment={out.get('etom_alignment')}"
    )
    out["provenance_audit_note"] = (
        "v1.3 executable dependency-graph correction; numerical gold unchanged; "
        "free HKEXnews MCP smoke verified field edges."
    )
    return out


def main() -> int:
    for src, dst, prior, sup, exposure in (
        (
            DEV_SRC,
            DEV_OUT,
            "superseded_before_any_agent_run",
            str(DEV_SRC.relative_to(ROOT)),
            (
                "taxonomy_corrected_after_nonformal_pilot: Development fin_cf_* content "
                "was exposed in mcp-free/flat/OV pilots before v1.2; v1.2 itself was never "
                "Agent/LLM-run and is superseded by v1.3 before any Agent evaluation."
            ),
        ),
        (
            HELD_SRC,
            HELD_OUT,
            "superseded_before_any_agent_run",
            str(HELD_SRC.relative_to(ROOT)),
            None,
        ),
    ):
        rows = load_jsonl(src)
        out_rows = []
        for row in rows:
            new = transform(
                row,
                prior_status=prior,
                supersedes=sup,
                development_exposure_history=exposure,
            )
            if gold_core(row) != gold_core(new):
                raise SystemExit(f"gold changed for {row['id']}")
            out_rows.append(new)
        write_jsonl(dst, out_rows)
        print(f"wrote {dst} n={len(out_rows)}")

    # Mark v1.2 files superseded (metadata only; gold untouched).
    for path in (DEV_SRC, HELD_SRC):
        rows = load_jsonl(path)
        for row in rows:
            row["supersession_status"] = "superseded_before_any_agent_run"
            row["superseded_by"] = (
                "benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl"
                if "coverage" in path.name
                else "benchmarks/heldout/financial_test_30_v1_3.jsonl"
            )
        write_jsonl(path, rows)
        print(f"marked superseded {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
