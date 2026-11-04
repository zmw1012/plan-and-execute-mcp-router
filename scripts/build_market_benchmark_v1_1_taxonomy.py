#!/usr/bin/env python3
"""Build Market Dev15/Held-out20 v1.1 taxonomy/schema correction (no live MCP / no LLM).

Preserves numerical gold and membership. Marks v1 as superseded_before_any_agent_run
in docs only for JSONL (files retained unchanged except optional supersession sidecar
notes live in docs/README).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEV_V1 = ROOT / "benchmarks" / "questions" / "market_development_15_v1.jsonl"
HELD_V1 = ROOT / "benchmarks" / "heldout" / "market_test_20_v1.jsonl"
DEV_V11 = ROOT / "benchmarks" / "questions" / "market_development_15_v1_1.jsonl"
HELD_V11 = ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl"

# project_difficulty_level inherits former balanced L* slots (not ETOM).
# ETOM assigned only when original orchestration semantics hold.
TAXONOMY: Dict[str, Dict[str, Any]] = {
    # ---- Development ----
    "mkt_dev_001": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool direct close retrieval via stock_hk_hist.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_002": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool volume field retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_series",
    },
    "mkt_dev_003": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool OHLC retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_004": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool HSI level retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "not_applicable_index_level",
        "adjustment_mode_reason": "HSI_index_level_has_no_equity_adjust_mode",
    },
    "mkt_dev_005": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L2",
        "etom_alignment": "adapted",
        "etom_alignment_reason": (
            "Context-aware session disambiguation (calendar date vs previous-valid "
            "trading session) among overlapping date-handling paths; single operation."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_006": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L2",
        "etom_alignment": "adapted",
        "etom_alignment_reason": (
            "Field/path disambiguation: turnover vs volume/spot distractors among "
            "functionally overlapping Market tools; still one gold operation."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_series",
    },
    "mkt_dev_007": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "repeated_same_operation",
        "operation_count": 2,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L2",
        "etom_alignment": "adapted",
        "etom_alignment_reason": (
            "Disambiguation among adjustment modes ('' vs qfq) on the same hist "
            "interface; parallel repeated invokes, not sequential O→I chaining."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": ["", "qfq"],
        "adjustment_mode_reason": "explicit_dual_mode_compare_no_auto_mix",
    },
    "mkt_dev_008": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "One stock_hk_hist series retrieval plus local deterministic percentage "
            "return; calculation difficulty ≠ ETOM L3/L4 orchestration."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_009": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "Single hist retrieval + local MA20; not intra-server sequential chaining."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_010": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "repeated_same_operation",
        "operation_count": 2,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "Two independent HSI retrieves (or one series) then local daily return; "
            "no tool output→input dependency; same single server."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "not_applicable_index_level",
        "adjustment_mode_reason": "HSI_index_level_has_no_equity_adjust_mode",
    },
    "mkt_dev_011": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "Annualized volatility is local deterministic calc on one hist series; "
            "not cross-server ETOM L4."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_012": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "MDD is local calc on one hist series; not ETOM L4.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_dev_013": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "multi_operation_same_server",
        "operation_count": 2,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "stock_hk_hist + stock_hk_index_daily_sina are two operations on the same "
            "AKTools server without O→I dependency (parallel then local excess return). "
            "Cannot be exact ETOM L3 or L4; Market-only catalog has one executable server."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "equity_leg_unadjusted; HSI_leg_index_na",
    },
    "mkt_dev_014": {
        "project_difficulty_level": "D5",
        "orchestration_scope": "policy_scope",
        "operation_count": 0,
        "server_count": 0,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "latest/current/today is frozen temporal-policy refusal. Active spot "
            "distractors exist, so this is NOT a capability gap / exact ETOM L5."
        ),
        "robustness_subtype": "temporal_policy",
        "adjustment_mode": None,
        "adjustment_mode_reason": "refusal_item_no_formal_price_gold",
    },
    "mkt_dev_015": {
        "project_difficulty_level": "D5",
        "orchestration_scope": "capability_gap",
        "operation_count": 0,
        "server_count": 0,
        "has_output_input_dependency": False,
        "etom_level": "L5",
        "etom_alignment": "exact",
        "etom_alignment_reason": (
            "Minute-bar gold via stock_hk_hist_min_em is deferred/unavailable under "
            "cutoff; no feasible active-catalog path for verified 1-minute gold."
        ),
        "robustness_subtype": "capability_gap",
        "adjustment_mode": None,
        "adjustment_mode_reason": "refusal_capability_gap_no_price_gold",
        "secondary_robustness_note": "FTShare path also unavailable_provider",
    },
    # ---- Held-out ----
    "mkt_ht_001": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool close retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_002": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool volume retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_series",
    },
    "mkt_ht_003": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool HSI retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "not_applicable_index_level",
        "adjustment_mode_reason": "HSI_index_level_has_no_equity_adjust_mode",
    },
    "mkt_ht_004": {
        "project_difficulty_level": "D1",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "exact",
        "etom_alignment_reason": "Foundational single-tool OHLC retrieval.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_005": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L2",
        "etom_alignment": "adapted",
        "etom_alignment_reason": "Previous-valid-day session disambiguation; single operation.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_006": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "repeated_same_operation",
        "operation_count": 2,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L2",
        "etom_alignment": "adapted",
        "etom_alignment_reason": "Adjustment-mode disambiguation ('' vs hfq); no O→I chaining.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": ["", "hfq"],
        "adjustment_mode_reason": "explicit_dual_mode_compare_no_auto_mix",
    },
    "mkt_ht_007": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L1",
        "etom_alignment": "adapted",
        "etom_alignment_reason": (
            "Single-tool OHLC retrieval with trivial local high−low; project D2 for "
            "derived field, still foundational retrieval (not ETOM L3)."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_008": {
        "project_difficulty_level": "D2",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": "L2",
        "etom_alignment": "adapted",
        "etom_alignment_reason": (
            "Ticker vs index identity disambiguation using profile tool among overlapping "
            "identity/spot/index paths."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": None,
        "adjustment_mode_reason": "identity_metadata_not_a_price_series",
    },
    "mkt_ht_009": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Local percentage return on one hist series ≠ ETOM L3.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_010": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Local MA10 on one hist series ≠ ETOM L3.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_011": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "repeated_same_operation",
        "operation_count": 2,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Independent HSI retrieves + local simple return; no O→I.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "not_applicable_index_level",
        "adjustment_mode_reason": "HSI_index_level_has_no_equity_adjust_mode",
    },
    "mkt_ht_012": {
        "project_difficulty_level": "D3",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Local sum of turnover on one hist window ≠ ETOM L3.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_series",
    },
    "mkt_ht_013": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Local ann. vol on one series ≠ cross-server ETOM L4.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_014": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Local MDD ≠ ETOM L4.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_015": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "multi_operation_same_server",
        "operation_count": 2,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "Two same-server ops without O→I; Market-only cannot be exact ETOM L4."
        ),
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "equity_leg_unadjusted; HSI_leg_index_na",
    },
    "mkt_ht_016": {
        "project_difficulty_level": "D4",
        "orchestration_scope": "single_operation",
        "operation_count": 1,
        "server_count": 1,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": "Local MA5+return combo on one series ≠ ETOM L4.",
        "robustness_subtype": "not_applicable",
        "adjustment_mode": "",
        "adjustment_mode_reason": "formal_unadjusted_equity_price",
    },
    "mkt_ht_017": {
        "project_difficulty_level": "D5",
        "orchestration_scope": "policy_scope",
        "operation_count": 0,
        "server_count": 0,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "current/right-now refusal is temporal policy; spot tools exist in active "
            "catalog, so not exact capability-gap L5."
        ),
        "robustness_subtype": "temporal_policy",
        "adjustment_mode": None,
        "adjustment_mode_reason": "refusal_item_no_formal_price_gold",
    },
    "mkt_ht_018": {
        "project_difficulty_level": "D5",
        "orchestration_scope": "capability_gap",
        "operation_count": 0,
        "server_count": 0,
        "has_output_input_dependency": False,
        "etom_level": "L5",
        "etom_alignment": "exact",
        "etom_alignment_reason": (
            "No active operation provides HKEX official trading calendar; "
            "tool_trade_date_hist_sina is an explicit non-official distractor."
        ),
        "robustness_subtype": "capability_gap",
        "adjustment_mode": None,
        "adjustment_mode_reason": "refusal_capability_gap_no_price_gold",
    },
    "mkt_ht_019": {
        "project_difficulty_level": "D5",
        "orchestration_scope": "policy_scope",
        "operation_count": 0,
        "server_count": 0,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "Post-cutoff date is temporal_policy_scope, not a missing-tool capability gap."
        ),
        "robustness_subtype": "temporal_policy",
        "adjustment_mode": None,
        "adjustment_mode_reason": "refusal_temporal_policy_no_price_gold",
    },
    "mkt_ht_020": {
        "project_difficulty_level": "D5",
        "orchestration_scope": "capability_gap",
        "operation_count": 0,
        "server_count": 0,
        "has_output_input_dependency": False,
        "etom_level": None,
        "etom_alignment": "not_applicable",
        "etom_alignment_reason": (
            "Forced blocked FTShare provider path is unavailable_provider; answerable "
            "via aktools.stock_hk_hist exists, so not exact capability-gap ETOM L5."
        ),
        "robustness_subtype": "unavailable_provider",
        "adjustment_mode": None,
        "adjustment_mode_reason": "refusal_blocked_provider_not_formal_gold",
    },
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_values_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def transform(row: Dict[str, Any]) -> Dict[str, Any]:
    qid = row["id"]
    meta = TAXONOMY[qid]
    out = deepcopy(row)
    # Preserve numerical gold / membership; fix schema + taxonomy.
    out["benchmark_schema_version"] = "market_v1_1"
    out["catalog_id"] = "market_real_mcp_v1_1"
    out["supersedes_question_file_version"] = "v1"
    out["project_difficulty_level"] = meta["project_difficulty_level"]
    out["orchestration_scope"] = meta["orchestration_scope"]
    out["operation_count"] = meta["operation_count"]
    out["server_count"] = meta["server_count"]
    out["has_output_input_dependency"] = meta["has_output_input_dependency"]
    out["etom_level"] = meta["etom_level"]
    out["etom_alignment"] = meta["etom_alignment"]
    out["etom_alignment_reason"] = meta["etom_alignment_reason"]
    out["robustness_subtype"] = meta["robustness_subtype"]
    out["adjustment_mode"] = meta["adjustment_mode"]
    out["adjustment_mode_reason"] = meta["adjustment_mode_reason"]
    if "secondary_robustness_note" in meta:
        out["secondary_robustness_note"] = meta["secondary_robustness_note"]

    # Taxonomy string should not claim forced ETOM distribution.
    d = meta["project_difficulty_level"]
    et = meta["etom_level"]
    out["taxonomy"] = (
        f"Market Real-MCP project_difficulty={d}; "
        f"etom_level={et}; etom_alignment={meta['etom_alignment']}"
    )

    # Fix ± ± typo only (content-preserving schema/typo fix).
    out["question"] = out["question"].replace("± ±", "±")

    # Sync gold_values adjustment_mode with top-level for price-like items.
    for gv in out.get("gold_values") or []:
        if meta["adjustment_mode"] == "not_applicable_index_level":
            gv["adjustment_mode"] = "not_applicable_index_level"
        elif isinstance(meta["adjustment_mode"], list):
            gv["adjustment_mode"] = list(meta["adjustment_mode"])
        elif meta["adjustment_mode"] == "":
            gv["adjustment_mode"] = ""
        elif meta["adjustment_mode"] is None and not gv:
            pass

    # L5 / refusal markers
    if meta["robustness_subtype"] in {
        "capability_gap",
        "temporal_policy",
        "unavailable_provider",
    }:
        out["confirmed_infeasible"] = True
        if meta["etom_alignment"] == "exact" and meta["etom_level"] == "L5":
            out["answerability_class"] = "should_refuse_or_clarify"
            out["gold_status"] = "unanswerable"
        out["gold_operations"] = []

    # Provenance reuse note
    out["provenance_audit_note"] = (
        "v1 provenance hashes reused because raw invocations and numerical gold are unchanged"
    )
    return out


def main() -> int:
    for src, dst in ((DEV_V1, DEV_V11), (HELD_V1, HELD_V11)):
        rows = load_jsonl(src)
        out_rows = []
        for row in rows:
            new = transform(row)
            if not gold_values_equal(row.get("gold_values"), new.get("gold_values")):
                # Allow only adjustment_mode field sync inside gold_values
                old_gv = deepcopy(row.get("gold_values") or [])
                new_gv = deepcopy(new.get("gold_values") or [])
                for a, b in zip(old_gv, new_gv):
                    a.pop("adjustment_mode", None)
                    b.pop("adjustment_mode", None)
                if not gold_values_equal(old_gv, new_gv):
                    raise SystemExit(f"Numerical gold changed for {row['id']}")
            # membership / ids
            assert new["id"] == row["id"]
            assert new["dataset_split"] == row["dataset_split"] or (
                row["dataset_split"] == "heldout" and new["dataset_split"] == "heldout"
            )
            out_rows.append(new)
        write_jsonl(dst, out_rows)
        print(f"wrote {dst} n={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
