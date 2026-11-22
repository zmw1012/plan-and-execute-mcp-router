#!/usr/bin/env python3
"""Build Financial Dev/Held-out taxonomy v1.2 (metadata only; no LLM; no gold changes)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEV_SRC = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
HELD_SRC = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_1.jsonl"
DEV_OUT = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_2.jsonl"
HELD_OUT = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_2.jsonl"

# project_difficulty inherits former balanced L* slots (D1..D5).
# Strict etom_level assigned only from orchestration topology.


def T(
    *,
    d: str,
    scope: str,
    oc: int,
    sc: int,
    dep: bool,
    graph: List[Dict[str, str]],
    tools: List[List[str]],
    etom: Optional[str],
    align: str,
    reason: str,
    rob: str = "not_applicable",
) -> Dict[str, Any]:
    return {
        "project_difficulty_level": d,
        "orchestration_scope": scope,
        "operation_count": oc,
        "server_count": sc,
        "has_output_input_dependency": dep,
        "dependency_graph": graph,
        "acceptable_tool_sets": tools,
        "etom_level": etom,
        "etom_alignment": align,
        "etom_alignment_reason": reason,
        "robustness_subtype": rob,
    }


TAXONOMY: Dict[str, Dict[str, Any]] = {
    # -------------------- Development --------------------
    "fin_cf_001": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="exact",
        reason="Single AKTools analysis_indicator direct revenue retrieval.",
    ),
    "fin_cf_002": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_financial_hk_analysis_indicator_em"],
            ["aktools.stock_financial_hk_report_em"],
        ],
        etom="L2",
        align="exact",
        reason="Single-server fact with verified functionally overlapping tool set (analysis_indicator vs report_em).",
    ),
    "fin_cf_003": T(
        d="D2",
        scope="multi_operation_same_server",
        oc=4,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.download_hkexnews_announcement",
                "via": "announcement_id",
            },
            {
                "from": "hkexnews.download_hkexnews_announcement",
                "to": "hkexnews.parse_hkexnews_announcement",
                "via": "document_bytes",
            },
            {
                "from": "hkexnews.parse_hkexnews_announcement",
                "to": "hkexnews.keyword_search",
                "via": "parsed_text",
            },
        ],
        tools=[
            [
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.download_hkexnews_announcement",
                "hkexnews.parse_hkexnews_announcement",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L3",
        align="exact",
        reason="Intra-server HKEX sequential chain with announcement_id/document output→input edges; metric-basis discrimination is project difficulty D2.",
    ),
    "fin_cf_004": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_financial_hk_analysis_indicator_em"],
            ["aktools.stock_financial_hk_report_em"],
        ],
        etom="L2",
        align="exact",
        reason="Overlapping AKTools structured paths for the same interim revenue fact.",
    ),
    "fin_cf_005": T(
        d="D4",
        scope="cross_server",
        oc=2,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L4",
        align="adapted",
        reason="Cross-server IFRS vs non-IFRS basis pair is parallel complementary retrieval, not a compositional cross-server DAG.",
    ),
    "fin_cf_006": T(
        d="D4",
        scope="cross_server",
        oc=2,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L4",
        align="adapted",
        reason="Parallel cross-server multi-basis retrieval without operation-to-operation dependency edge.",
    ),
    "fin_cf_007": T(
        d="D3",
        scope="repeated_same_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_financial_hk_analysis_indicator_em"],
            ["aktools.stock_financial_hk_report_em"],
        ],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: independent metric retrieve(s) plus local division; no tool output→input edge.",
    ),
    "fin_cf_008": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Full-year FY2026 unavailable at cutoff is temporal_policy, not missing-tool capability gap.",
        rob="temporal_policy",
    ),
    "fin_cf_009": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Ambiguous period ('2026 revenue'); requires clarification, not exact L5 capability gap.",
        rob="ambiguous_request",
    ),
    "fin_cf_010": T(
        d="D1",
        scope="multi_operation_same_server",
        oc=3,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.download_hkexnews_announcement",
                "via": "announcement_id",
            },
            {
                "from": "hkexnews.download_hkexnews_announcement",
                "to": "hkexnews.keyword_search",
                "via": "document",
            },
        ],
        tools=[
            [
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.download_hkexnews_announcement",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L3",
        align="exact",
        reason="Former L1 reclassified: HKEX list→download→keyword is intra-server sequential chaining.",
    ),
    "fin_cf_011": T(
        d="D1",
        scope="multi_operation_same_server",
        oc=2,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            }
        ],
        tools=[["hkexnews.list_hkexnews_announcements", "hkexnews.keyword_search"]],
        etom="L3",
        align="exact",
        reason="List selection feeds keyword evidence read on the same server.",
    ),
    "fin_cf_012": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["hkexnews.keyword_search"]],
        etom="L2",
        align="adapted",
        reason="APM vs reported metric discrimination; single tool so not exact L2 (no alternate tool set), adapted L2.",
    ),
    "fin_cf_013": T(
        d="D1",
        scope="multi_operation_same_server",
        oc=2,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            }
        ],
        tools=[["hkexnews.list_hkexnews_announcements", "hkexnews.keyword_search"]],
        etom="L3",
        align="exact",
        reason="HKEX list→keyword sequential evidence path.",
    ),
    "fin_cf_014": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["hkexnews.keyword_search"]],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: values given in question / single-tool evidence plus local arithmetic.",
    ),
    "fin_cf_015": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["hkexnews.keyword_search"]],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: single-tool retrieve + local margin formula.",
    ),
    "fin_cf_016": T(
        d="D2",
        scope="multi_operation_same_server",
        oc=2,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            }
        ],
        tools=[["hkexnews.list_hkexnews_announcements", "hkexnews.keyword_search"]],
        etom="L3",
        align="exact",
        reason="Same-server list→keyword chain; period/metric pairing is project difficulty D2.",
    ),
    "fin_cf_017": T(
        d="D5",
        scope="capability_gap",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom="L5",
        align="exact",
        reason="Active catalog has no feasible HSBC gross-margin path; banking gross margin not a disclosed gold metric.",
        rob="capability_gap",
    ),
    "fin_cf_018": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="exact",
        reason="Single AKTools direct revenue retrieval.",
    ),
    "fin_cf_019": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_financial_hk_analysis_indicator_em"],
            ["aktools.stock_financial_hk_report_em"],
        ],
        etom="L2",
        align="exact",
        reason="Overlapping structured AKTools tools for attributable NI.",
    ),
    "fin_cf_020": T(
        d="D2",
        scope="multi_operation_same_server",
        oc=2,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            }
        ],
        tools=[["hkexnews.list_hkexnews_announcements", "hkexnews.keyword_search"]],
        etom="L3",
        align="exact",
        reason="HKEX sequential evidence chain for non-GAAP NI; basis discrimination is D2.",
    ),
    "fin_cf_021": T(
        d="D4",
        scope="cross_server",
        oc=2,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L4",
        align="adapted",
        reason="Cross-server reported vs non-GAAP EPS pair without compositional dependency edge.",
    ),
    "fin_cf_022": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_hk_company_profile_em"],
            ["aktools.stock_hk_security_profile_em"],
        ],
        etom="L2",
        align="exact",
        reason="Fiscal-calendar identity with overlapping profile tools.",
    ),
    "fin_cf_023": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_financial_hk_analysis_indicator_em"],
            ["aktools.stock_financial_hk_report_em"],
        ],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: local attributable margin on retrieved/structured fields; no O→I tool edge.",
    ),
    "fin_cf_024": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Calendar-year 2026 revenue unavailable under frozen Alibaba FY snapshot — temporal_policy.",
        rob="temporal_policy",
    ),
    "fin_cf_025": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Underspecified e-commerce profitability request — ambiguous_request.",
        rob="ambiguous_request",
    ),
    "fin_cf_026": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="exact",
        reason="Single-tool basic EPS retrieval.",
    ),
    "fin_cf_027": T(
        d="D4",
        scope="single_operation",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Rejected former L4: both IFRS and non-IFRS values are provided in the question; task is local arithmetic.",
    ),
    "fin_cf_028": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["hkexnews.keyword_search"]],
        etom="L1",
        align="exact",
        reason="Single HKEX keyword evidence retrieval for basic EPS.",
    ),
    "fin_cf_029": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["hkexnews.keyword_search"]],
        etom="L2",
        align="adapted",
        reason="APM vs reported interim revenue discrimination on one evidence tool; adapted (no alternate tool set).",
    ),
    "fin_cf_030": T(
        d="D4",
        scope="single_operation",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Rejected former L4: both figures given in-question; local comparison/percentage only.",
    ),
    # -------------------- Held-out --------------------
    "fin_ht_001": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="exact",
        reason="Single-tool interim basic EPS retrieval.",
    ),
    "fin_ht_002": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="exact",
        reason="Single-tool IFRS profit retrieval.",
    ),
    "fin_ht_003": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L2",
        align="adapted",
        reason="Non-IFRS vs IFRS EPS discrimination; single structured tool → adapted L2.",
    ),
    "fin_ht_004": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L2",
        align="adapted",
        reason="Non-IFRS profit discrimination on one tool; adapted L2.",
    ),
    "fin_ht_005": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: single operation + local margin arithmetic.",
    ),
    "fin_ht_006": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: single-tool multi-period fields + local percentage; no O→I edge.",
    ),
    "fin_ht_007": T(
        d="D4",
        scope="cross_server",
        oc=3,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.keyword_search",
                "hkexnews.get_announcement_pages",
            ]
        ],
        etom="L4",
        align="adapted",
        reason="Cross-server IFRS/non-IFRS EPS basis pair is parallel complementary evidence, not compositional DAG.",
    ),
    "fin_ht_008": T(
        d="D4",
        scope="cross_server",
        oc=3,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.keyword_search",
            ]
        ],
        etom=None,
        align="not_applicable",
        reason="Rejected former exact-L4 claim: parallel corroboration of the same revenue figure across servers.",
    ),
    "fin_ht_009": T(
        d="D5",
        scope="capability_gap",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom="L5",
        align="exact",
        reason="No active catalog operation returns Tencent VAS segment revenue.",
        rob="capability_gap",
    ),
    "fin_ht_010": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="9M2026 period outside frozen snapshot — temporal_policy, not capability_gap.",
        rob="temporal_policy",
    ),
    "fin_ht_011": T(
        d="D1",
        scope="multi_operation_same_server",
        oc=3,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            },
            {
                "from": "hkexnews.keyword_search",
                "to": "hkexnews.get_announcement_pages",
                "via": "hit_locator",
            },
        ],
        tools=[
            [
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.keyword_search",
                "hkexnews.get_announcement_pages",
            ]
        ],
        etom="L3",
        align="exact",
        reason="HKEX list→keyword→pages sequential chain.",
    ),
    "fin_ht_012": T(
        d="D1",
        scope="multi_operation_same_server",
        oc=3,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            },
            {
                "from": "hkexnews.keyword_search",
                "to": "hkexnews.get_announcement_pages",
                "via": "hit_locator",
            },
        ],
        tools=[
            [
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.keyword_search",
                "hkexnews.get_announcement_pages",
            ]
        ],
        etom="L3",
        align="exact",
        reason="Same-server sequential evidence chain for PAT.",
    ),
    "fin_ht_013": T(
        d="D2",
        scope="multi_operation_same_server",
        oc=3,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.download_hkexnews_announcement",
                "via": "announcement_id",
            },
            {
                "from": "hkexnews.download_hkexnews_announcement",
                "to": "hkexnews.keyword_search",
                "via": "document",
            },
        ],
        tools=[
            [
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.download_hkexnews_announcement",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L3",
        align="exact",
        reason="Evidence-path chaining on HKEX; preferring announcement over EM tables is D2 project difficulty.",
    ),
    "fin_ht_014": T(
        d="D2",
        scope="multi_operation_same_server",
        oc=3,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.list_hkexnews_announcements",
                "to": "hkexnews.keyword_search",
                "via": "selected_announcement",
            },
            {
                "from": "hkexnews.keyword_search",
                "to": "hkexnews.get_announcement_pages",
                "via": "hit_locator",
            },
        ],
        tools=[
            [
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.keyword_search",
                "hkexnews.get_announcement_pages",
            ]
        ],
        etom="L3",
        align="exact",
        reason="HKEX-only sequential path with explicit avoidance of East Money tables (D2).",
    ),
    "fin_ht_015": T(
        d="D3",
        scope="multi_operation_same_server",
        oc=2,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.keyword_search",
                "to": "hkexnews.get_announcement_pages",
                "via": "hit_locator",
            }
        ],
        tools=[["hkexnews.keyword_search", "hkexnews.get_announcement_pages"]],
        etom="L3",
        align="exact",
        reason="True intra-server O→I (keyword hits → page fetch); margin arithmetic is local post-processing.",
    ),
    "fin_ht_016": T(
        d="D3",
        scope="multi_operation_same_server",
        oc=2,
        sc=1,
        dep=True,
        graph=[
            {
                "from": "hkexnews.keyword_search",
                "to": "hkexnews.get_announcement_pages",
                "via": "hit_locator",
            }
        ],
        tools=[["hkexnews.keyword_search", "hkexnews.get_announcement_pages"]],
        etom="L3",
        align="exact",
        reason="Keyword→pages dependency to obtain both reported and APM figures before local subtract.",
    ),
    "fin_ht_017": T(
        d="D4",
        scope="cross_server",
        oc=3,
        sc=2,
        dep=True,
        graph=[
            {
                "from": "aktools.stock_hk_company_profile_em",
                "to": "hkexnews.keyword_search",
                "via": "year_end_constrains_period_interpretation",
            }
        ],
        tools=[
            [
                "aktools.stock_hk_company_profile_em",
                "aktools.stock_hk_security_profile_em",
                "hkexnews.keyword_search",
            ]
        ],
        etom="L4",
        align="exact",
        reason="Cross-server compositional use: profile year-end informs/gates HKEX FY revenue evidence selection.",
    ),
    "fin_ht_018": T(
        d="D4",
        scope="cross_server",
        oc=3,
        sc=2,
        dep=True,
        graph=[
            {
                "from": "aktools.stock_hk_company_profile_em",
                "to": "hkexnews.keyword_search",
                "via": "issuer_identity_gates_evidence",
            },
            {
                "from": "hkexnews.keyword_search",
                "to": "hkexnews.get_announcement_pages",
                "via": "hit_locator",
            },
        ],
        tools=[
            [
                "aktools.stock_hk_company_profile_em",
                "hkexnews.keyword_search",
                "hkexnews.get_announcement_pages",
            ]
        ],
        etom="L4",
        align="exact",
        reason="Profile identity confirmation then HKEX EPS evidence forms a cross-server compositional DAG.",
    ),
    "fin_ht_019": T(
        d="D5",
        scope="capability_gap",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom="L5",
        align="exact",
        reason="No frozen Financial catalog operation returns CET1 ratio gold.",
        rob="capability_gap",
    ),
    "fin_ht_020": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Unsafe cross-company gross-margin comparison; unsafe_substitution, not capability_gap.",
        rob="unsafe_substitution",
    ),
    "fin_ht_021": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="exact",
        reason="Single-tool reported basic EPS retrieval.",
    ),
    "fin_ht_022": T(
        d="D1",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_hk_company_profile_em"],
            ["aktools.stock_hk_security_profile_em"],
            ["aktools.stock_financial_hk_analysis_indicator_em"],
        ],
        etom="L2",
        align="exact",
        reason="FY period-end fact with multiple acceptable overlapping AKTools tool sets.",
    ),
    "fin_ht_023": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom="L1",
        align="adapted",
        reason="Row/REPORT_DATE selection within one tool is not exact L2 (no overlapping tool set); adapted foundational retrieval.",
    ),
    "fin_ht_024": T(
        d="D2",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[
            ["aktools.stock_financial_hk_analysis_indicator_em"],
            ["aktools.stock_financial_hk_report_em"],
        ],
        etom="L2",
        align="exact",
        reason="HOLDER_PROFIT field selection with overlapping analysis/report tools.",
    ),
    "fin_ht_025": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: multi-field read from one operation is not sequential tool chaining.",
    ),
    "fin_ht_026": T(
        d="D3",
        scope="single_operation",
        oc=1,
        sc=1,
        dep=False,
        graph=[],
        tools=[["aktools.stock_financial_hk_analysis_indicator_em"]],
        etom=None,
        align="not_applicable",
        reason="Rejected former L3: single-operation multi-metric label task.",
    ),
    "fin_ht_027": T(
        d="D4",
        scope="cross_server",
        oc=3,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.keyword_search",
                "hkexnews.get_announcement_pages",
            ]
        ],
        etom="L4",
        align="adapted",
        reason="Cross-server reported vs non-GAAP NI pair without required cross-server O→I edge.",
    ),
    "fin_ht_028": T(
        d="D4",
        scope="cross_server",
        oc=3,
        sc=2,
        dep=False,
        graph=[],
        tools=[
            [
                "aktools.stock_financial_hk_analysis_indicator_em",
                "hkexnews.list_hkexnews_announcements",
                "hkexnews.keyword_search",
            ]
        ],
        etom=None,
        align="not_applicable",
        reason="Rejected former exact L4: parallel corroboration of the same revenue amount.",
    ),
    "fin_ht_029": T(
        d="D5",
        scope="capability_gap",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom="L5",
        align="exact",
        reason="No active operation returns Alibaba Cloud segment revenue.",
        rob="capability_gap",
    ),
    "fin_ht_030": T(
        d="D5",
        scope="policy_scope",
        oc=0,
        sc=0,
        dep=False,
        graph=[],
        tools=[],
        etom=None,
        align="not_applicable",
        reason="Requested period outside frozen Alibaba snapshot — temporal_policy.",
        rob="temporal_policy",
    ),
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_core(row: Dict[str, Any]) -> Any:
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "gold_values": row.get("gold_values"),
        "gold_status": row.get("gold_status"),
    }


def transform(
    row: Dict[str, Any],
    *,
    prior_file_status: str,
    history_note: str,
    supersedes: str,
) -> Dict[str, Any]:
    qid = row["id"]
    meta = TAXONOMY[qid]
    out = deepcopy(row)
    out["benchmark_schema_version"] = "financial_v1_2"
    out["taxonomy_version"] = "financial_taxonomy_v1_2"
    out["supersession_status"] = "active"
    out["supersedes"] = supersedes
    out["prior_file_status"] = prior_file_status
    out["run_history_note"] = history_note
    out["project_difficulty_level"] = meta["project_difficulty_level"]
    out["orchestration_scope"] = meta["orchestration_scope"]
    out["operation_count"] = meta["operation_count"]
    out["server_count"] = meta["server_count"]
    out["has_output_input_dependency"] = meta["has_output_input_dependency"]
    out["dependency_graph"] = meta["dependency_graph"]
    out["acceptable_tool_sets"] = meta["acceptable_tool_sets"]
    out["etom_level"] = meta["etom_level"]
    out["etom_alignment"] = meta["etom_alignment"]
    out["etom_alignment_reason"] = meta["etom_alignment_reason"]
    out["robustness_subtype"] = meta["robustness_subtype"]
    out["former_etom_level"] = row.get("etom_level")
    out["taxonomy"] = (
        f"Financial Real-MCP project_difficulty={meta['project_difficulty_level']}; "
        f"etom_level={meta['etom_level']}; etom_alignment={meta['etom_alignment']}"
    )
    out["provenance_audit_note"] = (
        "Taxonomy metadata correction only; numerical gold unchanged; no live MCP re-audit."
    )
    # Keep original gold_operations for answered items; refusals stay empty.
    if meta["robustness_subtype"] in {
        "capability_gap",
        "temporal_policy",
        "ambiguous_request",
        "unsafe_substitution",
    }:
        if (
            meta["etom_alignment"] == "exact"
            and meta["etom_level"] == "L5"
        ) or meta["etom_level"] is None:
            out["gold_operations"] = []
            out["confirmed_infeasible"] = True
    return out


def main() -> int:
    missing = set(TAXONOMY) - {
        r["id"] for r in load_jsonl(DEV_SRC) + load_jsonl(HELD_SRC)
    }
    extra = {r["id"] for r in load_jsonl(DEV_SRC) + load_jsonl(HELD_SRC)} - set(TAXONOMY)
    if missing or extra:
        raise SystemExit(f"taxonomy ID mismatch missing={missing} extra={extra}")

    dev_hist = (
        "fin_cf_* Development content was used in nonformal controlled-local / mcp-free / "
        "flat / operation_visible pilots via financial_mcp_free_30_v1.jsonl (same IDs/questions/"
        "gold_values as financial_real_mcp_coverage_v1.jsonl). Marked "
        "taxonomy_corrected_after_nonformal_pilot."
    )
    held_hist = (
        "No predictions.jsonl contains fin_ht_* IDs. Held-out v1.1 had provenance audit only, "
        "never an Agent/LLM evaluation run. Marked superseded_before_any_agent_run."
    )

    for src, dst, prior_status, note, supersedes in (
        (
            DEV_SRC,
            DEV_OUT,
            "taxonomy_corrected_after_nonformal_pilot",
            dev_hist,
            "benchmarks/questions/financial_real_mcp_coverage_v1.jsonl",
        ),
        (
            HELD_SRC,
            HELD_OUT,
            "superseded_before_any_agent_run",
            held_hist,
            "benchmarks/heldout/financial_test_30_v1_1.jsonl",
        ),
    ):
        rows = load_jsonl(src)
        out_rows = []
        for row in rows:
            new = transform(
                row,
                prior_file_status=prior_status,
                history_note=note,
                supersedes=supersedes,
            )
            if gold_core(row) != gold_core(new):
                # allow taxonomy/gold_operations clearing only for refusals already empty
                if row.get("gold_values") != new.get("gold_values"):
                    raise SystemExit(f"gold_values changed for {row['id']}")
                if row.get("reference_answer") != new.get("reference_answer"):
                    raise SystemExit(f"reference_answer changed for {row['id']}")
                if row.get("question") != new.get("question"):
                    raise SystemExit(f"question changed for {row['id']}")
            out_rows.append(new)
        write_jsonl(dst, out_rows)
        print(f"wrote {dst} n={len(out_rows)} prior_file_status={prior_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
