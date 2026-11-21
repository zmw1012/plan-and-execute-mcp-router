"""PE-Fin RePlan-7 HK-155 final offline audit (read-only, zero-network)."""

AUDIT_VERSION = "pe_fin_replan_r7_hk155_final_audit_v1"
RESULT_ROLE = "posthoc_exploratory_offline_descriptive"
DEFAULT_AUDIT_SUBDIR = "final_audit_v1"
EXPECTED_N = 155
EXPECTED_DOMAIN_COUNTS = {
    "financial": 60,
    "market": 35,
    "news": 35,
    "cross_domain": 25,
}
EXPECTED_METHOD = "pe_fin_replan_v1_4_1"
EXPECTED_PROTOCOL = "planner_executor_replanner_contract_v4_1"
EXPECTED_CONDITION = "pe_fin_replan"
EXPECTED_ARCHITECTURE = "plan_execute_replan"
EXPECTED_ROUND_CAP = 7
EXPECTED_RUN_COMMIT = "84447cf54b4ac8b98d94818c828dd7958a2c1882"
BENCHMARK_CUTOFF = "2026-08-12"
EXPECTED_FORMULA_NAME = "R5_global_lexical@0.12"
EXPECTED_FORMULA_HASH = (
    "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"
)
EXPECTED_INDEX_IDENTITY = (
    "ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df"
)
EXPECTED_PRED_SHA = (
    "736f6d4b064a9dafdba17d37e147f0e57028e120050f67fa346ba52da180a4c7"
)
EXPECTED_META_SHA = (
    "e820627cb9363b3bcaaece71733d8e145187d892d7f8713828e5a38a458d609c"
)

RAW_NAMES = [
    "predictions.jsonl",
    "run_metadata.json",
    "cost_summary.json",
    "error_summary.json",
    "grading_summary.json",
    "grouped_aggregates.json",
    "manual_grading.csv",
    "pe_fin_lite_metrics.json",
    "per_task_metrics.csv",
    "retrieval_analysis.json",
    "step_funnel_summary.json",
]

__all__ = [
    "AUDIT_VERSION",
    "RESULT_ROLE",
    "DEFAULT_AUDIT_SUBDIR",
    "EXPECTED_N",
    "RAW_NAMES",
]
