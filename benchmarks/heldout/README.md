# Held-out benchmarks

## Financial

| File | Status | Notes |
|---|---|---|
| `financial_test_30_v1.jsonl` | **`superseded_before_any_agent_run`** | Candidate freeze at `f85d53f`. Superseded because 14 `evidence_answerable` items cited formal Annual/Interim **Reports** and report PDF pages while `gold_operations` used **mcp-hkexnews v0.1**, which only supports results/performance **announcements**. No agent/LLM run was ever executed on this file. |
| `financial_test_30_v1_1.jsonl` | **`active`** | Provenance-audited held-out Test 30. |

## Market

| File | Status | Notes |
|---|---|---|
| `market_test_20_v1.jsonl` | **`superseded_before_any_agent_run`** | Initial Market freeze at `c91e910`. Superseded for **taxonomy/provenance/schema** correction (package name, blocked FTShare IDs, ETOM vs project-difficulty separation, adjustment_mode schema) **before any Agent/LLM/embedding run**. Numerical gold unchanged; retained as provenance trail. |
| `market_test_20_v1_1.jsonl` | **`active`** | Taxonomy-corrected Market held-out 20. Locked/`final_evaluation_only`. Catalog: `configs/operations/market_real_mcp_v1_1.yaml`. |

## Why Market v1 was superseded

v1 incorrectly:

1. Labeled the Python package as `mcp-server-aktools` instead of `aktools==0.1.4`
2. Used non-candidate blocked FTShare aliases instead of `ftshare.ft_v2_hk_hk_candlesticks` / `ftshare.ft_v1_hk_hsi_daily_weight`
3. Treated local calculation difficulty as ETOM L3/L4 despite a single-server Market catalog

v1.1 keeps the same questions, membership, and numerical gold; it adds strict ETOM fields and project difficulty distribution.

## Validation

- Market: `scripts/validate_market_benchmark_v1_1.py`, `tests/test_market_benchmark_v1_1.py`
- Financial: `scripts/validate_financial_heldout_test_v1_1.py`

Do not expose gold / provenance fields to agent prompts.
