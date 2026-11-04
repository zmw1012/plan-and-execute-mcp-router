# Held-out benchmarks

## Financial

| File | Status | Notes |
|---|---|---|
| `financial_test_30_v1.jsonl` | **`superseded_before_any_agent_run`** | Candidate freeze at `f85d53f`. Superseded for evidence-document provenance (report vs results announcement). No agent/LLM run. |
| `financial_test_30_v1_1.jsonl` | **`superseded_before_any_agent_run`** | Provenance-audited golds. Superseded by taxonomy v1.2 **before any Agent evaluation** (`fin_ht_*` absent from all `predictions.jsonl`). Numerical gold retained as trail. |
| `financial_test_30_v1_2.jsonl` | **`superseded_before_any_agent_run`** | Taxonomy dual-label v1.2. Superseded by v1.3 executable dependency-graph correction **before any Agent/LLM run**. Numerical gold identical to v1.1. |
| `financial_test_30_v1_3.jsonl` | **`active`** | Field-level dependency graphs; `heldout_agent_run_status=never_agent_run`. Strict exact L4 = 0. |

## Market

| File | Status | Notes |
|---|---|---|
| `market_test_20_v1.jsonl` | **`superseded_before_any_agent_run`** | Initial Market freeze. Superseded for taxonomy/provenance/schema correction before any Agent run. |
| `market_test_20_v1_1.jsonl` | **`active`** | Taxonomy-corrected Market held-out 20. Catalog: `market_real_mcp_v1_1.yaml`. |

## Financial Development (non-heldout, for context)

| File | Status |
|---|---|
| `../questions/financial_real_mcp_coverage_v1.jsonl` | **`taxonomy_corrected_after_nonformal_pilot`** (same `fin_cf_*` content used in mcp-free/flat/OV pilots via `financial_mcp_free_30_v1.jsonl`) |
| `../questions/financial_real_mcp_coverage_v1_2.jsonl` | **`superseded_before_any_agent_run`** (v1.2 never Agent/LLM-run; superseded by v1.3) |
| `../questions/financial_real_mcp_coverage_v1_3.jsonl` | **`active`** Development taxonomy v1.3 (keeps `development_exposure_history` for pre-v1.2 nonformal pilot) |

## Validation

- Financial taxonomy v1.3: `scripts/validate_financial_taxonomy_v1_3.py`, `tests/test_financial_taxonomy_v1_3.py`
- Financial taxonomy v1.2 (historical): `scripts/validate_financial_taxonomy_v1_2.py`, `tests/test_financial_taxonomy_v1_2.py`
- Market: `scripts/validate_market_benchmark_v1_1.py`
- HKEXnews dependency smoke evidence: `docs/financial_hkexnews_dependency_smoke_v1/`

Do not expose gold / provenance fields to agent prompts.
