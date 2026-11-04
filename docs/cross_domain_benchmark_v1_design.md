# Cross-domain Benchmark v1 Design

**Branch:** `feature/cross-domain-benchmark-v1`  
**Base:** News freeze `508adcb`  
**Manifest:** `configs/operations/cross_domain_real_mcp_v1.yaml` (references frozen domain catalogs; does not copy/modify them)  
**Cutoff:** 2026-08-12 Asia/Hong_Kong  

## Splits

| Split | File | N |
|---|---|---|
| Development | `benchmarks/questions/cross_domain_development_10_v1.jsonl` | 10 |
| Held-out | `benchmarks/heldout/cross_domain_test_15_v1.jsonl` | 15 |

Held-out: `locked=true`, `heldout_agent_run_status=never_agent_run`, `benchmark_role=final_evaluation`.

## Domain combinations

| Combo | Dev | Held |
|---|---|---|
| Financial + Market | 3 | 4 |
| Financial + News | 3 | 4 |
| Market + News | 2 | 4 |
| Financial + Market + News | 2 | 3 |

## Company mix (reported)

| Split | Tencent | HSBC | Alibaba |
|---|---|---|---|
| Dev | 4 | 3 | 3 |
| Held | 5 | 5 | 5 |
| **Total** | **9** | **8** | **8** |

Slight Dev imbalance kept to preserve combo coverage without near-duplicating single-domain items.

## Dependency types

| Type | Meaning |
|---|---|
| `independent_parallel` | ≥2 domain facts; no required cross-domain field pass |
| `sequential_field_dependency` | News `release_date` → Market `stock_hk_hist` `start_date`/`end_date` |
| `mixed` | Sequential News→Market edge plus independent Financial leg |

## Exact ETOM policy

- Exact L4 only when a live-verified cross-domain field edge is **necessary**.
- Parallel multi-domain synthesis → `project_taxonomy=cross_domain_parallel_reasoning`, `exact_etom_alignment=false`.
- Currency/period/temporal conflicts → adapted robustness / refusal (not forged Exact L5).

## Gold sources

- Financial: frozen benchmark reference values (`fin_cf_*`)
- Market: free AKTools MCP smoke `docs/cross_domain_discovery_v1/market_cross_domain_smoke.json`
- News: frozen list/annual-report evidence under `docs/news_mcp_discovery_v1/`
