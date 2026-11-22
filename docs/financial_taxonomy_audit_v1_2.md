# Financial Taxonomy Audit v1.2

**Scope:** Development `financial_real_mcp_coverage_v1` → `v1_2`; Held-out `financial_test_30_v1_1` → `v1_2`  
**Catalog:** frozen Financial Real-MCP **unchanged**  
**Numerical gold / question text:** unchanged  
**No Agent / LLM / embedding / paid API / Operation-Visible run in this pass**

## Historical run audit

| Artifact | Evidence | Conclusion |
| --- | --- | --- |
| `financial_mcp_free_30_v1.jsonl` | Full/smoke runs under `outputs/full_*_30_v1`, `outputs/real_smoke_*`, `outputs/mcp_free_financial_30_v1` with `fin_cf_*` predictions | **Agent-run** (controlled-local / mcp-free / flat / OV pilots) |
| `financial_real_mcp_coverage_v1.jsonl` | Same `fin_cf_*` IDs, identical question + `gold_values` hash as mcp_free file; referenced in Real-MCP smoke `dataset_note` | Same Development content → treat as **nonformal pilot exposure** |
| `financial_test_30_v1_1.jsonl` | `fin_ht_*` **absent** from all `predictions.jsonl`; only provenance audit under `outputs/financial_heldout_provenance_audit_v1_1/` | **Never Agent-run** |

### Supersession labels (evidence-based)

- Development prior file: **`taxonomy_corrected_after_nonformal_pilot`** (do **not** claim `superseded_before_any_agent_run`)
- Held-out prior file (`v1_1`): **`superseded_before_any_agent_run`**
- Active files: `*_v1_2.jsonl` with `supersession_status=active`

## Policy

Project difficulty captures financial reasoning complexity. Strict ETOM level captures tool-orchestration topology. They are related but not interchangeable.

## Project difficulty (former L* slots)

| Split | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- |
| Development | 10 | 6 | 4 | 5 | 5 |
| Held-out | 6 | 6 | 6 | 6 | 6 |

## Strict exact ETOM counts (Dev+Held)

| Level | Count |
| --- | --- |
| L1 | 7 |
| L2 | 6 |
| L3 | 12 |
| L4 | 2 |
| L5 | 4 |

## Alignment

| Alignment | Count |
| --- | --- |
| exact | 31 |
| adapted | 10 |
| not_applicable | 19 |

## Robustness subtype

| Subtype | Count |
| --- | --- |
| not_applicable | 49 |
| capability_gap | 4 |
| temporal_policy | 4 |
| ambiguous_request | 2 |
| unsafe_substitution | 1 |

## True exact L3 IDs

`fin_cf_003`, `fin_cf_010`, `fin_cf_011`, `fin_cf_013`, `fin_cf_016`, `fin_cf_020`, `fin_ht_011`, `fin_ht_012`, `fin_ht_013`, `fin_ht_014`, `fin_ht_015`, `fin_ht_016`

## True exact L4 IDs

`fin_ht_017`, `fin_ht_018` (profile → HKEX compositional cross-server DAG)

## Rejected / downgraded former L3

| ID | Reason |
| --- | --- |
| fin_cf_007, fin_cf_014, fin_cf_015, fin_cf_023 | single/independent retrieve + local arithmetic |
| fin_ht_005, fin_ht_006, fin_ht_025, fin_ht_026 | single operation multi-field / local calc |

## Rejected / downgraded former L4

| ID | New status | Reason |
| --- | --- | --- |
| fin_cf_005, fin_cf_006, fin_cf_021, fin_ht_007, fin_ht_027 | adapted L4 | parallel cross-server basis pair, no compositional edge |
| fin_cf_027, fin_cf_030 | not_applicable | values in-question; local arithmetic only |
| fin_ht_008, fin_ht_028 | not_applicable | parallel corroboration of the same figure |

## Exact L5 (capability_gap only)

`fin_cf_017`, `fin_ht_009`, `fin_ht_019`, `fin_ht_029`
