# Financial ETOM Taxonomy — Read-only Audit

**Scope:** frozen Financial Development (`financial_real_mcp_coverage_v1.jsonl`) and Held-out (`financial_test_30_v1_1.jsonl`)  
**Action:** report only — **no** Financial catalog / question / gold / frozen-file edits  
**No Agent / LLM / embedding run in this audit**

## Method

For each labeled L3/L4/L5 item, inspect `gold_operations`, `gold_server_ids`, and task wording against strict ETOM:

- L3: same server, ≥2 operations, real output→input dependency
- L4: ≥2 servers, ≥2 operations, compositional dependency
- L5: capability gap / out-of-scope (not merely hard arithmetic)

## Findings summary

| Issue class | Severity | Count (approx.) |
| --- | --- | --- |
| L3 labeled but single-operation + local arithmetic | high | several held-out + some Dev |
| L3 labeled with ≥2 same-server ops but O→I unclear / parallel retrieve | medium | some Dev/held-out |
| L4 labeled with true AKTools + mcp-hkexnews | OK / likely exact | multiple |
| L4 labeled while taxonomy text still says “L2 analogue” | low (metadata inconsistency) | Dev fin_cf_005/006 |
| L5 true capability / period / segment gaps | OK | multiple |
| L5 temporal / clarification robustness (may be policy rather than missing tool) | medium | Dev temporal/clarify items |

## Development (`financial_real_mcp_coverage_v1.jsonl`)

### Likely OK cross-server L4

Examples with `aktools.*` + `hkexnews.*` in `gold_server_ids` / `gold_operations`:

- `fin_cf_005`, `fin_cf_006`, `fin_cf_021`, `fin_cf_027`, `fin_cf_030`

These satisfy **server_count ≥ 2**. Whether every item has a true **output→input** dependency (vs parallel corroboration) should be reviewed before claiming exact ETOM L4; corroboration pairs are often adapted L4 rather than sequential composition.

### L3 risk: calculation difficulty ≠ orchestration

- `fin_cf_014`, `fin_cf_015`: labeled L3 with primarily `hkexnews.keyword_search` (+ arithmetic taxonomy). If only one operation feeds local math, this mirrors the Market mistake (project difficulty ≠ ETOM L3).
- `fin_cf_007`, `fin_cf_023`: two AKTools ops (`analysis_indicator` + `report_em`) — plausible adapted/exact L3 **only if** one op’s output is required as input to the next; if both are independent retrieves then local divide, mark `not_applicable` / adapted.

### L5

- Unsupported metric / temporal hard negatives / clarification items exist (`fin_cf_008/009/017/024/025`). Confirm each is true **capability gap** vs temporal policy / ambiguity (same distinction applied in Market v1.1).

## Held-out (`financial_test_30_v1_1.jsonl`)

### Clear L3 overclaim risk (single structured op + local calc)

- `fin_ht_005`, `fin_ht_006`, `fin_ht_025`, `fin_ht_026`: `etom_level=L3` with `gold_operations` effectively a **single** AKTools indicator path and task_type `multi_step_calculation` / `period_comparison_calculation` / `multi_metric_retrieval`. Under strict ETOM, these are **project-difficulty / local reasoning**, not intra-server tool chaining.

### Plausible same-server L3 (needs O→I check)

- `fin_ht_015`, `fin_ht_016`: `hkexnews.keyword_search` + `hkexnews.get_announcement_pages` — good L3 candidates **if** keyword hit IDs feed page fetch.

### Likely cross-server L4 (good candidates)

- `fin_ht_007`, `fin_ht_008`, `fin_ht_017`, `fin_ht_018`, `fin_ht_027`, `fin_ht_028`: AKTools + mcp-hkexnews. Prefer documenting `has_output_input_dependency` explicitly in a future Financial taxonomy pass (read-only here).

### L5

- Segment / period / unsupported metric / unsafe comparison refusals (`fin_ht_009/010/019/020/029/030`) generally look like true catalog gaps or policy refusals; split `capability_gap` vs `temporal_policy` / `unsafe_substitution` in a future pass without changing golds.

## Recommendations (do not apply in this pass)

1. Introduce Market-style fields (`project_difficulty_level`, `orchestration_scope`, `etom_alignment`, `has_output_input_dependency`) on Financial items in a **future v1.2 taxonomy correction**, without silent overwrite of frozen held-out.
2. Stop calling single-op arithmetic “ETOM L3”.
3. Keep true AKTools↔HKEXnews items as the Financial L4 backbone; Cross-domain can still add harder composition.

## Confirmation

- Financial catalog YAML: **not modified**
- Financial JSONL golds: **not modified**
- This document is the sole Financial deliverable of the Market taxonomy correction branch pass
