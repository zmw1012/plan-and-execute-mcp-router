# Market Taxonomy Audit v1.1

Audit date: schema/taxonomy correction only. **No Agent / LLM / embedding / paid API run.**  
**No re-run of full live MCP provenance** — v1 provenance hashes reused because raw invocations and numerical gold are unchanged.

## Policy statement

Project difficulty levels are balanced for analysis, whereas ETOM levels are assigned only when the original orchestration semantics are satisfied. Market-only tasks are not forced into ETOM L4 because the frozen Market catalog contains one executable MCP server. Cross-server ETOM L4 is evaluated later in the cross-domain benchmark.

## ETOM original semantics (enforced)

| Level | Requirement |
| --- | --- |
| L1 | foundational single-tool direct retrieval |
| L2 | context-aware retrieval/disambiguation among overlapping tools |
| L3 | intra-server sequential chaining; ≥2 ops; true output→input dependency |
| L4 | cross-server compositional chaining; ≥2 servers; tool dependency |
| L5 | capability gap / out-of-scope under active toolset |

## Counts (Dev15 + Held-out20 = 35)

### Project difficulty

| Split | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- |
| Development | 4 | 3 | 3 | 3 | 2 |
| Held-out | 4 | 4 | 4 | 4 | 4 |

### Exact ETOM-compatible (`etom_alignment=exact`)

| Level | Count | IDs |
| --- | --- | --- |
| L1 | 8 | mkt_dev_001–004, mkt_ht_001–004 |
| L2 | 0 | — |
| L3 | 0 | — |
| L4 | 0 | — |
| L5 | 2 | mkt_dev_015, mkt_ht_018 |

### Alignment

| Alignment | Count |
| --- | --- |
| exact | 10 |
| adapted | 7 |
| not_applicable | 18 |

### Robustness subtype

| Subtype | Count |
| --- | --- |
| not_applicable | 29 |
| temporal_policy | 3 |
| capability_gap | 2 |
| unavailable_provider | 1 |

## Key reclassification decisions

1. **Single hist + local return/MA/vol/MDD** → high `project_difficulty_level`, `etom_level=null` (`not_applicable`).
2. **hist + HSI same-server dual retrieve without O→I** → not L3, not L4.
3. **Market-only catalog** → **zero exact L4**.
4. **latest/current** → `temporal_policy` (spot distractors exist), not capability gap.
5. **post-cutoff** → `temporal_policy`.
6. **forced FTShare** → `unavailable_provider` (AKTools path exists).
7. **minute hist / official HKEX calendar** → `capability_gap` exact L5.

## Adjustment-mode corrections

- Equity unadjusted: top-level `adjustment_mode=""` (never null while saying unadjusted).
- Dual compare: `["", "qfq"]` or `["", "hfq"]`.
- HSI: `not_applicable_index_level`.
- Refusals: `adjustment_mode=null` + `adjustment_mode_reason`.
- Typo `± ±0.05` → `±0.05` on mkt_dev_007 / mkt_ht_004.
