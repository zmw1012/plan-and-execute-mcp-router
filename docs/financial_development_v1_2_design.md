# Financial Development Coverage v1.2 Design

- **Active file:** `benchmarks/questions/financial_real_mcp_coverage_v1_2.jsonl`
- **Prior file:** `financial_real_mcp_coverage_v1.jsonl`
- **Prior status:** `taxonomy_corrected_after_nonformal_pilot`
- **Numerical gold / questions:** unchanged vs v1
- **Catalog:** frozen `financial_real_mcp.yaml` unchanged

## Why not `superseded_before_any_agent_run`

`fin_cf_*` Development content (same IDs/questions/gold_values as `financial_mcp_free_30_v1.jsonl`) was used in completed nonformal Agent pilots (`outputs/full_*_30_v1`, smoke runs). Taxonomy is corrected after those pilots.

## Project difficulty distribution

D1:10 / D2:6 / D3:4 / D4:5 / D5:5

Project difficulty captures financial reasoning complexity. Strict ETOM level captures tool-orchestration topology. They are related but not interchangeable.

## Strict ETOM highlights

- Several former “L1” HKEX evidence paths reclassified to **exact L3** (list→download/keyword chains).
- Former calculation “L3” items → `etom_level=null`.
- Cross-server basis pairs → **adapted L4**, not exact.
- In-question arithmetic “L4” (`fin_cf_027/030`) → `not_applicable`.
