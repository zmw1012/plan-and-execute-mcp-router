# Financial 30Q Real-MCP Answerability Audit

**Branch:** `feature/real-financial-mcp-discovery`  
**Catalog:** `configs/operations/financial_real_mcp.yaml` (AKTools + mcp-hkexnews)  
**Source questions (unchanged):** `benchmarks/questions/financial_mcp_free_30_v1.jsonl`  
**Annotated coverage:** `benchmarks/questions/financial_real_mcp_coverage_v1.jsonl`  
**Gold snapshots:** `data/mock/financial.json` (reconciliation only — never agent tool data)

No paid LLM was used. Reference answers were **not** edited to match tool output.

### Dataset role (important)

Every coverage row sets:

- `dataset_split: development`
- `benchmark_role: method_development_and_error_analysis`

These 30 questions **may** be used in the formal evaluation *pipeline* and for method development / error analysis.  
They are **not** a held-out final test set, because they already participated in tool discovery and system design.

---

## 1. Answerability class counts

| Class | Count | Meaning |
|-------|------:|---------|
| `structured_answerable` | **10** | Curated AKTools numeric path reconciles (or safely computes) without relying on unreliable HSBC absolutes |
| `evidence_answerable` | **15** | Requires mcp-hkexnews page/text evidence (non-IFRS/non-GAAP, HSBC IR figures, APMs) |
| `intentionally_infeasible` | **5** | Shared pool correctly cannot answer; tests abstain/clarify |
| `unsupported_invalid` | **0** | None — every item has a refuse path or a real MCP path |

Answerability classes were **not** changed in the L3/L4 fix.

## 2. ETOM level distribution (corrected)

| Level | Count | Definition used |
|-------|------:|-----------------|
| L1 | 10 | Explicit single-tool call |
| L2 | 6 | Implicit / equivalent tool choice |
| L3 | 4 | Multi-step chain **within one server** |
| L4 | 5 | **Cross-server** call chain (AKTools + mcp-hkexnews) |
| L5 | 5 | Infeasible / refuse / clarify |

### Cross-server → L4

These require both `aktools` and `mcp_hkexnews` in `gold_server_ids` and are labeled **L4**:

- `fin_cf_005`
- `fin_cf_006`
- `fin_cf_021`
- `fin_cf_027`
- `fin_cf_030`

## 3. Formal-test eligibility

| Eligible | Count |
|----------|------:|
| `formal_test_eligible=true` | 30 |
| `formal_test_eligible=false` | 0 |

Usable under the annotated class for development evaluation — **not** held-out final test.

### Items that must **not** be graded as ordinary structured numeric failures

| ID | Reason |
|----|--------|
| fin_cf_008, 009, 017, 024, 025 | Intentionally infeasible (unavailable / clarify / bank gross margin) |
| fin_cf_010–016, 028–029 | HSBC — AKTools absolute amounts unreliable; evidence path only |
| fin_cf_003, 005, 006, 020, 021, 027, 030 | non-IFRS / non-GAAP components need HKEX evidence |

## 4. Rules applied

1. AKTools ≠ automatic gold when disagreeing with `data/mock/financial.json`.  
2. HSBC absolute revenue/PAT/EPS **never** `structured_answerable`.  
3. non-IFRS/non-GAAP → `evidence_answerable` only (HKEX PDF keywords/pages).  
4. Refusal/clarify benchmarks → `intentionally_infeasible`.  
5. Cross-server gold paths → ETOM **L4** (not L3).  
6. No `unsupported_invalid` in this set because each question either has a real path or an intentional refuse target.

## 5. Per-question summary

See `benchmarks/questions/financial_real_mcp_coverage_v1.jsonl` for full fields including  
`etom_level`, `answerability_class`, `dataset_split`, `benchmark_role`, `gold_server_ids`, `gold_operations`, `required_capabilities`, `expected_tool_path`, `known_data_risks`, `formal_test_eligible`, `exclusion_reason`.

## 6. Shared catalog boundary (L5)

No catalog operations for: segments, CET1/NIM/RoTE, numeric non-GAAP, HSBC IR-faithful structured USD. Agents must abstain or use filing evidence with uncertainty — not invent values.
