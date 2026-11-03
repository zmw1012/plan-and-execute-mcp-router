# Financial 30Q Real-MCP Answerability Audit

**Branch:** `feature/real-financial-mcp-discovery`  
**Catalog:** `configs/operations/financial_real_mcp.yaml` (AKTools + mcp-hkexnews)  
**Source questions (unchanged):** `benchmarks/questions/financial_mcp_free_30_v1.jsonl`  
**Annotated coverage:** `benchmarks/questions/financial_real_mcp_coverage_v1.jsonl`  
**Gold snapshots:** `data/mock/financial.json` (reconciliation only — never agent tool data)

No paid LLM was used. Reference answers were **not** edited to match tool output.

---

## 1. Answerability class counts

| Class | Count | Meaning |
|-------|------:|---------|
| `structured_answerable` | **10** | Curated AKTools numeric path reconciles (or safely computes) without relying on unreliable HSBC absolutes |
| `evidence_answerable` | **15** | Requires mcp-hkexnews page/text evidence (non-IFRS/non-GAAP, HSBC IR figures, APMs) |
| `intentionally_infeasible` | **5** | Shared pool correctly cannot answer; tests abstain/clarify |
| `unsupported_invalid` | **0** | None — every item has a refuse path or a real MCP path |

## 2. ETOM level distribution

| Level | Count |
|-------|------:|
| L1 | 10 |
| L2 | 6 |
| L3 | 9 |
| L4 | 0 |
| L5 | 5 |

## 3. Formal-test eligibility

| Eligible | Count |
|----------|------:|
| `formal_test_eligible=true` | 30 |
| `formal_test_eligible=false` | 0 |

All 30 remain usable as formal tests **under the annotated class**:

- Structured items grade against verified mock snapshots when MCP reconciles.  
- Evidence items grade on evidence process / extracted figures from filings, **not** East Money HSBC absolutes.  
- Intentionally infeasible items grade abstention/clarification.

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
5. No `unsupported_invalid` in this set because each question either has a real path or an intentional refuse target.

## 5. Per-question summary

See `benchmarks/questions/financial_real_mcp_coverage_v1.jsonl` for full fields:
`etom_level`, `answerability_class`, `gold_server_ids`, `gold_operations`, `required_capabilities`, `expected_tool_path`, `known_data_risks`, `formal_test_eligible`, `exclusion_reason`.

## 6. Shared catalog boundary (L5)

No catalog operations for: segments, CET1/NIM/RoTE, numeric non-GAAP, HSBC IR-faithful structured USD. Agents must abstain or use filing evidence with uncertainty — not invent values.
