# Financial held-out benchmarks

| File | Status | Notes |
|---|---|---|
| `financial_test_30_v1.jsonl` | **`superseded_before_any_agent_run`** | Candidate freeze at `f85d53f`. Superseded because 14 `evidence_answerable` items cited formal Annual/Interim **Reports** and report PDF pages while `gold_operations` used **mcp-hkexnews v0.1**, which only supports results/performance **announcements** (full Annual/Interim Report PDFs are out of scope). No agent/LLM run was ever executed on this file. |
| `financial_test_30_v1_1.jsonl` | **`active`** | Provenance-audited held-out Test 30. Evidence golds re-anchored to in-scope official results announcements retrieved through frozen mcp-hkexnews. Structured golds live-verified against raw AKTools MCP output. |

## Why v1 was superseded

A formal Annual Report and an Annual Results Announcement are different documents even when they contain the same number. Document title, URL and page must match the document actually retrievable through the named gold MCP path. v1 conflated those metadata fields; v1.1 corrects them before any agent evaluation.

## Catalog binding

- Catalog: frozen Financial Real MCP v1 (`configs/operations/financial_real_mcp.yaml`)
- Servers: AKTools + mcp-hkexnews only
- No catalog mutation for this freeze

## Validation

- `scripts/validate_financial_heldout_test_v1_1.py`
- `tests/test_financial_heldout_benchmark_v1_1.py`

Do not expose gold / provenance fields to agent prompts.
