# Financial Held-out Test 30 v1 — Design

**Branch:** `feature/financial-heldout-test-v1`  
**Frozen Development checkpoint:** `91c3360` (`feature/real-financial-mcp-discovery`)  
**Dataset path:** `benchmarks/heldout/financial_test_30_v1.jsonl`  
**Catalog:** frozen Financial Real MCP v1 (`configs/operations/financial_real_mcp.yaml`)

## Purpose

Freeze a **held-out Test** benchmark of exactly **30** Financial questions for **final evaluation only**.

This split is **not** for method development, prompt iteration, or error-analysis mining.

| Field | Value |
|---|---|
| `dataset_split` | `heldout_test` |
| `benchmark_role` | `final_evaluation_only` |
| `locked` | `true` |

The Development / error-analysis set `benchmarks/questions/financial_mcp_free_30_v1.jsonl` (and its coverage annotation `financial_real_mcp_coverage_v1.jsonl`) **must not** be edited, copied, or lightly paraphrased into this held-out set.

## Non-goals (this freeze)

- Do **not** run Development or Held-out experiments with any LLM/agent.
- Do **not** call paid LLM or embedding APIs while authoring or validating.
- Do **not** push, open a PR, or merge.
- Do **not** modify `.env` or expose credentials.
- Do **not** wire new MCP servers/tools into the frozen v1 catalog without explicit user confirmation (see gap proposals doc).

## Design principles

1. **Schema continuity** — reuse the Real-MCP coverage field names (`id`/`question_id`, `etom_level`, `answerability_class`, `gold_operations`, `gold_server_ids`, gold values with currency/unit/basis, evidence URLs/pages, `formal_test_eligible`, etc.) and add lock/split metadata.
2. **No Development near-duplicates** — different reasoning compositions and evidence paths (e.g. interim-only metrics, fetch-then-compute vs numbers-in-question, dual-path corroboration, REPORT_DATE row selection, cross-company unsafe comparison).
3. **Catalog-faithful gold paths** — every L1–L4 item maps to operations that exist in the frozen Financial v1 catalog; L5 items define refusal/clarification behaviour.
4. **Official gold only** — numeric/reference answers come from the project’s verified official-company / HKEX-aligned mock snapshot (`data/mock/financial.json`), not model-generated facts.
5. **Fiscal discipline** — respect official cutoff / publication conventions; no invented FY2026 full-year Tencent; Alibaba FY naming remains March year-end distinct from calendar year.

## Target distributions (frozen)

| Axis | Target | Actual |
|---|---|---|
| Questions | 30 | 30 |
| ETOM L1…L5 | 6 each | 6 each |
| Tencent / HSBC / Alibaba | 10 / 10 / 10 | 10 / 10 / 10 |
| Structured / Evidence / Infeasible | ~10 / ~14 / 6 | **10 / 14 / 6** |

Company × level matrix: each issuer contributes **2 questions per ETOM level**.

## Answerability policy

| Class | Count | Rule |
|---|---|---|
| `structured_answerable` | 10 | Verified path via AKTools structured interfaces under frozen v1 (Tencent/Alibaba reported metrics, fiscal row selection, same-server multi-step). |
| `evidence_answerable` | 14 | Requires HKEXnews evidence (and/or cross-server corroboration). Includes all HSBC absolute amounts (East Money absolute USD tables are **not** gold). |
| `intentionally_infeasible` | 6 | L5 only: unsupported segment/metric, unavailable period, or unsafe comparison — with explicit refusal/clarification gold behaviour. |

## Coverage intents (held-out compositions)

| Intent | Example held-out IDs |
|---|---|
| Single metric retrieval | `fin_ht_001`, `fin_ht_011`, `fin_ht_021` |
| Period comparison | `fin_ht_006` |
| Reported vs adjusted / non-IFRS / non-GAAP | `fin_ht_003`, `fin_ht_004`, `fin_ht_007`, `fin_ht_027` |
| Annual vs interim | `fin_ht_001`/`fin_ht_002` vs annual Development items; Alibaba interim refusal `fin_ht_030` |
| Banking semantics / evidence preference | `fin_ht_013`, `fin_ht_014`, `fin_ht_016` |
| Multi-step calculation | `fin_ht_005`, `fin_ht_015`, `fin_ht_016` |
| Evidence / page retrieval | HSBC L1–L3; keyword/page ops |
| Ambiguity / unavailable period clarification or refusal | `fin_ht_010`, `fin_ht_030` |
| Unsupported metric / unsafe comparison refusal | `fin_ht_009`, `fin_ht_019`, `fin_ht_020`, `fin_ht_029` |
| Cross-server (ETOM L4) | `fin_ht_007`, `fin_ht_008`, `fin_ht_017`, `fin_ht_018`, `fin_ht_027`, `fin_ht_028` |

## Gold and leakage rules

- Gold answers are for **offline scoring / human adjudication only**.
- Do **not** expose `reference_answer`, `gold_values`, or gold operation sequences in agent-facing prompts.
- L1–L4 items include official `source_document` / `source_url` / `source_page_pdf` where applicable.
- Currency, unit, reported/adjusted basis, and EPS basis are explicit in gold fields.

## Validation

Deterministic checks live in:

- `scripts/validate_financial_heldout_test_v1.py`
- `tests/test_financial_heldout_benchmark.py`

No LLM, embedding, or experiment runner invocation is part of validation.
