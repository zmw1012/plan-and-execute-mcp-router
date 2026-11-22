# PE-Fin Lite v1 — Fidelity and Deviations

## Fidelity to the research brief

| Requirement | Status |
|---|---|
| Architecture `plan_first_step_level_retrieval_v1` | Yes |
| Static capability profile (no LLM compiler) | Yes |
| One-shot planner + ≤1 JSON repair | Yes |
| Semantic steps only (no operation_id in plan) | Yes |
| Per-step embedding Top-K=5, describe 2–3 | Yes |
| Binder primary+fallback one call | Yes |
| Local fallback ≤1; no global replan | Yes |
| Required-output + scope success definition | Yes |
| Disclosure composite fixed chain | Yes |
| Safe calculator (no eval) | Yes |
| Final synthesizer + FINAL_ANSWER_CONTRACT | Yes |
| Budgets LLM≤8 MCP≤8 | Yes |
| Live identity zhizengzeng / deepseek-v4.1-flash / text-embedding-3-small | Yes |
| No Monitor / capability compiler / Tier A/B/C | Yes |
| Diagnostic-12 same order as APV | Yes |
| Smoke-5 engineering only | Yes |

## Intentional deviations / simplifications

1. **Binder has no format-repair loop** — brief allows primary+fallback in one response; failed binder → structured step failure.
2. **Reason parse failure** soft-falls back to a non-fabricating gap note (still no tool calls / no replan).
3. **Embedding index module** adapted from APV shared-index algorithm with a **separate** namespace `pe_fin_lite_operation_embeddings_v1` (no silent lexical fallback).
4. **Calculator** reuses APV-Fin AST-safe evaluator (deterministic; no eval()).
5. **Metadata sidecar** reused from frozen HK unified metadata (company-generic).

## Explicit non-goals this round

- No real DeepSeek / embedding / MCP network runs
- No formal Smoke/Diagnostic accuracy claims
- No push / PR / merge
- No gold mutation / baseline raw mutation
