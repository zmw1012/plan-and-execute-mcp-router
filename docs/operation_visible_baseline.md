# Operation-Visible Baseline

This document describes a **simplified reimplementation adapted from Ghoshal et al. (2026)** (*Scaling Enterprise Cloud Agents to Thousands of APIs...*, EMNLP 2026 Industry Track).

It is **not** an official reproduction. The paper's production code, proprietary benchmark, and ~8K-operation catalog are unavailable. This project adapts the Operation-Visible control pattern to the local HK-equity financial tool registry.

## Architecture (paper-faithful pattern)

Centralized ReAct loop + deterministic Operation-Resolver MCP.

Meta-tools exposed to the central model (only):

1. `find_operations(query, top_k?)`
2. `describe_operation(operation_id)`
3. `invoke_api(operation_id, arguments)`

Underlying Financial tools (`get_financial_metric`, `get_filing_summary`) are **not** directly exposed in Operation-Visible mode.

## Retrieval (adapted embedding backbone)

Paper ranking form (preserved):

`score(q, operation) = cosine(embedding(q), embedding(operation_doc)) + 0.12 * lexical_score(q, operation_doc)`

| Component | Paper | This project |
|-----------|-------|--------------|
| Embedding model | `cohere.embed-v4.0` | relay-hosted `text-embedding-3-small` (**adapted**) |
| Lexical weight | 0.12 | 0.12 (exact) |
| Top-K | Top-20 analysis | default/max `top_k=20` |
| Generation model | GPT-5.4 family | Alibaba `deepseek-v4.1-flash` via `.env.aliyun` (**adapted**) |

No silent fallback to lexical-only, local embeddings, or the generation endpoint. Missing embedding config fails clearly.

Operation-document embeddings are cached under `.cache/operation_embeddings/` with keys including embedding model, registry fingerprint, and document text hash. API keys are never cached.

## Safeguards (defaults)

See `configs/operation_visible.yaml`:

- `max_llm_turns: 8`
- `max_search_reformulations: 3`
- `max_invocations: 6`
- `timeout_sec: 120`
- `temperature: 0.0`

## Parallelism

Independent **read-only** `invoke_api` calls whose arguments are already known may execute in parallel in one turn. Dependent calls remain sequential.

## Flat mode

Flat mode remains a separate comparison condition. This baseline does not change Flat semantics.

## Metrics logging

Per-question traces record messages, meta-tool calls/args/results, token usage, latency, resolver calls vs underlying `invoke_api` counts, and a placeholder status label (`full` / `partial` / `failed`). Resolver search/describe calls are **not** counted as underlying financial API calls.

## Secrets

Use local git-ignored files only:

- `ENV_FILE=.env.aliyun` for generation
- `EMBEDDING_ENV_FILE=.env.embedding-relay` for embeddings

Never commit, print, or log API keys.

## Fidelity table

| Paper component | Implemented behavior | Exact / adapted / omitted | Reason |
|-----------------|----------------------|---------------------------|--------|
| Central ReAct loop | `OperationVisibleAgent` single loop | exact (pattern) | Core architecture |
| No hidden LLM planner | Resolver is deterministic | exact (pattern) | Paper §4 |
| Meta-tools find/describe/invoke | `OperationResolverServer` | exact (names/roles) | Paper §4 |
| Ranking cosine + 0.12 lexical | `OperationRetriever.rank` | exact (formula) | Paper §4.1 |
| Embedding backbone | OpenAI-compatible `text-embedding-3-small` | adapted | Paper used Cohere embed-v4.0 |
| Generation model | pinned `LLM_MODEL` (Aliyun deepseek flash) | adapted | Paper used GPT-5.4 family |
| Top-20 retrieval | `top_k` default/max 20 | exact | Paper §4.1 / App E.3 |
| Parallel independent read-only calls | ThreadPool for independent invokes | exact (pattern) | Paper §4 |
| Visible traces | full tool/LLM events without private CoT | exact (pattern) | Paper §4 |
| ~8K SDK catalog | local financial ops only | adapted/omitted scale | Catalog unavailable |
| Proprietary cloud benchmark | not included | omitted | Unavailable |
| Finance-aware evidence repair | not included | omitted | Deferred proposed method |
| Lead-worker / Tool-Planner configs | not implemented in this PR | omitted | Baseline scope = Operation-Visible |
