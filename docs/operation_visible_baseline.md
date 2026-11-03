# Operation-Visible Baseline

This document describes a **simplified reimplementation adapted from Ghoshal et al. (2026)** (*Scaling Enterprise Cloud Agents to Thousands of APIs...*, EMNLP 2026 Industry Track).

It is **not** an official reproduction. The paper's production code, proprietary benchmark, and ~8K-operation catalog are unavailable. This project adapts the Operation-Visible *control pattern* to the local HK-equity financial tool registry.

## What this experiment can and cannot claim

- **Can measure:** whether exposing the two Financial tools (via Flat or Operation-Visible) improves answer quality versus MCP-Free closed-book on the same 30-question set, model, temperature, and grading sheet.
- **Cannot claim:** large-scale tool-retrieval advantage over thousands of APIs. The current catalog has **exactly 2** Financial operations (`financial.get_financial_metric`, `financial.get_filing_summary`). Market/News tools will be added only when their real local servers are merged—do not invent APIs to inflate catalog size.
- **`status=full` / agent completion:** means the agent loop produced a final-answer object. It does **not** mean the answer is correct. Correctness is decided by the shared manual grading sheet or a separate grading program.

## Architecture (adapted pattern)

Centralized ReAct loop + deterministic Operation-Resolver boundary.

Meta-tools exposed to the central model (only):

1. `find_operations(query, top_k?)`
2. `describe_operation(operation_id)`
3. `invoke_api(operation_id, arguments)`

Underlying Financial tools are **not** directly exposed in Operation-Visible mode.

**Tool protocol (adapted):** this project uses a **prompt-based JSON tool-calling protocol** (model returns JSON with `tool_calls`), not the paper's original production MCP client/runtime. Marked **adapted** in the fidelity table.

## Retrieval

Published ranking form (**formula preserved**):

`score(q, operation) = cosine(embedding(q), embedding(operation_doc)) + 0.12 * lexical_score(q, operation_doc)`

| Component | Paper | This project |
|-----------|-------|--------------|
| Embedding model | `cohere.embed-v4.0` | relay-hosted `text-embedding-3-small` (**adapted**) |
| Outer lexical weight `0.12` | published | **preserved** |
| Inner `lexical_score` feature weights (e.g. 0.40/0.25/0.20/…) | not fully published as a complete implementation | **project adaptation** — do not call exact |
| Top-K | Top-20 analysis | default/max `top_k=20` |
| Generation model | GPT-5.4 family | Alibaba `deepseek-v4.1-flash` via `.env.aliyun` (**adapted**) |
| Tool-call transport | paper production MCP stack | prompt-based JSON protocol (**adapted**) |
| Catalog size | ~8K operations | **2** Financial operations |

No silent fallback to lexical-only, local embeddings, or the generation endpoint. Missing embedding config fails clearly.

Operation-document embeddings are cached under `.cache/operation_embeddings/` with keys including embedding model, registry fingerprint, and document text hash. API keys are never cached.

## Safeguards (defaults)

See `configs/operation_visible.yaml` / `configs/financial_experiment.yaml`:

- `max_llm_turns: 8`
- `max_search_reformulations: 3`
- `max_invocations: 6`
- `timeout_sec: 120`
- `temperature: 0.0`
- `max_tokens: 1024` (shared across MCP-Free / Flat / Operation-Visible)

## Parallelism

Only a **contiguous** run of independent **read-only** `invoke_api` calls whose arguments are already known may execute as a parallel batch. `find_operations` / `describe_operation` (and dependent invokes) that appear earlier in the same turn always dispatch first. Dependent calls remain sequential.

## Flat and MCP-Free

- **MCP-Free:** question only; no tools; no embedding.
- **Flat:** central model calls `get_financial_metric` / `get_filing_summary` on FinancialServer directly (no resolver, no embedding). The Flat system prompt embeds the **full OperationSpec contracts** (description, required/optional params, enums, defaults, constraints, invocation guidance) generated from the same registry Operation-Visible uses, so schema visibility is fair even though Flat skips `describe_operation`.
- **Operation-Visible:** meta-tools only; embedding required.

Use `scripts/run_financial_experiment.py` for all three conditions with the same questions, model, temperature, answer contract, and grading columns.

## Metrics logging

Per-question rows record messages, meta-tool calls/args/results, token usage, latency, resolver calls vs underlying `invoke_api` counts, and predicted answer status (`answered` / `clarification_needed` / `unavailable`). Resolver search/describe calls are **not** counted as underlying financial API calls.

## Secrets

Use local git-ignored files only:

- `ENV_FILE=.env.aliyun` for generation
- `EMBEDDING_ENV_FILE=.env.embedding-relay` for embeddings (Operation-Visible only)

Never commit, print, or log API keys. Shell-exported variables take precedence over env-file values.

## Fidelity table

| Paper component | Implemented behavior | Exact / adapted / omitted | Reason |
|-----------------|----------------------|---------------------------|--------|
| Central ReAct loop | `OperationVisibleAgent` single loop | adapted (pattern) | Prompt-JSON tool protocol, not paper runtime |
| No hidden LLM planner | Deterministic resolver | adapted (pattern) | Same control idea; local scale |
| Meta-tools find/describe/invoke | `OperationResolverServer` | adapted (names/roles) | Same boundary; 2-op catalog |
| Outer ranking `cosine + 0.12 * lexical` | `OperationRetriever.rank` | preserved (formula) | Paper §4.1 published form |
| Inner lexical feature weights | local 0.40/0.25/0.20/… blend | project adaptation | Full paper lexical implementation not public |
| Embedding backbone | OpenAI-compatible `text-embedding-3-small` | adapted | Paper used Cohere embed-v4.0 |
| Generation model | pinned `LLM_MODEL` (Aliyun deepseek flash) | adapted | Paper used GPT-5.4 family |
| Tool-call transport | prompt-based JSON `tool_calls` | adapted | Not paper MCP client stack |
| Top-20 retrieval | `top_k` default/max 20 | preserved | Paper §4.1 / App E.3 |
| Parallel contiguous independent read-only calls | ThreadPool for consecutive invokes | adapted (pattern) | Order-preserving batches |
| Visible traces | tool/LLM events without private CoT | adapted (pattern) | Local logging schema |
| ~8K SDK catalog | **2** Financial operations | adapted/omitted scale | Catalog unavailable; no invented APIs |
| Proprietary cloud benchmark | not included | omitted | Unavailable |
| Finance-aware evidence repair | not included | omitted | Deferred proposed method |
| Lead-worker / Tool-Planner configs | not implemented here | omitted | Baseline scope = Operation-Visible |
