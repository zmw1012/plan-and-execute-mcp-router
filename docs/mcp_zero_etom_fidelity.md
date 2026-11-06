# MCP-Zero × ETOM Fidelity Report (routing-only baseline)

## What this is

MCP-Zero × ETOM is a **routing baseline**: the agent predicts tool routes /
rejection only. It is **not** Hong Kong equity Q&A. It does **not** execute
external MCP tools (`real_tool_executions=0`). **Route exact match is not
final-answer accuracy.**

## Data provenance

| Field | Value |
| --- | --- |
| Repository | https://github.com/snooow1029/ETOM |
| Source commit | `790f28727f567b88918420d163c524d46644a694` |
| Audit source commit | `d1934e052a32c0677a43f3b46ceaab2609e16cbe` |
| Formal freeze commit | `f4bfefec49eb28ed749bf600679819cb54d0e1a6` |
| Tools | 2375 |
| Queries | L1 781, L2 773, L3 327, L4 103, L5 91 (total 2075) |
| Formal 100 | L1–L5 × 20 |
| `sample_sha256` | `99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1` |
| `ordered_task_ids_sha256` | `3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479` |
| Pilot overlap | 0 |
| Seed | 20261106 |

Machine-readable import: `benchmarks/etom/import_manifest_mcp_zero_v1.json`.

Forbidden imports: Operation-Visible predictions, scores, final-audit per-task
errors, and any OV-derived special cases.

## MCP-Zero paper mechanisms

Paper: Fei et al., 2025 — *MCP-Zero* (arXiv:2506.01056).

| Mechanism | Status |
| --- | --- |
| Active Tool Request (`<tool_assistant>`) | **Exact** |
| Hierarchical server → tool routing | **Exact** |
| Official combined score `(s_s*s_t)*max(s_s,s_t)` | **Exact** |
| Deterministic tie-break `(-score, server_id, tool_name, operation_id)` | **Added** |
| Iterative extension (max 3) | **Adapted policy** |
| Embedding `text-embedding-3-small` | **Adapted** (paper: large) |
| `server_top_k=3` / `tool_top_k=5` | **Adapted** (ETOM pin; paper 5/3) |
| Query decomposition / multi-LLM ensemble | **Omitted** |
| Real tool execution | **Omitted** (`real_tool_executions=0`) |

## Shared method parameters with HK MCP-Zero

HK method-card hash (unchanged):
`77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f`

Shared: server_top_k=3, tool_top_k=5, max_extension_rounds=3,
generation=`deepseek-v4.1-flash`, embedding=`text-embedding-3-small`,
temperature=0.0, max_llm_turns=8, official_combined, ATR protocol v2.

## ETOM-specific output contract

Dataset-specific prompt is separate from routing knobs. Method knobs are
unchanged; only the answer/route contract differs from HK Q&A.

| Field | Value |
| --- | --- |
| Prompt version | `mcp_zero_etom_routing_v1` |
| Parser protocol | `mcp_zero_protocol_v2` (`json_mode=false`) |
| ETOM system prompt SHA256 | `8ebac6ce045af0176c1fb9e580bad483b7ca4564e3521adc433c09476db285d7` |
| ETOM method-card SHA256 | `0c14b7aec5b6bdf1156b9d1ab0e330c3e0bbf8264e2db0950d2efbdce09e05f6` |
| Final statuses | `route_complete` \| `infeasible` |
| Task-id special cases | **Forbidden** |

## Hierarchical index

Namespace: `.cache/mcp_zero_etom_v1/`.
Server units + tool units; tool retrieval only inside selected servers.
Identity includes model + source/registry fingerprint + namespace.
Partial append-only checkpoints, fsync after batch, atomic final write.
Incomplete / foreign / dimension-mismatch indexes fail closed.
Formal runner never silently rebuilds when `--require-complete-index`.

## Cache / resume

Resume identity includes method-card hash, ETOM prompt hash, source pin,
sample/ordered hashes, registry hash, hierarchical index identity, provider/
host/model, top_k, extension rounds, temperature/max_tokens/max_llm_turns,
parser protocol version. `max_tasks` is cumulative locked prefix and is **not**
in resume identity.

## Official metrics

- L1 Exact Match
- L2 Equal Function Set EM
- L3 node-set P/R/F1 + dependency/order + complete route + single-server
- L4 recorded separately: union node-set EM, EFS-covering, cross-server
  dependency correctness, complete route success, plus P/R/F1
- L5 exact rejection + false-positive tool selection

Strict denominator keeps failures (score 0). Also emit completed-only
diagnostic, overall exact route, macro/micro F1, completion/failure rates.

## Pilot → formal order

1. `index-status`
2. `build-index` (paid embeddings; not this round)
3. cost-pilot 10 smoke
4. formal-100 batched with `--max-tasks` + `--resume`

## Anti-leakage

- No OV prediction/score imports
- No per-item tuning from formal-100
- Gold only post-hoc for scoring / retrieval analysis
- No cross-worktree symlinks for source

## Known limits

- `server_top_k=3` structurally limits recall on wide L4 gold graphs
- Hash/mock embeddings are for offline engineering only
- Ordered dependency accuracy in audited formal_metrics mirrors edge accuracy
  (copied as-is for OV comparability)
