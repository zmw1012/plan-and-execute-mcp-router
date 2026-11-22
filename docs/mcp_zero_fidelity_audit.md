# MCP-Zero Fidelity Audit (HK-155 adapted baseline)

Paper: Fei et al., 2025 — *MCP-Zero: Active Tool Discovery for Autonomous LLM Agents* (arXiv:2506.01056)  
Official repo pin: `xfey/MCP-Zero@fd666c44…`  
ETOM MCP0 pin: `snooow1029/ETOM@790f287…`

This baseline is an **adapted reimplementation**, not a byte-for-byte reproduction.

## Fidelity table

| Mechanism | Paper / official | This HK baseline | Status |
|---|---|---|---|
| Active Tool Request (`<tool_assistant>`) | Required | Supported + parsed | **Exact format** |
| Hierarchical Semantic Routing | Server then tool | `ServerRetriever` → `ToolRetriever` | **Exact structure** |
| Server dual-match (desc + summary max) | Official matcher | Implemented (`use_summary`) | **Exact** |
| Tool final score `(s_s*s_t)*max(s_s,s_t)` | Official matcher | `score_mode=official_combined` | **Exact formula** |
| ETOM tool-only final score | ETOM MCP0 override | `score_mode=tool_similarity_only` | Available (not default) |
| `server_top_k` / `tool_top_k` | Official / paper default **5 / 3** | **Frozen 3 / 5** (ETOM pinned adaptation) | **Adapted default. This is not the paper's original exact configuration.** |
| Embedding model | Paper/official: `text-embedding-3-large` | **`text-embedding-3-small`** (shared with Operation-Visible) | **Adapted** |
| Iterative Capability Extension | Model may refine request | `ExtensionController` + max rounds | **Adapted policy** |
| Extension termination | Model decides adequacy | Explicit reasons: max rounds / duplicate / no new tools / empty / proceeded | **Documented adaptation** |
| Deterministic tie-break | Not specified | `(-score, server_id, tool_name, operation_id)` | **Added** |
| Visible retrieval trace | Implied | Written to predictions + `retrieval_analysis.json` | **Added** |
| Multi-LLM roles (router/retriever/…) | ETOM MCP0 multi-role | Single generation model (`deepseek-v4.1-flash`) | **Adapted** |
| Query decomposition | ETOM MCP0 | Omitted for HK fair single-model compare | **Omitted** |
| Conversational classification gate | ETOM MCP0 | Omitted (all HK tasks are tool-capable Q&A) | **Omitted** |
| Real MCP execution | Paper executes tools | `UnifiedRealMcpGateway` (same as OV) | **Exact backend** |
| Answer contract | Paper free-form | Shared OV final-answer JSON contract | **Adapted for HK eval** |
| Blocked ops exposure | N/A | Never in catalog / never selectable | **HK constraint** |
| Gold leakage | N/A | Forbidden (prompt + runtime) | **HK constraint** |

## Intentionally shared with Operation-Visible

- Dataset: HK-155 manifest + run order  
- Catalog: 42 unique / 39 runtime-exposed  
- Gateway: `UnifiedRealMcpGateway`  
- Evaluator: `hk_unified_evaluator_v1.2`  
- Generation: `deepseek-v4.1-flash`, `temperature=0.0`, `max_tokens=1024`, `max_llm_turns=8`  
- Embedding backbone: `text-embedding-3-small`  

**Primary experiment variable:** MCP router / orchestration method (`mcp_zero` vs `operation_visible`).

## Frozen before any HK-155 real result

The paper and official matcher default is `server_top_k=5`, `tool_top_k=3`.

This baseline freezes the ETOM pinned adaptation `server_top_k=3`, `tool_top_k=5` before any HK-155 real prediction exists. The rest of the method card is also frozen:

- deterministic tie-break `(-score, server_id, tool_name, operation_id)`
- `max_extension_rounds=3`
- `generation_model=deepseek-v4.1-flash`
- `embedding_model=text-embedding-3-small`
- `temperature=0.0`
- `max_tokens=1024`
- `max_llm_turns=8`
- evaluator `hk_unified_evaluator_v1.2`

Do not change `server_top_k` or `tool_top_k` after seeing HK-155 results. A later run with the paper's 5/3 is an independent ablation and must not replace this baseline's primary numbers. The canonical **routing** method card is `frozen_method_identity()` in `src/hk_equity_mcp/mcp_zero/__init__.py`. Its sha256 is `77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f`. Provider/host are runtime env metadata (`openai_compatible` @ `api.zhizengzeng.com`), recorded in run/resume identity via `frozen_runtime_env_identity()`, and intentionally excluded from the routing method-card hash. `prepare()` refuses a config that drifts from the method card or those env pins.

## Cursor vs experiment models

| Role | Model |
|---|---|
| Cursor coding agent | Claude / other coding models (not recorded as experiment model) |
| Experiment generation | `deepseek-v4.1-flash` only |
| Experiment embedding | `text-embedding-3-small` |

## Omitted / deferred

- ETOM-100 formal runner / live ETOM MCP0 comparison (separate workstream)  
- Official multi-server MCP-tools corpus evaluation  
- Live DeepSeek / embedding / MCP network calls in this commit (mock/dry-run only)
