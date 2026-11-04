# ETOM external cost pilot v1

Operation-Visible adapted for ETOM routing-only evaluation.

This is an external tool-routing benchmark. It does not replace the HK-equity Financial, Market, News, or held-out benchmarks, and it does not edit their frozen catalogs.

ETOM scores whether a system predicts the official tool route. The pinned registry does not execute those MCP tools, and this pilot does not turn `invoke_api` into a live tool call. A high score here is graph agreement with the official route, not evidence that a business answer was produced.

## Pinned source

| Field | Value |
| --- | --- |
| Repository | https://github.com/snooow1029/ETOM |
| Commit | `790f28727f567b88918420d163c524d46644a694` |
| License | MIT, as stated in that commit's README. The commit has no `LICENSE` file. |
| Registry | 491 servers, 2375 tools |
| Queries | L1 781, L2 773, L3 327, L4 103, L5 91 |

File hashes live in `benchmarks/etom/etom_source_pin_v1.json`. `scripts/fetch_etom_source.sh` checks out that commit under `data/etom/raw/`, which is gitignored. The loader refuses to run if the commit or any dataset hash changes. Conversion reads the raw JSON and writes only under `data/etom/derived/`.

## What is exact, adapted, or omitted

| Piece | Relation to official ETOM | What this pilot does |
| --- | --- | --- |
| Queries and gold tool graphs | exact | Read from the pinned JSON. The sampler only chooses ids. |
| Registry names, descriptions, input schemas | exact | `describe_operation` returns those official fields. |
| Tool count | exact | 2375 operations are searchable. |
| Pseudo-output schema | omitted | The pinned registry has no output schema. The official LLM schema generator was not run. |
| MCP tool execution | omitted | `invoke_api` records a predicted node and returns `business_result: null`. |
| EM / Precision / Recall / F1 | exact rules, local port | Graph match and level rules follow `eval/graph_evaluator.py` and `eval/eval.py` at the pinned commit. |
| Embedding model | adapted | `text-embedding-3-small`, not ETOM's default Qwen embedding. |
| Generation model | adapted | `deepseek-v4.1-flash` via the existing Aliyun env profile. Temperature 0, max tokens 2048, JSON mode on. `top_p` and generation `top_k` are not sent. |
| Retrieval | adapted | Operation-Visible ranking: cosine plus `0.12 * lexical`, top-k 20. |
| Orchestrator | adapted | One Operation-Visible loop. MCP-Zero, ToolShed, ReAct, Hybrid, and Proposed Method are not run. |
| Answer grading | omitted | There is no end-to-end answer score. |

## Cost-pilot sample

Seed `5018`, sampler `seeded_weighted_server_coverage_v1`. Quotas are two tasks per level. L3's first draw is restricted to a single-server dependency chain. L4's first draw is restricted to a cross-server dependency graph. L5 uses official infeasible tasks with an empty gold route. Later draws prefer servers that have not been used yet.

The locked ids are in `benchmarks/etom/cost_pilot_10_v1.lock.json`:

| Task | Level | Constraint |
| --- | --- | --- |
| L1-q185 | L1 | single tool, Excel MCP Server |
| L1-q233 | L1 | single tool, QASE MCP Server |
| L2-q522 | L2 | equivalence-style route, MindManager MCP Server |
| L2-q732 | L2 | equivalence-style route across Needle, Vectorize, docs-mcp |
| L3-q37 | L3 | single-server sequence, NPM Sentinel MCP |
| L3-q105 | L3 | single-server sequence, Obsidian Local REST API MCP Server |
| L4-q91 | L4 | cross-server dependency, Hugeicons and Obsidian |
| L4-q13 | L4 | cross-server dependency, fetch/scrape equivalence graph |
| L5-q31 | L5 | official infeasible task |
| L5-q66 | L5 | official infeasible task |

Every locked row has `dataset_split=cost_pilot` and `exclude_from_final_etom_sample=true`. `assert_not_excluded` rejects these ids for a later 100/200 draw. This pilot hard-stops at 10 tasks.

## Cost accounting

Catalog embeddings are built once and cached under `.cache/etom_embeddings/`. Query embeddings are counted only when `find_operations` runs. The summary keeps those two buckets separate.

USD figures use the peak cache-miss list price for DeepSeek-V4.1-Flash (input $0.30 / 1M, output $1.20 / 1M) and OpenAI `text-embedding-3-small` ($0.02 / 1M). Cache-hit discounts are not applied. The Aliyun and embedding relays can invoice differently, so the dollar amounts are a conservative estimate, not an invoice.

The 100-task and 200-task projections add the one-time catalog cost and the one verification probe once, then multiply the mean per-task variable cost. The buffer is `N * sample standard deviation` of that per-task cost. With n=10 it shows spread. It is not a confidence interval.

HTTP retries are capped at 1. Parse repairs are capped at 2. A provider or model echo that is not `aliyun` / `deepseek-v4.1-flash` / `text-embedding-3-small` stops the run.

## Commands

From the worktree `hk-equity-mcp-agent-etom`:

```bash
# 1. Preflight. No model calls.
bash scripts/run_etom_operation_visible_pilot_10.sh preflight

# 2. Formal 10-task pilot. caffeinate keeps the Mac awake.
bash scripts/run_etom_operation_visible_pilot_10.sh run

# 3. Resume after sleep, Ctrl+C, or a dropped network.
bash scripts/run_etom_operation_visible_pilot_10.sh resume

# 4. Follow the log.
bash scripts/run_etom_operation_visible_pilot_10.sh log
```

Stop a foreground run with Ctrl+C. Each finished task is already on disk. Resume skips completed tasks and retries tasks that failed. The env files default to the sibling checkout's `.env.aliyun` and `.env.embedding-relay`. The scripts do not print those files.

Outputs, all under `outputs/etom_external_pilot_v1/`:

- `etom_cost_pilot_10_v1.json`
- `predictions.jsonl`
- `per_task_metrics.csv`
- `cost_summary.json`
- `run_metadata.json`
- `run.log`

The raw checkout, embedding cache, and `outputs/` stay untracked.

## Run results

The measured 10-task figures are written by the live command into `outputs/etom_external_pilot_v1/cost_summary.json`. This section is updated after that run and is not a substitute for the JSON.
