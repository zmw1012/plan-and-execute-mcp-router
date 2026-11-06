# ETOM Formal-100 Operation-Visible Final Audit v1

Offline scoring and error analysis only. No LLM, embedding, MCP, or paid API call was made. No task was rerun. Raw predictions, gold routes, the formal lock, the prompt, the registry, and run metadata were not modified.

## 1. Raw artifact integrity

`grouped_aggregates.json` is not in the run directory. The run artifact is `grouped_results.json`.

| File | SHA256 | Bytes |
| --- | --- | ---: |
| `predictions.jsonl` | `c8ec6b49533c92b10dcfec7ec52cda6585bd1eaa1f4f13387ed98bd5c00a478d` | 2235284 |
| `run_metadata.json` | `9086f74327b7cad1a59ff824255705b8e45c80b478e3a1a0adef8567b657eae8` | 5965 |
| `per_task_metrics.csv` | `8dabd604069e2a9e17c9dc42edaf9f3bdf274e04cc3d67bb445e0b4775e795a2` | 12479 |
| `cost_summary.json` | `869432af5cc835d96617b98b508eec75ea8dd3ea5e138e75567c39cf0a135c06` | 4016 |
| `retrieval_analysis.json` | `0b3f6f049c3904acb68fad7bd4c05787d772a28b506e839ef03daeeb293d32b4` | 302 |
| `error_summary.json` | `77beda4e23033cd24476b1bc6f25889321c44044e7362c4a970086db0dde2d01` | 254 |
| `grouped_results.json` | `ac66771f6c71be09cf5b20fac66ca43ebeee19cd042abc618b28236409d0f558` | 2577 |

| Identity field | Value |
| --- | --- |
| predictions rows / unique task IDs | 100 / 100 |
| L1–L5 counts | {'L1': 20, 'L2': 20, 'L3': 20, 'L4': 20, 'L5': 20} |
| matches formal_100_v1.lock.json | True |
| formal sample SHA256 | `99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1` |
| ordered task IDs SHA256 | `3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479` |
| pilot overlap | 0 |
| registry SHA256 | `3c362877ec5733cf0fa1d3451b2f353fb97674e66075da0d7a6f41c0f98cc608` |
| catalog index | 2375/2375 `catalog-6172d05c839b110e.json` |
| catalog file SHA256 | `8720d7d2a18f66866c3c8e74ee01a34953e0bb2c719ec6a40e66743fb2db66e2` |
| prompt SHA256 | `3da9b3d8316a1613e75bebb38bdbfe694fbc809a5ad5c5ae363340dd47d705cd` |
| generation model | `deepseek-v4.1-flash` |
| embedding model | `text-embedding-3-small` |
| provider host | `api.zhizengzeng.com` |
| temperature | 0.0 |
| max_tokens | 2048 |
| max_llm_turns limit | 8 |
| real_tool_executions | 0 |

Provider host is the frozen relay host only. No API key is recorded here.

## 2. Freeze

Derived files are written only under `outputs/etom_external_formal100_operation_visible_v1/final_audit_v1/` and mirrored to `docs/etom_operation_visible_formal100_final_audit_v1/`.

Predictions SHA256 after the audit: `c8ec6b49533c92b10dcfec7ec52cda6585bd1eaa1f4f13387ed98bd5c00a478d` (unchanged=True).

## 3. Official ETOM metrics

Denominator policy:

- Strict: all 20 tasks in the level. A failed task scores exact-match false and precision/recall/F1 = 0.
- Completed-only: the 92 tasks whose frozen status is `completed`. This matches `cost_summary.json`.
- L2 exact match is Equal Function Set exact match from the pinned GraphEvaluator. It accepts any listed equivalent tool, and a single listed gold tool is not the whole gold set.
- L3/L4 complete route success is the official graph exact match, including dependency order and no extra tools. Node-set exact match, precision, recall, F1, and dependency correctness are reported separately.
- L4 node-set exact match is the official union of every EFS member. EFS-covering match is the set version aligned with one accepted tool per step.

| Level | Metric | Strict | Completed-only |
| --- | --- | ---: | ---: |
| L1 | Exact Match | 17/20 = 0.850 | 17/19 = 0.895 |
| L2 | Equal Function Set Exact Match | 12/20 = 0.600 | 12/17 = 0.706 |
| L3 | Node Set Exact Match (official union) | 13/20 = 0.650 | 13/19 = 0.684 |
| L3 | EFS-covering node set | 13/20 = 0.650 | 13/19 = 0.684 |
| L3 | Precision | 0.792 | 0.833 |
| L3 | Recall | 0.813 | 0.856 |
| L3 | F1 | 0.785 | 0.826 |
| L3 | Dependency/order correctness | 13/20 = 0.650 | 13/19 = 0.684 |
| L3 | Complete route success | 13/20 = 0.650 | 13/19 = 0.684 |
| L4 | Node Set Exact Match (official union) | 3/20 = 0.150 | 3/17 = 0.176 |
| L4 | EFS-covering node set | 5/20 = 0.250 | 5/17 = 0.294 |
| L4 | Precision | 0.365 | 0.429 |
| L4 | Recall | 0.472 | 0.556 |
| L4 | F1 | 0.385 | 0.453 |
| L4 | Cross-server dependency correctness | 6/20 = 0.300 | 6/17 = 0.353 |
| L4 | Complete route success | 5/20 = 0.250 | 5/17 = 0.294 |
| L5 | Exact Rejection Match | 19/20 = 0.950 | 19/20 = 0.950 |
| L5 | False-positive tool selection rate | 1/20 = 0.050 | 1/20 = 0.050 |

| Aggregate | Value |
| --- | ---: |
| strict exact-route accuracy | 66/100 = 0.660 |
| completion rate | 92/100 = 0.920 |
| failure rate | 8/100 = 0.080 |
| macro exact-route across L1–L5 | 0.660 |
| micro exact-route | 0.660 |
| macro F1 strict | 0.734 |
| micro F1 strict | 0.734 |
| macro F1 completed-only | 0.788 |
| micro F1 completed-only | 0.797 |

Micro exact-route is 66/100. Macro exact-route is the unweighted mean of the five level rates. They match because the formal sample has 20 tasks in every level. Completed-only macro and micro F1 differ because L1–L5 do not have the same number of completed tasks. L3/L4 strict exact match is the complete-route number only; node set, F1, and dependency metrics stay visible above.

L2 gold shape: 13 / 20 tasks are one step with one gold tool; 7 / 20 tasks are multi-step equal-function sets (each step accepts any listed equivalent). Exact match requires every step, not the first listed tool alone. Multi-step equal-function exact matches: 0 / 7. Singleton exact matches: 12 / 13; those gold sets contain one tool, so there is no alternate equivalent to select. Predictions that select a non-first equivalent and still exact-match: 0.

## 4. Failure audit

The eight failed tasks stay failed in the official score. `max_llm_turns` is a resource limit. Parse errors are not repaired per task id.

### L1-q591 (parse_error)

- Level: L1
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 0; gold operation retrieved: False
- Retrieved candidate count: 0
- Last parsed model output stored: False
- Last parsed output: `raw model text was not stored`
- Failure stage: parse_before_any_action
- Partial correct route: False (0/1 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: False
- Resource limit: False
- Note: Every stored turn is a JSON parse failure. The assistant string itself was not persisted, so the route cannot be reconstructed.

### L2-q766 (max_llm_turns)

- Level: L2
- Raw response stored: False
- Selected server/tools: ['mailmodo::currentDateTime', 'Outlook Meetings Scheduler MCP Server::find-person', 'Outlook Meetings Scheduler MCP Server::find-person', 'Outlook Meetings Scheduler MCP Server::find-person', 'Outlook Meetings Scheduler MCP Server::create-event-with-attendees']
- Retrieval searches: 3; gold operation retrieved: True
- Retrieved candidate count: 40
- Last parsed model output stored: True
- Last parsed output: `{"turn": 8, "tool_calls": [{"name": "invoke_api", "arguments": {"operation_id": "mailmodo::currentDateTime", "arguments": {}, "depends_on": []}}, {"name": "invoke_api", "arguments": {"operation_id": "Outlook Meetings Scheduler MCP Server::find-person", "arguments": {"name": "John"}, "depends_on": []}}, {"name": "invoke_api", "arguments": {"operation_id": "Outlook Meetings Scheduler MCP Server::find-person", "arguments": {"name": "Jane"}, "depends_on": []}}, {"name": "invoke_api", "arguments": {"operation_id": "Outlook Meetings Scheduler MCP Server::find-person", "arguments": {"name": "Bob"}, "depends_on": []}}, {"name": "invoke_api", "arguments": {"operation_id": "Outlook Meetings Scheduler `
- Failure stage: turn_budget_exhausted
- Partial correct route: True (1/4 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: True
- Resource limit: True
- Note: The frozen turn limit was reached. The task stays a resource_limit failure; no extra model call was added.

### L2-q720 (max_llm_turns)

- Level: L2
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 3; gold operation retrieved: True
- Retrieved candidate count: 60
- Last parsed model output stored: True
- Last parsed output: `{"turn": 6, "tool_calls": [{"name": "describe_operation", "arguments": {"operation_id": "DocsFetcher MCP Server::fetch-url-docs"}}, {"name": "describe_operation", "arguments": {"operation_id": "mcp-jinaai-reader::read_url"}}, {"name": "describe_operation", "arguments": {"operation_id": "Fetch MCP Server::fetch"}}, {"name": "describe_operation", "arguments": {"operation_id": "SushiMCP::list_llms_txt_sources"}}, {"name": "describe_operation", "arguments": {"operation_id": "SushiMCP::fetch_llms_txt"}}], "final_answer": null}`
- Failure stage: turn_budget_exhausted
- Partial correct route: False (0/4 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: True
- Resource limit: True
- Note: The frozen turn limit was reached. The task stays a resource_limit failure; no extra model call was added.

### L2-q518 (parse_error)

- Level: L2
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 0; gold operation retrieved: False
- Retrieved candidate count: 0
- Last parsed model output stored: False
- Last parsed output: `raw model text was not stored`
- Failure stage: parse_before_any_action
- Partial correct route: False (0/1 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: False
- Resource limit: False
- Note: Every stored turn is a JSON parse failure. The assistant string itself was not persisted, so the route cannot be reconstructed.

### L3-q195 (parse_error)

- Level: L3
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 6; gold operation retrieved: True
- Retrieved candidate count: 80
- Last parsed model output stored: True
- Last parsed output: `{"turn": 4, "tool_calls": [{"name": "find_operations", "arguments": {"query": "Redis Cloud fixed plans list plan ID", "top_k": 20}}], "final_answer": null}`
- Failure stage: parse_after_partial_trace
- Partial correct route: False (0/4 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: False
- Resource limit: False
- Note: The run stopped after repeated JSON parse failures. Earlier turns contain tool observations, and the recorded route does not cover the gold chain.

### L4-q48 (parse_error)

- Level: L4
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 4; gold operation retrieved: True
- Retrieved candidate count: 80
- Last parsed model output stored: True
- Last parsed output: `{"turn": 2, "tool_calls": [{"name": "describe_operation", "arguments": {"operation_id": "SushiMCP::list_llms_txt_sources"}}, {"name": "describe_operation", "arguments": {"operation_id": "SushiMCP::fetch_llms_txt"}}, {"name": "describe_operation", "arguments": {"operation_id": "MCP Chrome Google Search Tool::web-search"}}, {"name": "describe_operation", "arguments": {"operation_id": "MCP Webscan Server::crawl-site"}}, {"name": "describe_operation", "arguments": {"operation_id": "MCP Webscan Server::extract-links"}}, {"name": "describe_operation", "arguments": {"operation_id": "MCP Webscan Server::check-links"}}], "final_answer": null}`
- Failure stage: parse_after_partial_trace
- Partial correct route: False (0/5 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: False
- Resource limit: False
- Note: The run stopped after repeated JSON parse failures. Earlier turns contain tool observations, and the recorded route does not cover the gold chain.

### L4-q75 (max_llm_turns)

- Level: L4
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 17; gold operation retrieved: True
- Retrieved candidate count: 80
- Last parsed model output stored: True
- Last parsed output: `{"turn": 8, "tool_calls": [{"name": "find_operations", "arguments": {"query": "mobile", "top_k": 50}}, {"name": "describe_operation", "arguments": {"operation_id": "Mobile Next::screen_capture"}}, {"name": "describe_operation", "arguments": {"operation_id": "Mobile Next::save_screenshot"}}], "final_answer": null}`
- Failure stage: turn_budget_exhausted
- Partial correct route: False (0/6 gold steps)
- Harness/evaluation failure: False
- Agent reasoning failure: True
- Resource limit: True
- Note: The frozen turn limit was reached. The task stays a resource_limit failure; no extra model call was added.

### L4-q71 (parse_error)

- Level: L4
- Raw response stored: False
- Selected server/tools: []
- Retrieval searches: 0; gold operation retrieved: False
- Retrieved candidate count: 0
- Last parsed model output stored: False
- Last parsed output: `raw model text was not stored`
- Failure stage: parse_before_any_action
- Partial correct route: False (0/6 gold steps)
- Harness/evaluation failure: True
- Agent reasoning failure: False
- Resource limit: False
- Note: Every stored turn is a JSON parse failure. The assistant string itself was not persisted, so the route cannot be reconstructed.

## 5. Error taxonomy

Each task has one primary error. Exact-route successes have `primary_error = null`. Secondary errors describe the recorded route and may co-occur.

| Primary error | Count |
| --- | ---: |
| exact_route_success | 66 |
| incomplete_tool_chain | 22 |
| parse/evaluation_failure | 5 |
| max_turn_resource_limit | 3 |
| unnecessary_tool | 2 |
| tool_selection | 1 |
| infeasible_detection | 1 |

| Secondary error | Count |
| --- | ---: |
| tool_selection | 19 |
| incomplete_tool_chain | 6 |
| server_selection | 3 |

Primary counts by level are in `error_summary.json`.

## 6. Retrieval

| Metric | Value |
| --- | ---: |
| tasks with find_operations | 91/100 |
| L1–L4 gold operation hit | 77/80 = 0.963 |
| L1–L4 gold hit among tasks that searched | 77/77 = 1.000 |
| L1–L4 tasks that never searched | 3 |
| L1–L4 gold step coverage | 0.943 |
| L1–L4 gold step coverage given a search | 0.980 |
| L1 gold operation hit | 19/20 = 0.950 |
| L2 gold operation hit | 19/20 = 0.950 |
| L3 gold operation hit | 20/20 = 1.000 |
| L4 gold operation hit | 19/20 = 0.950 |
| non-exact tasks whose primary error is retrieval_miss | 0 |

Gold hit means at least one gold operation id appeared in a stored find_operations candidate list. L5 has an empty gold route, so it is outside the hit rate. The 3 L1–L4 tasks with no gold hit never called find_operations (L1-q591, L2-q518, L4-q71). Conditional on a search, every L1–L4 task retrieved at least one gold operation.

## 7. Cost and efficiency

Token totals below are recomputed from `predictions.jsonl`. USD figures are copied from the frozen `cost_summary.json` price configuration (`llm input $0.30 / 1M`, `llm output $1.20 / 1M`, `embedding $0.02 / 1M`). This audit does not invent a second price.

| Metric | Value |
| --- | ---: |
| task LLM calls | 434 |
| verification probe LLM calls | 1 |
| cost_summary llm_calls | 435 |
| mean / median LLM calls per task | 4.340 / 4.000 |
| L1 LLM calls | 90 |
| L2 LLM calls | 104 |
| L3 LLM calls | 92 |
| L4 LLM calls | 99 |
| L5 LLM calls | 49 |
| task input / output / total tokens | 1746891 / 265658 / 2012549 |
| probe input / output tokens | 38 / 32 |
| embedding calls / tokens | 263 / 370102 |
| query embedding calls / tokens | 188 / 1304 |
| catalog embedding calls / tokens | 75 / 368798 |
| catalog cache_reused | True |
| catalog paid on this run | False |
| latency mean / median / p95 ms | 30222.047 / 28587.482 / 60429.346 |
| frozen this_run_estimated_usd | 0.842936 |
| frozen variable_per_task.total_usd | 0.842886 |
| failed-task LLM calls | 44 |
| failed-task input / output / total tokens | 253478 / 54621 / 308099 |
| real_tool_executions | 0 |

Terminal total LLM calls=434 matches the sum of per-task llm_calls. cost_summary.llm_calls=435 equals that sum plus verification_probe.llm_calls=1. Both numbers are reported. Neither number is rewritten.

## 8. Terminal reconciliation

| Claim | Raw artifact | Match |
| --- | --- | --- |
| L1 strict EM 17/20 | 17/20 | True |
| L2 strict EM 12/20 | 12/20 | True |
| L3 strict EM 13/20 | 13/20 | True |
| L4 strict EM 5/20 | 5/20 | True |
| L5 strict EM 19/20 | 19/20 | True |
| overall strict EM 66/100 | 66/100 | True |
| completed 92 | 92 | True |
| failed 8 | 8 | True |
| parse_error 5 | 5 | True |
| max_llm_turns failures 3 | 3 | True |
| task LLM calls 434 | 434 | True |
| cost_summary llm_calls includes probe | 435 | True |

Terminal total LLM calls=434 matches the sum of per-task llm_calls. cost_summary.llm_calls=435 equals that sum plus verification_probe.llm_calls=1. Both numbers are reported. Neither number is rewritten.

## 9. Parser v1 and parser v1.1

Parser v1 is `_parse_agent_json`: strip a surrounding fence when the whole reply starts with ```, then `json.loads`, and require an object.

Parser v1.1 adds only generic, deterministic normalizations: drop a BOM, map smart quotes, strip one fence, take the first balanced object, and delete trailing commas outside strings. The same function is applied to every task. It does not branch on task id, and it does not edit stored responses.

Stored raw assistant strings across 100 tasks: 0.
Strings that fail v1 and parse under v1.1: 0.
Routes changed by v1.1: 0.

| Metric | Parser v1 | Parser v1.1 | Delta |
| --- | ---: | ---: | ---: |
| strict exact-route | 0.66 | 0.66 | 0.0 |
| strict mean F1 | 0.733679 | 0.733679 | 0.0 |
| L1 exact-route | 0.85 | 0.85 | 0.0 |
| L2 exact-route | 0.6 | 0.6 | 0.0 |
| L3 exact-route | 0.65 | 0.65 | 0.0 |
| L4 exact-route | 0.25 | 0.25 | 0.0 |
| L5 exact-route | 0.95 | 0.95 | 0.0 |

Per-task score changes:

- None. v1.1 does not change any task score.

Parse-error tasks that remain failed under both parsers:

- L1-q591
- L2-q518
- L3-q195
- L4-q48
- L4-q71

The five parse failures have no stored model text, so a parser cannot recover a route from this artifact. Their official scores stay failed.

## 10. Proposed method targets

- L4 complete route success is 5/20 = 0.250, with official union node-set exact match 3/20 = 0.150 and EFS-covering node set 5/20 = 0.250. Strict F1 is 0.385. The union node-set is lower because a gold step can list several equivalent tools; predicting one valid equivalent does not equal the full union. Completed L4 misses are mostly incomplete tool chains. dependency_order appears 0 times as a primary or secondary label, so the gap is missing steps rather than a reordering of a complete chain.
- Completed non-exact primary errors: incomplete_tool_chain 22, unnecessary_tool 2, tool_selection 1, infeasible_detection 1. L2 has 7 multi-step equal-function tasks and 0 of them are exact; the 12 L2 exact matches are single-tool tasks. A method should finish every equal-function step, not stop after the first acceptable tool.
- Among L1–L4 tasks that called find_operations, gold-operation hit is 77/77. The 3 tasks without a gold hit never searched; all three died in parse_before_any_action. Primary retrieval_miss count is 0. Gold-step coverage given a search is 0.980, so a few individual gold steps are absent from the candidate lists. That is smaller than the incomplete-chain gap.
- Parser robustness matters for the next run: five tasks stopped on JSON parse errors, and three of them never reached retrieval. Parser v1.1 does not change this run's score because those assistant strings were not stored.
- Three max_llm_turns tasks remain resource_limit failures. L4-q75 used 17 searches and never committed a route. Earlier route commitment is the relevant method change; these three tasks were not rerun.
- L5 exact rejection is 19/20. The one miss, L5-q33, is infeasible_detection from a false tool selection.

## 11. Confirmation

- No live model, embedding, MCP, or paid API client is imported or called by this auditor.
- No task was rerun.
- Raw prediction bytes are unchanged.
- Gold, prompt, registry, formal lock, and run metadata were not written.

