# ETOM-100 Formal External Sample v1

Equal-level stratified external validation sample for Operation-Visible
routing-only evaluation. This is **not** a population-proportional draw from
the ETOM query mix, and the pooled 100-task result must **not** be described as
an ETOM natural-distribution population estimate.

## Source pin

| Field | Value |
| --- | --- |
| Repository | https://github.com/snooow1029/ETOM |
| Commit | `790f28727f567b88918420d163c524d46644a694` |
| Source pin | `benchmarks/etom/etom_source_pin_v1.json` |
| Source pin SHA256 | `5f1ae09e7a8fbb7b554ed1302433feacaffb8645ccc560c7bb87ff64893f2692` |
| Registry tools | 2375 |
| Official queries | L1 781, L2 773, L3 327, L4 103, L5 91 (total 2075) |

Official JSON file hashes match the source pin. The checkout is not re-downloaded
or retargeted in this freeze.

## Sampler

| Field | Value |
| --- | --- |
| Sampler version | `etom_stratified_external_100_v1` |
| Seed | `20261106` |
| Quotas | L1=20, L2=20, L3=20, L4=20, L5=20 |
| Sample size | 100 |
| Lock | `benchmarks/etom/formal_100_v1.lock.json` |
| `sample_sha256` | `99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1` |
| `ordered_task_ids_sha256` | `3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479` |

### Declared constraints

1. Exclude every task id in `benchmarks/etom/cost_pilot_10_v1.lock.json`.
2. Per level: sort eligible ids by `task_id`, then draw with fixed seed.
3. Keep official ETOM level labels; do not re-label difficulty.
4. L1: single-step only (`step_count == 1`).
5. L3: all selected tasks are official single-server sequential dependency tasks.
6. L4: all selected tasks are official cross-server dependency tasks.
7. L5: official infeasible empty-route tasks only.
8. Within each level, hard cap `max_server_frequency=2`, plus seeded weights that
   prefer not-yet-selected servers.
9. If a constraint cannot be met, sampling fails closed (no silent relaxation).

## Official population vs formal sample

| Level | Official count | Formal sample |
| --- | ---: | ---: |
| L1 | 781 | 20 |
| L2 | 773 | 20 |
| L3 | 327 | 20 |
| L4 | 103 | 20 |
| L5 | 91 | 20 |
| Total | 2075 | 100 |

This is an **equal-level stratified** sample. Later result reporting must include:

- macro average across the five levels
- per-level results
- pooled 100-task result

The pooled figure summarizes this stratified freeze only. It is not an estimate
of ETOM’s natural level mix.

## Exclusion / leakage checks

| Check | Result |
| --- | --- |
| Cost-pilot lock SHA256 | `bb0dff91fe999b86b21162617c8a62b3b067c0a18b86227d4308068a94abf065` |
| Pilot overlap | **0** |
| Exact duplicate task ids | **0** |
| Selection based on pilot correctness | **No** (ids excluded before sampling; no model scores used) |

### Anti-leakage statement

- The cost-pilot 10 tasks were exposed only for cost accounting and runner validation.
- The formal 100 has `overlap=0` with that pilot lock.
- The formal 100 is frozen after this sampling step.
- ETOM-100 must not be used for per-item tuning of a Proposed Method.
- Proposed Method design must be driven primarily by the HK-155 development error taxonomy.
- ETOM-100 is reserved for external comparison of Operation-Visible against the
  final frozen method.

## Coverage statistics

| Metric | Value |
| --- | --- |
| Unique servers in sample | 138 |
| Tasks with dependency | 40 |
| L3 single-server sequential | 20 / 20 |
| L4 cross-server dependency | 20 / 20 |
| L5 official empty-route infeasible | 20 / 20 |

### Top server frequencies (sample-wide)

| Server | Count |
| --- | ---: |
| CCXT MCP Server | 2 |
| Financial Modeling Prep MCP Server | 2 |
| Outlook Meetings Scheduler MCP Server | 2 |
| Outlook Calendar MCP Tool | 2 |
| Software Planning Tool | 2 |
| Puppeteer MCP Server | 2 |
| playwright-server MCP server | 2 |
| mcp-tavily-search | 2 |
| Enhanced Documentation Search MCP Server | 2 |
| FetchSERP MCP Server | 2 |
| cognee MCP server | 2 |
| RAG Documentation MCP Server | 2 |
| SkyDeckAI Code | 2 |
| Baidu Cloud AI Content Safety MCP Server | 2 |
| WSB Analyst MCP Server | 1 |

Per-level max server frequency stays at most 2 by construction.

### Step-count distribution

| Steps | Tasks |
| ---: | ---: |
| 0 | 20 |
| 1 | 33 |
| 2 | 9 |
| 3 | 10 |
| 4 | 14 |
| 5 | 11 |
| 6 | 2 |
| 7 | 1 |

### L5 subtype distribution

Official `level_5.json` rows carry no subtype labels. Every gold route is an
empty tool path. In this freeze:

| Subtype | Count |
| --- | ---: |
| `official_empty_ground_truth_route` | 20 |

No custom refusal items were authored.

## Example tasks (for human understanding only)

These examples are not selection criteria and must not be used to swap ids.

### L1

- `L1-q556` — CCXT recent BTC/USDT trades (1 step).
- `L1-q160` — WallStreetBets unique external-link extraction (1 step).
- `L1-q267` — Atlassian Jira project issue retrieval (1 step).

### L2

- `L2-q766` — schedule a multi-attendee meeting (equivalence-style multi-server route).
- `L2-q702` — mark multiple tasks completed across todo servers.
- `L2-q492` — accordion component example via Shadcn UI v4.

### L3

- `L3-q108` — TrueNAS sequential pool/status chain.
- `L3-q302` — MindManager sequential library/version/document chain.
- `L3-q267` — Rootly sequential incident/alert attachment chain.

### L4

- `L4-q57` — cross-server architecture/documentation dependency graph.
- `L4-q4` — Nefino + Voyp cross-server supply-chain analysis route.
- `L4-q37` — documentation/image cross-server presentation route.

### L5

- `L5-q68` — adjust bedroom thermostat (official empty route).
- `L5-q4` — recommend a jazz streaming platform (official empty route).
- `L5-q39` — gift-card balance/extension (official empty route).

## Evaluation contract

ETOM here remains **routing-only evaluation**:

- `real_mcp_tool_executed=false`
- scores measure agreement with the official tool route / refusal, not live
  external MCP business execution success

Catalog embeddings reuse the already-built 2,375-operation complete index.
Formal live runs require that complete index and must not rebuild it.
