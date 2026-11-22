# Market Real-MCP Candidate Audit (Discovery v1)

**Branch:** `feature/real-market-mcp-discovery-v1`  
**Catalog status:** `candidate_v1` (not frozen)  
**Candidate file:** `configs/operations/market_real_mcp_candidate_v1.yaml`  
**Raw outputs:** `outputs/mcp_discovery_market_v1/` (gitignored)  
**Method:** direct MCP protocol calls only — no generation LLM / embedding / agent experiment.

## Research answers (short)

| # | Question | Finding |
|---|---|---|
| 1 | Which real MCP servers expose reproducible HK equity market data? | **AKTools** (`aktools` 0.1.4 / `akshare` 1.20.22, stdio) live-verified. **FTShare-MCP** implements MCP (250 tools) but Market data calls are **blocked** by anonymous 429 / missing API key. |
| 2 | Historical OHLCV? | Yes — `stock_hk_hist` (primary) and `stock_hk_daily` (supporting overlap). |
| 3 | Adjusted vs unadjusted? | Yes — explicit `adjust` (`''`/`qfq`/`hfq` on hist; `''`/`qfq`/`qfq-factor` on daily). |
| 4 | Hang Seng Index history? | Yes — symbol `HSI` via `stock_hk_index_daily_sina` / `_em` using **negative get_data offset** + filter `<=2026-08-12`. |
| 5 | Volume/turnover/currency/dates consistent? | Volume+turnover present on hist/daily/Sina index. Currency not a dedicated column — treat equities as **HKD** by HK feed convention; index as points. Dates are session calendar dates. |
| 6 | Corporate actions / factors? | Partial — `stock_hk_daily` `adjust=qfq-factor` returns factor series; FTShare adjust-factor tool schema exists but data blocked. |
| 7 | Useful / overlapping / distracting / unsafe? | See capability matrix; spot/hot/yield/valuation/calendar listed as distractors; FTShare blocked. |
| 8 | OV metadata needed? | See `docs/market_operation_metadata_for_retrieval.md`. |

## Servers audited

### A. AKTools MCP — **accepted primary server**

| Field | Value |
|---|---|
| Package / version | `aktools==0.1.4` (live) |
| Upstream data | `akshare==1.20.22` |
| Repo | https://github.com/akfamily/aktools |
| Implements MCP | Yes (stdio) |
| Raw MCP tools | **3**: `search_interfaces`, `describe_interface`, `get_data` |
| Auth / cost | None / free local |
| License | MIT |
| Live smoke | Pass for HK hist/daily/index/spot/meta |

**Important counting distinction:** the 3 raw MCP tools gate hundreds of AKShare **logical interfaces**. Market candidate operations name those interfaces under `get_data`.

### B. FTShare-MCP — **blocked**

| Field | Value |
|---|---|
| Endpoint | `https://market.ft.tech/gateway/mcp` |
| Transport | Streamable HTTP |
| Raw tools listed | **250** (live `tools/list`) |
| HK Market tools | `ft_v2_hk_hk_candlesticks`, `ft_v1_hk_hsi_daily_weight`, `ft_v1_hk_stk_ah_comparison` (+ related adjust/candles) |
| Live data calls | `UPSTREAM_UNAVAILABLE` / HTTP 429 anonymous quota (“最近 1 天最多访问 500 次”) |
| Local API key | Absent (`FTSHARE_API_KEY` not used; `.env` untouched) |
| Decision | **Blocked** — not reproducible for free/local formal gold |

### C. Other candidates considered (not live-accepted)

| Candidate | Why not accepted now |
|---|---|
| `mcp-aktools` (aahl) | Install failed in discovery Python 3.9 venv (no distribution); not live-listed/called here |
| `vearne-akshare-mcp` | README claims HK support; **not** installed/live-audited this round (avoid unverified wrappers) |
| Project mock `MarketServer` | Not Real-MCP; offline mock only |

## Live coverage (fixed date 2026-08-12, Asia/Hong_Kong)

| Issuer | Code | `stock_hk_hist` unadj close | Notes |
|---|---|---|---|
| Tencent | 00700 | 461.6 | qfq matched unadj on this date |
| HSBC | 00005 | 161.4 | qfq close ≈160.616 (materially different) |
| Alibaba | 09988 | 122.6 | qfq matched unadj on this date |

HSI (`stock_hk_index_daily_sina`): 2026-08-12 close ≈ **25440.17** (EM `latest` reconciles).

Weekend evidence for previous-valid-day rule: hist 00700 for 2026-08-07..12 → sessions **07,10,11,12** (08/09 absent).

## Reconciliation (aligned modes)

On 2026-08-12, Sina `stock_hk_daily` vs EM `stock_hk_hist` OHLC agree within float noise (~1e-4) for unadjusted/qfq samples across the three names. Amounts can differ slightly. Do **not** compare hfq vs unadjusted as inconsistency.

## Candidate catalog summary

See YAML `counts` section. **Not frozen.** Enough overlapping ops (`hist` vs `daily`, Sina vs EM index) and distractors (spot/hot/yield/valuation/calendar/minute) to stress operation selection later.

## Explicit non-actions

- Financial catalogs/benchmarks/docs **unchanged**
- No Market final Development/Held-out questions authored
- No OV retrieval code changes
- No proposed-method implementation
- No paid LLM/embedding calls
