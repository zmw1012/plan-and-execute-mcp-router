# Market Held-out Test 20 v1 Design

- **File:** `benchmarks/heldout/market_test_20_v1.jsonl`
- **Flags:** `locked=true`, `frozen=true`, `dataset_split=heldout`, `benchmark_role=final_evaluation_only`
- **Catalog:** frozen `market_real_mcp_v1`
- **Gold method:** direct Real-MCP audit only; **do not** place gold into future model prompts
- **No Agent / LLM run against held-out in this pass**

## ETOM distribution

| Level | Count |
| --- | --- |
| L1 | 4 |
| L2 | 4 |
| L3 | 4 |
| L4 | 4 |
| L5 | 4 |

## Separation from Development 15

- Different calendar dates / windows / task combinations (not company/date swaps only)
- Validator enforces: exact normalized duplicate = 0; high-overlap near-duplicate (Jaccard ≥ 0.85) = 0
- Examples of distinct tasks: Feb HSBC close, Jan Alibaba volume, Apr Tencent OHLC, May Sunday previous-valid, `hfq` vs unadj (Dev used `qfq`), high−low range, security profile identity, May HSBC % return, Mar Alibaba MA10, Jun HSI simple return, Feb turnover sum, Apr HSBC vol, Jan–Feb Tencent MDD, Jul Alibaba−HSI relative, Mar MA5+return combo

## L5 held-out (confirmed infeasible)

1. “current … right now”
2. `tool_trade_date_hist_sina` as official HKEX calendar
3. Post-cutoff date `2026-08-20`
4. FTShare `stock_hk_daily` as gold

## Usage

Held-out is for **final evaluation only**. Development/error analysis must use `market_development_15_v1.jsonl`.
