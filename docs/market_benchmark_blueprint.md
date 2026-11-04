# Market Benchmark Blueprint (Next Phase Only)

**Status:** proposal for user review after capability discovery.  
**Does not contain** final Development/Held-out question wording or final gold answers.

## Planned split sizes

| Split | Questions | Role |
|---|---|---|
| Development | **15** | method development / error analysis |
| Held-out | **20** | final evaluation only (locked later) |

## Candidate ETOM distribution (proposal)

### Development (15)

| Level | Count | Intent |
|---|---|---|
| L1 | 4 | single-tool explicit hist/index retrieval |
| L2 | 3 | adjust-mode or hist-vs-daily discrimination |
| L3 | 3 | multi-step returns / range-from-series |
| L4 | 3 | equity + HSI cross-series comparison |
| L5 | 2 | unavailable date/symbol or unsafe spot-as-gold refusal |

### Held-out (20)

| Level | Count |
|---|---|
| L1 | 4 |
| L2 | 4 |
| L3 | 4 |
| L4 | 4 |
| L5 | 4 |

(Exact held-out wording deferred until user accepts the capability matrix.)

## Task-type distribution (templates)

| Template | Dev | Held-out | Supported by |
|---|---|---|---|
| Fixed-date close (unadjusted) | 2 | 3 | `stock_hk_hist` |
| Date-range OHLC retrieval | 1 | 2 | `stock_hk_hist` |
| Volume/turnover on date | 1 | 1 | `stock_hk_hist` |
| Adjusted vs unadjusted discrimination | 2 | 2 | hist/daily adjust |
| Simple return between two sessions | 2 | 2 | hist closes (calc offline) |
| Multi-day volatility / MDD / MA inputs | 2 | 3 | hist series (calc offline) |
| HSI fixed-date level | 1 | 2 | index_daily_sina/em |
| Equity vs HSI relative return | 2 | 2 | hist + HSI (L4) |
| Ticker/index identity resolve | 1 | 1 | profile / index spot |
| L5 refusal/clarification | 2 | 4 | see below |

Company balance target (across both splits eventually): mix of Tencent / HSBC / Alibaba; include ≥1 HSI-focused item per split.

## Required gold fields (for future question JSONL)

Reuse Financial-style discipline where applicable:

- `id` / `question_id`
- `question`, `domain=market`
- `company` / `ticker` or `index_symbol`
- `etom_level`, `task_type`, `answerability_class`
- `session_date` or `session_dates` (≤2026-08-12)
- `adjustment_mode` (explicit)
- `timezone=Asia/Hong_Kong`
- `currency` or `unit` (HKD / index points)
- `gold_values` + tolerances
- `gold_server_ids`, `gold_operations`, `expected_operation_sequence`
- `previous_valid_trading_day_applied` (bool + rule citation when used)
- `dataset_split`, `benchmark_role`, `locked` (held-out)
- evidence notes for live recon (not formal Annual Reports)

**Do not** put gold into agent prompts.

## Capabilities supporting each template

| Template | Capability IDs from matrix |
|---|---|
| Fixed-date close | 1,3,11,15,16 |
| Range OHLC | 1,4,2 |
| Returns / vol / MDD / MA | 5–9 over raw closes |
| Adjust discrimination | 11,12 |
| HSI level / relative | 13,14 |
| Identity | 15 |
| L5 | 16 + distractor traps (17) |

## Unsupported / weak templates (do not force into L1–L4 yet)

- Official HKEX corporate-action event narrative Q&A
- Minute-bar microstructure gold
- FTShare-only weights/candles while blocked
- Using SSE calendar as HK session authority

## Likely L5 refusal / clarification categories

1. **Unavailable session** — date after cutoff, or non-trading day without allowing previous-valid-day (when question forbids substitution).  
2. **Unsafe spot substitution** — “current/latest price” when fixed historical gold is required.  
3. **Wrong adjustment semantics** — asking for unadjusted but only accepting hfq without disclosure.  
4. **Index identity confusion** — HSI yield (`stock_hk_gxl_lg`) requested as HSI level.  
5. **Unsupported venue/symbol** — non-HK or invalid code under frozen catalog.  
6. **Ambiguous ticker format** — `0700.HK` vs `00700` vs `hk00700` without resolution rules stated.

## Next-phase gate

Stop here until the user reviews:

1. `docs/market_mcp_capability_matrix.md`
2. `configs/operations/market_real_mcp_candidate_v1.yaml`

Only then author final Market Development 15 / Held-out 20 questions.
