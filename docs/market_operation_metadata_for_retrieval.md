# Market Operation Metadata for Operation Visible Retrieval

**Phase:** discovery only. **No changes** to `retrieval.py` / `schemas.py`.

This document maps each accepted/conditionally accepted logical Market operation to:

1. the **current generic Operation Visible document fields**;
2. **Market-specific metadata** that is not yet part of the generic index, for later comparison between generic OV retrieval and Market/Finance-aware retrieval.

## Generic OV fields (current)

For each logical operation, OV already indexes approximately:

| OV field | Market candidate usage |
|---|---|
| summary | short capability sentence |
| service/server ID | `aktools` (FTShare blocked ops documented but not accepted) |
| operation name | `operation_id` / interface name |
| action terms | price/history/HSI/adjust/… |
| required/optional parameter signatures | symbol, dates, adjust, offset, max_rows |
| constraints | cutoff filter, neg-offset guidance, invalid-symbol errors |
| description | full description from candidate YAML |
| invocation guidance | get_data routing + adjust/cutoff rules |

## Market-specific metadata (proposed extras; not wired)

| Extra field | Why it matters for Market selection |
|---|---|
| `ticker_exchange_coverage` | HKEX 5-digit vs index codes vs `hk#####` distractors |
| `historical_vs_spot` | Prevent selecting spot boards for fixed-date gold |
| `date_range_support` | hist has start/end; daily needs offset paging |
| `frequency` | daily / minute / spot |
| `adjustment_mode` | `''` / `qfq` / `hfq` / `qfq-factor` / n/a |
| `currency` | HKD vs index points vs percent yield |
| `timezone` | Asia/Hong_Kong trading dates |
| `field_family` | OHLC / volume / turnover / yield / valuation / weights |
| `benchmark_index_support` | HSI history yes/no |
| `source_reliability` | live-verified free vs blocked quota |
| `max_gold_date_policy` | clamp to 2026-08-12 |
| `previous_valid_trading_day_rule` | weekend/holiday handling |

## Per-operation retrieval document drafts

### `aktools.stock_hk_hist` (primary)

**OV document (generic):**  
summary=HK equity historical OHLCV with date range + adjust; server=`aktools`; name=`stock_hk_hist`; action_terms=[price,history,OHLCV,daily,close,volume,turnover,复权]; params=`symbol,period,start_date,end_date,adjust`; constraints=explicit adjust; clamp dates<=2026-08-12; description=East Money hist via get_data; invocation=get_data(interface=stock_hk_hist).

**Extras:** historical; daily; adjust modes `''|qfq|hfq`; currency HKD; TZ Asia/Hong_Kong; field_family=OHLCV+turnover; benchmark_index_support=false; reliability=live_verified_three_issuers.

### `aktools.stock_hk_daily` (supporting overlap)

**OV:** Sina full-history daily / qfq / qfq-factor; constraints=use negative offset for recent rows; filter <=cutoff; invocation=get_data(stock_hk_daily).

**Extras:** historical; daily; adjust `''|qfq|qfq-factor`; overlaps hist; reliability=live_verified_three_issuers.

### `aktools.stock_hk_index_daily_sina` (primary index)

**OV:** HSI daily history; params=`symbol=HSI`; constraints=negative offset + filter cutoff; action_terms=[HSI,Hang Seng,index,benchmark].

**Extras:** historical; index_points; benchmark_index_support=true; field_family=OHLCV.

### `aktools.stock_hk_index_daily_em` (supporting index)

**OV:** EM HSI history; close field=`latest`; constraints=map latest→close; filter cutoff.

**Extras:** overlaps Sina HSI; reliability=live_verified.

### `aktools.stock_hk_index_spot_sina` (supporting resolve)

**OV:** resolve HSI code from spot list; constraints=do not use 最新价 as gold.

**Extras:** spot; historical_vs_spot=spot.

### `aktools.stock_hk_security_profile_em` (supporting identity)

**OV:** security profile for HK codes.

**Extras:** identity metadata; not prices.

### Meta `search_interfaces` / `describe_interface` (supporting)

**OV:** discovery/describe helpers for Market interfaces.

**Extras:** meta; not time series.

### Distractors (indexed to test selection)

| operation_id | Distractor signal to encode |
|---|---|
| `stock_hk_famous_spot_em` | spot/latest |
| `stock_hk_hot_rank_em` | rank/popularity |
| `stock_hk_gxl_lg` | HSI **yield** not level |
| `stock_hk_indicator_eniu` | valuation; symbol format `hk#####` |
| `tool_trade_date_hist_sina` | calendar likely SSE |
| `stock_hk_hist_min_em` | minute/intraday |

Blocked FTShare ops should **not** enter the accepted OV registry for formal runs; keep them in audit docs only unless credentials become available.

## Later experiment contrast (planned, not built)

1. **Generic OV retrieval** using only current fields.  
2. **Market-aware retrieval** boosting `historical_vs_spot`, `adjustment_mode`, `benchmark_index_support`, and cutoff constraints.  

No retrieval code changes in this discovery phase.
