# Market MCP Capability Matrix (Discovery v1)

Temporal policy: **max gold date = 2026-08-12**, timezone **Asia/Hong_Kong**, previous-valid-trading-day rule documented in candidate YAML.

Legend: ✅ live-verified usable · ⚠️ usable with constraints · ❌ unsupported/blocked · D distractor

## Capability × operation

| Capability | Primary path | Supporting / overlap | Distractor / blocked |
|---|---|---|---|
| 1 Historical daily OHLC | ✅ `aktools.stock_hk_hist` | ⚠️ `stock_hk_daily` (neg offset + filter) | D spot boards |
| 2 Volume & turnover | ✅ hist `成交量`/`成交额` | ⚠️ daily `volume`/`amount` | — |
| 3 Fixed-date close | ✅ hist date filter | ⚠️ daily + filter `<=cutoff` | ❌ floating spot |
| 4 Date-range series | ✅ hist `start_date`/`end_date` | ⚠️ daily via paging | — |
| 5 Simple return inputs | ✅ hist closes | ⚠️ daily closes | — |
| 6 Multi-period return inputs | ✅ hist ranges | ⚠️ daily | — |
| 7 Volatility inputs | ✅ hist close series | ⚠️ daily | — |
| 8 Max-drawdown inputs | ✅ hist close series | ⚠️ daily | — |
| 9 Moving-average inputs | ✅ hist close series | ⚠️ daily | — |
| 10 Trading-day/calendar | ⚠️ derive from hist sessions | D `tool_trade_date_hist_sina` (likely SSE) | — |
| 11 Adjusted vs unadjusted | ✅ hist `''|qfq|hfq` | ✅ daily `''|qfq` + `qfq-factor` | — |
| 12 Corporate actions / factors | ⚠️ daily `qfq-factor` | ❌ FTShare adjust_factor (429) | — |
| 13 HSI history | ✅ `stock_hk_index_daily_sina` | ✅ `stock_hk_index_daily_em` | D `stock_hk_gxl_lg` yield |
| 14 Equity vs index inputs | ✅ hist + HSI history | spot only for code resolve | D yield/spot |
| 15 Ticker identity | ✅ 5-digit codes live | ✅ profile / index spot HSI | D eniu `hk#####` format |
| 16 Explicit failure unavailable | ✅ invalid symbol errors | ✅ invalid index errors | — |
| 17 Distractors present | — | — | D spot/hot/yield/valuation/calendar/minute; blocked FTShare |

## Issuer coverage (primary hist)

| Operation | Tencent 00700 | HSBC 00005 | Alibaba 09988 |
|---|---|---|---|
| `stock_hk_hist` | ✅ | ✅ | ✅ |
| `stock_hk_daily` (recent via offset&lt;0) | ✅ | ✅ | ✅ |

## Index coverage

| Operation | HSI code resolved | 2026-08-12 row | Post-cutoff filter needed |
|---|---|---|---|
| `stock_hk_index_spot_sina` | ✅ `HSI` | spot only (not gold) | n/a |
| `stock_hk_index_daily_sina` | ✅ | ✅ close≈25440.17 | yes (neg-offset windows include later dates) |
| `stock_hk_index_daily_em` | ✅ | ✅ latest≈25440.17 | yes |

## Counts (candidate catalog)

| Class | Count |
|---|---|
| primary | 2 |
| supporting | 6 |
| distractor | 6 |
| blocked | 2 |
| rejected | 0 |
| **logical ops in YAML** | **16** |
| AKTools raw MCP tools | 3 |
| FTShare raw MCP tools listed | 250 (data blocked) |

## Remaining gaps

1. No free/live corporate-action **event** feed (factors only via qfq-factor).
2. No verified **HKEX official** trading calendar interface (SSE calendar is a distractor).
3. Intraday minute history not reliable in probe window.
4. FTShare remains blocked without API key (do not accept for formal gold).
5. Currency not an explicit field on OHLCV rows (convention HKD).
6. Calculations (return/vol/MDD/MA) intentionally left as offline transforms over raw inputs — no separate calc MCP tools added.
