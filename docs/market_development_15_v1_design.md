# Market Development 15 v1 Design

> **Status: `superseded_before_any_agent_run`.** Active file is `market_development_15_v1_1.jsonl`. See `docs/market_development_15_v1_1_design.md`.

- **File:** `benchmarks/questions/market_development_15_v1.jsonl`
- **Split:** `development` / `method_development_and_error_analysis`
- **Catalog:** frozen `market_real_mcp_v1` (13 active ops)
- **Gold method:** direct Real-MCP audit only (`scripts/audit_market_benchmark_real_mcp_v1.py`)
- **No LLM / Agent / Operation-Visible scoring in this pass**

## ETOM distribution

| Level | Count | Role |
| --- | --- | --- |
| L1 | 4 | Direct retrieval (close / volume / OHLC / HSI) |
| L2 | 3 | Previous-valid-day, turnover, adjust compare |
| L3 | 3 | Percentage return, MA20, HSI daily return |
| L4 | 3 | Ann. vol, MDD, equity vs HSI relative |
| L5 | 2 | Confirmed refuse/clarify |

## Coverage

- Names: Tencent, HSBC, Alibaba, HSI
- Fixed-date close, OHLC/range, volume/turnover
- Adjusted vs unadjusted (`""` vs `qfq`)
- Simple / percentage / daily return
- Volatility / MDD / MA
- HSI level + equity vs HSI relative
- Ticker/index identity appears in held-out (Dev focuses on price/calc; L5 covers blocked minute/FTShare)

## Date policy

- All formal dates `<= 2026-08-12`
- Avoid mass single-day `2026-08-12` copies; Dev uses Jul/Jun/May/Mar/Aug-weekend windows
- No `latest` / `current` / `today` as numeric gold (L5 negative)

## Gold ops

L1–L4 gold operations ⊆ active catalog; formal price gold uses:

- `aktools.stock_hk_hist` (default `adjust=""`)
- `aktools.stock_hk_index_daily_sina` (HSI + post-filter)

Blocked FTShare and deferred `stock_hk_hist_min_em` never appear in `gold_operations`.
