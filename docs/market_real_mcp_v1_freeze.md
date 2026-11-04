# Market Real-MCP v1 Freeze

- **Catalog:** `configs/operations/market_real_mcp_v1.yaml`
- **Candidate source:** `configs/operations/market_real_mcp_candidate_v1.yaml`
- **Cutoff:** `2026-08-12` (`Asia/Hong_Kong`)
- **Gateway:** `hk_equity_mcp.real_mcp.market_gateway.MarketRealMcpGateway`
- **No Agent / LLM / embedding / paid API in this freeze pass**

## Schema layers

| Layer | Meaning |
| --- | --- |
| raw MCP tools | `search_interfaces`, `describe_interface`, `get_data` |
| logical operations | `operations[].operation_id` (curated contracts) |
| active searchable operations | only `operations[]` in the frozen YAML (OV registry) |
| blocked / deferred | `blocked_operations` + `deferred_operations` (audit only) |

## Active role distribution

| Role | Count | Notes |
| --- | --- | --- |
| primary | 2 | Formal gold price: `stock_hk_hist`, `stock_hk_index_daily_sina` |
| supporting | 6 | Discovery, alternate series, identity/spot helpers |
| distractor | 5 | spot / hot rank / HK yield / valuation / non-HK calendar |
| **total active** | **13** | |

## Excluded from active registry

| Operation | Status | Reason |
| --- | --- | --- |
| `ftshare.hk_hist` | blocked | Streamable HTTP 429 / unreliable; never gold |
| `ftshare.stock_hk_daily` | blocked | Streamable HTTP 429 / unreliable; never gold |
| `aktools.stock_hk_hist_min_em` | deferred | Empty-window evidence only under cutoff |

## Frozen rules

1. **`stock_hk_hist`:** explicit `start_date`/`end_date`; explicit `adjust`; formal gold default `adjust=""`; no auto-mix of qfq/hfq.
2. **`stock_hk_daily` / HSI negative-offset tools:** MCP response **must** be post-filtered to `date <= 2026-08-12`.
3. **Previous-valid-day:** `max(available session date <= requested date)` (and `<= cutoff`). **No forward-fill** from future sessions.
4. **HKD:** when inferred from exchange convention, record `currency_provenance=exchange_convention` (not an explicit API field).
5. **`tool_trade_date_hist_sina`:** distractor only; **not** HKEX official calendar; previous-valid-day uses price sessions.
6. **Fail-closed:** missing SDK/binary/catalog/operation/required data → structured error; **never** silent fallback to mock/`FinancialServer`.

## Calculation definitions (tested)

- Simple return: `close_end / close_start - 1`
- Percentage return: `100 * (close_end / close_start - 1)`
- Daily return: `P_t / P_(t-1) - 1`
- Annualized volatility: `sample_std(daily simple returns, ddof=1) * sqrt(252)`
- Maximum drawdown: `min(P_t / running_max(P_1...P_t) - 1)`
- Moving average N: arithmetic mean of the most recent N valid **unadjusted** closes **including** the target session

Every scored question states: percent vs decimal, rounding, tolerance, adjustment mode, price field, date window.

## Wiring (no Agent run in this pass)

```python
from hk_equity_mcp.real_mcp import MarketRealMcpGateway, load_real_mcp_registry
from hk_equity_mcp.operation_resolver.server import OperationResolverServer

catalog = "configs/operations/market_real_mcp_v1.yaml"
gw = MarketRealMcpGateway(catalog_path=catalog).open()
registry = load_real_mcp_registry(catalog)
# Pass gw as real_mcp_gateway to OperationResolverServer when running OV later.
```

## Benchmarks

- Development 15: `benchmarks/questions/market_development_15_v1.jsonl`
- Held-out 20: `benchmarks/heldout/market_test_20_v1.jsonl`
- Provenance: `docs/market_real_mcp_provenance_audit_v1.md`
