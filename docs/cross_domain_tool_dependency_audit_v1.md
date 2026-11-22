# Cross-domain Tool Dependency Audit v1

## Exact L4 field edges (live-verified)

All Exact L4 items use:

`hkexnews.list_hkexnews_announcements.release_date`
→ `aktools.stock_hk_hist.start_date` / `end_date`

(with `release_date` YYYY-MM-DD mapped to hist `YYYYMMDD`).

| ID | Issuer | Disclosure used | Session close |
|---|---|---|---|
| xd_dev_007 | Tencent | Monthly Return 2026-08-06 | 479.2 |
| xd_dev_009 | Tencent | Results ann. 2026-08-12 (+ independent 1H revenue) | 461.6 |
| xd_ht_009 | Alibaba | DATE OF BOARD MEETING 2026-08-07 | 123.8 |
| xd_ht_010 | HSBC | Transaction in own shares 2026-08-12 | 161.4 |
| xd_ht_013 | Tencent | Monthly Return 2026-08-06 (+ independent FY EPS) | 479.2 |

Market legs audited in `docs/cross_domain_discovery_v1/market_cross_domain_smoke.json`.

## Non-Exact multi-domain items

Most remaining items are `independent_parallel`: each domain answerable alone, then composed. Marked `exact_etom_alignment=false` / `etom_alignment=not_applicable` even if `etom_level` slot shows L4 for project bookkeeping.

## Same-server cross-domain note

Some Financial + News pairs use only `mcp_hkexnews` (e.g., HSBC EPS keyword + list). They remain **cross-domain** by task definition (Financial metric vs News disclosure metadata) but are **not** Exact L4 unless a cross-domain field edge exists.
