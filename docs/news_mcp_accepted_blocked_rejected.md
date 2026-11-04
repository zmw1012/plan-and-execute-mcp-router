# News MCP Accepted / Blocked / Rejected (candidate_v1)

## Accepted primary

| operation_id | source_class | Notes |
|---|---|---|
| `hkexnews.list_hkexnews_announcements` | official_disclosure | Date window + timestamps + URL |
| `hkexnews.download_hkexnews_announcement` | official_disclosure | Field consumer of list |
| `hkexnews.parse_hkexnews_announcement` | official_disclosure | Required before search |
| `hkexnews.keyword_search` | official_disclosure | `keywords: list`; snippets |
| `hkexnews.get_announcement_pages` | official_disclosure | `local_key` + `start_page` |

## Accepted supporting

| operation_id | source_class | Notes |
|---|---|---|
| `hkexnews.lookup_stock_code` | official_disclosure | Identity helper |
| `aktools.search_interfaces` | metadata_only | Discovery meta |
| `aktools.describe_interface` | metadata_only | Discovery meta |
| `aktools.stock_news_em` | financial_media | Live; **not** historical cutoff gold |
| `ah_disclosure.resolve_company` | metadata_only | market/symbol normalize |
| `ah_disclosure.search_annual_report` | official_disclosure | Title/URL/date only |

## Distractors

| operation_id | Why |
|---|---|
| `ah_disclosure.search_filings` | Date window not honored in live probes |
| `aktools.stock_notice_report` | A-share board |
| `aktools.stock_gsrl_gsdt_em` | A-share event calendar |
| `aktools.stock_info_global_em` | Non-issuer global flash |
| `aktools.news_cctv` | Macro CCTV transcript |
| `aktools.stock_news_main_cx` | Caixin feed, no issuer/date filter |

## Blocked

| operation_id | Why |
|---|---|
| `ftshare.ft_v1_report_announcements_list` | No free reproducible data path (API key / quota) |

## Rejected

| operation_id | Why |
|---|---|
| `aktools.stock_individual_notice_report` | Live failure on HK code |
| `mock.news_server` | Offline mock, not Real-MCP |
