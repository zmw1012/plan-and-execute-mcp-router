# News MCP Capability Matrix (Discovery v1)

Temporal policy: **max gold date = 2026-08-12**, timezone **Asia/Hong_Kong**.  
Legend: ✅ live-verified usable for gold · ⚠️ usable with constraints · ❌ unsupported/blocked · D distractor

## Capability × operation

| # | Capability | Primary path | Supporting | Distractor / blocked / gap |
|---|---|---|---|---|
| 1 | Search by stock code | ✅ `hkexnews.list_*` | ⚠️ `stock_news_em` (floating) | D A-share notice |
| 2 | Search by company name | ⚠️ media keyword via `stock_news_em` | — | ❌ no official name-search equal to code list |
| 3 | Date-range filter | ✅ hkexnews `start_date`/`end_date` | — | D ah `search_filings` (date filter **unreliable**) |
| 4 | Accurate publication timestamp | ✅ `release_datetime` / `release_date` | ⚠️ media `发布时间` (recent only) | — |
| 5 | Source name + URL | ✅ `file_link` (HKEXnews) | ⚠️ media `文章来源`+`新闻链接` | — |
| 6 | Headline | ✅ `title` | ⚠️ media `新闻标题` | — |
| 7 | Body / snippet / citable fragment | ✅ parse→keyword snippet / pages | ⚠️ media `新闻内容` snippet | — |
| 8 | HKEX official announcement list | ✅ list | ⚠️ ah annual title discovery | D ah filings (date-unsafe) |
| 9 | Download / parse / keyword / page evidence | ✅ download→parse→keyword→pages (field deps) | — | — |
| 10 | Dedup | ⚠️ offline on `news_id`/`local_key` | — | ❌ no dedicated MCP dedup tool |
| 11 | Chinese + English | ✅ HKEX EN titles live; ZH via lang/local | ⚠️ media mostly ZH | — |
| 12 | Chronological ordering | ✅ list sorted by DateTime | ⚠️ media provider order | — |
| 13 | Latest-before-cutoff | ✅ max `release_datetime` ≤ cutoff in window | ❌ `stock_news_em` empty before cutoff | D ah filings latest ignores cutoff |
| 14 | Fail closed (empty / future / gap) | ✅ invalid code error; policy on post-cutoff | ⚠️ empty historical media → refuse | ❌ FTShare blocked |

## Issuer coverage

| Operation | Tencent | HSBC | Alibaba | Notes |
|---|---|---|---|---|
| hkexnews list (Jul1–Aug12 2026) | ✅ | ✅ | ✅ | official_disclosure |
| hkexnews download/parse/keyword | sample OK (HSBC) | ✅ | chain transferable | same server deps |
| `stock_news_em` | ✅ floating / 0≤cutoff | ✅ / 0 | ✅ / 0 | financial_media |
| ah `search_annual_report` | ✅ | prior | prior | title/URL only |
| A-share notice / gsrl | n/a | n/a | n/a | distractors |

## Timestamp / source / content fidelity

| Path | Timestamp | Source URL | Content |
|---|---|---|---|
| HKEXnews list | `release_date` + `release_datetime` | `file_link` | metadata; body via download chain |
| HKEXnews keyword | n/a | via local_key doc | `snippet` + `page_number` |
| `stock_news_em` | `发布时间` (recent) | `新闻链接` | `新闻内容` snippet |
| ah annual search | `publish_time` | `pdf_url` | title/metadata (not parsed here) |

## Counts (candidate catalog)

| Class | Count |
|---|---|
| primary | 5 |
| supporting | 6 |
| distractor | 6 |
| blocked | 1 |
| rejected | 2 |
| **logical ops** | **20** |
| equal_function_sets | **0** |

## Remaining gaps

1. **No free historical HK media archive** with date-range filter (East Money feed is floating ~10).
2. **No verified media↔disclosure dedup** tool across servers.
3. **ah `search_filings` date window untrustworthy** — cannot replace hkexnews list for cutoff gold.
4. **Company-name → official disclosure** without stock code is weak (lookup expects code).
5. **FTShare** announcement APIs remain blocked without key.
6. Full Annual Report **numeric** reading stays Financial — News only uses title/date/URL/event tasks.
7. Explicit multilingual ZH list smoke not exhaustively paired for every title (EN list verified live).
