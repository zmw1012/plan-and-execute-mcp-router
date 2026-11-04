# Financial Taxonomy Audit v1.3

**Scope:** Development / Held-out `*_v1_2` → `*_v1_3` executable dependency-graph correction  
**Catalog:** frozen Financial Real-MCP **unchanged**  
**Numerical gold / question / reference_answer:** unchanged  
**No Agent / LLM / embedding / paid API / Operation-Visible run in this pass**  
**Market active v1.1:** untouched

## Historical run audit (unchanged facts)

| Artifact | Conclusion |
| --- | --- |
| Development `fin_cf_*` (mcp-free / flat / OV pilots) | **Nonformal Agent pilot** before v1.2 |
| Financial taxonomy **v1.2** | **Never Agent/LLM-run** → superseded as `superseded_before_any_agent_run` |
| Held-out `fin_ht_*` | **`never_agent_run`** (still true for v1.3) |

### Supersession labels

- Prior file (v1.2, both splits): **`superseded_before_any_agent_run`**
- Active files: `*_v1_3.jsonl` with `supersession_status=active`
- Development rows keep `development_exposure_history` documenting pre-v1.2 nonformal pilot
- Held-out rows keep `heldout_agent_run_status=never_agent_run`

## HKEXnews live dataflow (free direct MCP smoke)

Evidence: `docs/financial_hkexnews_dependency_smoke_v1/`

| Step | Operation | Notes |
| --- | --- | --- |
| 1 | `list_hkexnews_announcements` | Emits `local_key`, `news_id`, `release_date`, `title`, `stock_code`, … |
| 2 | `download_hkexnews_announcement` | Consumes list fields incl. `local_key` / `news_id` / `release_date` / `title_keyword` / `stock_code`; returns `local_key` |
| 3 | `parse_hkexnews_announcement` | **Required** before search; consumes `local_key` |
| 4 | `keyword_search` | Requires `local_key`; hits include `page_number` |
| 5 | `get_announcement_pages` | Requires `local_key` + `start_page` (from `page_number`) |

Keyword before download/parse fails (`Parsed file not found`). No warm-cache assumption.

### Field-level edges (producer_output → consumer_input)

- `list.local_key` → `download.local_key`
- `list.news_id` → `download.news_id`
- `list.release_date` → `download.release_date`
- `list.title` → `download.title_keyword`
- `list.stock_code` → `download.stock_code`
- `download.local_key` → `parse.local_key`
- `parse.local_key` → `keyword_search.local_key`
- `parse.local_key` → `get_announcement_pages.local_key` (when pages used)
- `keyword_search.page_number` → `get_announcement_pages.start_page`

## Project difficulty (unchanged from v1.2)

| Split | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- |
| Development | 10 | 6 | 4 | 5 | 5 |
| Held-out | 6 | 6 | 6 | 6 | 6 |

## Strict exact ETOM counts (Dev+Held, v1.3)

| Level | Count |
| --- | --- |
| L1 | 7 |
| L2 | 6 |
| L3 | 12 |
| L4 | **0** |
| L5 | 4 |

Exact L1–L5 balance is **not** required. True compositional L4 deferred to Cross-domain.

## Alignment

| Alignment | Count |
| --- | --- |
| exact | 29 |
| adapted | 12 |
| not_applicable | 19 |

## True exact L3 IDs and paths

| ID | Path |
| --- | --- |
| `fin_cf_003`, `fin_cf_010`, `fin_cf_011`, `fin_cf_013`, `fin_cf_016`, `fin_cf_020`, `fin_ht_013` | list → download → parse → keyword_search |
| `fin_ht_011`, `fin_ht_012`, `fin_ht_014`, `fin_ht_015`, `fin_ht_016` | list → download → parse → keyword_search → get_announcement_pages |

All edges are live-verified field transfers (`live_verified_dependency=true`).

## Exact L4 correction

| ID | v1.2 | v1.3 | Reason |
| --- | --- | --- | --- |
| `fin_ht_017` | exact L4 | **adapted L4** | Conceptual edges (`year_end_constrains_period_interpretation`, `issuer_identity_gates_evidence`) are not parameter field transfers; question already names HSBC + period |
| `fin_ht_018` | exact L4 | **adapted L4** | Same: AKTools profile and HKEXnews evidence answer distinct sub-questions; no AKTools output field required as HKEXnews input |

Strict Financial exact L4 count = **0**.

## Validator upgrades

`scripts/validate_financial_taxonomy_v1_3.py` now checks:

- edge schema: `producer_operation`, `producer_output_field`, `consumer_operation`, `consumer_input_field`
- operations ∈ frozen catalog
- `consumer_input_field` ∈ tool parameter schema (catalog + live smoke schemas)
- rejects conceptual field / via names
- exact L3/L4 require `live_verified_dependency`
- exact L4 requires ≥1 cross-server field edge

## Artifacts

| Path | Role |
| --- | --- |
| `benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl` | active Development |
| `benchmarks/heldout/financial_test_30_v1_3.jsonl` | active Held-out |
| `benchmarks/questions/financial_real_mcp_coverage_v1_2.jsonl` | superseded |
| `benchmarks/heldout/financial_test_30_v1_2.jsonl` | superseded |
| `docs/financial_hkexnews_dependency_smoke_v1/` | free MCP smoke evidence |
| `scripts/build_financial_benchmark_v1_3_dependency.py` | rebuild |
| `scripts/validate_financial_taxonomy_v1_3.py` | schema-aware validator |
| `tests/test_financial_taxonomy_v1_3.py` | offline tests |
