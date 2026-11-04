# News / Corporate Disclosure Benchmark Blueprint (Proposal — not frozen)

**Domain label:** **News / Corporate Disclosure**  
**Status:** proposal after capability discovery + dependency-smoke correction. **No final 35 questions** yet.  
**Catalog note:** `equal_function_sets: []` ⇒ **Exact ETOM L2 = 0**.

## Planned split sizes

| Split | Questions | Role |
|---|---|---|
| Development | **15** | method / error analysis |
| Held-out | **20** | final evaluation (lock later) |

## Dual labels

- `project_difficulty_level` (D1–D5): disclosure/event reasoning complexity (may include chronological / latest-before-cutoff **reasoning**)
- `etom_level` + `etom_alignment` (exact / adapted / not_applicable)
- Do **not** force L1–L5 balance
- **Exact L2 = 0** while `equal_function_sets` is empty (official ≠ media ≠ annual report ≠ results announcement as equal tools)
- **Exact L3** only via live HKEXnews field edges: list→download→parse→keyword_search(`keywords:list`)→pages(`start_page`)
- **Exact L4 = 0** for now (no required cross-server field transfer; do not invent)
- Chronological / latest-before-cutoff over an official list raises **project difficulty**, not ETOM level, unless a real multi-op field DAG is required
- Source discrimination (official vs media) is **project difficulty / adapted**, **not** Exact L2

## Strict Exact ETOM planning counts

### Development (15)

| Bucket | Count | Intent |
|---|---|---|
| Exact L1 | 5 | single official list/lookup; date window; title/category |
| Exact L2 | **0** | no equal-function set |
| Exact L3 | 4 | HKEXnews field-dependency evidence chain |
| Exact L4 | **0** | no cross-server field edge required |
| Exact L5 (`capability_gap`) | 2 | historical media unavailable under frozen catalog |
| Adapted robustness | 2 | invalid code / post-cutoff temporal / name ambiguity (see subtypes) |
| Project-reasoning (D↑, ETOM not upgraded) | 2 | chronological / latest-before-cutoff on official list |

### Held-out (20)

| Bucket | Count |
|---|---|
| Exact L1 | 6 |
| Exact L2 | **0** |
| Exact L3 | 5 |
| Exact L4 | **0** |
| Exact L5 (`capability_gap`) | 3 |
| Adapted robustness | 3 |
| Project-reasoning (D↑, ETOM not upgraded) | 3 |

## Robustness / L5 subtype split (do not collapse)

| Situation | Label | Exact L5? |
|---|---|---|
| Historical media unavailable (`stock_news_em` no date filter; n_before_cutoff=0) | `robustness_subtype=capability_gap` | **yes (candidate)** |
| Invalid stock code | `adapted` + `invalid_symbol` / fail-closed | **no** |
| Post-cutoff request | `adapted` + `temporal_policy` | **no** |
| Ambiguous company name | `adapted` + `ambiguous_request` | **no** |

## Task-type templates (formal gold — no floating media)

| Template | Dev | Held | Supported by | Notes |
|---|---|---|---|---|
| Fixed-window official announcement title/date | 3 | 4 | hkexnews list | Exact L1 |
| Latest official disclosure ≤ cutoff | 2 | 2 | list + offline max datetime | project reasoning; ETOM L1 unless multi-op DAG |
| Event category / title keyword filter | 1 | 2 | list category/title_keyword | Exact L1 |
| PDF evidence snippet (keyword/pages) | 3 | 4 | download→parse→keyword→pages | Exact L3 |
| Annual report **existence**/date/URL (not numbers) | 1 | 1 | ah `search_annual_report` | supporting; not Financial numerics |
| Chronological order of ≥2 official announcements | 1 | 2 | list + offline sort | project difficulty |
| Exact L5: historical media gap refuse | 2 | 3 | frozen catalog gap | capability_gap |
| Adapted: invalid code / post-cutoff / ambiguity | 2 | 2 | hkexnews errors + policy | not Exact L5 |

### Explicitly excluded from formal gold

- `aktools.stock_news_em` content as answer evidence
- Official-vs-media factual comparison gold
- Cross-source adapted L4 that depends on media text
- Revenue / profit / EPS / non-IFRS extraction (Financial domain)
- Floating “latest media” as cutoff gold

`stock_news_em` may appear only as **tool-selection distractor**, **capability-gap**, **fail-closed**, or **informal demo**.

## Required future gold fields (when freezing)

- `domain=news_corporate_disclosure` (or `news` with `domain_scope=corporate_disclosure`)
- `source_class=official_disclosure` for formal gold rows
- `cutoff_date=2026-08-12`, `timezone=Asia/Hong_Kong`
- field-level `dependency_graph` + `live_verified_dependency` for Exact L3
- Held-out: `locked=true`, `heldout_agent_run_status=never_agent_run`

## Stop point

Capability matrix correction complete after dependency smoke. Still **do not** author/freeze the final 35 until explicit go-ahead.
