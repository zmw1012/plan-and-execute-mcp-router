# News Benchmark Blueprint (Proposal — not frozen)

**Status:** proposal after capability discovery. **No final 35 questions** in this phase.  
Await user confirmation of the capability matrix before drafting/freezing items.

## Planned split sizes

| Split | Questions | Role |
|---|---|---|
| Development | **15** | method / error analysis |
| Held-out | **20** | final evaluation (lock later) |

## Dual labels (carry Financial/Market discipline)

- `project_difficulty_level` (D1–D5): news/event reasoning complexity  
- `etom_level` + `etom_alignment` (exact / adapted / not_applicable)  
- Do **not** force L1–L5 count balance  
- Exact L3 requires same-server field edges (list→download→parse→keyword/pages)  
- Exact L4 only if a real cross-server **field** transfer is required (likely rare; may be 0 until Cross-domain)  
- Parallel official + media retrieval without field passing → adapted L4 at most  
- Exact L5 = frozen-catalog capability gap (e.g. historical media before cutoff)

## Suggested Dev 15 distribution

| Bucket | Count | Intent |
|---|---|---|
| Exact L1 | 3 | single list/lookup with explicit code + date window |
| Exact L2 | 2 | choose official vs media / annual vs results title (unequal tools) |
| Exact L3 | 3 | download→parse→keyword (or pages) field chain |
| Adapted L4 | 2 | official list + media snippet without field pass |
| Exact L5 / adapted robustness | 3 | historical media gap; invalid code; post-cutoff refuse |
| Project-hard / multi-event | 2 | chronological / latest-before-cutoff over official list |

## Suggested Held-out 20 distribution

| Bucket | Count |
|---|---|
| Exact L1 | 4 |
| Exact L2 | 3 |
| Exact L3 | 4 |
| Adapted L4 | 3 |
| Exact L5 / robustness | 4 |
| Multi-event / category | 2 |

## Task-type templates (no final wording yet)

| Template | Dev | Held | Supported by |
|---|---|---|---|
| Fixed-window official announcement title/date | 2 | 3 | hkexnews list |
| Latest official disclosure ≤ cutoff | 2 | 2 | list + offline max datetime |
| Event category / title keyword filter | 1 | 2 | list category/title_keyword |
| PDF evidence snippet (keyword/pages) | 2 | 3 | download→parse→keyword[/pages] |
| Official vs media source discrimination | 2 | 2 | list vs `stock_news_em` (L2) |
| Annual report **existence**/date/URL (not numbers) | 1 | 1 | ah `search_annual_report` |
| Chronological order of ≥2 announcements | 1 | 2 | list + offline sort |
| Multi-source adapted (no field edge) | 1 | 1 | list ∥ media |
| L5 refuse: historical media / bad code / future gold | 3 | 4 | gaps above |

Company mix: Tencent / HSBC / Alibaba across both splits.  
**Exclude:** revenue/profit/EPS extraction; non-IFRS numerics; formal report table digging as News gold.

## Required future gold fields (when freezing)

- `domain=news`, `source_class`, `cutoff_date=2026-08-12`, `timezone=Asia/Hong_Kong`
- `gold_operations` / `dependency_graph` with producer/consumer **fields** for exact L3+
- `live_verified_dependency` when edges claimed
- Held-out: `locked=true`, `heldout_agent_run_status=never_agent_run` until first eval

## Stop point

Confirm capability matrix + candidate catalog before any question authoring / freeze.
