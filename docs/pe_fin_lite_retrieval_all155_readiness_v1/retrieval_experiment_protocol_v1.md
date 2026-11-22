# PE-Fin Lite Retrieval / Embedding Formula Experiment Protocol (v1)

**Role:** exploratory / descriptive · `no_unseen_heldout_claim=true`  
**Method (frozen):** `pe_fin_lite_v1.2` · architecture `plan_first_step_level_retrieval_v1`  
**Embedding model (only):** `text-embedding-3-small` · dim **1536** · host `api.zhizengzeng.com`  
**This document does not authorize real API runs**; see `future_commands.md`.

---

## 1. Why only one Embedding model

The experiment asks whether **Plan-first step-level retrieval** beats **full-question global retrieval** on HK-155 (especially Financial / News / Cross-domain multi-tool tasks).

It does **not** compare embedding backbones. A single frozen model keeps cost bounded and attributes gains to:

| Signal | Origin |
|--------|--------|
| Cosine on catalog docs | Standard dense retrieval |
| Step vs question query construction | PE-Fin Lite design |
| Deterministic domain hard filter | PE-Fin Lite design |
| Required-output coverage hybrid | PE-Fin Lite design |
| Lexical term (weight 0.12) | Paper / current runtime control |

---

## 2. Formulas

### R0 — Question-level dense baseline

`score = cosine(embed(full_question), embed(operation_document))`

- Global; no domain filter; no coverage; no lexical  
- Top-K=5 · **baseline only** · **excluded from Plan-first formal selection**

### R1 — Step-level global dense

`score = cosine(embed(step_query), embed(operation_document))`

- Step query from Planner retrieve step (`pe_fin_lite_step_query_v1`)  
- No domain filter; no coverage

### R2 — Step-level domain-filtered dense

1. `candidate_pool = hard_filter(runtime_ops, step.domain)`  
2. `score = cosine(...)` on pool  

Diagnostics: before/after counts, rejected reasons, gold before/after, `filter_miss`, `embedding_miss`.

### R3 — Step-level domain-filtered hybrid

Same pool as R2.

```
coverage = matched_required_outputs / total_required_outputs
# empty required_outputs → coverage = 1.0 (unit-neutral; preserves cosine order)
score = alpha * cosine + (1 - alpha) * coverage
alpha ∈ {1.0, 0.9, 0.8, 0.7}   # discrete only; no grid search
```

### R_runtime — Current Agent runtime parity control

Exact mirror of PE-Fin Lite retrieve path:

1. Build step search query (`build_step_search_query`)  
2. Score **full registry**: `cosine + 0.12 * lexical_score` (`operation_resolver.retrieval`)  
3. Stable sort: score desc, `operation_id` asc  
4. Cut global Top-K  
5. Apply domain/runtime hard filter **after** Top-K (`select_embedding_topk`)  

Lexical weight is **fixed at 0.12** (not tuned).  
Blocked / non-runtime ops must not survive the hard-filter stage.

---

## 3. Formula selection (development-70 only)

**Selectable:** R1, R2, R3@α, R_runtime  
**Not selectable:** R0  

Priority (deterministic tie-break):

1. `complete_route@3`  
2. macro `Recall@3`  
3. minimum per-domain `Recall@3`  
4. MRR  
5. simpler formula  
6. if R3 tied, prefer **larger** alpha  

| Selected | `runtime_sync_required` |
|----------|-------------------------|
| R_runtime | `false` |
| R1 / R2 / R3 | `true` + minimal sync spec (do **not** patch runtime in this round) |

Reserved-85: evaluate locked formula only; **no retune**.  
All-155: descriptive aggregate only.

---

## 4. Identity / invalidation / caches

| Artifact | Rule |
|----------|------|
| Real catalog | identity `ac4c109b…`, host+model+dim=1536, complete |
| Question-query cache | 155×1536; **reuse**; no re-embed if identity matches |
| Mixed-dim invalid R0 | archived under `question_level_invalid_mixed_dim64_v1/` + INVALIDATED marker; **not** for report/selection |
| Offline ranking without emb identity | fail-closed / inventory-only; never silent host=null mock catalog |
| Planner / step-query caches | bind prompt/schema/canonicalizer/docs/index hashes; mismatch fail-closed |

Gold / accepted ops: **evaluation only** — never enter Planner prompt, step query, docs, filter, ranking, or Binder.

---

## 5. API cost boundary

| Stage | Network |
|-------|---------|
| index-status (with emb env identity), R0–R3/R_runtime rank, alpha compare, select, lock, reserved85, report, catalog coverage, planner-cache-audit | **zero** |
| Planner-only 155 | generation LLM (future) |
| Step-query embed | Embedding API (future) |
| Full PE-Fin Lite HK-155 | after lock + optional runtime sync (separate) |

---

## 6. Result table template

| Split | Formula | Hit@1/3/5 | Recall@1/3/5 | MRR | complete-route@3/5 | min-domain R@3 | filter_miss | embedding_miss | distractor@5 | blocked_leak |
|-------|---------|-----------|--------------|-----|--------------------|----------------|-------------|----------------|--------------|--------------|
| development70 | R0 | | | | | | | | | |
| development70 | R1…R_runtime | | | | | | | | | |
| reserved85 | locked | | | | | | | | | |
| all155 | descriptive | | | | | | | | | |

Separate rows: disclosure composite; no-tool unnecessary retrieval (excluded from tool-recall denominator).

---

## 7. Limitations

- Single embedding model; no cross-encoder / LLM reranker / Monitor / Replan.  
- R_runtime post-Top-K filter can yield &lt;K candidates (runtime quirk; mirrored on purpose).  
- Planner quality bounds step-query usefulness.  
- Development/reserved split is origin-based; not an unseen held-out claim.  
- Formal lock is exploratory/descriptive engineering evidence, not production auto-promote.
