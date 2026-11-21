# Results

Authoritative summary of frozen HK-155 and ETOM-100 experiments for **Plan-and-Execute MCP Router**.

HK-155 and ETOM-100 use different task definitions, tool catalogs, and success criteria. **Do not combine or average their scores.**

---

## Metric definitions

### HK-155 (real MCP execution)

| Metric | Meaning |
|--------|---------|
| **Answer Accuracy** | Fraction of tasks with a correct final answer under the sealed deterministic grader (+ human grades where required). |
| **Common Strict** | Strict end-to-end success used for cross-method comparison (answer correct **and** applicable tool/protocol constraints). |
| **Exact Tool Set** | Predicted invoked operation set equals the gold operation set (N/A for MCP-Free). |
| **Total Tokens** | Sum of generation (+ embedding where billed) tokens over **all** tasks in the sealed run—not a per-task average. |

Notes:

- MCP-Free has no tool stage; Exact Tool Set is N/A; Common Strict coincides with Answer Accuracy for this closed-book condition.
- MCP-PlanRoute Lite has one manual-pending bound on the sealed sheet; lower/upper Answer Accuracy bounds are reported.
- MCP-PlanRoute RePlan raw answer-correct count was 40/155; the **conservative** Answer Accuracy is **39/155 (25.16%)** because one otherwise-correct task used post-cutoff evidence.
- RePlan HK-155 is **engineering-valid** (protocol/runtime sealed) but **not formal-accuracy-valid** as a sole headline claim; treat conservative figures as the comparable accuracy.

### ETOM-100 (routing-only)

| Metric | Meaning |
|--------|---------|
| **Route Exact Match (EM)** | Predicted tool route equals the official ETOM gold route. |
| **Precision / Recall / F1** | Set-overlap metrics over predicted vs gold route nodes (when reported). |
| **Failures-as-zero (conservative)** | Incomplete / failed trajectories count as incorrect for the headline EM (not dropped from the denominator). |
| **Per-level L1–L5** | Route EM stratified by official ETOM difficulty levels (20 tasks each). |

ETOM runs **do not execute** business MCP tools; scores measure route agreement only.

---

## HK-155 results

| Method | Answer Accuracy | Common Strict | Exact Tool Set | Total Tokens |
|--------|-----------------|---------------|----------------|--------------|
| MCP-Free | 12.90% | 12.90%* | N/A | 79,133 |
| Flat-All-Tools | 56.77% | 54.84% | 23.23% | 9,464,124 |
| Operation-Visible | **61.29%** | 29.03% | 41.29% | 9,959,757 |
| MCP-Zero | 58.06% | **58.06%** | ~43.9% | 4,050,513 |
| MCP-PlanRoute Lite | 22.58% lower / 23.23% upper | 5.81% | 14.19% | 1,146,716 |
| MCP-PlanRoute RePlan | **25.16% conservative** | 11.61% | 14.19% | 6,164,706 |

\*MCP-Free: no tool stage; Common Strict equals Answer Accuracy for this condition.

**Headline reading (HK-155):** Operation-Visible leads Answer Accuracy (61.29%). MCP-Zero leads Common Strict (58.06%). Both proposed PlanRoute methods underperform ReAct-style / hierarchical baselines on accuracy despite much lower token budgets for Lite.

---

## ETOM-100 results

| Method | Route EM | Route F1 | L1 | L2 | L3 | L4 | L5 | Tokens |
|--------|----------|----------|----|----|----|----|----|--------|
| Operation-Visible | **66%** | 0.734 | 85% | 60% | 65% | 25% | 95% | 2,012,549 |
| MCP-Zero | 49% | *(no frozen unified overall F1 headline)* | 85% | 50% | 45% | 10% | 55% | 1,420,146 |
| MCP-PlanRoute RePlan | **35% conservative** / 45% raw | 0.446 cons. / 0.643 raw | 75% | 50% | 20% | 0% | 30% | 1,669,390 |

Flat-All-Tools and MCP-PlanRoute Lite were **not run** on ETOM-100. MCP-Free is N/A (no routing).

**Headline reading (ETOM-100):** Operation-Visible leads Route EM (66%). MCP-PlanRoute RePlan trails both baselines under conservative failures-as-zero scoring (35%).

---

## Retrieval ablation (HK PlanRoute / R5 selection)

Frozen offline ablation over planner/query retrieval settings (R0–R5):

| ID | Query type | Domain filter | Score formula | Recall@3 | Complete Route@3 | MRR |
|----|------------|---------------|---------------|----------|------------------|-----|
| R0 | question-level | none | cosine | 0.711 | 0.653 | 0.759 |
| R1 | step-level | none | cosine | 0.796 | 0.735 | 0.740 |
| R2 | step-level | hard domain | cosine | 0.585 | 0.517 | 0.619 |
| R3 | step-level | hard domain | α·cosine + (1−α)·coverage | 0.585 | 0.517 | 0.619 |
| R4 | step-level | none | α·cosine + (1−α)·coverage | 0.796 | 0.735 | 0.740 |
| **R5** | step-level | **none** | **cosine + 0.12×lexical** | **0.827** | **0.776** | **0.772** |

Shared R5 settings used by proposed methods: Top-K = 5, Describe-N = 3, lexical weight 0.12, no hard domain filter.

**Why R5:** best Recall@3 / Complete Route@3 / MRR among the frozen candidates while preserving the Operation-Visible published outer lexical weight.

### R5 vs R1

| Delta | Value |
|-------|-------|
| Recall@3 | +3.1 percentage points |
| Complete Route@3 | +4.1 points |
| MRR | +0.032 |

### RePlan vs Lite (HK-155)

| Quantity | Change |
|----------|--------|
| Common Strict | 5.81% → 11.61% (9 → 18 strict tasks) |
| Task-level flips | 17 improved, 12 regressed |
| Net raw-correct tasks | +5 |
| Tokens | ×5.38 (1,146,716 → 6,164,706) |

Replanning helps some failures but does **not** close the gap to Operation-Visible / MCP-Zero, and multiplies control-LLM cost.

---

## Error analysis (sealed audits)

### HK-155 (proposed / PlanRoute lineage)

| Failure theme | Count (sealed taxonomy) |
|---------------|-------------------------|
| Incomplete accepted evidence | 63 |
| Evidence grounding mismatch | 18 |
| Rigid replanning constraints (total) | 22 |
| — step cap | 11 |
| — schema/format | 6 |
| — LLM budget | 5 |

### ETOM-100 (MCP-PlanRoute RePlan)

| Failure theme | Count |
|---------------|-------|
| Over-strict route validation | 24 |
| LLM budget exhaustion | 8 |
| Replanning without progress | 3 |

---

## Scientific conclusion

R5 improves **candidate retrieval**, but the **rigid Planner → Execute-one → Replanner** implementation does **not** outperform ReAct-style Operation-Visible or hierarchical MCP-Zero on either benchmark under conservative scoring.

These experiments **reject the current rigid implementation**, not Plan-and-Execute as a general design principle. Recommended future direction: **Plan-guided ReAct** (soft plan guidance inside a visible Find→Describe→Invoke loop) rather than hard step caps and exclusive replanner turns.

---

## Provenance (run / audit tips)

Tips below are ancestors of `release/code-submission-v1` (verified at Phase 1 / Phase 1.5).

| Condition | Sealed tip (short) | Role | Merged via |
|-----------|--------------------|------|------------|
| MCP-Free HK-155 | `36c27bd` | Final audit + human grade | `7243ca9` |
| Flat-All-Tools HK-155 | `9ca2242` | Final audit + human grade | `b1b9db3` |
| Operation-Visible HK-155 | `b299300` | Sealed OV audit metrics | `c68fd55` |
| MCP-Zero HK-155 | `8c2a25f` | Sealed MCP-Zero audit | `a2f14ea` |
| MCP-Zero ETOM-100 | `1870f53` | Formal-100 final audit | `9d31f07` |
| Operation-Visible ETOM-100 | `d1934e0` | Formal-100 audit lock | `1ee9200` |
| MCP-PlanRoute Lite HK-155 | `f30530f` | R5 HK-155 final audit | via RePlan HK lineage / release merges |
| MCP-PlanRoute RePlan HK-155 | `cdddeaf` | RePlan-7 HK-155 offline audit | `912ecb1` |
| MCP-PlanRoute RePlan ETOM-100 | `e5e48ae` | Formal-100 offline audit | release base + `912ecb1` |

Integration baseline for the release branch: `e5e48ae`. Stabilization: `b3584dd` (tests/grader compatibility only; **no metric change**).

Detailed audit folders live under `docs/` (see [docs/README.md](docs/README.md)). Frozen raw `outputs/` trees are **not** committed; attach via `MCP_ROUTER_*_ARTIFACT_ROOT` (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [DATA.md](DATA.md)).
