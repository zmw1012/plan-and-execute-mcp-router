# PE-Fin RePlan ETOM Architecture (v1)

Cross-benchmark transfer of frozen PE-Fin RePlan-7 onto ETOM-100.
Condition: `pe_fin_replan_etom`. Independent of HK runners.

## Loop

```
Question
  → Static ETOM Capability Profile
  → Planner (exactly once)
  → Executor executes exactly one ready step
  → Replanner (action = plan | response)
  → execute next step or END
  → maximum 7 rounds
  → final response from Replanner only
```

Hard edges:

- No Executor → Executor
- No Monitor / no global replan
- Round 7 is response-only
- No eighth round
- Planner is not an execution round

## Components

| Module | Role |
| --- | --- |
| ETOM capability profile | Static, hash-versioned server/capability summary (no gold/task leakage) |
| Planner | Question + profile + generic step schema → structured plan |
| Step retrieval (R5) | `cosine + 0.12 * lexical`; Top-K=5; Describe-N=3; safety-only filter |
| Tool binder | Described candidates only; primary + ≤1 fallback; required-args grounded |
| Gateway | Prefer ETOM routing resolver / future MCP transport; budgeted invokes |
| Envelope / Evidence | Generic projection; provenance `observed \| grounded_input \| derived` |
| Compute | Deterministic safe calculator; operands from Evidence Store |
| Reason | Strict structured output; ≤1 repair; fail closed |
| Replanner | Sees compressed evidence + budgets; ≤2 new steps; grounding |
| Evaluator | Reuse `hk_equity_mcp.etom.evaluator` (routing graph EM/P/R/F1) |

## Step types

`retrieve` | `compute` | `reason`

Tool-related retrieve steps carry capability area, objective, required_outputs,
depends_on, retrieval_query_hint, and scope hints. Planner plans must not embed
`operation_id` or gold routes.

## Budgets

- LLM calls / task ≤ 16
- MCP invocations / task ≤ 8
- Execution / replan / executed-step rounds ≤ 7
- Binder / schema repair ≤ 1 full regeneration
- Max new steps per replan = 2

## Isolation from HK

Do not import HK Financial/Market/News/Disclosure capability profile,
Sina/Eastmoney providers, HKEX disclosure workflow, HK cutoff, or HK
stock-code normalization into the ETOM condition path.
Embedding index namespace is dedicated:
`.cache/pe_fin_replan_etom_operation_embeddings_v1`.
