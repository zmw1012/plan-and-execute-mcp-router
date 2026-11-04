# Financial Held-out Test 30 v1.2 Design

- **Active file:** `benchmarks/heldout/financial_test_30_v1_2.jsonl`
- **Prior file:** `financial_test_30_v1_1.jsonl` → **`superseded_before_any_agent_run`**
- **Flags:** `locked=true`, `dataset_split=heldout_test`, `benchmark_role=final_evaluation_only`
- **Numerical gold / questions:** unchanged vs v1.1
- **Catalog:** frozen Financial Real-MCP unchanged

## Run-history basis for supersession label

No `predictions.jsonl` contains `fin_ht_*`. Held-out v1.1 only received direct Real-MCP provenance audit, never an Agent evaluation. Therefore prior file may be labeled `superseded_before_any_agent_run`.

## Project difficulty distribution

D1–D5: 6 each

Project difficulty captures financial reasoning complexity. Strict ETOM level captures tool-orchestration topology. They are related but not interchangeable.

## Strict exact L4 (only)

- `fin_ht_017`, `fin_ht_018` — AKTools profile → HKEX evidence compositional DAG

Parallel corroboration / basis-pair items remain adapted or not_applicable.
