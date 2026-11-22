# Final proposed method freeze — PE-Fin Lite v1.2

## Declaration

| Field | Frozen value |
|---|---|
| final proposed method version | **`pe_fin_lite_v1.2`** |
| architecture | Plan-first + step-level retrieval |
| Monitor | **0** |
| global replan | **0** |
| retrieval TopK | **5** |
| Describe | **2–3** |
| local fallback per retrieve | **≤1** |
| LLM calls / task | **≤8** |
| MCP invocations / task | **≤8** |
| binder design | primary + fallback in one response; ≤1 format repair |
| reason design | ≤1 JSON repair; sticky-fail cleared only on valid repair |
| disclosure | shortest-required-prefix; cutoff-enforced; ≤1 title broaden |

## Explicit non-goals after this freeze

- Do **not** create `pe_fin_lite_v1.3`, `v2`, or any new controller from Diagnostic-12 residuals.
- Do **not** add Monitor / global replan / task-id / company whitelist / gold announcement patches.
- Do **not** treat Diagnostic-12 as formal accuracy or significance evidence.

## Allowed subsequent work only

1. HK-155 **retrieval audit** (planner-only + ranking / formula selection; evaluation-only)
2. Formal **HK-155** agent evaluation (separate run directory; not D12)
3. Offline final audit / report writing

Commit that freezes the method runtime used by D12 v3:  
`baf572461dce670fddb71291128e82360dbef2f0`
