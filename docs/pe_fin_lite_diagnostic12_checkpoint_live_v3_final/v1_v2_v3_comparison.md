# Diagnostic-12 v1 / v2 / v3 comparison (report-ready)

| Metric | v1 | v2 | v3 |
|---|---:|---:|---:|
| method_version | pe_fin_lite_v1.1* | pe_fin_lite_v1.1 | **pe_fin_lite_v1.2** |
| full / partial / failed | 4 / 4 / 4 | 4 / 5 / 3 | **5 / 5 / 2** |
| answer_correct | 2/12 | 3/12 | 3/12 |
| LLM calls | 38 | 39 | 42 |
| tokens | 88726 | 86418 | 95419 |
| MCP calls | 28 | 26 | 33 |
| engineering_pass | true (pre zero-inv fix nuance) | false (vacuous tool success) | **true** |
| explicitly_not_formal_accuracy | true | true | true |

\*v1 freeze docs record post-hardening method bump to v1.1; v1 live raw was the first checkpoint live run.

## Conclusions (required)

1. **Engineering stability improved.**  
   Completeness moved 4/4/4 → 4/5/3 → 5/5/2; v3 achieves `engineering_pass=true` with zero-invocation metrics and dual-layer N/A semantics intact.

2. **Method accuracy plateaued.**  
   answer_correct remains **3/12** from v2 to v3 despite disclosure/binder/reason hardening.

3. **No further Diagnostic-12 structural changes.**  
   Remaining failures are reportable method/data limitations (evidence acceptance, disclosure recall/selection, Reason schema extras, EPS basis/unit semantics)—not justification for Monitor, global replan, task-id patches, or a new controller.

4. **D12 is not formal accuracy** and must not be used for significance claims or unseen held-out claims.

5. **Next work is outside D12 structure:** retrieval audit (all-155), formal HK-155 agent run (separate), offline final audit.
