# PE-Fin Lite Planner Diagnostic-10 v2 Offline Audit

## Status

`invalid_for_engineering_pass`

- Role: engineering_protocol_diagnostic_only
- explicitly_not_formal_accuracy: true
- Not an accuracy / formula / reserved85 result

## Freeze

- planner_outputs.jsonl SHA256: `34115f2aba3aec91e639e35dc6b5be9c80c593bc35c97238b69d9e227b2795e0`
- protocol: `pe_fin_lite_planner_only_v2`
- provider/host/model: `openai_compatible` / `api.zhizengzeng.com` / `deepseek-v4.1-flash`

## Funnel

| Metric | Value |
|--------|------:|
| rows / unique | 10 / 10 |
| ok / failed | 4 / 6 |
| repair tasks | 8 |
| LLM calls | 18 |
| total tokens | 69670 |
| engineering_pass | False |

## Failures

| task_id | error |
|---------|-------|
| xd_ht_003 | scope_constraints unexpected currency |
| mkt_dev_007 | initial + repair empty |
| news_ht_011 | repair JSON truncated/invalid |
| xd_ht_006 | initial + repair empty |
| fin_ht_024 | metric_basis maxLength (actual 125 > 80) |
| fin_cf_029 | truncated/non-JSON |

## Root causes (engineering)

1. Missing `currency` in scope_constraints schema
2. Output budget / empty provider content under planner max_tokens=2048
3. Truncated repair JSON
4. metric_basis maxLength too tight
5. Observability gap (no finish_reason / reasoning_content lengths in v2 traces)

## Next

Planner protocol v3 implemented offline (see `protocol_v3_hardening.md`): compact JSON prompt, scope currency, longer metric_basis/adjustment limits, planner_max_tokens=4096, response observability. New live output dir `outputs/pe_fin_lite_planner_protocol_diagnostic10_v3`. Do not overwrite v2. Do not treat this v2 run as accuracy.

### v2 raw freeze (re-verified after hardening)

`planner_outputs.jsonl` SHA256 unchanged: `34115f2aba3aec91e639e35dc6b5be9c80c593bc35c97238b69d9e227b2795e0`

### Cost trade-off (planner stage only)

Raising planner/planner_repair max_tokens 2048→4096 may reduce empty/truncated responses and avoid some repair calls; total calls/task still ≤8 with ≤1 planner repair. Binder/Reason/Final budgets unchanged.
