# D12 v2 Offline Hardening + Root-Cause Audit

## engineering_pass=false direct cause

Raw predictions for `mkt_dev_014` and `fin_cf_024` have:

- `real_mcp_execution_count=0`
- `tool_execution_success=true` (vacuous)

Invariant `zero_invocation_not_tool_success` fails. Not hard-coded.

## Zero-invocation metrics fix (future runtime)

`agent.py`: when `mcp_count==0` and no invoke records →  
`tool_execution_success=null`, `all_tool_invocations_successful=null`.

Offline audit:

- temporal no-tool → C07 `N/A_not_required`
- binder failed before invoke → C07 `N/A_not_reached`
- does not rewrite v2 raw

## Checkpoint headline semantics

Headline/funnel `status` = `stage_outcome_status` (never contract pass as stage pass).

| CP | Rule |
|---|---|
| C07 | N/A_not_required / N/A_not_reached / pass / fail |
| C11 | N/A_not_planned / N/A_upstream_blocked / pass / fail |
| C12 | N/A_not_planned / N/A_upstream_blocked / pass / fail; recovered ≠ pass |
| C13 | schema pass; semantic correctness separate |
| C14 | evaluator executed; answer_correct separate |

## A. Binder — fin_cf_024

- failure_reason: `binder_failed:empty content`
- binding.ok=false; repair_count=0 (binder has **no** repair loop by design)
- primary/fallback null; described candidates present (3 ops)
- No binder `call_observability` / finish_reason on funnel → **insufficient observability** for length vs null vs provider
- Planner call obs present (`finish_reason=stop`) — not binder
- Classification: **provider/protocol empty binder content** (schema path raises `empty content`); not selection-outside-described

## B. Disclosure taxonomy (offline)

| Task | Stage | Class | Evidence |
|---|---|---|---|
| fin_cf_003 | list | no_matching_result | No announcements found for 00700; n_list_rows=0 |
| fin_cf_020 | list | no_matching_result | No announcements found for 09988; n_list_rows=0 |
| fin_ht_016 | download | envelope_adapter_failure / download_budget_exhausted | Download budget exhausted (max 1…) after list success |
| xd_ht_014 | list | no_matching_result | No announcements found for 09988 (with date range) |

No network retry. No invalid_arguments / network_service_failure evidenced.

## C. Reason — fin_cf_023 / news_ht_004

Artifacts contain valid summary+claims and revalidate against reason schema **today**.

Root cause (code path, not re-run): `execute_reason_step` sets `reason_parse_failed=True` on **first** parse failure and **never clears** it after successful repair (`repair_count=1`, `runtime_recovered=false`).

So stage is marked fail even though repair payload is schema-valid; Final still synthesizes from verified evidence/compute.

**Next-round minimal fix (do not implement this round):** clear `reason_parse_failed` when repair parse succeeds; optionally record `initial_parse_failed` separately. No Monitor/Replan.

## D. xd_dev_002 evidence semantic

- required outputs satisfied; volume OK
- EPS observed `8.504848` USD vs benchmark `1.21` → evidence semantic (metric basis/scale/row), not evaluator bug
- No gold-number adapter patch

## v1 → v2 comparison

| Signal | v1 | v2 |
|---|---|---|
| partial_merge_value_conflict | 3 | **0** |
| answer_correct | 2/12 | **3/12** |
| failed | 4 | **3** |
| compute direct Call failure | 1 | **0** |
| temporal latest/current | wrong historical answer | **unavailable (correct)** |
| remaining | — | binder empty; disclosure list/download; reason sticky parse flag; EPS semantic |
| LLM / tokens / MCP / emb | 38 / 88726 / 28 / 12 | 39 / 86418 / 26 / 10 |

## Future D12 v3 (suggestion only — do not run)

After confirming runtime metrics fix + optional reason sticky-flag fix + rebuild embedding index if needed:

```bash
PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --diagnostic12 \
  --split exploratory_all \
  --require-complete-index \
  --allow-dirty \
  --env-file ../hk-equity-mcp-agent/.env.relay-v41 \
  --embedding-env-file ../hk-equity-mcp-agent/.env.embedding-relay \
  --output-dir outputs/pe_fin_lite_diagnostic12_checkpoint_live_v3
```
