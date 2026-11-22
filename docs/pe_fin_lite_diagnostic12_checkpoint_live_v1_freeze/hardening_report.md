# Offline Hardening Report — D12 Checkpoint Live v1

## 1. Corrected checkpoint semantics (dual-layer)

Each checkpoint now records:

- `contract_status` / `observability_status` — trace/contract presence
- `stage_outcome_status` — semantic stage success
- `downstream_task_effect` — effect on later stages / answer

Trace existence alone is no longer treated as stage success.

| CP | Key correction |
|---|---|
| C02 | structural pass = no blocked leakage; quality = `not_evaluated` (no accepted-op set in D12) |
| C03 | structural pass = ranked candidates; retrieval quality separated from Recall@K / all155 |
| C08 | `normalization_success=null` → unknown; gateway success ≠ normalization success |
| C09 | acceptance_logic_pass vs retrieve incomplete; false_accept / false_reject / correctly_rejected_incomplete |
| C11 | upstream retrieve fail → `N/A_upstream_blocked`; only executed calc fail → `direct_compute_failure` |
| C12 | parse fail = stage fail; `runtime_recovered=true` does not equal stage pass |
| C13 | schema-valid ≠ semantically correct |
| C14 | evaluator executed ≠ answer_correct |

## 2. issuer_code pollution — root cause & fix

**Root cause:** `aktools.stock_financial_hk_report_em` legitimately uses `symbol="利润表"` (statement type) with `stock="00005"|"09988"`. `build_grounded_input_field_map` preferred `args["symbol"]` via `or`-chain and wrote `issuer_code="利润表"`. Later merge with real code `00005`/`09988` raised `partial_merge_value_conflict` (fin_cf_011, fin_cf_023, xd_ht_014).

**Fix (generic):**

- `canonicalize_hk_issuer_code` rejects CJK/statement labels; accepts 1–5 digit / `.HK` forms
- grounded_input tries `stock` / `stock_code` / `code` / `symbol` / question code independently
- merge drops non-canonical pollution; equivalent codes canonicalize before compare
- envelope projection never promotes statement labels through `symbol→issuer_code`
- real entity conflicts still fail closed
- no task_id branches

## 3. Compute contract — root cause & fix

**Root cause:** planner emitted `round_half_up(close,2)` / `abs(...)` but calculator rejected all `ast.Call`.

**Frozen strategy:** whitelist `abs|min|max|round_half_up`; forbid attribute access / imports / arbitrary calls; unknown functions fail closed; planner prompt lists only these functions.

## 4. Reason parse — root cause & fix

**v1 evidence (xd_dev_002, mkt_dev_014):** reason artifacts contain `gaps=["reason_parse_failed"]` with soft completed status and verified-evidence fallback text.

**Fix:** ≤1 repair still under LLM budget; second fail → verified-evidence fallback with `reason_parse_failed=true`, `runtime_recovered=true`, stage status `failed`; structured JSON contract in reason prompt; no Monitor/Replan.

## 5. Temporal latest/current

**v1:** mkt_dev_014 asked latest/current/today but answered with 2026-07-02 historical close.

**Fix:** generic cutoff policy refuses silent historical substitution (`temporal_latest_current_unsatisfied_under_cutoff`).

## 6. Disclosure offline taxonomy

| Task | Stage | Class | Evidence |
|---|---|---|---|
| fin_cf_020 | download | `envelope_adapter_failure` (subtype download_budget_exhausted) | "Download budget exhausted (max 1 announcement per adapter session)." |
| fin_ht_016 | list | `no_matching_result` | "No announcements found for 00005"; n_list_rows=0 |
| fin_cf_024 | list | `no_matching_result` | "No announcements found for 09988"; n_list_rows=0 |

No network retry. Invalid_arguments / network_service_failure / scope mismatch not evidenced for these three.

## 7. xd_dev_002 evidence semantic audit

- `required_outputs_satisfied=true` for both retrieves: analysis_indicator returned `basic_eps=8.504848`, currency 美元, FY2025; market hist returned volume `13123569` on 2026-06-30.
- Final numeric mismatch vs gold basic EPS `1.21` USD; volume matches.
- Classification: **evidence semantic issue** (EPS metric basis / scale / source row from analysis_indicator), **not** evaluator bug (`auto_grade_reason=numeric_mismatch;numeric_match_labeled` correctly fails).
- Reason stage parse-failed with verified-evidence fallback (did not invent numbers).
- No gold-number adapter patch in this round; only generic issuer/period/unit/canonicalization rules.

## 8. Constraints honored

- No real LLM / Embedding / MCP
- No Diagnostic-12 / HK-155 re-run
- No raw predictions/run_metadata mutation
- No Monitor / global replan / new multi-round controller
- No task_id or gold patches
- No push / PR / merge
