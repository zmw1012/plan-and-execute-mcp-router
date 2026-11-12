# INVALIDATED — PE-Fin RePlan Diagnostic-20 live v1

**Marker:** `invalidated_engineering_diagnostic_due_to_replanner_prompt_schema_contract_mismatch`

## Status

This run is **not usable** for method accuracy and **not usable** for RePlan effectiveness.

Do **not** modify files in this directory. Treat as a frozen, invalidated engineering diagnostic.

## Frozen raw SHA256

| File | SHA256 |
|------|--------|
| `predictions.jsonl` | `bd4424e4b6c4dc6e51f3439833a7c7a3f55a0a69ac6a81414013efd6db1c86e8` |
| `run_metadata.json` | `8ba1803907957c48bd1b021551dc8f2095841f6b716ce13e04afb6c5273e6917` |

## Outcome summary

- **20** prediction rows
- **19** Replanner schema / contract failures (dominant: missing required property `decision_reason`; also related plan/response shape mismatches after repair)
- **1** transient generation failure: `xd_dev_004` — `RuntimeError: LLM request failed ... URLError`

## Identity of invalidated run

- `condition=pe_fin_replan`
- `method_version=pe_fin_replan_v1_1` (pre-contract-fix)
- `protocol_version=planner_executor_replanner_seven_round_v1`
- `subset_kind=diagnostic20_replan`

## Next run

Use a **fresh** directory only:

```text
outputs/pe_fin_replan_diagnostic20_live_v2
```

Do not resume or overwrite this v1 directory.
