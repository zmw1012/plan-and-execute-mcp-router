# Planner protocol v3 hardening (offline; no live Diagnostic-10 v3)

## Protocol identity

`pe_fin_lite_planner_only_v3`

| Field | Value |
|-------|-------|
| prompt hash | `98817b913fcaff9d3fc402d6a1ce912b061a9459ec2d6cac64bf1a9bb68bda85` |
| schema hash | `fdf728361fd74c3d9e6c5807b35a51ac8f87123cc0e3b3e8ea6139627da42d68` |
| structural canonicalizer hash | `1c0cf80e1a3940a14983b93fb8938fab74e13f48917f0141d18e86531ba949cd` |
| scope contract hash | `f282016dc2ffd94ed09c53927dab960a5fd68c2595953ced1aeeca383d5715d8` |
| capability profile hash | `4cefb249bad15010519ff6f9d3b2fdab206781ef9d767aca6f3d902a184ae8e9` |
| planner_max_tokens | `4096` |
| diagnostic manifest/order hash | manifest `4d474fbfd170f2cfc8e53748dd6ee7fea77082d488ccb3ca70a5886721e8305e` / order `764cbdb12cbdae1a8b925379f19959bff23d75939bf428c46374bcb2cb4bb4d3` |
| provider/host/model (formal live) | `openai_compatible` / `api.zhizengzeng.com` / `deepseek-v4.1-flash` |

v1/v2 planner caches cannot resume into v3 (protocol + max_tokens + scope/schema/prompt identity).

## Changes

1. Compact minified JSON planner prompt + generic compact example
2. `scope_constraints.currency` (string|null, maxLength=16)
3. `metric_basis` maxLength 80→160; `adjustment` 40→80 (fail-closed, no silent truncate)
4. Planner/planner_repair `max_tokens=4096` only
5. Response observability on planner calls (lengths/flags only; no hidden CoT text; no secrets)

## Live output dir (future)

`outputs/pe_fin_lite_planner_protocol_diagnostic10_v3`

Do not overwrite v2. Do not run live in this offline hardening turn.
