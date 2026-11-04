# Financial Held-out Test 30 v1 — MCP Gap Proposals

**Status:** proposals only. **Do not wire** any new server/tool or mutate the frozen Financial v1 catalog until the user explicitly confirms.

Audit basis: every held-out L1–L4 item was checked against `configs/operations/financial_real_mcp.yaml`. Unsupported L1–L4 candidates were **replaced** with catalog-feasible compositions; residual gaps that still matter for general Financial coverage are recorded below.

## Proposal gate

A new MCP/tool is proposed only when the gap affects **≥ 3 meaningful candidate questions** and improves general Financial coverage (not a one-off trick for a single ID).

---

## Proposal 1 — Segment / business-line revenue evidence

| Field | Content |
|---|---|
| **Capability gap** | No frozen v1 operation returns verified **segment or business-line revenue** (Tencent VAS, Alibaba Cloud, HSBC CMB/WPB, etc.) with official page evidence. |
| **Affected held-out items** | `fin_ht_009` (Tencent VAS — L5 refuse), `fin_ht_029` (Alibaba Cloud — L5 refuse). |
| **Additional meaningful candidates** (≥3 total) | HSBC Wealth & Personal Banking / Commercial Banking revenue; Tencent FinTech & Business Services; Alibaba China commerce vs Cloud — all repeatedly useful for L2–L4 Financial QA. |
| **Candidate server/tool** | Prefer extending **`mcp_hkexnews`** with a guided “results PDF table/segment extract” helper (e.g. `extract_segment_table` / structured keyword+page pack), **or** a dedicated read-only IR segment parser bound to official annual/interim PDFs. Do **not** invent segments from East Money consolidated rows. |
| **Live evidence** | Segment notes already exist in official annual/interim PDFs referenced by the project mock (Tencent/Alibaba/HSBC IR URLs). |
| **Reproducibility** | Medium–high if scoped to named report + page anchors; brittle if free-form PDF table detection drifts. |
| **Data-quality risk** | High if OCR/table parse silently mis-aligns columns; medium if human-verified page packs are frozen per report. |
| **Recommendation** | **ACCEPT (design only)** — high leverage for ≥3 segment questions and general coverage. **STOP** before implementation pending user confirmation. |

---

## Proposal 2 — Banking regulatory capital / ratio pack (CET1, NIM, etc.)

| Field | Content |
|---|---|
| **Capability gap** | No frozen v1 operation returns verified **CET1**, **NIM**, **RWA**, or similar banking regulatory/APM ratios as structured gold. |
| **Affected held-out items** | `fin_ht_019` (CET1 — L5 refuse). |
| **Additional meaningful candidates** (≥3 total) | CET1 ratio; net interest margin; risk-weighted assets / leverage ratio — standard HSBC results KPIs beyond P&L totals. |
| **Candidate server/tool** | HKEXnews keyword/page pack specialised for HSBC “key metrics” / capital tables, or a narrow `banking_key_metrics_extract` over official results PDFs. |
| **Live evidence** | HSBC Annual Report / Interim Report key metrics pages (same IR PDF family already used for revenue/PAT gold). |
| **Reproducibility** | Medium — metric synonyms and basis footnotes (transitional vs reported CET1) create ambiguity. |
| **Data-quality risk** | High without explicit basis tags (phased-in vs reported, including/excluding notables). |
| **Recommendation** | **CONDITIONAL ACCEPT** — only if the project wants banking-ratio coverage in a future catalog v2. For held-out v1, L5 refusal is the correct behaviour. **STOP** before wiring. |

---

## Gaps explicitly **not** proposed as new tools

| Gap | Why not proposed now |
|---|---|
| East Money absolute USD amounts for HSBC as structured gold | Known data-quality risk; held-out correctly forces HKEXnews evidence. Fixing via “trust East Money” would **lower** quality. |
| Full-year Tencent FY2026 / Alibaba calendar-2026 / 9M2026 inventing | Period availability is a **snapshot policy** issue, not an MCP gap. |
| Non-IFRS / non-GAAP via AKTools | Already answerable via HKEXnews evidence under v1; no new server required for held-out L1–L4. |

## Action required from user

1. Review Proposal 1 (segment evidence) and Proposal 2 (banking ratios).  
2. Explicitly confirm which (if any) should be designed/wired.  
3. Until confirmation: **frozen Financial v1 catalog remains unchanged**; held-out L5 items keep refusal gold.
