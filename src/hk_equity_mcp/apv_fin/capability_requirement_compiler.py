"""Capability Requirement Compiler (APV-Fin v1.3).

Planner semantic requirement → canonical capability → canonical contract
outputs → capability filter → operation binding.

Company-generic and deterministic. Never branches on task identifiers, gold,
or company allow-lists.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hk_equity_mcp.apv_fin.capability_ontology import (
    CAPABILITY_COMPATIBILITY,
    CAPABILITY_IDS,
    DEFAULT_REQUIRED_OUTPUTS,
)

COMPILER_VERSION = "apv_fin_capability_requirement_compiler_v1"

# Informal / legacy planner vocabulary → formal CAPABILITY_IDS.
CAPABILITY_ALIAS_MAP: Dict[str, str] = {
    "market.equity_eod_price": "market.latest_close",
    "market.equity_quote_snapshot": "market.latest_close",
    "market.daily_trading_volume": "market.daily_ohlcv",
    "market.ohlcv": "market.daily_ohlcv",
    "market.price_history": "market.daily_ohlcv",
    "financial.standard_metric_bundle": "financial.reported_metric_bundle",
    "financial.official_filing_narrative_metric": "financial.official_narrative_evidence",
    "financial.narrative_metric": "financial.official_narrative_evidence",
    "official.filing_narrative_metric": "financial.official_narrative_evidence",
}

# Semantic planner fields → contract canonical provides_fields.
SEMANTIC_OUTPUT_ALIAS_MAP: Dict[str, str] = {
    "closing_price": "close",
    "latest_close": "close",
    "close_price": "close",
    "trading_volume": "volume",
    "daily_volume": "volume",
    "volume_unit": "volume",
    "net_income_attributable_to_ordinary_shareholders": "attributable_profit",
    "profit_attributable_to_equity_holders": "attributable_profit",
    "non_ifrs_profit_attributable_to_equity_holders": "snippet",
    "adjusted_profit_attributable_to_equity_holders": "snippet",
    "reporting_period": "report_date",
    "period": "report_date",
    "eps_currency": "currency",
    "reporting_currency": "currency",
    "basic_earnings_per_share": "basic_eps",
    "accounting_basis": "snippet",
    "adjustment": "adjust",
}

# Fields that are already canonical contract outputs (pass-through).
CANONICAL_OUTPUT_FIELDS = frozenset(
    {
        "attributable_profit",
        "revenue",
        "basic_eps",
        "currency",
        "unit",
        "report_date",
        "trade_date",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "adjust",
        "local_key",
        "snippet",
        "page_number",
        "title",
        "value",
        "entity",
        "issuer_code",
        "news_id",
        "release_date",
    }
)

_NARRATIVE_METRIC_RE = re.compile(
    r"(non[-_ ]?ifrs|\badjusted\b|\bnon\s+gaap\b|"
    r"alternative[\s_-]*performance|"
    r"\bnarrative\b|official[\s_-]*filing|announcement[\s_-]*body)",
    re.IGNORECASE,
)


class CapabilityCompilerError(ValueError):
    """Fail-closed compiler error with a stable error_type."""

    def __init__(self, message: str, *, error_type: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def capability_compiler_hash() -> str:
    blob = json.dumps(
        {
            "compiler_version": COMPILER_VERSION,
            "capability_aliases": CAPABILITY_ALIAS_MAP,
            "semantic_output_aliases": SEMANTIC_OUTPUT_ALIAS_MAP,
            "capability_ids": list(CAPABILITY_IDS),
            "compatibility": {k: list(v) for k, v in CAPABILITY_COMPATIBILITY.items()},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def output_alias_map_hash() -> str:
    blob = json.dumps(SEMANTIC_OUTPUT_ALIAS_MAP, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def formal_capability_ids() -> Tuple[str, ...]:
    return CAPABILITY_IDS


def normalize_capability_id(
    raw: Optional[str],
    *,
    allow_null: bool = False,
) -> Optional[str]:
    """Map informal capability vocabulary onto CAPABILITY_IDS. Fail closed."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if allow_null:
            return None
        raise CapabilityCompilerError(
            "required_capability missing",
            error_type="missing_required_capability",
        )
    text = str(raw).strip()
    if text in CAPABILITY_IDS:
        return text
    if text in CAPABILITY_ALIAS_MAP:
        mapped = CAPABILITY_ALIAS_MAP[text]
        if mapped not in CAPABILITY_IDS:
            raise CapabilityCompilerError(
                f"alias {text!r} maps to unknown capability {mapped!r}",
                error_type="unknown_capability",
            )
        return mapped
    raise CapabilityCompilerError(
        f"unknown capability {text!r}; not in CAPABILITY_IDS and no alias",
        error_type="unknown_capability",
    )


def looks_like_narrative_metric(
    *,
    metric_semantics: Sequence[str] = (),
    required_outputs: Sequence[str] = (),
    objective: str = "",
    capability_hint: Optional[str] = None,
    source_preference: Sequence[str] = (),
    basis_hints: Sequence[str] = (),
) -> bool:
    """Detect non-IFRS / adjusted / narrative evidence needs (company-generic)."""
    blobs: List[str] = []
    blobs.extend(str(x) for x in metric_semantics or [])
    blobs.extend(str(x) for x in required_outputs or [])
    blobs.extend(str(x) for x in basis_hints or [])
    blobs.append(str(objective or ""))
    blobs.append(str(capability_hint or ""))
    blobs.extend(str(x) for x in source_preference or [])
    joined = " | ".join(blobs)
    if _NARRATIVE_METRIC_RE.search(joined):
        return True
    # Semantic field names that structured bundles do not provide exactly.
    for out in required_outputs or []:
        low = str(out).lower()
        if low.startswith("non_ifrs") or "adjusted_profit" in low:
            return True
    return False


def canonicalize_outputs(
    semantic_outputs: Sequence[str],
    *,
    target_capability: Optional[str] = None,
) -> Dict[str, Any]:
    """Separate semantic requirements from contract-canonical outputs."""
    semantic = [str(x) for x in semantic_outputs or [] if str(x).strip()]
    mapping: Dict[str, str] = {}
    canonical: List[str] = []
    dropped_semantic_only: List[str] = []

    for sem in semantic:
        if sem in CANONICAL_OUTPUT_FIELDS:
            mapping[sem] = sem
            if sem not in canonical:
                canonical.append(sem)
            continue
        if sem in SEMANTIC_OUTPUT_ALIAS_MAP:
            can = SEMANTIC_OUTPUT_ALIAS_MAP[sem]
            mapping[sem] = can
            if can not in canonical:
                canonical.append(can)
            continue
        # Unknown semantic field: keep provenance, do not hard-filter contracts
        # on the raw NL token.
        dropped_semantic_only.append(sem)
        mapping[sem] = f"__semantic_only__:{sem}"

    # Capability-default fill when planner omitted essentials.
    if target_capability and target_capability in DEFAULT_REQUIRED_OUTPUTS:
        for d in DEFAULT_REQUIRED_OUTPUTS[target_capability]:
            if d not in canonical:
                # Only add defaults when we have at least one related semantic cue
                # or empty list would leave filter too loose for market price.
                pass

    # Market latest close: ensure trade_date + close when price semantics present.
    # Drop currency/unit — OHLCV contracts typically do not provide them.
    # Do not invent hard required_outputs when the planner left them empty.
    if target_capability == "market.latest_close":
        canonical = [c for c in canonical if c not in {"currency", "unit", "adjust"}]
        if semantic:
            for need in ("trade_date", "close"):
                if need not in canonical:
                    canonical.append(need)
                    mapping.setdefault(need, need)
    if target_capability == "market.daily_ohlcv":
        # Volume routing: keep trade_date + volume when volume was requested.
        vol_asked = any(
            mapping.get(s) == "volume" or s in {"volume", "trading_volume"}
            for s in semantic
        ) or ("volume" in canonical)
        # volume_unit → volume already; drop non-OHLCV hard filters.
        soft_drop = {"currency", "unit"}
        if vol_asked:
            soft_drop.add("close")  # volume ask need not force close
        canonical = [c for c in canonical if c not in soft_drop]
        if semantic:
            if vol_asked:
                for need in ("trade_date", "volume"):
                    if need not in canonical:
                        canonical.append(need)
                        mapping.setdefault(need, need)
            else:
                price_asked = any(
                    mapping.get(s) == "close" or s in {"close", "closing_price", "latest_close"}
                    for s in semantic
                ) or ("close" in canonical)
                if price_asked:
                    for need in ("trade_date", "close"):
                        if need not in canonical:
                            canonical.append(need)
                            mapping.setdefault(need, need)

    if target_capability == "financial.official_narrative_evidence":
        # Narrative path: contract fields are evidence snippets, not IFRS rows.
        narrative_canonical = [
            c for c in canonical if c in {"snippet", "local_key", "page_number", "title"}
        ]
        if "snippet" not in narrative_canonical:
            narrative_canonical.insert(0, "snippet")
            mapping.setdefault("snippet", "snippet")
        if "local_key" not in narrative_canonical:
            narrative_canonical.append("local_key")
            mapping.setdefault("local_key", "local_key")
        canonical = narrative_canonical

    if target_capability == "financial.reported_metric_bundle":
        # Drop fields that cannot exist on structured bundle contracts.
        filtered = []
        for c in canonical:
            if c in {"snippet", "local_key", "page_number"}:
                continue
            filtered.append(c)
        canonical = filtered

    return {
        "semantic_output_requirements": semantic,
        "canonical_required_outputs": canonical,
        "semantic_to_canonical_output_map": mapping,
        "unmapped_semantic_outputs": dropped_semantic_only,
    }


def _normalize_argument_slots(slots: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in dict(slots or {}).items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = v
    return out


def _select_fallbacks(
    capability: str,
    declared: Sequence[str],
    *,
    narrative: bool,
) -> List[str]:
    selected: List[str] = []
    for raw in declared or []:
        try:
            norm = normalize_capability_id(str(raw), allow_null=False)
        except CapabilityCompilerError:
            continue
        if norm and norm != capability and norm not in selected:
            selected.append(norm)
    for alt in CAPABILITY_COMPATIBILITY.get(capability, ()):
        if alt != capability and alt not in selected:
            selected.append(alt)
    if narrative and "disclosure.list" not in selected:
        selected.append("disclosure.list")
    return selected


def compile_capability_requirement(
    *,
    task_spec: Optional[Mapping[str, Any]] = None,
    step_objective: str = "",
    metric_semantics: Optional[Sequence[str]] = None,
    period: Optional[Sequence[str]] = None,
    source_preference: Optional[Sequence[str]] = None,
    capability_hint: Optional[str] = None,
    required_output_semantics: Optional[Sequence[str]] = None,
    argument_slots: Optional[Mapping[str, Any]] = None,
    basis_adjustment_constraints: Optional[Sequence[str]] = None,
    fallback_capabilities: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compile one retrieve-step capability requirement (deterministic)."""
    task_spec = dict(task_spec or {})
    metrics = list(metric_semantics or task_spec.get("metrics") or [])
    periods = list(period or task_spec.get("periods") or [])
    basis = list(
        basis_adjustment_constraints
        or task_spec.get("metric_basis_hints")
        or task_spec.get("constraints")
        or []
    )
    prefs = list(source_preference or [])
    semantic_outs = list(required_output_semantics or [])

    narrative = looks_like_narrative_metric(
        metric_semantics=metrics,
        required_outputs=semantic_outs,
        objective=step_objective,
        capability_hint=capability_hint,
        source_preference=prefs,
        basis_hints=basis,
    )

    reasons: List[str] = []
    if narrative:
        canonical_cap = "financial.official_narrative_evidence"
        reasons.append("non_ifrs_or_narrative_metric_semantics")
        # Prefer official disclosure source.
        if "official_disclosure" not in prefs:
            prefs = ["official_disclosure"] + prefs
    else:
        canonical_cap = normalize_capability_id(
            capability_hint, allow_null=False
        )
        if capability_hint and capability_hint != canonical_cap:
            reasons.append(
                f"capability_alias:{capability_hint}->{canonical_cap}"
            )
        else:
            reasons.append("capability_passthrough")

    out_pack = canonicalize_outputs(semantic_outs, target_capability=canonical_cap)
    # Financial margin: keep attributable_profit + revenue for compute.
    if canonical_cap == "financial.reported_metric_bundle":
        for need in ("revenue", "attributable_profit"):
            # Only ensure when semantic map referenced them or metrics imply.
            mapped_vals = set(out_pack["semantic_to_canonical_output_map"].values())
            if need in mapped_vals and need not in out_pack["canonical_required_outputs"]:
                out_pack["canonical_required_outputs"].append(need)

    selected_fallback = _select_fallbacks(
        canonical_cap,
        fallback_capabilities or [],
        narrative=narrative,
    )
    slots = _normalize_argument_slots(argument_slots)

    return {
        "canonical_required_capability": canonical_cap,
        "canonical_required_outputs": list(out_pack["canonical_required_outputs"]),
        "normalized_argument_slots": slots,
        "semantic_output_requirements": list(out_pack["semantic_output_requirements"]),
        "semantic_to_canonical_output_map": dict(
            out_pack["semantic_to_canonical_output_map"]
        ),
        "unmapped_semantic_outputs": list(out_pack["unmapped_semantic_outputs"]),
        "selected_fallback_capabilities": selected_fallback,
        "source_preference": prefs,
        "period": periods,
        "metric_semantics": metrics,
        "narrative_routing": narrative,
        "compilation_reason": ";".join(reasons) or "compiled",
        "compiler_version": COMPILER_VERSION,
        "ok": True,
        "error_type": None,
    }


def compile_retrieve_step(
    step: Mapping[str, Any],
    *,
    task_spec: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile a single plan step dict in place (returns new dict)."""
    step = dict(step)
    if str(step.get("step_type") or "") != "retrieve":
        # Non-retrieve: null capability fields stay null.
        if step.get("required_capability"):
            try:
                step["required_capability"] = normalize_capability_id(
                    step.get("required_capability"), allow_null=True
                )
            except CapabilityCompilerError:
                step["required_capability"] = None
        return {
            "step": step,
            "compilation": {
                "ok": True,
                "skipped": True,
                "compiler_version": COMPILER_VERSION,
            },
        }

    try:
        compiled = compile_capability_requirement(
            task_spec=task_spec,
            step_objective=str(step.get("objective") or ""),
            metric_semantics=list((task_spec or {}).get("metrics") or []),
            period=list((task_spec or {}).get("periods") or []),
            source_preference=list(step.get("source_preference") or []),
            capability_hint=step.get("required_capability"),
            required_output_semantics=list(step.get("required_outputs") or []),
            argument_slots=dict(step.get("argument_slots") or {}),
            basis_adjustment_constraints=list(
                (task_spec or {}).get("metric_basis_hints") or []
            ),
            fallback_capabilities=list(step.get("fallback_capabilities") or []),
        )
    except CapabilityCompilerError as exc:
        return {
            "step": step,
            "compilation": {
                "ok": False,
                "error_type": exc.error_type,
                "error": exc.message,
                "compiler_version": COMPILER_VERSION,
            },
        }

    step["required_capability"] = compiled["canonical_required_capability"]
    step["required_outputs"] = list(compiled["canonical_required_outputs"])
    step["argument_slots"] = dict(compiled["normalized_argument_slots"])
    step["source_preference"] = list(compiled["source_preference"])
    step["fallback_capabilities"] = list(compiled["selected_fallback_capabilities"])
    # Provenance retained on step (model-visible; not runtime-only).
    step["semantic_output_requirements"] = list(
        compiled["semantic_output_requirements"]
    )
    step["semantic_to_canonical_output_map"] = dict(
        compiled["semantic_to_canonical_output_map"]
    )
    return {"step": step, "compilation": compiled}


def compile_plan_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Compile all retrieve steps in an apv_fin_plan_v1 payload."""
    payload = dict(payload)
    task_spec = dict(payload.get("task_spec") or {})
    plan = dict(payload.get("initial_plan") or {})
    steps_in = list(plan.get("steps") or [])
    steps_out: List[Dict[str, Any]] = []
    compilations: List[Dict[str, Any]] = []
    for raw in steps_in:
        result = compile_retrieve_step(raw, task_spec=task_spec)
        if not result["compilation"].get("ok", True):
            raise CapabilityCompilerError(
                result["compilation"].get("error") or "compile_failed",
                error_type=str(
                    result["compilation"].get("error_type") or "compile_failed"
                ),
            )
        steps_out.append(result["step"])
        compilations.append(result["compilation"])
    plan["steps"] = steps_out
    payload["task_spec"] = task_spec
    payload["initial_plan"] = plan
    payload["_capability_compilations"] = compilations
    return payload


def precompile_capability_vocabulary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize informal capability IDs before JSON-schema enum validation.

    Unknown capabilities raise CapabilityCompilerError (fail closed).
    Output canonicalization is deferred to compile_plan_payload / compile_retrieve_step
    so schema still accepts planner semantic required_outputs strings.
    """
    if not isinstance(payload, dict):
        return payload
    # Plan payload
    plan = (payload.get("initial_plan") or {}) if "initial_plan" in payload else None
    steps_containers: List[List[Dict[str, Any]]] = []
    if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
        steps_containers.append(plan["steps"])
    # Monitor patch
    patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else None
    if patch:
        for key in ("new_plan_steps", "updated_plan_steps"):
            entries = patch.get(key)
            if isinstance(entries, list):
                steps_containers.append(entries)

    for steps in steps_containers:
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "required_capability" not in step:
                continue
            cap = step.get("required_capability")
            if cap is None:
                continue
            # allow_null=False only when retrieve; for updates may set capability
            step_type = str(step.get("step_type") or "")
            allow_null = step_type != "retrieve" and step_type != ""
            if step_type == "retrieve" or (
                step_type == "" and cap is not None
            ):
                # Monitor updates often omit step_type; still normalize capability.
                step["required_capability"] = normalize_capability_id(
                    cap, allow_null=allow_null
                )
            for i, fb in enumerate(list(step.get("fallback_capabilities") or [])):
                try:
                    step["fallback_capabilities"][i] = normalize_capability_id(
                        fb, allow_null=False
                    )
                except CapabilityCompilerError:
                    # Drop invalid fallback rather than fail whole plan here;
                    # compile_plan_payload will finalize.
                    continue
    return payload
