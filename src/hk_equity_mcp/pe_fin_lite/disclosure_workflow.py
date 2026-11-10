"""Fixed HKEX disclosure composite: list → download → parse → keyword → pages.

Company-generic; no task_id / issuer whitelist branching.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

DISCLOSURE_COMPOSITE_ID = "retrieve_official_disclosure_evidence"
DISCLOSURE_CHAIN = (
    "list",
    "download",
    "parse",
    "keyword",
    "pages",
)

# Capability tags preferred when selecting operations from metadata.
_STAGE_CAPABILITY = {
    "list": ("disclosure.list", "disclosure_search", "list"),
    "download": ("disclosure.download", "download"),
    "parse": ("disclosure.parse", "parse"),
    "keyword": ("disclosure.keyword_search", "keyword"),
    "pages": ("disclosure.page_read", "page"),
}

_STAGE_PROVIDES = {
    "list": ("local_key", "news_id", "title", "release_date"),
    "download": ("local_key",),
    "parse": ("local_key",),
    "keyword": ("snippet", "local_key", "page_number"),
    "pages": ("snippet", "page_number", "local_key"),
}


def disclosure_workflow_hash() -> str:
    blob = json.dumps(
        {
            "workflow_id": DISCLOSURE_COMPOSITE_ID,
            "chain": list(DISCLOSURE_CHAIN),
            "no_task_id_branching": True,
            "passes": ["local_key", "news_id", "release_date", "page_number"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_disclosure_composite_step(step: Dict[str, Any]) -> bool:
    wf = str(step.get("workflow") or "")
    if wf == DISCLOSURE_COMPOSITE_ID:
        return True
    domain = str(step.get("domain") or "").lower()
    objective = str(step.get("objective") or "").lower()
    if domain == "disclosure" and any(
        tok in objective
        for tok in ("official disclosure", "hkex", "announcement evidence", "filing text")
    ):
        return True
    return False


def _score_operation_for_stage(
    *,
    operation_id: str,
    stage: str,
    metadata: Dict[str, Any],
) -> int:
    caps = [str(c).lower() for c in (metadata.get("capabilities") or [])]
    provides = [str(p).lower() for p in (metadata.get("provides") or [])]
    oid = operation_id.lower()
    score = 0
    for tag in _STAGE_CAPABILITY.get(stage, ()):
        tag_l = tag.lower()
        if tag_l in caps or tag_l in oid:
            score += 5
    for need in _STAGE_PROVIDES.get(stage, ()):
        if need.lower() in provides or need.lower() in oid:
            score += 2
    # Prefer hkexnews chain for official disclosure.
    if "hkexnews" in oid:
        score += 3
    if "ah_disclosure" in oid and stage == "list":
        score += 1
    return score


def select_stage_operation(
    *,
    stage: str,
    metadata_by_id: Dict[str, Dict[str, Any]],
    allowed_ids: Sequence[str],
) -> Optional[str]:
    best = None
    best_score = -1
    for oid in allowed_ids:
        meta = metadata_by_id.get(oid) or {}
        score = _score_operation_for_stage(
            operation_id=oid, stage=stage, metadata=meta
        )
        if score > best_score:
            best_score = score
            best = oid
    if best_score <= 0:
        return None
    return best


def _extract_chain_fields(payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    objs: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        objs.append(payload)
        for key in ("normalized", "result", "values", "adapted"):
            if isinstance(payload.get(key), dict):
                objs.append(payload[key])
        for obj in objs:
            if isinstance(obj.get("rows"), list):
                rows.extend([r for r in obj["rows"] if isinstance(r, dict)])
            for field in ("local_key", "news_id", "release_date", "page_number", "title", "snippet"):
                if obj.get(field) is not None and field not in out:
                    out[field] = obj.get(field)
    for row in rows:
        for field in ("local_key", "news_id", "release_date", "page_number", "title", "snippet"):
            if row.get(field) is not None and field not in out:
                out[field] = row.get(field)
        if "local_keys" not in out and row.get("local_key"):
            out.setdefault("local_keys", [])
            if isinstance(out["local_keys"], list):
                out["local_keys"].append(row["local_key"])
    return out


def build_stage_arguments(
    *,
    stage: str,
    contract: Dict[str, Any],
    chain_state: Dict[str, Any],
    step: Dict[str, Any],
    grounding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    schema = contract.get("json_schema") or {}
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    args: Dict[str, Any] = {}
    g = grounding or {}
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    stock_code = (
        g.get("stock_code")
        or chain_state.get("stock_code")
        or chain_state.get("issuer_code")
        or step.get("entity")
    )
    from_date = g.get("from_date") or scope.get("from_date") or step.get("date")
    to_date = g.get("to_date") or scope.get("to_date") or step.get("date")
    title_kw = g.get("title_keyword") or scope.get("title_keyword")

    def maybe_set(name: str, value: Any) -> None:
        if name in props and value is not None and value != "":
            args[name] = value

    maybe_set("local_key", chain_state.get("local_key"))
    maybe_set("news_id", chain_state.get("news_id"))
    maybe_set("release_date", chain_state.get("release_date") or scope.get("trade_date"))
    maybe_set("page_number", chain_state.get("page_number") or 1)
    maybe_set("keyword", title_kw or step.get("objective"))
    maybe_set("query", title_kw or step.get("objective"))
    maybe_set("title", title_kw)
    maybe_set("stock_code", stock_code)
    maybe_set("symbol", stock_code)
    maybe_set("code", stock_code)
    maybe_set("issuer_code", stock_code)
    maybe_set("start_date", from_date or step.get("period"))
    maybe_set("end_date", to_date or step.get("period"))
    maybe_set("from_date", from_date)
    maybe_set("to_date", to_date)

    # Fill remaining required with chain/step values when possible.
    for name in required:
        if name in args:
            continue
        if name in chain_state and chain_state[name] is not None:
            args[name] = chain_state[name]
        elif name in ("stock", "symbol", "code", "stock_code", "issuer_code") and stock_code:
            args[name] = stock_code
        elif name in ("start_date", "from_date") and from_date:
            args[name] = from_date
        elif name in ("end_date", "to_date") and to_date:
            args[name] = to_date
        elif name in ("keyword", "query", "title") and title_kw:
            args[name] = title_kw
    return args


InvokeFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]
DescribeFn = Callable[[str], Dict[str, Any]]


def run_disclosure_composite(
    *,
    step: Dict[str, Any],
    metadata_by_id: Dict[str, Dict[str, Any]],
    runtime_operation_ids: Sequence[str],
    invoke_fn: InvokeFn,
    describe_fn: DescribeFn,
    allow_hkex_download: bool = False,
    max_stages: int = 5,
    question: str = "",
    grounding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute fixed disclosure chain; each underlying call recorded in trace."""
    from hk_equity_mcp.pe_fin_lite.argument_grounding import (
        apply_grounding_to_step,
        collect_step_grounding,
    )

    allowed = list(runtime_operation_ids)
    chain_state: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    values: Dict[str, Any] = {}
    invoked: List[str] = []
    missing: List[str] = []
    last_error: Optional[str] = None

    g = grounding or collect_step_grounding(question=question or "", step=step)
    step = apply_grounding_to_step(step, g)
    if g.get("stock_code"):
        chain_state["stock_code"] = g["stock_code"]
        chain_state["issuer_code"] = g["stock_code"]
        values["issuer_code"] = g["stock_code"]

    # Optional company-name → code lookup (never invent). Only when no explicit code.
    if g.get("needs_lookup") and not g.get("stock_code"):
        lookup_oid = None
        for oid in allowed:
            if "lookup_stock_code" in oid.lower():
                lookup_oid = oid
                break
        if lookup_oid is not None:
            described = describe_fn(lookup_oid)
            contract = (described or {}).get("contract") or {}
            props = (contract.get("json_schema") or {}).get("properties") or {}
            args: Dict[str, Any] = {}
            name = g.get("company_name")
            for key in ("company_name", "name", "query", "keyword"):
                if key in props and name:
                    args[key] = name
            try:
                lookup_res = invoke_fn(lookup_oid, args)
                invoked.append(lookup_oid)
                success = bool(lookup_res.get("success")) if isinstance(lookup_res, dict) else False
                extracted = _extract_chain_fields(
                    lookup_res.get("result") if isinstance(lookup_res, dict) else lookup_res
                )
                if isinstance(lookup_res, dict):
                    extracted.update(_extract_chain_fields(lookup_res))
                code = (
                    extracted.get("issuer_code")
                    or extracted.get("stock_code")
                    or extracted.get("symbol")
                    or extracted.get("code")
                )
                trace.append(
                    {
                        "stage": "lookup_stock_code",
                        "operation_id": lookup_oid,
                        "arguments": args,
                        "success": success,
                        "extracted_fields": sorted(extracted.keys()),
                        "error": None if success else "lookup_failed",
                    }
                )
                if code:
                    chain_state["stock_code"] = code
                    chain_state["issuer_code"] = code
                    values["issuer_code"] = code
                    g = dict(g)
                    g["stock_code"] = code
                    g["code_available"] = True
                    g["needs_lookup"] = False
                    step = apply_grounding_to_step(step, g)
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    {
                        "stage": "lookup_stock_code",
                        "operation_id": lookup_oid,
                        "arguments": args,
                        "success": False,
                        "error": f"lookup_exception:{type(exc).__name__}:{exc}",
                    }
                )

    for stage in DISCLOSURE_CHAIN[:max_stages]:
        oid = select_stage_operation(
            stage=stage,
            metadata_by_id=metadata_by_id,
            allowed_ids=allowed,
        )
        if oid is None:
            last_error = f"no_operation_for_stage:{stage}"
            missing.extend([p for p in _STAGE_PROVIDES.get(stage, ()) if p not in values])
            break
        if stage == "download" and not allow_hkex_download and "hkexnews.download" in oid:
            last_error = "hkex_download_not_allowed"
            missing.append("local_key")
            break
        described = describe_fn(oid)
        contract = (described or {}).get("contract") or {}
        args = build_stage_arguments(
            stage=stage,
            contract=contract,
            chain_state=chain_state,
            step=step,
            grounding=g,
        )
        # Required args gate — never bypass argument validation.
        required = list((contract.get("json_schema") or {}).get("required") or [])
        unmet = [r for r in required if r not in args]
        if unmet:
            last_error = f"missing_stage_args:{stage}:{unmet}"
            missing.extend(unmet)
            trace.append(
                {
                    "stage": stage,
                    "operation_id": oid,
                    "arguments": args,
                    "success": False,
                    "error": last_error,
                    "grounding": {
                        "stock_code": g.get("stock_code"),
                        "from_date": g.get("from_date"),
                        "to_date": g.get("to_date"),
                        "title_keyword": g.get("title_keyword"),
                    },
                }
            )
            break
        try:
            result = invoke_fn(oid, args)
        except Exception as exc:  # noqa: BLE001 — convert to StepResult, do not crash runner
            last_error = f"stage_exception:{stage}:{type(exc).__name__}:{exc}"
            trace.append(
                {
                    "stage": stage,
                    "operation_id": oid,
                    "arguments": {
                        k: args[k] for k in args if "key" not in k.lower() or k == "local_key"
                    },
                    "success": False,
                    "extracted_fields": [],
                    "error": last_error,
                }
            )
            break
        invoked.append(oid)
        success = bool(result.get("success")) if isinstance(result, dict) else False
        extracted = _extract_chain_fields(
            result.get("result") if isinstance(result, dict) else result
        )
        if isinstance(result, dict):
            extracted.update(_extract_chain_fields(result))
        chain_state.update({k: v for k, v in extracted.items() if v is not None})
        values.update({k: v for k, v in extracted.items() if v is not None})
        # Propagate list identity fields for later stages.
        for pass_key in ("news_id", "local_key", "release_date", "page_number", "title"):
            if pass_key in extracted and extracted[pass_key] is not None:
                chain_state[pass_key] = extracted[pass_key]
        if g.get("stock_code") and "issuer_code" not in values:
            values["issuer_code"] = g["stock_code"]
        trace.append(
            {
                "stage": stage,
                "operation_id": oid,
                "arguments": {
                    k: args[k] for k in args if "key" not in k.lower() or k == "local_key"
                },
                "success": success,
                "extracted_fields": sorted(extracted.keys()),
                "error": None
                if success
                else (
                    result.get("error")
                    if isinstance(result, dict)
                    else "invoke_failed"
                ),
            }
        )
        if not success:
            last_error = f"stage_failed:{stage}"
            break

    required_outputs = list(step.get("required_outputs") or ["snippet", "local_key"])
    missing_outputs = [r for r in required_outputs if r not in values or values.get(r) in (None, "")]
    status = "completed" if not missing_outputs and not last_error else "unavailable"
    return {
        "workflow_id": DISCLOSURE_COMPOSITE_ID,
        "chain": list(DISCLOSURE_CHAIN),
        "trace": trace,
        "invoked_operations": invoked,
        "values": values,
        "chain_state": chain_state,
        "missing_outputs": missing_outputs,
        "status": status,
        "error": last_error,
        "execution_ok": status == "completed",
        "scope_match": True,
        "mcp_transport_ok": last_error not in {"hkex_download_not_allowed"},
        "tool_result_valid": bool(values) and last_error is None,
        "grounding": g,
    }
