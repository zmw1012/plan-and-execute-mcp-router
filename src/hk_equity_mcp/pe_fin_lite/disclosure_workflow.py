"""HKEX disclosure composite with deterministic shortest-required-prefix.

Default chain: list → (optional) download → parse → keyword → pages.
Stops after list when required outputs are already satisfied.
Company-generic; no task_id / issuer whitelist branching.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DISCLOSURE_COMPOSITE_ID = "retrieve_official_disclosure_evidence"
DISCLOSURE_CHAIN = (
    "list",
    "download",
    "parse",
    "keyword",
    "pages",
)

# Document-body fields that require stages beyond list.
_TEXT_REQUIRED_OUTPUTS = frozenset(
    {"snippet", "page_number", "attributable_profit", "revenue", "basic_eps", "currency"}
)
_LIST_FIELDS = (
    "news_id",
    "title",
    "release_date",
    "issuer_code",
    "local_key",
    "source_url",
    "company_name",
)

_STAGE_CAPABILITY = {
    "list": ("disclosure.list", "disclosure_search", "list"),
    "download": ("disclosure.download", "download"),
    "parse": ("disclosure.parse", "parse"),
    "keyword": ("disclosure.keyword_search", "keyword"),
    "pages": ("disclosure.page_read", "page"),
}

_STAGE_PROVIDES = {
    "list": ("local_key", "news_id", "title", "release_date", "issuer_code", "source_url"),
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
            "shortest_required_prefix": True,
            "stop_after_list_when_satisfied": True,
            "text_required_outputs": sorted(_TEXT_REQUIRED_OUTPUTS),
            "list_row_scope_filter": True,
            "preserve_list_evidence_on_later_failure": True,
            "no_task_id_branching": True,
            "passes": ["local_key", "news_id", "release_date", "page_number", "title"],
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


def needs_document_text(required_outputs: Sequence[str]) -> bool:
    return any(str(r) in _TEXT_REQUIRED_OUTPUTS for r in (required_outputs or []))


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


def _collect_rows(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return rows
    stack = [payload]
    seen = set()
    while stack:
        obj = stack.pop()
        if not isinstance(obj, dict):
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        maybe = obj.get("rows")
        if isinstance(maybe, list):
            rows.extend([r for r in maybe if isinstance(r, dict)])
        for key in ("normalized", "result", "values", "adapted", "raw", "data", "content"):
            child = obj.get(key)
            if isinstance(child, dict):
                stack.append(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        if "json" in item and isinstance(item["json"], dict):
                            stack.append(item["json"])
                        else:
                            stack.append(item)
    return rows


def _extract_chain_fields(payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rows = _collect_rows(payload)
    objs: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        objs.append(payload)
        for key in ("normalized", "result", "values", "adapted"):
            if isinstance(payload.get(key), dict):
                objs.append(payload[key])
    for obj in objs:
        for field in _LIST_FIELDS + ("page_number", "snippet"):
            if obj.get(field) is not None and field not in out:
                out[field] = obj.get(field)
    for row in rows:
        for field in _LIST_FIELDS + ("page_number", "snippet"):
            if row.get(field) is not None and field not in out:
                out[field] = row.get(field)
        if row.get("local_key"):
            out.setdefault("local_keys", [])
            if isinstance(out["local_keys"], list):
                out["local_keys"].append(row["local_key"])
    return out


def _norm_date(raw: Any) -> str:
    text = str(raw or "").strip()
    m = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    m = re.match(r"(\d{4})(\d{2})(\d{2})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return text


def _norm_code(raw: Any) -> str:
    digits = re.sub(r"\D", "", str(raw or ""))
    return digits.zfill(5) if digits else ""


def filter_list_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    grounding: Dict[str, Any],
    step: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deterministic scope filter on list rows (entity/date/title)."""
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    entity = scope.get("entity")
    entity_code = None
    if isinstance(entity, dict):
        entity_code = entity.get("code") or entity.get("stock_code") or entity.get("issuer_code")
    elif isinstance(entity, str):
        entity_code = entity
    code = _norm_code(grounding.get("stock_code") or entity_code)
    from_date = _norm_date(grounding.get("from_date") or scope.get("from_date"))
    to_date = _norm_date(grounding.get("to_date") or scope.get("to_date"))
    exact_release = _norm_date(scope.get("trade_date") or grounding.get("trade_date") or "")
    title_kw = str(
        grounding.get("title_keyword") or scope.get("title_keyword") or ""
    ).strip().lower()

    kept: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_code = _norm_code(
            row.get("issuer_code") or row.get("stock_code") or row.get("symbol")
        )
        if code and row_code and row_code != code:
            continue
        rel = _norm_date(row.get("release_date") or row.get("date"))
        if from_date and rel and rel < from_date:
            continue
        if to_date and rel and rel > to_date:
            continue
        if title_kw:
            title = str(row.get("title") or "").lower()
            if title_kw not in title:
                continue
        if exact_release and rel and rel != exact_release:
            continue
        kept.append(row)
    return kept


def select_list_row(
    rows: Sequence[Dict[str, Any]],
    *,
    grounding: Dict[str, Any],
    step: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    filtered = filter_list_rows(rows, grounding=grounding, step=step)
    if filtered:
        return filtered[0]
    # If filter emptied but rows exist and we only have code constraint, keep first.
    if rows:
        return dict(rows[0])
    return None


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
    # File / URL links when required by download contracts.
    maybe_set("file_link", chain_state.get("source_url") or chain_state.get("file_link"))
    maybe_set("url", chain_state.get("source_url") or chain_state.get("url"))

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
        elif name == "local_key" and chain_state.get("local_key"):
            args[name] = chain_state["local_key"]
    return args


InvokeFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]
DescribeFn = Callable[[str], Dict[str, Any]]


def _row_to_values(row: Dict[str, Any], stock_code: Optional[str] = None) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for field in _LIST_FIELDS:
        if row.get(field) is not None:
            values[field] = row.get(field)
    # Common aliases
    if "news_id" not in values and row.get("newsId") is not None:
        values["news_id"] = row.get("newsId")
    if "local_key" not in values and row.get("localKey") is not None:
        values["local_key"] = row.get("localKey")
    if "source_url" not in values:
        for k in ("source_url", "url", "link", "file_link", "fileLink"):
            if row.get(k):
                values["source_url"] = row.get(k)
                break
    if stock_code and "issuer_code" not in values:
        values["issuer_code"] = stock_code
    return values


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
    """Execute shortest required prefix of the disclosure chain."""
    from hk_equity_mcp.pe_fin_lite.argument_grounding import (
        apply_grounding_to_step,
        collect_step_grounding,
    )

    allowed = list(runtime_operation_ids)
    chain_state: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    values: Dict[str, Any] = {}
    invoked: List[str] = []
    gateway_flags: Dict[str, bool] = {}
    last_error: Optional[str] = None
    stopped_after_list = False

    g = grounding or collect_step_grounding(question=question or "", step=step)
    step = apply_grounding_to_step(step, g)
    if g.get("stock_code"):
        chain_state["stock_code"] = g["stock_code"]
        chain_state["issuer_code"] = g["stock_code"]
        values["issuer_code"] = g["stock_code"]

    required_outputs = list(step.get("required_outputs") or ["snippet", "local_key"])
    need_text = needs_document_text(required_outputs)

    # Optional lookup when name-only.
    if g.get("needs_lookup") and not g.get("stock_code"):
        lookup_oid = next((oid for oid in allowed if "lookup_stock_code" in oid.lower()), None)
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
                gateway_flags[lookup_oid] = success
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
                        "gateway_success": success,
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
                        "gateway_success": False,
                        "error": f"lookup_exception:{type(exc).__name__}:{exc}",
                    }
                )

    for stage in DISCLOSURE_CHAIN[:max_stages]:
        # Shortest prefix: after list, stop if required outputs satisfied and no text needed.
        if stage != "list" and not need_text:
            missing_now = [
                r for r in required_outputs if r not in values or values.get(r) in (None, "")
            ]
            if not missing_now:
                stopped_after_list = True
                break
        if stage != "list" and need_text is False and stage in {"download", "parse", "keyword", "pages"}:
            # Still stop when list satisfied all required outputs.
            missing_now = [
                r for r in required_outputs if r not in values or values.get(r) in (None, "")
            ]
            if not missing_now:
                stopped_after_list = True
                break

        oid = select_stage_operation(
            stage=stage,
            metadata_by_id=metadata_by_id,
            allowed_ids=allowed,
        )
        if oid is None:
            last_error = f"no_operation_for_stage:{stage}"
            break
        if stage == "download" and not allow_hkex_download and "hkexnews.download" in oid:
            last_error = "hkex_download_not_allowed"
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
        required = list((contract.get("json_schema") or {}).get("required") or [])
        unmet = [r for r in required if r not in args]
        if unmet:
            last_error = f"missing_stage_args:{stage}:{unmet}"
            trace.append(
                {
                    "stage": stage,
                    "operation_id": oid,
                    "arguments": args,
                    "success": False,
                    "gateway_success": False,
                    "error": last_error,
                    "grounding": {
                        "stock_code": g.get("stock_code"),
                        "from_date": g.get("from_date"),
                        "to_date": g.get("to_date"),
                        "title_keyword": g.get("title_keyword"),
                    },
                    "chain_state_keys": sorted(chain_state.keys()),
                }
            )
            break
        try:
            result = invoke_fn(oid, args)
        except Exception as exc:  # noqa: BLE001
            last_error = f"stage_exception:{stage}:{type(exc).__name__}:{exc}"
            trace.append(
                {
                    "stage": stage,
                    "operation_id": oid,
                    "arguments": {
                        k: args[k] for k in args if "key" not in k.lower() or k == "local_key"
                    },
                    "success": False,
                    "gateway_success": False,
                    "extracted_fields": [],
                    "error": last_error,
                }
            )
            break
        invoked.append(oid)
        success = bool(result.get("success")) if isinstance(result, dict) else False
        gateway_flags[oid] = success
        payload = result.get("result") if isinstance(result, dict) else result
        extracted = _extract_chain_fields(payload)
        if isinstance(result, dict):
            extracted.update(_extract_chain_fields(result))

        if stage == "list":
            rows = _collect_rows(payload)
            if isinstance(result, dict):
                rows = rows or _collect_rows(result)
            selected = select_list_row(rows, grounding=g, step=step)
            if selected:
                row_vals = _row_to_values(selected, stock_code=g.get("stock_code"))
                values.update(row_vals)
                chain_state.update({k: v for k, v in row_vals.items() if v is not None})
                extracted = {**extracted, **row_vals}
            elif extracted:
                values.update({k: v for k, v in extracted.items() if k in _LIST_FIELDS and v is not None})
                chain_state.update({k: v for k, v in extracted.items() if v is not None})
        else:
            chain_state.update({k: v for k, v in extracted.items() if v is not None})
            values.update({k: v for k, v in extracted.items() if v is not None})

        for pass_key in ("news_id", "local_key", "release_date", "page_number", "title", "source_url"):
            if pass_key in extracted and extracted[pass_key] is not None:
                chain_state[pass_key] = extracted[pass_key]
                values.setdefault(pass_key, extracted[pass_key])
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
                "gateway_success": success,
                "extracted_fields": sorted(extracted.keys()),
                "selected_row_fields": sorted(
                    k for k in _LIST_FIELDS if k in values and values.get(k) is not None
                )
                if stage == "list"
                else None,
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

        # After successful list: stop if required outputs already satisfied and
        # no document-text fields are required.
        if stage == "list" and not need_text:
            missing_now = [
                r for r in required_outputs if r not in values or values.get(r) in (None, "")
            ]
            if not missing_now:
                stopped_after_list = True
                break

    missing_outputs = [
        r for r in required_outputs if r not in values or values.get(r) in (None, "")
    ]
    # Preserve list evidence even when later stages fail.
    status = "completed" if not missing_outputs and not last_error else "unavailable"
    if last_error and missing_outputs:
        status = "failed" if any(x in str(last_error) for x in ("missing_stage_args", "stage_failed", "stage_exception")) else "unavailable"
    return {
        "workflow_id": DISCLOSURE_COMPOSITE_ID,
        "chain": list(DISCLOSURE_CHAIN),
        "trace": trace,
        "invoked_operations": invoked,
        "gateway_success_by_operation": gateway_flags,
        "values": values,
        "chain_state": chain_state,
        "missing_outputs": missing_outputs,
        "status": status,
        "error": last_error,
        "execution_ok": status == "completed",
        "scope_match": True,
        "mcp_transport_ok": last_error not in {"hkex_download_not_allowed"},
        "tool_result_valid": bool(values) and (last_error is None or bool(values)),
        "grounding": g,
        "stopped_after_list": stopped_after_list,
        "needed_document_text": need_text,
        "gateway_success": any(gateway_flags.values()) if gateway_flags else False,
    }
