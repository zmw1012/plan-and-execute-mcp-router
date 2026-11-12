"""HKEX disclosure composite with deterministic shortest-required-prefix.

Default chain: list → (optional) download → parse → keyword → pages.
Stops after list when required outputs are already satisfied.
Company-generic; no task_id / issuer whitelist branching.

Disclosure policy v1.2:
- propagate explicit from_date/to_date; always honor benchmark cutoff
- period → bounded date window only via versioned generic rules
- local row selection by issuer/date/cutoff then document-type relevance
- at most one deterministic title-broadening fallback (not global replan)
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF

DISCLOSURE_COMPOSITE_ID = "retrieve_official_disclosure_evidence"
DISCLOSURE_POLICY_VERSION = "pe_fin_lite_disclosure_policy_v1_2"
PERIOD_WINDOW_POLICY_VERSION = "period_window_v1"
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
    "file_link",
    "company_name",
)

# Real HKEX list envelopes use announcements / normalized.data, not only "rows".
_ROW_LIST_KEYS = (
    "rows",
    "announcements",
    "items",
    "sample",
    "results",
    "records",
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


# Metric phrases are not announcement titles; map to document-type intents.
_METRIC_PHRASE_MARKERS = (
    "non-ifrs",
    "non ifrs",
    "non-gaap",
    "non gaap",
    "net income",
    "attributable profit",
    "attributable to",
    "basic eps",
    "diluted eps",
    "earnings per share",
)

# Generic document-type intents (company-agnostic).
_DOC_TYPE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "annual_results": (
        "annual results",
        "final results",
        "yearly results",
        "annual report",
    ),
    "interim_results": (
        "interim results",
        "half-year results",
        "half year results",
        "semi-annual results",
        "interim report",
    ),
    "results": ("results announcement", "results"),
    "agm_notice": (
        "agm notice",
        "notice of annual general meeting",
        "annual general meeting",
        "agm",
    ),
}

_FY_PERIOD = re.compile(
    r"\b(?:FY|financial\s+year)\s*[-/]?\s*(20\d{2})\b", re.IGNORECASE
)
_HY_PERIOD = re.compile(
    r"\b(?:HY|H1|1H|interim)\s*[-/]?\s*(20\d{2})\b", re.IGNORECASE
)


def disclosure_workflow_hash() -> str:
    blob = json.dumps(
        {
            "workflow_id": DISCLOSURE_COMPOSITE_ID,
            "policy_version": DISCLOSURE_POLICY_VERSION,
            "period_window_policy": PERIOD_WINDOW_POLICY_VERSION,
            "chain": list(DISCLOSURE_CHAIN),
            "shortest_required_prefix": True,
            "stop_after_list_when_satisfied": True,
            "text_required_outputs": sorted(_TEXT_REQUIRED_OUTPUTS),
            "list_row_scope_filter": True,
            "list_row_list_keys": list(_ROW_LIST_KEYS),
            "normalize_announcement_aliases": True,
            "preserve_list_evidence_on_later_failure": True,
            "no_task_id_branching": True,
            "benchmark_cutoff_enforced": True,
            "default_cutoff": DEFAULT_DATASET_CUTOFF,
            "deterministic_title_broadening_max": 1,
            "no_default_first_row": True,
            "document_type_intents": sorted(_DOC_TYPE_PATTERNS.keys()),
            "metric_phrase_not_title": True,
            "passes": [
                "local_key",
                "news_id",
                "release_date",
                "title",
                "file_link",
                "source_url",
                "page_number",
                "start_page",
            ],
            "stage_arg_aliases": {
                "keywords_from_title_keyword": True,
                "start_page_from_page_number": True,
                "title_keyword_passthrough": True,
                "document_type_list_title": True,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def derive_period_date_window(
    period: Any,
    *,
    cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Versioned, company-generic period → bounded window under cutoff.

    Returns (from_date, to_date, rule_id) or (None, None, None) when unsafe.
    """
    text = str(period or "").strip()
    if not text:
        return None, None, None
    cut = _norm_date(cutoff) or DEFAULT_DATASET_CUTOFF
    m_fy = _FY_PERIOD.search(text)
    if m_fy:
        year = int(m_fy.group(1))
        # FY results typically publish in the following calendar year.
        from_date = f"{year}-01-01"
        to_date = f"{year + 1}-08-12"
        if to_date > cut:
            to_date = cut
        if from_date > cut:
            return None, None, None
        return from_date, to_date, f"{PERIOD_WINDOW_POLICY_VERSION}:FY"
    m_hy = _HY_PERIOD.search(text)
    if m_hy:
        year = int(m_hy.group(1))
        from_date = f"{year}-06-01"
        to_date = f"{year}-12-31"
        if to_date > cut:
            to_date = cut
        if from_date > cut:
            return None, None, None
        return from_date, to_date, f"{PERIOD_WINDOW_POLICY_VERSION}:HY"
    return None, None, None


def is_metric_phrase_title(title_keyword: Any) -> bool:
    text = str(title_keyword or "").strip().lower()
    if not text:
        return False
    return any(m in text for m in _METRIC_PHRASE_MARKERS)


def infer_document_type_intent(
    *,
    title_keyword: Any = None,
    objective: Any = None,
    question: Any = None,
    metric_basis: Any = None,
) -> Optional[str]:
    """Map free text to a generic document-type intent (no issuer/gold specifics)."""
    blobs = [
        str(title_keyword or ""),
        str(objective or ""),
        str(question or ""),
        str(metric_basis or ""),
    ]
    joined = " ".join(blobs).lower()
    # Prefer more specific intents first.
    for intent in ("agm_notice", "annual_results", "interim_results", "results"):
        for pat in _DOC_TYPE_PATTERNS[intent]:
            if pat in joined:
                return intent
    if is_metric_phrase_title(title_keyword) or is_metric_phrase_title(metric_basis):
        return "results"
    return None


def document_type_list_title(intent: Optional[str]) -> Optional[str]:
    """Title string safe to send to list APIs (document type, not metric phrase)."""
    if intent == "agm_notice":
        return "AGM"
    if intent == "annual_results":
        return "Annual Results"
    if intent == "interim_results":
        return "Interim Results"
    if intent == "results":
        return "Results"
    return None


def enrich_disclosure_grounding(
    grounding: Dict[str, Any],
    *,
    step: Optional[Dict[str, Any]] = None,
    question: str = "",
    cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> Dict[str, Any]:
    """Fill dates / document-type intent without inventing issuer codes."""
    g = dict(grounding or {})
    step = step or {}
    cut = _norm_date(cutoff) or DEFAULT_DATASET_CUTOFF
    g["benchmark_cutoff"] = cut

    from_date = _norm_date(g.get("from_date"))
    to_date = _norm_date(g.get("to_date"))
    period = g.get("period") or step.get("period")
    if (not from_date or not to_date) and period:
        d_from, d_to, rule = derive_period_date_window(period, cutoff=cut)
        if d_from and not from_date:
            from_date = d_from
            g["from_date"] = d_from
            g["period_window_rule"] = rule
        if d_to and not to_date:
            to_date = d_to
            g["to_date"] = d_to
            g["period_window_rule"] = rule

    if from_date and from_date > cut:
        from_date = cut
        g["from_date"] = cut
    if to_date and to_date > cut:
        to_date = cut
        g["to_date"] = cut
    if not to_date:
        to_date = cut
        g["to_date"] = cut

    intent = infer_document_type_intent(
        title_keyword=g.get("title_keyword"),
        objective=step.get("objective"),
        question=question,
        metric_basis=g.get("metric_basis"),
    )
    g["document_type_intent"] = intent
    raw_title = str(g.get("title_keyword") or "").strip()
    if raw_title and is_metric_phrase_title(raw_title):
        g["title_is_metric_phrase"] = True
        g["list_title_keyword"] = document_type_list_title(intent)
    elif raw_title:
        g["title_is_metric_phrase"] = False
        # Use raw title when it already looks like a document-type phrase.
        g["list_title_keyword"] = raw_title
    else:
        g["title_is_metric_phrase"] = False
        g["list_title_keyword"] = document_type_list_title(intent)
    return g


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


def _is_announcement_row(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("local_key") is not None or obj.get("localKey") is not None:
        return True
    if obj.get("news_id") is not None or obj.get("newsId") is not None:
        return True
    if obj.get("file_link") is not None or obj.get("fileLink") is not None:
        return True
    title = obj.get("title")
    release = obj.get("release_date") or obj.get("releaseDate")
    return bool(title) and bool(release)


def _extend_rows_from_list(rows: List[Dict[str, Any]], items: Any) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if _is_announcement_row(item):
            rows.append(item)
        elif isinstance(item, dict) and "json" in item and isinstance(item["json"], dict):
            nested = item["json"]
            if _is_announcement_row(nested):
                rows.append(nested)


def _collect_rows(payload: Any) -> List[Dict[str, Any]]:
    """Collect announcement/list rows from gateway envelopes and MCP payloads.

    Accepts recorded HKEX shapes:
    - ``rows`` / ``announcements`` lists
    - ``normalized`` as list or ``{status, data:[...]}`` / ``{announcements:[...]}``
    - ``raw.content[].json`` MCP content blocks
    - resolver wrapper ``{success, result: RealMcpResult.to_dict()}``
    """
    rows: List[Dict[str, Any]] = []
    if payload is None:
        return rows
    if isinstance(payload, list):
        _extend_rows_from_list(rows, payload)
        return rows
    if not isinstance(payload, dict):
        return rows

    stack: List[Any] = [payload]
    seen: set = set()
    while stack:
        obj = stack.pop()
        if isinstance(obj, list):
            _extend_rows_from_list(rows, obj)
            for item in obj:
                if isinstance(item, dict):
                    if "json" in item and isinstance(item["json"], (dict, list)):
                        stack.append(item["json"])
                    else:
                        stack.append(item)
            continue
        if not isinstance(obj, dict):
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if _is_announcement_row(obj):
            rows.append(obj)
        for key in _ROW_LIST_KEYS:
            maybe = obj.get(key)
            if isinstance(maybe, list):
                _extend_rows_from_list(rows, maybe)
        for key in (
            "normalized",
            "result",
            "values",
            "adapted",
            "raw",
            "data",
            "content",
            "payload",
        ):
            child = obj.get(key)
            if isinstance(child, (dict, list)):
                stack.append(child)
    # Stable de-dupe by local_key/news_id/title+date when possible.
    deduped: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for row in rows:
        key = (
            str(row.get("local_key") or row.get("localKey") or ""),
            str(row.get("news_id") or row.get("newsId") or ""),
            str(row.get("title") or ""),
            str(row.get("release_date") or row.get("releaseDate") or ""),
        )
        if key in seen_keys and any(key):
            continue
        seen_keys.add(key)
        deduped.append(row)
    return deduped


def _extract_chain_fields(payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rows = _collect_rows(payload)
    objs: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        objs.append(payload)
        for key in ("normalized", "result", "values", "adapted", "data"):
            child = payload.get(key)
            if isinstance(child, dict):
                objs.append(child)
    for obj in objs:
        for field in _LIST_FIELDS + ("page_number", "snippet", "start_page"):
            if obj.get(field) is not None and field not in out:
                out[field] = obj.get(field)
        # Alias hydration for single-object payloads.
        if "local_key" not in out and obj.get("localKey") is not None:
            out["local_key"] = obj.get("localKey")
        if "news_id" not in out and obj.get("newsId") is not None:
            out["news_id"] = obj.get("newsId")
        if "source_url" not in out:
            for k in ("source_url", "url", "link", "file_link", "fileLink"):
                if obj.get(k):
                    out["source_url"] = obj.get(k)
                    out.setdefault("file_link", obj.get(k))
                    break
    for row in rows:
        norm = _row_to_values(row)
        for field, value in norm.items():
            if value is not None and field not in out:
                out[field] = value
        if norm.get("local_key"):
            out.setdefault("local_keys", [])
            if isinstance(out["local_keys"], list):
                out["local_keys"].append(norm["local_key"])
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
    require_title_match: bool = True,
) -> List[Dict[str, Any]]:
    """Deterministic hard filter: issuer / date window / cutoff (title optional)."""
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    entity = scope.get("entity")
    entity_code = None
    if isinstance(entity, dict):
        entity_code = entity.get("code") or entity.get("stock_code") or entity.get("issuer_code")
    elif isinstance(entity, str):
        entity_code = entity
    code = _norm_code(grounding.get("stock_code") or entity_code)
    from_date = _norm_date(
        grounding.get("from_date") or scope.get("from_date") or scope.get("start_date")
    )
    to_date = _norm_date(
        grounding.get("to_date") or scope.get("to_date") or scope.get("end_date")
    )
    cutoff = _norm_date(grounding.get("benchmark_cutoff") or DEFAULT_DATASET_CUTOFF)
    if to_date and cutoff and to_date > cutoff:
        to_date = cutoff
    if not to_date:
        to_date = cutoff
    exact_release = _norm_date(
        scope.get("release_date")
        or grounding.get("release_date")
        or scope.get("trade_date")
        or grounding.get("trade_date")
        or ""
    )
    title_kw = ""
    if require_title_match:
        title_kw = str(
            grounding.get("list_title_keyword")
            or grounding.get("title_keyword")
            or scope.get("title_keyword")
            or ""
        ).strip().lower()
        # Never hard-filter on metric phrases that are not announcement titles.
        if is_metric_phrase_title(title_kw):
            title_kw = str(grounding.get("list_title_keyword") or "").strip().lower()

    kept: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_code = _norm_code(
            row.get("issuer_code")
            or row.get("stock_code")
            or row.get("symbol")
            or row.get("code")
        )
        if code and row_code and row_code != code:
            continue
        rel = _norm_date(
            row.get("release_date") or row.get("releaseDate") or row.get("date")
        )
        if from_date and rel and rel < from_date:
            continue
        if to_date and rel and rel > to_date:
            continue
        if cutoff and rel and rel > cutoff:
            continue
        if title_kw:
            title = str(row.get("title") or "").lower()
            if title_kw not in title:
                continue
        if exact_release and rel and rel != exact_release:
            continue
        kept.append(row)
    return kept


def _doc_type_relevance_score(title: str, intent: Optional[str]) -> int:
    t = (title or "").lower()
    if not intent:
        return 0
    score = 0
    for pat in _DOC_TYPE_PATTERNS.get(intent, ()):
        if pat in t:
            score = max(score, 10 + len(pat))
    # Penalize clearly mismatched document types for results/AGM intents.
    if intent in {"results", "annual_results", "interim_results"}:
        if "tender offer" in t or "tender offers" in t:
            score -= 50
        if "next day disclosure" in t and "result" not in t:
            score -= 20
    if intent == "agm_notice":
        if "tender offer" in t:
            score -= 50
        if "agm" in t or "annual general meeting" in t:
            score += 20
    return score


def _row_sort_key(
    row: Dict[str, Any],
    *,
    intent: Optional[str] = None,
) -> Tuple[int, str, str, str]:
    """Relevance first, then newest release_date, then news_id/local_key."""
    title = str(row.get("title") or "")
    rel = _norm_date(row.get("release_date") or row.get("releaseDate") or "")
    news = str(row.get("news_id") or row.get("newsId") or "")
    local = str(row.get("local_key") or row.get("localKey") or "")
    return (_doc_type_relevance_score(title, intent), rel, news, local)


def rank_list_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    grounding: Dict[str, Any],
    step: Dict[str, Any],
    require_title_match: bool = True,
) -> Dict[str, Any]:
    """Filter + rank with explicit accept/reject reasons (no default first-row)."""
    intent = grounding.get("document_type_intent")
    hard = filter_list_rows(
        rows, grounding=grounding, step=step, require_title_match=False
    )
    soft = filter_list_rows(
        rows, grounding=grounding, step=step, require_title_match=require_title_match
    )
    rejected: List[Dict[str, Any]] = []
    hard_ids = {id(r) for r in hard}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if id(row) in hard_ids:
            continue
        rel = _norm_date(row.get("release_date") or row.get("releaseDate") or "")
        cutoff = _norm_date(grounding.get("benchmark_cutoff") or DEFAULT_DATASET_CUTOFF)
        reason = "issuer_or_date_mismatch"
        if cutoff and rel and rel > cutoff:
            reason = "beyond_benchmark_cutoff"
        rejected.append(
            {
                "title": str(row.get("title") or "")[:160],
                "release_date": rel,
                "reason": reason,
            }
        )

    pool = list(soft)
    title_rejects: List[Dict[str, Any]] = []
    if require_title_match and hard and not soft:
        for row in hard:
            title_rejects.append(
                {
                    "title": str(row.get("title") or "")[:160],
                    "release_date": _norm_date(
                        row.get("release_date") or row.get("releaseDate") or ""
                    ),
                    "reason": "title_or_document_type_mismatch",
                }
            )

    # Soft relevance filter: drop strongly negative scores when intent known.
    scored_pool: List[Dict[str, Any]] = []
    for row in pool:
        score = _doc_type_relevance_score(str(row.get("title") or ""), intent)
        if intent and score < 0:
            rejected.append(
                {
                    "title": str(row.get("title") or "")[:160],
                    "release_date": _norm_date(
                        row.get("release_date") or row.get("releaseDate") or ""
                    ),
                    "reason": "document_type_penalty",
                    "relevance_score": score,
                }
            )
            continue
        scored_pool.append(row)

    ordered = sorted(
        scored_pool,
        key=lambda r: _row_sort_key(r if isinstance(r, dict) else {}, intent=intent),
        reverse=True,
    )
    selected = dict(ordered[0]) if ordered and isinstance(ordered[0], dict) else None
    selection_reason = None
    if selected is not None:
        selection_reason = {
            "relevance_score": _doc_type_relevance_score(
                str(selected.get("title") or ""), intent
            ),
            "document_type_intent": intent,
            "release_date": _norm_date(
                selected.get("release_date") or selected.get("releaseDate") or ""
            ),
            "require_title_match": require_title_match,
        }
    return {
        "selected": selected,
        "ordered": ordered,
        "n_hard": len(hard),
        "n_soft": len(soft),
        "n_scored": len(scored_pool),
        "rejected": rejected[:20] + title_rejects[:10],
        "selection_reason": selection_reason,
    }


def select_list_row(
    rows: Sequence[Dict[str, Any]],
    *,
    grounding: Dict[str, Any],
    step: Dict[str, Any],
    require_title_match: bool = True,
) -> Optional[Dict[str, Any]]:
    ranked = rank_list_rows(
        rows,
        grounding=grounding,
        step=step,
        require_title_match=require_title_match,
    )
    return ranked.get("selected")


def build_stage_arguments(
    *,
    stage: str,
    contract: Dict[str, Any],
    chain_state: Dict[str, Any],
    step: Dict[str, Any],
    grounding: Optional[Dict[str, Any]] = None,
    omit_title: bool = False,
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
    to_date = (
        g.get("to_date")
        or scope.get("to_date")
        or g.get("benchmark_cutoff")
        or DEFAULT_DATASET_CUTOFF
    )
    cutoff = _norm_date(g.get("benchmark_cutoff") or DEFAULT_DATASET_CUTOFF)
    to_date_n = _norm_date(to_date)
    if to_date_n and cutoff and to_date_n > cutoff:
        to_date = cutoff
    # List title: document-type intent only; never raw metric phrase.
    if omit_title:
        title_kw = None
    else:
        title_kw = g.get("list_title_keyword")
        if title_kw is None and not g.get("title_is_metric_phrase"):
            title_kw = (
                g.get("title_keyword")
                or scope.get("title_keyword")
                or chain_state.get("title")
            )
        if is_metric_phrase_title(title_kw):
            title_kw = document_type_list_title(g.get("document_type_intent"))
    # Keyword stage may still use metric / objective phrases for body search.
    body_kw = (
        g.get("title_keyword")
        or scope.get("title_keyword")
        or step.get("objective")
    )
    page_number = chain_state.get("page_number") or chain_state.get("start_page") or 1

    def maybe_set(name: str, value: Any) -> None:
        if name in props and value is not None and value != "":
            args[name] = value

    maybe_set("local_key", chain_state.get("local_key"))
    maybe_set("news_id", chain_state.get("news_id"))
    maybe_set(
        "release_date",
        chain_state.get("release_date")
        or scope.get("release_date")
        or scope.get("trade_date"),
    )
    maybe_set("page_number", page_number)
    maybe_set("start_page", page_number)
    if stage in {"keyword", "pages"}:
        maybe_set("keyword", body_kw)
        maybe_set("query", body_kw)
    else:
        maybe_set("keyword", title_kw)
        maybe_set("query", title_kw)
        maybe_set("title", title_kw or chain_state.get("title"))
        maybe_set("title_keyword", title_kw)
    maybe_set("stock_code", stock_code)
    maybe_set("symbol", stock_code)
    maybe_set("code", stock_code)
    maybe_set("issuer_code", stock_code)
    # Dates: never substitute bare period strings into start/end.
    maybe_set("start_date", from_date)
    maybe_set("end_date", to_date)
    maybe_set("from_date", from_date)
    maybe_set("to_date", to_date)
    # File / URL links when required by download contracts.
    maybe_set(
        "file_link",
        chain_state.get("file_link") or chain_state.get("source_url"),
    )
    maybe_set("url", chain_state.get("source_url") or chain_state.get("url"))
    if "keywords" in props:
        kw_src = body_kw if stage == "keyword" else (title_kw or body_kw)
        if isinstance(kw_src, list):
            args["keywords"] = [str(x) for x in kw_src if str(x).strip()]
        elif kw_src:
            args["keywords"] = [str(kw_src)]

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
        elif name in ("keyword", "query", "title", "title_keyword") and title_kw:
            args[name] = title_kw
        elif name == "local_key" and chain_state.get("local_key"):
            args[name] = chain_state["local_key"]
        elif name == "news_id" and chain_state.get("news_id"):
            args[name] = chain_state["news_id"]
        elif name == "start_page":
            args[name] = page_number
        elif name == "keywords":
            kw_src = body_kw if stage == "keyword" else (title_kw or body_kw)
            if kw_src:
                args[name] = [str(kw_src)] if not isinstance(kw_src, list) else list(kw_src)
    return args


InvokeFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]
DescribeFn = Callable[[str], Dict[str, Any]]


def _row_to_values(row: Dict[str, Any], stock_code: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a list/announcement row into workflow values (issuer-generic)."""
    values: Dict[str, Any] = {}
    if not isinstance(row, dict):
        return values
    for field in _LIST_FIELDS:
        if row.get(field) is not None:
            values[field] = row.get(field)
    # Common aliases from real HKEX / MCP envelopes.
    if "news_id" not in values and row.get("newsId") is not None:
        values["news_id"] = row.get("newsId")
    if "local_key" not in values and row.get("localKey") is not None:
        values["local_key"] = row.get("localKey")
    if "release_date" not in values and row.get("releaseDate") is not None:
        values["release_date"] = row.get("releaseDate")
    if "issuer_code" not in values:
        code = row.get("stock_code") or row.get("symbol") or row.get("code")
        if code is not None:
            values["issuer_code"] = _norm_code(code) or code
    if "company_name" not in values and row.get("sec_name") is not None:
        values["company_name"] = row.get("sec_name")
    link = None
    for k in ("file_link", "fileLink", "source_url", "url", "link"):
        if row.get(k):
            link = row.get(k)
            break
    if link:
        values.setdefault("file_link", link)
        values.setdefault("source_url", link)
    if stock_code and "issuer_code" not in values:
        values["issuer_code"] = _norm_code(stock_code) or stock_code
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
    eea_enabled: bool = False,
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
    list_evidence: Dict[str, Any] = {}
    invoked: List[str] = []
    gateway_flags: Dict[str, bool] = {}
    last_error: Optional[str] = None
    stopped_after_list = False

    g = grounding or collect_step_grounding(question=question or "", step=step)
    g = enrich_disclosure_grounding(g, step=step, question=question or "")
    step = apply_grounding_to_step(step, g)
    if g.get("stock_code"):
        chain_state["stock_code"] = g["stock_code"]
        chain_state["issuer_code"] = g["stock_code"]
        values["issuer_code"] = g["stock_code"]

    required_outputs = list(step.get("required_outputs") or ["snippet", "local_key"])
    need_text = needs_document_text(required_outputs)
    title_broadened = False
    selection_obs: Dict[str, Any] = {}
    list_ordered_rows: List[Dict[str, Any]] = []
    alternate_list_row_used = False

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
        omit_title = False
        args = build_stage_arguments(
            stage=stage,
            contract=contract,
            chain_state=chain_state,
            step=step,
            grounding=g,
            omit_title=omit_title,
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
                        "list_title_keyword": g.get("list_title_keyword"),
                        "document_type_intent": g.get("document_type_intent"),
                        "benchmark_cutoff": g.get("benchmark_cutoff"),
                    },
                    "chain_state_keys": sorted(chain_state.keys()),
                }
            )
            break

        def _invoke_once(call_args: Dict[str, Any]) -> Dict[str, Any]:
            return invoke_fn(oid, call_args)

        try:
            result = _invoke_once(args)
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
        # Resolver wraps RealMcpResult under "result"; also accept bare envelopes.
        payload = result.get("result") if isinstance(result, dict) else result
        extracted = _extract_chain_fields(payload)
        if isinstance(result, dict):
            extracted.update(_extract_chain_fields(result))
        if isinstance(payload, dict) and isinstance(payload.get("normalized"), (dict, list)):
            extracted.update(_extract_chain_fields(payload.get("normalized")))

        if stage == "list":
            rows = _collect_rows(payload)
            if isinstance(result, dict):
                rows = rows or _collect_rows(result)
            if isinstance(payload, dict) and not rows:
                rows = _collect_rows(payload.get("normalized"))
            ranked = rank_list_rows(rows, grounding=g, step=step, require_title_match=True)
            selected = ranked.get("selected")
            # Deterministic broadening (≤1): keep issuer/date/cutoff, drop title.
            used_title = bool(
                args.get("title_keyword") or args.get("title") or args.get("keyword")
            )
            if (
                selected is None
                and not title_broadened
                and (g.get("list_title_keyword") or used_title)
                and success
            ):
                title_broadened = True
                if rows:
                    # Local broaden: first list already returned rows.
                    ranked = rank_list_rows(
                        rows, grounding=g, step=step, require_title_match=False
                    )
                    selected = ranked.get("selected")
                else:
                    # API returned 0 under title constraint → one re-list without title.
                    broad_args = build_stage_arguments(
                        stage=stage,
                        contract=contract,
                        chain_state=chain_state,
                        step=step,
                        grounding=g,
                        omit_title=True,
                    )
                    try:
                        broad_result = _invoke_once(broad_args)
                    except Exception as exc:  # noqa: BLE001
                        last_error = (
                            f"stage_exception:list_broaden:{type(exc).__name__}:{exc}"
                        )
                        trace.append(
                            {
                                "stage": "list_broaden",
                                "operation_id": oid,
                                "arguments": {
                                    k: broad_args[k]
                                    for k in broad_args
                                    if "key" not in k.lower() or k == "local_key"
                                },
                                "success": False,
                                "gateway_success": False,
                                "error": last_error,
                                "title_broadened": True,
                            }
                        )
                        break
                    invoked.append(oid)
                    broad_success = (
                        bool(broad_result.get("success"))
                        if isinstance(broad_result, dict)
                        else False
                    )
                    gateway_flags[oid] = gateway_flags.get(oid, False) or broad_success
                    broad_payload = (
                        broad_result.get("result")
                        if isinstance(broad_result, dict)
                        else broad_result
                    )
                    rows = _collect_rows(broad_payload) or _collect_rows(broad_result)
                    ranked = rank_list_rows(
                        rows, grounding=g, step=step, require_title_match=False
                    )
                    selected = ranked.get("selected")
                    result = broad_result
                    success = broad_success
                    payload = broad_payload
                    args = broad_args
                    extracted = _extract_chain_fields(payload)
                    if isinstance(result, dict):
                        extracted.update(_extract_chain_fields(result))
            selection_obs = {
                "n_list_rows": len(rows),
                "n_hard": ranked.get("n_hard"),
                "n_soft": ranked.get("n_soft"),
                "n_scored": ranked.get("n_scored"),
                "selection_reason": ranked.get("selection_reason"),
                "rejected_sample": ranked.get("rejected") or [],
                "title_broadened": title_broadened,
            }
            list_ordered_rows = [
                r for r in (ranked.get("ordered") or []) if isinstance(r, dict)
            ]
            if selected:
                row_vals = _row_to_values(selected, stock_code=g.get("stock_code"))
                list_evidence.update(row_vals)
                values.update(row_vals)
                chain_state.update({k: v for k, v in row_vals.items() if v is not None})
                extracted = {**extracted, **row_vals}
            elif extracted:
                row_vals = {
                    k: v
                    for k, v in extracted.items()
                    if k in _LIST_FIELDS and v is not None
                }
                list_evidence.update(row_vals)
                values.update(row_vals)
                chain_state.update({k: v for k, v in extracted.items() if v is not None})
        else:
            chain_state.update({k: v for k, v in extracted.items() if v is not None})
            values.update({k: v for k, v in extracted.items() if v is not None})

        for pass_key in (
            "news_id",
            "local_key",
            "release_date",
            "page_number",
            "start_page",
            "title",
            "source_url",
            "file_link",
        ):
            if pass_key in extracted and extracted[pass_key] is not None:
                chain_state[pass_key] = extracted[pass_key]
                values.setdefault(pass_key, extracted[pass_key])
        # Keyword hits often expose page_number → start_page for pages stage.
        if chain_state.get("page_number") is not None and "start_page" not in chain_state:
            chain_state["start_page"] = chain_state["page_number"]
        if g.get("stock_code") and "issuer_code" not in values:
            values["issuer_code"] = g["stock_code"]

        selected_fields = None
        if stage == "list":
            selected_fields = sorted(
                k
                for k in _LIST_FIELDS
                if k in list_evidence and list_evidence.get(k) is not None
            )

        trace.append(
            {
                "stage": stage if not title_broadened else (
                    "list_broaden" if stage == "list" and title_broadened else stage
                ),
                "operation_id": oid,
                "arguments": {
                    k: args[k] for k in args if "key" not in k.lower() or k == "local_key"
                },
                "success": success,
                "gateway_success": success,
                "extracted_fields": sorted(extracted.keys()),
                "selected_row_fields": selected_fields,
                "n_list_rows": selection_obs.get("n_list_rows")
                if stage == "list"
                else None,
                "selection_reason": selection_obs.get("selection_reason")
                if stage == "list"
                else None,
                "rejected_sample": selection_obs.get("rejected_sample")
                if stage == "list"
                else None,
                "title_broadened": title_broadened if stage == "list" else False,
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
            # EEA: at most one second-candidate switch on download/parse failure.
            # Never erase already-verified list evidence.
            if (
                eea_enabled
                and stage in {"download", "parse"}
                and not alternate_list_row_used
                and len(list_ordered_rows) >= 2
            ):
                alt = list_ordered_rows[1]
                alt_vals = _row_to_values(alt, stock_code=g.get("stock_code"))
                # Keep list_evidence; replace document pointers only.
                for k in ("news_id", "local_key", "title", "release_date", "source_url", "file_link"):
                    if alt_vals.get(k) is not None:
                        chain_state[k] = alt_vals[k]
                        values.setdefault(k, alt_vals[k])
                alternate_list_row_used = True
                trace.append(
                    {
                        "stage": f"{stage}_alternate_list_row",
                        "success": False,
                        "gateway_success": False,
                        "error": f"stage_failed:{stage};switching_to_second_candidate",
                        "alternate_list_row_used": True,
                    }
                )
                # Retry same stage once with second candidate (counts as MCP via re-invoke).
                args = build_stage_arguments(
                    stage=stage,
                    contract=contract,
                    chain_state=chain_state,
                    step=step,
                    grounding=g,
                    omit_title=False,
                )
                try:
                    result = _invoke_once(args)
                except Exception as exc:  # noqa: BLE001
                    last_error = f"stage_exception:{stage}_alt:{type(exc).__name__}:{exc}"
                    break
                invoked.append(oid)
                success = bool(result.get("success")) if isinstance(result, dict) else False
                gateway_flags[oid] = gateway_flags.get(oid, False) or success
                payload = result.get("result") if isinstance(result, dict) else result
                extracted = _extract_chain_fields(payload)
                if isinstance(result, dict):
                    extracted.update(_extract_chain_fields(result))
                trace.append(
                    {
                        "stage": f"{stage}_second_candidate",
                        "operation_id": oid,
                        "arguments": {
                            k: args[k]
                            for k in args
                            if "key" not in k.lower() or k == "local_key"
                        },
                        "success": success,
                        "gateway_success": success,
                        "extracted_fields": sorted(extracted.keys()),
                        "alternate_list_row_used": True,
                        "error": None if success else (
                            result.get("error") if isinstance(result, dict) else "invoke_failed"
                        ),
                    }
                )
                if not success:
                    last_error = f"stage_failed:{stage}_second_candidate"
                    break
                chain_state.update({k: v for k, v in extracted.items() if v is not None})
                values.update({k: v for k, v in extracted.items() if v is not None})
            else:
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

    # Later-stage failures must not erase list evidence already obtained.
    if list_evidence:
        for k, v in list_evidence.items():
            if v is not None:
                values.setdefault(k, v)
                chain_state.setdefault(k, v)

    missing_outputs = [
        r for r in required_outputs if r not in values or values.get(r) in (None, "")
    ]
    status = "completed" if not missing_outputs and not last_error else "unavailable"
    if last_error and missing_outputs:
        status = (
            "failed"
            if any(
                x in str(last_error)
                for x in ("missing_stage_args", "stage_failed", "stage_exception")
            )
            else "unavailable"
        )
    return {
        "workflow_id": DISCLOSURE_COMPOSITE_ID,
        "chain": list(DISCLOSURE_CHAIN),
        "trace": trace,
        "invoked_operations": invoked,
        "gateway_success_by_operation": gateway_flags,
        "values": values,
        "list_evidence": list_evidence,
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
        "title_broadened": title_broadened,
        "selection_observability": selection_obs,
        "disclosure_policy_version": DISCLOSURE_POLICY_VERSION,
        "alternate_list_row_used": alternate_list_row_used,
        "eea_enabled": bool(eea_enabled),
    }
