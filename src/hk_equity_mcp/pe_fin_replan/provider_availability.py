"""RePlan-only runtime provider availability overlay (Sina market).

Does not mutate frozen catalogs. Filters market+eastmoney ops before R5 ranking
when condition=pe_fin_replan and a profile is loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

PROVIDER_AVAILABILITY_CONTRACT_VERSION = "pe_fin_replan_provider_availability_v1"
TOOL_READINESS_CONTRACT_VERSION = "pe_fin_replan_tool_readiness_v1"

DEFAULT_PROFILE_REL = (
    "configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml"
)

_INDEX_MARKERS = (
    "hsi",
    "hang seng",
    "恒生",
    "index",
    "指数",
)
_EQUITY_MARKERS = (
    "equity",
    "stock",
    "share",
    "ticker",
    "阿里",
    "腾讯",
    "汇丰",
    "alibaba",
    "tencent",
    "hsbc",
)


class ProviderAvailabilityError(RuntimeError):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provider_availability_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": PROVIDER_AVAILABILITY_CONTRACT_VERSION,
            "rules": [
                "condition_scope_pe_fin_replan_only",
                "disable_market_and_eastmoney_via_akshare",
                "match_domain_and_provider_not_em_suffix",
                "retain_financial_eastmoney",
                "retain_sina_market",
                "retain_hkexnews",
                "stock_hk_daily_unadjusted_only",
                "no_index_tool_for_single_equity",
                "fail_closed_disabled_binding",
                "no_catalog_mutation",
                "no_embedding_rebuild",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tool_readiness_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": TOOL_READINESS_CONTRACT_VERSION,
            "modes": ["contract_audit", "critical_live_audit", "market_sina_probe"],
            "runtime_operation_count_expected": 39,
            "live_modes_require_network": True,
            "mock_dry_run_cannot_forge_live_pass": True,
            "readiness_uses_agent_project_invoke_to_required_outputs": True,
            "adjustment_empty_string_legal": True,
            "requires_projection_success": True,
            "requires_effective_session_date": "2026-05-15",
            "reject_invalidated_probe_dirs": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_provider_availability_profile(
    path: Path,
    *,
    condition: str = "pe_fin_replan",
) -> Dict[str, Any]:
    raw_bytes = Path(path).read_bytes()
    data = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    if not isinstance(data, dict):
        raise ProviderAvailabilityError(f"invalid provider profile: {path}")
    scope = list(data.get("condition_scope") or [])
    if condition not in scope:
        raise ProviderAvailabilityError(
            f"provider profile {data.get('profile_id')!r} does not allow "
            f"condition={condition!r}; scope={scope}"
        )
    profile = deepcopy(data)
    profile["_source_path"] = str(Path(path).resolve())
    profile["_profile_sha256"] = _sha_bytes(raw_bytes)
    profile["_raw_bytes_len"] = len(raw_bytes)
    return profile


def profile_hash(profile: Dict[str, Any]) -> str:
    return str(profile.get("_profile_sha256") or "")


def _op_domain(
    operation_id: str,
    *,
    metadata_by_id: Dict[str, Dict[str, Any]],
    raw_ops: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    meta = metadata_by_id.get(operation_id) or {}
    if meta.get("domain"):
        return str(meta.get("domain")).strip().lower()
    raw = (raw_ops or {}).get(operation_id) or {}
    if raw.get("domain"):
        return str(raw.get("domain")).strip().lower()
    # Preferred domain from unified record may be embedded in raw
    return str(raw.get("preferred_domain") or meta.get("preferred_domain") or "").strip().lower()


def _op_provider(
    operation_id: str,
    *,
    metadata_by_id: Dict[str, Dict[str, Any]],
    raw_ops: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    meta = metadata_by_id.get(operation_id) or {}
    if meta.get("provider"):
        return str(meta.get("provider")).strip().lower()
    raw = (raw_ops or {}).get(operation_id) or {}
    return str(raw.get("provider") or "").strip().lower()


def apply_provider_availability_filter(
    *,
    operation_ids: Sequence[str],
    metadata_by_id: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
    raw_ops: Optional[Dict[str, Dict[str, Any]]] = None,
    records: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Remove market+eastmoney ops. Never uses `_em` name heuristics."""
    before = [str(x) for x in operation_ids]
    rule = dict(profile.get("disable_rule") or {})
    disable_domain = str(rule.get("domain") or "").strip().lower()
    disable_provider = str(rule.get("provider") or "").strip().lower()
    disabled: List[str] = []
    kept: List[str] = []

    # Enrich domain from unified records when available
    rec_domain: Dict[str, str] = {}
    if records:
        for oid, rec in records.items():
            doms = getattr(rec, "domains", None) or (rec.get("domains") if isinstance(rec, dict) else None)
            if doms:
                # Prefer market when present for disable matching
                dlist = [str(d).lower() for d in doms]
                if "market" in dlist:
                    rec_domain[str(oid)] = "market"
                elif dlist:
                    rec_domain[str(oid)] = dlist[0]

    for oid in before:
        domain = rec_domain.get(oid) or _op_domain(
            oid, metadata_by_id=metadata_by_id, raw_ops=raw_ops
        )
        provider = _op_provider(
            oid, metadata_by_id=metadata_by_id, raw_ops=raw_ops
        )
        if domain == disable_domain and provider == disable_provider:
            disabled.append(oid)
        else:
            kept.append(oid)

    preferred = dict(profile.get("preferred_operations") or {})
    return {
        "provider_profile_id": profile.get("profile_id"),
        "provider_profile_hash": profile_hash(profile),
        "operations_before_provider_filter": before,
        "operations_before_count": len(before),
        "disabled_operation_ids": disabled,
        "disabled_provider_groups": [
            {
                "domain": disable_domain,
                "provider": disable_provider,
                "operation_ids": list(disabled),
            }
        ],
        "operations_after_provider_filter": kept,
        "operations_after_count": len(kept),
        "preferred_single_equity_operation": (
            (preferred.get("single_equity_unadjusted_daily_ohlcv") or {}).get(
                "operation_id"
            )
        ),
        "preferred_index_operation": (
            (preferred.get("index_daily_ohlc") or {}).get("operation_id")
        ),
        "disabled_market_provider": disable_provider,
    }


def step_requests_adjusted_price(step: Dict[str, Any]) -> Optional[str]:
    """Return qfq/hfq if requested; else ''/None for unadjusted/unknown."""
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    for key in ("adjustment", "adjust"):
        val = scope.get(key) if scope else None
        if val is None:
            val = step.get(key)
        if val is None:
            continue
        s = str(val).strip().lower()
        if s in {"qfq", "hfq"}:
            return s
        if s in {"", "none", "unadjusted", "raw"}:
            return ""
    text = " ".join(
        str(step.get(k) or "")
        for k in ("objective", "period", "metric_basis")
    ).lower()
    if re.search(r"\bqfq\b|前复权", text):
        return "qfq"
    if re.search(r"\bhfq\b|后复权", text):
        return "hfq"
    if "unadjusted" in text or 'adjust=""' in text or "adjust=''" in text:
        return ""
    return None


def step_looks_like_index(step: Dict[str, Any], question: str = "") -> bool:
    blob = " ".join(
        [
            str(step.get("objective") or ""),
            str(step.get("entity") or ""),
            str((step.get("scope_constraints") or {}).get("entity") or "")
            if isinstance(step.get("scope_constraints"), dict)
            else "",
            question or "",
        ]
    ).lower()
    strong = ("hsi", "hang seng", "恒生", "index level", "指数行情", "index close")
    if any(m in blob for m in strong):
        return True
    # Generic "index" only when not clearly a single equity code request
    if "index" in blob or "指数" in blob:
        if not re.search(r"\b\d{4,5}\b", blob):
            return True
    return False


def step_looks_like_single_equity(step: Dict[str, Any], question: str = "") -> bool:
    blob = " ".join(
        [
            str(step.get("objective") or ""),
            question or "",
        ]
    ).lower()
    if any(m in blob for m in ("hsi", "hang seng", "恒生指数")):
        return False
    scope = step.get("scope_constraints") if isinstance(step.get("scope_constraints"), dict) else {}
    ent = scope.get("entity") if scope else None
    code = None
    if isinstance(ent, dict):
        code = ent.get("code") or ent.get("stock_code")
    elif isinstance(ent, str):
        code = ent
    code = code or step.get("entity")
    if code and re.search(r"\d{4,5}", str(code)) and "hsi" not in str(code).lower():
        return True
    return any(m in blob for m in _EQUITY_MARKERS)


def validate_bound_operation(
    *,
    operation_id: str,
    arguments: Optional[Dict[str, Any]],
    step: Dict[str, Any],
    profile: Dict[str, Any],
    filter_trace: Dict[str, Any],
    question: str = "",
    metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    raw_ops: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fail closed if binder selects disabled / capability-mismatched ops."""
    oid = str(operation_id)
    disabled = set(filter_trace.get("disabled_operation_ids") or [])
    if oid in disabled:
        raise ProviderAvailabilityError(
            f"disabled_operation_selected:{oid}"
        )
    preferred = dict(profile.get("preferred_operations") or {})
    equity_pref = preferred.get("single_equity_unadjusted_daily_ohlcv") or {}
    index_pref = preferred.get("index_daily_ohlc") or {}
    adj = step_requests_adjusted_price(step)
    args = arguments or {}
    arg_adj = str(args.get("adjust") or args.get("adjustment") or "").strip().lower()
    requested_adjustment = arg_adj if arg_adj in {"qfq", "hfq", ""} else (adj if adj is not None else "")

    provider = _op_provider(
        oid, metadata_by_id=metadata_by_id or {}, raw_ops=raw_ops
    )
    substitution = False
    substitution_reason = None
    capability_match = True

    if oid == equity_pref.get("operation_id"):
        if requested_adjustment in {"qfq", "hfq"} or arg_adj in {"qfq", "hfq"}:
            raise ProviderAvailabilityError(
                "unsupported_provider_capability:stock_hk_daily_does_not_support_"
                f"{requested_adjustment or arg_adj}"
            )
        if step_looks_like_index(step, question) and not step_looks_like_single_equity(
            step, question
        ):
            # Index-only tasks should not use single-equity daily
            capability_match = False
            raise ProviderAvailabilityError(
                "unsupported_provider_capability:stock_hk_daily_not_for_index"
            )
        substitution = True
        substitution_reason = "eastmoney_market_disabled_use_sina_equity_daily"

    if oid == index_pref.get("operation_id"):
        if step_looks_like_single_equity(step, question) and not step_looks_like_index(
            step, question
        ):
            raise ProviderAvailabilityError(
                "unsupported_provider_capability:index_tool_for_single_equity"
            )
        substitution = True
        substitution_reason = "eastmoney_market_disabled_use_sina_index_daily"

    return {
        "selected_operation_id": oid,
        "selected_operation_provider": provider,
        "provider_substitution": substitution,
        "provider_substitution_reason": substitution_reason,
        "requested_adjustment": requested_adjustment,
        "provider_capability_match": capability_match,
    }


def resume_identity_fields(profile: Optional[Dict[str, Any]], filter_trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from hk_equity_mcp.pe_fin_lite.envelope_projection import (
        MARKET_GATEWAY_PROJECTION_VERSION,
        market_gateway_projection_hash,
    )
    from hk_equity_mcp.pe_fin_replan.sina_symbol import (
        SINA_HK_SYMBOL_NORMALIZER_VERSION,
        sina_symbol_normalizer_hash,
    )

    if not profile:
        return {
            "market_provider_profile_id": None,
            "market_provider_profile_hash": None,
            "disabled_market_provider": None,
            "disabled_operation_ids": None,
            "preferred_single_equity_operation": None,
            "preferred_index_operation": None,
            "provider_availability_contract_version": None,
            "provider_availability_contract_hash": None,
            "tool_readiness_contract_version": None,
            "tool_readiness_contract_hash": None,
            "sina_symbol_normalizer_version": None,
            "sina_symbol_normalizer_hash": None,
            "market_gateway_projection_version": None,
            "market_gateway_projection_hash": None,
        }
    trace = filter_trace or {}
    preferred = dict(profile.get("preferred_operations") or {})
    return {
        "market_provider_profile_id": profile.get("profile_id"),
        "market_provider_profile_hash": profile_hash(profile),
        "disabled_market_provider": str(
            (profile.get("disable_rule") or {}).get("provider") or ""
        ),
        "disabled_operation_ids": list(trace.get("disabled_operation_ids") or []),
        "preferred_single_equity_operation": (
            (preferred.get("single_equity_unadjusted_daily_ohlcv") or {}).get(
                "operation_id"
            )
        ),
        "preferred_index_operation": (
            (preferred.get("index_daily_ohlc") or {}).get("operation_id")
        ),
        "provider_availability_contract_version": PROVIDER_AVAILABILITY_CONTRACT_VERSION,
        "provider_availability_contract_hash": provider_availability_contract_hash(),
        "tool_readiness_contract_version": TOOL_READINESS_CONTRACT_VERSION,
        "tool_readiness_contract_hash": tool_readiness_contract_hash(),
        "sina_symbol_normalizer_version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
        "sina_symbol_normalizer_hash": sina_symbol_normalizer_hash(),
        "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
        "market_gateway_projection_hash": market_gateway_projection_hash(),
    }


def load_tool_readiness_artifact(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProviderAvailabilityError("invalid tool readiness artifact")
    return data


def assert_tool_readiness_for_live(
    *,
    artifact_path: Optional[Path],
    expected_profile_hash: str,
    mock: bool,
    dry_run: bool,
    expected_normalizer_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Live runs require a matching readiness artifact. Mock/dry-run must not forge pass."""
    from hk_equity_mcp.pe_fin_replan.sina_symbol import (
        SINA_HK_SYMBOL_NORMALIZER_VERSION,
        sina_symbol_normalizer_hash,
    )

    if mock or dry_run:
        return {
            "required": False,
            "status": "skipped_offline",
            "live_pass_forged": False,
        }
    if artifact_path is None or not Path(artifact_path).is_file():
        raise ProviderAvailabilityError(
            "require_tool_readiness: readiness artifact missing"
        )
    # Reject invalidated v1 probe directory by path marker / sibling INVALIDATED.json
    art_path = Path(artifact_path).resolve()
    if (art_path.parent / "INVALIDATED.json").is_file():
        raise ProviderAvailabilityError(
            "require_tool_readiness: readiness artifact invalidated; "
            "use market_sina_probe_v3"
        )
    from hk_equity_mcp.pe_fin_lite.envelope_projection import (
        MARKET_GATEWAY_PROJECTION_VERSION,
        market_gateway_projection_hash,
    )

    raw = Path(artifact_path).read_bytes()
    artifact_hash = _sha_bytes(raw)
    data = json.loads(raw.decode("utf-8"))
    norm_hash = expected_normalizer_hash or sina_symbol_normalizer_hash()
    proj_hash = market_gateway_projection_hash()

    required_true = {
        "status": "pass",
        "gateway_success": True,
        "normalization_success": True,
        "projection_success": True,
        "scope_match": True,
        "provider": "sina_via_akshare",
        "network_called": True,
        "live_probe_pass": True,
        "normalized_symbol": "09988",
        "effective_session_date": "2026-05-15",
    }
    for key, expected in required_true.items():
        if data.get(key) != expected:
            raise ProviderAvailabilityError(
                f"require_tool_readiness: {key}={data.get(key)!r} "
                f"expected={expected!r} (invalidated/fail artifacts not accepted)"
            )
    if list(data.get("missing_outputs") or []) != []:
        raise ProviderAvailabilityError(
            "require_tool_readiness: missing_outputs must be empty"
        )
    if data.get("market_provider_profile_hash") != expected_profile_hash:
        raise ProviderAvailabilityError(
            "require_tool_readiness: market_provider_profile_hash mismatch"
        )
    if data.get("mode") not in {"market_sina_probe", "critical_live_audit"}:
        raise ProviderAvailabilityError(
            f"require_tool_readiness: unexpected mode={data.get('mode')!r}"
        )
    if data.get("sina_symbol_normalizer_version") != SINA_HK_SYMBOL_NORMALIZER_VERSION:
        raise ProviderAvailabilityError(
            "require_tool_readiness: sina_symbol_normalizer_version mismatch"
        )
    if data.get("sina_symbol_normalizer_hash") != norm_hash:
        raise ProviderAvailabilityError(
            "require_tool_readiness: sina_symbol_normalizer_hash mismatch"
        )
    if data.get("market_gateway_projection_version") != MARKET_GATEWAY_PROJECTION_VERSION:
        raise ProviderAvailabilityError(
            "require_tool_readiness: market_gateway_projection_version mismatch"
        )
    if data.get("market_gateway_projection_hash") != proj_hash:
        raise ProviderAvailabilityError(
            "require_tool_readiness: market_gateway_projection_hash mismatch"
        )
    return {
        "required": True,
        "status": "pass",
        "artifact_path": str(art_path),
        "artifact_sha256": artifact_hash,
        "live_pass_forged": False,
        "sina_symbol_normalizer_version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
        "sina_symbol_normalizer_hash": norm_hash,
        "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
        "market_gateway_projection_hash": proj_hash,
    }
