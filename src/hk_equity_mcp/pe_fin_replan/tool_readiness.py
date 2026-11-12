"""PE-Fin RePlan tool readiness audit (contract / critical-live / sina probe).

Contract mode is zero-network. Live modes are opt-in and must not run in this
offline engineering turn unless explicitly invoked.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.pe_fin_replan.provider_availability import (
    DEFAULT_PROFILE_REL,
    TOOL_READINESS_CONTRACT_VERSION,
    apply_provider_availability_filter,
    load_provider_availability_profile,
    profile_hash,
    tool_readiness_contract_hash,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

EXPECTED_RUNTIME_OPERATION_COUNT = 39


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_contract_audit(
    *,
    root: Path,
    output_dir: Path,
    profile_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Zero-network contract audit over all runtime operations + provider overlay."""
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    bundle = build_unified_catalog(root=root)
    meta_path = root / "configs/operations/hk_unified_operation_metadata_v1.yaml"
    metadata_by_id = load_and_validate_metadata(
        meta_path, catalog=bundle, root=root
    )
    runtime_ids = list(bundle.stats.runtime_operation_ids)
    profile_path = Path(profile_path) if profile_path else root / DEFAULT_PROFILE_REL
    profile = load_provider_availability_profile(
        profile_path, condition="pe_fin_replan"
    )
    filt = apply_provider_availability_filter(
        operation_ids=runtime_ids,
        metadata_by_id=metadata_by_id,
        profile=profile,
        raw_ops=dict(bundle.raw_ops),
        records=dict(bundle.records),
    )

    rows: List[Dict[str, Any]] = []
    issues: List[str] = []
    seen: set = set()
    for oid in runtime_ids:
        if oid in seen:
            issues.append(f"duplicate_operation_id:{oid}")
        seen.add(oid)
        raw = bundle.raw_ops.get(oid) or {}
        rec = bundle.records.get(oid)
        meta = metadata_by_id.get(oid) or {}
        runtime_exposed = bool(getattr(rec, "runtime_exposed", False))
        is_blocked = bool(getattr(rec, "is_blocked", False))
        server_id = str(
            raw.get("server_id")
            or getattr(rec, "server_id", None)
            or meta.get("server_id")
            or ""
        )
        params = raw.get("parameters") or raw.get("arguments") or []
        if isinstance(params, dict):
            param_names = sorted(params.keys())
            required = [
                k
                for k, v in params.items()
                if isinstance(v, dict) and v.get("required")
            ]
        elif isinstance(params, list):
            param_names = [
                str(p.get("name"))
                for p in params
                if isinstance(p, dict) and p.get("name")
            ]
            required = [
                str(p.get("name"))
                for p in params
                if isinstance(p, dict) and p.get("required") and p.get("name")
            ]
        else:
            param_names = []
            required = []
        provider = meta.get("provider") or raw.get("provider")
        domains = list(getattr(rec, "domains", None) or [])
        if not domains and meta.get("domain"):
            domains = [meta.get("domain")]
        provides = meta.get("provides") or raw.get("provides") or []
        execution = raw.get("execution") or {}
        row = {
            "operation_id": oid,
            "runtime_exposed": runtime_exposed,
            "is_blocked": is_blocked,
            "server_id": server_id,
            "server_available": bool(server_id),
            "argument_param_names": param_names,
            "required_arguments": required,
            "provider": provider,
            "domains": domains,
            "adapter_mapping": {
                "kind": execution.get("kind"),
                "interface": execution.get("interface")
                or raw.get("underlying_aktools_interface"),
                "tool": execution.get("tool") or raw.get("exposed_mcp_tool"),
            },
            "provides": provides,
            "normalized_output_contract_present": bool(
                raw.get("normalized_output")
                or raw.get("output_schema")
                or provides
                or execution
            ),
            "disabled_by_provider_profile": oid
            in set(filt.get("disabled_operation_ids") or []),
        }
        if not runtime_exposed:
            issues.append(f"runtime_not_exposed:{oid}")
        if is_blocked:
            issues.append(f"blocked_leakage:{oid}")
        if oid in (bundle.stats.blocked_operation_ids or []):
            issues.append(f"blocked_id_in_runtime:{oid}")
        if not server_id:
            issues.append(f"missing_server:{oid}")
        rows.append(row)

    if len(runtime_ids) != EXPECTED_RUNTIME_OPERATION_COUNT:
        issues.append(
            f"runtime_count_mismatch:expected={EXPECTED_RUNTIME_OPERATION_COUNT}"
            f":got={len(runtime_ids)}"
        )

    disabled = list(filt.get("disabled_operation_ids") or [])
    retained_sina = [
        oid
        for oid in runtime_ids
        if oid
        in {
            "aktools.stock_hk_daily",
            "aktools.stock_hk_index_daily_sina",
        }
    ]
    for need in (
        "aktools.stock_hk_daily",
        "aktools.stock_hk_index_daily_sina",
    ):
        if need not in runtime_ids:
            issues.append(f"missing_preferred:{need}")
        if need in disabled:
            issues.append(f"preferred_disabled:{need}")

    fin_em_disabled = [
        oid
        for oid in disabled
        if "financial" in str(
            (metadata_by_id.get(oid) or {}).get("domain")
            or ""
        ).lower()
        or "financial"
        in [
            str(d).lower()
            for d in (getattr(bundle.records.get(oid), "domains", None) or [])
        ]
    ]
    if fin_em_disabled:
        issues.append(f"financial_disabled:{fin_em_disabled}")

    status = "pass" if not issues else "fail"
    summary = {
        "mode": "contract_audit",
        "status": status,
        "network_called": False,
        "live_pass_forged": False,
        "tool_readiness_contract_version": TOOL_READINESS_CONTRACT_VERSION,
        "tool_readiness_contract_hash": tool_readiness_contract_hash(),
        "market_provider_profile_id": profile.get("profile_id"),
        "market_provider_profile_hash": profile_hash(profile),
        "runtime_operation_count": len(runtime_ids),
        "expected_runtime_operation_count": EXPECTED_RUNTIME_OPERATION_COUNT,
        "unique_operation_ids": len(seen),
        "disabled_operation_ids": disabled,
        "operations_after_provider_filter_count": filt.get("operations_after_count"),
        "retained_sina_market_operations": retained_sina,
        "issue_count": len(issues),
        "issues": issues,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(output_dir / "contract_audit_summary.json", summary)
    _atomic_write_json(
        output_dir / "contract_audit_operations.json",
        {"operations": rows, "provider_filter": filt},
    )
    # Explicitly not a live readiness pass artifact.
    readiness_stub = {
        "mode": "contract_audit",
        "status": "not_live_pass",
        "network_called": False,
        "live_pass_forged": False,
        "note": "contract_audit cannot satisfy --require-tool-readiness",
        "market_provider_profile_hash": profile_hash(profile),
        "tool_readiness_contract_hash": tool_readiness_contract_hash(),
    }
    _atomic_write_json(output_dir / "readiness_stub.json", readiness_stub)
    return summary


def run_critical_live_audit(*, root: Path, output_dir: Path, **_: Any) -> Dict[str, Any]:
    raise RuntimeError(
        "critical_live_audit not implemented in this turn; use --market-sina-probe "
        "for market readiness before Smoke-5 v4"
    )


def _load_env_file(path: Optional[Path]) -> None:
    if path is None:
        return
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"env file not found: {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, val = text.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in __import__("os").environ:
            __import__("os").environ[key] = val


def _classify_probe_failure(
    *,
    error: Optional[str],
    error_type: Optional[str],
    gateway_success: bool,
    bar: Dict[str, Any],
    missing_outputs: Sequence[str],
    scope_match: bool,
) -> Optional[str]:
    if error_type == "invalid_symbol_format" or (
        error and "invalid_symbol_format" in str(error)
    ):
        return "invalid_symbol_format"
    err = str(error or "").lower()
    if any(x in err for x in ("rate limit", "429", "too many requests")):
        return "rate_limit"
    if any(x in err for x in ("timeout", "connection", "transport", "network")):
        return "transport_failure"
    if gateway_success:
        # Success path but schema/date missing
        has_date = bool(bar.get("trade_date") or bar.get("date"))
        if not has_date or "'date'" in err or "keyerror" in err:
            return "provider_schema_drift"
        if missing_outputs and not scope_match:
            return "scope_mismatch"
        if missing_outputs:
            return "normalization_failure"
        return None
    if any(x in err for x in ("empty", "no session", "no_session", "no data")):
        return "empty_result"
    if "'date'" in err or "调用接口出错" in str(error or ""):
        # Often caused by bad symbol → empty/malformed frame; prefer schema drift
        # only when gateway claimed success. On hard fail after format fix, keep
        # market_provider_unavailable as last resort.
        if "09988.hk" in err:
            return "invalid_symbol_format"
        return "provider_schema_drift"
    if error_type in {
        "invalid_symbol_format",
        "provider_schema_drift",
        "transport_failure",
        "rate_limit",
        "empty_result",
        "normalization_failure",
        "scope_mismatch",
    }:
        return str(error_type)
    return "market_provider_unavailable"


def run_market_sina_probe(
    *,
    root: Path,
    output_dir: Path,
    profile_path: Optional[Path] = None,
    env_file: Optional[str] = None,
) -> Dict[str, Any]:
    """One-shot live Sina equity OHLCV readiness probe (no LLM / embedding).

    v2 invoke symbol must be five-digit numeric ``09988`` (not ``09988.HK``).
    """
    import time

    from hk_equity_mcp.pe_fin_lite.issuer_code import canonicalize_hk_issuer_code
    from hk_equity_mcp.pe_fin_replan.sina_symbol import (
        SINA_HK_SYMBOL_NORMALIZER_VERSION,
        normalize_sina_hk_symbol_v1,
        sina_symbol_normalizer_hash,
    )
    from hk_equity_mcp.real_mcp.market_gateway import MarketRealMcpGateway
    from hk_equity_mcp.real_mcp.market_normalize import MARKET_CUTOFF, to_ymd

    root = Path(root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_env_file(Path(env_file) if env_file else None)

    profile_path = Path(profile_path) if profile_path else root / DEFAULT_PROFILE_REL
    profile = load_provider_availability_profile(
        profile_path, condition="pe_fin_replan"
    )

    requested_symbol = "09988"
    symbol_meta = normalize_sina_hk_symbol_v1(requested_symbol)
    args = {
        "symbol": requested_symbol,
        "target_date": "2026-05-15",
        "adjustment": "",
        "adjust": "",
        "offset": -400,
        "max_rows": 400,
    }
    t0 = time.perf_counter()
    network_called = False
    failure_class = None
    gateway_success = False
    normalization_success = False
    missing_outputs: List[str] = []
    required_outputs = [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjustment",
        "currency",
        "issuer_code",
    ]
    raw_ref = None
    scope_match = False
    bar: Dict[str, Any] = {}
    error_text: Optional[str] = None
    error_type: Optional[str] = None
    invoked_symbol = symbol_meta["normalized_symbol"]

    gw = MarketRealMcpGateway()
    try:
        gw.open()
        network_called = True
        result = gw.invoke("aktools.stock_hk_daily", args)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        gateway_success = bool(result.success)
        error_text = result.error
        error_type = getattr(result, "error_type", None)
        norm = result.normalized if isinstance(result.normalized, dict) else {}
        invoked_symbol = str(
            norm.get("symbol_format_normalized")
            or norm.get("symbol")
            or args["symbol"]
        )
        data = norm.get("data")
        if isinstance(data, dict):
            bar = data
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            bar = data[0]
        raw_path = output_dir / "raw_stock_hk_daily_09988.json"
        _atomic_write_json(
            raw_path,
            {
                "operation_id": "aktools.stock_hk_daily",
                "original_requested_symbol": requested_symbol,
                "normalized_invoked_symbol": invoked_symbol,
                "arguments_redacted": {
                    "symbol": requested_symbol,
                    "target_date": args["target_date"],
                    "adjust": "",
                },
                "success": gateway_success,
                "error": result.error,
                "error_type": error_type,
                "normalized": norm,
            },
        )
        raw_ref = str(raw_path)

        if gateway_success:
            # Schema drift: success envelope but no date column/fields.
            if not (bar.get("trade_date") or bar.get("date")):
                failure_class = "provider_schema_drift"
                missing_outputs.append("trade_date")
            for key in required_outputs:
                if key in missing_outputs:
                    continue
                val = bar.get(key)
                if key == "trade_date" and val in (None, ""):
                    val = bar.get("date")
                if val in (None, ""):
                    missing_outputs.append(key)
            issuer = canonicalize_hk_issuer_code(
                bar.get("issuer_code") or invoked_symbol
            )
            if issuer != "09988":
                missing_outputs.append("issuer_code_mismatch")
            adj = bar.get("adjustment")
            if adj is None:
                adj = norm.get("adjustment_mode")
            if adj != "":
                missing_outputs.append("adjustment")
            currency = bar.get("currency") or norm.get("currency_or_unit")
            if currency != "HKD":
                missing_outputs.append("currency")
            eff = (
                bar.get("trade_date")
                or bar.get("date")
                or norm.get("effective_session_date")
            )
            if str(eff) != args["target_date"]:
                missing_outputs.append("target_date_mismatch")
            from hk_equity_mcp.real_mcp.market_normalize import parse_session_date

            d = parse_session_date(eff)
            if d is None or d > MARKET_CUTOFF:
                missing_outputs.append("cutoff_violation")
            normalization_success = not missing_outputs
            scope_match = normalization_success and str(eff) == args["target_date"]
            if not normalization_success and failure_class is None:
                failure_class = _classify_probe_failure(
                    error=error_text,
                    error_type=error_type,
                    gateway_success=True,
                    bar=bar,
                    missing_outputs=missing_outputs,
                    scope_match=scope_match,
                )
        else:
            failure_class = _classify_probe_failure(
                error=error_text,
                error_type=error_type,
                gateway_success=False,
                bar=bar,
                missing_outputs=missing_outputs,
                scope_match=False,
            )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        err_s = str(exc)
        error_text = err_s
        if "invalid_symbol_format" in err_s:
            failure_class = "invalid_symbol_format"
        else:
            failure_class = "transport_failure"
        gateway_success = False
        normalization_success = False
        missing_outputs = [f"exception:{type(exc).__name__}"]
        raw_ref = err_s[:500]
    finally:
        try:
            gw.close()
        except Exception:  # noqa: BLE001
            pass

    live_probe_pass = bool(
        gateway_success
        and normalization_success
        and scope_match
        and not missing_outputs
        and invoked_symbol == "09988"
    )
    status = "pass" if live_probe_pass else "fail"
    artifact = {
        "mode": "market_sina_probe",
        "probe_dir_version": "v2",
        "status": status,
        "network_called": network_called,
        "live_pass_forged": False,
        "live_probe_pass": live_probe_pass,
        "market_provider_profile_id": profile.get("profile_id"),
        "market_provider_profile_hash": profile_hash(profile),
        "tool_readiness_contract_version": TOOL_READINESS_CONTRACT_VERSION,
        "tool_readiness_contract_hash": tool_readiness_contract_hash(),
        "sina_symbol_normalizer_version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
        "sina_symbol_normalizer_hash": sina_symbol_normalizer_hash(),
        "operation_id": "aktools.stock_hk_daily",
        "provider": "sina_via_akshare",
        "original_requested_symbol": requested_symbol,
        "normalized_symbol": invoked_symbol,
        "symbol_format_original": symbol_meta["symbol_format_original"],
        "symbol_format_normalized": symbol_meta["symbol_format_normalized"],
        "argument_transform_applied": True,
        "arguments_redacted": {
            "symbol": requested_symbol,
            "target_date": "2026-05-15",
            "adjust": "",
        },
        "gateway_success": gateway_success,
        "normalization_success": normalization_success,
        "required_outputs": required_outputs,
        "missing_outputs": missing_outputs,
        "scope_match": scope_match,
        "latency_ms": latency_ms,
        "failure_class": failure_class,
        "raw_result_ref": raw_ref,
        "cutoff": to_ymd(MARKET_CUTOFF),
        "engineering_readiness_only": True,
        "not_accuracy_result": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    readiness_path = output_dir / "readiness.json"
    _atomic_write_json(readiness_path, artifact)
    artifact["artifact_path"] = str(readiness_path)
    artifact["artifact_sha256"] = _sha_bytes(readiness_path.read_bytes())
    return artifact


def build_future_sina_probe_command(root: Path) -> str:
    root = Path(root).resolve()
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path("python")
    return (
        f'cd "{root}" && \\\n'
        f'PYTHONPATH=src "{py}" scripts/run_pe_fin_replan_tool_readiness_audit_v1.py \\\n'
        f"  --market-sina-probe \\\n"
        f"  --market-provider-profile "
        f"configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml \\\n"
        f"  --output-dir outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe_v2 \\\n"
        f"  --env-file .env"
    )


def build_future_smoke5_v4_command(root: Path) -> str:
    root = Path(root).resolve()
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path("python")
    cache = root / ".cache" / "pe_fin_lite_operation_embeddings_v1"
    return (
        f'cd "{root}" && \\\n'
        f'PYTHONPATH=src "{py}" scripts/run_hk_unified_pe_fin_lite_experiment.py \\\n'
        f"  --split exploratory_all \\\n"
        f"  --replan-smoke5 \\\n"
        f"  --replan \\\n"
        f"  --confirm-retire-heldout \\\n"
        f"  --allow-hkex-download \\\n"
        f"  --require-complete-index \\\n"
        f"  --market-provider-profile "
        f"configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml \\\n"
        f"  --require-tool-readiness \\\n"
        f"  --tool-readiness-artifact "
        f"outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe_v2/readiness.json \\\n"
        f"  --env-file .env \\\n"
        f"  --embedding-env-file .env \\\n"
        f"  --embedding-cache-dir \"{cache}\" \\\n"
        f"  --output-dir outputs/pe_fin_replan_smoke5_live_v4"
    )
