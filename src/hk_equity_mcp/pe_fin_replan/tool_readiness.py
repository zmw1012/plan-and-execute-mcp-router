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
        "critical_live_audit is deferred: do not run in offline engineering turn"
    )


def run_market_sina_probe(*, root: Path, output_dir: Path, **_: Any) -> Dict[str, Any]:
    raise RuntimeError(
        "market_sina_probe is deferred: do not run in offline engineering turn "
        "(future live readiness only)"
    )


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
        f"  --output-dir outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe \\\n"
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
        f"outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe/readiness.json \\\n"
        f"  --env-file .env \\\n"
        f"  --embedding-env-file .env \\\n"
        f"  --embedding-cache-dir \"{cache}\" \\\n"
        f"  --output-dir outputs/pe_fin_replan_smoke5_live_v4"
    )
