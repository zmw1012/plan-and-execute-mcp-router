"""Load official ETOM registry and queries without rewriting them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from hk_equity_mcp.etom.source import raw_root, verify_source


@dataclass(frozen=True)
class EtomTask:
    task_id: str
    level: int
    query_id: int
    query: str
    ground_truth_tools: Any
    source_file: str
    server_names: Set[str] = field(default_factory=set)
    step_count: int = 0
    has_dependency: bool = False
    single_server_sequential: bool = False
    cross_server_dependency: bool = False
    infeasible: bool = False
    query_sha256: str = ""
    ground_truth_sha256: str = ""

    def public_lock_row(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "query_id": self.query_id,
            "source_file": self.source_file,
            "dataset_split": "cost_pilot",
            "exclude_from_final_etom_sample": True,
            "server_names": sorted(self.server_names),
            "step_count": self.step_count,
            "has_dependency": self.has_dependency,
            "single_server_sequential": self.single_server_sequential,
            "cross_server_dependency": self.cross_server_dependency,
            "infeasible": self.infeasible,
            "query_sha256": self.query_sha256,
            "ground_truth_sha256": self.ground_truth_sha256,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_tool_dicts(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        if "tool_name" in node or "server_name" in node or "server" in node:
            yield node
            return
        for value in node.values():
            yield from iter_tool_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_tool_dicts(item)


def _step_count(ground_truth: Any) -> int:
    if not isinstance(ground_truth, list):
        return 0
    return len(ground_truth)


def task_from_official(level: int, row: Dict[str, Any], source_file: str) -> EtomTask:
    query_id = int(row["query_id"])
    query = row["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Official L{level} query_id={query_id} has an empty query.")
    ground_truth = row.get("ground_truth_tools")
    tools = list(iter_tool_dicts(ground_truth))
    servers = {
        str(tool.get("server_name") or tool.get("server") or "").strip()
        for tool in tools
    }
    servers.discard("")
    has_dependency = any(bool(tool.get("dependencies")) for tool in tools)
    steps = _step_count(ground_truth)
    return EtomTask(
        task_id=f"L{level}-q{query_id}",
        level=level,
        query_id=query_id,
        query=query,
        ground_truth_tools=ground_truth,
        source_file=source_file,
        server_names=servers,
        step_count=steps,
        has_dependency=has_dependency,
        single_server_sequential=(
            level == 3 and len(servers) == 1 and has_dependency and steps >= 2
        ),
        cross_server_dependency=(
            level == 4 and len(servers) >= 2 and has_dependency
        ),
        infeasible=(level == 5 and not tools),
        query_sha256=_sha256_text(query),
        ground_truth_sha256=_sha256_text(_canonical(ground_truth)),
    )


def load_official_tasks(root: Path, *, verify: bool = True) -> List[EtomTask]:
    if verify:
        verify_source(root)
    checkout = raw_root(root)
    tasks: List[EtomTask] = []
    for level in range(1, 6):
        rel = f"data/queries/level_{level}.json"
        rows = json.loads((checkout / rel).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{rel} is not a JSON list.")
        for row in rows:
            tasks.append(task_from_official(level, row, rel))
    return tasks


def load_official_registry(root: Path, *, verify: bool = True) -> Dict[str, Any]:
    if verify:
        verify_source(root)
    path = raw_root(root) / "data" / "mcp_registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    tools = sum(len(server.get("tools") or []) for server in payload.get("servers") or [])
    if tools != 2375:
        raise ValueError(
            f"Pinned ETOM registry has {tools} tools; this pilot expects 2375."
        )
    return payload
