#!/usr/bin/env python3
"""Write a derived ETOM operation registry. Never writes into the raw checkout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.etom.convert import convert_registry, derived_registry_document  # noqa: E402
from hk_equity_mcp.etom.loader import load_official_registry  # noqa: E402
from hk_equity_mcp.etom.source import file_sha256, raw_root  # noqa: E402


def main() -> None:
    registry_path = raw_root(ROOT) / "data" / "mcp_registry.json"
    before = file_sha256(registry_path)
    payload = load_official_registry(ROOT)
    specs, official = convert_registry(payload)
    after = file_sha256(registry_path)
    if before != after:
        raise SystemExit("Conversion modified the official registry. Aborting.")
    destination = ROOT / "data" / "etom" / "derived" / "operation_registry.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = derived_registry_document(specs, official)
    destination.write_text(
        json.dumps(
            {
                "operation_count": document["operation_count"],
                "conversion": document["conversion"],
                "fidelity": document["fidelity"],
                "pseudo_output_available": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"operations={len(specs)} raw_registry_unchanged=true")
    print(f"derived_summary={destination}")


if __name__ == "__main__":
    main()
