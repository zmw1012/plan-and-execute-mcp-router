# pe_fin_replan_etom test fixtures

`mini_registry.json` is a tiny fake registry in the official ETOM payload shape
(`{"servers": [{"server_name", "categories", "tools": [{"tool_name",
"tool_description", "input_schema"}]}]}`). It is loaded through
`hk_equity_mcp.etom.convert.convert_registry` so unit tests exercise the same
conversion path as the real 2375-tool catalog, without needing the raw ETOM
checkout. It contains no gold routes and no task ids.
