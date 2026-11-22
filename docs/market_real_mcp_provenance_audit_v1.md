# Market Real-MCP Provenance Audit v1

> v1.1 taxonomy/schema correction **reuses these provenance hashes** because raw MCP invocations and numerical gold are unchanged. No full live re-audit was run for v1.1.

- Generated (UTC): 2026-11-04T05:00:08.389007+00:00
- Catalog: `/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/configs/operations/market_real_mcp_v1.yaml`
- Cutoff: 2026-08-12
- Method: direct Real-MCP invokes only (no Agent/LLM/embedding).
- Mock FinancialServer: **not used**.

Records: 35

## mkt_dev_001

```json
{
  "question_id": "mkt_dev_001",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00700",
      "period": "daily",
      "start_date": "20260710",
      "end_date": "20260731",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-07-31",
  "effective_session_date": "2026-07-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 475.2,
  "unit": "HKD",
  "calculation_inputs": {
    "close": 475.2
  },
  "calculation_formula": "unadjusted close field",
  "rounded_gold": 475.2,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "c05fda11a53df03cd5b52b533247a235a91717417efaae8ce92945a543b6f37d",
  "audit_timestamp": "2026-11-04T04:59:51.315185+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_002

```json
{
  "question_id": "mkt_dev_002",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00005",
      "period": "daily",
      "start_date": "20260609",
      "end_date": "20260630",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-06-30",
  "effective_session_date": "2026-06-30",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 13123569.0,
  "unit": "shares",
  "calculation_inputs": {
    "volume": 13123569.0
  },
  "calculation_formula": "volume field",
  "rounded_gold": 13123569.0,
  "tolerance": {
    "rel": 0.001
  },
  "raw_response_hash": "65a4adbcf04e69c27346a2ef4ee027bea13874f3fdfa6615acee5f102a4289a7",
  "audit_timestamp": "2026-11-04T04:59:55.404448+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_003

```json
{
  "question_id": "mkt_dev_003",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "09988",
      "period": "daily",
      "start_date": "20260424",
      "end_date": "20260515",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-05-15",
  "effective_session_date": "2026-05-15",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": {
    "open": 138.0,
    "high": 138.0,
    "low": 131.1,
    "close": 132.3
  },
  "unit": "HKD",
  "calculation_inputs": {
    "date": "2026-05-15",
    "open": 138.0,
    "high": 138.0,
    "low": 131.1,
    "close": 132.3,
    "volume": 90144879.0,
    "turnover": 12041180928.0,
    "raw": {
      "日期": "2026-05-15T00:00:00.000",
      "开盘": 138.0,
      "收盘": 132.3,
      "最高": 138.0,
      "最低": 131.1,
      "成交量": 90144879,
      "成交额": 12041180928.0,
      "振幅": 5.0,
      "涨跌幅": -4.06,
      "涨跌额": -5.6,
      "换手率": 0.47
    }
  },
  "calculation_formula": "OHLC fields",
  "rounded_gold": {
    "open": 138.0,
    "high": 138.0,
    "low": 131.1,
    "close": 132.3
  },
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "03b8c360bb7673947c5d3ecd731d31265b7a4592a3d3830be7108f8bfc4270de",
  "audit_timestamp": "2026-11-04T04:59:56.776230+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_004

```json
{
  "question_id": "mkt_dev_004",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "sina_via_akshare",
  "source_operation": "aktools.stock_hk_index_daily_sina",
  "raw_invocation_arguments": {
    "symbol": "HSI",
    "offset": -500,
    "max_rows": 500,
    "target_date": "2026-03-18"
  },
  "requested_date_or_range": "2026-03-18",
  "effective_session_date": "2026-03-18",
  "previous_valid_day_triggered": false,
  "adjustment_mode": null,
  "value": 26025.41992,
  "unit": "index_points",
  "calculation_inputs": {
    "date": "2026-03-18",
    "open": 25923.03,
    "high": 26117.95,
    "low": 25791.82,
    "close": 26025.41992,
    "volume": 15086716097.0,
    "turnover": 240373872827.0,
    "raw": {
      "date": "2026-03-18T00:00:00.000",
      "open": 25923.03,
      "high": 26117.95,
      "low": 25791.82,
      "close": 26025.41992,
      "volume": 15086716097,
      "amount": 240373872827
    }
  },
  "calculation_formula": "HSI close after cutoff post-filter",
  "rounded_gold": 26025.42,
  "tolerance": {
    "abs": 0.5
  },
  "raw_response_hash": "44a8ad0b736af511a86c5cbdb08a8bfabe9e62c236b90ca45e58095c04a28600",
  "audit_timestamp": "2026-11-04T04:59:57.047052+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_005

```json
{
  "question_id": "mkt_dev_005",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00700",
      "period": "daily",
      "start_date": "20260719",
      "end_date": "20260809",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-08-09",
  "effective_session_date": "2026-08-07",
  "previous_valid_day_triggered": true,
  "adjustment_mode": "",
  "value": 478.8,
  "unit": "HKD",
  "calculation_inputs": {
    "date": "2026-08-07",
    "open": 479.0,
    "high": 483.2,
    "low": 475.4,
    "close": 478.8,
    "volume": 16319939.0,
    "turnover": 7803757312.0,
    "raw": {
      "日期": "2026-08-07T00:00:00.000",
      "开盘": 479.0,
      "收盘": 478.8,
      "最高": 483.2,
      "最低": 475.4,
      "成交量": 16319939,
      "成交额": 7803757312.0,
      "振幅": 1.63,
      "涨跌幅": -0.08,
      "涨跌额": -0.4,
      "换手率": 0.18
    }
  },
  "calculation_formula": "previous_valid_day max(session<=requested)",
  "rounded_gold": 478.8,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "8820a85e9c3aef3e27214b12939ca3461c94e3433c0f6ae65353a060eaca79c2",
  "audit_timestamp": "2026-11-04T04:59:57.535019+00:00",
  "notes": "Sunday request → prior session",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_006

```json
{
  "question_id": "mkt_dev_006",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00005",
      "period": "daily",
      "start_date": "20260624",
      "end_date": "20260715",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-07-15",
  "effective_session_date": "2026-07-15",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 2372985456.0,
  "unit": "HKD",
  "calculation_inputs": {
    "turnover": 2372985456.0
  },
  "calculation_formula": "turnover/成交额 field",
  "rounded_gold": 2372985456.0,
  "tolerance": {
    "rel": 0.001
  },
  "raw_response_hash": "31c8898ca53590971eb5d31900573d0d014a614946766c36fd0cda91ca1c33cd",
  "audit_timestamp": "2026-11-04T04:59:58.029262+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_007

```json
{
  "question_id": "mkt_dev_007",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "unadj": {
      "name": "stock_hk_hist",
      "params": {
        "symbol": "09988",
        "period": "daily",
        "start_date": "20260409",
        "end_date": "20260430",
        "adjust": ""
      },
      "offset": 0,
      "max_rows": 1000
    },
    "qfq": {
      "name": "stock_hk_hist",
      "params": {
        "symbol": "09988",
        "period": "daily",
        "start_date": "20260409",
        "end_date": "20260430",
        "adjust": "qfq"
      },
      "offset": 0,
      "max_rows": 1000
    }
  },
  "requested_date_or_range": "2026-04-30",
  "effective_session_date": "2026-04-30",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "compare '' vs qfq",
  "value": {
    "unadjusted": 126.0,
    "qfq": 124.97
  },
  "unit": "HKD",
  "calculation_inputs": {
    "unadj_close": 126.0,
    "qfq_close": 124.972
  },
  "calculation_formula": "two explicit adjust modes; no auto-mix",
  "rounded_gold": {
    "unadjusted": 126.0,
    "qfq": 124.97
  },
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "4c7536560d9a14811f38ee55a5f1cb2be9f0c085596804b7f5dbf09845d681f5",
  "audit_timestamp": "2026-11-04T04:59:58.752780+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_008

```json
{
  "question_id": "mkt_dev_008",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00700",
    "start_date": "20260702",
    "end_date": "20260731",
    "adjust": ""
  },
  "requested_date_or_range": {
    "start": "2026-07-02",
    "end": "2026-07-31"
  },
  "effective_session_date": "2026-07-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 10.460251046025103,
  "unit": "percent",
  "calculation_inputs": {
    "close_start": 430.2,
    "close_end": 475.2,
    "dates": [
      "2026-07-02",
      "2026-07-31"
    ]
  },
  "calculation_formula": "100*(close_end/close_start-1)",
  "rounded_gold": 10.4603,
  "tolerance": {
    "abs": 0.01
  },
  "raw_response_hash": "0b7afd89c3e3c06594ab78ae794109422d1ea980debebdfe298000bb98c657e2",
  "audit_timestamp": "2026-11-04T04:59:59.162343+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_009

```json
{
  "question_id": "mkt_dev_009",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00005",
    "start_date": "20260520",
    "end_date": "20260630",
    "adjust": ""
  },
  "requested_date_or_range": "2026-06-30",
  "effective_session_date": "2026-06-30",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 145.42000000000002,
  "unit": "HKD",
  "calculation_inputs": {
    "closes": [
      147.2,
      146.9,
      146.4,
      141.8,
      142.3,
      142.1,
      135.3,
      138.5,
      142.8,
      145.8,
      146.0,
      148.0,
      149.0,
      147.9,
      148.7,
      148.2,
      148.3,
      147.7,
      147.6,
      147.9
    ],
    "dates": [
      "2026-06-02",
      "2026-06-03",
      "2026-06-04",
      "2026-06-05",
      "2026-06-08",
      "2026-06-09",
      "2026-06-10",
      "2026-06-11",
      "2026-06-12",
      "2026-06-15",
      "2026-06-16",
      "2026-06-17",
      "2026-06-18",
      "2026-06-22",
      "2026-06-23",
      "2026-06-24",
      "2026-06-25",
      "2026-06-26",
      "2026-06-29",
      "2026-06-30"
    ]
  },
  "calculation_formula": "mean of last 20 unadjusted closes including target session",
  "rounded_gold": 145.42,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "5dfe641d863ca3b65319021e611cfbc2519f19b980bc24d79195cfc01a27b356",
  "audit_timestamp": "2026-11-04T04:59:59.638378+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_010

```json
{
  "question_id": "mkt_dev_010",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "sina_via_akshare",
  "source_operation": "aktools.stock_hk_index_daily_sina",
  "raw_invocation_arguments": {
    "d0": {
      "symbol": "HSI",
      "offset": -500,
      "max_rows": 500,
      "target_date": "2026-08-11"
    },
    "d1": {
      "symbol": "HSI",
      "offset": -500,
      "max_rows": 500,
      "target_date": "2026-08-12"
    }
  },
  "requested_date_or_range": [
    "2026-08-11",
    "2026-08-12"
  ],
  "effective_session_date": "2026-08-12",
  "previous_valid_day_triggered": false,
  "adjustment_mode": null,
  "value": -0.00828947669670399,
  "unit": "decimal_return",
  "calculation_inputs": {
    "P_t-1": 25652.81836,
    "P_t": 25440.16992,
    "dates": [
      "2026-08-11",
      "2026-08-12"
    ]
  },
  "calculation_formula": "P_t/P_(t-1)-1",
  "rounded_gold": -0.008289,
  "tolerance": {
    "abs": 1e-05
  },
  "raw_response_hash": "f7fb84cae489f1df86403def7a655fe0eb62278fd9f2fa68aa8da3174dc3e3ee",
  "audit_timestamp": "2026-11-04T04:59:59.677583+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_011

```json
{
  "question_id": "mkt_dev_011",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00700",
    "start_date": "20260701",
    "end_date": "20260731",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-07-02",
    "2026-07-31"
  ],
  "effective_session_date": "2026-07-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 0.47608224362237883,
  "unit": "decimal_volatility",
  "calculation_inputs": {
    "closes": [
      430.2,
      431.2,
      452.0,
      461.2,
      478.8,
      469.6,
      460.2,
      457.6,
      456.2,
      474.0,
      484.0,
      461.6,
      477.8,
      474.0,
      440.6,
      445.2,
      434.6,
      443.0,
      447.2,
      466.4,
      471.8,
      475.2
    ],
    "dates": [
      "2026-07-02",
      "2026-07-03",
      "2026-07-06",
      "2026-07-07",
      "2026-07-08",
      "2026-07-09",
      "2026-07-10",
      "2026-07-13",
      "2026-07-14",
      "2026-07-15",
      "2026-07-16",
      "2026-07-17",
      "2026-07-20",
      "2026-07-21",
      "2026-07-22",
      "2026-07-23",
      "2026-07-24",
      "2026-07-27",
      "2026-07-28",
      "2026-07-29",
      "2026-07-30",
      "2026-07-31"
    ]
  },
  "calculation_formula": "sample_std(daily_simple_returns,ddof=1)*sqrt(252)",
  "rounded_gold": 0.476082,
  "tolerance": {
    "abs": 0.0001
  },
  "raw_response_hash": "fe11c52d6c531fdb558e91e9cf0025a03f75c68f4af9eb096485d6618eb752d1",
  "audit_timestamp": "2026-11-04T05:00:00.573955+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_012

```json
{
  "question_id": "mkt_dev_012",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "09988",
    "start_date": "20260602",
    "end_date": "20260731",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-06-02",
    "2026-07-31"
  ],
  "effective_session_date": "2026-07-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": -0.3162719633307869,
  "unit": "decimal_drawdown",
  "calculation_inputs": {
    "closes": [
      130.9,
      126.6,
      123.5,
      122.4,
      118.8,
      117.1,
      113.5,
      107.4,
      110.2,
      109.3,
      107.0,
      106.9,
      104.9,
      102.9,
      98.95,
      99.4,
      95.0,
      89.5,
      93.0,
      92.85,
      94.5,
      94.1,
      95.95,
      95.8,
      107.5,
      108.0,
      110.2,
      110.7,
      110.8,
      113.4,
      116.9,
      112.6,
      116.8,
      117.0,
      113.6,
      114.9,
      110.0,
      111.0,
      112.0,
      113.6,
      111.8,
      117.0
    ],
    "dates": [
      "2026-06-02",
      "2026-06-03",
      "2026-06-04",
      "2026-06-05",
      "2026-06-08",
      "2026-06-09",
      "2026-06-10",
      "2026-06-11",
      "2026-06-12",
      "2026-06-15",
      "2026-06-16",
      "2026-06-17",
      "2026-06-18",
      "2026-06-22",
      "2026-06-23",
      "2026-06-24",
      "2026-06-25",
      "2026-06-26",
      "2026-06-29",
      "2026-06-30",
      "2026-07-02",
      "2026-07-03",
      "2026-07-06",
      "2026-07-07",
      "2026-07-08",
      "2026-07-09",
      "2026-07-10",
      "2026-07-13",
      "2026-07-14",
      "2026-07-15",
      "2026-07-16",
      "2026-07-17",
      "2026-07-20",
      "2026-07-21",
      "2026-07-22",
      "2026-07-23",
      "2026-07-24",
      "2026-07-27",
      "2026-07-28",
      "2026-07-29",
      "2026-07-30",
      "2026-07-31"
    ]
  },
  "calculation_formula": "min(P_t/running_max(P_1..P_t)-1)",
  "rounded_gold": -0.316272,
  "tolerance": {
    "abs": 0.0001
  },
  "raw_response_hash": "a5398a19d82d9048370ff53bc313eba7ec83ee2e13c0a91b34490b55fd18d8be",
  "audit_timestamp": "2026-11-04T05:00:00.901016+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_013

```json
{
  "question_id": "mkt_dev_013",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney+sina",
  "source_operation": "aktools.stock_hk_hist+aktools.stock_hk_index_daily_sina",
  "raw_invocation_arguments": {
    "equity": {
      "symbol": "00700",
      "start_date": "20260504",
      "end_date": "20260529",
      "adjust": ""
    },
    "index": {
      "symbol": "HSI",
      "offset": -500,
      "max_rows": 500,
      "window": [
        "2026-05-04",
        "2026-05-29"
      ]
    },
    "common": [
      "2026-05-04",
      "2026-05-29"
    ]
  },
  "requested_date_or_range": [
    "2026-05-04",
    "2026-05-29"
  ],
  "effective_session_date": "2026-05-29",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": -0.061823602408432765,
  "unit": "decimal_excess_return",
  "calculation_inputs": {
    "equity": [
      473.0,
      427.2
    ],
    "hsi": [
      26095.88086,
      25182.39063
    ],
    "dates": [
      "2026-05-04",
      "2026-05-29"
    ]
  },
  "calculation_formula": "(P_e_end/P_e_start-1)-(P_i_end/P_i_start-1)",
  "rounded_gold": -0.061824,
  "tolerance": {
    "abs": 0.0001
  },
  "raw_response_hash": "42a5a353e03faed515c1585046554258b07d648c2dc8e7a6e0ea6a8bc98b59ee",
  "audit_timestamp": "2026-11-04T05:00:01.419777+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_014

```json
{
  "question_id": "mkt_dev_014",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": null,
  "provider": "n/a",
  "source_operation": "none",
  "raw_invocation_arguments": {},
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": null,
  "unit": "n/a",
  "calculation_inputs": {},
  "calculation_formula": "infeasible",
  "rounded_gold": null,
  "tolerance": null,
  "raw_response_hash": "",
  "audit_timestamp": "2026-11-04T05:00:01.419803+00:00",
  "notes": "Uses latest/current/today; forbidden as formal numeric gold under fixed cutoff.",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_dev_015

```json
{
  "question_id": "mkt_dev_015",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": null,
  "provider": "n/a",
  "source_operation": "none",
  "raw_invocation_arguments": {},
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": null,
  "unit": "n/a",
  "calculation_inputs": {},
  "calculation_formula": "infeasible",
  "rounded_gold": null,
  "tolerance": null,
  "raw_response_hash": "",
  "audit_timestamp": "2026-11-04T05:00:01.419816+00:00",
  "notes": "FTShare blocked; minute hist deferred (empty-window). Not in active frozen catalog.",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_001

```json
{
  "question_id": "mkt_ht_001",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00005",
      "period": "daily",
      "start_date": "20260206",
      "end_date": "20260227",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-02-27",
  "effective_session_date": "2026-02-27",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 147.3,
  "unit": "HKD",
  "calculation_inputs": {
    "date": "2026-02-27",
    "open": 146.8,
    "high": 147.8,
    "low": 145.8,
    "close": 147.3,
    "volume": 18075411.0,
    "turnover": 2656255984.0,
    "raw": {
      "日期": "2026-02-27T00:00:00.000",
      "开盘": 146.8,
      "收盘": 147.3,
      "最高": 147.8,
      "最低": 145.8,
      "成交量": 18075411,
      "成交额": 2656255984.0,
      "振幅": 1.38,
      "涨跌幅": 1.59,
      "涨跌额": 2.3,
      "换手率": 0.11
    }
  },
  "calculation_formula": "close",
  "rounded_gold": 147.3,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "1691532c536e43af0ff94ea45abad017cd5e45aa7eb9a67809ce5c195b8c7d77",
  "audit_timestamp": "2026-11-04T05:00:01.881879+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_002

```json
{
  "question_id": "mkt_ht_002",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "09988",
      "period": "daily",
      "start_date": "20251225",
      "end_date": "20260115",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-01-15",
  "effective_session_date": "2026-01-15",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 133061563.0,
  "unit": "shares",
  "calculation_inputs": {
    "volume": 133061563.0
  },
  "calculation_formula": "volume",
  "rounded_gold": 133061563.0,
  "tolerance": {
    "rel": 0.001
  },
  "raw_response_hash": "b6e500455d064f30ef3615e1d55cc41eee44fb120de5d8152e79ce4d285644c3",
  "audit_timestamp": "2026-11-04T05:00:02.150686+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_003

```json
{
  "question_id": "mkt_ht_003",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "sina_via_akshare",
  "source_operation": "aktools.stock_hk_index_daily_sina",
  "raw_invocation_arguments": {
    "symbol": "HSI",
    "offset": -500,
    "max_rows": 500,
    "target_date": "2026-07-15"
  },
  "requested_date_or_range": "2026-07-15",
  "effective_session_date": "2026-07-15",
  "previous_valid_day_triggered": false,
  "adjustment_mode": null,
  "value": 24681.10156,
  "unit": "index_points",
  "calculation_inputs": {
    "date": "2026-07-15",
    "open": 24551.34,
    "high": 24774.84,
    "low": 24449.53,
    "close": 24681.10156,
    "volume": 13491923494.0,
    "turnover": 304297167603.0,
    "raw": {
      "date": "2026-07-15T00:00:00.000",
      "open": 24551.34,
      "high": 24774.84,
      "low": 24449.53,
      "close": 24681.10156,
      "volume": 13491923494,
      "amount": 304297167603
    }
  },
  "calculation_formula": "close",
  "rounded_gold": 24681.1,
  "tolerance": {
    "abs": 0.5
  },
  "raw_response_hash": "935706d0b43a6b9e95204246826495fe2f765f7ea1dc0e8ae10a66604b21ee5e",
  "audit_timestamp": "2026-11-04T05:00:02.171322+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_004

```json
{
  "question_id": "mkt_ht_004",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00700",
      "period": "daily",
      "start_date": "20260312",
      "end_date": "20260402",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-04-02",
  "effective_session_date": "2026-04-02",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": {
    "open": 496.0,
    "high": 496.0,
    "low": 485.0,
    "close": 489.2
  },
  "unit": "HKD",
  "calculation_inputs": {
    "date": "2026-04-02",
    "open": 496.0,
    "high": 496.0,
    "low": 485.0,
    "close": 489.2,
    "volume": 17030828.0,
    "turnover": 8323962112.0,
    "raw": {
      "日期": "2026-04-02T00:00:00.000",
      "开盘": 496.0,
      "收盘": 489.2,
      "最高": 496.0,
      "最低": 485.0,
      "成交量": 17030828,
      "成交额": 8323962112.0,
      "振幅": 2.22,
      "涨跌幅": -1.49,
      "涨跌额": -7.4,
      "换手率": 0.19
    }
  },
  "calculation_formula": "ohlc",
  "rounded_gold": {
    "open": 496.0,
    "high": 496.0,
    "low": 485.0,
    "close": 489.2
  },
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "074a22953d1de13f6b1d34bb5a44c143af4de350f9a2ff249e3d1c7a47d203f8",
  "audit_timestamp": "2026-11-04T05:00:02.632108+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_005

```json
{
  "question_id": "mkt_ht_005",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "00005",
      "period": "daily",
      "start_date": "20260419",
      "end_date": "20260510",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-05-10",
  "effective_session_date": "2026-05-08",
  "previous_valid_day_triggered": true,
  "adjustment_mode": "",
  "value": 138.7,
  "unit": "HKD",
  "calculation_inputs": {
    "date": "2026-05-08",
    "open": 139.4,
    "high": 140.2,
    "low": 138.2,
    "close": 138.7,
    "volume": 15750818.0,
    "turnover": 2188456032.0,
    "raw": {
      "日期": "2026-05-08T00:00:00.000",
      "开盘": 139.4,
      "收盘": 138.7,
      "最高": 140.2,
      "最低": 138.2,
      "成交量": 15750818,
      "成交额": 2188456032.0,
      "振幅": 1.4,
      "涨跌幅": -3.07,
      "涨跌额": -4.4,
      "换手率": 0.09
    }
  },
  "calculation_formula": "previous_valid_day",
  "rounded_gold": 138.7,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "74954b20bb057d92449c3b74cd839d8795399527fb517751bb5ef82630980a1b",
  "audit_timestamp": "2026-11-04T05:00:03.365630+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_006

```json
{
  "question_id": "mkt_ht_006",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "u": {
      "name": "stock_hk_hist",
      "params": {
        "symbol": "00700",
        "period": "daily",
        "start_date": "20260310",
        "end_date": "20260331",
        "adjust": ""
      },
      "offset": 0,
      "max_rows": 1000
    },
    "hfq": {
      "name": "stock_hk_hist",
      "params": {
        "symbol": "00700",
        "period": "daily",
        "start_date": "20260310",
        "end_date": "20260331",
        "adjust": "hfq"
      },
      "offset": 0,
      "max_rows": 1000
    }
  },
  "requested_date_or_range": "2026-03-31",
  "effective_session_date": "2026-03-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "'' vs hfq",
  "value": {
    "unadjusted": 484.0,
    "hfq": 2683.31
  },
  "unit": "HKD",
  "calculation_inputs": {
    "u": 484.0,
    "hfq": 2683.313
  },
  "calculation_formula": "explicit dual adjust",
  "rounded_gold": {
    "unadjusted": 484.0,
    "hfq": 2683.31
  },
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "973318fa16e0fda39f5b0ff0025d56d44bf34521428a1bce6a9a6f5234d5e755",
  "audit_timestamp": "2026-11-04T05:00:04.171016+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_007

```json
{
  "question_id": "mkt_ht_007",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "name": "stock_hk_hist",
    "params": {
      "symbol": "09988",
      "period": "daily",
      "start_date": "20260120",
      "end_date": "20260210",
      "adjust": ""
    },
    "offset": 0,
    "max_rows": 1000
  },
  "requested_date_or_range": "2026-02-10",
  "effective_session_date": "2026-02-10",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 3.9,
  "unit": "HKD",
  "calculation_inputs": {
    "high": 162.6,
    "low": 158.7
  },
  "calculation_formula": "high-low",
  "rounded_gold": 3.9,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "9069f4458268316563a8d44d6edadac78888c227e3ac5fbbdf6f7fad78a553b6",
  "audit_timestamp": "2026-11-04T05:00:04.590682+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_008

```json
{
  "question_id": "mkt_ht_008",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_security_profile_em",
  "raw_invocation_arguments": {
    "symbol": "00700"
  },
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": {
    "success": true,
    "source_server": "aktools",
    "source_operation": "aktools.stock_hk_security_profile_em",
    "provider": "eastmoney_via_akshare",
    "symbol": "00700",
    "requested_date_or_range": null,
    "effective_session_date": null,
    "timezone": "Asia/Hong_Kong",
    "currency_or_unit": null,
    "currency_provenance": null,
    "adjustment_mode": null,
    "data": [],
    "provenance": {
      "exposed_mcp_tool": "get_data",
      "underlying_aktools_interface": "stock_hk_security_profile_em",
      "raw_invocation_arguments": {
        "name": "stock_hk_security_profile_em",
        "params": {
          "symbol": "00700"
        },
        "offset": 0,
        "max_rows": 20
      },
      "row_count_after_filter": 0,
      "cutoff": "2026-08-12",
      "server_version": "0.1.4",
      "real_mcp": true,
      "mock_fallback": false,
      "tool_trade_date_hist_sina_is_hkex_official_calendar": false
    },
    "previous_valid_day_triggered": null,
    "error": null
  },
  "unit": "identity",
  "calculation_inputs": {
    "success": true,
    "source_server": "aktools",
    "source_operation": "aktools.stock_hk_security_profile_em",
    "provider": "eastmoney_via_akshare",
    "symbol": "00700",
    "requested_date_or_range": null,
    "effective_session_date": null,
    "timezone": "Asia/Hong_Kong",
    "currency_or_unit": null,
    "currency_provenance": null,
    "adjustment_mode": null,
    "data": [],
    "provenance": {
      "exposed_mcp_tool": "get_data",
      "underlying_aktools_interface": "stock_hk_security_profile_em",
      "raw_invocation_arguments": {
        "name": "stock_hk_security_profile_em",
        "params": {
          "symbol": "00700"
        },
        "offset": 0,
        "max_rows": 20
      },
      "row_count_after_filter": 0,
      "cutoff": "2026-08-12",
      "server_version": "0.1.4",
      "real_mcp": true,
      "mock_fallback": false,
      "tool_trade_date_hist_sina_is_hkex_official_calendar": false
    },
    "previous_valid_day_triggered": null,
    "error": null
  },
  "calculation_formula": "profile identity",
  "rounded_gold": {
    "is_tencent": true,
    "hsi_is_index": true
  },
  "tolerance": {
    "abs": 0
  },
  "raw_response_hash": "8ba9a917c73f7a443f510d7a70a9afb45e20cd1d73c6271d7543f2c0fc0d8d33",
  "audit_timestamp": "2026-11-04T05:00:04.711389+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_009

```json
{
  "question_id": "mkt_ht_009",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00005",
    "start_date": "20260504",
    "end_date": "20260529",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-05-04",
    "2026-05-29"
  ],
  "effective_session_date": "2026-05-29",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 1.1854951185495066,
  "unit": "percent",
  "calculation_inputs": {
    "start": 143.4,
    "end": 145.1
  },
  "calculation_formula": "100*(close_end/close_start-1)",
  "rounded_gold": 1.1855,
  "tolerance": {
    "abs": 0.01
  },
  "raw_response_hash": "e0de70da0fbd893b34739bf53ad5a0805d5be4ed4d3358051f014c39bf36dba0",
  "audit_timestamp": "2026-11-04T05:00:05.150479+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_010

```json
{
  "question_id": "mkt_ht_010",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "09988",
    "start_date": "20260301",
    "end_date": "20260331",
    "adjust": ""
  },
  "requested_date_or_range": "2026-03-31",
  "effective_session_date": "2026-03-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 125.03,
  "unit": "HKD",
  "calculation_inputs": {
    "closes": [
      137.7,
      132.0,
      123.7,
      119.7,
      123.2,
      128.9,
      123.0,
      122.6,
      120.5,
      119.0
    ]
  },
  "calculation_formula": "MA10",
  "rounded_gold": 125.03,
  "tolerance": {
    "abs": 0.05
  },
  "raw_response_hash": "eab92283a8907112ba7b873d6336979c548e9f2611415c96dfc8ac7271db4e1a",
  "audit_timestamp": "2026-11-04T05:00:05.560833+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_011

```json
{
  "question_id": "mkt_ht_011",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "sina_via_akshare",
  "source_operation": "aktools.stock_hk_index_daily_sina",
  "raw_invocation_arguments": {
    "h0": {
      "symbol": "HSI",
      "offset": -500,
      "max_rows": 500,
      "target_date": "2026-06-03"
    },
    "h1": {
      "symbol": "HSI",
      "offset": -500,
      "max_rows": 500,
      "target_date": "2026-06-30"
    }
  },
  "requested_date_or_range": [
    "2026-06-03",
    "2026-06-30"
  ],
  "effective_session_date": "2026-06-30",
  "previous_valid_day_triggered": false,
  "adjustment_mode": null,
  "value": -0.10736804362447794,
  "unit": "decimal_return",
  "calculation_inputs": {
    "start": 25633.20898,
    "end": 22881.02148
  },
  "calculation_formula": "close_end/close_start-1",
  "rounded_gold": -0.107368,
  "tolerance": {
    "abs": 1e-05
  },
  "raw_response_hash": "f032beaca027a6790033b12384721652ac57604099e17c50324e8281fd83934c",
  "audit_timestamp": "2026-11-04T05:00:05.603822+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_012

```json
{
  "question_id": "mkt_ht_012",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00700",
    "start_date": "20260203",
    "end_date": "20260227",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-02-03",
    "2026-02-27"
  ],
  "effective_session_date": "2026-02-27",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 301579761968.0,
  "unit": "HKD",
  "calculation_inputs": {
    "turnovers": [
      37562016512.0,
      34030535936.0,
      43460897280.0,
      19472346880.0,
      14680204032.0,
      16193933312.0,
      12991880960.0,
      21258224384.0,
      14233832960.0,
      3838968112.0,
      11171863808.0,
      11883268352.0,
      18828896768.0,
      11981117696.0,
      13216109056.0,
      16775665920.0
    ]
  },
  "calculation_formula": "sum(turnover)",
  "rounded_gold": 301579761968.0,
  "tolerance": {
    "rel": 0.001
  },
  "raw_response_hash": "330106cf3215cc7252bf1a93ff8411303b029149744658c24bb7ba9bca0052a2",
  "audit_timestamp": "2026-11-04T05:00:06.057047+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_013

```json
{
  "question_id": "mkt_ht_013",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00005",
    "start_date": "20260401",
    "end_date": "20260430",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-04-01",
    "2026-04-30"
  ],
  "effective_session_date": "2026-04-30",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 0.2757502621734871,
  "unit": "decimal_volatility",
  "calculation_inputs": {
    "closes": [
      130.3,
      130.0,
      138.6,
      138.9,
      139.7,
      138.8,
      141.0,
      141.4,
      142.3,
      141.5,
      142.2,
      143.7,
      144.0,
      141.1,
      140.0,
      140.2,
      140.6,
      141.0,
      140.2
    ]
  },
  "calculation_formula": "ann_vol",
  "rounded_gold": 0.27575,
  "tolerance": {
    "abs": 0.0001
  },
  "raw_response_hash": "1d113620f2372926c1a84978cfc0a43ad455587f8b9979b9e1dad872d486f1bf",
  "audit_timestamp": "2026-11-04T05:00:06.581902+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_014

```json
{
  "question_id": "mkt_ht_014",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00700",
    "start_date": "20260105",
    "end_date": "20260227",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-01-05",
    "2026-02-27"
  ],
  "effective_session_date": "2026-02-27",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": -0.1911532385466035,
  "unit": "decimal_drawdown",
  "calculation_inputs": {
    "closes": [
      624.5,
      632.5,
      624.5,
      616.0,
      611.0,
      623.0,
      627.5,
      633.0,
      622.0,
      617.5,
      610.0,
      601.0,
      602.5,
      597.5,
      595.0,
      599.5,
      607.0,
      621.0,
      622.0,
      606.0,
      598.5,
      581.0,
      558.0,
      558.5,
      547.5,
      560.0,
      551.0,
      548.0,
      535.5,
      532.0,
      533.0,
      522.0,
      538.0,
      520.0,
      522.5,
      512.0,
      518.0
    ]
  },
  "calculation_formula": "MDD",
  "rounded_gold": -0.191153,
  "tolerance": {
    "abs": 0.0001
  },
  "raw_response_hash": "f4f2164858754f77527b93862fcf06741d777742a3a0baea0f60997da7562c40",
  "audit_timestamp": "2026-11-04T05:00:07.208790+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_015

```json
{
  "question_id": "mkt_ht_015",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney+sina",
  "source_operation": "aktools.stock_hk_hist+aktools.stock_hk_index_daily_sina",
  "raw_invocation_arguments": {
    "eq": {
      "symbol": "09988",
      "start_date": "20260702",
      "end_date": "20260731",
      "adjust": ""
    },
    "ix": {
      "symbol": "HSI",
      "offset": -500,
      "max_rows": 500,
      "window": [
        "2026-07-02",
        "2026-07-31"
      ]
    }
  },
  "requested_date_or_range": [
    "2026-07-02",
    "2026-07-31"
  ],
  "effective_session_date": "2026-07-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": 0.11537155104753571,
  "unit": "decimal_excess_return",
  "calculation_inputs": {
    "eq": [
      94.5,
      117.0
    ],
    "ix": [
      23055.03125,
      25884.42969
    ]
  },
  "calculation_formula": "simple_return(equity)-simple_return(HSI)",
  "rounded_gold": 0.115372,
  "tolerance": {
    "abs": 0.0001
  },
  "raw_response_hash": "143ffe222e2b5314987a0df70d07f885eaa209a1e73a2cb37bc69ec021e4bc4d",
  "audit_timestamp": "2026-11-04T05:00:07.681412+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_016

```json
{
  "question_id": "mkt_ht_016",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": "0.1.4",
  "provider": "eastmoney_via_akshare",
  "source_operation": "aktools.stock_hk_hist",
  "raw_invocation_arguments": {
    "symbol": "00005",
    "start_date": "20260302",
    "end_date": "20260331",
    "adjust": ""
  },
  "requested_date_or_range": [
    "2026-03-02",
    "2026-03-31"
  ],
  "effective_session_date": "2026-03-31",
  "previous_valid_day_triggered": false,
  "adjustment_mode": "",
  "value": {
    "ma5": 124.8,
    "simple_return": -0.099499
  },
  "unit": "mixed",
  "calculation_inputs": {
    "closes": [
      139.7,
      135.8,
      130.8,
      133.9,
      135.2,
      129.9,
      135.9,
      133.0,
      128.9,
      122.5,
      124.3,
      124.8,
      127.9,
      124.0,
      124.5,
      118.9,
      123.6,
      125.0,
      124.8,
      124.8,
      123.6,
      125.8
    ]
  },
  "calculation_formula": "MA5+simple_return",
  "rounded_gold": {
    "ma5": 124.8,
    "simple_return": -0.099499
  },
  "tolerance": {
    "ma5_abs": 0.05,
    "simple_return_abs": 0.0001
  },
  "raw_response_hash": "05c2ca7cfdc131fdc9a8d0b786abdccfe5ec41acdfc7232ee6a5dfbce049bed8",
  "audit_timestamp": "2026-11-04T05:00:08.228497+00:00",
  "notes": "",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_017

```json
{
  "question_id": "mkt_ht_017",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": null,
  "provider": "n/a",
  "source_operation": "none",
  "raw_invocation_arguments": {},
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": null,
  "unit": "n/a",
  "calculation_inputs": {},
  "calculation_formula": "infeasible",
  "rounded_gold": null,
  "tolerance": null,
  "raw_response_hash": "",
  "audit_timestamp": "2026-11-04T05:00:08.228531+00:00",
  "notes": "current/right now forbidden as formal gold under fixed cutoff.",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_018

```json
{
  "question_id": "mkt_ht_018",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": null,
  "provider": "n/a",
  "source_operation": "none",
  "raw_invocation_arguments": {},
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": null,
  "unit": "n/a",
  "calculation_inputs": {},
  "calculation_formula": "infeasible",
  "rounded_gold": null,
  "tolerance": null,
  "raw_response_hash": "",
  "audit_timestamp": "2026-11-04T05:00:08.228548+00:00",
  "notes": "tool_trade_date_hist_sina is explicitly NOT HKEX official calendar.",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_019

```json
{
  "question_id": "mkt_ht_019",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": null,
  "provider": "n/a",
  "source_operation": "none",
  "raw_invocation_arguments": {},
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": null,
  "unit": "n/a",
  "calculation_inputs": {},
  "calculation_formula": "infeasible",
  "rounded_gold": null,
  "tolerance": null,
  "raw_response_hash": "",
  "audit_timestamp": "2026-11-04T05:00:08.228561+00:00",
  "notes": "Requested date after frozen cutoff; post-cutoff gold forbidden.",
  "mock_used": false,
  "llm_used": false
}
```

## mkt_ht_020

```json
{
  "question_id": "mkt_ht_020",
  "mcp_server": "aktools",
  "package_command": "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent/.venv-aktools-discovery/bin/aktools-mcp",
  "server_version": null,
  "provider": "n/a",
  "source_operation": "none",
  "raw_invocation_arguments": {},
  "requested_date_or_range": null,
  "effective_session_date": null,
  "previous_valid_day_triggered": null,
  "adjustment_mode": null,
  "value": null,
  "unit": "n/a",
  "calculation_inputs": {},
  "calculation_formula": "infeasible",
  "rounded_gold": null,
  "tolerance": null,
  "raw_response_hash": "",
  "audit_timestamp": "2026-11-04T05:00:08.228571+00:00",
  "notes": "FTShare operations blocked; not in active registry; not gold.",
  "mock_used": false,
  "llm_used": false
}
```
