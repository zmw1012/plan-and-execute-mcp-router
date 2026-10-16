from hk_equity_mcp.common.schema import validate_response
from hk_equity_mcp.mcp_servers.financial import FinancialServer
from hk_equity_mcp.mcp_servers.financial.data_store import get_store, reload_store


def setup_module():
    reload_store()


def test_ticker_normalization_0700_and_0700_hk():
    server = FinancialServer()
    a = server.get_financial_metric("0700", "revenue", period="FY2025", report_type="annual")
    b = server.get_financial_metric("0700.HK", "revenue", period="FY2025", report_type="annual")
    assert a["success"] and b["success"]
    assert a["ticker"] == b["ticker"] == "0700.HK"
    assert a["data"]["value"] == b["data"]["value"]


def test_default_latest_annual():
    server = FinancialServer()
    result = server.get_financial_metric("0700.HK", "revenue")
    assert validate_response(result)
    assert result["success"] is True
    assert result["data"]["report_type"] == "annual"
    assert result["data"]["document_type"] == "annual_report"
    assert result["data"]["fiscal_year_label"] == "FY2025"
    assert result["data"]["period_end"] == "2025-12-31"
    assert result["data"]["value"] == 751766000000
    assert result["data"]["currency"] == "CNY"
    assert result["data"]["unit"] == "CNY"
    assert result["data"]["canonical_metric"] == "revenues"
    assert result["data"]["audit_status"] == "audited"


def test_explicit_latest_interim():
    server = FinancialServer()
    result = server.get_financial_metric(
        "0700.HK", "revenue", report_type="interim"
    )
    assert result["success"] is True
    assert result["data"]["report_type"] == "interim"
    assert result["data"]["document_type"] == "interim_report"
    assert result["data"]["fiscal_year_label"] == "1H2026"
    assert result["data"]["period_end"] == "2026-06-30"
    assert result["data"]["value"] == 401243000000
    assert result["data"]["audit_status"] == "reviewed"


def test_fy2025_not_confused_with_1h2026():
    server = FinancialServer()
    annual = server.get_financial_metric(
        "0700.HK", "revenue", period="FY2025", report_type="annual"
    )
    interim = server.get_financial_metric(
        "0700.HK", "revenue", period="1H2026", report_type="interim"
    )
    assert annual["data"]["value"] != interim["data"]["value"]
    assert annual["data"]["period_end"] == "2025-12-31"
    assert interim["data"]["period_end"] == "2026-06-30"


def test_alibaba_fy2026_period_end_and_audited_annual_report():
    server = FinancialServer()
    result = server.get_financial_metric(
        "9988.HK", "revenue", period="FY2026", report_type="annual"
    )
    assert result["success"] is True
    assert result["data"]["period_end"] == "2026-03-31"
    assert result["data"]["fiscal_year_label"] == "FY2026"
    assert result["data"]["document_type"] == "annual_report"
    assert result["data"]["audit_status"] == "audited"
    assert result["data"]["value"] == 1023670000000
    assert result["data"]["canonical_metric"] == "revenue"


def test_unknown_ticker():
    server = FinancialServer()
    result = server.get_financial_metric("9999.HK", "revenue")
    assert result["success"] is False
    assert "Unknown ticker" in result["error"]


def test_unknown_metric():
    server = FinancialServer()
    result = server.get_financial_metric(
        "0700.HK", "ebitda_magic", period="FY2025", report_type="annual"
    )
    assert result["success"] is False
    assert "Unknown metric" in result["error"]


def test_period_mismatch():
    server = FinancialServer()
    result = server.get_financial_metric(
        "0700.HK", "revenue", period="FY2099", report_type="annual"
    )
    assert result["success"] is False
    assert "Period mismatch" in result["error"]


def test_annual_interim_report_type_mismatch():
    server = FinancialServer()
    result = server.get_financial_metric(
        "9988.HK", "revenue", period="FY2026", report_type="interim"
    )
    assert result["success"] is False
    assert "Report type mismatch" in result["error"] or "Period mismatch" in result["error"]


def test_reported_vs_adjusted_and_canonical_metric():
    server = FinancialServer()
    reported = server.get_financial_metric(
        "0700.HK",
        "net_profit",
        period="FY2025",
        report_type="annual",
        metric_type="reported",
    )
    adjusted = server.get_financial_metric(
        "0700.HK",
        "net_profit",
        period="FY2025",
        report_type="annual",
        metric_type="adjusted",
    )
    assert reported["success"] and adjusted["success"]
    assert reported["data"]["value"] == 224842000000
    assert adjusted["data"]["value"] == 259626000000
    assert reported["data"]["canonical_metric"] == "profit_attributable_to_equity_holders"
    assert adjusted["data"]["canonical_metric"] == "non_ifrs_profit_attributable_to_equity_holders"
    assert adjusted["data"]["adjustment_basis"]


def test_hsbc_adjustment_basis_is_specific():
    server = FinancialServer()
    result = server.get_financial_metric(
        "0005.HK",
        "revenue",
        period="FY2025",
        report_type="annual",
        metric_type="adjusted",
    )
    assert result["success"] is True
    assert result["data"]["canonical_metric"] == (
        "constant_currency_revenue_excluding_notable_items"
    )
    assert "constant-currency" in result["data"]["adjustment_basis"]
    assert "notable items" in result["data"]["adjustment_basis"]


def test_eps_basis_returned():
    server = FinancialServer()
    result = server.get_financial_metric(
        "0700.HK", "eps", period="FY2025", report_type="annual", metric_type="reported"
    )
    assert result["data"]["eps_basis"] == "basic"


def test_currency_unit_and_evidence_metadata():
    server = FinancialServer()
    result = server.get_financial_metric(
        "0005.HK", "revenue", period="FY2025", report_type="annual"
    )
    assert result["data"]["currency"] == "USD"
    assert result["data"]["unit"] == "USD"
    assert result["data"]["source_url"]
    assert result["data"]["source_document"]
    assert result["data"]["source_page_pdf"] == 15
    assert result["evidence"]
    assert result["data"]["verification_status"] == "verified"
    assert result["data"]["canonical_metric"] == "net_operating_income_before_ecl_change"


def test_filing_summary_validates_period():
    server = FinancialServer()
    ok = server.get_filing_summary("0005.HK", period="FY2025", report_type="annual")
    bad = server.get_filing_summary("0005.HK", period="FY2099", report_type="annual")
    assert ok["success"] is True
    assert "68274000000" in ok["data"]["summary"]
    assert ok["data"]["period"]["period_end"] == "2025-12-31"
    assert ok["data"]["source"]["document_type"] == "annual_report"
    assert ok["data"]["key_reported_metrics"]
    assert bad["success"] is False
    assert "Period mismatch" in bad["error"]


def test_ambiguous_calendar_year_period():
    server = FinancialServer()
    result = server.get_financial_metric("0700.HK", "revenue", period="2026")
    assert result["success"] is False
    assert "Ambiguous period" in result["error"]


def test_store_allows_future_alibaba_interim_slot():
    company = get_store().get_company("9988.HK")
    assert company is not None
    assert isinstance(company["reports"], list)
    assert all(r["report_type"] in {"annual", "interim"} for r in company["reports"])


def test_snapshot_metadata_not_period_end():
    store = get_store()
    assert store.snapshot_created_date == "2026-10-16"
    assert store.latest_source_publication_date == "2026-08-12"
    assert store.latest_source_publication_date != "2026-06-30"


def test_verified_metrics_have_source_url_and_page_pdf():
    store = get_store()
    for company in store.companies.values():
        for report in company["reports"]:
            assert report.get("document_type") in {
                "annual_report",
                "interim_report",
                "annual_results_announcement",
                "interim_results_announcement",
            }
            for metric in report["metrics"]:
                if metric.get("verification_status") != "verified":
                    continue
                assert report.get("source_url")
                page = metric.get("source_page_pdf")
                assert page is not None
                assert isinstance(page, int)
                assert page >= 1
                assert metric.get("evidence_text")
                assert metric.get("canonical_metric")
                assert metric.get("metric_definition")
