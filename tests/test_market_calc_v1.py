"""Unit tests for frozen Market calculation definitions."""

from __future__ import annotations

import math

import pytest

from hk_equity_mcp.real_mcp.market_calc import (
    annualized_volatility,
    daily_returns,
    maximum_drawdown,
    moving_average,
    percentage_return,
    round_half_up,
    sample_std,
    simple_return,
)


def test_simple_and_percentage_return():
    assert simple_return(100.0, 110.0) == pytest.approx(0.1)
    assert percentage_return(100.0, 110.0) == pytest.approx(10.0)


def test_daily_returns_and_vol():
    closes = [100.0, 101.0, 99.0, 102.0]
    rets = daily_returns(closes)
    assert len(rets) == 3
    assert rets[0] == pytest.approx(0.01)
    vol = annualized_volatility(closes)
    expected = sample_std(rets, ddof=1) * math.sqrt(252)
    assert vol == pytest.approx(expected)


def test_maximum_drawdown():
    closes = [100.0, 120.0, 90.0, 95.0]
    # peak 120 then 90 → -0.25
    assert maximum_drawdown(closes) == pytest.approx(-0.25)


def test_moving_average_includes_target_session():
    closes = [10.0, 20.0, 30.0, 40.0]
    assert moving_average(closes, 3) == pytest.approx(30.0)


def test_round_half_up():
    assert round_half_up(1.235, 2) == 1.24
    assert round_half_up(-1.235, 2) == -1.24
