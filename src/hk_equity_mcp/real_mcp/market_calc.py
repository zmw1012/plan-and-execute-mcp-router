"""Frozen Market Real-MCP calculation definitions (deterministic, no LLM).

Formulas (Market v1 freeze):
- Simple return: close_end / close_start - 1
- Percentage return: 100 * (close_end / close_start - 1)
- Daily return: P_t / P_(t-1) - 1
- Annualized volatility: sample_std(daily simple returns, ddof=1) * sqrt(252)
- Maximum drawdown: min(P_t / running_max(P_1...P_t) - 1)
- Moving average N: arithmetic mean of the most recent N valid unadjusted closes
  including the target session.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


TRADING_DAYS_PER_YEAR = 252


def simple_return(close_start: float, close_end: float) -> float:
    if close_start == 0:
        raise ValueError("close_start must be non-zero")
    return close_end / close_start - 1.0


def percentage_return(close_start: float, close_end: float) -> float:
    return 100.0 * simple_return(close_start, close_end)


def daily_returns(closes: Sequence[float]) -> List[float]:
    if len(closes) < 2:
        raise ValueError("need at least 2 closes for daily returns")
    out: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0:
            raise ValueError("zero close in series")
        out.append(closes[i] / prev - 1.0)
    return out


def sample_std(values: Sequence[float], *, ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        raise ValueError(f"need more than {ddof} observations for sample_std")
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - ddof)
    return math.sqrt(var)


def annualized_volatility(closes: Sequence[float]) -> float:
    rets = daily_returns(closes)
    return sample_std(rets, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)


def maximum_drawdown(closes: Sequence[float]) -> float:
    if not closes:
        raise ValueError("empty closes")
    peak = closes[0]
    mdd = 0.0
    for px in closes:
        if px > peak:
            peak = px
        if peak == 0:
            raise ValueError("zero peak in series")
        dd = px / peak - 1.0
        if dd < mdd:
            mdd = dd
    return mdd


def moving_average(closes: Sequence[float], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(closes) < n:
        raise ValueError(f"need at least {n} closes for MA")
    window = closes[-n:]
    return sum(window) / float(n)


def round_half_up(value: float, decimals: int) -> float:
    """Deterministic round half away from zero via decimal-style scale."""
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    factor = 10**decimals
    scaled = value * factor
    if scaled >= 0:
        return math.floor(scaled + 0.5) / factor
    return math.ceil(scaled - 0.5) / factor


def as_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]
