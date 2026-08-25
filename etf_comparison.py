"""Side-by-side comparison of up to three funds.

WHAT IS COMPARED AND WHAT IS NOT. The task asks for a grid of "ER, AUM,
PE, yield, returns, sharpe, tracking error". Six of those come off data
this build already has. Tracking error does not: it needs each fund's
stated benchmark index as a return series, and Yahoo returns neither the
benchmark mapping nor the index series — asset_class.MISSING_SOURCES has
recorded that since the spine was added, and PHASE 1.3 is where it
lives. It is listed as unavailable rather than approximated against a
guessed benchmark, because "tracking error vs. something we picked for
you" is a different number wearing the same name.

HOLDINGS OVERLAP IS TOP-TEN OVERLAP, and says so everywhere it appears.
Yahoo returns ten holdings per fund, which is 37-46% of a typical fund
(measured when the decomposition panel was built). "In A but not B" over
that slice is a real and useful signal about concentration, but it is
NOT the full overlap between two funds, and a reader who thinks it is
would badly misjudge how correlated two funds are.

PERFORMANCE IS REBASED, NOT RAW. Two funds at $60 and $700 cannot be
read on one axis; both are rebased to 100 at the first common date, so
the lines answer "what would a pound have done in each".
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

import etf_analysis
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("etf_comparison")

MAX_FUNDS = 3
CACHE_TTL_SECONDS = 3600

TRACKING_ERROR_UNAVAILABLE = (
    "Tracking error is not shown. It needs each fund's stated benchmark "
    "as a return series, and this data source returns neither the "
    "benchmark mapping nor the index history. Approximating it against a "
    "benchmark we picked would be a different number under the same name."
)

HOLDINGS_CAVEAT = (
    "Overlap is measured over the TOP TEN holdings each fund discloses — "
    "about 37-46% of a typical fund, not all of it. Read it as a "
    "concentration signal, not as how correlated the two funds are."
)


@dataclass(frozen=True)
class ComparisonRow:
    """One metric across the compared funds. `values` is per symbol."""
    key: str
    label: str
    values: Dict[str, Optional[float]]
    unit: str = ""
    decimals: int = 2
    lower_is_better: Optional[bool] = None   # None = no ranking applies
    note: str = ""


# The grid, in reading order. `lower_is_better` drives the "best" mark;
# it is None where better/worse is a matter of mandate rather than
# quality — a fund is not superior for holding more assets.
METRICS: Tuple[Tuple[str, str, str, int, Optional[bool], str], ...] = (
    ("expense_ratio_pct", "Expense ratio", "%", 2, True,
     "Net annual cost. The one figure here that is certain in advance."),
    ("net_assets", "Assets under management", "$", 0, None,
     "Size, not quality."),
    ("price_earnings", "Price / Earnings", "", 2, None,
     "Whole-fund, not the top ten."),
    ("dividend_yield_pct", "Dividend yield", "%", 2, None, ""),
    ("return_1y_pct", "1-year return", "%", 2, False, ""),
    ("return_3y_pct", "3-year return (annualised)", "%", 2, False, ""),
    ("volatility_pct", "Volatility (annualised)", "%", 2, True,
     "Standard deviation of daily returns over the loaded window."),
    ("sharpe", "Sharpe (excess/vol)", "", 2, False,
     "Computed from the loaded price window, not the fund's own "
     "published figure."),
)


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_prices(symbols: Tuple[str, ...], period: str = "1y"
                ) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    """Adjusted closes for the compared funds. Never raises."""
    symbols = tuple(s for s in symbols if s)
    if not symbols:
        return None, "No funds selected."
    try:
        import yfinance as yf

        frame = yf.download(list(symbols), period=period, progress=False,
                            auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else None
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "etf_comparison.prices_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return None, f"Price history could not be loaded: {exc}"
    if closes is None or closes.empty:
        return None, "Price history came back empty."
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    log_event(logger, logging.INFO, "etf_comparison.prices_loaded",
              symbols=len(symbols), rows=len(closes))
    return closes, None


def rebased(closes: Optional["pd.DataFrame"]) -> Optional["pd.DataFrame"]:
    """Every series to 100 at the first date they ALL have.

    Rebasing each column at its own first valid date would start two
    funds on different days and silently compare different windows; the
    common start is the only honest one.
    """
    if closes is None or closes.empty:
        return None
    frame = closes.dropna(how="any")
    if frame.empty or len(frame) < 2:
        return None
    return (frame / frame.iloc[0]) * 100.0


def _returns(closes: Optional["pd.DataFrame"], symbol: str) -> Optional["pd.Series"]:
    if closes is None or symbol not in getattr(closes, "columns", []):
        return None
    series = pd.to_numeric(closes[symbol], errors="coerce").dropna()
    return series.pct_change().dropna() if len(series) > 2 else None


def volatility_pct(closes, symbol: str) -> Optional[float]:
    rets = _returns(closes, symbol)
    if rets is None or len(rets) < 2:
        return None
    return _number(rets.std() * (252 ** 0.5) * 100.0)


def sharpe(closes, symbol: str, risk_free_pct: float = 0.0) -> Optional[float]:
    """Annualised excess return over annualised volatility.

    Computed from the loaded window rather than reported, and labelled
    that way on screen: a Sharpe over one year and a Sharpe over five are
    different statistics and must not be set beside each other as though
    they were the same.
    """
    rets = _returns(closes, symbol)
    if rets is None or len(rets) < 2:
        return None
    vol = rets.std() * (252 ** 0.5)
    if not vol:
        return None
    annual = ((1 + rets.mean()) ** 252) - 1
    return _number((annual - risk_free_pct / 100.0) / vol)


def total_return_pct(closes, symbol: str, bars: Optional[int] = None
                     ) -> Optional[float]:
    """Percent change over `bars`, or over the whole window when None.

    Returns None rather than a shorter window when there are not enough
    bars — a "3-year return" computed over one year is not one.
    """
    if closes is None or symbol not in getattr(closes, "columns", []):
        return None
    series = pd.to_numeric(closes[symbol], errors="coerce").dropna()
    if len(series) < 2:
        return None
    if bars is None:
        first, last = series.iloc[0], series.iloc[-1]
    else:
        if len(series) <= bars:
            return None
        first, last = series.iloc[-1 - bars], series.iloc[-1]
    first, last = _number(first), _number(last)
    if first is None or last is None or first == 0:
        return None
    return (last / first - 1.0) * 100.0


def build_rows(profiles: Dict[str, "etf_analysis.EtfProfile"],
               closes=None,
               yields: Optional[Dict[str, Optional[float]]] = None
               ) -> List[ComparisonRow]:
    """The metric grid. A fund that does not report a metric holds None
    for it, which renders blank — never a zero that a reader would then
    average."""
    yields = yields or {}
    symbols = list(profiles)
    rows: List[ComparisonRow] = []
    for key, label, unit, decimals, lower_better, note in METRICS:
        values: Dict[str, Optional[float]] = {}
        for symbol in symbols:
            profile = profiles.get(symbol)
            if key == "dividend_yield_pct":
                values[symbol] = _number(yields.get(symbol))
            elif key == "return_1y_pct":
                values[symbol] = total_return_pct(closes, symbol)
            elif key == "return_3y_pct":
                # 3 years of trading days. None when the window is shorter.
                values[symbol] = total_return_pct(closes, symbol, bars=756)
            elif key == "volatility_pct":
                values[symbol] = volatility_pct(closes, symbol)
            elif key == "sharpe":
                values[symbol] = sharpe(closes, symbol)
            else:
                values[symbol] = (_number(getattr(profile, key, None))
                                  if profile is not None else None)
        rows.append(ComparisonRow(key=key, label=label, values=values,
                                  unit=unit, decimals=decimals,
                                  lower_is_better=lower_better, note=note))
    return rows


def best_symbol(row: ComparisonRow) -> Optional[str]:
    """Which fund wins this row, or None when the row is not a ranking or
    fewer than two funds reported it."""
    if row.lower_is_better is None:
        return None
    measured = {s: v for s, v in row.values.items() if v is not None}
    if len(measured) < 2:
        return None
    picker = min if row.lower_is_better else max
    best = picker(measured, key=lambda s: measured[s])
    # A tie has no winner; marking one arbitrarily would invent a
    # difference the data does not show.
    if list(measured.values()).count(measured[best]) > 1:
        return None
    return best


@dataclass(frozen=True)
class Overlap:
    shared: Tuple[str, ...]
    only_a: Tuple[str, ...]
    only_b: Tuple[str, ...]
    shared_weight_pct: float = 0.0     # of A's disclosed top ten

    @property
    def ok(self) -> bool:
        return bool(self.shared or self.only_a or self.only_b)


def holdings_overlap(a: Optional["etf_analysis.EtfProfile"],
                     b: Optional["etf_analysis.EtfProfile"]) -> Overlap:
    """Which names the two funds share across their disclosed top tens.

    See HOLDINGS_CAVEAT: this is the top ten, not the fund.
    """
    def names(profile):
        if profile is None or not getattr(profile, "ok", False):
            return {}
        out = {}
        for holding in getattr(profile, "top_holdings", ()) or ():
            symbol = (getattr(holding, "symbol", "") or "").strip().upper()
            if symbol:
                out[symbol] = _number(getattr(holding, "weight_pct", None)) or 0.0
        return out

    left, right = names(a), names(b)
    shared = tuple(sorted(set(left) & set(right)))
    return Overlap(
        shared=shared,
        only_a=tuple(sorted(set(left) - set(right))),
        only_b=tuple(sorted(set(right) - set(left))),
        shared_weight_pct=sum(left[s] for s in shared),
    )


def format_value(row: ComparisonRow, symbol: str) -> str:
    """A cell, or an explicit blank. Never a fabricated zero."""
    value = row.values.get(symbol)
    if value is None:
        return "Not reported"
    if row.unit == "$":
        for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= cutoff:
                return f"${value / cutoff:,.1f}{suffix}"
        return f"${value:,.0f}"
    return f"{value:,.{row.decimals}f}{row.unit}"
