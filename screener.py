"""Multi-ticker stock screening engine.

Evaluates a ticker universe against user-defined filter criteria spanning
fundamentals (P/E, PEG, margins, ROE, ...), technical/risk metrics (RSI,
Sharpe, Sortino, volatility, max drawdown), and statement-derived metrics
(Altman Z-Score). See finance.py for the filter-builder UI.

Deliberately a thin orchestration layer: every metric value is computed by
the same functions the single-ticker analysis already uses
(financial_standardization.py, fundamental_analysis.py, risk_analytics.py,
technical_indicators.py) — no parallel metric-calculation logic here.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from config import CHART_DEFAULTS, RISK
from data_loader import load_price_history_only, load_ticker_bundle
from financial_standardization import standardize_financials
from fundamental_analysis import FundamentalAnalysisEngine
from logging_setup import get_logger, log_event, log_exception
from price_processing import process_price_data
from risk_analytics import (
    compute_annualized_volatility,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)
from technical_indicators import compute_rsi

logger = get_logger("screener")

MAX_UNIVERSE_SIZE = 30  # Yahoo Finance rate limits scale with universe size (see task notes)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    tier: str  # "shallow" (info-only), "deep" (statements), or "price" (OHLCV)
    unit: str = ""
    decimals: int = 2


# Every metric usable as a screen criterion. Tiers determine which fetch(es)
# a ticker needs — "shallow" and "deep" both come from load_ticker_bundle()
# (deep=False vs deep=True), "price" comes from load_price_history_only().
# Kept broader than the task notes' 4 named examples (P/E, PEG, ROE, RSI/
# Sharpe/Volatility/Altman Z) deliberately: every "shallow" metric here is
# already present on the same single info-only fetch a P/E or PEG criterion
# already requires, so adding Price/Book, Debt/Equity, Current Ratio, and
# Beta costs nothing extra: it's free breadth, not scope creep. Same
# reasoning for Sortino Ratio and Max Drawdown alongside Sharpe/RSI/
# Volatility — one price-history fetch computes all of them.
METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("pe_ratio", "P/E Ratio", "shallow"),
    MetricSpec("peg_ratio", "PEG Ratio", "shallow"),
    MetricSpec("price_to_book", "Price/Book", "shallow"),
    MetricSpec("net_margin_pct", "Net Margin", "shallow", "%"),
    MetricSpec("roe_pct", "Return on Equity", "shallow", "%"),
    MetricSpec("debt_to_equity", "Debt/Equity", "shallow"),
    MetricSpec("current_ratio", "Current Ratio", "shallow"),
    MetricSpec("beta", "Beta", "shallow"),
    MetricSpec("rsi", f"RSI ({CHART_DEFAULTS.rsi_default})", "price"),
    MetricSpec("sharpe_ratio", "Sharpe Ratio", "price"),
    MetricSpec("sortino_ratio", "Sortino Ratio", "price"),
    MetricSpec("annual_volatility_pct", "Annualized Volatility", "price", "%"),
    MetricSpec("max_drawdown_pct", "Max Drawdown", "price", "%"),
    MetricSpec("altman_z", "Altman Z-Score", "deep"),
)
METRICS_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in METRICS}

OPERATORS: Dict[str, "callable"] = {
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
}


@dataclass(frozen=True)
class ScreenCriterion:
    metric: str    # key into METRICS_BY_KEY
    operator: str  # key into OPERATORS
    threshold: float


@dataclass
class ScreenResult:
    """One ticker's outcome: every screened metric's value, one pass/fail
    per criterion, and an overall status. `criteria_passes[i]` is None (not
    False) when the metric behind criterion i couldn't be computed for this
    ticker — a missing metric is never silently treated as a failure, per
    this codebase's "never fabricate, always disclose" convention."""
    ticker: str
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    criteria_passes: List[Optional[bool]] = field(default_factory=list)
    status: str = "ok"   # "ok" | "insufficient_data" | "fetch_error"
    detail: str = ""

    @property
    def passed_all(self) -> Optional[bool]:
        if not self.criteria_passes:
            return None
        if any(p is None for p in self.criteria_passes):
            return None
        return all(self.criteria_passes)


def _shallow_values(std, wanted: set) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    if "pe_ratio" in wanted:
        out["pe_ratio"] = std.pe_ratio
    if "peg_ratio" in wanted:
        out["peg_ratio"] = std.peg_ratio
    if "price_to_book" in wanted:
        out["price_to_book"] = std.price_to_book
    if "net_margin_pct" in wanted:
        out["net_margin_pct"] = std.net_margin * 100 if std.net_margin is not None else None
    if "roe_pct" in wanted:
        out["roe_pct"] = std.return_on_equity * 100 if std.return_on_equity is not None else None
    if "debt_to_equity" in wanted:
        out["debt_to_equity"] = std.debt_to_equity
    if "current_ratio" in wanted:
        out["current_ratio"] = std.current_ratio
    if "beta" in wanted:
        out["beta"] = std.beta
    return out


def _deep_values(std, wanted: set) -> Tuple[Dict[str, Optional[float]], List[str]]:
    out: Dict[str, Optional[float]] = {}
    notes: List[str] = []
    if "altman_z" in wanted:
        z, verdict, missing = FundamentalAnalysisEngine(std).altman_z_score()
        out["altman_z"] = z
        if z is None:
            notes.append(f"Altman Z-Score unavailable ({verdict}: missing {', '.join(missing)})" if missing else f"Altman Z-Score unavailable ({verdict})")
    return out, notes


def _price_values(ticker: str, wanted: set, start, end, risk_free_rate: float) -> Tuple[Dict[str, Optional[float]], List[str]]:
    out: Dict[str, Optional[float]] = {key: None for key in wanted}
    notes: List[str] = []

    history, errors = load_price_history_only(ticker, start, end)
    if errors or history.empty:
        notes.append("No price history available" + (f" ({'; '.join(errors)})" if errors else ""))
        return out, notes

    df = process_price_data(history, ticker=ticker).df

    if "rsi" in wanted:
        rsi_series = compute_rsi(df, CHART_DEFAULTS.rsi_default)
        out["rsi"] = rsi_series.iloc[-1] if rsi_series.notna().any() else None
    if "sharpe_ratio" in wanted:
        out["sharpe_ratio"] = compute_sharpe_ratio(df, risk_free_rate)
    if "sortino_ratio" in wanted:
        out["sortino_ratio"] = compute_sortino_ratio(df, risk_free_rate)
    if "annual_volatility_pct" in wanted:
        vol = compute_annualized_volatility(df)
        out["annual_volatility_pct"] = vol * 100 if vol is not None else None
    if "max_drawdown_pct" in wanted:
        dd = compute_max_drawdown(df["Close"])
        out["max_drawdown_pct"] = dd.max_drawdown * 100 if dd is not None else None

    missing = [METRICS_BY_KEY[k].label for k, v in out.items() if v is None]
    if missing:
        notes.append(f"Insufficient history to compute: {', '.join(missing)}")
    return out, notes


def _screen_one(ticker: str, criteria: Sequence[ScreenCriterion], start, end, risk_free_rate: float) -> ScreenResult:
    wanted_by_tier: Dict[str, set] = {"shallow": set(), "deep": set(), "price": set()}
    for c in criteria:
        wanted_by_tier[METRICS_BY_KEY[c.metric].tier].add(c.metric)

    needs_deep = bool(wanted_by_tier["deep"])
    try:
        bundle = load_ticker_bundle(ticker, deep=needs_deep)
    except Exception as e:
        log_exception(logger, "calc.error", section="screener", ticker=ticker)
        return ScreenResult(ticker=ticker, status="fetch_error", detail=f"Unexpected error: {type(e).__name__}: {e}",
                             criteria_passes=[None] * len(criteria))

    if not bundle.is_valid:
        return ScreenResult(ticker=ticker, status="fetch_error", detail="; ".join(bundle.errors) or "Ticker could not be loaded",
                             criteria_passes=[None] * len(criteria))

    std = standardize_financials(bundle)
    values: Dict[str, Optional[float]] = {}
    notes: List[str] = []

    values.update(_shallow_values(std, wanted_by_tier["shallow"] | wanted_by_tier["deep"]))
    if needs_deep:
        deep_vals, deep_notes = _deep_values(std, wanted_by_tier["deep"])
        values.update(deep_vals)
        notes.extend(deep_notes)

    if wanted_by_tier["price"]:
        price_vals, price_notes = _price_values(ticker, wanted_by_tier["price"], start, end, risk_free_rate)
        values.update(price_vals)
        notes.extend(price_notes)

    missing_shallow = [METRICS_BY_KEY[k].label for k in (wanted_by_tier["shallow"] - wanted_by_tier["deep"]) if values.get(k) is None]
    if missing_shallow:
        notes.append(f"Not reported by this ticker: {', '.join(missing_shallow)}")

    criteria_passes: List[Optional[bool]] = []
    for c in criteria:
        v = values.get(c.metric)
        criteria_passes.append(OPERATORS[c.operator](v, c.threshold) if v is not None else None)

    status = "ok" if all(p is not None for p in criteria_passes) else "insufficient_data"
    return ScreenResult(ticker=ticker, values=values, criteria_passes=criteria_passes, status=status, detail="; ".join(notes))


@st.cache_data(ttl=3600, show_spinner=False)
def run_screen(
    tickers: Tuple[str, ...],
    criteria: Tuple[ScreenCriterion, ...],
    start=None,
    end=None,
    risk_free_rate: Optional[float] = None,
) -> List[ScreenResult]:
    """Screen `tickers` against `criteria`. Tuples (not lists) so this is
    cacheable — repeated screens of the same universe/criteria don't
    re-fetch. Underlying per-ticker fetches are independently cached too
    (data_loader.py), so overlapping universes across separate screens
    still only pay for what isn't already cached.
    """
    start = start or (pd.Timestamp.today() - pd.Timedelta(days=CHART_DEFAULTS.default_lookback_days)).date()
    end = end or pd.Timestamp.today().date()
    risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK.risk_free_rate

    log_event(logger, logging.INFO, "screener.run", universe_size=len(tickers), criteria_count=len(criteria))
    results = [_screen_one(t, criteria, start, end, risk_free_rate) for t in tickers]
    log_event(logger, logging.INFO, "screener.complete", universe_size=len(tickers),
              passed=sum(1 for r in results if r.passed_all), insufficient=sum(1 for r in results if r.status == "insufficient_data"),
              errors=sum(1 for r in results if r.status == "fetch_error"))
    return results
