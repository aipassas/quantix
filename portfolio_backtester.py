"""Portfolio Backtester — runs the currently-configured strategy (from
strategy_builder.py) across a MULTI-TICKER basket with position sizing and
rebalancing, instead of one ticker at a time.

Reuses strategy_builder.run_backtest() UNCHANGED for each ticker's own
entry/exit signal generation and daily strategy-return series — this
module's only job is combining N independently-backtested tickers' daily
returns into ONE portfolio equity curve under a configurable rebalancing
rule, and reporting each ticker's contribution. No indicator math, no
signal logic, and no per-ticker return calculation is reimplemented here.

REBALANCING MODEL (the part a naive implementation gets wrong): weights
DRIFT with each ticker's own daily return between rebalances, exactly like
a real portfolio's dollar allocations move with each holding's
performance, and are reset back to target only on a rebalance event
(periodic and/or threshold-triggered). A "reweight to target every single
day" model would silently assume costless daily reallocation and is not
what "periodic" or "threshold-based" rebalancing means.

CONTRIBUTION METRIC — a disclosed approximation, not exact: each ticker's
reported contribution is the sum of its own (weight_held_that_day × that
day's return) across the backtest — an ADDITIVE decomposition of the
portfolio's daily returns (by construction, the per-day contributions
across all tickers always sum to that day's Portfolio_Return exactly).
Summed over the whole period, ticker contributions therefore sum to the
arithmetic total of daily returns, which is close to but not exactly the
same as the COMPOUNDED total_return_pct once returns are geometrically
linked — the standard trade-off of simple additive return attribution
(the same category of approximation Brinson-style attribution makes
without further geometric smoothing), not a bug.
"""
import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from data_loader import load_ticker_bundle
from logging_setup import get_logger, log_event, log_exception
from price_processing import process_price_data
from risk_analytics import compute_max_drawdown, compute_sharpe_ratio
from strategy_builder import BacktestResult, StrategyRule, run_backtest
from technical_indicators import compute_bollinger_bands, compute_macd, compute_rsi, compute_sma_lines

logger = get_logger("portfolio_backtester")

REBALANCE_FREQUENCIES: Tuple[str, ...] = ("none", "monthly", "quarterly", "annually")
REBALANCE_FREQUENCY_LABELS: Dict[str, str] = {
    "none": "None (buy-and-hold weights, drift forever)",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "annually": "Annually",
}


def normalize_weights(weights: Dict[str, float]) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Rescale target weights to sum to 1.0. Returns (None, reason) instead
    of raising or silently proceeding with a nonsensical allocation when
    the input can't be normalized — negative weights (short positions)
    aren't supported by this engine, since run_backtest() itself is
    long-only throughout."""
    if any(w < 0 for w in weights.values()):
        return None, "Negative weights (short positions) aren't supported — this engine is long-only, matching run_backtest()."
    total = sum(weights.values())
    if total <= 0:
        return None, "Target weights must sum to a positive number."
    return {t: w / total for t, w in weights.items()}, None


def _period_key(date: pd.Timestamp, frequency: str) -> tuple:
    """A comparable key that changes exactly when a new rebalance PERIOD
    begins — comparing consecutive dates' keys is how period boundaries
    (first trading day of a new month/quarter/year) are detected, since
    trading calendars have gaps (weekends, holidays) a fixed day-count
    can't reliably land on."""
    if frequency == "monthly":
        return (date.year, date.month)
    if frequency == "quarterly":
        return (date.year, (date.month - 1) // 3)
    if frequency == "annually":
        return (date.year,)
    raise ValueError(f"Unknown rebalance frequency: {frequency!r}")


@dataclass
class PortfolioBacktestResult:
    df: pd.DataFrame  # index=aligned dates; columns Portfolio_Return, Cum_Portfolio, Cum_Buy_Hold, and Weight_{ticker} (the drifted weight actually held that day, post any same-day rebalance)
    included_tickers: Tuple[str, ...]
    excluded_tickers: Tuple[str, ...]
    exclusion_reasons: Dict[str, str]
    target_weights: Dict[str, float]      # normalized, re-spread across only the included tickers — see run_portfolio_backtest()'s docstring
    rebalance_frequency: str
    rebalance_threshold_pct: Optional[float]
    rebalance_dates: Tuple[pd.Timestamp, ...]
    total_return_pct: float
    total_buy_hold_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    cost_bps: float
    ticker_backtests: Dict[str, BacktestResult] = field(default_factory=dict)  # each ticker's OWN standalone strategy result, for solo comparison
    contribution_pct: Dict[str, float] = field(default_factory=dict)          # see module docstring's CONTRIBUTION METRIC note


def run_portfolio_backtest(
    ticker_dfs: Dict[str, pd.DataFrame],
    rule: StrategyRule,
    target_weights: Dict[str, float],
    sma_length: int,
    rsi_length: int,
    rebalance_frequency: str = "none",
    rebalance_threshold_pct: Optional[float] = None,
    cost_bps: float = 0.0,
) -> Tuple[Optional[PortfolioBacktestResult], Optional[str]]:
    """Backtest `rule` on every ticker in `ticker_dfs` independently (via
    the unmodified single-ticker engine), then combine the results into one
    weighted, rebalanced portfolio equity curve.

    `ticker_dfs` values must already have every indicator column the
    strategy's conditions need (SMA_*/RSI_*/MACD_*/BB_*/Z_Score) — the
    exact same preparation finance.py already does for the single loaded
    ticker before calling run_backtest() directly.

    Returns (None, reason) instead of raising for every input-shaped
    failure (bad weights, no ticker survives), so the caller can render an
    explicit message instead of a traceback — same convention as
    watchlist_panel.add_ticker().

    Weight re-spreading: if a ticker is excluded (missing data, or no
    overlapping trading days with the rest of the basket after alignment),
    its target weight is NOT silently dropped — that would understate the
    portfolio's total exposure to less than 100%. It's redistributed
    proportionally across the tickers that DO survive, and this is
    disclosed via `target_weights` on the result (which reflects the
    re-spread values, not the caller's raw input) plus `exclusion_reasons`.
    """
    normalized, err = normalize_weights(target_weights)
    if err:
        return None, err

    excluded: Dict[str, str] = {}
    per_ticker_returns: Dict[str, pd.Series] = {}
    ticker_backtests: Dict[str, BacktestResult] = {}

    for ticker, weight in normalized.items():
        if weight <= 0:
            continue  # a zero-weight ticker was in the basket UI but isn't actually allocated capital
        df = ticker_dfs.get(ticker)
        if df is None or df.empty or len(df) < 2:
            excluded[ticker] = "no usable price history"
            continue
        try:
            result = run_backtest(df, rule, sma_length, rsi_length, cost_bps=cost_bps)
        except Exception as e:
            log_event(logger, logging.WARNING, "portfolio_backtest.ticker_error", ticker=ticker, error=str(e))
            excluded[ticker] = f"backtest failed: {type(e).__name__}: {e}"
            continue
        returns = result.df["Net_Strategy_Returns"].dropna()
        if returns.empty:
            excluded[ticker] = "no computable strategy returns (indicator warm-up longer than the loaded history)"
            continue
        per_ticker_returns[ticker] = returns
        ticker_backtests[ticker] = result

    if len(per_ticker_returns) < 1:
        return None, "No ticker in the basket produced a usable backtest — " + "; ".join(f"{t}: {r}" for t, r in excluded.items())

    # Inner-join on date so every included ticker actually traded on every
    # date used — same alignment principle portfolio_analytics.build_aligned_returns()
    # uses, applied here to STRATEGY returns instead of raw asset returns.
    returns_df = pd.DataFrame(per_ticker_returns).dropna(how="any")
    for ticker in list(per_ticker_returns):
        if ticker not in returns_df.columns or returns_df[ticker].isna().all():
            excluded[ticker] = "no overlapping trading days with the rest of the basket"
            per_ticker_returns.pop(ticker, None)
            ticker_backtests.pop(ticker, None)
    returns_df = returns_df[[t for t in per_ticker_returns if t in returns_df.columns]]

    if returns_df.empty or len(returns_df.columns) < 1:
        return None, "No overlapping trading days across the basket after alignment."

    included = list(returns_df.columns)
    # Re-spread weights across only the surviving tickers — see docstring.
    surviving_raw = {t: normalized[t] for t in included}
    respread, respread_err = normalize_weights(surviving_raw)
    if respread_err:
        return None, respread_err

    dates = returns_df.index
    weights = dict(respread)  # weights actually held TODAY; starts at target on day 1
    portfolio_returns = []
    weight_rows = []
    rebalance_dates = []
    contribution_totals = {t: 0.0 for t in included}
    prev_date = None

    for date in dates:
        # Rebalance decision happens BEFORE today's return is earned — the
        # standard convention (rebalance at the start of the new period /
        # as soon as drift is detected, so that day's return already
        # reflects the reset weights), not after. `weights` at this point
        # is what drifted forward from yesterday's close.
        should_rebalance = False
        if rebalance_frequency != "none" and prev_date is not None:
            if _period_key(date, rebalance_frequency) != _period_key(prev_date, rebalance_frequency):
                should_rebalance = True
        if rebalance_threshold_pct is not None:
            max_drift = max(abs(weights[t] - respread[t]) for t in included) * 100
            if max_drift > rebalance_threshold_pct:
                should_rebalance = True
        if should_rebalance:
            weights = dict(respread)
            rebalance_dates.append(date)

        day_returns = returns_df.loc[date]
        day_contribution = {t: weights[t] * float(day_returns[t]) for t in included}
        port_return_today = sum(day_contribution.values())
        for t in included:
            contribution_totals[t] += day_contribution[t]
        portfolio_returns.append(port_return_today)
        weight_rows.append({f"Weight_{t}": weights[t] for t in included})

        # Drift weights forward by today's per-ticker return, then renormalize
        # (drift alone can push the raw values slightly off 1.0 due to floating
        # point; renormalizing keeps "weight" meaning "share of current portfolio
        # value" exactly, which is what it's supposed to mean) — this becomes
        # tomorrow's starting `weights`, subject to tomorrow's own rebalance check.
        drifted_raw = {t: weights[t] * (1 + float(day_returns[t])) for t in included}
        drifted_total = sum(drifted_raw.values())
        weights = {t: v / drifted_total for t, v in drifted_raw.items()} if drifted_total > 0 else dict(respread)
        prev_date = date

    portfolio_df = pd.DataFrame({"Portfolio_Return": portfolio_returns}, index=dates)
    weights_df = pd.DataFrame(weight_rows, index=dates)
    portfolio_df = portfolio_df.join(weights_df)
    portfolio_df["Cum_Portfolio"] = (1 + portfolio_df["Portfolio_Return"]).cumprod()

    buy_hold_returns = sum(returns_df[t] * respread[t] for t in included)  # static-weight reference line, no rebalancing
    portfolio_df["Cum_Buy_Hold"] = (1 + buy_hold_returns).cumprod()

    total_return_pct = (portfolio_df["Cum_Portfolio"].iloc[-1] - 1) * 100
    total_buy_hold_return_pct = (portfolio_df["Cum_Buy_Hold"].iloc[-1] - 1) * 100

    dd_result = compute_max_drawdown(portfolio_df["Cum_Portfolio"])
    max_drawdown_pct = dd_result.max_drawdown * 100 if dd_result is not None else 0.0

    sharpe_ratio = compute_sharpe_ratio(pd.DataFrame({"Close": portfolio_df["Cum_Portfolio"]}))

    contribution_pct = {t: v * 100 for t, v in contribution_totals.items()}

    log_event(
        logger, logging.INFO, "portfolio_backtest.run",
        tickers=len(included), excluded=len(excluded), rebalance_frequency=rebalance_frequency,
        rebalance_count=len(rebalance_dates), total_return_pct=round(total_return_pct, 2),
    )

    return PortfolioBacktestResult(
        df=portfolio_df,
        included_tickers=tuple(included),
        excluded_tickers=tuple(excluded.keys()),
        exclusion_reasons=excluded,
        target_weights=respread,
        rebalance_frequency=rebalance_frequency,
        rebalance_threshold_pct=rebalance_threshold_pct,
        rebalance_dates=tuple(rebalance_dates),
        total_return_pct=total_return_pct,
        total_buy_hold_return_pct=total_buy_hold_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        cost_bps=cost_bps,
        ticker_backtests=ticker_backtests,
        contribution_pct=contribution_pct,
    ), None


@st.cache_data(ttl=3600, show_spinner=False)
def prepare_ticker_for_backtest(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    sma_length: int,
    rsi_length: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """One basket ticker's indicator-prepared DataFrame — the exact same
    Returns/SMA/RSI/MACD/Bollinger/Z-Score preparation finance.py already
    does inline for the single loaded ticker before calling
    strategy_builder.run_backtest() directly, factored out here so every
    basket ticker gets it identically rather than a second, potentially
    drifting copy of that pipeline. Cached like every other deep fetch in
    this app (data_loader.PRICE_HISTORY_TTL's 1-hour convention) since a
    basket run re-touches the same tickers on every widget tweak.

    Returns (None, reason) instead of raising for a ticker Yahoo can't
    resolve or that returns no usable history — the caller reports it as
    an excluded basket member rather than the whole run failing.
    """
    try:
        bundle = load_ticker_bundle(ticker, start_date, end_date, deep=True)
    except Exception as e:
        log_exception(logger, "prepare_ticker.error", section="portfolio_backtester", ticker=ticker)
        return None, f"unexpected error: {type(e).__name__}: {e}"

    if not bundle.is_valid or bundle.price_history.empty:
        return None, "; ".join(bundle.errors) or "no price history returned"

    df = process_price_data(bundle.price_history, ticker=ticker).df
    if df.empty or len(df) < max(sma_length, rsi_length) + 2:
        return None, f"only {len(df)} usable bar(s) — not enough to warm up a {max(sma_length, rsi_length)}-period indicator"

    df["Returns"] = df["Close"].pct_change()
    df = compute_sma_lines(df, [sma_length])
    df[f"RSI_{rsi_length}"] = compute_rsi(df, rsi_length)
    df = compute_macd(df)
    df = compute_bollinger_bands(df, sma_length)
    df["Mean"] = df["Close"].rolling(window=sma_length).mean()
    df["Std"] = df["Close"].rolling(window=sma_length).std()
    df["Z_Score"] = (df["Close"] - df["Mean"]) / df["Std"]
    return df, None
