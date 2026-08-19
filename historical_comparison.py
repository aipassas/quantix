"""Replay the analysis as it would have looked on an earlier date.

THE TRAP THIS MODULE EXISTS TO AVOID: the naive way to build "show me the
analysis as of 2024" is to keep today's fundamentals and just move the
price cursor. That is look-ahead bias — it would present a balance sheet
filed in 2026 as though it were known in 2024, producing a Scorecard that
never existed. This app's standing convention is never to fabricate a
number, so nothing here is reused unless it was genuinely knowable on the
chosen date.

WHAT IS HONESTLY REPLAYABLE, and how:

1. Everything price-derived — technicals and every risk metric. The price
   series is truncated to the as-of date and the existing indicator/risk
   functions are re-run over it. This is exact: the bars really are the
   bars that existed then.

2. Statement-derived fundamentals. Yahoo returns roughly five ANNUAL
   periods per statement, so the filing in force on a given date is
   recoverable: the statement columns are sliced to those with a period
   end on or before the as-of date, and the existing standardize/scorecard
   pipeline then naturally treats the most recent survivor as "current".
   No parallel extraction logic, so the historical Scorecard is computed
   by exactly the same code as the live one.

   The cost is RESOLUTION, and it is disclosed rather than hidden:
   annual filings mean the fundamentals only step at ~yearly boundaries,
   so a date in March 2024 is backed by the FY2023 filing. statement_period
   on the result says which filing was used, and the UI shows it.

3. P/E and market cap are RECONSTRUCTED, not reused: the balance sheet
   carries "Ordinary Shares Number", so market cap = historical close ×
   shares-then, and P/E = market cap ÷ net-income-then. Both inputs are
   point-in-time, so the output is too.

WHAT IS DISCARDED: every numeric field on Yahoo's `info` dict is a
snapshot of today with no history behind it (trailingPE, marketCap, beta,
currentPrice, profitMargins, heldPercent*, …). Those are stripped from the
historical bundle entirely rather than silently carried forward — see
_STATIC_INFO_KEYS. Anything that then can't be computed comes back None
and renders as unavailable, never as a plausible-looking guess.
"""
import copy
import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import CHART_DEFAULTS, RISK
from financial_standardization import standardize_financials
from fundamental_analysis import FundamentalAnalysisEngine
from logging_setup import get_logger
from price_processing import process_price_data
from risk_analytics import (
    compute_annualized_volatility,
    compute_calmar_ratio,
    compute_expected_shortfall,
    compute_historical_var,
    compute_max_drawdown,
    compute_risk_score,
    compute_rolling_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)
from technical_indicators import compute_rsi, compute_sma_lines

logger = get_logger("historical_comparison")

# The only `info` keys carried into a historical bundle. Everything else on
# that dict is a today-snapshot with no history, so keeping it would be exactly
# the look-ahead bias this module exists to prevent. These five are
# descriptive and effectively static — a company's name and sector are not
# a market observation.
_STATIC_INFO_KEYS = ("longName", "shortName", "longBusinessSummary", "website", "sector")

# Minimum price bars before a truncated series is worth computing over at
# all. Below this, everything price-derived is reported unavailable rather
# than computed from a handful of points.
_MIN_BARS = 30


@dataclass(frozen=True)
class ComparedMetric:
    label: str
    group: str                       # "Fundamentals" | "Technicals" | "Risk"
    then: Optional[float]
    now: Optional[float]
    unit: str = ""                   # "%", "$", "x" or ""
    decimals: int = 2
    higher_is_better: Optional[bool] = None   # None = directionless

    @property
    def delta(self) -> Optional[float]:
        if self.then is None or self.now is None:
            return None
        return self.now - self.then


@dataclass(frozen=True)
class ComparisonResult:
    as_of: datetime.date
    statement_period: Optional[datetime.date]   # the filing backing "then"
    price_bars: int
    metrics: Tuple[ComparedMetric, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def has_fundamentals(self) -> bool:
        return self.statement_period is not None

    @property
    def has_price_metrics(self) -> bool:
        return self.price_bars >= _MIN_BARS


def _as_date(value) -> Optional[datetime.date]:
    """Coerce a statement column label / index entry to a date."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def statement_periods(bundle) -> Tuple[datetime.date, ...]:
    """Every statement period end available, oldest first."""
    dates = set()
    for df in (bundle.income_stmt, bundle.balance_sheet, bundle.cash_flow):
        if df is None or getattr(df, "empty", True):
            continue
        for col in df.columns:
            d = _as_date(col)
            if d is not None:
                dates.add(d)
    return tuple(sorted(dates))


def statement_period_for(bundle, as_of: datetime.date) -> Optional[datetime.date]:
    """The filing in force on `as_of` — the latest period ending on or
    before it. None when the as-of date predates every filing available,
    which is a real and common case (Yahoo returns ~5 annual periods)."""
    prior = [d for d in statement_periods(bundle) if d <= as_of]
    return prior[-1] if prior else None


def available_range(bundle) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    """The (earliest, latest) dates a replay can honestly be run for,
    bounded by the price history actually loaded."""
    ph = bundle.price_history
    if ph is None or ph.empty:
        return None, None
    idx = ph.index
    return _as_date(idx[0]), _as_date(idx[-1])


def _truncate_statements(df: Optional[pd.DataFrame], as_of: datetime.date) -> pd.DataFrame:
    """Keep only period columns ending on or before as_of."""
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    keep = [c for c in df.columns if (_as_date(c) or datetime.date.max) <= as_of]
    return df[keep] if keep else pd.DataFrame()


def bundle_as_of(bundle, as_of: datetime.date):
    """A copy of `bundle` containing only what was knowable on `as_of`.

    Every time series is truncated and the `info` dict is reduced to
    static descriptors. Feeding this to the ordinary standardize/scorecard
    pipeline is what makes the historical analysis run through exactly the
    same code as the live one rather than a parallel implementation that
    could drift from it.
    """
    hist = copy.copy(bundle)
    hist.info = {k: bundle.info.get(k) for k in _STATIC_INFO_KEYS if bundle.info.get(k) is not None}
    ph = bundle.price_history
    if ph is not None and not ph.empty:
        hist.price_history = ph[[(_as_date(i) or datetime.date.max) <= as_of for i in ph.index]]
    hist.income_stmt = _truncate_statements(bundle.income_stmt, as_of)
    hist.balance_sheet = _truncate_statements(bundle.balance_sheet, as_of)
    hist.cash_flow = _truncate_statements(bundle.cash_flow, as_of)
    # Ownership snapshots have no history either.
    hist.institutional_holders = None
    hist.insider_transactions = None
    return hist


def _reconstruct_info(hist_bundle, close_then: Optional[float]) -> Dict[str, float]:
    """Point-in-time values for the three fields standardize_financials
    happens to source from `info` rather than from the statements.

    net margin, current ratio and market cap are all derivable from data
    that WAS available on the as-of date — net income / revenue and current
    assets / liabilities come straight off the truncated statements, and
    market cap is shares-then × close-then. Stripping `info` wholesale
    (correctly, to avoid look-ahead) would otherwise leave these three
    permanently unavailable in every replay, and market cap in particular
    is a required Altman Z input, so its absence silently cost the
    historical Z-score too.

    Reconstructing them is NOT look-ahead: every input is period-appropriate.
    Anything whose inputs are missing is simply omitted, so the field falls
    back to unavailable rather than to a guess.
    """
    std = standardize_financials(hist_bundle)
    out: Dict[str, float] = {}

    if std.total_revenue and std.net_income is not None and std.total_revenue != 0:
        out["profitMargins"] = std.net_income / std.total_revenue
    if std.current_assets is not None and std.current_liabilities:
        out["currentRatio"] = std.current_assets / std.current_liabilities

    shares = _shares_from(hist_bundle)
    if shares and close_then:
        out["marketCap"] = shares * close_then
    return out


def _shares_from(bundle) -> Optional[float]:
    """Shares outstanding off an ALREADY-truncated balance sheet, taken
    from the LATEST surviving period.

    Selected by column date rather than by position. yfinance happens to
    return statement columns most-recent-first today, so `.iloc[0]` would
    usually give the same answer — but only by luck, and a silently
    wrong share count would feed a wrong historical market cap and P/E
    without anything looking broken. A test pins this by supplying columns
    in ascending order.
    """
    bs = bundle.balance_sheet
    if bs is None or getattr(bs, "empty", True):
        return None
    for alias in ("Ordinary Shares Number", "Share Issued"):
        if alias in bs.index:
            row = bs.loc[alias].dropna()
            if row.empty:
                continue
            latest = max(row.index, key=lambda c: _as_date(c) or datetime.date.min)
            try:
                return float(row[latest])
            except (TypeError, ValueError):
                return None
    return None


def _price_metrics(df: pd.DataFrame, risk_free_rate: float) -> Dict[str, Optional[float]]:
    """Every price-derived metric over whatever series is handed in. The
    same functions the live panels use, so a replayed number is directly
    comparable to the current one rather than an approximation of it."""
    out: Dict[str, Optional[float]] = {}
    if df is None or df.empty or len(df) < _MIN_BARS:
        return out

    work = df.copy()
    work["Returns"] = work["Close"].pct_change()

    out["close"] = float(work["Close"].iloc[-1])
    out["annual_vol"] = compute_annualized_volatility(work)
    rolling = compute_rolling_volatility(work, CHART_DEFAULTS.vol_window_default)
    out["rolling_vol"] = float(rolling.dropna().iloc[-1]) if rolling.notna().any() else None
    out["var"] = compute_historical_var(work, RISK.var_confidence_default, lookback=CHART_DEFAULTS.var_lookback_default)
    out["cvar"] = compute_expected_shortfall(work, RISK.var_confidence_default, lookback=CHART_DEFAULTS.var_lookback_default)
    dd = compute_max_drawdown(work["Close"])
    out["max_drawdown"] = dd.max_drawdown if dd is not None else None
    out["sharpe"] = compute_sharpe_ratio(work, risk_free_rate)
    out["sortino"] = compute_sortino_ratio(work, risk_free_rate)
    out["calmar"] = compute_calmar_ratio(work)

    rsi = compute_rsi(work, CHART_DEFAULTS.rsi_default)
    out["rsi"] = float(rsi.dropna().iloc[-1]) if rsi is not None and rsi.notna().any() else None

    # compute_sma_lines returns a COPY of the frame with SMA_{n} columns
    # added, and leaves the first n-1 rows genuinely NaN rather than
    # averaging fewer bars than asked for — so a truncated series simply
    # reports None for a window it can't fill, which is what we want.
    smas = compute_sma_lines(work, (50, 200))
    for window in (50, 200):
        col = f"SMA_{window}"
        series = smas[col] if col in smas.columns else None
        out[f"sma{window}"] = (float(series.dropna().iloc[-1])
                               if series is not None and series.notna().any() else None)

    out["risk_score"] = None  # filled by the caller, which also has altman_z
    return out


def build_comparison(bundle, as_of: datetime.date, risk_free_rate: Optional[float] = None) -> ComparisonResult:
    """The full then-vs-now comparison for one ticker.

    Never raises: every metric that can't be computed from the data
    genuinely available on `as_of` comes back None and is reported as
    unavailable, with a warning explaining why.
    """
    risk_free_rate = RISK.risk_free_rate if risk_free_rate is None else risk_free_rate
    warnings: List[str] = []

    hist_bundle = bundle_as_of(bundle, as_of)
    period = statement_period_for(bundle, as_of)

    # --- price-derived, both sides ---
    now_df = process_price_data(bundle.price_history, ticker=bundle.ticker).df
    hist_df = (process_price_data(hist_bundle.price_history, ticker=bundle.ticker).df
               if hist_bundle.price_history is not None and not hist_bundle.price_history.empty
               else pd.DataFrame())
    bars = len(hist_df)
    if bars < _MIN_BARS:
        warnings.append(
            f"Only {bars} trading day{'' if bars == 1 else 's'} of price history "
            f"{'exists' if bars == 1 else 'exist'} before {as_of:%d %b %Y} in the loaded "
            f"range — too few to compute technicals or risk metrics, so those are shown as unavailable. "
            f"Widen the Start Date in the sidebar to replay an earlier date."
        )
    then_px = _price_metrics(hist_df, risk_free_rate)
    now_px = _price_metrics(now_df, risk_free_rate)

    # --- statement-derived, both sides, through the SAME pipeline ---
    then_metrics: Dict[str, Optional[float]] = {}
    now_std = standardize_financials(bundle)
    now_fund = FundamentalAnalysisEngine(now_std, raw_info=bundle.info).analyze()
    now_altman, _, _ = FundamentalAnalysisEngine(now_std, raw_info=bundle.info).altman_z_score()

    then_altman = None
    if period is None:
        warnings.append(
            f"No financial statement was filed on or before {as_of:%d %b %Y} in the data available "
            f"(Yahoo returns roughly five annual periods), so the historical fundamentals and "
            f"Blueprint score can't be reconstructed for that date."
        )
        then_fund = None
    else:
        # Two passes on purpose: the first recovers the statement values
        # needed to rebuild the three info-sourced fields point-in-time,
        # the second standardizes with those present so the historical
        # Scorecard and Altman Z have the same inputs the live one does.
        hist_bundle.info = {**hist_bundle.info,
                            **_reconstruct_info(hist_bundle, then_px.get("close"))}
        hist_std = standardize_financials(hist_bundle)
        engine = FundamentalAnalysisEngine(hist_std)
        then_fund = engine.analyze()
        then_altman, _, _ = engine.altman_z_score()
        then_metrics["net_margin"] = hist_std.net_margin
        then_metrics["debt_to_equity"] = hist_std.debt_to_equity
        then_metrics["current_ratio"] = hist_std.current_ratio
        then_metrics["total_revenue"] = hist_std.total_revenue
        then_metrics["net_income"] = hist_std.net_income

        # P/E and market cap RECONSTRUCTED from point-in-time inputs rather
        # than reused from today's info dict.
        then_metrics["net_margin"] = hist_std.net_margin
        then_metrics["current_ratio"] = hist_std.current_ratio
        then_metrics["market_cap"] = hist_std.market_cap
        ni = hist_std.net_income
        if hist_std.market_cap and ni and ni > 0:
            then_metrics["pe_ratio"] = hist_std.market_cap / ni
        else:
            then_metrics["pe_ratio"] = None
        if not hist_std.market_cap:
            # Distinguish the two causes rather than blaming the wrong one.
            # Market cap needs shares-then AND a close-then; when the price
            # side is what's missing, the bars warning above already says so,
            # and asserting "shares weren't reported" would be false.
            if _shares_from(hist_bundle) is None:
                warnings.append(
                    "Shares outstanding wasn't reported on the balance sheet in force then, so the "
                    "historical P/E and market cap couldn't be reconstructed and are shown as unavailable."
                )
            elif then_px.get("close") is None:
                warnings.append(
                    "There isn't enough price history before that date to price the company, so the "
                    "historical P/E and market cap couldn't be reconstructed and are shown as unavailable."
                )

    # --- composite risk score, both sides ---
    def _score(px: Dict[str, Optional[float]], altman: Optional[float]) -> Optional[float]:
        if not px:
            return None
        result = compute_risk_score(
            rolling_volatility=px.get("rolling_vol"), historical_var=px.get("var"),
            expected_shortfall=px.get("cvar"), max_drawdown=px.get("max_drawdown"),
            sharpe_ratio=px.get("sharpe"), sortino_ratio=px.get("sortino"),
            calmar_ratio=px.get("calmar"), altman_z=altman,
        )
        return result.score if result is not None else None

    rows: List[ComparedMetric] = [
        # --- Fundamentals ---
        ComparedMetric("Blueprint Alignment", "Fundamentals",
                       then_fund.score_pct if then_fund else None,
                       now_fund.score_pct if now_fund else None, "%", 1, True),
        ComparedMetric("Net Margin", "Fundamentals",
                       _pct(then_metrics.get("net_margin")), _pct(now_std.net_margin), "%", 2, True),
        ComparedMetric("Debt / Equity", "Fundamentals",
                       then_metrics.get("debt_to_equity"), now_std.debt_to_equity, "", 2, False),
        ComparedMetric("Current Ratio", "Fundamentals",
                       then_metrics.get("current_ratio"), now_std.current_ratio, "", 2, True),
        ComparedMetric("Altman Z-Score", "Fundamentals", then_altman, now_altman, "", 2, True),
        ComparedMetric("P/E Ratio", "Fundamentals",
                       then_metrics.get("pe_ratio"), now_std.pe_ratio, "", 1, False),
        ComparedMetric("Market Cap", "Fundamentals",
                       then_metrics.get("market_cap"), now_std.market_cap, "$", 0, None),
        ComparedMetric("Total Revenue", "Fundamentals",
                       then_metrics.get("total_revenue"), now_std.total_revenue, "$", 0, True),
        ComparedMetric("Net Income", "Fundamentals",
                       then_metrics.get("net_income"), now_std.net_income, "$", 0, True),
        # --- Technicals ---
        ComparedMetric("Close Price", "Technicals", then_px.get("close"), now_px.get("close"), "$", 2, None),
        ComparedMetric("RSI", "Technicals", then_px.get("rsi"), now_px.get("rsi"), "", 1, None),
        ComparedMetric("SMA 50", "Technicals", then_px.get("sma50"), now_px.get("sma50"), "$", 2, None),
        ComparedMetric("SMA 200", "Technicals", then_px.get("sma200"), now_px.get("sma200"), "$", 2, None),
        # --- Risk ---
        ComparedMetric("Composite Risk Score", "Risk",
                       _score(then_px, then_altman), _score(now_px, now_altman), "", 1, True),
        ComparedMetric("Annualized Volatility", "Risk",
                       _pct(then_px.get("annual_vol")), _pct(now_px.get("annual_vol")), "%", 2, False),
        ComparedMetric("1-Day Historical VaR", "Risk",
                       _pct(then_px.get("var")), _pct(now_px.get("var")), "%", 2, True),
        ComparedMetric("Expected Shortfall (CVaR)", "Risk",
                       _pct(then_px.get("cvar")), _pct(now_px.get("cvar")), "%", 2, True),
        ComparedMetric("Max Drawdown", "Risk",
                       _pct(then_px.get("max_drawdown")), _pct(now_px.get("max_drawdown")), "%", 2, True),
        ComparedMetric("Sharpe Ratio", "Risk", then_px.get("sharpe"), now_px.get("sharpe"), "", 2, True),
        ComparedMetric("Sortino Ratio", "Risk", then_px.get("sortino"), now_px.get("sortino"), "", 2, True),
        ComparedMetric("Calmar Ratio", "Risk", then_px.get("calmar"), now_px.get("calmar"), "", 2, True),
    ]

    log_bars = bars
    logger.log(logging.INFO, "historical_comparison.built as_of=%s bars=%d period=%s",
               as_of, log_bars, period)
    return ComparisonResult(
        as_of=as_of, statement_period=period, price_bars=bars,
        metrics=tuple(rows), warnings=tuple(warnings),
    )


def _pct(value: Optional[float]) -> Optional[float]:
    """Fraction -> percentage points, preserving None."""
    return None if value is None else value * 100
