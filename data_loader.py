"""Centralized Yahoo Finance data loading for Quantix.

Every part of finance.py that needs market data should go through this
module instead of calling yfinance directly. Each dataset is fetched once
here and the resulting bundle is reused everywhere else, instead of every
section making its own independent yf.Ticker() call.

Yahoo Finance is an unofficial, occasionally flaky data source, so every
fetch here is retried with backoff and validated before being handed back.
Bundles never raise on partial failure: they carry `errors` (data that's
required and missing after all retries) and `warnings` (data that's
optional and came back empty) so callers can decide how to degrade and can
surface data-quality issues to the user instead of crashing.

Caching strategy: each dataset is cached independently with a TTL matched
to how often it actually changes, instead of one blanket TTL for
everything (see the *_TTL constants below). This means:
  - Changing the selected date range only invalidates price history, not
    the (unrelated, quarterly) financial statements or ownership data.
  - Switching the benchmark symbol doesn't force a refetch of VIX/TNX,
    which are always the same two tickers regardless of benchmark choice.
  - The same ticker's quote/profile data is fetched at most once per TTL
    window no matter how many places ask for it (main analysis, watchlist
    scan, peer comparison all share the same cached _load_info call).
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

from data_providers import DataProvider, get_provider_for_ticker
from logging_setup import get_logger, log_event

logger = get_logger("data_loader")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0
REQUEST_TIMEOUT = 10

# Cache lifetimes, tuned per data type rather than one blanket value.
INFO_TTL = 1800            # 30 min  — semi-real-time quote/profile fields
PRICE_HISTORY_TTL = 3600   # 1 hour  — daily OHLC bars
STATEMENTS_TTL = 86400     # 24 hours — quarterly financial statements
OWNERSHIP_TTL = 43200      # 12 hours — 13F / Form 4 filing cadence
MACRO_TTL = 3600           # 1 hour  — benchmark/VIX/TNX daily OHLC
SEASONALITY_TTL = 86400    # 24 hours — 10y monthly history, effectively static


def _fetch_with_retry(
    fetch_fn: Callable[[], Any],
    *,
    label: str,
    validate: Optional[Callable[[Any], bool]] = None,
    retries: int = MAX_RETRIES,
) -> Tuple[Any, Optional[str]]:
    """Call fetch_fn, retrying on exceptions (network errors, timeouts, rate limits)
    or failed validation, with linear backoff between attempts.

    Returns (value, None) on success. Returns (None, error_message) if every
    retry is exhausted, so callers can fall back to an empty default and
    record why instead of letting the exception crash the app.
    """
    # This function only executes on a cache miss (callers are @st.cache_data
    # wrapped), so every log line here corresponds to a real outbound request.
    last_error: Optional[Exception] = None
    started = time.monotonic()
    for attempt in range(1, retries + 1):
        log_event(logger, logging.DEBUG, "api.request", dataset=label, attempt=attempt)
        try:
            value = fetch_fn()
            if validate is not None and not validate(value):
                raise ValueError("returned an empty or invalid dataset")
            log_event(
                logger, logging.INFO, "api.success", dataset=label,
                attempt=attempt, ms=round((time.monotonic() - started) * 1000),
            )
            return value, None
        except Exception as e:
            last_error = e
            if attempt < retries:
                log_event(
                    logger, logging.WARNING, "api.retry", dataset=label,
                    attempt=attempt, of=retries, error=f"{type(e).__name__}: {e}",
                )
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    log_event(
        logger, logging.ERROR, "api.failed", dataset=label, attempts=retries,
        ms=round((time.monotonic() - started) * 1000),
        error=f"{type(last_error).__name__}: {last_error}",
    )
    return None, f"{label}: failed after {retries} attempt(s) — {last_error}"


def _is_nonempty_frame(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _is_valid_info(info: Any) -> bool:
    # yfinance's known failure mode for an invalid/delisted ticker or a blocked
    # request is a near-empty dict (often just 1-2 keys) instead of an exception.
    return isinstance(info, dict) and len(info) > 5


@dataclass
class TickerBundle:
    """Single source of truth for everything Yahoo Finance knows about one ticker."""
    ticker: str
    info: dict = field(default_factory=dict)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    institutional_holders: Optional[pd.DataFrame] = None
    insider_transactions: Optional[pd.DataFrame] = None
    errors: List[str] = field(default_factory=list)    # required data missing after retries
    warnings: List[str] = field(default_factory=list)  # optional data missing/degraded

    @property
    def is_valid(self) -> bool:
        """False when required data (info, and price history for deep bundles) is missing."""
        return not self.errors


@dataclass
class MacroBundle:
    """Benchmark and macro (VIX / 10Y Treasury) series for the selected date range."""
    benchmark_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    vix_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    tnx_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: List[str] = field(default_factory=list)


def _load_statement_field(stock: "yf.Ticker", attr: str, ticker: str, label: str, warnings: List[str]) -> pd.DataFrame:
    """Load an optional financial statement. Missing/empty is a warning, not an error —
    plenty of legitimate tickers (ETFs, indices) simply don't have one."""
    value, err = _fetch_with_retry(lambda: getattr(stock, attr), label=f"{ticker} {label}")
    if err:
        warnings.append(err)
        return pd.DataFrame()
    if not _is_nonempty_frame(value):
        warnings.append(f"{ticker}: {label} data unavailable.")
        return pd.DataFrame()
    return value


def _load_ownership_field(stock: "yf.Ticker", attr: str, ticker: str, label: str, warnings: List[str]) -> Optional[pd.DataFrame]:
    """Load an optional ownership dataset (institutional holders / insider transactions)."""
    value, err = _fetch_with_retry(lambda: getattr(stock, attr), label=f"{ticker} {label}")
    if err:
        warnings.append(err)
        return None
    return value


@st.cache_data(ttl=INFO_TTL)
def _load_info(ticker: str) -> Tuple[dict, List[str]]:
    """Fetch just the .info dict (quote + company profile).

    Cached by ticker alone (no date range), and shared by every caller —
    main ticker analysis, watchlist scan, and peer comparison all hit this
    same cache entry, so the same ticker is fetched at most once per
    INFO_TTL window no matter how many places ask for it.
    """
    stock = yf.Ticker(ticker)
    info, err = _fetch_with_retry(lambda: stock.info, label=f"{ticker} company info", validate=_is_valid_info)
    if err:
        return {}, [err]
    return info, []


@st.cache_data(ttl=PRICE_HISTORY_TTL)
def _load_price_history(ticker: str, start, end) -> Tuple[pd.DataFrame, List[str]]:
    stock = yf.Ticker(ticker)
    price_history, err = _fetch_with_retry(
        lambda: stock.history(start=start, end=end, timeout=REQUEST_TIMEOUT),
        label=f"{ticker} price history",
        validate=_is_nonempty_frame,
    )
    if err:
        return pd.DataFrame(), [err]
    return price_history, []


@st.cache_data(ttl=STATEMENTS_TTL)
def _load_financial_statements(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Fetch the 3 financial statements, cached for a full day.

    These only change on a quarterly filing cadence, so this is where the
    biggest reduction in repeated downloads comes from — and since this is
    cached independently of the date range, changing the sidebar date
    pickers no longer forces a refetch of statements that haven't changed.
    """
    stock = yf.Ticker(ticker)
    warnings: List[str] = []
    income_stmt = _load_statement_field(stock, "financials", ticker, "income statement", warnings)
    balance_sheet = _load_statement_field(stock, "balance_sheet", ticker, "balance sheet", warnings)
    cash_flow = _load_statement_field(stock, "cashflow", ticker, "cash flow statement", warnings)
    return income_stmt, balance_sheet, cash_flow, warnings


@st.cache_data(ttl=OWNERSHIP_TTL)
def _load_ownership_data(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    stock = yf.Ticker(ticker)
    warnings: List[str] = []
    institutional_holders = _load_ownership_field(stock, "institutional_holders", ticker, "institutional holders", warnings)
    insider_transactions = _load_ownership_field(stock, "insider_transactions", ticker, "insider transactions", warnings)
    return institutional_holders, insider_transactions, warnings


def load_ticker_bundle(ticker: str, start=None, end=None, deep: bool = True) -> TickerBundle:
    """Assemble a TickerBundle from independently-cached sub-fetches (see the
    _load_* functions above). This function itself is NOT cached: it's cheap
    glue code, and caching it too would risk serving a bundle assembled from
    a stale mix if the sub-caches later refresh at different times.

    deep=True pulls price history, financial statements, and ownership data
    on top of `.info` (used for the main analyzed ticker). deep=False only
    pulls `.info` (used for watchlist scans and peer comparisons, which
    never need the heavier statements/ownership data).

    Every underlying fetch is retried on failure and validated; a ticker
    that's temporarily unreachable or that Yahoo returns garbage for comes
    back as a bundle with `errors`/`warnings` populated instead of raising,
    so a single flaky ticker can't crash the whole app.

    A `ticker` prefixed "crypto:" (e.g. "crypto:bitcoin") is routed to an
    alternate data_providers.DataProvider instead of Yahoo Finance — see
    data_providers.py's module docstring for what that covers and why. This
    branch is checked FIRST and returns via a separate helper specifically
    so the Yahoo path below it is untouched byte-for-byte: existing callers
    passing an ordinary ticker get identical behavior to before this layer
    existed.
    """
    provider = get_provider_for_ticker(ticker)
    if provider is not None:
        return _load_ticker_bundle_from_provider(provider, ticker, start, end, deep)

    bundle = TickerBundle(ticker=ticker)

    info, info_errors = _load_info(ticker)
    bundle.info = info
    bundle.errors.extend(info_errors)

    if deep:
        price_history, price_errors = _load_price_history(ticker, start, end)
        bundle.price_history = price_history
        bundle.errors.extend(price_errors)

        bundle.income_stmt, bundle.balance_sheet, bundle.cash_flow, stmt_warnings = _load_financial_statements(ticker)
        bundle.warnings.extend(stmt_warnings)

        bundle.institutional_holders, bundle.insider_transactions, ownership_warnings = _load_ownership_data(ticker)
        bundle.warnings.extend(ownership_warnings)

    log_event(
        logger, logging.ERROR if bundle.errors else logging.INFO, "bundle.loaded",
        ticker=ticker, deep=deep, valid=bundle.is_valid,
        errors=len(bundle.errors), warnings=len(bundle.warnings),
        price_rows=len(bundle.price_history),
    )
    for message in bundle.errors:
        log_event(logger, logging.ERROR, "bundle.error", ticker=ticker, detail=message)
    for message in bundle.warnings:
        log_event(logger, logging.WARNING, "bundle.warning", ticker=ticker, detail=message)

    return bundle


def _load_ticker_bundle_from_provider(provider: DataProvider, ticker: str, start, end, deep: bool) -> TickerBundle:
    """The non-Yahoo path, kept as a separate function (not interleaved
    with the Yahoo path above) precisely so that path stays provably
    unchanged. Mirrors data_providers.DataProvider's own (value, messages)
    convention: messages alongside an EMPTY result are required-data-
    missing (-> bundle.errors); messages alongside a NON-EMPTY result are
    disclosures (-> bundle.warnings), same distinction TickerBundle.is_valid
    already relies on.

    Financial statements and ownership data are Yahoo/equity-specific
    concepts with no analogue for a provider like CoinGecko — left at
    TickerBundle's own empty defaults with a disclosed reason rather than
    silently absent with no explanation.
    """
    bundle = TickerBundle(ticker=ticker)

    info, info_messages = provider.fetch_info(ticker)
    bundle.info = info
    bundle.errors.extend(info_messages)  # a fetch_info failure is always required-data-missing

    if deep:
        price_history, price_messages = provider.fetch_price_history(ticker, start, end)
        bundle.price_history = price_history
        if price_history.empty:
            bundle.errors.extend(price_messages)
        else:
            bundle.warnings.extend(price_messages)
        bundle.warnings.append(f"{ticker}: financial statements and ownership data are not available through this provider.")

    log_event(
        logger, logging.ERROR if bundle.errors else logging.INFO, "bundle.loaded",
        ticker=ticker, deep=deep, valid=bundle.is_valid,
        errors=len(bundle.errors), warnings=len(bundle.warnings),
        price_rows=len(bundle.price_history),
    )
    for message in bundle.errors:
        log_event(logger, logging.ERROR, "bundle.error", ticker=ticker, detail=message)
    for message in bundle.warnings:
        log_event(logger, logging.WARNING, "bundle.warning", ticker=ticker, detail=message)

    return bundle


def load_price_history_only(ticker: str, start=None, end=None) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch just price history for `ticker`, without the rest of a deep
    bundle (statements, ownership) — for basket-style use cases like
    portfolio correlation that only need OHLCV for several tickers at
    once and would otherwise pay for financial-statement fetches they
    never use. Reuses the same independently-cached _load_price_history()
    load_ticker_bundle(deep=True) already calls, so this never duplicates
    a fetch already in flight for the main analyzed ticker.
    """
    return _load_price_history(ticker, start, end)


@st.cache_data(ttl=MACRO_TTL)
def _load_symbol_history(symbol: str, start, end) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch history for one symbol. Cached per-symbol so VIX and TNX (always
    the same two tickers) aren't refetched just because the user changed the
    benchmark symbol."""
    value, err = _fetch_with_retry(
        lambda: yf.Ticker(symbol).history(start=start, end=end, timeout=REQUEST_TIMEOUT),
        label=f"{symbol} history",
        validate=_is_nonempty_frame,
    )
    if err:
        return pd.DataFrame(), [err]
    return value, []


def load_macro_bundle(benchmark: str, start=None, end=None) -> MacroBundle:
    """Fetch benchmark, VIX, and 10Y Treasury history. Each symbol is cached
    independently (see _load_symbol_history) so switching the benchmark
    symbol doesn't force a refetch of VIX/TNX."""
    bundle = MacroBundle()
    bench_df, bench_warnings = _load_symbol_history(benchmark, start, end)
    vix_df, vix_warnings = _load_symbol_history("^VIX", start, end)
    tnx_df, tnx_warnings = _load_symbol_history("^TNX", start, end)
    bundle.benchmark_history = bench_df
    bundle.vix_history = vix_df
    bundle.tnx_history = tnx_df
    bundle.warnings = bench_warnings + vix_warnings + tnx_warnings

    log_event(
        logger, logging.WARNING if bundle.warnings else logging.INFO, "macro.loaded",
        benchmark=benchmark, warnings=len(bundle.warnings),
        bench_rows=len(bench_df), vix_rows=len(vix_df), tnx_rows=len(tnx_df),
    )
    for message in bundle.warnings:
        log_event(logger, logging.WARNING, "macro.warning", detail=message)

    return bundle


@st.cache_data(ttl=SEASONALITY_TTL)
def load_seasonality_history(ticker: str, period: str = "10y", interval: str = "1mo") -> pd.DataFrame:
    """Fetch long-range history for seasonality analysis.

    Kept separate from price_history since it uses a different period/interval
    (10 years of monthly bars vs. the user-selected daily range), but still
    routed through this module so it's not an independent yfinance call site.
    Cached for a full day since a decade of monthly bars is effectively
    static within any given day.

    Yahoo-only, deliberately not routed through data_providers.py: a true
    10-year seasonality view isn't even possible against CoinGecko's free
    tier (365-day cap, see data_providers.py's module docstring), so this
    isn't "not yet wired up" so much as genuinely out of reach for that
    provider. Short-circuits immediately for a non-Yahoo ticker instead of
    silently falling through to Yahoo and burning 3 retries against a
    symbol Yahoo was never going to recognize — caught live: without this
    check, analyzing a crypto ticker cost an extra ~7 real seconds of
    failed retries before the (already-graceful) empty result appeared.
    """
    if get_provider_for_ticker(ticker) is not None:
        return pd.DataFrame()

    value, err = _fetch_with_retry(
        lambda: yf.Ticker(ticker).history(period=period, interval=interval, timeout=REQUEST_TIMEOUT),
        label=f"{ticker} seasonality history",
        validate=_is_nonempty_frame,
    )
    if err:
        return pd.DataFrame()
    return value


def clear_all_caches() -> None:
    """Clear every cached dataset in this module. Used by the sidebar
    'Force Refresh' button so the user can bypass the TTLs on demand
    (e.g. right after market open) instead of waiting for them to expire.
    """
    _load_info.clear()
    _load_price_history.clear()
    _load_financial_statements.clear()
    _load_ownership_data.clear()
    _load_symbol_history.clear()
    load_seasonality_history.clear()
