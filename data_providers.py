"""Pluggable data-provider layer for Quantix.

Yahoo Finance (via data_loader.py) remains the default and only provider
for equities — nothing about that path changes. This module adds a
genuinely alternate, independently-verified provider for a different
asset class (crypto): CoinGecko's free, keyless public API. It also
defines the common interface a further provider would need to implement
to plug in the same way, without any downstream code (finance.py, every
analysis module) changing — they only ever see a data_loader.TickerBundle
back from load_ticker_bundle(), unaware of which provider filled it in.

PROVIDER SELECTION is explicit, not sniffed: a ticker prefixed "crypto:"
(e.g. "crypto:bitcoin" — CoinGecko's own coin-id scheme, NOT a ticker
symbol; CoinGecko is id-keyed, and symbols like "btc" collide across many
listed coins) routes to CoinGeckoProvider. Everything else routes to
Yahoo Finance, unchanged. A silent heuristic ("does this look like
crypto?") was considered and rejected — an explicit prefix is honest
about which provider is being asked for, matching this app's disclosure
conventions everywhere else (e.g. watchlist_panel's never-fabricate
day-change rule).

SCOPE, DECIDED WITH THE USER BEFORE BUILDING: the originating task named
Interactive Brokers, Alpaca, and CryptoCompare as example providers. None
can be integrated END-TO-END from this environment — Interactive Brokers
needs a locally-running TWS/IB Gateway application connected to a funded
or paper brokerage account; Alpaca and CryptoCompare both need an account
and an API key, neither of which this assistant can create on the user's
behalf. Writing adapter code for them that was never once called against
the real API would mean claiming "integrated end-to-end" without it being
true. CoinGecko was chosen instead specifically because its public API
needs no key or account and every behavior documented below was verified
against the live endpoints while building this, not assumed. The
DataProvider interface is written so a future credentialed adapter
(Alpaca, CryptoCompare, IBKR) is a drop-in: same two methods, same
(value, messages) return convention data_loader.py's Yahoo path already
uses for _load_info/_load_price_history.

TWO DISCLOSED LIMITATIONS OF COINGECKO'S FREE TIER (each confirmed live,
not assumed from documentation):
  1. No historical data older than 365 days on the public tier — a real,
     paid-plan-only restriction (confirmed via a live request beyond that
     window, which returned CoinGecko's own error_code 10012). A
     requested start date further back is silently clamped to 365 days
     before the end date, and the clamp is disclosed via
     bundle.warnings — never silently substituted without a trace.
  2. No true OHLC over an arbitrary historical range on the free tier —
     only a price time series (the `market_chart/range` endpoint),
     confirmed by inspecting the actual response shape (it returns
     `prices`/`total_volumes` arrays, no `open`/`high`/`low`). Open/High/
     Low are therefore set equal to that bar's Close rather than
     fabricated, and this is disclosed via bundle.warnings on every
     crypto fetch. Volume IS genuinely returned by this same endpoint and
     is not an approximation.

GRANULARITY: CoinGecko's `market_chart/range` returns HOURLY points for
ranges under ~90 days and DAILY points beyond that (confirmed live by
counting returned points against known ranges — not documented precisely
by CoinGecko itself). Resampled to exactly one bar per calendar day here
unconditionally, so indicator math elsewhere in this app — built entirely
around daily bars, where "SMA-20" means 20 DAYS everywhere else — never
silently operates on 20 HOURS of hourly data instead for a short-range
crypto fetch.
"""
import datetime
import logging
import time
from typing import List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from logging_setup import get_logger, log_event

logger = get_logger("data_providers")

CRYPTO_PREFIX = "crypto:"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_MAX_HISTORY_DAYS = 365   # free-tier cap, confirmed live (error_code 10012 beyond this)
COINGECKO_TIMEOUT_SECONDS = 10
COINGECKO_MAX_RETRIES = 3
COINGECKO_INFO_TTL = 1800          # matches data_loader.INFO_TTL
COINGECKO_PRICE_TTL = 3600         # matches data_loader.PRICE_HISTORY_TTL


def is_crypto_ticker(ticker: str) -> bool:
    return ticker.strip().lower().startswith(CRYPTO_PREFIX)


def _coingecko_coin_id(ticker: str) -> str:
    return ticker.strip()[len(CRYPTO_PREFIX):].strip().lower()


def _fetch_json_with_retry(url: str, params: dict, label: str, retries: int = COINGECKO_MAX_RETRIES) -> Tuple[Optional[dict], Optional[str]]:
    """Same linear-backoff retry discipline as data_loader.py's own
    _fetch_with_retry, kept local to this module rather than importing
    that private helper across modules — every other module in this app
    only ever imports data_loader's PUBLIC names (TickerBundle,
    load_ticker_bundle, etc.), never its underscore-prefixed internals.

    Distinguishes a definitive "not found" (404 — bad coin id, not worth
    retrying) from a transient failure (network error, 429 rate limit,
    5xx) worth retrying with backoff.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=COINGECKO_TIMEOUT_SECONDS)
            if response.status_code == 404:
                return None, f"{label}: unknown CoinGecko coin id"
            if response.status_code == 429:
                raise RuntimeError("rate limited by CoinGecko's free public tier")
            response.raise_for_status()
            return response.json(), None
        except Exception as e:
            last_error = e
            if attempt < retries:
                log_event(logger, logging.WARNING, "coingecko.retry", label=label, attempt=attempt, of=retries, error=f"{type(e).__name__}: {e}")
                time.sleep(1.0 * attempt)
    log_event(logger, logging.ERROR, "coingecko.failed", label=label, attempts=retries, error=f"{type(last_error).__name__}: {last_error}")
    return None, f"{label}: failed after {retries} attempt(s) — {type(last_error).__name__}: {last_error}"


@st.cache_data(ttl=COINGECKO_INFO_TTL, show_spinner=False)
def _coingecko_fetch_info(coin_id: str) -> Tuple[dict, List[str]]:
    data, err = _fetch_json_with_retry(
        f"{COINGECKO_BASE_URL}/coins/{coin_id}",
        {"localization": "false", "tickers": "false", "market_data": "true", "community_data": "false", "developer_data": "false", "sparkline": "false"},
        label=f"crypto:{coin_id} info",
    )
    if err:
        return {}, [err]

    market = data.get("market_data") or {}
    current_price = (market.get("current_price") or {}).get("usd")
    change_24h = market.get("price_change_24h")
    # Derived, not fetched directly — CoinGecko's market_data has no
    # separate "previous close" field, so this is built from the two
    # values it DOES give (matches this app's existing derive-don't-trust
    # convention for day-change: see watchlist_panel.py's own docstring on
    # why a derived day-change beats trusting a single opaque field).
    previous_close = (current_price - change_24h) if current_price is not None and change_24h is not None else None

    info = {
        "longName": data.get("name"),
        "shortName": data.get("name"),
        "symbol": (data.get("symbol") or "").upper(),
        "currency": "USD",
        "currentPrice": current_price,
        "regularMarketPrice": current_price,
        "previousClose": previous_close,
        "regularMarketPreviousClose": previous_close,
        "marketCap": (market.get("market_cap") or {}).get("usd"),
        "quoteType": "CRYPTOCURRENCY",
        # Deliberately absent, not fabricated: crypto has no GICS sector,
        # no P/E (no earnings), no shares-outstanding-style equity
        # metrics. financial_standardization.py already treats a missing
        # info key as None everywhere, never a fabricated default — so
        # simply not setting these keys is the correct, already-supported
        # way to say "not applicable," not a gap that needs filling.
    }
    return info, []


@st.cache_data(ttl=COINGECKO_PRICE_TTL, show_spinner=False)
def _coingecko_fetch_price_history(coin_id: str, start: Optional[datetime.date], end: Optional[datetime.date]) -> Tuple[pd.DataFrame, List[str]]:
    messages: List[str] = []
    end_date = end or datetime.date.today()
    earliest_allowed = end_date - datetime.timedelta(days=COINGECKO_MAX_HISTORY_DAYS)
    start_date = start or earliest_allowed
    if start_date < earliest_allowed:
        messages.append(
            f"crypto:{coin_id}: CoinGecko's free tier only allows the last {COINGECKO_MAX_HISTORY_DAYS} days of "
            f"history — clamped the requested start date to {earliest_allowed.isoformat()}."
        )
        start_date = earliest_allowed

    # Anchored to UTC explicitly — a NAIVE datetime's .timestamp() is
    # interpreted as the SERVER's local time, which would silently shift
    # the requested day boundary by that server's UTC offset (caught while
    # writing this module's own tests: on a UTC+3 machine, "start of June
    # 1st" naively resolved to 21:00 UTC on May 31st, a full day off).
    # CoinGecko's own returned timestamps are UTC, so the request window
    # must be too, deterministically, regardless of where this app runs.
    from_ts = int(datetime.datetime.combine(start_date, datetime.time.min, tzinfo=datetime.timezone.utc).timestamp())
    to_ts = int(datetime.datetime.combine(end_date, datetime.time.max, tzinfo=datetime.timezone.utc).timestamp())

    data, err = _fetch_json_with_retry(
        f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range",
        {"vs_currency": "usd", "from": from_ts, "to": to_ts},
        label=f"crypto:{coin_id} price history",
    )
    if err:
        return pd.DataFrame(), [err]

    prices = data.get("prices") or []
    volumes = data.get("total_volumes") or []
    if not prices:
        return pd.DataFrame(), [f"crypto:{coin_id}: no price data returned for the selected range"]

    price_series = pd.Series({pd.Timestamp(ts, unit="ms"): value for ts, value in prices}).sort_index()
    volume_series = pd.Series({pd.Timestamp(ts, unit="ms"): value for ts, value in volumes}).sort_index()

    # One bar per calendar day regardless of source granularity — see
    # module docstring's GRANULARITY note. Close = last observation of the
    # day; Open/High/Low set equal to Close (disclosed below, never
    # fabricated as if it were real intraday range); Volume = that day's
    # last cumulative reading from the same endpoint (genuine, not derived).
    daily_close = price_series.resample("D").last().dropna()
    daily_volume = volume_series.resample("D").last().reindex(daily_close.index)

    df = pd.DataFrame({
        "Open": daily_close, "High": daily_close, "Low": daily_close,
        "Close": daily_close, "Volume": daily_volume,
    })
    df.index.name = "Date"

    messages.append(
        f"crypto:{coin_id}: Open/High/Low are set equal to Close — CoinGecko's free tier doesn't expose true "
        f"intraday OHLC over an arbitrary historical range. Volume is genuine, not derived."
    )
    return df, messages


class DataProvider:
    """The common interface a data source must implement to plug into
    data_loader.load_ticker_bundle() without any downstream code changing.

    Both methods return (value, messages) — the exact convention
    data_loader.py's own _load_info/_load_price_history already use:
    `messages` accompanying an EMPTY result means required data is
    missing (the caller routes it to TickerBundle.errors); `messages`
    accompanying a NON-EMPTY result are disclosures, not failures (routed
    to TickerBundle.warnings). This lets a provider disclose a real,
    non-fatal caveat (like CoinGecko's OHLC approximation above) through
    the same channel this app already uses for "the data is real but you
    should know this about it," without inventing a second interface.

    Only price history and a quote/profile "info" dict are required.
    Fundamentals (statements) and account/positions are deliberately NOT
    part of this interface: no provider integrated so far exposes
    brokerage account data, and forcing an unused method onto every
    adapter would be exactly the kind of premature abstraction this
    codebase avoids elsewhere (see watchlist_panel.py's own note on not
    over-generalizing for a case that isn't here yet).
    """

    def fetch_info(self, ticker: str) -> Tuple[dict, List[str]]:
        raise NotImplementedError

    def fetch_price_history(self, ticker: str, start: Optional[datetime.date], end: Optional[datetime.date]) -> Tuple[pd.DataFrame, List[str]]:
        raise NotImplementedError


class CoinGeckoProvider(DataProvider):
    """Crypto asset data from CoinGecko's free, keyless public API — see
    this module's docstring for the two disclosed limitations of its free
    tier and exactly how each was verified against the live API."""

    def fetch_info(self, ticker: str) -> Tuple[dict, List[str]]:
        return _coingecko_fetch_info(_coingecko_coin_id(ticker))

    def fetch_price_history(self, ticker: str, start: Optional[datetime.date], end: Optional[datetime.date]) -> Tuple[pd.DataFrame, List[str]]:
        return _coingecko_fetch_price_history(_coingecko_coin_id(ticker), start, end)


def get_provider_for_ticker(ticker: str) -> Optional[DataProvider]:
    """None means "use Yahoo Finance" — data_loader.py's existing,
    completely unchanged default path. Only a non-default provider is
    ever returned here; Yahoo itself isn't wrapped in this interface, to
    avoid touching a large amount of already-tested code purely for
    symmetry (see data_loader.load_ticker_bundle()'s own comment on this
    choice)."""
    if is_crypto_ticker(ticker):
        return CoinGeckoProvider()
    return None
