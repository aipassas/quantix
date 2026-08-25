"""Ticker autocomplete — a searchable list of symbols the app already
knows, plus company-name lookup against Yahoo for everything else.

WHAT STREAMLIT CAN AND CANNOT DO HERE, since it shapes the whole design:
there is no true per-keystroke autocomplete. Streamlit doesn't round-trip
on every keystroke, and a <script> injected through st.markdown never
executes (proven in this codebase already — see onboarding.py). So a
typeahead that queries a server as you type is not available without
building a full bidirectional component.

Two things ARE available, and this module serves both:

1. A searchable st.selectbox filters its OPTIONS client-side as you type,
   instantly and with no network call. build_universe() assembles those
   options — every ticker the app already knows about (the institutional
   baskets, the user's watchlists, their favorites and recents), labelled
   "AAPL — Apple Inc." so a half-remembered name is enough to find it.
   The selectbox is rendered with accept_new_options=True, so a symbol
   that isn't in the list can still simply be typed: the suggestion list
   is a shortcut, never a restriction.

2. search_symbols() queries Yahoo by company name on submit, which is one
   round-trip rather than one per keystroke. That's what makes genuine
   discovery possible ("apple", "nvid") across equities, ETFs and crypto
   rather than only over tickers already on screen.

Both degrade rather than fail: a name that can't be resolved falls back to
showing the bare ticker, and a failed search returns an error string for
the UI to show instead of raising.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import streamlit as st

from data_loader import load_ticker_bundle
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("ticker_search")

MAX_SEARCH_RESULTS = 8


@dataclass(frozen=True)
class SymbolMatch:
    symbol: str
    name: str = ""
    quote_type: str = ""
    exchange: str = ""

    @property
    def label(self) -> str:
        """What the dropdown shows. Falls back to the bare symbol when no
        name resolved, rather than rendering a dangling separator."""
        return f"{self.symbol} — {self.name}" if self.name else self.symbol

    @property
    def detail(self) -> str:
        """Type/exchange line for search results. Yahoo returns foreign
        cross-listings and tokenized proxies alongside the primary listing
        (searching "apple" surfaces APC.DE and D90.F next to AAPL), so
        showing where a match trades is what lets someone tell them apart
        rather than picking the wrong one."""
        bits = [b for b in (self.quote_type, self.exchange) if b]
        return " · ".join(bits)


# Cached, unlike search_symbols itself. The failure path calls this on
# EVERY rerun while an unresolvable ticker is loaded, and this page reruns
# constantly (polling fragments), so an uncached call would mean several
# Yahoo search requests a minute for a symbol that has not changed.
# A day is right: which exchanges a fund lists on is not news.
@st.cache_data(ttl=86400, show_spinner=False)
def suggest_alternatives(symbol: str, limit: int = 6,
                         searcher: Optional[Callable] = None
                         ) -> Tuple[Tuple["SymbolMatch", ...], Optional[str]]:
    """Real listings for a symbol that did not resolve.

    The case this exists for: a fund quoted by its bare ticker everywhere
    except this data source, which wants the exchange suffix. VWCE does
    not resolve; VWCE.DE, VWCE.MI and VWCE.AS all do, and Yahoo's own
    search maps one to the other — measured, "VWCE" returns seven
    listings with VWCE.DE first.

    The symbol itself is filtered out of the results: offering someone
    the exact thing that just failed is noise.

    `searcher` is injectable for the same reason search_symbols' is —
    so the tests exercise this function rather than reimplementing its
    filter and proving only that the test's own copy works.
    """
    symbol = (symbol or "").strip().upper()
    if len(symbol) < 2:
        return (), None
    matches, error = search_symbols(symbol, max_results=limit + 2,
                                    searcher=searcher)
    matches = tuple(m for m in matches if m.symbol.upper() != symbol)
    return matches[:limit], error


@st.cache_data(ttl=86400, show_spinner=False)
def name_for(ticker: str) -> str:
    """A ticker's company name, or "" if it can't be resolved.

    Uses the same shallow (info-only) bundle the Institutional Watchlist
    scan and the sidebar quotes already fetch, so by the time the sidebar
    renders these are usually served from that fetch's own cache rather
    than costing a new call. Never raises — an unresolvable name simply
    means the dropdown shows the bare ticker.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return ""
    try:
        bundle = load_ticker_bundle(ticker, deep=False)
        info = bundle.info or {}
        return str(info.get("longName") or info.get("shortName") or "")
    except Exception:
        log_exception(logger, "ticker_search.name_lookup_failed", section="ticker_search")
        return ""


def build_universe(tickers: Sequence[str], resolve_names: bool = True) -> Tuple[SymbolMatch, ...]:
    """The dropdown's option list, deduplicated and alphabetical.

    Order is stable and alphabetical rather than by recency: this list is
    typed into, not scanned, so a symbol jumping position between renders
    would be actively unhelpful.
    """
    seen = []
    for raw in tickers:
        symbol = (raw or "").strip().upper()
        if symbol and symbol not in seen:
            seen.append(symbol)
    matches = [SymbolMatch(symbol=s, name=name_for(s) if resolve_names else "") for s in sorted(seen)]
    return tuple(matches)


def _default_searcher(query: str, max_results: int):
    """Yahoo symbol search. Imported lazily so a yfinance version without
    Search doesn't break module import for everything else."""
    import yfinance as yf
    return yf.Search(query, max_results=max_results).quotes


def search_symbols(
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
    searcher: Optional[Callable] = None,
) -> Tuple[Tuple[SymbolMatch, ...], Optional[str]]:
    """Search Yahoo by company name or partial ticker.

    Returns (matches, error). Never raises: a network failure or an API
    shape change comes back as an error string for the UI to show, because
    a broken search must not take down the sidebar it lives in.

    `searcher` is injectable so tests never touch the network.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return (), "Type at least two characters to search."

    searcher = searcher or _default_searcher
    try:
        raw = searcher(query, max_results)
    except Exception as e:
        log_exception(logger, "ticker_search.search_failed", section="ticker_search")
        return (), f"Symbol search is unavailable right now ({type(e).__name__})."

    matches = []
    seen = set()
    for row in raw or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        matches.append(SymbolMatch(
            symbol=symbol,
            name=str(row.get("shortname") or row.get("longname") or "").strip(),
            quote_type=str(row.get("quoteType") or "").strip(),
            exchange=str(row.get("exchange") or "").strip(),
        ))

    if not matches:
        return (), f'No symbols matched "{query}".'
    log_event(logger, logging.INFO, "ticker_search.searched", results=len(matches))
    return tuple(matches[:max_results]), None


def symbol_from_label(label: str) -> str:
    """The ticker back out of a dropdown label.

    The selectbox accepts new options, so `label` may be a label this
    module produced ("AAPL — Apple Inc.") or something the user typed
    freehand ("aapl"). Both have to resolve to a usable symbol, so this
    splits on the separator only when it's actually present.
    """
    text = (label or "").strip()
    if "—" in text:
        text = text.split("—", 1)[0]
    return text.strip().upper()
