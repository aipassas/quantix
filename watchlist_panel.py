"""Symbol navigation aids — the two ways to change the analyzed ticker
without retyping it.

1. The sidebar watchlist: a small, user-maintained list of tickers with a
   live quote line for each, rendered so it stays visible alongside
   whichever analysis panel is open.
2. The recently-viewed strip (record_recent below): an automatic
   most-recently-used list of symbols visited this session, rendered as
   chips under the symbol header. Curated by the user in the first case,
   accumulated by simply navigating in the second.

Deliberately reuses data_loader.load_ticker_bundle(deep=False), the exact
same shallow (info-only) fetch the Institutional Watchlist scan and the
Peer Comparison already use — so a ticker already loaded elsewhere this
session is served from that fetch's own 30-minute cache rather than
costing a new Yahoo call.

Day-change convention: computed here as (price - previous_close) /
previous_close, NOT read from Yahoo's own `regularMarketChangePercent`.
Yahoo's percent fields have historically been inconsistent about whether
they're a fraction or already-multiplied percent (the same class of
scaling ambiguity financial_standardization.py documents for
`debtToEquity`), and both inputs needed for the honest calculation are
right there in the same payload — so this computes it rather than trusting
a field whose scale can't be verified per-ticker.

Never-fabricate convention: a ticker Yahoo can't resolve, or one missing
either price leg, comes back with status != "ok" and None values, and the
panel renders it as an explicit unavailable row — never silently dropped
and never shown with a made-up 0.00.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st

from data_loader import load_ticker_bundle
from logging_setup import get_logger, log_event

logger = get_logger("watchlist_panel")


@dataclass(frozen=True)
class QuoteSnapshot:
    """One watchlist row. `status` is "ok" only when price AND
    previous_close both resolved, since day-change needs both."""
    ticker: str
    price: Optional[float] = None
    previous_close: Optional[float] = None
    change_pct: Optional[float] = None
    pe_ratio: Optional[float] = None
    status: str = "ok"          # "ok" | "unavailable"
    detail: str = ""

    @property
    def direction_icon(self) -> str:
        """Green/red/white ball matching the app's status-colour vocabulary.
        White for an unavailable quote or a genuinely flat day — never a
        directional colour for a direction that isn't there."""
        if self.change_pct is None:
            return "⚪"
        if self.change_pct > 0:
            return "🟢"
        if self.change_pct < 0:
            return "🔴"
        return "⚪"


def parse_tickers(raw: str) -> Tuple[str, ...]:
    """Split a comma/space-separated string into deduplicated uppercase
    ticker symbols, preserving the order they were typed in."""
    parts = [p.strip().upper() for p in raw.replace(",", " ").split()]
    return tuple(dict.fromkeys(p for p in parts if p))


def add_ticker(current: Tuple[str, ...], raw: str, max_tickers: int) -> Tuple[Tuple[str, ...], Optional[str]]:
    """Append parsed ticker(s) to `current`, returning (new_list, error).

    Returns the list UNCHANGED with an error message when the input is
    empty, already present, or would exceed `max_tickers` — the caller
    surfaces that message rather than silently no-op'ing, so a click that
    appears to do nothing always has a stated reason.
    """
    incoming = parse_tickers(raw)
    if not incoming:
        return current, "Enter a ticker symbol first."

    already = [t for t in incoming if t in current]
    if already:
        return current, f"Already on the watchlist: {', '.join(already)}."

    room = max_tickers - len(current)
    if room <= 0:
        return current, f"Watchlist is full ({max_tickers} max) — remove one first."
    if len(incoming) > room:
        return current, f"Only room for {room} more ticker(s) ({max_tickers} max)."

    return current + incoming, None


def remove_ticker(current: Tuple[str, ...], ticker: str) -> Tuple[str, ...]:
    return tuple(t for t in current if t != ticker)


def record_recent(current: Tuple[str, ...], ticker: str, max_recent: int) -> Tuple[str, ...]:
    """Most-recently-used list with `ticker` moved to the front.

    Idempotent by design: Streamlit re-runs the whole script on every
    widget interaction (a slider nudge, a tab click), so this is called
    constantly with the SAME ticker. Re-recording the current symbol must
    therefore be a no-op rather than duplicating it or churning the order
    — which is why this moves-to-front and dedupes instead of appending.

    Returns `current` unchanged for an empty ticker, and an empty tuple
    for a non-positive `max_recent`, rather than raising.
    """
    if max_recent <= 0:
        return ()
    if not ticker:
        return current
    rest = tuple(t for t in current if t != ticker)
    return (ticker,) + rest[:max_recent - 1]


@st.cache_data(ttl=300, show_spinner=False)
def _load_quote(ticker: str) -> QuoteSnapshot:
    """One ticker's quote line, from the shared shallow bundle.

    Cached for 5 minutes rather than the underlying info fetch's own 30
    minutes: this is the shortest-lived, most quote-like view of that
    data, and a cheap cache hit on the already-cached bundle underneath
    costs nothing when it does re-run.
    """
    try:
        bundle = load_ticker_bundle(ticker, deep=False)
    except Exception:
        # load_ticker_bundle handles routine failures internally via
        # bundle.errors, so anything raised here is genuinely unexpected.
        log_event(logger, logging.WARNING, "quote.error", ticker=ticker)
        return QuoteSnapshot(ticker=ticker, status="unavailable", detail="unexpected error loading quote")

    if not bundle.info:
        return QuoteSnapshot(ticker=ticker, status="unavailable", detail="; ".join(bundle.errors) or "no quote data returned")

    info = bundle.info
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    pe_ratio = info.get("trailingPE")

    if price is None or not previous_close:
        return QuoteSnapshot(
            ticker=ticker, price=price, previous_close=previous_close, pe_ratio=pe_ratio,
            status="unavailable", detail="quote missing price or previous close",
        )

    change_pct = (price - previous_close) / previous_close * 100
    return QuoteSnapshot(
        ticker=ticker, price=price, previous_close=previous_close,
        change_pct=change_pct, pe_ratio=pe_ratio, status="ok",
    )


def load_quote_snapshots(tickers: Tuple[str, ...]) -> Tuple[QuoteSnapshot, ...]:
    """Quote lines for every watchlist ticker, in the caller's order."""
    return tuple(_load_quote(t) for t in tickers)
