"""Discovery for the "Find a ticker" panel: what's active now, and what's
in a sector — both from real market data.

WHY "TRENDING" IS NOT A HARD-CODED LIST. The task illustrates this
feature as [Tesla] [Nvidia] [Apple], which is a static list wearing the
word "trending". Three famous names do not become trending by being
labelled so, and this app's standing rule is that a figure is never
invented. Yahoo's predefined screens give the real thing — most active by
volume, biggest gainers, biggest losers, as of today — so that is what
these chips show, with the ranking metric named on each one. The heading
says "most active by volume", not the vaguer "trending", because that is
the claim the data actually supports.

WHY THE CATEGORY FILTER AND THE DESCRIPTION BOX ARE THE SAME MACHINERY.
"Category filters: Tech, Healthcare, Finance" and "Show me biotech
stocks" are the same question asked two ways, and Yahoo's screener
answers both: sector for the categories, industry for the finer ones like
biotech. So the description box is a lookup from words to a screen, not a
second search path with its own quirks.

THE DESCRIPTION BOX MATCHES KEYWORDS, AND SAYS SO. It does not understand
language. "Show me biotech stocks" works because "biotech" is in the
table below; "companies that might benefit from rate cuts" will not, and
the UI says as much rather than quietly returning something adjacent and
letting it look like comprehension. Silently answering a question you did
not understand is worse than declining it.

CROSS-LISTINGS ARE DROPPED. A sector screen returns the OTC shadow of
every foreign name beside its primary listing — argenx came back as both
ARGX (NasdaqGS) and ARGNF (OTC Pink), CSL twice, UCB twice. Those are not
different companies and they push real results off a short list, so
anything on an OTC venue is filtered out. The existing name search warns
about this instead of filtering, which is right there: the user typed a
specific name and deserves everything matching it. Here the app chose the
list, so it should choose well.

Nothing here raises. Every entry point returns (results, error) and the
panel shows the error, because a discovery widget that fails must not
take down the sidebar it lives in.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import streamlit as st

from logging_setup import get_logger, log_event, log_exception
from screener import SECTORS

logger = get_logger("ticker_discovery")

# Yahoo pages these in 25s regardless of what we ask for, so the cap is
# applied after the fetch. Six chips is what fits the sidebar's width.
TRENDING_LIMIT = 6
SECTOR_LIMIT = 8
# Five, as the task asks. The symbol currently on screen is skipped when
# this list is drawn, so these are five OTHER tickers rather than four
# plus the one already filling the header.
RECENTS_LIMIT = 5

# Intraday figures, so a short life. Long enough that flipping between
# the three screens does not re-fetch, short enough that "most active
# today" is not yesterday's answer.
TRENDING_TTL_SECONDS = 600
# Sector membership shifts on a scale of quarters, not minutes.
SECTOR_TTL_SECONDS = 3600

# (key, label, what the ranking actually measures)
TRENDING_SCREENS: Tuple[Tuple[str, str, str], ...] = (
    ("most_actives", "Most active", "by share volume traded today"),
    ("day_gainers", "Gainers", "by percent gain today"),
    ("day_losers", "Losers", "by percent loss today"),
)
TRENDING_LABELS: Dict[str, str] = {k: lbl for k, lbl, _ in TRENDING_SCREENS}
TRENDING_BASIS: Dict[str, str] = {k: basis for k, _, basis in TRENDING_SCREENS}

# Venues whose listings are secondary shadows of a primary one elsewhere.
# Matched on the short code and on the display name, because Yahoo is not
# consistent about which it populates.
_OTC_CODES = frozenset({"PNK", "OQX", "OQB", "OTC", "PINK"})


@dataclass(frozen=True)
class Listing:
    """One row of a discovery screen. Every figure is Yahoo's, or None."""
    symbol: str
    name: str = ""
    exchange: str = ""
    price: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None

    @property
    def label(self) -> str:
        return f"{self.symbol} — {self.name}" if self.name else self.symbol

    def change_text(self) -> str:
        """Signed percent, or an explicit blank. Never 0.00% for missing."""
        if self.change_pct is None:
            return "not reported"
        return f"{self.change_pct:+.2f}%"


def _is_otc(row: dict) -> bool:
    code = str(row.get("exchange") or "").strip().upper()
    full = str(row.get("fullExchangeName") or "").strip().upper()
    return code in _OTC_CODES or "OTC" in full


def _to_listing(row: dict) -> Optional[Listing]:
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return None

    def number(key):
        value = row.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    volume = number("regularMarketVolume")
    return Listing(
        symbol=symbol,
        name=str(row.get("shortName") or row.get("longName") or "").strip(),
        exchange=str(row.get("fullExchangeName") or row.get("exchange") or "").strip(),
        price=number("regularMarketPrice"),
        change_pct=number("regularMarketChangePercent"),
        volume=int(volume) if volume is not None else None,
        market_cap=number("marketCap"),
    )


def _clean(rows, limit: int) -> Tuple[Listing, ...]:
    out, seen = [], set()
    for row in rows or []:
        if not isinstance(row, dict) or _is_otc(row):
            continue
        listing = _to_listing(row)
        if listing is None or listing.symbol in seen:
            continue
        seen.add(listing.symbol)
        out.append(listing)
        if len(out) >= limit:
            break
    return tuple(out)


def _screen(body, sort_field: Optional[str], label: str):
    """Run one Yahoo screen. Returns (rows, error); never raises.

    `body` is either a predefined screen's name or an EquityQuery. Both
    go through yf.screen, which is why they share this one path.
    """
    try:
        import yfinance as yf

        kwargs = {"count": 25}
        if sort_field:
            kwargs.update(sortField=sort_field, sortAsc=False)
        result = yf.screen(body, **kwargs)
    except Exception as e:
        log_exception(logger, "ticker_discovery.screen_failed",
                      section="ticker_discovery")
        return [], f"{label} is unavailable right now ({type(e).__name__})."

    rows = result.get("quotes") if isinstance(result, dict) else None
    if not rows:
        return [], f"Yahoo returned no results for {label.lower()}."
    return rows, None


@st.cache_data(ttl=TRENDING_TTL_SECONDS, show_spinner=False)
def _trending_cached(kind: str, limit: int):
    rows, error = _screen(kind, None, TRENDING_LABELS.get(kind, kind))
    if error:
        return (), error
    return _clean(rows, limit), None


def trending(kind: str = "most_actives", limit: int = TRENDING_LIMIT):
    """The live movers for one predefined screen. (listings, error)."""
    if kind not in TRENDING_LABELS:
        return (), f"Unknown screen “{kind}”."
    listings, error = _trending_cached(kind, limit)
    if not error:
        log_event(logger, logging.INFO, "ticker_discovery.trending",
                  screen=kind, results=len(listings))
    return listings, error


@st.cache_data(ttl=SECTOR_TTL_SECONDS, show_spinner=False)
def _by_field_cached(field: str, value: str, limit: int):
    try:
        from yfinance import EquityQuery
    except Exception as e:
        return (), f"Sector screening is unavailable ({type(e).__name__})."

    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("eq", [field, value]),
    ])
    # Largest first: on a list of eight, market cap is the ordering most
    # likely to surface names the reader recognises.
    rows, error = _screen(query, "intradaymarketcap", value)
    if error:
        return (), error
    return _clean(rows, limit), None


def by_sector(sector: str, limit: int = SECTOR_LIMIT):
    if sector not in SECTORS:
        return (), f"“{sector}” is not one of Yahoo's sectors."
    return _by_field_cached("sector", sector, limit)


def by_industry(industry: str, limit: int = SECTOR_LIMIT):
    return _by_field_cached("industry", industry, limit)


# --- the description box ------------------------------------------------------

@dataclass(frozen=True)
class Intent:
    """What a phrase was understood to mean. `field` is sector|industry."""
    field: str
    value: str
    label: str
    matched: str      # the word that triggered it, quoted back to the user


# Keyword -> (field, Yahoo's value, human label). Ordered longest-first at
# match time so "biotech" wins over "tech" in "biotech stocks" — the bug
# a naive substring scan would ship.
DESCRIPTION_TERMS: Tuple[Tuple[str, str, str, str], ...] = (
    ("biotech", "industry", "Biotechnology", "Biotechnology"),
    ("biotechnology", "industry", "Biotechnology", "Biotechnology"),
    ("pharma", "industry", "Drug Manufacturers - General", "Pharmaceuticals"),
    ("pharmaceutical", "industry", "Drug Manufacturers - General", "Pharmaceuticals"),
    ("semiconductor", "industry", "Semiconductors", "Semiconductors"),
    ("chip", "industry", "Semiconductors", "Semiconductors"),
    ("bank", "industry", "Banks - Diversified", "Banks"),
    ("airline", "industry", "Airlines", "Airlines"),
    ("automaker", "industry", "Auto Manufacturers", "Automakers"),
    ("carmaker", "industry", "Auto Manufacturers", "Automakers"),
    ("software", "industry", "Software - Infrastructure", "Software"),
    ("insurance", "industry", "Insurance - Diversified", "Insurance"),
    ("mining", "industry", "Gold", "Gold mining"),
    ("oil", "industry", "Oil & Gas Integrated", "Oil & gas"),
    ("retail", "industry", "Discount Stores", "Retail"),
    ("aerospace", "industry", "Aerospace & Defense", "Aerospace & defense"),
    ("defense", "industry", "Aerospace & Defense", "Aerospace & defense"),
    ("reit", "sector", "Real Estate", "Real Estate"),
    ("real estate", "sector", "Real Estate", "Real Estate"),
    ("healthcare", "sector", "Healthcare", "Healthcare"),
    ("health care", "sector", "Healthcare", "Healthcare"),
    ("health", "sector", "Healthcare", "Healthcare"),
    ("finance", "sector", "Financial Services", "Financial Services"),
    ("financial", "sector", "Financial Services", "Financial Services"),
    ("energy", "sector", "Energy", "Energy"),
    ("utility", "sector", "Utilities", "Utilities"),
    ("utilities", "sector", "Utilities", "Utilities"),
    ("industrial", "sector", "Industrials", "Industrials"),
    ("materials", "sector", "Basic Materials", "Basic Materials"),
    ("telecom", "sector", "Communication Services", "Communication Services"),
    ("media", "sector", "Communication Services", "Communication Services"),
    ("consumer", "sector", "Consumer Cyclical", "Consumer Cyclical"),
    ("tech", "sector", "Technology", "Technology"),
    ("technology", "sector", "Technology", "Technology"),
)


def interpret(text: str) -> Optional[Intent]:
    """Map a phrase to a screen, or None if no term is recognised.

    Longest term first, so "biotech stocks" resolves to Biotechnology
    rather than to Technology via the "tech" inside it. That collision is
    the whole reason this is ordered rather than a plain dict lookup.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    for term, field, value, label in sorted(
            DESCRIPTION_TERMS, key=lambda t: len(t[0]), reverse=True):
        if term in lowered:
            return Intent(field=field, value=value, label=label, matched=term)
    return None


def for_description(text: str):
    """(listings, error, intent). `intent` is None when nothing matched."""
    intent = interpret(text)
    if intent is None:
        return (), None, None
    if intent.field == "sector":
        listings, error = by_sector(intent.value)
    else:
        listings, error = by_industry(intent.value)
    return listings, error, intent
