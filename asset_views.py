"""What the PAGE looks like for each kind of instrument.

asset_class.py answers "what is this symbol, and which analyses apply".
This module answers the next question down: given that answer, what
should the reader actually SEE — which pill is lit, what the search box
says, which header stats mean anything, what the tabs are called, and
what badge the watchlist row carries. Keeping that here rather than in
finance.py means adding a class is one row, and means the tests can read
the mapping without importing a 7000-line script.

BONDS AND COMMODITIES ARE NOT SYMMETRIC WITH THE OTHERS, and pretending
otherwise would be the same category error this codebase keeps paying
for. The task asks for six pills — Stocks, ETFs, Bonds, Crypto,
Commodities, Forex — but:

  - Commodities IS a real class here. GC=F classifies as a futures
    contract and gets the price-derived analysis in full.
  - Bonds is NOT. There is no bond asset class: a bond FUND (TLT, AGG,
    BND) classifies as an ETF and is analysed as one, and a Treasury
    yield arrives as an index level (^TNX). Yield-to-maturity, duration
    and credit spreads need a source this build has no credentials for —
    asset_class.MISSING_SOURCES has said so since the spine was added,
    and PHASE 2 is where that work lives.

So the Bonds pill exists, because a reader looking for bonds should find
something, but it routes to the instruments that DO work and says what
is not built yet. A pill that silently did nothing would be worse than
no pill.

THE PILL IS A LENS, NOT A MODE. Selecting one filters what the search
suggests and which examples are offered; it does not override what a
symbol actually IS. Typing SPY while "Stocks" is lit still analyses SPY
as a fund, because the classification comes from the data and not from a
UI toggle — the alternative is a page confidently applying the wrong
analysis because a pill was left selected.

TABS ARE RE-LABELLED, NOT REMOVED, and that is deliberate. finance.py
unpacks its tab objects positionally and each `with tab_x:` block is
several hundred lines; dropping a tab for one class would renumber the
rest, and the ⌘1–⌘8 shortcuts are bound to those positions. Re-labelling
gets the reader the right words ("Holdings & Fund Profile" rather than
"Fundamentals & Valuation") with none of that risk, and the panels
inside already render an explicit not-applicable note where an analysis
does not apply. The one structural change is additive: a Fund Comparison
tab appended at the END for funds only, which leaves every existing
index untouched.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import asset_class

# --- the header strip ---------------------------------------------------------

# Which quick-stat keys mean something for each class. A stat absent here
# is not "not reported" for that class, it is not a question you can ask
# — an ETF has no net margin, and offering one in the header is how the
# data-quality badge came to grade funds on corporate filings.
HEADER_STATS: Dict[str, Tuple[str, ...]] = {
    asset_class.EQUITY: (
        "price", "change_pct", "pe_ratio", "market_cap",
        "dividend_yield_pct", "beta", "price_to_book", "net_margin",
        "return_on_equity", "debt_to_equity", "current_ratio", "sector",
    ),
    asset_class.ETF: (
        # Beta is deliberately absent: standardized.beta never resolves
        # for a fund (measured — SPY and TLT both read "Not reported"),
        # so offering it would guarantee a permanently-blank stat, which
        # is the thing this mapping exists to remove. Yahoo does carry
        # beta3Year in the info dict; wiring that source is PHASE 1.3.
        "price", "change_pct", "expense_ratio_pct", "net_assets",
        "dividend_yield_pct", "fund_category", "fund_pe",
    ),
    asset_class.CRYPTO: ("price", "change_pct", "market_cap"),
    asset_class.FOREX: ("price", "change_pct"),
    asset_class.FUTURE: ("price", "change_pct"),
    asset_class.INDEX: ("price", "change_pct"),
    asset_class.UNKNOWN: ("price", "change_pct"),
}


@dataclass(frozen=True)
class AssetView:
    key: str                 # an asset_class key, or a pseudo-class (bond)
    pill: str                # the selector label
    badge: str               # one letter, for a watchlist row
    placeholder: str         # the search box's prompt
    examples: Tuple[str, ...] = ()
    note: str = ""           # what is and is not built for this class
    asset_class_key: str = ""   # the class it actually resolves to


# The selector, in the task's own order. `key` is what the app stores;
# `asset_class_key` is what a symbol of this kind classifies AS.
VIEWS: Tuple[AssetView, ...] = (
    AssetView(
        asset_class.EQUITY, "Stocks", "S", "Search stocks — ticker or name",
        ("AAPL", "MSFT", "NVDA"),
        "Full coverage: filings, the scorecard, a discounted cash flow, "
        "sector percentiles and peers.",
        asset_class.EQUITY),
    AssetView(
        asset_class.ETF, "ETFs & Funds", "E", "Search ETFs — ticker, name or theme",
        ("SPY", "QQQ", "VTI"),
        "Holdings, cost, style and sector momentum in place of company "
        "filings. Premium/discount to NAV is not available from this data "
        "source.",
        asset_class.ETF),
    AssetView(
        "bond", "Bonds", "B", "Search bond funds — ticker or name",
        ("TLT", "AGG", "BND"),
        "Bond ANALYTICS are not built yet — yield-to-maturity, duration "
        "and credit spreads need a data source this build has no "
        "credentials for. A bond FUND is analysed as a fund, which "
        "covers price, risk and cost; a Treasury yield arrives as an "
        "index level (^TNX).",
        asset_class.ETF),
    AssetView(
        asset_class.CRYPTO, "Crypto", "C", "Search crypto — e.g. BTC-USD",
        ("BTC-USD", "ETH-USD", "SOL-USD"),
        "Price-derived analysis in full. There are no filings to value "
        "and no on-chain metrics in this build.",
        asset_class.CRYPTO),
    AssetView(
        asset_class.FUTURE, "Commodities", "F", "Search commodities — e.g. GC=F",
        ("GC=F", "CL=F", "SI=F"),
        "Front-month contracts only. The forward curve needs a quote per "
        "contract month, which this source does not return.",
        asset_class.FUTURE),
    AssetView(
        asset_class.FOREX, "Forex", "X", "Search currency pairs — e.g. EURUSD=X",
        ("EURUSD=X", "GBPUSD=X", "USDJPY=X"),
        "Price-derived analysis in full. Interest-rate parity needs a "
        "policy-rate source that is not wired up.",
        asset_class.FOREX),
)

VIEWS_BY_KEY: Dict[str, AssetView] = {v.key: v for v in VIEWS}

DEFAULT_VIEW = asset_class.EQUITY

# Classes a symbol can classify as that are not offered as a pill. An
# index is a real class and is analysed, it is simply not something a
# reader browses FOR — they arrive at ^GSPC by typing it.
_FALLBACK_BADGES: Dict[str, str] = {
    asset_class.INDEX: "I",
    asset_class.UNKNOWN: "?",
}


def view(key: Optional[str]) -> AssetView:
    """The view for a pill key, falling back to stocks."""
    return VIEWS_BY_KEY.get(key or "", VIEWS_BY_KEY[DEFAULT_VIEW])


def pill_labels() -> Tuple[str, ...]:
    return tuple(v.pill for v in VIEWS)


def key_for_pill(label: str) -> str:
    for v in VIEWS:
        if v.pill == label:
            return v.key
    return DEFAULT_VIEW


def badge(asset_class_key: str) -> str:
    """The one-letter badge for a CLASSIFIED symbol.

    Takes an asset_class key, not a pill key: a watchlist row is badged
    by what the symbol is, not by which pill happened to be lit when it
    was added. Bond funds therefore badge as E, which is what they are —
    the app has no separate bond analysis to promise otherwise.
    """
    for v in VIEWS:
        if v.asset_class_key == asset_class_key and v.key == asset_class_key:
            return v.badge
    return _FALLBACK_BADGES.get(asset_class_key, "?")


def badge_title(asset_class_key: str) -> str:
    """Hover text for a badge, so a bare letter is never a puzzle."""
    return asset_class.label(asset_class_key)


def header_stats(asset_class_key: str) -> Tuple[str, ...]:
    """Which quick-stat keys apply to a classified symbol."""
    return HEADER_STATS.get(asset_class_key, HEADER_STATS[asset_class.UNKNOWN])


def applies(stat_key: str, asset_class_key: str) -> bool:
    return stat_key in header_stats(asset_class_key)


# --- the tab strip ------------------------------------------------------------

# The eight panels finance.py builds, in order, with the label each class
# should see. A class missing from a row inherits the base label.
BASE_TABS: Tuple[str, ...] = (
    "Overview", "Chart Workspace", "Fundamentals & Valuation",
    "Risk & Technicals", "Monte Carlo & Seasonality", "Smart Money & Peers",
    "Portfolio", "CIO Tear Sheet",
)

_TAB_OVERRIDES: Dict[str, Dict[int, str]] = {
    asset_class.ETF: {
        2: "Holdings & Fund Profile",
        3: "Risk & Fund Technicals",
        5: "Peers & Flows",
    },
    asset_class.CRYPTO: {2: "Valuation (n/a)", 5: "Peers (n/a)"},
    asset_class.FOREX: {2: "Valuation (n/a)", 5: "Peers (n/a)"},
    asset_class.FUTURE: {2: "Valuation (n/a)", 5: "Peers (n/a)"},
    asset_class.INDEX: {2: "Valuation (n/a)", 5: "Peers (n/a)"},
    asset_class.UNKNOWN: {2: "Valuation (n/a)", 5: "Peers (n/a)"},
}

COMPARISON_TAB = "Fund Comparison"


def tab_labels(asset_class_key: str) -> Tuple[str, ...]:
    """The tab strip for a classified symbol.

    The comparison tab is APPENDED, never inserted, so every existing tab
    keeps its index and the ⌘1–⌘8 bindings keep pointing at the panel
    they always did.
    """
    overrides = _TAB_OVERRIDES.get(asset_class_key, {})
    labels = [overrides.get(i, name) for i, name in enumerate(BASE_TABS)]
    if has_comparison(asset_class_key):
        labels.append(COMPARISON_TAB)
    return tuple(labels)


def has_comparison(asset_class_key: str) -> bool:
    """Comparison is a fund feature: it compares cost, holdings overlap
    and tracking, none of which a single equity or a currency has."""
    return asset_class_key == asset_class.ETF


def search_placeholder(pill_key: Optional[str]) -> str:
    return view(pill_key).placeholder
