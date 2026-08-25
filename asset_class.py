"""What KIND of thing is this symbol, and which analyses actually apply.

THE PROBLEM THIS EXISTS TO FIX. Quantix was built for equities, and every
panel assumed one. Nothing in the analysis path branched on asset type —
`quoteType` appeared twice in the whole codebase, both times for display.
So typing BTC-USD produced a bundle marked valid, with pe=None,
sector=None and no statements, and then a discounted cash flow, an
eight-point fundamental scorecard and a sector-percentile ranking were
run against it anyway. A cryptocurrency has no cash flows to discount and
no sector to be ranked within; presenting those panels as merely "not
reported" implies the question was sensible and the data was missing. It
was not: the question does not apply.

THE FOUNDATION FOR THE PHASED ROLLOUT. This is the spine every later
asset-class phase hangs off — ETF holdings, bond duration, on-chain
metrics all need to know what they are looking at first. Adding a class
means adding one row here and declaring what it supports; adding an
ANALYSIS means adding one capability and saying which classes have it.
Neither requires hunting through finance.py for assumptions.

WHAT YAHOO ALONE CAN AND CANNOT DO, measured rather than assumed. It
returns price history and a quoteType for equities, ETFs, crypto,
currencies, futures and indices, and for ETFs it also returns top
holdings, sector weightings and a fund overview. It does NOT return bond
yield-to-maturity or duration, on-chain crypto metrics, or a futures
curve across contract months — those need sources this app does not have
credentials for, and MISSING_SOURCES records that so a later phase does
not rediscover it.

CAPABILITIES ARE DECLARED, NOT INFERRED. It would be tempting to decide
"if pe_ratio is None, hide the valuation panel". That conflates a metric
this issuer did not report with one that cannot exist for this kind of
instrument — and it would hide a genuine data gap on an equity, which is
exactly the disclosure this app is built around.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

# --- the classes --------------------------------------------------------------

EQUITY = "equity"
ETF = "etf"
CRYPTO = "crypto"
FOREX = "forex"
FUTURE = "future"
INDEX = "index"
UNKNOWN = "unknown"

# Yahoo's quoteType -> our class. Yahoo's vocabulary is wider than ours
# (MUTUALFUND, OPTION, ...) and is matched case-insensitively because it
# is not consistent about case across endpoints.
_QUOTE_TYPES: Dict[str, str] = {
    "equity": EQUITY,
    "etf": ETF,
    "mutualfund": ETF,          # same analysis shape: a basket with holdings
    "cryptocurrency": CRYPTO,
    "currency": FOREX,
    "future": FUTURE,
    "index": INDEX,
}


# --- the capabilities ---------------------------------------------------------

# Each is one analysis surface the app offers. A class either supports it
# or does not; there is no "maybe".
FUNDAMENTALS = "fundamentals"      # statements, ratios, the 8-point scorecard
DCF = "dcf"                        # discounted cash flow / intrinsic value
SECTOR_PERCENTILE = "sector_percentile"
PEERS = "peers"                    # competitor benchmarking
DIVIDENDS = "dividends"
TECHNICALS = "technicals"          # price-derived: SMA, RSI, MACD, Bollinger
RISK = "risk"                      # return-derived: VaR, Sharpe, drawdown
SIMULATION = "simulation"          # Monte Carlo over returns
HOLDINGS = "holdings"              # what a basket contains
ON_CHAIN = "on_chain"              # settlement activity on a public ledger

ALL_CAPABILITIES: Tuple[str, ...] = (
    FUNDAMENTALS, DCF, SECTOR_PERCENTILE, PEERS, DIVIDENDS,
    TECHNICALS, RISK, SIMULATION, HOLDINGS, ON_CHAIN,
)


@dataclass(frozen=True)
class AssetClassSpec:
    key: str
    label: str
    supports: Tuple[str, ...]
    # Why the unsupported analyses do not apply — shown to the reader in
    # place of an empty panel, so "nothing here" reads as an answer
    # rather than a failure.
    absence_reason: str = ""
    example: str = ""


SPECS: Tuple[AssetClassSpec, ...] = (
    AssetClassSpec(
        EQUITY, "Stock",
        (FUNDAMENTALS, DCF, SECTOR_PERCENTILE, PEERS, DIVIDENDS,
         TECHNICALS, RISK, SIMULATION),
        "", "AAPL"),
    AssetClassSpec(
        ETF, "ETF / fund",
        (DIVIDENDS, TECHNICALS, RISK, SIMULATION, HOLDINGS),
        "A fund has no income statement of its own — it holds other "
        "companies. Its value comes from what it holds, so the "
        "company-level scorecard and a discounted cash flow do not apply.",
        "SPY"),
    AssetClassSpec(
        CRYPTO, "Cryptocurrency",
        (TECHNICALS, RISK, SIMULATION, ON_CHAIN),
        "A cryptocurrency has no issuer, no filings and no cash flows, so "
        "there is nothing to discount and no sector to rank it within. "
        "Price-derived analysis still applies in full, and a public "
        "ledger supports a valuation read no equity has: what the "
        "network actually settles.",
        "BTC-USD"),
    AssetClassSpec(
        FOREX, "Currency pair",
        (TECHNICALS, RISK, SIMULATION),
        "A currency pair is a relative price between two currencies. It "
        "has no earnings, no balance sheet and no dividend.",
        "EURUSD=X"),
    AssetClassSpec(
        FUTURE, "Futures contract",
        (TECHNICALS, RISK, SIMULATION),
        "A futures contract is a dated claim on a deliverable, not a "
        "business. Valuation depends on the forward curve and carry, "
        "which this build does not yet source.",
        "GC=F"),
    AssetClassSpec(
        INDEX, "Index",
        (TECHNICALS, RISK, SIMULATION),
        "An index is a calculated level, not a security you can own. It "
        "has no financials of its own.",
        "^GSPC"),
    AssetClassSpec(
        UNKNOWN, "Unrecognised instrument",
        (TECHNICALS, RISK, SIMULATION),
        "This symbol's type was not reported, so only price-derived "
        "analysis is offered. Anything depending on filings is withheld "
        "rather than guessed at.",
        ""),
)
SPECS_BY_KEY: Dict[str, AssetClassSpec] = {s.key: s for s in SPECS}


# What this build cannot source for a class, recorded so a later phase
# does not have to rediscover it. Yahoo is the only provider wired up.
MISSING_SOURCES: Dict[str, Tuple[str, ...]] = {
    ETF: ("tracking error and down-capture are not in Yahoo's fund data. "
          "The expense ratio IS — twice, 100x apart: a percent in "
          "info.netExpenseRatio and a fraction in funds_data."
          "fund_operations. Top holdings and sector weightings are "
          "present for equity funds and absent by nature for bond and "
          "commodity ones (SPY and QQQ disclose them; TLT and GLD do not)",),
    CRYPTO: ("MVRV, realized cap, whale concentration and exchange "
             "reserves — all need UTXO-level or address-labelled data "
             "that no free provider publishes; Glassnode or CoinMetrics "
             "would",
             "on-chain metrics for any coin OTHER than Bitcoin — "
             "blockchain.info indexes the Bitcoin chain only, so every "
             "other coin has price and supply but no chain data",
             "social sentiment — CoinGecko still returns the fields and "
             "they are empty, reading zero for Bitcoin and Dogecoin "
             "alike, which is a retired field rather than a quiet "
             "community",
             "order-book depth — Binance and Kraken quote the prices "
             "CoinGecko already aggregates, and their depth endpoints "
             "need per-exchange integration",),
    FUTURE: ("the forward curve needs quotes for every contract month; "
             "Yahoo returns only the front month per symbol",),
    INDEX: ("treasury yields arrive as an index level (^TNX); "
            "yield-to-maturity, duration and convexity need FRED or a "
            "bond data provider",),
    FOREX: ("interest-rate parity needs policy rates for both currencies; "
            "no rates provider is wired up",),
}


# --- classification -----------------------------------------------------------

def classify(info: Optional[dict], symbol: str = "") -> str:
    """The asset class for a Yahoo info dict.

    Falls back to UNKNOWN rather than to EQUITY. Defaulting to equity is
    what produced the original bug: an unrecognised instrument would get
    the full stock treatment on the strength of a missing field.
    """
    quote_type = ""
    if isinstance(info, dict):
        quote_type = str(info.get("quoteType") or "").strip().lower()
    return _QUOTE_TYPES.get(quote_type, UNKNOWN)


def spec(asset_class: str) -> AssetClassSpec:
    return SPECS_BY_KEY.get(asset_class, SPECS_BY_KEY[UNKNOWN])


def supports(asset_class: str, capability: str) -> bool:
    return capability in spec(asset_class).supports


def label(asset_class: str) -> str:
    return spec(asset_class).label


def with_article(asset_class: str) -> str:
    """The class label with the right indefinite article, e.g. "an ETF /
    fund", "a stock", "an index".

    Naively lowercasing the label produced "a etf / fund" on the data
    quality badge — wrong article, and an acronym written as a word. The
    article is chosen on the leading letter, which gets "an ETF" right
    because E is read as a vowel, and an all-caps first word keeps its
    case.
    """
    text = label(asset_class)
    if not text:
        return "an instrument"
    if not text.split()[0].isupper():
        text = text[0].lower() + text[1:]
    return ("an " if text[0].upper() in "AEIOU" else "a ") + text


def unavailable_note(asset_class: str, capability: str) -> str:
    """What to show where an analysis would have gone.

    Names the class and the reason, because "not available" alone reads
    as a fetch that failed — and a reader who thinks the data is merely
    missing will go looking for it.
    """
    detail = spec(asset_class).absence_reason
    return (f"Not applicable to {with_article(asset_class)}. {detail}").strip()


def missing_sources(asset_class: str) -> Tuple[str, ...]:
    return MISSING_SOURCES.get(asset_class, ())
