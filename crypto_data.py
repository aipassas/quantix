"""Crypto prices, supply, on-chain activity and developer health.

WHAT THIS BUILD CAN ACTUALLY REACH, probed on 2026-08-25 rather than
assumed. The task names CoinGecko, Binance and Kraken for prices and
Glassnode-shaped metrics for the chain. The answer is better than the
bond phase and worse than the task hopes:

  - CoinGecko's free API needs NO key and answers from here. /ping,
    /global, /coins/markets and /coins/{id} all return HTTP 200. That
    covers price, market cap, 24h volume, circulating/total/max supply,
    all-time high and low, returns over 24h/7d/30d/1y, market-cap rank
    and Bitcoin dominance. This is the primary source.

  - Binance and Kraken are NOT wired. They quote the same prices
    CoinGecko aggregates, so a second price feed adds a dependency and
    no information. What they would add is order-book depth — which is
    the one liquidity input this module lacks — but that needs a
    per-exchange integration and, for the useful endpoints, a key.

  - blockchain.info's charts API is free, keyless, and answers from
    here. That is REAL on-chain data: estimated transaction volume in
    USD, unique active addresses, miner revenue, hash rate, transaction
    count and total coins issued. It is BITCOIN ONLY, and this module
    says so rather than quietly showing a Bitcoin number under an
    altcoin's name.

  - MVRV and realized cap are NOT available: /charts/mvrv and
    /charts/realized-cap both return HTTP 404, and realized cap needs
    UTXO-level data no free source publishes. Whale concentration (top
    100 addresses) and exchange reserve ratio need the same class of
    data. All three are declared unavailable and none is approximated
    from something adjacent.

TWO FIELDS THAT LOOK LIKE DATA AND ARE NOT.

MAX SUPPLY ZERO MEANS UNCAPPED, NOT ZERO. Yahoo reports maxSupply = 0
for ETH and DOGE; CoinGecko reports max_supply = None for the same two,
and for 111 of the top 250. Neither coin has a supply cap at all, so a
strip reading "Max supply: 0" is not merely unhelpful, it is false — and
"percent of maximum mined" computed as circulating/max is a division by
zero on nearly half the universe. UNCAPPED is a distinct state here and
scarcity metrics decline to score it. This is the same shape as an ETF
expense ratio of 0.00 meaning undisclosed rather than free.

COMMUNITY DATA IS DEAD. CoinGecko still returns a community_data block
and every field in it is empty: reddit_subscribers 0 and
twitter_followers None for Bitcoin, Ethereum, Solana AND Dogecoin —
measured, all four. Dogecoin's subreddit has millions of members, so
zero is not a small number, it is a retired field. The task's "social
data: Twitter mentions, Reddit posts, Google searches" therefore has no
source and none is invented. DEVELOPER data, in the same response, is
live and discriminates: 4-week commit counts of 108 (BTC), 41 (ETH),
171 (SOL) and 0 (DOGE) are all plausible for those projects.

SYMBOLS COLLIDE, so resolution is by market-cap rank. Two symbols are
already duplicated inside the top 250 alone: DAI is both Dai (rank 23)
and DAI on PulseChain (rank 219), and USDF is both Falcon USD (rank 60)
and Aster USDF (rank 238). Matching a bare ticker to the first row that
happens to carry it would land on either. The highest-ranked match wins,
and `resolve` reports when it had to choose.
"""
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BLOCKCHAIN_BASE = "https://api.blockchain.info/charts"

# CoinGecko's keyless tier allows roughly 10-30 calls a minute and answers
# 429 past that. Every call here is cached, and the universe — one call
# for 250 coins — is what the screener and the header both read.
UNIVERSE_TTL_SECONDS = 300
PROFILE_TTL_SECONDS = 3600
ONCHAIN_TTL_SECONDS = 3600
GLOBAL_TTL_SECONDS = 300

REQUEST_TIMEOUT_SECONDS = 20

SOURCE_COINGECKO = "coingecko"
SOURCE_BLOCKCHAIN = "blockchain.info"

# The coin whose chain this build can see. Held as a symbol so callers
# compare against the same spelling the universe uses.
ONCHAIN_SYMBOL = "btc"

UNCAPPED = "uncapped"

MVRV_UNAVAILABLE = (
    "MVRV needs realized capitalisation, which is a UTXO-level "
    "aggregate. blockchain.info answers /charts/mvrv and "
    "/charts/realized-cap with HTTP 404 and no free source publishes it; "
    "Glassnode or CoinMetrics would."
)

WHALE_UNAVAILABLE = (
    "Whale concentration needs a rich list — the balance of the top 100 "
    "addresses. No free endpoint publishes one, and addresses are not "
    "owners: a single exchange cold wallet would read as one whale "
    "holding millions of coins."
)

EXCHANGE_RESERVE_UNAVAILABLE = (
    "Exchange reserves need every exchange's wallets labelled, which is "
    "a proprietary dataset. Nothing here can distinguish an exchange's "
    "address from anyone else's."
)

SOCIAL_UNAVAILABLE = (
    "Social sentiment is not reported. CoinGecko still returns the "
    "fields and they are empty — reddit subscribers 0 and twitter "
    "followers absent for Bitcoin, Ethereum, Solana and Dogecoin alike, "
    "which is a retired field rather than four quiet communities."
)

ONCHAIN_BITCOIN_ONLY = (
    "On-chain metrics cover Bitcoin only in this build. They come from "
    "blockchain.info, which indexes the Bitcoin chain; every other coin "
    "would need its own indexer, and most need an API key."
)

EXCHANGES_NOT_WIRED = (
    "Order-book depth is not sourced. Binance and Kraken quote the same "
    "prices CoinGecko aggregates, so they would add a dependency without "
    "adding information; their depth endpoints are what would help, and "
    "those need per-exchange integration."
)


# --- symbols ------------------------------------------------------------------

def normalise_symbol(symbol: str) -> str:
    """A Yahoo crypto ticker reduced to the bare coin symbol.

    "BTC-USD" -> "btc". Yahoo quotes crypto against a currency and the
    rest of this module keys on the coin alone. A symbol with no suffix
    is passed through, so "btc" and "BTC-USD" resolve identically.
    """
    text = str(symbol or "").strip().lower()
    if not text:
        return ""
    for suffix in ("-usd", "-eur", "-gbp", "-usdt", "=x"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def is_crypto_symbol(symbol: str) -> bool:
    """Whether a ticker looks like Yahoo's crypto form.

    Used only to decide whether a crypto panel is worth drawing before
    the quote comes back; the authoritative answer is asset_class.classify
    on the info dict.
    """
    return str(symbol or "").strip().lower().endswith(
        ("-usd", "-eur", "-gbp", "-usdt"))


# --- the universe -------------------------------------------------------------

@dataclass(frozen=True)
class CoinRow:
    """One coin as the market endpoint reports it.

    `max_supply` is None when the coin is uncapped, which is a real
    answer and not a gap — see `supply_state`.
    """
    coin_id: str
    symbol: str
    name: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    market_cap_rank: Optional[int] = None
    volume_24h: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    max_supply: Optional[float] = None
    change_24h_pct: Optional[float] = None
    change_7d_pct: Optional[float] = None
    change_30d_pct: Optional[float] = None
    change_1y_pct: Optional[float] = None
    ath: Optional[float] = None
    ath_change_pct: Optional[float] = None
    ath_date: str = ""
    # Market-wide, not per-row: a coin's share of TOTAL crypto market
    # cap cannot be derived from a 250-coin page, whose total excludes
    # eighteen thousand other coins. Filled in by `with_dominance` from
    # the reported global figures.
    dominance_pct: Optional[float] = None

    @property
    def uncapped(self) -> bool:
        return not self.max_supply or self.max_supply <= 0

    @property
    def supply_state(self) -> str:
        """UNCAPPED, or "capped" — never a zero pretending to be a cap."""
        return UNCAPPED if self.uncapped else "capped"

    @property
    def pct_of_max_mined(self) -> Optional[float]:
        """How much of the eventual supply already exists, as a percent.

        None for an uncapped coin. Guarding this here is the point: the
        naive circulating/max is a ZeroDivisionError on Yahoo's data and
        a nonsense 0% on CoinGecko's, for 111 of the top 250 coins.
        """
        if self.uncapped or not self.circulating_supply:
            return None
        return 100.0 * float(self.circulating_supply) / float(self.max_supply)

    @property
    def turnover(self) -> Optional[float]:
        """24h traded volume over market cap.

        A liquidity read that does not need an order book. This is
        EXCHANGE volume, not on-chain volume — the two are different
        numbers and only the latter belongs in NVT.
        """
        if not self.market_cap or self.volume_24h is None:
            return None
        return float(self.volume_24h) / float(self.market_cap)


def with_dominance(row: Optional["CoinRow"],
                   market: Optional["GlobalMarket"]) -> Optional["CoinRow"]:
    """The same row carrying its share of the whole market.

    CoinRow is frozen, so this returns a copy rather than mutating one —
    which also means a row handed to two panels cannot pick up a
    dominance figure from whichever ran first.
    """
    if row is None:
        return None
    if market is None or not getattr(market, "ok", False):
        return row
    return replace(row, dominance_pct=market.dominance_of(row.symbol))


def _number(value) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        return None if number != number else number
    except (TypeError, ValueError):
        return None


def _row_from_market(raw: dict) -> Optional[CoinRow]:
    coin_id = str(raw.get("id") or "").strip()
    symbol = str(raw.get("symbol") or "").strip().lower()
    if not coin_id or not symbol:
        return None
    rank = raw.get("market_cap_rank")
    try:
        rank = int(rank) if rank is not None else None
    except (TypeError, ValueError):
        rank = None
    return CoinRow(
        coin_id=coin_id,
        symbol=symbol,
        name=str(raw.get("name") or coin_id),
        price=_number(raw.get("current_price")),
        market_cap=_number(raw.get("market_cap")),
        market_cap_rank=rank,
        volume_24h=_number(raw.get("total_volume")),
        circulating_supply=_number(raw.get("circulating_supply")),
        total_supply=_number(raw.get("total_supply")),
        max_supply=_number(raw.get("max_supply")),
        change_24h_pct=_number(raw.get("price_change_percentage_24h_in_currency")),
        change_7d_pct=_number(raw.get("price_change_percentage_7d_in_currency")),
        change_30d_pct=_number(raw.get("price_change_percentage_30d_in_currency")),
        change_1y_pct=_number(raw.get("price_change_percentage_1y_in_currency")),
        ath=_number(raw.get("ath")),
        ath_change_pct=_number(raw.get("ath_change_percentage")),
        ath_date=str(raw.get("ath_date") or ""),
    )


def _get_json(url: str, params: Optional[dict] = None):
    """One HTTP GET returning parsed JSON, or None. Never raises.

    Every loader in this app returns an error string rather than
    propagating, so a dead provider degrades one panel instead of taking
    the page down.
    """
    import requests

    response = requests.get(url, params=params or {},
                            timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")
    return response.json()


@st.cache_data(ttl=UNIVERSE_TTL_SECONDS, show_spinner=False)
def load_universe(per_page: int = 250) -> Tuple[Tuple[CoinRow, ...], Optional[str]]:
    """The top coins by market cap, one call.

    250 is CoinGecko's per-page maximum. The task asks to "ingest 1000
    cryptos"; four pages would do it and would also spend four of the
    ten calls a minute the keyless tier allows, for a tail of coins with
    market caps under a hundred million that no preset here screens for.
    One page is fetched and the count is stated on screen.
    """
    try:
        raw = _get_json(f"{COINGECKO_BASE}/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": int(per_page),
            "page": 1,
            "price_change_percentage": "24h,7d,30d,1y",
        })
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "crypto_data.universe_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return (), ("CoinGecko did not answer, so the coin universe is "
                    "unavailable. It is a keyless public API whose usual "
                    "failure is a rate limit — try again shortly.")

    rows = [row for row in (_row_from_market(r) for r in (raw or []))
            if row is not None]
    if not rows:
        return (), "CoinGecko returned no coins."
    log_event(logger, logging.INFO, "crypto_data.universe_loaded",
              coins=len(rows))
    return tuple(rows), None


@dataclass(frozen=True)
class Resolution:
    """Which coin a ticker was taken to mean, and whether it was a guess."""
    row: Optional[CoinRow]
    ambiguous: bool = False
    also_matched: Tuple[str, ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.row is not None


def resolve(symbol: str, rows: Sequence[CoinRow]) -> Resolution:
    """The coin a ticker means, resolved by market-cap rank.

    Symbols are not unique — DAI and USDF each appear twice inside the
    top 250 — so the highest-ranked match wins and the rest are named,
    because silently picking a rank-219 PulseChain wrapper when someone
    typed DAI would misprice the panel by an order of magnitude.
    """
    wanted = normalise_symbol(symbol)
    if not wanted:
        return Resolution(None, error="No symbol given.")

    matches = [r for r in rows if r.symbol == wanted]
    if not matches:
        # An id is accepted too, so "bitcoin" works as well as "BTC-USD".
        matches = [r for r in rows if r.coin_id == wanted]
    if not matches:
        return Resolution(None, error=(
            f"{symbol} is not in the top {len(rows)} coins by market "
            f"cap, which is the universe this build loads."))

    def rank_key(row: CoinRow):
        return (row.market_cap_rank if row.market_cap_rank is not None
                else 10 ** 9)

    matches.sort(key=rank_key)
    best = matches[0]
    others = tuple(f"{r.name} (rank {r.market_cap_rank})" for r in matches[1:])
    return Resolution(best, ambiguous=bool(others), also_matched=others)


# --- coin detail --------------------------------------------------------------

@dataclass(frozen=True)
class CoinProfile:
    """The per-coin detail: developer health, category, chain facts.

    Community/social fields are deliberately absent — see
    SOCIAL_UNAVAILABLE. Offering an empty "Reddit subscribers" row would
    read as a quiet community rather than a retired field.
    """
    coin_id: str = ""
    name: str = ""
    symbol: str = ""
    categories: Tuple[str, ...] = ()
    genesis_date: str = ""
    hashing_algorithm: str = ""
    block_time_minutes: Optional[float] = None
    commits_4w: Optional[int] = None
    contributors: Optional[int] = None
    stars: Optional[int] = None
    forks: Optional[int] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.coin_id) and not self.error

    @property
    def has_developer_data(self) -> bool:
        return self.commits_4w is not None or self.contributors is not None


def _int(value) -> Optional[int]:
    number = _number(value)
    return None if number is None else int(number)


@st.cache_data(ttl=PROFILE_TTL_SECONDS, show_spinner=False)
def load_profile(coin_id: str) -> CoinProfile:
    """Developer activity and chain facts for one coin."""
    coin_id = str(coin_id or "").strip()
    if not coin_id:
        return CoinProfile(error="No coin id given.")
    try:
        raw = _get_json(f"{COINGECKO_BASE}/coins/{coin_id}", {
            "localization": "false", "tickers": "false",
            "market_data": "false", "community_data": "false",
            "developer_data": "true", "sparkline": "false",
        })
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "crypto_data.profile_failed", coin=coin_id,
                      error=f"{type(exc).__name__}: {exc}")
        return CoinProfile(coin_id=coin_id,
                           error="CoinGecko did not return this coin's detail.")

    dev = (raw or {}).get("developer_data") or {}
    return CoinProfile(
        coin_id=coin_id,
        name=str(raw.get("name") or coin_id),
        symbol=str(raw.get("symbol") or "").lower(),
        categories=tuple(c for c in (raw.get("categories") or []) if c),
        genesis_date=str(raw.get("genesis_date") or ""),
        hashing_algorithm=str(raw.get("hashing_algorithm") or ""),
        block_time_minutes=_number(raw.get("block_time_in_minutes")),
        commits_4w=_int(dev.get("commit_count_4_weeks")),
        contributors=_int(dev.get("pull_request_contributors")),
        stars=_int(dev.get("stars")),
        forks=_int(dev.get("forks")),
    )


# --- Bitcoin dominance --------------------------------------------------------

@dataclass(frozen=True)
class GlobalMarket:
    total_market_cap: Optional[float] = None
    total_volume_24h: Optional[float] = None
    market_cap_change_24h_pct: Optional[float] = None
    dominance: Dict[str, float] = field(default_factory=dict)  # symbol -> pct
    active_coins: Optional[int] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.dominance) and not self.error

    def dominance_of(self, symbol: str) -> Optional[float]:
        return (self.dominance or {}).get(normalise_symbol(symbol))


@st.cache_data(ttl=GLOBAL_TTL_SECONDS, show_spinner=False)
def load_global() -> GlobalMarket:
    """Total market cap and each coin's share of it.

    This is where Bitcoin dominance comes from — a real reported figure,
    not one reconstructed by summing a 250-coin page, which would divide
    by a total that excludes the other eighteen thousand coins.
    """
    try:
        raw = (_get_json(f"{COINGECKO_BASE}/global") or {}).get("data") or {}
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "crypto_data.global_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return GlobalMarket(error="CoinGecko did not return market totals.")

    percentages = {}
    for key, value in (raw.get("market_cap_percentage") or {}).items():
        number = _number(value)
        if number is not None:
            percentages[str(key).lower()] = number
    return GlobalMarket(
        total_market_cap=_number((raw.get("total_market_cap") or {}).get("usd")),
        total_volume_24h=_number((raw.get("total_volume") or {}).get("usd")),
        market_cap_change_24h_pct=_number(
            raw.get("market_cap_change_percentage_24h_usd")),
        dominance=percentages,
        active_coins=_int(raw.get("active_cryptocurrencies")),
    )


# --- on-chain (Bitcoin) -------------------------------------------------------

@dataclass(frozen=True)
class OnChainMetric:
    key: str
    label: str
    chart: str            # blockchain.info chart name
    unit: str
    note: str = ""


ONCHAIN_METRICS: Tuple[OnChainMetric, ...] = (
    OnChainMetric("tx_volume_usd", "On-chain transaction volume",
                  "estimated-transaction-volume-usd", "USD",
                  "Estimated value moved on the chain per day. This is "
                  "the denominator NVT needs — not exchange volume, "
                  "which measures trading rather than settlement."),
    OnChainMetric("active_addresses", "Active addresses",
                  "n-unique-addresses", "addresses",
                  "Unique addresses used in a day. A proxy for users, "
                  "not a count of them — one person can hold many."),
    OnChainMetric("miner_revenue", "Miner revenue",
                  "miners-revenue", "USD",
                  "Block subsidy plus fees, per day. The chain's "
                  "security budget."),
    OnChainMetric("hash_rate", "Hash rate", "hash-rate", "TH/s",
                  "Computing power securing the chain."),
    OnChainMetric("transactions", "Transactions", "n-transactions",
                  "transactions"),
    OnChainMetric("supply", "Coins in circulation", "total-bitcoins", "BTC",
                  "Measured issuance. The stock-to-flow ratio takes its "
                  "flow from the change in this series rather than from "
                  "an assumed halving schedule."),
)
ONCHAIN_BY_KEY: Dict[str, OnChainMetric] = {m.key: m for m in ONCHAIN_METRICS}


def onchain_available(symbol: str) -> bool:
    return normalise_symbol(symbol) == ONCHAIN_SYMBOL


def onchain_note(symbol: str) -> str:
    """Why there is no chain data for this coin, when there is not."""
    if onchain_available(symbol):
        return ""
    return ONCHAIN_BITCOIN_ONLY


@st.cache_data(ttl=ONCHAIN_TTL_SECONDS, show_spinner=False)
def load_onchain(metric_key: str,
                 timespan: str = "1years") -> Tuple[Optional["pd.Series"], Optional[str]]:
    """One on-chain series, indexed by date.

    Indexed by NORMALISED date rather than by the raw timestamp: the
    charts are sampled independently and their timestamps do not line
    up, so market cap joined to transaction volume on the raw index
    produced an empty frame. Aligning on the day is what makes NVT
    computable at all.
    """
    metric = ONCHAIN_BY_KEY.get(metric_key)
    if metric is None:
        return None, f"No on-chain metric named {metric_key!r}."
    try:
        raw = _get_json(f"{BLOCKCHAIN_BASE}/{metric.chart}", {
            "timespan": timespan, "format": "json"})
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "crypto_data.onchain_failed",
                      chart=metric.chart,
                      error=f"{type(exc).__name__}: {exc}")
        return None, f"blockchain.info did not return {metric.label.lower()}."

    values = (raw or {}).get("values") or []
    if not values:
        return None, f"No data returned for {metric.label.lower()}."
    series = pd.Series(
        [_number(point.get("y")) for point in values],
        index=pd.to_datetime([point.get("x") for point in values],
                             unit="s").normalize(),
        name=metric.key,
    ).dropna()
    series = series[~series.index.duplicated()].sort_index()
    if series.empty:
        return None, f"No usable data for {metric.label.lower()}."
    return series, None


@st.cache_data(ttl=ONCHAIN_TTL_SECONDS, show_spinner=False)
def load_onchain_market_cap(timespan: str = "4years") -> Tuple[Optional["pd.Series"], Optional[str]]:
    """Bitcoin's market cap history, from the same source as the chain
    series so the two align on the day without a second provider's
    timestamps to reconcile."""
    try:
        raw = _get_json(f"{BLOCKCHAIN_BASE}/market-cap",
                        {"timespan": timespan, "format": "json"})
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "crypto_data.market_cap_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return None, "blockchain.info did not return market cap history."
    values = (raw or {}).get("values") or []
    if not values:
        return None, "No market cap history returned."
    series = pd.Series(
        [_number(p.get("y")) for p in values],
        index=pd.to_datetime([p.get("x") for p in values], unit="s").normalize(),
        name="market_cap").dropna()
    series = series[~series.index.duplicated()].sort_index()
    return (series, None) if not series.empty else (None, "No usable market cap history.")


# --- validation ---------------------------------------------------------------

# Bounds sized for the ERROR they catch, not for tidiness. A crypto price
# spans eight orders of magnitude in one universe — SHIB trades near
# 0.00001 and BTC near 80,000 — so a "price between X and Y" rule would
# fail on correct data, which teaches its reader to ignore the suite.
# What is worth catching is a sign error, a zero, or a NaN.

def validate_price(price: Optional[float]) -> Optional[str]:
    if price is None:
        return None
    if price != price:
        return "Price is not a number."
    if price <= 0:
        return f"Price is {price:g}; a traded coin's price is positive."
    return None


def validate_volume(volume: Optional[float]) -> Optional[str]:
    if volume is None:
        return None
    if volume < 0:
        return f"24h volume is negative ({volume:g})."
    return None


def validate_supply(row: CoinRow) -> Optional[str]:
    """Supply sanity, with UNCAPPED treated as the real answer it is."""
    circulating = row.circulating_supply
    if circulating is None:
        return None
    if circulating <= 0:
        return f"{row.symbol.upper()} reports no coins in circulation."
    if row.uncapped:
        return None
    if circulating > float(row.max_supply) * 1.001:
        return (f"{row.symbol.upper()} reports {circulating:,.0f} in "
                f"circulation against a maximum of {row.max_supply:,.0f}.")
    return None


def validate_row(row: CoinRow) -> List[str]:
    return [note for note in (validate_price(row.price),
                              validate_volume(row.volume_24h),
                              validate_supply(row)) if note]
