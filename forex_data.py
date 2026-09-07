"""Currency pairs, policy rates, purchasing power and the calendar.

THE RECORDED NOTE WAS WRONG AGAIN. `asset_class.MISSING_SOURCES[FOREX]`
said "interest-rate parity needs policy rates for both currencies; no
rates provider is wired up". Probed 2026-09-07, four official sources
answer without a key:

  - BIS, central bank policy rates for 49 economies, monthly end-of-
    period. USD 3.625, EUR 2.25, JPY 1.00, GBP 3.75, AUD 4.35, NZD 2.50,
    CAD 2.25, CHF 0.00. That is interest-rate parity and the carry trade
    unlocked. NOTE the v1 path is the one that works: the v2 dataflow
    URL returns HTTP 404.
  - The ECB Data Portal, daily, for the euro area specifically.
  - The World Bank, PPP conversion factors per country.
  - A public mirror of the Forex Factory calendar: 81 events in a week
    across nine currencies, each with an impact level, a forecast and a
    previous value.

WHICH EURO RATE, AND WHY THE TWO DISAGREE. BIS reports 2.25 for the euro
area; a headline elsewhere says 2.40. Both are right and they are
different instruments — the ECB's deposit facility rate against its main
refinancing rate, measured 15bp apart on the same day through the ECB's
own API. Since 2022 the deposit facility is the rate that actually binds,
so BIS's choice is the correct one, and this module says which it is
rather than leaving a reader to reconcile two numbers.

POLICY RATES LAG, AND THE CALENDAR IS THE ANTIDOTE. BIS is monthly and
end-of-period: on 7 September its newest euro-area observation was
August, and its UK observation was July. A carry computed on a rate a
central bank has since moved is wrong in the direction nobody checks. So
every rate carries its observation date, and the economic calendar is
read alongside it to flag a decision that is scheduled or has just
happened.

YAHOO'S FX BID AND ASK ARE NOT USABLE. Measured across fourteen major
pairs: FOUR quote an ask BELOW the bid — EURUSD -5.81bp, AUDUSD -7.20bp,
NZDUSD -5.88bp, GBPUSD -0.68bp — which is not a tight spread but an
impossible one. Every JPY cross reports a price about 1.1% away from its
own bid/ask midpoint, so the two fields are not even snapshots of the
same moment. USDCHF shows a 21bp spread on a pair that really trades
inside 1bp.

The task's PHASE 5.7 validation asks to "verify bid/ask spread" and
"bid < ask". That test would fail on the source, not on the code. So no
spread is reported anywhere in this module, the mid price is used, and
`BID_ASK_UNUSABLE` says why on screen. Same shape as the ETF phase,
where bid and ask came back as 0.0 for most funds.
"""
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger(__name__)

RATES_TTL_SECONDS = 21600        # monthly data; six hours is generous
PPP_TTL_SECONDS = 86400          # annual data
CALENDAR_TTL_SECONDS = 1800      # a week of events, refreshed often enough
COT_TTL_SECONDS = 21600          # published weekly
SPOT_TTL_SECONDS = 300

REQUEST_TIMEOUT_SECONDS = 30

# The v1 path. The v2 dataflow URL returns HTTP 404 — probed, not assumed.
BIS_POLICY_RATES_URL = (
    "https://stats.bis.org/api/v1/data/WS_CBPOL/M../all"
    "?lastNObservations=1&format=csv")
WORLD_BANK_PPP_URL = (
    "https://api.worldbank.org/v2/country/{codes}/indicator/PA.NUS.PPP"
    "?format=json&per_page=100&mrnev=1")
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


BID_ASK_UNUSABLE = (
    "Bid and ask are not shown. Measured across fourteen major pairs, "
    "four quote an ask BELOW the bid — an impossible spread, not a "
    "tight one — and every yen cross reports a price about 1.1% away "
    "from its own bid/ask midpoint, so the fields are not snapshots of "
    "the same moment. The mid price is used instead."
)

IMPLIED_VOL_UNAVAILABLE = (
    "Implied volatility and the volatility smile are not reported. They "
    "come from FX option quotes, and no free source publishes them; "
    "what is shown is REALISED volatility, computed from the price "
    "series itself."
)

BROKER_FEEDS_UNCONFIGURED = (
    "OANDA and Polygon.io are not wired — both need an account. Spot "
    "rates come from Yahoo, policy rates from the BIS, purchasing power "
    "from the World Bank and the calendar from a public mirror, none of "
    "which needs a key."
)

FORWARDS_ARE_DERIVED = (
    "Forward rates are DERIVED from covered interest parity, not "
    "quoted. No free source publishes an FX forward curve. Parity is an "
    "arbitrage identity rather than a forecast, so the number is a fair "
    "value under the quoted rates — a real forward can trade away from "
    "it when currency basis is wide."
)

CALENDAR_IS_THIRD_PARTY = (
    "The calendar comes from a public mirror of Forex Factory's feed, "
    "not from an official statistical agency. It is the schedule "
    "traders actually watch, and it carries no guarantee: treat the "
    "times as indicative and confirm anything you would act on."
)


# --- currencies ---------------------------------------------------------------

@dataclass(frozen=True)
class Currency:
    code: str            # ISO 4217, e.g. "EUR"
    name: str
    bis_area: str        # BIS REF_AREA code
    wb_country: str      # World Bank ISO3 used for PPP
    note: str = ""

    @property
    def has_ppp(self) -> bool:
        return bool(self.wb_country)


# The World Bank publishes NO euro-area aggregate PPP — EMU and EUU both
# return no observation — so Germany stands in, and the panel says so.
# France implies a rate about 5% different (0.678 against 0.710), which
# is the width of the proxy and is worth a reader knowing.
CURRENCIES: Tuple[Currency, ...] = (
    Currency("USD", "US dollar", "US", "USA"),
    Currency("EUR", "Euro", "XM", "DEU",
             "PPP uses Germany: the World Bank publishes no euro-area "
             "aggregate. France would imply a rate about 5% different."),
    Currency("JPY", "Japanese yen", "JP", "JPN"),
    Currency("GBP", "Pound sterling", "GB", "GBR"),
    Currency("AUD", "Australian dollar", "AU", "AUS"),
    Currency("NZD", "New Zealand dollar", "NZ", "NZL"),
    Currency("CAD", "Canadian dollar", "CA", "CAN"),
    Currency("CHF", "Swiss franc", "CH", "CHE"),
)
CURRENCIES_BY_CODE: Dict[str, Currency] = {c.code: c for c in CURRENCIES}

# Currencies bought when risk appetite falls. Held here because the
# carry-unwind read needs to know which side of a trade runs for cover.
SAFE_HAVENS: Tuple[str, ...] = ("JPY", "CHF", "USD")


@dataclass(frozen=True)
class Pair:
    base: str            # you buy one unit of this...
    quote: str           # ...for this many of these
    note: str = ""

    @property
    def code(self) -> str:
        return f"{self.base}{self.quote}"

    @property
    def label(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def symbol(self) -> str:
        """Yahoo's form. EURUSD=X quotes dollars per euro."""
        return f"{self.base}{self.quote}=X"


# The majors, the yen crosses that carry trades actually use, and the
# two euro crosses worth watching.
PAIRS: Tuple[Pair, ...] = (
    Pair("EUR", "USD"), Pair("GBP", "USD"), Pair("AUD", "USD"),
    Pair("NZD", "USD"), Pair("USD", "JPY"), Pair("USD", "CHF"),
    Pair("USD", "CAD"),
    Pair("AUD", "JPY", "The classic carry pair: high-yielder funded in "
                       "the lowest-yielding major."),
    Pair("NZD", "JPY"), Pair("EUR", "JPY"), Pair("GBP", "JPY"),
    Pair("CHF", "JPY"),
    Pair("EUR", "GBP"), Pair("EUR", "CHF"),
)
PAIRS_BY_CODE: Dict[str, Pair] = {p.code: p for p in PAIRS}
PAIRS_BY_SYMBOL: Dict[str, Pair] = {p.symbol.upper(): p for p in PAIRS}


def parse_pair(text: str) -> Optional[Pair]:
    """A pair from anything a user might type.

    "EURUSD=X", "EUR/USD", "eurusd" and "EUR-USD" all resolve. Returns
    None rather than guessing when the six letters are not two known
    currencies — inventing a pair would produce a symbol that quotes
    nothing.
    """
    cleaned = "".join(ch for ch in str(text or "").upper()
                      if ch.isalpha())
    if cleaned.endswith("X") and len(cleaned) == 7:
        cleaned = cleaned[:6]
    if len(cleaned) != 6:
        return None
    base, quote = cleaned[:3], cleaned[3:]
    if base not in CURRENCIES_BY_CODE or quote not in CURRENCIES_BY_CODE:
        return None
    return PAIRS_BY_CODE.get(base + quote) or Pair(base, quote)


def is_forex_symbol(text: str) -> bool:
    return str(text or "").strip().upper().endswith("=X")


# --- policy rates -------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRate:
    currency: str
    rate_pct: Optional[float] = None
    as_of: str = ""              # the BIS TIME_PERIOD, e.g. "2026-08"
    source: str = "BIS"

    @property
    def ok(self) -> bool:
        return self.rate_pct is not None


@dataclass(frozen=True)
class PolicyRates:
    rates: Dict[str, PolicyRate] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.rates) and not self.error

    def get(self, currency: str) -> Optional[PolicyRate]:
        return self.rates.get(str(currency or "").upper())

    def differential(self, base: str, quote: str) -> Optional[float]:
        """Base rate minus quote rate, in percentage points.

        This is the carry on a long position in the pair: hold the base,
        fund it in the quote. Positive means the position earns.
        """
        first, second = self.get(base), self.get(quote)
        if first is None or second is None or not (first.ok and second.ok):
            return None
        return first.rate_pct - second.rate_pct

    @property
    def stalest(self) -> Optional[str]:
        periods = [r.as_of for r in self.rates.values() if r.as_of]
        return min(periods) if periods else None


def _float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return None if number != number else number
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=RATES_TTL_SECONDS, show_spinner=False)
def load_policy_rates() -> PolicyRates:
    """Central bank policy rates from the BIS, one call for all of them.

    Monthly and end-of-period, so a rate can be several weeks old and a
    central bank may have moved since. The observation date travels with
    every rate for exactly that reason.
    """
    import requests

    try:
        response = requests.get(BIS_POLICY_RATES_URL,
                                timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        rows = list(pd.read_csv(io.StringIO(response.text)).to_dict("records"))
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "forex_data.policy_rates_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return PolicyRates(error=("The BIS policy-rate feed did not "
                                  "answer, so interest-rate parity and "
                                  "carry are unavailable this run."))

    by_area = {c.bis_area: c.code for c in CURRENCIES}
    rates: Dict[str, PolicyRate] = {}
    for row in rows:
        area = str(row.get("REF_AREA") or "").strip()
        code = by_area.get(area)
        if code is None:
            continue
        value = _float(row.get("OBS_VALUE"))
        if value is None:
            continue
        rates[code] = PolicyRate(code, value,
                                 str(row.get("TIME_PERIOD") or "").strip())
    if not rates:
        return PolicyRates(error="The BIS feed carried no rate for any "
                                 "currency this module tracks.")
    log_event(logger, logging.INFO, "forex_data.policy_rates_loaded",
              currencies=len(rates))
    return PolicyRates(rates=rates)


# --- purchasing power ---------------------------------------------------------

@dataclass(frozen=True)
class PppFactor:
    currency: str
    factor: Optional[float] = None      # local currency per international $
    year: str = ""
    country: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.factor is not None and self.factor > 0


@dataclass(frozen=True)
class PppFactors:
    factors: Dict[str, PppFactor] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.factors) and not self.error

    def get(self, currency: str) -> Optional[PppFactor]:
        return self.factors.get(str(currency or "").upper())


@st.cache_data(ttl=PPP_TTL_SECONDS, show_spinner=False)
def load_ppp() -> PppFactors:
    """PPP conversion factors from the World Bank.

    Annual, and a year or two behind by nature — a purchasing-power
    comparison is a slow-moving anchor, not a quote.
    """
    import requests

    codes = ";".join(c.wb_country for c in CURRENCIES if c.has_ppp)
    try:
        response = requests.get(WORLD_BANK_PPP_URL.format(codes=codes),
                                timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        payload = response.json()
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "forex_data.ppp_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return PppFactors(error="The World Bank PPP series did not answer.")

    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else None
    if not rows:
        return PppFactors(error="The World Bank returned no PPP observations.")

    by_country = {c.wb_country: c for c in CURRENCIES if c.has_ppp}
    factors: Dict[str, PppFactor] = {}
    for row in rows:
        country = str(row.get("countryiso3code") or "").strip()
        currency = by_country.get(country)
        value = _float(row.get("value"))
        if currency is None or value is None:
            continue
        factors[currency.code] = PppFactor(
            currency.code, value, str(row.get("date") or ""),
            (row.get("country") or {}).get("value", country),
            currency.note)
    if not factors:
        return PppFactors(error="No usable PPP factor was returned.")
    return PppFactors(factors=factors)


# --- economic calendar --------------------------------------------------------

IMPACT_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Holiday": 0}


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    currency: str
    when: Optional["pd.Timestamp"] = None
    impact: str = ""
    forecast: str = ""
    previous: str = ""

    @property
    def high_impact(self) -> bool:
        return self.impact == "High"

    @property
    def is_rate_decision(self) -> bool:
        """Whether this event sets a policy rate.

        Matters because a scheduled decision makes every carry number on
        the page provisional — the differential it rests on is about to
        change.
        """
        text = self.title.lower()
        # Widened after two real titles slipped through: the feed writes
        # "RBA Rate Statement" and "Federal Funds Rate", neither of
        # which matches "rate decision" or "fed funds". A missed
        # decision is the one case this flag exists to catch.
        return any(k in text for k in
                   ("rate decision", "rate statement", "refinancing rate",
                    "cash rate", "bank rate", "official cash",
                    "federal funds", "funds rate", "interest rate",
                    "policy rate", "rate vote", "ocr"))


@st.cache_data(ttl=CALENDAR_TTL_SECONDS, show_spinner=False)
def load_calendar() -> Tuple[Tuple[CalendarEvent, ...], Optional[str]]:
    """This week's scheduled releases.

    The `country` field carries a CURRENCY code rather than a country —
    "USD", "EUR", "All" — which is convenient here and would be a trap
    for anyone expecting ISO country codes.
    """
    import requests

    try:
        response = requests.get(CALENDAR_URL,
                                timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        payload = response.json()
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "forex_data.calendar_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return (), ("The economic calendar did not answer. It is a "
                    "third-party mirror rather than an official feed.")

    events: List[CalendarEvent] = []
    for row in payload or []:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        when = None
        try:
            when = pd.Timestamp(str(row.get("date") or "")).tz_convert(None)
        except Exception:                          # noqa: BLE001
            try:
                when = pd.Timestamp(str(row.get("date") or ""))
            except Exception:                      # noqa: BLE001
                when = None
        events.append(CalendarEvent(
            title=title,
            currency=str(row.get("country") or "").strip().upper(),
            when=when, impact=str(row.get("impact") or "").strip(),
            forecast=str(row.get("forecast") or "").strip(),
            previous=str(row.get("previous") or "").strip()))
    if not events:
        return (), "The calendar feed returned no events."
    events.sort(key=lambda e: (e.when is None, e.when))
    log_event(logger, logging.INFO, "forex_data.calendar_loaded",
              events=len(events))
    return tuple(events), None


def events_for(events: Sequence[CalendarEvent], pair: Pair,
               high_impact_only: bool = False) -> Tuple[CalendarEvent, ...]:
    """Events touching either leg of a pair.

    "All" is included: a global event moves both sides and excluding it
    would drop exactly the releases that move everything at once.
    """
    wanted = {pair.base, pair.quote, "ALL"}
    out = [e for e in events if e.currency in wanted]
    if high_impact_only:
        out = [e for e in out if e.high_impact]
    return tuple(out)


# --- positioning --------------------------------------------------------------

# The CFTC market code carries a TRAILING SPACE — "CME " — so an equality
# filter on "CME" silently returns zero rows. Found the hard way.
COT_MARKET_CODE = "CME "

COT_CONTRACTS: Dict[str, str] = {
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    "NZD": "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
}


@dataclass(frozen=True)
class Positioning:
    currency: str
    net: Optional[int] = None          # long minus short, non-commercial
    long: Optional[int] = None
    short: Optional[int] = None
    as_of: str = ""

    @property
    def ok(self) -> bool:
        return self.net is not None

    @property
    def stance(self) -> str:
        """Speculative positioning, phrased as a fact about contracts.

        Never as a signal: a crowded position is information about who
        else is in the trade, not a direction.
        """
        if not self.ok:
            return "Unavailable"
        if self.net > 0:
            return "Net long"
        if self.net < 0:
            return "Net short"
        return "Flat"


@st.cache_data(ttl=COT_TTL_SECONDS, show_spinner=False)
def load_positioning() -> Tuple[Dict[str, Positioning], Optional[str]]:
    """Non-commercial net positioning in CME currency futures.

    This is USD-relative by construction — every contract is that
    currency against the dollar — so a net short in the yen contract is
    a bet on dollar strength, not a view on the yen in isolation.
    """
    import requests

    try:
        response = requests.get(COT_URL, timeout=REQUEST_TIMEOUT_SECONDS,
                                params={
                                    "$limit": 500,
                                    "$order": "report_date_as_yyyy_mm_dd DESC",
                                    "$select": ("market_and_exchange_names,"
                                                "report_date_as_yyyy_mm_dd,"
                                                "noncomm_positions_long_all,"
                                                "noncomm_positions_short_all")})
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        rows = response.json()
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "forex_data.cot_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return {}, "The CFTC positioning report did not answer."

    by_name = {name: code for code, name in COT_CONTRACTS.items()}
    out: Dict[str, Positioning] = {}
    for row in rows or []:
        name = str(row.get("market_and_exchange_names") or "").strip()
        code = by_name.get(name)
        if code is None or code in out:
            continue
        long_ = _float(row.get("noncomm_positions_long_all"))
        short = _float(row.get("noncomm_positions_short_all"))
        if long_ is None or short is None:
            continue
        out[code] = Positioning(
            code, int(long_ - short), int(long_), int(short),
            str(row.get("report_date_as_yyyy_mm_dd") or "")[:10])
    if not out:
        return {}, "No currency contracts were found in the report."
    return out, None


# --- validation ---------------------------------------------------------------

def validate_rate(rate: Optional[float]) -> Optional[str]:
    """A policy rate can legitimately be NEGATIVE.

    The ECB, the SNB and the BoJ all held rates below zero within the
    last decade — Switzerland at -0.75% for years — so a "rates
    positive" bound fails on recent history. What is impossible is a
    NaN, and what is implausible is a rate outside the range any
    inflation-targeting central bank has used.
    """
    if rate is None:
        return None
    if rate != rate:
        return "Policy rate is not a number."
    if rate < -5 or rate > 100:
        return f"Policy rate of {rate:g}% is outside any plausible range."
    return None


def validate_price(price: Optional[float]) -> Optional[str]:
    """An exchange rate is a ratio of two positive prices.

    Unlike a futures price it cannot be negative or zero. But it spans
    four orders of magnitude across pairs — EURGBP near 0.86 and USDJPY
    near 154 — so no range bound applies.
    """
    if price is None:
        return None
    if price != price:
        return "Exchange rate is not a number."
    if price <= 0:
        return f"Exchange rate is {price:g}; a rate is a positive ratio."
    return None


# Deliberately NOT a validator: the task asks to verify "bid < ask" and a
# "reasonable" spread. Four of fourteen majors fail the first on the live
# feed and USDCHF fails the second, so such a test would report the
# source's defects as the app's. See BID_ASK_UNUSABLE.
SPREAD_VALIDATION_DECLINED = BID_ASK_UNUSABLE


def validate_pair_symbol(symbol: str) -> Optional[str]:
    if parse_pair(symbol) is None:
        return (f"{symbol} is not two currencies this build tracks — "
                f"{', '.join(c.code for c in CURRENCIES)}.")
    return None
