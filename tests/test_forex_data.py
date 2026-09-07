"""Pairs, policy rates, purchasing power, the calendar and positioning.

Measured live on 2026-09-07 and defended here:

  - the BIS publishes policy rates for 49 economies without a key, so
    the old "no rates provider is wired up" note was wrong;
  - Yahoo's FX bid/ask is unusable — four of fourteen majors quote an
    ask BELOW the bid;
  - the CFTC market code carries a TRAILING SPACE, so an equality filter
    on "CME" silently returns nothing.
"""
import json

import pandas as pd
import pytest

import forex_data as fd


# --- pairs --------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("EURUSD=X", "EURUSD=X"), ("EUR/USD", "EURUSD=X"),
    ("eurusd", "EURUSD=X"), ("EUR-USD", "EURUSD=X"),
    ("AUDJPY", "AUDJPY=X"), ("  usd/jpy  ", "USDJPY=X"),
])
def test_a_pair_resolves_from_anything_a_user_types(given, expected):
    pair = fd.parse_pair(given)
    assert pair is not None and pair.symbol == expected


@pytest.mark.parametrize("given", ["XXXYYY", "EUR", "", None, "AAPL",
                                   "EURUSDGBP"])
def test_an_unknown_pair_returns_none_rather_than_guessing(given):
    """Inventing a pair produces a symbol that quotes nothing, and the
    page then reports a fetch failure for a question nobody asked."""
    assert fd.parse_pair(given) is None


def test_the_quote_convention_is_base_then_quote():
    """EURUSD=X is dollars per euro. Reversing this inverts every carry
    number and every forward without raising."""
    pair = fd.parse_pair("EUR/USD")
    assert pair.base == "EUR" and pair.quote == "USD"
    assert pair.label == "EUR/USD" and pair.code == "EURUSD"


def test_every_declared_pair_is_reachable_both_ways():
    for pair in fd.PAIRS:
        assert fd.PAIRS_BY_CODE[pair.code] is pair
        assert fd.PAIRS_BY_SYMBOL[pair.symbol.upper()] is pair


def test_every_pair_leg_is_a_currency_with_a_rate_and_a_ppp_source():
    for pair in fd.PAIRS:
        for leg in (pair.base, pair.quote):
            currency = fd.CURRENCIES_BY_CODE[leg]
            assert currency.bis_area, leg
            assert currency.has_ppp, leg


def test_the_yen_crosses_carry_trades_use_are_present():
    codes = {p.code for p in fd.PAIRS}
    for code in ("AUDJPY", "NZDJPY", "GBPJPY"):
        assert code in codes


def test_the_safe_havens_are_declared():
    assert set(fd.SAFE_HAVENS) == {"JPY", "CHF", "USD"}


def test_a_forex_symbol_is_recognised_by_its_suffix():
    assert fd.is_forex_symbol("EURUSD=X") is True
    assert fd.is_forex_symbol("GC=F") is False
    assert fd.is_forex_symbol("") is False


# --- policy rates -------------------------------------------------------------

def _rates(**pairs):
    return fd.PolicyRates(rates={
        code: fd.PolicyRate(code, value, "2026-08")
        for code, value in pairs.items()})


def test_the_differential_is_base_minus_quote():
    """The carry on a long position: hold the base, fund in the quote.
    AUD at 4.35 funded in JPY at 1.00 earns 3.35 points."""
    rates = _rates(AUD=4.35, JPY=1.0)
    assert rates.differential("AUD", "JPY") == pytest.approx(3.35)
    assert rates.differential("JPY", "AUD") == pytest.approx(-3.35)


def test_a_missing_rate_gives_no_differential_rather_than_zero():
    """Treating an absent rate as 0% would report a carry of exactly the
    other leg's rate, which looks like a measurement."""
    rates = _rates(AUD=4.35)
    assert rates.differential("AUD", "JPY") is None
    assert rates.differential("JPY", "AUD") is None


def test_the_stalest_observation_is_reported():
    """BIS is monthly and lags unevenly — measured, GBP was a month
    behind USD on the same day."""
    rates = fd.PolicyRates(rates={
        "USD": fd.PolicyRate("USD", 3.625, "2026-08"),
        "GBP": fd.PolicyRate("GBP", 3.75, "2026-07")})
    assert rates.stalest == "2026-07"


def _install(monkeypatch, text, status=200, json_payload=None):
    class _Response:
        status_code = status
        @property
        def text(self):
            return text
        def json(self):
            if json_payload is None:
                raise ValueError("not json")
            return json_payload

    monkeypatch.setattr("requests.get", lambda *a, **k: _Response())


_BIS_CSV = (
    "FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\n"
    "M,US,2026-08,3.625\n"
    "M,XM,2026-08,2.25\n"
    "M,JP,2026-08,1\n"
    "M,GB,2026-07,3.75\n"
    "M,BR,2026-08,14\n"          # an economy this module does not track
)


def test_policy_rates_parse_and_map_bis_areas_to_currencies(monkeypatch):
    _install(monkeypatch, _BIS_CSV)
    rates = fd.load_policy_rates.__wrapped__()
    assert rates.ok
    assert rates.get("USD").rate_pct == pytest.approx(3.625)
    assert rates.get("EUR").rate_pct == pytest.approx(2.25)   # XM -> EUR
    assert rates.get("GBP").as_of == "2026-07"
    # An economy outside the table is ignored rather than mis-assigned.
    assert "BRL" not in rates.rates


def test_a_dead_bis_feed_returns_an_error_rather_than_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("requests.get", boom)
    rates = fd.load_policy_rates.__wrapped__()
    assert not rates.ok and rates.error
    assert rates.differential("AUD", "JPY") is None


def test_a_non_200_from_bis_is_an_error(monkeypatch):
    _install(monkeypatch, _BIS_CSV, status=503)
    assert not fd.load_policy_rates.__wrapped__().ok


def test_a_bis_response_with_no_tracked_currency_is_an_error(monkeypatch):
    _install(monkeypatch, "FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nM,BR,2026-08,14\n")
    assert not fd.load_policy_rates.__wrapped__().ok


# --- purchasing power ---------------------------------------------------------

_WB_JSON = [
    {"page": 1},
    [{"countryiso3code": "USA", "date": "2025", "value": 1.0,
      "country": {"value": "United States"}},
     {"countryiso3code": "DEU", "date": "2025", "value": 0.709983,
      "country": {"value": "Germany"}},
     {"countryiso3code": "JPN", "date": "2025", "value": 97.08005,
      "country": {"value": "Japan"}},
     {"countryiso3code": "XXX", "date": "2025", "value": 5.0,
      "country": {"value": "Nowhere"}}],
]


def test_ppp_factors_map_countries_to_currencies(monkeypatch):
    _install(monkeypatch, "", json_payload=_WB_JSON)
    factors = fd.load_ppp.__wrapped__()
    assert factors.ok
    assert factors.get("JPY").factor == pytest.approx(97.08005)
    # Germany stands in for the euro area, and the row says so.
    assert factors.get("EUR").country == "Germany"
    assert "euro-area" in factors.get("EUR").note


def test_the_euro_proxy_is_disclosed_on_the_currency_itself():
    """The World Bank publishes no euro-area aggregate — EMU and EUU
    both return nothing — so the substitution has to be visible."""
    note = fd.CURRENCIES_BY_CODE["EUR"].note
    assert "Germany" in note and "France" in note


def test_a_failed_world_bank_call_is_not_ok(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr("requests.get", boom)
    assert not fd.load_ppp.__wrapped__().ok


def test_an_empty_world_bank_payload_is_an_error(monkeypatch):
    _install(monkeypatch, "", json_payload=[{"page": 1}, []])
    assert not fd.load_ppp.__wrapped__().ok


# --- the calendar -------------------------------------------------------------

_CAL_JSON = [
    {"title": "Main Refinancing Rate", "country": "EUR",
     "date": "2026-09-10T08:15:00-04:00", "impact": "High",
     "forecast": "2.65%", "previous": "2.40%"},
    {"title": "Core PPI m/m", "country": "USD",
     "date": "2026-09-10T08:30:00-04:00", "impact": "High",
     "forecast": "0.3%", "previous": "0.2%"},
    {"title": "Leading Indicators", "country": "JPY",
     "date": "2026-09-07T01:00:00-04:00", "impact": "Low",
     "forecast": "", "previous": ""},
    {"title": "OPEC Meetings", "country": "All",
     "date": "2026-09-08T05:00:00-04:00", "impact": "Medium",
     "forecast": "", "previous": ""},
    {"title": "", "country": "USD", "date": "", "impact": "Low"},
]


def test_the_calendar_parses_and_sorts_by_time(monkeypatch):
    _install(monkeypatch, "", json_payload=_CAL_JSON)
    events, error = fd.load_calendar.__wrapped__()
    assert error is None
    assert len(events) == 4          # the untitled row is dropped
    times = [e.when for e in events if e.when is not None]
    assert times == sorted(times)


def test_the_country_field_actually_carries_a_CURRENCY(monkeypatch):
    """"USD", "EUR", "All" — not ISO country codes. A reader expecting
    countries would filter on nothing."""
    _install(monkeypatch, "", json_payload=_CAL_JSON)
    events, _ = fd.load_calendar.__wrapped__()
    assert {e.currency for e in events} == {"EUR", "USD", "JPY", "ALL"}


def test_events_for_a_pair_include_BOTH_legs_and_global_events():
    """A global event moves both sides; excluding it would drop exactly
    the releases that move everything at once."""
    events = (
        fd.CalendarEvent("ECB", "EUR", pd.Timestamp("2026-09-10"), "High"),
        fd.CalendarEvent("NFP", "USD", pd.Timestamp("2026-09-11"), "High"),
        fd.CalendarEvent("OPEC", "ALL", pd.Timestamp("2026-09-08"), "Medium"),
        fd.CalendarEvent("Tankan", "JPY", pd.Timestamp("2026-09-09"), "High"),
    )
    picked = fd.events_for(events, fd.PAIRS_BY_CODE["EURUSD"])
    assert {e.title for e in picked} == {"ECB", "NFP", "OPEC"}


def test_high_impact_only_filters_but_keeps_both_legs():
    events = (
        fd.CalendarEvent("ECB", "EUR", pd.Timestamp("2026-09-10"), "High"),
        fd.CalendarEvent("Retail", "USD", pd.Timestamp("2026-09-11"), "Low"),
    )
    picked = fd.events_for(events, fd.PAIRS_BY_CODE["EURUSD"],
                           high_impact_only=True)
    assert {e.title for e in picked} == {"ECB"}


@pytest.mark.parametrize("title", [
    "Main Refinancing Rate", "RBA Rate Statement", "Official Cash Rate",
    "BOE Bank Rate Votes", "Federal Funds Rate", "OCR",
])
def test_a_rate_decision_is_recognised(title):
    """It matters because a scheduled decision makes every carry number
    on the page provisional."""
    assert fd.CalendarEvent(title, "EUR").is_rate_decision


@pytest.mark.parametrize("title", ["Core PPI m/m", "Trade Balance",
                                   "Unemployment Claims"])
def test_an_ordinary_release_is_not_a_rate_decision(title):
    assert not fd.CalendarEvent(title, "USD").is_rate_decision


def test_a_dead_calendar_returns_an_error_and_names_what_it_is(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr("requests.get", boom)
    events, error = fd.load_calendar.__wrapped__()
    assert events == () and "third-party" in error


def test_the_calendar_is_disclosed_as_unofficial():
    assert "Forex Factory" in fd.CALENDAR_IS_THIRD_PARTY
    assert "official" in fd.CALENDAR_IS_THIRD_PARTY


# --- positioning --------------------------------------------------------------

def test_the_cftc_market_code_keeps_its_trailing_space():
    """Found the hard way: filtering on "CME" returns zero rows because
    the published value is "CME " with a trailing space."""
    assert fd.COT_MARKET_CODE == "CME "
    assert fd.COT_MARKET_CODE != fd.COT_MARKET_CODE.strip()


_COT_JSON = [
    {"market_and_exchange_names": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
     "report_date_as_yyyy_mm_dd": "2026-09-01T00:00:00.000",
     "noncomm_positions_long_all": "203477",
     "noncomm_positions_short_all": "228402"},
    {"market_and_exchange_names": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
     "report_date_as_yyyy_mm_dd": "2026-09-01T00:00:00.000",
     "noncomm_positions_long_all": "117169",
     "noncomm_positions_short_all": "209396"},
    {"market_and_exchange_names": "GULF # 6 FUEL OIL CRACK",
     "report_date_as_yyyy_mm_dd": "2026-09-01T00:00:00.000",
     "noncomm_positions_long_all": "10", "noncomm_positions_short_all": "5"},
]


def test_positioning_reads_net_non_commercial_contracts(monkeypatch):
    _install(monkeypatch, "", json_payload=_COT_JSON)
    positions, error = fd.load_positioning.__wrapped__()
    assert error is None
    assert positions["EUR"].net == 203477 - 228402
    assert positions["EUR"].stance == "Net short"
    assert positions["JPY"].as_of == "2026-09-01"
    # A non-currency contract is not mistaken for one.
    assert len(positions) == 2


def test_positioning_keeps_only_the_most_recent_report(monkeypatch):
    older = dict(_COT_JSON[0])
    older["report_date_as_yyyy_mm_dd"] = "2026-08-25T00:00:00.000"
    older["noncomm_positions_long_all"] = "1"
    _install(monkeypatch, "", json_payload=[_COT_JSON[0], older])
    positions, _ = fd.load_positioning.__wrapped__()
    assert positions["EUR"].long == 203477


def test_a_flat_book_reads_flat_not_unavailable():
    assert fd.Positioning("EUR", 0, 100, 100).stance == "Flat"
    assert fd.Positioning("EUR").stance == "Unavailable"


def test_a_dead_cot_feed_returns_an_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr("requests.get", boom)
    positions, error = fd.load_positioning.__wrapped__()
    assert positions == {} and error


# --- validation ---------------------------------------------------------------

def test_a_NEGATIVE_policy_rate_is_not_flagged():
    """The SNB held -0.75% for years and the ECB and BoJ both went below
    zero. A "rates positive" bound fails on recent history."""
    for rate in (-0.75, -0.5, 0.0, 5.25):
        assert fd.validate_rate(rate) is None


def test_an_absurd_policy_rate_is_flagged():
    assert fd.validate_rate(-40.0) is not None
    assert fd.validate_rate(500.0) is not None
    assert fd.validate_rate(float("nan")) is not None
    assert fd.validate_rate(None) is None


def test_an_exchange_rate_must_be_positive_unlike_a_futures_price():
    """A rate is a ratio of two prices, so unlike a futures contract it
    cannot go below zero."""
    assert fd.validate_price(-1.0) is not None
    assert fd.validate_price(0.0) is not None
    assert fd.validate_price(float("nan")) is not None


def test_rates_across_orders_of_magnitude_all_pass():
    """EURGBP near 0.86 and USDJPY near 154 sit in the same universe."""
    for price in (0.8588, 1.1631, 154.53, 209.17):
        assert fd.validate_price(price) is None


def test_the_spread_validation_the_task_asks_for_is_DECLINED():
    """PHASE 5.7 asks to verify "bid < ask". Four of fourteen majors
    fail that on the live feed, so the test would report the source's
    defect as the app's."""
    assert fd.SPREAD_VALIDATION_DECLINED is fd.BID_ASK_UNUSABLE
    assert "ask BELOW the bid" in fd.BID_ASK_UNUSABLE
    names = [n for n in dir(fd) if not n.startswith("_")]
    assert not [n for n in names if "spread" in n.lower()
                and n != "SPREAD_VALIDATION_DECLINED"]


def test_an_unknown_pair_symbol_explains_what_is_tracked():
    note = fd.validate_pair_symbol("XXXYYY")
    assert note and "EUR" in note and "JPY" in note
    assert fd.validate_pair_symbol("EURUSD=X") is None


def test_the_unavailable_notes_describe_an_absence_with_a_reason():
    for note in (fd.BID_ASK_UNUSABLE, fd.IMPLIED_VOL_UNAVAILABLE,
                 fd.BROKER_FEEDS_UNCONFIGURED, fd.FORWARDS_ARE_DERIVED,
                 fd.CALENDAR_IS_THIRD_PARTY):
        assert len(note) > 80
        assert note.strip().endswith(".")
