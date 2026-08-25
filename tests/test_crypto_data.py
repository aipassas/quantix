"""The crypto data pipeline: symbols, supply, resolution and validation.

The traps this file pins, all of them measured against the live sources
on 2026-08-25 rather than recalled:

  - max supply 0/None means UNCAPPED, and 111 of the top 250 coins are.
  - symbols collide inside the top 250 (DAI, USDF), so resolution is by
    market-cap rank.
  - on-chain data covers Bitcoin only.
  - the two on-chain charts are sampled independently, so they must be
    joined on the DAY or the join is empty.
"""
import sys
import types

import pandas as pd
import pytest

import crypto_data as cd


# --- symbols ------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("BTC-USD", "btc"), ("btc-usd", "btc"), ("ETH-EUR", "eth"),
    ("DOGE-USD", "doge"), ("btc", "btc"), ("bitcoin", "bitcoin"),
    ("  SOL-USD  ", "sol"), ("", ""), (None, ""),
])
def test_a_yahoo_crypto_ticker_reduces_to_the_bare_coin(given, expected):
    assert cd.normalise_symbol(given) == expected


def test_a_currency_suffix_is_not_mistaken_for_part_of_the_symbol():
    """USDT-USD must not become "usdt-" or "" — the coin is USDT."""
    assert cd.normalise_symbol("USDT-USD") == "usdt"


# --- the uncapped trap --------------------------------------------------------

def _row(**kwargs):
    base = dict(coin_id="x", symbol="x", name="X")
    base.update(kwargs)
    return cd.CoinRow(**base)


def test_a_max_supply_of_none_means_uncapped_not_zero():
    """CoinGecko's form of the trap."""
    row = _row(circulating_supply=1.2e8, max_supply=None)
    assert row.uncapped is True
    assert row.supply_state == cd.UNCAPPED
    assert row.pct_of_max_mined is None


def test_a_max_supply_of_zero_means_uncapped_not_zero():
    """Yahoo's form of the SAME trap: it reports maxSupply 0 for ETH and
    DOGE, neither of which has a cap. Treating that as a real maximum is
    a ZeroDivisionError, and treating it as "0 coins will ever exist" is
    simply false."""
    row = _row(circulating_supply=1.2e8, max_supply=0)
    assert row.uncapped is True
    assert row.pct_of_max_mined is None


def test_percent_mined_is_computed_for_a_genuinely_capped_coin():
    row = _row(circulating_supply=20_075_134.0, max_supply=21_000_000.0)
    assert row.uncapped is False
    assert row.pct_of_max_mined == pytest.approx(95.596, abs=0.01)


def test_percent_mined_never_divides_by_zero():
    """The whole point of the guard. Without it this raises."""
    for maximum in (0, 0.0, None):
        assert _row(circulating_supply=1e6, max_supply=maximum).pct_of_max_mined is None


def test_turnover_is_volume_over_market_cap():
    row = _row(market_cap=1000.0, volume_24h=250.0)
    assert row.turnover == pytest.approx(0.25)
    assert _row(market_cap=0, volume_24h=250.0).turnover is None


# --- resolution ---------------------------------------------------------------

def _universe():
    """The lower-ranked DAI is listed FIRST, deliberately.

    With the real coin first, an implementation that simply took the
    first match would pass — and one did, under a poison run. Order and
    rank have to disagree for the test to be about rank at all.
    """
    return (
        _row(coin_id="dai-on-pulsechain", symbol="dai",
             name="DAI on PulseChain", market_cap_rank=219, market_cap=1.3e8),
        _row(coin_id="bitcoin", symbol="btc", name="Bitcoin",
             market_cap_rank=1, market_cap=1.57e12),
        _row(coin_id="dai", symbol="dai", name="Dai", market_cap_rank=23,
             market_cap=4.59e9),
    )


def test_a_colliding_symbol_resolves_to_the_larger_coin():
    """THE BUG THIS PREVENTS. DAI is two different coins inside the top
    250 and they differ by a factor of 35 in market cap. Taking whichever
    appeared first in the response would misprice the panel silently."""
    result = cd.resolve("DAI-USD", _universe())
    assert result.ok
    assert result.row.name == "Dai"
    assert result.row.market_cap_rank == 23
    assert result.ambiguous is True
    assert "DAI on PulseChain (rank 219)" in result.also_matched


def test_an_unambiguous_symbol_is_not_flagged_as_ambiguous():
    result = cd.resolve("BTC-USD", _universe())
    assert result.ok and result.ambiguous is False
    assert result.also_matched == ()


def test_a_coin_id_resolves_as_well_as_a_symbol():
    assert cd.resolve("bitcoin", _universe()).row.symbol == "btc"


def test_a_coin_outside_the_universe_reports_why():
    result = cd.resolve("NOTACOIN-USD", _universe())
    assert not result.ok
    assert "top 3 coins" in result.error


def test_a_missing_rank_does_not_win_over_a_real_one():
    """A row with no rank must sort last, not first — None compares
    unpredictably and would otherwise be picked."""
    rows = (_row(coin_id="a", symbol="zz", name="Unranked",
                 market_cap_rank=None),
            _row(coin_id="b", symbol="zz", name="Ranked",
                 market_cap_rank=7))
    assert cd.resolve("ZZ-USD", rows).row.name == "Ranked"


# --- dominance ----------------------------------------------------------------

def test_dominance_is_attached_without_mutating_the_original_row():
    """CoinRow is frozen and with_dominance returns a copy, so a row
    handed to two panels cannot pick up the other's figure."""
    row = _row(symbol="btc")
    market = cd.GlobalMarket(dominance={"btc": 59.14})
    attached = cd.with_dominance(row, market)
    assert attached.dominance_pct == pytest.approx(59.14)
    assert row.dominance_pct is None
    assert attached is not row


def test_a_failed_market_fetch_leaves_the_row_alone():
    row = _row(symbol="btc")
    assert cd.with_dominance(row, cd.GlobalMarket(error="down")) is row
    assert cd.with_dominance(row, None) is row


def test_dominance_is_looked_up_on_the_normalised_symbol():
    market = cd.GlobalMarket(dominance={"btc": 59.14})
    assert market.dominance_of("BTC-USD") == pytest.approx(59.14)
    assert market.dominance_of("nosuchcoin") is None


# --- on-chain scope -----------------------------------------------------------

def test_on_chain_data_is_bitcoin_only_and_says_so():
    assert cd.onchain_available("BTC-USD") is True
    assert cd.onchain_available("btc") is True
    assert cd.onchain_available("ETH-USD") is False
    note = cd.onchain_note("ETH-USD")
    assert "Bitcoin only" in note
    assert cd.onchain_note("BTC-USD") == ""


def test_every_declared_on_chain_metric_names_a_chart():
    for metric in cd.ONCHAIN_METRICS:
        assert metric.chart, metric.key
        assert metric.label
        assert cd.ONCHAIN_BY_KEY[metric.key] is metric


def test_the_unavailable_notes_name_what_is_missing_and_why():
    """A bare "not available" reads as a fetch that failed, and sends the
    reader looking for data that does not exist."""
    for note in (cd.MVRV_UNAVAILABLE, cd.WHALE_UNAVAILABLE,
                 cd.EXCHANGE_RESERVE_UNAVAILABLE, cd.SOCIAL_UNAVAILABLE,
                 cd.ONCHAIN_BITCOIN_ONLY, cd.EXCHANGES_NOT_WIRED):
        assert len(note) > 60, note
        assert note.endswith((".", "empty")) or "." in note


# --- validation ---------------------------------------------------------------

def test_price_validation_catches_a_sign_error_not_a_small_price():
    """Crypto prices span eight orders of magnitude in one universe —
    SHIB near 0.00001 and BTC near 80,000 — so a range bound would fail
    on correct data. Only impossibilities are flagged."""
    assert cd.validate_price(0.000012) is None
    assert cd.validate_price(78_249.0) is None
    assert cd.validate_price(None) is None
    assert cd.validate_price(-1.0) is not None
    assert cd.validate_price(0.0) is not None
    assert cd.validate_price(float("nan")) is not None


def test_supply_validation_does_not_fail_an_uncapped_coin():
    """An uncapped coin has more circulating than its (absent) maximum by
    definition. Failing it would fail ETH, USDT and 109 others."""
    assert cd.validate_supply(_row(circulating_supply=1.2e8,
                                   max_supply=None)) is None


def test_supply_validation_catches_circulating_above_a_real_cap():
    note = cd.validate_supply(_row(symbol="btc", circulating_supply=22e6,
                                   max_supply=21e6))
    assert note is not None and "21,000,000" in note


def test_supply_validation_tolerates_rounding_at_the_cap():
    """BNB reports circulating marginally above total in the live feed;
    a strict > would flag a healthy coin."""
    assert cd.validate_supply(_row(circulating_supply=200_000_100.0,
                                   max_supply=200_000_000.0)) is None


def test_validate_row_gathers_every_complaint():
    notes = cd.validate_row(_row(price=-1.0, volume_24h=-5.0,
                                 circulating_supply=0))
    assert len(notes) == 3


# --- the loaders, mocked ------------------------------------------------------

def _install(monkeypatch, payloads):
    """Route _get_json to a dict of url-substring -> payload."""
    def fake(url, params=None):
        for fragment, payload in payloads.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise RuntimeError(f"unexpected url {url}")
    monkeypatch.setattr(cd, "_get_json", fake)


def test_the_universe_parses_a_market_row(monkeypatch):
    _install(monkeypatch, {"coins/markets": [{
        "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
        "current_price": 78249, "market_cap": 1.57e12, "market_cap_rank": 1,
        "total_volume": 4.15e10, "circulating_supply": 20075134.0,
        "total_supply": 20075134.0, "max_supply": 21000000.0,
        "price_change_percentage_24h_in_currency": -0.817,
        "price_change_percentage_1y_in_currency": -29.8,
        "ath": 126080, "ath_change_percentage": -37.9,
        "ath_date": "2025-10-06T10:57:42.000Z"}]})
    rows, error = cd.load_universe.__wrapped__()
    assert error is None and len(rows) == 1
    row = rows[0]
    assert row.symbol == "btc" and row.market_cap_rank == 1
    assert row.pct_of_max_mined == pytest.approx(95.596, abs=0.01)


def test_a_market_row_without_an_id_is_dropped_not_guessed(monkeypatch):
    _install(monkeypatch, {"coins/markets": [
        {"symbol": "btc", "name": "Bitcoin"},
        {"id": "ethereum", "symbol": "eth", "name": "Ethereum"}]})
    rows, error = cd.load_universe.__wrapped__()
    assert [r.coin_id for r in rows] == ["ethereum"]


def test_a_dead_provider_returns_an_error_rather_than_raising(monkeypatch):
    _install(monkeypatch, {"coins/markets": RuntimeError("HTTP 429")})
    rows, error = cd.load_universe.__wrapped__()
    assert rows == () and error and "CoinGecko" in error


def test_global_reads_dominance_and_ignores_unparseable_shares(monkeypatch):
    _install(monkeypatch, {"global": {"data": {
        "market_cap_percentage": {"btc": 59.139, "eth": 11.071,
                                  "bad": "not a number"},
        "total_market_cap": {"usd": 2.64e12},
        "total_volume": {"usd": 1.2e11},
        "market_cap_change_percentage_24h_usd": -3.14,
        "active_cryptocurrencies": 18683}}})
    market = cd.load_global.__wrapped__()
    assert market.ok
    assert market.dominance_of("BTC-USD") == pytest.approx(59.139)
    assert "bad" not in market.dominance
    assert market.active_coins == 18683


def test_the_profile_reads_developer_data_and_omits_social(monkeypatch):
    _install(monkeypatch, {"coins/bitcoin": {
        "name": "Bitcoin", "symbol": "btc",
        "categories": ["Layer 1 (L1)", None, "Proof of Work (PoW)"],
        "genesis_date": "2009-01-03", "hashing_algorithm": "SHA-256",
        "block_time_in_minutes": 10,
        "developer_data": {"commit_count_4_weeks": 108,
                           "pull_request_contributors": 846,
                           "stars": 73168, "forks": 36426},
        "community_data": {"reddit_subscribers": 0}}})
    profile = cd.load_profile.__wrapped__("bitcoin")
    assert profile.ok and profile.has_developer_data
    assert profile.commits_4w == 108 and profile.contributors == 846
    assert profile.categories == ("Layer 1 (L1)", "Proof of Work (PoW)")
    # The dead community block must not become a field anyone can render.
    assert not hasattr(profile, "reddit_subscribers")


def test_a_failed_profile_is_not_ok(monkeypatch):
    _install(monkeypatch, {"coins/x": RuntimeError("HTTP 404")})
    profile = cd.load_profile.__wrapped__("x")
    assert not profile.ok and profile.error


def test_an_empty_coin_id_never_calls_the_provider(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("should not fetch for an empty id")
    monkeypatch.setattr(cd, "_get_json", explode)
    assert not cd.load_profile.__wrapped__("").ok


# --- on-chain series ----------------------------------------------------------

def _chart(values):
    return {"values": [{"x": x, "y": y} for x, y in values], "unit": "USD"}


def test_an_on_chain_series_is_indexed_by_DAY(monkeypatch):
    """THE JOIN BUG. The market-cap and transaction-volume charts are
    sampled independently, so their raw timestamps differ by hours. An
    inner join on the raw index returned an EMPTY frame and NVT could
    not be computed at all. Normalising to the date is what fixes it."""
    _install(monkeypatch, {"estimated-transaction-volume-usd": _chart([
        (1787529600 + 3600, 11.5e9), (1787443200 + 7200, 10.0e9)])})
    series, error = cd.load_onchain.__wrapped__("tx_volume_usd")
    assert error is None
    assert all(ts == ts.normalize() for ts in series.index)
    assert series.index.is_monotonic_increasing


def test_an_on_chain_series_drops_duplicate_days(monkeypatch):
    _install(monkeypatch, {"n-unique-addresses": _chart([
        (1787529600, 489906.0), (1787529600 + 60, 489000.0)])})
    series, _ = cd.load_onchain.__wrapped__("active_addresses")
    assert len(series) == 1


def test_an_unknown_metric_is_refused_by_name(monkeypatch):
    series, error = cd.load_onchain.__wrapped__("no_such_metric")
    assert series is None and "no_such_metric" in error


def test_an_empty_chart_is_an_error_not_an_empty_series(monkeypatch):
    _install(monkeypatch, {"hash-rate": {"values": []}})
    series, error = cd.load_onchain.__wrapped__("hash_rate")
    assert series is None and error


def test_market_cap_history_comes_from_the_same_source_as_the_chain():
    """Both from blockchain.info, so the two align on the day without a
    second provider's timestamps to reconcile."""
    import inspect
    source = inspect.getsource(cd.load_onchain_market_cap)
    assert "BLOCKCHAIN_BASE" in source


# --- the remaining fetch-and-parse paths --------------------------------------
# These only ever run against a live provider, which is exactly where the
# unit traps in this project have hidden. Mocked here rather than left to
# a live run nobody makes.

def test_a_yahoo_style_ticker_is_recognised_before_the_quote_returns():
    assert cd.is_crypto_symbol("BTC-USD") is True
    assert cd.is_crypto_symbol("eth-eur") is True
    assert cd.is_crypto_symbol("AAPL") is False
    assert cd.is_crypto_symbol("") is False
    assert cd.is_crypto_symbol(None) is False


def test_with_dominance_passes_a_missing_row_straight_through():
    assert cd.with_dominance(None, cd.GlobalMarket(dominance={"btc": 1.0})) is None


def test_an_unparseable_rank_becomes_none_rather_than_raising(monkeypatch):
    _install(monkeypatch, {"coins/markets": [{
        "id": "x", "symbol": "x", "name": "X", "market_cap_rank": "not a rank"}]})
    rows, error = cd.load_universe.__wrapped__()
    assert error is None and rows[0].market_cap_rank is None


def test_an_empty_coin_list_is_an_error_not_an_empty_success(monkeypatch):
    _install(monkeypatch, {"coins/markets": []})
    rows, error = cd.load_universe.__wrapped__()
    assert rows == () and error


def test_resolving_an_empty_symbol_asks_for_one():
    result = cd.resolve("", _universe())
    assert not result.ok and "No symbol" in result.error


def test_a_failed_global_fetch_is_not_ok(monkeypatch):
    _install(monkeypatch, {"global": RuntimeError("HTTP 429")})
    market = cd.load_global.__wrapped__()
    assert not market.ok and market.error
    assert market.dominance_of("BTC-USD") is None


def test_a_failed_on_chain_fetch_names_the_metric(monkeypatch):
    _install(monkeypatch, {"miners-revenue": RuntimeError("timeout")})
    series, error = cd.load_onchain.__wrapped__("miner_revenue")
    assert series is None and "miner revenue" in error


def test_market_cap_history_is_indexed_by_day(monkeypatch):
    _install(monkeypatch, {"market-cap": _chart([
        (1787529600 + 3600, 1.57e12), (1787443200 + 60, 1.55e12)])})
    series, error = cd.load_onchain_market_cap.__wrapped__()
    assert error is None and len(series) == 2
    assert all(ts == ts.normalize() for ts in series.index)
    assert series.index.is_monotonic_increasing


def test_a_failed_market_cap_fetch_returns_an_error(monkeypatch):
    _install(monkeypatch, {"market-cap": RuntimeError("down")})
    series, error = cd.load_onchain_market_cap.__wrapped__()
    assert series is None and error


def test_an_empty_market_cap_chart_is_an_error(monkeypatch):
    _install(monkeypatch, {"market-cap": {"values": []}})
    series, error = cd.load_onchain_market_cap.__wrapped__()
    assert series is None and error


def test_a_chart_of_only_nulls_is_an_error_not_an_empty_series(monkeypatch):
    _install(monkeypatch, {"market-cap": {"values": [
        {"x": 1787529600, "y": None}]}})
    series, error = cd.load_onchain_market_cap.__wrapped__()
    assert series is None and "usable" in error


def test_a_non_200_response_raises_so_the_loader_can_report_it(monkeypatch):
    """_get_json is the single HTTP door. It must turn a bad status into
    an exception the caller converts to an on-screen error — never into
    a silent None that later reads as "no data"."""
    class _Response:
        status_code = 429
        def json(self):                      # pragma: no cover - not reached
            raise AssertionError("must not parse a failed response")

    monkeypatch.setattr("requests.get", lambda *a, **k: _Response())
    with pytest.raises(RuntimeError, match="429"):
        cd._get_json("https://example.invalid/x")


def test_a_200_response_is_parsed(monkeypatch):
    class _Response:
        status_code = 200
        def json(self):
            return {"ok": True}

    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(url=url, params=params, timeout=timeout)
        return _Response()

    monkeypatch.setattr("requests.get", fake_get)
    assert cd._get_json("https://example.invalid/x", {"a": 1}) == {"ok": True}
    # A request with no timeout can hang a Streamlit rerun indefinitely.
    assert captured["timeout"] == cd.REQUEST_TIMEOUT_SECONDS
