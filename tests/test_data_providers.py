"""Tests for data_providers.py — the pluggable data-provider layer's
routing, and CoinGeckoProvider's response-mapping/resampling/error
handling.

The real network boundary (requests.get) is stubbed throughout, so these
verify the mapping/math deterministically rather than depending on
CoinGecko's live API being reachable during a test run. Live-network
verification against the real API was done separately while building
this (see data_providers.py's module docstring for exactly what was
confirmed and how).
"""
import datetime

import pandas as pd
import pytest

import data_providers as dp
from data_providers import (
    CoinGeckoProvider,
    DataProvider,
    get_provider_for_ticker,
    is_crypto_ticker,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --- routing -----------------------------------------------------------------

def test_is_crypto_ticker_detects_prefix_case_insensitively():
    assert is_crypto_ticker("crypto:bitcoin")
    assert is_crypto_ticker("CRYPTO:Bitcoin")
    assert not is_crypto_ticker("AAPL")
    assert not is_crypto_ticker("BTC-USD")  # Yahoo's own crypto ticker format, deliberately NOT rerouted


def test_get_provider_for_ticker_routes_crypto_prefix_only():
    assert isinstance(get_provider_for_ticker("crypto:bitcoin"), CoinGeckoProvider)
    assert get_provider_for_ticker("AAPL") is None
    assert get_provider_for_ticker("MSFT") is None


def test_coingecko_coin_id_strips_prefix_and_lowercases():
    assert dp._coingecko_coin_id("crypto:Bitcoin") == "bitcoin"
    assert dp._coingecko_coin_id("CRYPTO:  ethereum  ") == "ethereum"


def test_data_provider_base_methods_are_not_implemented():
    base = DataProvider()
    with pytest.raises(NotImplementedError):
        base.fetch_info("x")
    with pytest.raises(NotImplementedError):
        base.fetch_price_history("x", None, None)


# --- _fetch_json_with_retry ----------------------------------------------------

def test_fetch_json_success_on_first_try(monkeypatch):
    monkeypatch.setattr(dp.requests, "get", lambda url, params, timeout: _FakeResponse(200, {"ok": True}))
    data, err = dp._fetch_json_with_retry("http://x", {}, "label")
    assert err is None
    assert data == {"ok": True}


def test_fetch_json_404_is_not_found_no_retry(monkeypatch):
    calls = {"n": 0}
    def fake_get(url, params, timeout):
        calls["n"] += 1
        return _FakeResponse(404)
    monkeypatch.setattr(dp.requests, "get", fake_get)
    data, err = dp._fetch_json_with_retry("http://x", {}, "label")
    assert data is None
    assert "unknown" in err.lower()
    assert calls["n"] == 1  # a definitive 404 shouldn't burn retries


def test_fetch_json_429_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    def fake_get(url, params, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponse(429)
        return _FakeResponse(200, {"ok": True})
    monkeypatch.setattr(dp.requests, "get", fake_get)
    monkeypatch.setattr(dp.time, "sleep", lambda s: None)
    data, err = dp._fetch_json_with_retry("http://x", {}, "label")
    assert err is None
    assert data == {"ok": True}
    assert calls["n"] == 3


def test_fetch_json_exhausts_retries_and_reports_failure(monkeypatch):
    monkeypatch.setattr(dp.requests, "get", lambda url, params, timeout: _FakeResponse(500))
    monkeypatch.setattr(dp.time, "sleep", lambda s: None)
    data, err = dp._fetch_json_with_retry("http://x", {}, "label", retries=2)
    assert data is None
    assert "failed after 2" in err


# --- _coingecko_fetch_info -----------------------------------------------------

def _canned_info_response(current_price=63632.0, change_24h=-425.47):
    return {
        "name": "Bitcoin", "symbol": "btc",
        "market_data": {
            "current_price": {"usd": current_price},
            "price_change_24h": change_24h,
            "market_cap": {"usd": 1_234_000_000_000},
        },
    }


def test_coingecko_fetch_info_maps_fields_correctly(monkeypatch):
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: (_canned_info_response(), None))
    info, errors = dp._coingecko_fetch_info.__wrapped__("bitcoin")
    assert errors == []
    assert info["longName"] == "Bitcoin"
    assert info["symbol"] == "BTC"
    assert info["currentPrice"] == 63632.0
    assert info["previousClose"] == pytest.approx(63632.0 - (-425.47))
    assert info["quoteType"] == "CRYPTOCURRENCY"


def test_coingecko_fetch_info_never_fabricates_equity_only_fields(monkeypatch):
    """A crypto asset genuinely has no P/E or GICS sector — these keys must
    be ABSENT, never a fabricated 0/None-that-looks-computed."""
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: (_canned_info_response(), None))
    info, _ = dp._coingecko_fetch_info.__wrapped__("bitcoin")
    assert "trailingPE" not in info
    assert "sector" not in info


def test_coingecko_fetch_info_propagates_fetch_failure(monkeypatch):
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: (None, "network error"))
    info, errors = dp._coingecko_fetch_info.__wrapped__("nonexistent-coin")
    assert info == {}
    assert errors == ["network error"]


# --- _coingecko_fetch_price_history --------------------------------------------

def _hourly_prices(day_count: int, start: datetime.date):
    """Synthetic hourly (prices, volumes) covering `day_count` days,
    each day's price rising from hour 0 to hour 23 so "last of day" is
    unambiguous and checkable. Built explicitly in UTC — real CoinGecko
    epoch-ms timestamps are UTC, and pd.Timestamp(ts, unit="ms") in the
    code under test parses them as UTC, so the fixture must match or the
    resulting calendar-day buckets shift by the local machine's offset."""
    prices, volumes = [], []
    base = datetime.datetime.combine(start, datetime.time.min, tzinfo=datetime.timezone.utc)
    for day in range(day_count):
        for hour in range(24):
            ts = base + datetime.timedelta(days=day, hours=hour)
            ts_ms = int(ts.timestamp() * 1000)
            prices.append([ts_ms, 100.0 + day + hour * 0.01])
            volumes.append([ts_ms, 1000.0 + day])
    return prices, volumes


def test_price_history_resamples_hourly_to_one_bar_per_day(monkeypatch):
    start = datetime.date(2024, 6, 1)
    end = datetime.date(2024, 6, 3)
    prices, volumes = _hourly_prices(3, start)
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: ({"prices": prices, "total_volumes": volumes}, None))

    df, messages = dp._coingecko_fetch_price_history.__wrapped__("bitcoin", start, end)

    assert len(df) == 3  # 3 calendar days, not 72 hourly rows
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # Last hour of day 0 (hour=23) should win the day's Close.
    assert df["Close"].iloc[0] == pytest.approx(100.0 + 0 + 23 * 0.01)


def test_price_history_open_high_low_all_equal_close(monkeypatch):
    """Disclosed limitation, verified directly: no fabricated intraday range."""
    start = datetime.date(2024, 6, 1)
    prices, volumes = _hourly_prices(2, start)
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: ({"prices": prices, "total_volumes": volumes}, None))
    df, messages = dp._coingecko_fetch_price_history.__wrapped__("bitcoin", start, start + datetime.timedelta(days=1))
    assert (df["Open"] == df["Close"]).all()
    assert (df["High"] == df["Close"]).all()
    assert (df["Low"] == df["Close"]).all()
    assert any("Open/High/Low are set equal to Close" in m for m in messages)


def test_price_history_volume_is_not_equal_to_close_ie_genuinely_separate(monkeypatch):
    start = datetime.date(2024, 6, 1)
    prices, volumes = _hourly_prices(2, start)
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: ({"prices": prices, "total_volumes": volumes}, None))
    df, _ = dp._coingecko_fetch_price_history.__wrapped__("bitcoin", start, start + datetime.timedelta(days=1))
    assert not (df["Volume"] == df["Close"]).any()


def test_price_history_clamps_start_date_beyond_365_days(monkeypatch):
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: ({"prices": [[1, 1.0]], "total_volumes": [[1, 1.0]]}, None))
    end = datetime.date(2024, 6, 1)
    far_past = end - datetime.timedelta(days=1000)
    df, messages = dp._coingecko_fetch_price_history.__wrapped__("bitcoin", far_past, end)
    expected_clamp = (end - datetime.timedelta(days=dp.COINGECKO_MAX_HISTORY_DAYS)).isoformat()
    assert any(expected_clamp in m for m in messages)


def test_price_history_no_data_returned_is_an_error_not_empty_success(monkeypatch):
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: ({"prices": [], "total_volumes": []}, None))
    df, messages = dp._coingecko_fetch_price_history.__wrapped__("bitcoin", None, None)
    assert df.empty
    assert messages  # a disclosed reason, not silence


def test_price_history_fetch_failure_returns_empty_df_and_error(monkeypatch):
    monkeypatch.setattr(dp, "_fetch_json_with_retry", lambda url, params, label: (None, "unknown CoinGecko coin id"))
    df, messages = dp._coingecko_fetch_price_history.__wrapped__("not-a-real-coin", None, None)
    assert df.empty
    assert messages == ["unknown CoinGecko coin id"]


# --- CoinGeckoProvider (the DataProvider-facing wrapper) -----------------------

def test_coingecko_provider_fetch_info_strips_prefix(monkeypatch):
    captured = {}
    def fake(coin_id):
        captured["coin_id"] = coin_id
        return ({"name": "Ethereum"}, [])
    monkeypatch.setattr(dp, "_coingecko_fetch_info", fake)
    provider = CoinGeckoProvider()
    info, errors = provider.fetch_info("crypto:ethereum")
    assert captured["coin_id"] == "ethereum"
    assert info == {"name": "Ethereum"}


def test_coingecko_provider_fetch_price_history_strips_prefix(monkeypatch):
    captured = {}
    def fake(coin_id, start, end):
        captured["coin_id"] = coin_id
        return (pd.DataFrame(), [])
    monkeypatch.setattr(dp, "_coingecko_fetch_price_history", fake)
    provider = CoinGeckoProvider()
    provider.fetch_price_history("crypto:ethereum", None, None)
    assert captured["coin_id"] == "ethereum"
