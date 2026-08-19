"""Tests for historical_comparison.py.

The tests that matter most here are the look-ahead ones. A replay that
quietly reuses today's balance sheet would still "work" and still render a
plausible table — it would just be wrong in a way nobody notices. So the
point-in-time boundary is asserted directly rather than inferred from the
output looking reasonable.
"""
import datetime

import pandas as pd
import pytest

import historical_comparison as hc
from data_loader import TickerBundle
from historical_comparison import (
    ComparedMetric,
    _STATIC_INFO_KEYS,
    available_range,
    bundle_as_of,
    statement_period_for,
    statement_periods,
)


# The REAL TickerBundle, not a stand-in: standardize_financials is
# @st.cache_data-wrapped and hashes its argument, so a hand-rolled fake
# fails there for reasons that have nothing to do with this module. Using
# the real type also means these tests exercise the same object the app
# passes in.
def _stmt(values_by_period, row_name):
    cols = [pd.Timestamp(d) for d in values_by_period]
    return pd.DataFrame([list(values_by_period.values())], index=[row_name], columns=cols)


def _bundle():
    idx = pd.date_range("2023-01-02", periods=600, freq="B")
    prices = pd.DataFrame({
        "Open": range(100, 100 + len(idx)),
        "High": range(101, 101 + len(idx)),
        "Low": range(99, 99 + len(idx)),
        "Close": range(100, 100 + len(idx)),
        "Volume": [1_000_000] * len(idx),
    }, index=idx)
    income = _stmt({"2023-09-30": 1000.0, "2024-09-30": 2000.0}, "Total Revenue")
    balance = _stmt({"2023-09-30": 50.0, "2024-09-30": 40.0}, "Ordinary Shares Number")
    cash = _stmt({"2023-09-30": 10.0, "2024-09-30": 20.0}, "Free Cash Flow")
    info = {
        "longName": "Test Co", "sector": "Technology", "website": "x.com",
        # every one of these is a today-snapshot that must NOT survive
        "trailingPE": 99.0, "marketCap": 1e12, "beta": 1.7, "currentPrice": 500.0,
        "profitMargins": 0.44, "currentRatio": 3.3, "heldPercentInsiders": 0.1,
        "totalDebt": 123.0, "sharesOutstanding": 999.0,
    }
    return TickerBundle(
        ticker="TEST", info=info, price_history=prices,
        income_stmt=income, balance_sheet=balance, cash_flow=cash,
        institutional_holders=pd.DataFrame({"x": [1]}),
        insider_transactions=pd.DataFrame({"y": [1]}),
    )


# --- statement period selection -------------------------------------------------

def test_statement_periods_are_discovered_and_sorted():
    assert statement_periods(_bundle()) == (datetime.date(2023, 9, 30), datetime.date(2024, 9, 30))


def test_period_in_force_is_the_latest_filing_on_or_before_the_date():
    b = _bundle()
    assert statement_period_for(b, datetime.date(2024, 3, 15)) == datetime.date(2023, 9, 30)
    assert statement_period_for(b, datetime.date(2025, 1, 1)) == datetime.date(2024, 9, 30)


def test_the_filing_date_itself_counts_as_in_force():
    assert statement_period_for(_bundle(), datetime.date(2023, 9, 30)) == datetime.date(2023, 9, 30)


def test_a_date_before_every_filing_has_no_period():
    """Common in practice — Yahoo only returns about five annual periods."""
    assert statement_period_for(_bundle(), datetime.date(2020, 1, 1)) is None


# --- the look-ahead boundary ----------------------------------------------------

def test_future_statement_columns_are_removed():
    hist = bundle_as_of(_bundle(), datetime.date(2024, 3, 15))
    kept = [pd.Timestamp(c).date() for c in hist.income_stmt.columns]
    assert kept == [datetime.date(2023, 9, 30)]


def test_future_price_bars_are_removed():
    as_of = datetime.date(2024, 3, 15)
    hist = bundle_as_of(_bundle(), as_of)
    assert len(hist.price_history) > 0
    assert max(i.date() for i in hist.price_history.index) <= as_of


def test_only_static_descriptors_survive_from_info():
    """The core anti-look-ahead assertion: every numeric field on Yahoo's
    info dict is today's snapshot with no history, so carrying any of them
    into a replay would fabricate knowledge of the future."""
    hist = bundle_as_of(_bundle(), datetime.date(2024, 3, 15))
    assert set(hist.info) <= set(_STATIC_INFO_KEYS)
    for leaked in ("trailingPE", "marketCap", "beta", "currentPrice",
                   "profitMargins", "currentRatio", "totalDebt", "sharesOutstanding"):
        assert leaked not in hist.info, f"{leaked} leaked into the historical bundle"


def test_ownership_snapshots_are_dropped():
    hist = bundle_as_of(_bundle(), datetime.date(2024, 3, 15))
    assert hist.institutional_holders is None
    assert hist.insider_transactions is None


def test_the_original_bundle_is_not_mutated():
    """bundle_as_of must not damage the live bundle the rest of the page
    is still rendering from."""
    b = _bundle()
    before_cols = list(b.income_stmt.columns)
    before_bars = len(b.price_history)
    before_info = dict(b.info)
    bundle_as_of(b, datetime.date(2024, 3, 15))
    assert list(b.income_stmt.columns) == before_cols
    assert len(b.price_history) == before_bars
    assert b.info == before_info


def test_shares_outstanding_is_read_from_the_period_in_force():
    """Point-in-time shares are what make a reconstructed market cap and
    P/E honest — today's share count would import the future."""
    b = _bundle()
    assert hc._shares_from(bundle_as_of(b, datetime.date(2024, 3, 15))) == 50.0
    assert hc._shares_from(bundle_as_of(b, datetime.date(2025, 1, 1))) == 40.0


# --- available range ------------------------------------------------------------

def test_available_range_follows_the_loaded_price_history():
    earliest, latest = available_range(_bundle())
    assert earliest == datetime.date(2023, 1, 2)
    assert latest is not None and latest > earliest


def test_available_range_is_empty_without_price_history():
    b = _bundle()
    b.price_history = pd.DataFrame()
    assert available_range(b) == (None, None)


# --- graceful degradation -------------------------------------------------------

def test_too_few_bars_yields_no_price_metrics_and_says_why():
    b = _bundle()
    result = hc.build_comparison(b, datetime.date(2023, 1, 5))
    assert result.has_price_metrics is False
    assert any("trading day" in w for w in result.warnings)


def test_a_date_before_every_filing_warns_and_omits_fundamentals():
    b = _bundle()
    result = hc.build_comparison(b, datetime.date(2023, 6, 1))
    assert result.has_fundamentals is False
    assert any("statement" in w.lower() for w in result.warnings)


def test_build_comparison_never_raises_on_an_empty_bundle():
    """Missing historical data is handled gracefully per the acceptance
    criteria — an empty bundle must degrade, not explode."""
    b = _bundle()
    b.price_history = pd.DataFrame()
    b.income_stmt = pd.DataFrame()
    b.balance_sheet = pd.DataFrame()
    b.cash_flow = pd.DataFrame()
    result = hc.build_comparison(b, datetime.date(2024, 3, 15))
    assert result.has_price_metrics is False
    assert result.has_fundamentals is False


# --- the compared-metric contract ------------------------------------------------

def test_delta_is_none_when_either_side_is_missing():
    """A delta against an unknown value would be fabricated."""
    assert ComparedMetric("m", "Risk", None, 5.0).delta is None
    assert ComparedMetric("m", "Risk", 5.0, None).delta is None


def test_delta_is_now_minus_then():
    assert ComparedMetric("m", "Risk", 2.0, 5.0).delta == 3.0


def test_every_metric_is_in_a_known_group():
    result = hc.build_comparison(_bundle(), datetime.date(2024, 3, 15))
    assert {m.group for m in result.metrics} <= {"Fundamentals", "Technicals", "Risk"}
    assert result.metrics, "expected metrics to be emitted even when values are unavailable"


def test_missing_price_is_not_blamed_on_missing_shares():
    """A warning that names the wrong cause is worse than none. This is the
    exact case seen live: the balance sheet DOES report shares, but there
    are too few bars before the date to price the company. The message must
    say that rather than claiming shares weren't reported.

    Needs a bundle whose price history starts just before the filing, so a
    filing is in force while the bar count is still under the floor."""
    b = _bundle()
    idx = pd.date_range("2024-09-24", periods=5, freq="B")
    b.price_history = pd.DataFrame({
        "Open": [100] * len(idx), "High": [101] * len(idx), "Low": [99] * len(idx),
        "Close": [100] * len(idx), "Volume": [1_000_000] * len(idx),
    }, index=idx)

    result = hc.build_comparison(b, datetime.date(2024, 9, 30))
    assert result.has_fundamentals is True          # a filing IS in force
    assert result.has_price_metrics is False        # but the price side is too thin
    joined = " ".join(result.warnings)
    assert "Shares outstanding wasn't reported" not in joined
    assert "enough price history" in joined


def test_missing_shares_is_reported_as_such():
    b = _bundle()
    b.balance_sheet = b.balance_sheet.drop(index="Ordinary Shares Number")
    result = hc.build_comparison(b, datetime.date(2025, 1, 1))
    assert any("Shares outstanding wasn't reported" in w for w in result.warnings)
