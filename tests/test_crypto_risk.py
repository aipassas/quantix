"""Risk for a market that never closes.

The measurement this file defends: Yahoo returns 365.6 bars a year for
BTC-USD and ETH-USD, against 251.0 for SPY and GLD, measured over the
three years to 2026-08-25. Annualising crypto with the equity factor of
252 understates Bitcoin's volatility by a fifth — 36.7% against a true
44.1% — and understates it in the direction that flatters the asset.
"""
import numpy as np
import pandas as pd
import pytest

import crypto_risk as cr


def _days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def _weekdays(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="B")


# --- the annualisation factor -------------------------------------------------

def test_crypto_annualises_over_the_calendar_year_not_the_trading_year():
    assert cr.BARS_PER_YEAR == 365
    assert cr.EQUITY_BARS_PER_YEAR == 252


def test_volatility_uses_365_and_the_difference_is_material():
    """Constructed so the answer is known in closed form: a series whose
    daily returns have a standard deviation of exactly 1% must annualise
    to 1% * sqrt(365) = 19.105%, not 1% * sqrt(252) = 15.875%."""
    rng = np.random.default_rng(11)
    raw = rng.normal(0, 0.01, 400)
    raw = (raw - raw.mean()) / raw.std() * 0.01      # exactly 1% sigma
    closes = pd.Series(100 * np.cumprod(1 + raw), index=_days(401)[1:])
    window = cr.volatility_windows(closes, (("90-day", 90),))[0]
    returns = closes.pct_change().dropna().iloc[-90:]
    expected = float(returns.std() * (365 ** 0.5) * 100)
    assert window.annualised_pct == pytest.approx(expected, rel=1e-9)
    wrong = float(returns.std() * (252 ** 0.5) * 100)
    assert window.annualised_pct == pytest.approx(wrong * (365 / 252) ** 0.5,
                                                  rel=1e-9)
    assert window.annualised_pct > wrong * 1.19       # ~20% higher


def test_a_window_longer_than_the_history_is_unavailable_not_zero():
    """A 365-day volatility computed off 60 bars would carry a label that
    lies about how much history stands behind it — and a coin listed
    last month would show a confident annual figure."""
    closes = pd.Series(np.linspace(100, 120, 60), index=_days(60))
    windows = {w.label: w for w in cr.volatility_windows(closes)}
    assert windows["30-day"].ok
    assert not windows["1-year"].ok
    assert windows["1-year"].annualised_pct is None
    assert windows["1-year"].status == "Unavailable"


def test_every_requested_window_is_returned_even_when_blind():
    """Silently dropping the windows that could not be computed would
    make a partial answer look complete."""
    windows = cr.volatility_windows(pd.Series(dtype="float64"))
    assert len(windows) == len(cr.VOL_WINDOWS)
    assert all(not w.ok for w in windows)


def test_the_annualisation_note_names_both_factors():
    assert "365" in cr.ANNUALISATION_NOTE and "252" in cr.ANNUALISATION_NOTE


# --- drawdown -----------------------------------------------------------------

def test_max_drawdown_is_the_worst_peak_to_trough_fall():
    closes = pd.Series([100, 120, 60, 90, 80], index=_days(5))
    profile = cr.drawdown_profile(closes)
    assert profile.max_drawdown_pct == pytest.approx(-50.0)
    assert profile.peak_date == closes.index[1]
    assert profile.trough_date == closes.index[2]


def test_the_peak_is_taken_BEFORE_the_trough_not_after():
    """A later, higher peak must not be reported as the one the trough
    fell from — the drawdown would be measured from a price that had not
    happened yet."""
    closes = pd.Series([100, 120, 60, 200], index=_days(4))
    profile = cr.drawdown_profile(closes)
    assert profile.peak_date == closes.index[1]
    assert profile.max_drawdown_pct == pytest.approx(-50.0)


def test_the_current_drawdown_is_measured_against_the_running_peak():
    closes = pd.Series([100, 200, 150], index=_days(3))
    profile = cr.drawdown_profile(closes)
    assert profile.current_drawdown_pct == pytest.approx(-25.0)
    assert profile.recovered is False
    assert profile.days_since_peak == 1


def test_a_series_at_a_new_high_reports_recovered():
    closes = pd.Series([100, 60, 140], index=_days(3))
    assert cr.drawdown_profile(closes).recovered is True


def test_no_history_is_an_error_not_a_zero_drawdown():
    assert not cr.drawdown_profile(None).ok
    assert not cr.drawdown_profile(pd.Series([1.0], index=_days(1))).ok


# --- correlation across two different calendars -------------------------------

def test_correlation_uses_only_the_days_both_markets_traded():
    """THE CALENDAR TRAP. Crypto trades 365 days and equities ~252, so
    the overlap is the equity calendar. The count must reflect that, not
    the length of the longer series."""
    rng = np.random.default_rng(5)
    crypto = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 400)),
                       index=_days(400))
    equity_index = _weekdays(280)
    equity = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 280)),
                       index=equity_index)
    reading = cr.correlate(crypto, equity, "US equities", "SPY")
    assert reading.ok
    assert reading.observations < 300
    assert reading.observations >= cr.MIN_CORRELATION_DAYS


def test_a_series_correlated_with_itself_is_exactly_one():
    rng = np.random.default_rng(7)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)),
                       index=_days(200))
    assert cr.correlate(closes, closes, "self", "X").coefficient == pytest.approx(1.0)


def test_an_exactly_inverted_series_is_strongly_negative():
    """A magnitude-only assertion passes an implementation that dropped
    the sign, so the sign is pinned."""
    rng = np.random.default_rng(9)
    returns = rng.normal(0, 0.02, 200)
    up = pd.Series(100 * np.cumprod(1 + returns), index=_days(200))
    down = pd.Series(100 * np.cumprod(1 - returns), index=_days(200))
    reading = cr.correlate(up, down, "inverse", "X")
    assert reading.coefficient < -0.98
    assert reading.strength == "Strong"


def test_too_little_overlap_returns_unavailable_rather_than_a_number():
    """Two weeks of overlap will produce a coefficient, and it will mean
    nothing."""
    rng = np.random.default_rng(3)
    a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)),
                  index=_days(200, "2024-01-01"))
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 20)),
                  index=_days(20, "2024-06-20"))
    reading = cr.correlate(a, b, "short", "X")
    assert not reading.ok
    assert reading.strength == "Unavailable"
    assert str(cr.MIN_CORRELATION_DAYS) in reading.error


def test_a_flat_series_reports_undefined_rather_than_nan():
    index = _days(200)
    rng = np.random.default_rng(4)
    moving = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)),
                       index=index)
    flat = pd.Series(100.0, index=index)
    reading = cr.correlate(moving, flat, "flat", "X")
    assert not reading.ok and "undefined" in reading.error


def test_timezone_stamped_indexes_still_overlap():
    """Yahoo stamps crypto and equity bars in different exchange
    timezones. A raw join across those produced almost no overlap."""
    rng = np.random.default_rng(6)
    a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)),
                  index=_days(200).tz_localize("UTC"))
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)),
                  index=_days(200).tz_localize("America/New_York"))
    assert cr.correlate(a, b, "tz", "X").observations >= 190


def test_a_coin_is_not_offered_a_correlation_against_itself():
    """It returns 1.00, which is not a finding, and takes a row a real
    comparison could have used."""
    pairs = cr.benchmarks_for("BTC-USD")
    assert "BTC-USD" not in [s for _, s in pairs]
    assert "ETH-USD" in [s for _, s in pairs]
    assert len(cr.benchmarks_for("SOL-USD")) == len(cr.CORRELATION_BENCHMARKS)


# --- leverage -----------------------------------------------------------------

def test_the_move_to_liquidation_is_the_inverse_of_leverage():
    read = cr.liquidation_distance(100.0, 10, maintenance_margin_pct=0.0)
    assert read.move_to_liquidation_pct == pytest.approx(10.0)
    assert read.liquidation_price == pytest.approx(90.0)
    assert cr.liquidation_distance(
        100.0, 20, maintenance_margin_pct=0.0).move_to_liquidation_pct == pytest.approx(5.0)


def test_the_maintenance_margin_shortens_the_distance():
    read = cr.liquidation_distance(100.0, 10, maintenance_margin_pct=0.5)
    assert read.move_to_liquidation_pct == pytest.approx(9.5)


def test_a_short_is_liquidated_ABOVE_the_entry():
    read = cr.liquidation_distance(100.0, 10, maintenance_margin_pct=0.0,
                                   long=False)
    assert read.liquidation_price == pytest.approx(110.0)


def test_an_unleveraged_position_cannot_be_liquidated():
    """1x has no borrowed margin. Reporting "100% move to liquidation"
    would suggest a spot holding can be closed out by the venue."""
    read = cr.liquidation_distance(100.0, 1)
    assert not read.ok and "unleveraged" in read.error.lower()


def test_an_impossible_leverage_and_margin_pair_is_refused():
    read = cr.liquidation_distance(100.0, 250, maintenance_margin_pct=1.0)
    assert not read.ok


def test_liquidation_context_is_expressed_in_the_coins_own_volatility():
    """A 9.5% move means nothing in isolation. Against a coin whose
    30-day volatility annualises to 45%, it is about 4 sigma."""
    windows = (cr.VolatilityWindow("30-day", 30, 45.0, 30),)
    text = cr.liquidation_context(9.5, windows)
    daily_sigma = 45.0 / (365 ** 0.5)
    assert f"{9.5 / daily_sigma:.1f}" in text
    assert "30-day" in text


def test_liquidation_context_is_silent_when_nothing_was_measured():
    assert cr.liquidation_context(9.5, ()) == ""
    assert cr.liquidation_context(None, (cr.VolatilityWindow("30-day", 30, 45.0, 30),)) == ""


# --- developer health ---------------------------------------------------------

class _Profile:
    def __init__(self, commits, contributors=100, stars=1):
        self.commits_4w = commits
        self.contributors = contributors
        self.stars = stars
        self.has_developer_data = True


def test_developer_activity_separates_the_measured_coins():
    """Measured: 108 (BTC), 41 (ETH), 171 (SOL), 0 (DOGE) commits in four
    weeks. The bands must actually separate those."""
    assert cr.developer_health(_Profile(108)).verdict == "Actively developed"
    assert cr.developer_health(_Profile(41)).verdict == "Actively developed"
    assert cr.developer_health(_Profile(171)).verdict == "Actively developed"
    assert cr.developer_health(_Profile(0)).verdict == "Dormant repository"
    assert cr.developer_health(_Profile(5)).verdict == "Low activity"


def test_a_dormant_repository_is_not_called_a_dead_network():
    """A chain can run untouched for years. The wording must not turn a
    quiet repository into a verdict on the protocol."""
    detail = cr.developer_health(_Profile(0)).detail
    assert "not the network" in detail


def test_no_repository_data_is_unavailable_rather_than_dormant():
    assert cr.developer_health(None).verdict == "Unavailable"


# --- what is deliberately not scored ------------------------------------------

def test_the_unscored_risks_say_why_rather_than_going_quiet():
    for note in (cr.REGULATORY_UNAVAILABLE, cr.AUDIT_UNAVAILABLE,
                 cr.HACK_HISTORY_UNAVAILABLE):
        assert len(note) > 80
        assert note.strip().endswith(".")


def test_no_regulatory_score_is_computed_anywhere_in_the_module():
    """A hand-assembled jurisdiction score is an opinion rendered as
    data, on a question where being wrong has legal consequences."""
    import inspect
    names = [n for n in dir(cr) if not n.startswith("_")]
    assert not [n for n in names
                if "regulat" in n.lower() and n != "REGULATORY_UNAVAILABLE"]
