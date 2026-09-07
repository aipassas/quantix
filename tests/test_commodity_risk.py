"""Volatility, season, hedging and shocks.

The measurement this file defends: futures return 251.6 bars a year
against SPY's 251.5, measured over three years to 2026-08-26. The 252
factor is right here, and that was checked rather than inherited from
either the equity default or the crypto exception.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_risk as cr


def _days(n, start="2016-01-01", freq="B"):
    return pd.date_range(start, periods=n, freq=freq)


# --- the annualisation factor, checked rather than assumed --------------------

def test_commodities_use_the_EQUITY_trading_year():
    """After crypto's 365 the tempting inference is that every non-equity
    class needs its own factor. Futures keep the exchange calendar."""
    import config

    assert cr.BARS_PER_YEAR == 252
    # It must AGREE with the app-wide constant, not shadow it: introducing
    # a local factor that happens to equal the shared one is how the two
    # drift apart later.
    assert cr.BARS_PER_YEAR == config.RISK.trading_days_per_year


def test_the_annualisation_note_records_the_measurement():
    assert "251.6" in cr.ANNUALISATION_NOTE
    assert "252" in cr.ANNUALISATION_NOTE


def test_volatility_is_the_standard_deviation_times_root_252():
    rng = np.random.default_rng(4)
    raw = rng.normal(0, 0.015, 400)
    closes = pd.Series(100 * np.cumprod(1 + raw), index=_days(400))
    window = cr.volatility_windows(closes, (("90-day", 90),))[0]
    returns = closes.pct_change().dropna().iloc[-90:]
    assert window.annualised_pct == pytest.approx(
        float(returns.std() * (252 ** 0.5) * 100), rel=1e-9)


def test_a_window_longer_than_the_history_is_unavailable_not_zero():
    closes = pd.Series(np.linspace(100, 110, 60), index=_days(60))
    windows = {w.label: w for w in cr.volatility_windows(closes)}
    assert windows["30-day"].ok
    assert not windows["1-year"].ok
    assert windows["1-year"].status == "Unavailable"


def test_every_requested_window_comes_back_even_when_blind():
    windows = cr.volatility_windows(pd.Series(dtype="float64"))
    assert len(windows) == len(cr.VOL_WINDOWS)
    assert all(not w.ok for w in windows)


# --- seasonality --------------------------------------------------------------

def _seasonal_series(bump_month, bump_pct=8.0, years=10):
    """A price path with a known, repeating bump in one calendar month."""
    index = pd.date_range("2016-01-31", periods=years * 12, freq="ME")
    level, values = 100.0, []
    for stamp in index:
        level *= 1 + (bump_pct / 100.0 if stamp.month == bump_month else 0.0)
        values.append(level)
    return pd.Series(values, index=index)


def test_seasonality_finds_a_month_that_is_really_there():
    """Constructed so the answer is known: only July moves."""
    reading = cr.seasonality(_seasonal_series(7))
    assert reading.ok
    july = next(m for m in reading.months if m.month == 7)
    assert july.average_pct == pytest.approx(8.0, abs=0.01)
    assert reading.strongest.month == 7


def test_every_month_is_returned_whether_or_not_it_scored():
    """Dropping the unscored months would make a partial history look
    like a set of months that happen not to matter."""
    reading = cr.seasonality(_seasonal_series(7))
    assert len(reading.months) == 12


def test_a_month_with_too_few_observations_is_not_scored():
    short = _seasonal_series(7, years=1)
    reading = cr.seasonality(short)
    scored = [m for m in reading.months if m.scored]
    assert not scored or all(
        m.observations >= cr.MIN_SEASONAL_OBSERVATIONS for m in scored)


def test_the_sample_size_reaches_the_reader():
    """A seasonal chart that does not say how many years it averaged
    invites trust in a shape drawn through three points."""
    text = cr.describe_seasonality(cr.seasonality(_seasonal_series(7)))
    assert "observations" in text
    assert "years" in text


def test_the_spliced_series_caveat_exists_and_is_not_duplicated():
    """Yahoo's continuous front-month series rolls one contract onto the
    next, so its month-to-month change mixes price with the roll. The
    caveat belongs on the chart caption; repeating it in the description
    printed the same paragraph twice on screen."""
    assert "splice" in cr.CONTINUOUS_SERIES_NOTE.lower()
    assert "roll" in cr.CONTINUOUS_SERIES_NOTE.lower()
    text = cr.describe_seasonality(cr.seasonality(_seasonal_series(7)))
    assert cr.CONTINUOUS_SERIES_NOTE not in text


def test_too_little_history_is_an_error_not_a_flat_pattern():
    assert not cr.seasonality(None).ok
    assert not cr.seasonality(pd.Series([1.0, 2.0])).ok


# --- correlation and the hedging thesis ---------------------------------------

def test_a_series_correlated_with_itself_is_one():
    rng = np.random.default_rng(8)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 300)),
                       index=_days(300))
    assert cr.correlate(closes, closes, "self", "X").coefficient == pytest.approx(1.0)


def test_an_inverted_series_is_strongly_negative():
    """A magnitude-only assertion passes an implementation that dropped
    the sign — and the sign is the whole content of a dollar hedge."""
    rng = np.random.default_rng(9)
    returns = rng.normal(0, 0.02, 300)
    up = pd.Series(100 * np.cumprod(1 + returns), index=_days(300))
    down = pd.Series(100 * np.cumprod(1 - returns), index=_days(300))
    reading = cr.correlate(up, down, "inverse", "X")
    assert reading.coefficient < -0.98
    assert reading.strength == "Strong"


def test_too_little_overlap_is_unavailable_rather_than_a_number():
    rng = np.random.default_rng(2)
    a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 300)),
                  index=_days(300, "2016-01-01"))
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 20)),
                  index=_days(20, "2016-06-01"))
    reading = cr.correlate(a, b, "short", "X")
    assert not reading.ok and reading.strength == "Unavailable"


def test_the_hedge_benchmarks_test_what_a_commodity_is_bought_for():
    symbols = {s for _, s, _ in cr.HEDGE_BENCHMARKS}
    assert "DX-Y.NYB" in symbols        # the dollar
    assert "TIP" in symbols             # inflation
    assert "SPY" in symbols             # diversification
    for label, symbol, why in cr.HEDGE_BENCHMARKS:
        assert len(why) > 40, symbol


def test_a_negative_dollar_correlation_reads_as_hedging():
    reading = cr.Correlation("US dollar", "DX-Y.NYB", -0.45, 500)
    verdict, detail = cr.hedge_verdict(reading)
    assert "dollar hedge" in verdict
    assert "-0.45" in detail


def test_a_positive_dollar_correlation_says_the_hedge_is_not_working():
    """Stated as an observation about co-movement, never as advice."""
    verdict, detail = cr.hedge_verdict(
        cr.Correlation("US dollar", "DX-Y.NYB", 0.35, 500))
    assert "Not moving against the dollar" == verdict
    assert "supply and" in detail


def test_an_unavailable_correlation_gives_an_unavailable_verdict():
    assert cr.hedge_verdict(
        cr.Correlation("US dollar", "DX-Y.NYB", error="no data"))[0] == "Unavailable"


# --- shocks -------------------------------------------------------------------

def test_a_shock_is_applied_linearly_to_the_position():
    rows = cr.stress_test(price=100.0, position_value=100_000.0,
                          shocks=(-20.0,))
    row = rows[0]
    assert row.resulting_value == pytest.approx(80_000.0)
    assert row.profit_loss == pytest.approx(-20_000.0)
    assert row.price_after == pytest.approx(80.0)


def test_the_task_named_twenty_percent_and_it_is_bracketed():
    assert -20.0 in cr.SHOCK_PCTS
    assert min(cr.SHOCK_PCTS) < -20.0 < max(cr.SHOCK_PCTS)


def test_upside_shocks_are_offered_too():
    """A one-sided table implies a view about direction."""
    assert any(s > 0 for s in cr.SHOCK_PCTS)


def test_no_position_value_yields_rows_without_invented_numbers():
    rows = cr.stress_test(price=100.0, position_value=None)
    assert all(r.resulting_value is None for r in rows)
    assert len(rows) == len(cr.SHOCK_PCTS)


def test_a_shock_is_expressed_in_the_contracts_own_volatility():
    """A 20% move means nothing in isolation. Against a 30% annualised
    contract it is about a ten-sigma day and two-thirds of a year."""
    windows = (cr.VolatilityWindow("1-year", 252, 30.0, 252),)
    text = cr.shock_in_sigmas(-20.0, windows)
    daily = 30.0 / (252 ** 0.5)
    assert f"{20.0 / daily:.0f}" in text
    assert "0.67 of a typical year" in text


def test_shock_context_is_silent_when_nothing_was_measured():
    assert cr.shock_in_sigmas(-20.0, ()) == ""


# --- basis risk is named, not faked -------------------------------------------

def test_basis_risk_is_declared_unmeasurable_with_its_reason():
    assert "no spot series" in cr.BASIS_RISK_NOTE
    assert "calendar spread" in cr.BASIS_RISK_NOTE


def test_spread_volatility_needs_both_legs():
    frame = pd.DataFrame({"a": [1.0] * 40}, index=_days(40))
    assert cr.spread_volatility(frame, "a", "b") is None
    assert cr.spread_volatility(None, "a", "b") is None


def test_spread_volatility_annualises_the_change_in_the_spread():
    rng = np.random.default_rng(12)
    index = _days(300)
    near = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 300)), index=index)
    far = near + pd.Series(np.cumsum(rng.normal(0, 0.2, 300)), index=index)
    value = cr.spread_volatility(pd.DataFrame({"n": near, "f": far}),
                                 "n", "f")
    assert value is not None and value > 0
