"""Dominance, trend, levels and on-chain activity."""
import numpy as np
import pandas as pd
import pytest

import crypto_data as cd
import crypto_market as cm


def _days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def _market(**kwargs):
    base = dict(dominance={"btc": 59.139, "eth": 11.071, "usdt": 6.899},
                total_market_cap=2.64e12)
    base.update(kwargs)
    return cd.GlobalMarket(**base)


# --- dominance ----------------------------------------------------------------

def test_dominance_comes_from_the_reported_total_not_a_page_sum():
    """Summing the 250-coin page would divide by a total that excludes
    eighteen thousand other coins, inflating every share. The loader must
    read /global."""
    import inspect
    assert "market_cap_percentage" in inspect.getsource(cd.load_global)


def test_dominance_reads_the_coins_share_and_its_rank():
    reading = cm.dominance("BTC-USD", _market())
    assert reading.ok
    assert reading.share_pct == pytest.approx(59.139)
    assert reading.rank == 1


def test_the_largest_share_is_not_called_the_1st_largest():
    assert "the largest share" in cm.describe_dominance(
        cm.Dominance("btc", 59.1, 1, 2.64e12))
    assert "2nd largest" in cm.describe_dominance(
        cm.Dominance("eth", 11.1, 2, 2.64e12))
    assert "3rd largest" in cm.describe_dominance(
        cm.Dominance("sol", 2.1, 3, 2.64e12))
    assert "11th largest" in cm.describe_dominance(
        cm.Dominance("x", 0.1, 11, 2.64e12))


def test_a_coin_the_totals_do_not_break_out_is_unavailable():
    reading = cm.dominance("SHIB-USD", _market())
    assert not reading.ok and reading.error


def test_a_failed_market_fetch_propagates_its_own_message():
    reading = cm.dominance("BTC-USD", cd.GlobalMarket(error="provider down"))
    assert not reading.ok and reading.error == "provider down"


# --- the cross that could not be checked --------------------------------------

def test_a_golden_cross_is_detected():
    closes = pd.Series(np.linspace(100, 300, 400), index=_days(400))
    cross = cm.moving_average_cross(closes)
    assert cross.state == cm.FIRED and cross.golden is True
    assert cross.fast > cross.slow
    assert "golden" in cross.detail


def test_a_death_cross_is_detected():
    closes = pd.Series(np.linspace(300, 100, 400), index=_days(400))
    cross = cm.moving_average_cross(closes)
    assert cross.state == cm.NOT_FIRED and cross.golden is False
    assert "death" in cross.detail


def test_too_little_history_is_UNCHECKED_not_NOT_FIRED():
    """A rule that could not be evaluated is not a rule that did not
    fire. Rendering it as "no cross" is an all-clear nobody performed."""
    closes = pd.Series(np.linspace(100, 200, 120), index=_days(120))
    cross = cm.moving_average_cross(closes)
    assert cross.state == cm.UNCHECKED
    assert cross.golden is None
    assert not cross.evaluable
    assert "200" in cross.detail and "120" in cross.detail


def test_the_cross_recomputes_its_own_averages():
    """Indicator columns on the chart frame exist only when the display
    checkbox is ticked, so a signal reading them would report missing
    data because a user unticked a box."""
    import inspect
    source = inspect.getsource(cm.moving_average_cross)
    assert "rolling" in source
    assert "SMA_50" not in source and "SMA_200" not in source


def test_exactly_the_slow_window_is_enough():
    """Off-by-one at the boundary: 200 bars must be evaluable."""
    closes = pd.Series(np.linspace(100, 200, 200), index=_days(200))
    assert cm.moving_average_cross(closes).evaluable


# --- range and pivots ---------------------------------------------------------

def test_range_position_is_zero_at_the_low_and_100_at_the_high():
    rising = pd.Series(np.linspace(50, 150, 100), index=_days(100))
    assert cm.range_position(rising).position_pct == pytest.approx(100.0)
    falling = pd.Series(np.linspace(150, 50, 100), index=_days(100))
    assert cm.range_position(falling).position_pct == pytest.approx(0.0)


def test_range_position_is_the_midpoint_for_a_price_halfway_up():
    closes = pd.Series([100.0] * 50 + [200.0] * 50 + [150.0], index=_days(101))
    assert cm.range_position(closes).position_pct == pytest.approx(50.0)


def test_a_flat_price_reports_no_movement_rather_than_dividing_by_zero():
    flat = pd.Series(100.0, index=_days(50))
    reading = cm.range_position(flat)
    assert not reading.ok and "did not move" in reading.error


def test_a_pivot_needs_bars_on_BOTH_sides():
    """The last `window` bars can never be pivots by construction. A high
    that has not been tested from the right is not a level anything has
    bounced off — and an implementation that treated the final bar as a
    pivot would always report the current price as resistance."""
    values = [10.0] * 12 + [99.0] + [10.0] * 12 + [98.0]   # 98 is last
    closes = pd.Series(values, index=_days(len(values)))
    highs, lows = cm.pivots(closes, window=10)
    assert 99.0 in highs
    assert 98.0 not in highs


def test_support_is_below_the_price_and_resistance_above_it():
    rng = np.random.default_rng(21)
    wave = 100 + 20 * np.sin(np.linspace(0, 12 * np.pi, 400))
    closes = pd.Series(wave + rng.normal(0, 0.5, 400), index=_days(400))
    levels = cm.support_resistance(closes)
    price = float(closes.iloc[-1])
    assert levels.ok
    if levels.support is not None:
        assert levels.support < price
    if levels.resistance is not None:
        assert levels.resistance > price
    assert levels.pivot_count > 0


def test_no_resistance_above_means_the_price_is_at_its_range_top():
    """Absence is information here, not a failure. Built as an uptrend
    WITH pullbacks, so pivot lows exist below while nothing has traded
    above the current price."""
    rng = np.random.default_rng(31)
    wave = np.linspace(100, 300, 400) + 12 * np.sin(np.linspace(0, 10 * np.pi, 400))
    wave[-1] = wave.max() + 5              # finish at a new high
    closes = pd.Series(wave + rng.normal(0, 0.2, 400), index=_days(400))
    levels = cm.support_resistance(closes)
    assert levels.resistance is None
    assert levels.support is not None
    assert levels.support < float(closes.iloc[-1])


def test_a_pure_trend_line_has_no_pivots_at_all():
    """Every bar of a strictly monotonic series is beaten by a neighbour
    on one side, so nothing is a local extreme. Reporting the window's
    endpoints as "levels" would invent support that nothing bounced
    off."""
    closes = pd.Series(np.linspace(100, 300, 400), index=_days(400))
    levels = cm.support_resistance(closes)
    assert levels.pivot_count == 0
    assert levels.support is None and levels.resistance is None
    assert not levels.ok
    # The extremes are still reported, because those ARE observed.
    assert levels.window_high == pytest.approx(300.0)


def test_a_short_history_refuses_to_produce_levels():
    closes = pd.Series(np.linspace(100, 110, 15), index=_days(15))
    assert not cm.support_resistance(closes).ok


# --- relative volume ----------------------------------------------------------

def test_a_doubling_of_volume_reads_as_exactly_two():
    """THE BASELINE TRAP. Including the bar under test in its own
    trailing average turns a true doubling into about 1.9."""
    volumes = pd.Series([100.0] * 30 + [200.0], index=_days(31))
    reading = cm.relative_volume(volumes, window_days=30)
    assert reading.ratio == pytest.approx(2.0)
    assert reading.average == pytest.approx(100.0)


def test_the_trailing_average_excludes_the_current_bar():
    import inspect
    source = inspect.getsource(cm.relative_volume)
    assert "-1]" in source          # the slice stops before the last bar


def test_relative_volume_needs_one_more_bar_than_the_window():
    volumes = pd.Series([100.0] * 30, index=_days(30))
    reading = cm.relative_volume(volumes, window_days=30)
    assert not reading.ok and "31" in reading.error


def test_zero_average_volume_is_an_error_not_an_infinity():
    volumes = pd.Series([0.0] * 30 + [50.0], index=_days(31))
    assert not cm.relative_volume(volumes, window_days=30).ok


# --- on-chain activity --------------------------------------------------------

def test_activity_reports_a_level_and_a_thirty_day_change():
    series = pd.Series(np.linspace(100.0, 200.0, 61), index=_days(61))
    reading = cm.activity("active_addresses", series)
    assert reading.ok
    assert reading.latest == pytest.approx(200.0)
    earlier = float(series.iloc[-31])
    assert reading.change_30d_pct == pytest.approx(
        100.0 * (200.0 - earlier) / earlier)
    assert reading.label == "Active addresses"


def test_a_short_series_reports_a_level_without_a_change():
    series = pd.Series([100.0, 110.0], index=_days(2))
    reading = cm.activity("hash_rate", series)
    assert reading.ok and reading.change_30d_pct is None


def test_a_fetch_error_is_carried_through_rather_than_swallowed():
    reading = cm.activity("miner_revenue", None, error="provider down")
    assert not reading.ok and reading.error == "provider down"
    assert reading.label == "Miner revenue"


def test_the_summary_counts_what_resolved_rather_than_scoring_out_of_all():
    readings = [cm.activity("hash_rate", pd.Series([1.0, 2.0], index=_days(2))),
                cm.activity("miner_revenue", None, error="down")]
    text = cm.activity_summary(readings)
    assert "1 of 2" in text


def test_a_wholly_failed_fetch_explains_the_bitcoin_only_scope():
    text = cm.activity_summary([cm.activity("hash_rate", None, error="down")])
    assert "Bitcoin only" in text
