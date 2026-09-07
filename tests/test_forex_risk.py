"""Volatility, tails, carry unwind and neighbours.

The measurement this file defends: FX returns 259.3 bars a year against
SPY's 251.5, because currency trades every weekday and keeps no exchange
holidays. Small — 1.6% on a volatility figure — but checked rather than
inherited from either the equity default or the crypto exception.
"""
import numpy as np
import pandas as pd
import pytest

import forex_data as fd
import forex_risk as fr


def _days(n, start="2016-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def _series(n=800, sigma=0.005, seed=7, drift=0.0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(drift, sigma, n)
    return pd.Series(100 * np.cumprod(1 + raw), index=_days(n))


# --- the annualisation factor -------------------------------------------------

def test_forex_annualises_over_260_days():
    assert fr.BARS_PER_YEAR == 260


def test_the_note_records_the_measurement_that_justifies_it():
    assert "259.3" in fr.ANNUALISATION_NOTE
    assert "251.5" in fr.ANNUALISATION_NOTE


def test_volatility_is_the_standard_deviation_times_root_260():
    closes = _series(600)
    window = fr.volatility_windows(closes, (("90-day", 90),))[0]
    returns = closes.pct_change().dropna().iloc[-90:]
    assert window.annualised_pct == pytest.approx(
        float(returns.std() * (260 ** 0.5) * 100), rel=1e-9)


def test_a_window_longer_than_the_history_is_unavailable_not_zero():
    closes = _series(60)
    windows = {w.label: w for w in fr.volatility_windows(closes)}
    assert windows["30-day"].ok
    assert not windows["1-year"].ok
    assert windows["1-year"].status == "Unavailable"


def test_no_typical_band_is_applied_and_the_reason_is_recorded():
    """The task states 8-12% as typical. Only four of twelve majors sit
    inside it — EUR/GBP is 3.5% — so a rule built on that band would
    flag most of a healthy board."""
    assert "8-12%" in fr.NO_TYPICAL_BAND
    assert "EUR/GBP" in fr.NO_TYPICAL_BAND
    names = [n for n in dir(fr) if not n.startswith("_")]
    assert not [n for n in names if "typical" in n.lower()
                and n != "NO_TYPICAL_BAND"]


# --- value at risk ------------------------------------------------------------

def test_var_is_sigma_times_the_z_for_the_confidence_asked_for():
    closes = _series(600)
    reading = fr.value_at_risk(closes, horizon_days=1, confidence=0.95)
    returns = closes.pct_change().dropna().iloc[-260:]
    assert reading.parametric_pct == pytest.approx(
        float(returns.std() * fr.Z_95 * 100), rel=1e-6)


def test_a_different_confidence_gets_a_DIFFERENT_z():
    """A one-entry lookup table silently reports a 95% figure under a
    99% label the moment a caller asks for one."""
    closes = _series(600)
    ninety_five = fr.value_at_risk(closes, confidence=0.95)
    ninety_nine = fr.value_at_risk(closes, confidence=0.99)
    assert ninety_nine.parametric_pct > ninety_five.parametric_pct * 1.3
    assert ninety_nine.label == "1-day VaR (99%)"


def test_a_longer_horizon_scales_with_the_square_root_of_time():
    closes = _series(600)
    one = fr.value_at_risk(closes, horizon_days=1)
    ten = fr.value_at_risk(closes, horizon_days=10)
    assert ten.parametric_pct == pytest.approx(
        one.parametric_pct * (10 ** 0.5), rel=1e-9)


def test_the_horizon_is_carried_in_the_label():
    """A VaR without its horizon is not a number anyone can act on."""
    assert fr.value_at_risk(_series(600), horizon_days=5).label.startswith("5-day")


def test_the_historical_figure_is_reported_beside_the_parametric_one():
    """Currency returns are not normal, so where the historical figure
    is larger the normal assumption is understating the tail — and the
    reader should be able to see that."""
    closes = _series(600)
    reading = fr.value_at_risk(closes)
    assert reading.historical_pct is not None
    assert str(reading.observations) in fr.describe_var(reading)


def test_too_little_history_is_an_error_not_a_zero_var():
    assert not fr.value_at_risk(pd.Series([1.0, 2.0])).ok
    assert not fr.value_at_risk(None).ok


# --- tails and carry unwind ---------------------------------------------------

def _skewed(n=800, seed=5, down=-0.06):
    """A series with a few large down days and no matching up days."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0002, 0.004, n)
    for i in range(60, n, 160):
        raw[i] = down
    return pd.Series(100 * np.cumprod(1 + raw), index=_days(n))


def test_the_tail_profile_reports_skew_with_its_extremes():
    """Skew is dominated by single days: EUR/GBP reads +2.15 purely on
    one bad print, so the extremes have to travel with it."""
    profile = fr.tail_profile(_skewed())
    assert profile.ok and profile.scored
    assert profile.skew < -0.1
    assert profile.worst_day_pct < profile.best_day_pct
    assert profile.worst_day_pct == pytest.approx(-6.0, abs=0.2)


def test_a_symmetric_series_is_not_called_asymmetric():
    """Checked across several seeds, because a single symmetric draw of
    800 points can skew past -0.1 by chance — one did, which is what
    made the threshold sample-size aware."""
    for seed in range(8):
        profile = fr.tail_profile(_series(800, seed=seed))
        assert profile.ok
        assert profile.asymmetric_down is False, seed


def test_the_skew_threshold_scales_with_the_sample():
    """Skew's standard error is sqrt(6/n), so what counts as asymmetric
    has to shrink as history grows."""
    small = fr.TailProfile(skew=-0.15, observations=800,
                           worst_day_pct=-1.0, best_day_pct=1.0)
    large = fr.TailProfile(skew=-0.15, observations=5000,
                           worst_day_pct=-1.0, best_day_pct=1.0)
    assert small.standard_error > large.standard_error
    assert small.asymmetric_down is False    # 0.15 < 2 x 0.087
    assert large.asymmetric_down is True     # 0.15 > 2 x 0.035


def test_too_few_observations_leaves_the_tail_UNSCORED():
    profile = fr.tail_profile(_series(100))
    assert profile.ok and not profile.scored


def test_carry_unwind_needs_BOTH_a_positive_carry_and_a_negative_skew():
    """Either alone is unremarkable. Being PAID to hold a position whose
    losses run larger than its gains is the shape that empties out."""
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    skewed = fr.tail_profile(_skewed())
    symmetric = fr.tail_profile(_series(800))

    label, detail = fr.unwind_risk(pair, 3.35, skewed)
    assert label == "Carry-unwind shape"
    # Funded in a haven, so the funding leg strengthens as it closes.
    assert "JPY" in detail

    assert fr.unwind_risk(pair, 3.35, symmetric)[0] == "Paid, without the tail"
    assert symmetric.asymmetric_down is False
    assert fr.unwind_risk(pair, -1.5, skewed)[0] == "Not a carry position"


def test_a_haven_funded_pair_says_why_that_matters():
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    assert pair.quote in fd.SAFE_HAVENS
    _, detail = fr.unwind_risk(pair, 3.35, fr.tail_profile(_skewed()))
    assert "risk appetite" in detail


def test_a_non_haven_funded_pair_omits_that_clause():
    pair = fd.PAIRS_BY_CODE["EURGBP"]
    assert pair.quote not in fd.SAFE_HAVENS
    _, detail = fr.unwind_risk(pair, 1.5, fr.tail_profile(_skewed()))
    assert "risk appetite" not in detail


def test_unwind_needs_a_carry_and_a_tail_to_say_anything():
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    assert fr.unwind_risk(pair, None, fr.tail_profile(_series(800)))[0] == "Unavailable"
    assert fr.unwind_risk(pair, 3.35, fr.tail_profile(None))[0] == "Unavailable"


def test_a_short_history_is_unscored_rather_than_given_a_verdict():
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    label, detail = fr.unwind_risk(pair, 3.35, fr.tail_profile(_series(100)))
    assert label == "Unscored" and "too few" in detail


# --- correlation and concentration --------------------------------------------

def test_a_series_correlated_with_itself_is_one():
    closes = _series(400)
    assert fr.correlate(closes, closes, "self", "X").coefficient == pytest.approx(1.0)


def test_an_inverted_series_is_strongly_negative():
    """The sign is the content: USD/CHF reads -0.82 against EUR/USD
    because it is the same trade the other way up."""
    rng = np.random.default_rng(11)
    raw = rng.normal(0, 0.005, 400)
    up = pd.Series(100 * np.cumprod(1 + raw), index=_days(400))
    down = pd.Series(100 * np.cumprod(1 - raw), index=_days(400))
    reading = fr.correlate(up, down, "inverse", "X")
    assert reading.coefficient < -0.98 and reading.strength == "Strong"


def test_too_little_overlap_is_unavailable_rather_than_a_number():
    a = _series(400, start="2016-01-01") if False else _series(400)
    b = pd.Series(_series(20).values, index=_days(20, "2016-06-01"))
    reading = fr.correlate(a, b, "short", "X")
    assert not reading.ok and reading.strength == "Unavailable"


def test_peers_are_the_pairs_sharing_a_leg():
    """A pair that moves with its neighbour is not a second position —
    it is the same bet twice. EUR/USD against GBP/USD reads 0.81."""
    peers = {sym for _, sym in fr.peers_for(fd.PAIRS_BY_CODE["EURUSD"])}
    assert "GBPUSD=X" in peers
    assert "EURUSD=X" not in peers          # never itself


def test_a_pair_is_never_offered_as_its_own_peer():
    for pair in fd.PAIRS:
        assert pair.symbol not in {s for _, s in fr.peers_for(pair)}


def test_the_concentration_note_names_the_pairs_that_move_together():
    readings = [fr.Correlation("GBP/USD", "GBPUSD=X", 0.81, 515),
                fr.Correlation("USD/CAD", "USDCAD=X", 0.15, 515)]
    note = fr.concentration_note(readings)
    assert "GBP/USD" in note and "USD/CAD" not in note
    assert "one bet sized larger" in note


def test_no_strong_neighbour_says_so_plainly():
    readings = [fr.Correlation("USD/CAD", "USDCAD=X", 0.15, 515)]
    assert "No neighbouring pair" in fr.concentration_note(readings)


# --- what is not measured -----------------------------------------------------

def test_implied_volatility_is_declared_unavailable_not_faked():
    """A volatility smile needs FX option quotes. What is here is
    REALISED volatility, which is a different quantity."""
    assert "option" in fr.IMPLIED_VOL_UNAVAILABLE
    assert "REALISED" in fr.IMPLIED_VOL_UNAVAILABLE
    names = [n for n in dir(fr) if not n.startswith("_")]
    assert not [n for n in names if "smile" in n.lower()]
    assert not [n for n in names if "implied" in n.lower()
                and n != "IMPLIED_VOL_UNAVAILABLE"]


def test_intervention_and_political_risk_are_refused_with_reasons():
    for note in (fr.INTERVENTION_UNAVAILABLE, fr.POLITICAL_RISK_UNAVAILABLE):
        assert len(note) > 80 and note.strip().endswith(".")
    assert "calendar" in fr.INTERVENTION_UNAVAILABLE
