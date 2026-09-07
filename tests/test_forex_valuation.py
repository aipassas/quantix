"""Parity, carry and purchasing power, tested against definitions.

The convention under test throughout: a pair written BASE/QUOTE is
priced in units of QUOTE per one BASE, so the forward is

    F = S x (1 + r_quote x T) / (1 + r_base x T)

Getting that backwards does not raise — it silently inverts every
result — which is why the direction is pinned by an economic fact rather
than by restating the formula: a currency you are PAID to hold must
trade at a forward DISCOUNT.
"""
import pytest

import forex_data as fd
import forex_valuation as fv


def _rates(**pairs):
    return fd.PolicyRates(rates={
        code: fd.PolicyRate(code, value, "2026-08")
        for code, value in pairs.items()})


def _factors(**pairs):
    return fd.PppFactors(factors={
        code: fd.PppFactor(code, value, "2025", code)
        for code, value in pairs.items()})


# --- forwards -----------------------------------------------------------------

def test_the_high_yielder_trades_at_a_forward_DISCOUNT():
    """THE DIRECTION TEST. AUD pays 4.35% and JPY 1.00%, so AUD/JPY must
    price BELOW spot forward — roughly the differential. An inverted
    convention returns a premium and looks perfectly plausible."""
    forward = fv.forward_rate(100.0, base_rate_pct=4.35,
                              quote_rate_pct=1.00, months=12)
    assert forward.ok
    assert forward.rate < 100.0
    assert forward.at_discount is True
    assert forward.premium_pct == pytest.approx(-3.21, abs=0.05)


def test_the_low_yielder_trades_at_a_forward_PREMIUM():
    forward = fv.forward_rate(100.0, base_rate_pct=1.00,
                              quote_rate_pct=4.35, months=12)
    assert forward.rate > 100.0 and forward.at_discount is False


def test_the_forward_is_exactly_covered_interest_parity():
    spot, base, quote, months = 1.1631, 2.25, 3.625, 12
    expected = spot * (1 + quote / 100 * 1.0) / (1 + base / 100 * 1.0)
    assert fv.forward_rate(spot, base, quote, months).rate == pytest.approx(expected)


def test_equal_rates_leave_the_forward_at_spot():
    forward = fv.forward_rate(1.2345, 3.0, 3.0, 12)
    assert forward.rate == pytest.approx(1.2345)
    assert forward.premium_pct == pytest.approx(0.0)


def test_a_longer_tenor_moves_the_forward_further_but_not_the_rate():
    """The annualised premium is a rate, so it should barely change with
    tenor while the outright forward moves proportionally."""
    near = fv.forward_rate(100.0, 4.35, 1.0, 3)
    far = fv.forward_rate(100.0, 4.35, 1.0, 12)
    assert abs(far.rate - 100.0) > abs(near.rate - 100.0)
    assert far.premium_pct == pytest.approx(near.premium_pct, abs=0.15)


def test_the_curve_covers_the_tenors_the_task_names():
    curve = fv.forward_curve(100.0, 4.35, 1.0)
    assert [f.tenor for f in curve] == ["1M", "3M", "6M", "1Y"]
    assert all(f.ok for f in curve)


def test_a_missing_policy_rate_yields_no_forward_rather_than_spot():
    """Falling back to spot would present "no data" as "parity holds
    exactly", which is a claim."""
    forward = fv.forward_rate(100.0, None, 1.0, 12)
    assert not forward.ok and forward.rate is None
    assert "policy rate is missing" in forward.error


def test_no_spot_yields_no_forward():
    assert not fv.forward_rate(None, 4.35, 1.0, 12).ok
    assert not fv.forward_rate(0.0, 4.35, 1.0, 12).ok


def test_parity_is_labelled_as_an_identity_not_a_forecast():
    assert "arbitrage identity" in fv.PARITY_IS_NOT_A_FORECAST
    assert "uncovered" in fv.PARITY_IS_NOT_A_FORECAST.lower()


# --- carry --------------------------------------------------------------------

def test_carry_is_the_rate_differential():
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    reading = fv.carry(pair, _rates(AUD=4.35, JPY=1.0))
    assert reading.ok
    assert reading.differential_pct == pytest.approx(3.35)
    assert reading.positive is True


def test_a_negative_carry_is_reported_as_a_cost():
    pair = fd.PAIRS_BY_CODE["EURUSD"]
    reading = fv.carry(pair, _rates(EUR=2.25, USD=3.625))
    assert reading.differential_pct == pytest.approx(-1.375)
    assert reading.positive is False
    assert "costs" in fv.describe_carry(reading)


def test_carry_is_scaled_by_the_pairs_own_volatility():
    """A 3.35% carry on a pair that swings 9% a year is a different
    proposition from the same carry on one that swings 25%."""
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    calm = fv.carry(pair, _rates(AUD=4.35, JPY=1.0), volatility_pct=9.0)
    wild = fv.carry(pair, _rates(AUD=4.35, JPY=1.0), volatility_pct=25.0)
    assert calm.ratio == pytest.approx(3.35 / 9.0)
    assert wild.ratio < calm.ratio
    assert calm.thin is False and wild.thin is False
    thin = fv.carry(pair, _rates(AUD=1.1, JPY=1.0), volatility_pct=9.0)
    assert thin.thin is True
    assert "thin" in fv.describe_carry(thin)


def test_the_rate_dates_travel_with_the_carry():
    """BIS is monthly and lags unevenly, so a carry number without its
    observation dates hides how stale it might be."""
    rates = fd.PolicyRates(rates={
        "AUD": fd.PolicyRate("AUD", 4.35, "2026-07"),
        "JPY": fd.PolicyRate("JPY", 1.0, "2026-08")})
    reading = fv.carry(fd.PAIRS_BY_CODE["AUDJPY"], rates)
    assert reading.base_as_of == "2026-07" and reading.quote_as_of == "2026-08"
    assert "2026-07" in fv.describe_carry(reading)


def test_a_missing_leg_names_which_currency_is_missing():
    reading = fv.carry(fd.PAIRS_BY_CODE["AUDJPY"], _rates(AUD=4.35))
    assert not reading.ok and "JPY" in reading.error


def test_unavailable_rates_give_an_unavailable_carry():
    reading = fv.carry(fd.PAIRS_BY_CODE["AUDJPY"],
                       fd.PolicyRates(error="feed down"))
    assert not reading.ok and reading.error == "feed down"


# --- purchasing power ---------------------------------------------------------

def test_the_ppp_rate_is_quote_factor_over_base_factor():
    """Inverting this turns every undervaluation into an
    overvaluation."""
    valuation = fv.ppp_valuation(
        fd.PAIRS_BY_CODE["EURUSD"], 1.1631,
        _factors(EUR=0.709983, USD=1.0))
    assert valuation.ppp_rate == pytest.approx(1.0 / 0.709983, rel=1e-9)
    assert valuation.ppp_rate == pytest.approx(1.4085, abs=0.001)


def test_the_yen_comes_out_undervalued_as_it_famously_is():
    """USD/JPY PPP is 97.08 against a spot near 154.5, so the dollar
    sits ~59% above parity and the yen ~37% below. Reproducing the
    best-known standing fact in the currency market is the check that
    the method measures what it claims."""
    valuation = fv.ppp_valuation(
        fd.PAIRS_BY_CODE["USDJPY"], 154.527,
        _factors(USD=1.0, JPY=97.08005))
    assert valuation.ppp_rate == pytest.approx(97.08005)
    assert valuation.deviation_pct == pytest.approx(59.2, abs=0.5)
    assert valuation.verdict == "Well above PPP"


def test_the_franc_comes_out_expensive_as_it_famously_is():
    valuation = fv.ppp_valuation(
        fd.PAIRS_BY_CODE["USDCHF"], 0.8093,
        _factors(USD=1.0, CHF=0.930746))
    assert valuation.deviation_pct < 0        # the dollar is cheap...
    assert valuation.verdict == "Below PPP"   # ...so the franc is dear


def test_the_bands_are_wide_because_ppp_gaps_persist():
    """A five-percent gap is not a finding: deviations of twenty percent
    are ordinary and can last a decade."""
    near = fv.ppp_valuation(fd.PAIRS_BY_CODE["EURUSD"], 1.05,
                            _factors(EUR=1.0, USD=1.0))
    assert near.verdict == "Near PPP"
    assert "anchor, not a trade" in fv.describe_ppp(near)


def test_a_missing_factor_names_the_currency():
    valuation = fv.ppp_valuation(fd.PAIRS_BY_CODE["EURUSD"], 1.16,
                                 _factors(EUR=0.71))
    assert not valuation.ok and "USD" in valuation.error


def test_no_spot_gives_no_valuation():
    assert not fv.ppp_valuation(fd.PAIRS_BY_CODE["EURUSD"], None,
                                _factors(EUR=0.71, USD=1.0)).ok


def test_the_euro_proxy_note_reaches_the_description():
    factors = fd.PppFactors(factors={
        "EUR": fd.PppFactor("EUR", 0.709983, "2025", "Germany",
                            "PPP uses Germany."),
        "USD": fd.PppFactor("USD", 1.0, "2025", "United States")})
    text = fv.describe_ppp(fv.ppp_valuation(fd.PAIRS_BY_CODE["EURUSD"],
                                            1.1631, factors))
    assert "Germany" in text


# --- the scorecard ------------------------------------------------------------

def test_the_scorecard_counts_what_it_could_measure():
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    card = fv.scorecard(
        fv.carry(pair, _rates(AUD=4.35, JPY=1.0), 8.8),
        fv.ppp_valuation(pair, 111.54, _factors(AUD=1.398943, JPY=97.08005)),
        fv.forward_curve(111.54, 4.35, 1.0))
    assert card.ok
    assert card.dimensions_scored == card.dimensions_possible == 3


def test_a_blind_dimension_lowers_the_count_not_the_grade():
    """The data-quality badge gave every ETF 18/100 by scoring absent
    evidence as bad news."""
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    card = fv.scorecard(fv.carry(pair, fd.PolicyRates(error="down")),
                        fv.ppp_valuation(pair, None, fd.PppFactors(error="x")),
                        ())
    assert card.dimensions_scored == 0
    assert card.dimensions_possible == 3
    assert all(line.verdict == "Unavailable" for line in card.lines)


def test_the_forward_line_is_never_called_a_forecast():
    pair = fd.PAIRS_BY_CODE["AUDJPY"]
    card = fv.scorecard(fv.carry(pair, _rates(AUD=4.35, JPY=1.0)),
                        fv.ppp_valuation(pair, 111.54,
                                         _factors(AUD=1.4, JPY=97.08)),
                        fv.forward_curve(111.54, 4.35, 1.0))
    line = next(l for l in card.lines if l.key == "forward")
    assert line.verdict in ("Discount", "Premium")
    assert "not a view" in line.detail
