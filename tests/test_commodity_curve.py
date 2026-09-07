"""Curve shape, carry, spread and roll — tested against definitions.

The rule from the bond phase holds: pin the maths against its
DEFINITION, not a remembered number. Carry is a log ratio over time, so
doubling the horizon at the same total move must halve it, exactly. Roll
yield is minus the calendar spread, so its sign must flip with the
curve's.
"""
import math

import pandas as pd
import pytest

import commodity_curve as cc
import commodity_data as cd


def _point(month, price, oi=100_000, year=2027):
    return cd.ContractPoint(symbol=f"X{month}{year}", year=year,
                            month=month, price=price, open_interest=oi)


def _curve(pairs, key="gold", as_of="2026-12-01"):
    """pairs: (year, month, price[, open_interest])."""
    points = []
    for pair in pairs:
        year, month, price = pair[0], pair[1], pair[2]
        oi = pair[3] if len(pair) > 3 else 100_000
        points.append(cd.ContractPoint(
            symbol=f"X{month}{year}", year=year, month=month, price=price,
            open_interest=oi))
    return cd.Curve(commodity=cd.COMMODITIES_BY_KEY[key],
                    points=tuple(points), as_of=pd.Timestamp(as_of),
                    probed=len(points))


# --- shape --------------------------------------------------------------------

def test_a_rising_curve_is_contango():
    curve = _curve([(2026, 12, 100.0), (2027, 3, 102.0), (2027, 6, 104.0)])
    reading = cc.shape(curve)
    assert reading.label == cc.CONTANGO
    assert reading.front_to_back_pct == pytest.approx(4.0)
    assert reading.turning_point == ""


def test_a_falling_curve_is_backwardation():
    curve = _curve([(2026, 12, 100.0), (2027, 3, 97.0), (2027, 6, 94.0)])
    reading = cc.shape(curve)
    assert reading.label == cc.BACKWARDATION
    assert reading.front_to_back_pct == pytest.approx(-6.0)


def test_a_flat_curve_is_called_flat_rather_than_given_a_direction():
    curve = _curve([(2026, 12, 100.0), (2027, 3, 100.01),
                    (2027, 6, 100.02)])
    assert cc.shape(curve).label == cc.FLAT


def test_a_HUMPED_curve_is_its_own_answer_not_a_rounding():
    """THE AGRICULTURAL CASE, from the live corn strip: 536.75, 552.25,
    559.75, 562.00, 534.75, 536.25. Front to back is -0.09% — a two-way
    classifier calls that flat and says nothing, when what the market is
    pricing is a harvest."""
    curve = _curve([(2026, 12, 536.75), (2027, 3, 552.25),
                    (2027, 5, 559.75), (2027, 7, 562.00),
                    (2027, 9, 534.75), (2027, 12, 536.25)], "corn")
    reading = cc.shape(curve)
    assert reading.label == cc.HUMPED
    assert reading.turning_point == "N27"          # July, the old-crop peak
    assert abs(reading.front_to_back_pct) < 0.5    # would have read "flat"


def test_an_inverted_hump_is_also_humped():
    """A curve that falls then rises — a trough — is equally not a
    single slope."""
    curve = _curve([(2026, 12, 100.0), (2027, 3, 94.0), (2027, 6, 99.0)])
    reading = cc.shape(curve)
    assert reading.label == cc.HUMPED
    assert reading.turning_point


def test_one_settlement_rounding_does_not_manufacture_a_hump():
    """Without a noise band, a single tick against the trend turns a
    clean contango into "humped" and the label becomes useless."""
    curve = _curve([(2026, 12, 100.0), (2027, 1, 101.0),
                    (2027, 2, 100.99), (2027, 3, 102.0),
                    (2027, 4, 103.0)])
    assert cc.shape(curve).label == cc.CONTANGO


def test_shape_reads_only_the_liquid_contracts():
    """A thin deferred month must not set the shape. Here the liquid
    strip rises and a 3-lot contract at the back falls."""
    curve = _curve([(2026, 12, 100.0, 300_000), (2027, 3, 102.0, 50_000),
                    (2027, 6, 80.0, 3)])
    reading = cc.shape(curve)
    assert reading.label == cc.CONTANGO
    assert reading.contracts_used == 2


def test_a_curve_that_could_not_load_is_unavailable_not_flat():
    assert cc.shape(None).label == cc.UNAVAILABLE
    assert cc.shape(cd.Curve(error="nothing quoted")).label == cc.UNAVAILABLE


# --- carry is a definition ----------------------------------------------------

def test_implied_carry_is_the_log_ratio_over_time():
    curve = _curve([(2026, 12, 100.0), (2027, 12, 105.0)])
    carry = cc.implied_carry(curve)
    assert carry.years == pytest.approx(1.0)
    assert carry.annualised_pct == pytest.approx(100 * math.log(1.05))


def test_doubling_the_horizon_halves_the_annualised_carry():
    """The identity. An implementation using a simple ratio rather than
    a log, or forgetting to divide by the horizon, fails this."""
    one_year = cc.implied_carry(_curve([(2026, 12, 100.0), (2027, 12, 110.0)]))
    two_year = cc.implied_carry(_curve([(2026, 12, 100.0), (2028, 12, 110.0)]))
    assert two_year.annualised_pct == pytest.approx(
        one_year.annualised_pct / 2, rel=1e-9)


def test_the_gold_strip_reproduces_a_known_carry():
    """Measured live: Dec-26 4,683.2 to Dec-27 4,909.0 implies 4.71%/yr
    continuously compounded, and against a 3.71% bill that leaves 1.00pp
    of storage net of convenience yield — the textbook figure for gold.
    Reproducing a known answer is the reason to trust it on an unknown
    one."""
    curve = _curve([(2026, 12, 4683.2), (2027, 12, 4909.0)])
    carry = cc.implied_carry(curve, risk_free_pct=3.71)
    assert carry.annualised_pct == pytest.approx(4.71, abs=0.02)
    assert carry.net_of_rate_pct == pytest.approx(1.00, abs=0.02)


def test_a_backwardated_curve_gives_a_negative_carry():
    carry = cc.implied_carry(_curve([(2026, 12, 100.0), (2027, 12, 90.0)]))
    assert carry.annualised_pct < 0


def test_carry_without_a_risk_free_rate_omits_the_net_figure():
    carry = cc.implied_carry(_curve([(2026, 12, 100.0), (2027, 12, 105.0)]))
    assert carry.net_of_rate_pct is None
    assert "storage" not in cc.describe_carry(carry)


def test_contracts_that_do_not_span_time_are_refused():
    curve = cd.Curve(
        commodity=cd.COMMODITIES_BY_KEY["gold"],
        points=(_point(12, 100.0, year=2026), _point(12, 101.0, year=2026)),
        as_of=pd.Timestamp("2026-12-01"))
    assert not cc.implied_carry(curve).ok


def test_the_module_does_not_apply_the_specs_additive_formula():
    """PHASE 4.2 gives P(T) = Spot + (r + s - y) x T, which adds a rate
    to a price and returns a one-year gold forward 0.001% above spot
    where the real contract trades 4.8% above it. The note must be
    present and the module must not contain a forward-price function
    taking storage and convenience yield as inputs."""
    assert cc.SPEC_FORMULA_NOTE
    assert "adds a rate to a price" in cc.SPEC_FORMULA_NOTE
    names = [n.lower() for n in dir(cc) if not n.startswith("_")]
    assert not [n for n in names if "forward_price" in n]


# --- calendar spread, in place of basis ---------------------------------------

def test_the_calendar_spread_uses_the_two_NEAREST_liquid_contracts():
    curve = _curve([(2026, 12, 100.0), (2027, 3, 103.0), (2027, 6, 110.0)])
    spread = cc.calendar_spread(curve)
    assert spread.near_label == "Z26" and spread.far_label == "H27"
    assert spread.absolute == pytest.approx(3.0)
    assert spread.percent == pytest.approx(3.0)


def test_the_spread_skips_a_thin_front_month():
    curve = _curve([(2026, 9, 99.0, 400), (2026, 12, 100.0, 300_000),
                    (2027, 3, 103.0, 50_000)])
    spread = cc.calendar_spread(curve)
    assert spread.near_label == "Z26"


def test_basis_is_not_computed_anywhere():
    """There is no spot price, so a "basis" function would have to
    relabel the front contract as spot."""
    names = [n.lower() for n in dir(cc) if not n.startswith("_")]
    assert not [n for n in names if n == "basis" or n.startswith("basis_")]


# --- roll yield ---------------------------------------------------------------

def test_contango_gives_a_NEGATIVE_roll_yield():
    """The sign is the easy thing to get backwards. In contango the next
    contract costs MORE, so rolling into it loses money."""
    roll = cc.roll_yield(_curve([(2026, 12, 100.0), (2027, 12, 110.0)]))
    assert roll.ok and roll.annualised_pct < 0
    assert roll.favourable is False
    assert "costs" in cc.describe_roll(roll)


def test_backwardation_gives_a_POSITIVE_roll_yield():
    roll = cc.roll_yield(_curve([(2026, 12, 110.0), (2027, 12, 100.0)]))
    assert roll.ok and roll.annualised_pct > 0
    assert roll.favourable is True
    assert "earns" in cc.describe_roll(roll)


def test_roll_yield_is_exactly_minus_the_annualised_spread():
    curve = _curve([(2026, 12, 100.0), (2027, 6, 104.0)])
    spread = cc.calendar_spread(curve)
    roll = cc.roll_yield(curve)
    assert roll.annualised_pct == pytest.approx(-spread.annualised_pct)


def test_the_annualisation_caveat_is_stated():
    """A 3.2% one-month spread annualises to 38.8%/yr, which is true and
    misleading without the assumption spelled out."""
    assert "IF the curve held" in cc.ROLL_ANNUALISATION_CAVEAT


# --- inventory against its own season -----------------------------------------

def _weekly(values, start="2021-01-01"):
    index = pd.date_range(start, periods=len(values), freq="W")
    return pd.Series(values, index=index, dtype="float64")


def test_inventory_is_compared_with_the_SAME_WEEK_of_prior_years():
    """A flat mean would report the season as news: crude stocks swing
    seasonally, and a level that is high for October is low for April.
    Here every prior year holds 100 in the matching week and the latest
    is 110, so the deviation is exactly +10%."""
    index = pd.date_range("2021-01-06", periods=6 * 52, freq="W")
    values = [100.0] * len(index)
    series = pd.Series(values, index=index)
    series.iloc[-1] = 110.0
    reading = cc.read_inventory(series)
    assert reading.ok and reading.scored
    assert reading.deviation_pct == pytest.approx(10.0, abs=0.01)
    assert reading.years_used >= 3


def test_a_short_history_is_UNSCORED_rather_than_called_normal():
    series = _weekly([100.0] * 20, start="2026-01-04")
    reading = cc.read_inventory(series)
    assert reading.ok
    assert not reading.scored
    label, detail = cc.inventory_verdict(reading)
    assert label == "Unscored" and "too few" in detail


def test_a_four_week_change_is_reported():
    series = _weekly([100.0, 100.0, 100.0, 100.0, 110.0])
    reading = cc.read_inventory(series)
    assert reading.change_4w_pct == pytest.approx(10.0)


def test_low_stocks_and_backwardation_are_reported_as_agreeing():
    index = pd.date_range("2021-01-06", periods=6 * 52, freq="W")
    series = pd.Series([100.0] * len(index), index=index)
    series.iloc[-1] = 85.0
    reading = cc.read_inventory(series)
    label, detail = cc.inventory_verdict(reading, cc.BACKWARDATION)
    assert label == "Tight"
    assert "agrees" in detail


def test_a_disagreement_between_stocks_and_curve_is_SHOWN_not_averaged():
    index = pd.date_range("2021-01-06", periods=6 * 52, freq="W")
    series = pd.Series([100.0] * len(index), index=index)
    series.iloc[-1] = 85.0
    label, detail = cc.inventory_verdict(cc.read_inventory(series),
                                         cc.CONTANGO)
    assert label == "Tight"
    assert "does not line up" in detail


def test_no_inventory_series_is_an_error_not_a_zero():
    assert not cc.read_inventory(None).ok
    assert cc.inventory_verdict(cc.read_inventory(None))[0] == cc.UNAVAILABLE
