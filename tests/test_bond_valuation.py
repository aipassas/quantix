"""Bond valuation: YTM, duration, convexity, scenarios.

Bond maths has exact answers, so most of this file asserts IDENTITIES
rather than remembered numbers — a par bond yields its coupon, a zero's
Macaulay duration IS its maturity, modified duration is Macaulay over
(1 + y/m), and repricing under a small shift must agree with the
analytic derivative. An implementation that satisfies all four is right
for reasons that do not depend on my arithmetic being right.

The two regression tests worth naming are the defects in the reference
implementation the task shipped: fractional maturities (50bp) and
negative yields (unreachable).
"""
import math

import pytest

import bond_valuation as bv


def _bond(coupon=5.0, years=10, m=2, par=100.0):
    return bv.Bond(coupon, years, par=par, periods_per_year=m)


# --- identities that must hold exactly ----------------------------------------

@pytest.mark.parametrize("coupon,years,m", [
    (5.0, 10, 1), (6.0, 10, 2), (3.25, 7, 2), (8.0, 30, 2), (0.5, 2, 4),
])
def test_a_par_bond_yields_its_coupon(coupon, years, m):
    """The cleanest check there is: price it at par and the solver must
    hand back the coupon rate."""
    ytm = bv.yield_to_maturity(_bond(coupon, years, m), 100.0)
    assert ytm == pytest.approx(coupon / 100.0, abs=1e-9)


@pytest.mark.parametrize("years", [1, 5, 10, 30])
def test_a_zeros_macaulay_duration_is_its_maturity(years):
    """A zero-coupon bond has one cash flow, so the weighted average time
    to its cash flows is exactly its maturity. Any other answer means the
    weighting or the period conversion is wrong."""
    bond = _bond(coupon=0.0, years=years)
    duration = bv.macaulay_duration(bond, 0.045)
    assert duration == pytest.approx(years, abs=1e-9)


def test_modified_duration_is_macaulay_over_one_plus_periodic_yield():
    bond, ytm = _bond(6.0, 10, 2), 0.05
    macaulay = bv.macaulay_duration(bond, ytm)
    assert bv.modified_duration(bond, ytm) == pytest.approx(
        macaulay / (1 + ytm / 2), abs=1e-12)


@pytest.mark.parametrize("coupon,years", [(0.0, 10), (6.0, 10), (2.0, 30)])
def test_repricing_agrees_with_the_analytic_duration(coupon, years):
    """Effective duration reprices under a shift; modified duration is the
    closed form. For an option-free bond they must agree — that is the
    test that the closed form is right."""
    bond, ytm = _bond(coupon, years), 0.05
    assert bv.effective_duration(bond, ytm) == pytest.approx(
        bv.modified_duration(bond, ytm), rel=2e-3)


def test_duration_is_returned_in_years_not_periods():
    """The two differ by periods_per_year — a silent 2x if confused."""
    annual = bv.macaulay_duration(_bond(0.0, 10, m=1), 0.05)
    semi = bv.macaulay_duration(_bond(0.0, 10, m=2), 0.05)
    assert annual == pytest.approx(10.0)
    assert semi == pytest.approx(10.0)


@pytest.mark.parametrize("coupon,years", [(0, 5), (0, 30), (6, 10), (12, 20)])
def test_convexity_is_positive_for_every_option_free_bond(coupon, years):
    assert bv.convexity(_bond(coupon, years), 0.05) > 0


def test_a_longer_bond_is_more_convex_and_longer_duration():
    short, long = _bond(0.0, 5), _bond(0.0, 30)
    assert bv.convexity(long, 0.05) > bv.convexity(short, 0.05)
    assert bv.modified_duration(long, 0.05) > bv.modified_duration(short, 0.05)


def test_price_falls_monotonically_as_yield_rises():
    bond = _bond(6.0, 10)
    prices = [bv.price_from_yield(bond, y / 100) for y in range(0, 15)]
    assert all(a > b for a, b in zip(prices, prices[1:]))


def test_price_and_yield_round_trip():
    bond = _bond(4.25, 12)
    for price in (72.0, 95.5, 100.0, 118.75):
        ytm = bv.yield_to_maturity(bond, price)
        assert bv.price_from_yield(bond, ytm) == pytest.approx(price, abs=1e-8)


# --- the defects in the task's reference implementation -----------------------

def test_a_fractional_maturity_is_priced_on_one_clock():
    """THE 50BP BUG. The reference truncates the coupon count with
    int(years) while discounting par over the fractional exponent, so the
    two halves describe different bonds. A 10.5-year 5% annual bond at
    par is 5.3127%, not the 4.8137% that code returns — verified against
    an independent solver."""
    bond = bv.Bond(5.0, 10.5, periods_per_year=1)
    ytm = bv.yield_to_maturity(bond, 100.0)
    assert ytm * 100 == pytest.approx(5.3127, abs=0.001)
    assert abs(ytm * 100 - 4.8137) > 0.4, "the truncating answer must not return"


def test_a_fractional_bonds_cash_flows_land_on_the_stub():
    """0.5, 1.5 … 10.5 — not 1 … 10 with the redemption dangling at 10.5."""
    flows = bv.cash_flows(bv.Bond(5.0, 10.5, periods_per_year=1))
    periods = [f.period for f in flows]
    assert periods[0] == pytest.approx(0.5)
    assert periods[-1] == pytest.approx(10.5)
    assert len(flows) == 11
    assert all(b - a == pytest.approx(1.0) for a, b in zip(periods, periods[1:]))
    # The redemption rides on the final flow, not a separate one.
    assert flows[-1].amount == pytest.approx(5.0 + 100.0)


def test_a_semiannual_fractional_bond_pays_on_half_periods():
    flows = bv.cash_flows(bv.Bond(6.0, 10.25, periods_per_year=2))
    periods = [f.period for f in flows]
    assert periods[-1] == pytest.approx(20.5)
    assert periods[0] == pytest.approx(0.5)
    assert len(flows) == 21


def test_a_negative_yield_is_found_rather_than_pinned_to_zero():
    """THE OTHER BUG. The reference brackets from 0.0001 up, so a 5-year
    1% bond at 130 comes back 0.0161% instead of -4.2561%. Deep-premium
    bonds do this routinely and negative sovereign yields were ordinary
    for years."""
    bond = bv.Bond(1.0, 5, periods_per_year=1)
    ytm = bv.yield_to_maturity(bond, 130.0)
    assert ytm < 0
    assert ytm * 100 == pytest.approx(-4.2561, abs=0.001)
    assert bv.price_from_yield(bond, ytm) == pytest.approx(130.0, abs=1e-8)


def test_a_zero_coupon_bond_above_par_yields_below_zero():
    bond = bv.Bond(0.0, 5, periods_per_year=1)
    ytm = bv.yield_to_maturity(bond, 103.0)
    assert ytm < 0
    assert bv.price_from_yield(bond, ytm) == pytest.approx(103.0, abs=1e-9)


def test_the_solver_reaches_a_distressed_yield():
    """A deeply discounted bond yields far more than the reference's 50%
    ceiling would allow near the top of its range."""
    bond = bv.Bond(2.0, 5, periods_per_year=1)
    ytm = bv.yield_to_maturity(bond, 25.0)
    assert ytm > 0.30
    assert bv.price_from_yield(bond, ytm) == pytest.approx(25.0, abs=1e-8)


# --- refusals -----------------------------------------------------------------

def test_an_unreachable_price_returns_nothing_rather_than_an_endpoint():
    """Pinning to the edge of the bracket returns a confident wrong
    number; None says the price is not one this bond can have."""
    bond = _bond(5.0, 10)
    assert bv.yield_to_maturity(bond, 1e9) is None
    assert bv.yield_to_maturity(bond, 0.0) is None
    assert bv.yield_to_maturity(bond, -5.0) is None
    assert bv.yield_to_maturity(bond, None) is None


def test_a_matured_bond_has_no_cash_flows_and_no_yield():
    for years in (0.0, -1.0):
        bond = bv.Bond(5.0, years)
        assert bv.cash_flows(bond) == []
        assert bv.yield_to_maturity(bond, 100.0) is None
        assert bv.macaulay_duration(bond, 0.05) is None
        assert bv.convexity(bond, 0.05) is None


def test_an_impossible_discount_rate_is_refused():
    """A periodic rate at or below -100% makes the discount factor
    undefined."""
    bond = _bond(5.0, 10, m=1)
    assert bv.price_from_yield(bond, -1.0) is None
    assert bv.macaulay_duration(bond, -1.5) is None
    assert bv.modified_duration(bond, -1.5) is None
    assert bv.convexity(bond, -1.5) is None


def test_current_yield_ignores_the_pull_to_par_and_says_nothing_about_it():
    """It is the coupon over the price and nothing more — which is why it
    is not a yield to maturity."""
    bond = _bond(4.0, 10)
    assert bv.current_yield_pct(bond, 100.0) == pytest.approx(4.0)
    assert bv.current_yield_pct(bond, 80.0) == pytest.approx(5.0)
    assert bv.current_yield_pct(bond, 0) is None
    # A discount bond's YTM exceeds its current yield: the pull to par is
    # extra return the current yield cannot see.
    ytm = bv.yield_to_maturity(bond, 80.0)
    assert ytm * 100 > bv.current_yield_pct(bond, 80.0)


# --- scenarios ----------------------------------------------------------------

def test_the_scenario_table_covers_the_tasks_ladder():
    bond = _bond(4.0, 10)
    ytm = bv.yield_to_maturity(bond, 95.0)
    rows = bv.scenario_table(bond, ytm)
    assert [r.shift_bps for r in rows] == list(bv.SCENARIO_SHIFTS_BPS)
    for shift in (-200, -100, -50, 50, 100, 200):
        assert shift in bv.SCENARIO_SHIFTS_BPS


def test_the_unshifted_scenario_reprices_to_the_starting_price():
    bond = _bond(4.0, 10)
    ytm = bv.yield_to_maturity(bond, 95.0)
    base = next(r for r in bv.scenario_table(bond, ytm) if r.shift_bps == 0)
    assert base.exact_price == pytest.approx(95.0, abs=1e-8)
    assert base.exact_change_pct == pytest.approx(0.0, abs=1e-9)


def test_a_duration_only_estimate_is_symmetric_but_the_bond_is_not():
    """This asymmetry IS convexity. A duration-only line understates the
    rally and overstates the sell-off, and by 200bp the gap is over a
    percentage point."""
    bond = _bond(4.0, 10)
    ytm = bv.yield_to_maturity(bond, 95.0)
    rows = {r.shift_bps: r for r in bv.scenario_table(bond, ytm)}
    up, down = rows[200], rows[-200]

    assert up.duration_only_change_pct == pytest.approx(
        -down.duration_only_change_pct, abs=1e-9)
    # The real move is not symmetric: the rally is bigger than the loss.
    assert abs(down.exact_change_pct) > abs(up.exact_change_pct)
    # Duration alone understates the rally...
    assert down.duration_only_change_pct < down.exact_change_pct
    # ...and overstates the sell-off.
    assert up.duration_only_change_pct < up.exact_change_pct


def test_adding_convexity_closes_most_of_the_gap():
    bond = _bond(4.0, 10)
    ytm = bv.yield_to_maturity(bond, 95.0)
    for row in bv.scenario_table(bond, ytm):
        if row.shift_bps == 0:
            continue
        duration_error = abs(row.duration_only_change_pct - row.exact_change_pct)
        both_error = abs(row.approx_change_pct - row.exact_change_pct)
        assert both_error <= duration_error, row.shift_bps


def test_the_approximation_drifts_furthest_at_the_biggest_shift():
    """Worth showing the exact reprice beside it: the estimate is at its
    worst exactly where a reader most wants to trust it."""
    bond = _bond(4.0, 10)
    ytm = bv.yield_to_maturity(bond, 95.0)
    rows = {r.shift_bps: r for r in bv.scenario_table(bond, ytm)}
    small = abs(rows[50].approx_change_pct - rows[50].exact_change_pct)
    large = abs(rows[200].approx_change_pct - rows[200].exact_change_pct)
    assert large > small


def test_the_estimate_formula_is_the_tasks_own():
    """dP = (-ModDur x dY) + (0.5 x Convexity x dY^2)."""
    estimate = bv.estimate_price_change_pct(8.0, 70.0, 100)
    expected = (-8.0 * 0.01 + 0.5 * 70.0 * 0.01 ** 2) * 100
    assert estimate == pytest.approx(expected)
    # Without a convexity it degrades to the duration term alone.
    assert bv.estimate_price_change_pct(8.0, None, 100) == pytest.approx(-8.0)
    assert bv.estimate_price_change_pct(None, 70.0, 100) is None


# --- par and curve positioning ------------------------------------------------

def test_a_bond_trades_above_par_exactly_when_its_coupon_beats_its_yield():
    bond = _bond(4.0, 10)
    for price, expected in ((95.0, bv.DISCOUNT), (100.0, bv.AT_PAR),
                            (112.0, bv.PREMIUM)):
        ytm = bv.yield_to_maturity(bond, price)
        position = bv.par_position(bond, price, ytm)
        assert position.label == expected, price
        if expected is bv.PREMIUM:
            assert ytm * 100 < bond.coupon_rate_pct
        elif expected is bv.DISCOUNT:
            assert ytm * 100 > bond.coupon_rate_pct


def test_premium_is_explained_rather_than_left_to_read_as_good():
    bond = _bond(4.0, 10)
    ytm = bv.yield_to_maturity(bond, 112.0)
    detail = bv.par_position(bond, 112.0, ytm).detail
    assert "coupon" in detail and "yield" in detail
    assert "pulls back to par" in detail


def test_the_premium_percentage_is_measured_against_par():
    bond = _bond(4.0, 10, par=1000.0)
    assert bv.par_position(bond, 1100.0, 0.03).premium_pct == pytest.approx(10.0)
    assert bv.par_position(bond, 900.0, 0.05).premium_pct == pytest.approx(-10.0)


class _FakeCurve:
    """Stands in for bond_data.Curve; interpolate_yield reads points."""
    def __init__(self, points):
        self.points = points

    @property
    def ok(self):
        return len(self.points) >= 2


def _curve():
    import bond_data as bd
    return bd.Curve((
        bd.CurvePoint(3, "3M", 3.70, bd.SOURCE_YAHOO),
        bd.CurvePoint(60, "5Y", 4.37, bd.SOURCE_YAHOO),
        bd.CurvePoint(120, "10Y", 4.66, bd.SOURCE_YAHOO),
        bd.CurvePoint(360, "30Y", 5.20, bd.SOURCE_YAHOO)))


def test_a_bond_yielding_more_than_the_curve_is_cheap():
    position = bv.curve_position(0.0503, _curve(), 10)
    assert position.label == bv.CHEAP
    assert position.spread_bps == pytest.approx(37.0, abs=1.0)
    assert "MORE than" in position.detail
    assert "compensation" in position.detail


def test_a_bond_yielding_less_than_the_curve_is_rich():
    position = bv.curve_position(0.0325, _curve(), 5)
    assert position.label == bv.RICH
    assert position.spread_bps < 0
    assert "LESS than" in position.detail


def test_a_bond_on_the_curve_is_not_described_as_a_comparison():
    """It read "4bp the same as than the 10.0-year point" before this."""
    position = bv.curve_position(0.0470, _curve(), 10)
    assert position.label == bv.FAIR
    assert "than" not in position.detail
    assert "Sits on" in position.detail


def test_the_spread_is_quoted_in_basis_points():
    """A credit spread quoted in percent gets misread by a hundred."""
    position = bv.curve_position(0.0566, _curve(), 10)
    assert position.spread_bps == pytest.approx(100.0, abs=1.0)
    assert "bp" in position.detail


def test_a_maturity_the_curve_does_not_span_is_not_guessed_at():
    position = bv.curve_position(0.05, _curve(), 40)
    assert position.spread_bps is None
    assert not position.ok
    assert "does not span" in position.detail


def test_no_yield_means_no_comparison():
    assert not bv.curve_position(None, _curve(), 10).ok


# --- bond funds ---------------------------------------------------------------

def test_expected_return_is_yield_minus_duration_times_the_rate_move():
    outlook = bv.fund_expected_return(4.4, 13.21, 100)
    assert outlook.price_change_pct == pytest.approx(-13.21, abs=1e-9)
    assert outlook.total_return_pct == pytest.approx(4.4 - 13.21, abs=1e-9)

    rally = bv.fund_expected_return(4.4, 13.21, -100)
    assert rally.price_change_pct == pytest.approx(13.21, abs=1e-9)
    assert rally.total_return_pct == pytest.approx(4.4 + 13.21, abs=1e-9)


def test_with_rates_unchanged_the_return_is_the_yield():
    outlook = bv.fund_expected_return(4.4, 13.21, 0)
    assert outlook.total_return_pct == pytest.approx(4.4)
    assert "unchanged" in outlook.detail


def test_expected_return_needs_both_inputs():
    assert bv.fund_expected_return(None, 13.0, 100).total_return_pct is None
    assert bv.fund_expected_return(4.4, None, 100).total_return_pct is None
    assert "Needs both" in bv.fund_expected_return(4.4, None, 100).detail


def test_the_breakeven_move_is_the_number_a_holder_actually_needs():
    """A 4.4% yield on a 13.21 duration fund is wiped out by 33bp, which
    is one ordinary week — and the task does not ask for it."""
    assert bv.breakeven_rate_move_bps(4.4, 13.21) == pytest.approx(33.3, abs=0.1)
    # A short fund can absorb far more.
    assert bv.breakeven_rate_move_bps(4.4, 1.43) > 300
    assert bv.breakeven_rate_move_bps(4.4, 0) is None
    assert bv.breakeven_rate_move_bps(None, 5.0) is None


# --- what is declared unavailable ---------------------------------------------

def test_merton_default_probability_is_declared_rather_than_approximated():
    """It needs the issuer's equity value and volatility, and there is no
    bond-to-equity mapping in this build."""
    assert "equity" in bv.MERTON_UNAVAILABLE
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "bond_valuation.py").read_text(encoding="utf-8")
    names = {n.name.lower() for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.FunctionDef)}
    assert not [n for n in names if "merton" in n or "default_prob" in n]


def test_effective_duration_says_it_assumes_no_embedded_option():
    """A callable bond's true effective duration is lower, and returning
    this figure for one without saying so would overstate its upside."""
    assert "callable" in bv.EMBEDDED_OPTION_NOTE
    assert "option-adjusted" in bv.EMBEDDED_OPTION_NOTE


# --- the solver's fallback ----------------------------------------------------

def test_the_bisection_fallback_agrees_with_brent(monkeypatch):
    """scipy is a dependency, but the fallback must not be wrong — an
    untested fallback is a bug waiting for the day scipy is missing."""
    bond = _bond(4.25, 12)
    for price in (72.0, 95.5, 118.75):
        expected = bv.yield_to_maturity(bond, price)

        def excess(ytm, _bond=bond, _price=price):
            return bv.price_from_yield(_bond, ytm) - _price

        fallback = bv._bisect(excess, bv.YIELD_MIN, bv.YIELD_MAX)
        assert fallback == pytest.approx(expected, abs=1e-6), price


def test_a_failing_solver_falls_back_rather_than_raising(monkeypatch):
    """If scipy's brentq ever fails, the yield must still come back — a
    solver that raises would take down every bond panel on the page."""
    import scipy.optimize

    def _boom(*args, **kwargs):
        raise RuntimeError("brentq unavailable")

    monkeypatch.setattr(scipy.optimize, "brentq", _boom)
    bond = _bond(4.25, 12)
    ytm = bv.yield_to_maturity(bond, 95.5)
    assert ytm is not None
    assert bv.price_from_yield(bond, ytm) == pytest.approx(95.5, abs=1e-6)


def test_a_price_that_makes_the_bracket_non_finite_is_refused():
    """An enormous par overflows the discounting at the bracket's floor;
    the answer is None, not an infinity."""
    bond = bv.Bond(5.0, 30, par=1e308, periods_per_year=2)
    assert bv.yield_to_maturity(bond, 1e307) is None


def test_the_bisection_fallback_terminates_on_a_flat_function():
    """A degenerate bracket must not spin for 500 iterations and then
    return something misleading — it returns the midpoint, which is the
    honest answer when the function never changes sign."""
    result = bv._bisect(lambda y: 1.0, bv.YIELD_MIN, bv.YIELD_MAX)
    assert result is not None
    assert bv.YIELD_MIN <= result <= bv.YIELD_MAX


# --- checks against the definitions, not against remembered numbers ------------

def test_price_matches_the_closed_form_annuity():
    """Textbook anchors: a 10y 8% annual bond at a 10% yield is 87.711, a
    5y 6% at 8% is 92.015, a 3y zero at 5% is 86.384."""
    assert bv.price_from_yield(bv.Bond(8.0, 10, periods_per_year=1),
                               0.10) == pytest.approx(87.711, abs=0.001)
    assert bv.price_from_yield(bv.Bond(6.0, 5, periods_per_year=1),
                               0.08) == pytest.approx(92.015, abs=0.001)
    assert bv.price_from_yield(bv.Bond(0.0, 3, periods_per_year=1),
                               0.05) == pytest.approx(86.384, abs=0.001)


@pytest.mark.parametrize("coupon,years,m", [
    (8.0, 10, 1), (6.0, 10, 2), (0.0, 30, 2), (12.0, 5, 2),
])
def test_convexity_is_the_second_derivative_over_price(coupon, years, m):
    """The definition, by central difference — a stronger check than any
    remembered figure, and it caught one of mine: I expected 61.6 for the
    10y 8% bond and both the module and the derivative say 56.14."""
    bond, ytm = bv.Bond(coupon, years, periods_per_year=m), 0.05
    price = bv.price_from_yield(bond, ytm)
    step = 1e-5
    second_derivative = (bv.price_from_yield(bond, ytm + step)
                         - 2 * price
                         + bv.price_from_yield(bond, ytm - step)) / step ** 2
    assert bv.convexity(bond, ytm) == pytest.approx(
        second_derivative / price, rel=1e-4)


@pytest.mark.parametrize("coupon,years,m", [(8.0, 10, 1), (4.0, 20, 2)])
def test_modified_duration_is_minus_the_first_derivative_over_price(coupon, years, m):
    bond, ytm = bv.Bond(coupon, years, periods_per_year=m), 0.05
    price = bv.price_from_yield(bond, ytm)
    step = 1e-6
    first_derivative = (bv.price_from_yield(bond, ytm + step)
                        - bv.price_from_yield(bond, ytm - step)) / (2 * step)
    assert bv.modified_duration(bond, ytm) == pytest.approx(
        -first_derivative / price, rel=1e-5)


def test_the_taylor_expansion_reprices_the_bond():
    """Duration and convexity together ARE the first two Taylor terms, so
    they must reconstruct the price. The residual is the third order, and
    it is what grows at the wide shifts."""
    bond, ytm = bv.Bond(8.0, 10, periods_per_year=1), 0.10
    price = bv.price_from_yield(bond, ytm)
    mod, convex = bv.modified_duration(bond, ytm), bv.convexity(bond, ytm)
    for bps in (50, 100, -50, -100):
        shift = bps * bv.BASIS_POINT
        taylor = price * (1 - mod * shift + 0.5 * convex * shift ** 2)
        exact = bv.price_from_yield(bond, ytm + shift)
        assert taylor == pytest.approx(exact, abs=0.01), bps
