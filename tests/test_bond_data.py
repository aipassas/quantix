"""The bond data pipeline: treasury curve, shape, and fund characteristics.

The load-bearing claims are about which provider fields can be trusted.
Two of them cannot, and the tests pin the replacements rather than the
fields — because a test written against `bond_holdings["Duration"]` would
lock in a number that says a twenty-year treasury fund is less
rate-sensitive than a one-to-three-year one.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bond_data as bd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "bond_data.py").read_text(encoding="utf-8")


def _curve(*pairs, source=bd.SOURCE_YAHOO):
    points = tuple(
        bd.CurvePoint(months, bd.MATURITIES_BY_MONTHS[months].label, y, source)
        for months, y in pairs)
    return bd.Curve(points, source)


# --- the maturity table -------------------------------------------------------

def test_the_table_carries_the_ten_maturities_the_task_asks_for():
    assert [m.label for m in bd.MATURITIES] == [
        "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    assert all(m.fred for m in bd.MATURITIES), "every one needs a FRED series"
    assert [m.months for m in bd.MATURITIES] == sorted(
        m.months for m in bd.MATURITIES), "kept in maturity order"


def test_only_the_four_maturities_yahoo_actually_publishes_have_a_symbol():
    """Probed, not assumed: no Yahoo symbol exists for 1Y, 2Y, 3Y, 7Y or
    20Y, and treasury futures are prices rather than yields."""
    with_symbol = {m.label: m.yahoo for m in bd.MATURITIES if m.yahoo}
    assert with_symbol == {"3M": "^IRX", "5Y": "^FVX",
                           "10Y": "^TNX", "30Y": "^TYX"}


# --- credentials --------------------------------------------------------------

def test_fred_is_unconfigured_by_default_and_that_is_quiet(monkeypatch):
    """st.secrets raises when there is no secrets file, which is the
    normal state of a fresh checkout — it must not surface as an error."""
    monkeypatch.delenv(bd.FRED_ENV_VAR, raising=False)
    assert bd.fred_api_key() is None
    assert not bd.fred_is_configured()


def test_a_key_in_the_environment_is_picked_up(monkeypatch):
    monkeypatch.setenv(bd.FRED_ENV_VAR, "  abc123  ")
    assert bd.fred_api_key() == "abc123"
    assert bd.fred_is_configured()


def test_an_empty_key_counts_as_unconfigured(monkeypatch):
    monkeypatch.setenv(bd.FRED_ENV_VAR, "   ")
    assert bd.fred_api_key() is None


def test_the_unconfigured_message_names_what_is_missing_and_how_to_fix_it():
    text = bd.FRED_UNCONFIGURED
    for label in ("6M", "1Y", "2Y", "3Y", "7Y", "20Y"):
        assert label in text, label
    assert bd.FRED_ENV_VAR in text
    assert "secrets.toml" in text


def test_bloomberg_is_declared_unavailable_rather_than_approximated():
    """Corporate bond CUSIPs, coupons and agency ratings need a licence
    this build does not have. Approximating them from something adjacent
    would be a different dataset under the same name."""
    assert "licence" in bd.BLOOMBERG_UNAVAILABLE
    tree = ast.parse(SOURCE)
    names = {n.name.lower() for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)}
    assert not [n for n in names if "cusip" in n or "bloomberg" in n]


# --- curve shape --------------------------------------------------------------

def test_an_upward_curve_reads_as_normal_or_steep():
    normal = bd.curve_shape(_curve((3, 4.0), (120, 4.8), (360, 5.0)))
    assert normal.label == "Normal"
    assert normal.spread_pp == pytest.approx(1.0)
    assert normal.short_label == "3M" and normal.long_label == "30Y"

    steep = bd.curve_shape(_curve((3, 3.0), (360, 5.0)))
    assert steep.label == "Steep"
    assert steep.spread_pp == pytest.approx(2.0)


def test_a_curve_where_long_yields_less_than_short_is_inverted():
    shape = bd.curve_shape(_curve((3, 5.2), (120, 4.4), (360, 4.6)))
    assert shape.label == "Inverted"
    assert shape.spread_pp == pytest.approx(-0.6)
    assert "LESS" in shape.detail


def test_a_curve_with_almost_no_slope_is_flat():
    shape = bd.curve_shape(_curve((3, 4.50), (360, 4.60)))
    assert shape.label == "Flat"


def test_the_shape_names_which_two_maturities_it_measured():
    """A "3M to 30Y" spread and a "2Y to 10Y" spread are different
    numbers, and quoting either without saying which is how they get
    confused."""
    shape = bd.curve_shape(_curve((24, 4.0), (120, 4.5)))
    assert shape.short_label == "2Y" and shape.long_label == "10Y"
    assert "2Y" in shape.detail and "10Y" in shape.detail


def test_a_hump_in_the_middle_is_reported_even_when_the_ends_look_normal():
    """A curve can slope up end to end while being out of order inside
    it, and the ends alone would call that normal."""
    shape = bd.curve_shape(_curve((3, 4.0), (60, 4.9), (120, 4.6), (360, 5.2)))
    assert shape.label in ("Normal", "Steep")
    assert shape.inversions == ("5Y→10Y",)
    assert "out of order" in shape.detail


def test_a_curve_with_too_few_points_has_no_shape():
    assert bd.curve_shape(bd.Curve()).label == "Unknown"
    assert bd.curve_shape(_curve((120, 4.5))).label == "Unknown"


# --- interpolation ------------------------------------------------------------

def test_interpolation_never_extrapolates_past_the_loaded_maturities():
    """A 40-year yield is not obtainable from a curve that stops at 30,
    and returning the 30-year for it would quietly answer a question
    nobody asked."""
    curve = _curve((3, 3.7), (60, 4.4), (120, 4.7), (360, 5.2))
    value, method = bd.interpolate_yield(curve, 480)
    assert value is None and "outside" in method
    value, method = bd.interpolate_yield(curve, 1)
    assert value is None and "outside" in method


def test_an_interpolated_point_sits_between_its_neighbours():
    curve = _curve((3, 3.7), (60, 4.4), (120, 4.7), (360, 5.2))
    value, method = bd.interpolate_yield(curve, 24)
    assert method == "linear"
    assert 3.7 < value < 4.4


def test_an_exact_maturity_returns_its_own_yield():
    curve = _curve((3, 3.7), (60, 4.4), (120, 4.7), (360, 5.2))
    value, _ = bd.interpolate_yield(curve, 120)
    assert value == pytest.approx(4.7)
    assert curve.at(120) == pytest.approx(4.7)
    assert curve.at(24) is None, "not loaded, so not reported"


def test_nelson_siegel_is_not_fitted_to_too_few_points():
    """It has four free parameters. Fitting it to four observations is an
    exact fit with zero residual — a line through the dots rather than a
    model of them, and it would swing wildly between them."""
    assert bd.NELSON_SIEGEL_MIN_POINTS > 4
    four = _curve((3, 3.7), (60, 4.4), (120, 4.7), (360, 5.2))
    _, method = bd.interpolate_yield(four, 24)
    assert method == "linear"


def test_nelson_siegel_is_used_once_there_are_enough_maturities():
    full = _curve((3, 3.70), (6, 3.80), (12, 3.90), (24, 4.00), (36, 4.10),
                  (60, 4.40), (84, 4.55), (120, 4.70), (240, 5.00), (360, 5.23))
    value, method = bd.interpolate_yield(full, 30)
    assert method == "Nelson-Siegel"
    # It must still land near the data it was fitted to.
    assert 3.9 < value < 4.3, value


def test_the_fitted_curve_tracks_the_points_it_was_fitted_to():
    """A fit that misses its own observations is not a fit."""
    pairs = [(3, 3.70), (6, 3.80), (12, 3.90), (24, 4.00), (36, 4.10),
             (60, 4.40), (84, 4.55), (120, 4.70), (240, 5.00), (360, 5.23)]
    full = _curve(*pairs)
    for months, actual in pairs:
        value, method = bd.interpolate_yield(full, months)
        assert method == "Nelson-Siegel"
        assert abs(value - actual) < 0.15, (months, value, actual)


def test_interpolating_an_empty_curve_is_unavailable():
    value, method = bd.interpolate_yield(bd.Curve(), 24)
    assert value is None and method == "unavailable"


# --- empirical duration -------------------------------------------------------

def _price_series(duration, days=400, seed=0):
    """Prices generated so that a 1pp rise in yield costs `duration`%."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=days, freq="B")
    yields = pd.Series(4.0 + np.cumsum(rng.normal(0, 0.04, days)), index=index)
    returns = -duration * yields.diff().fillna(0.0) / 100.0
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, index=index), yields


def test_duration_is_recovered_from_price_behaviour():
    """The whole reason this function exists — the provider's own field
    is unusable, so duration is measured instead."""
    for target in (1.5, 6.0, 13.0):
        prices, yields = _price_series(target)
        duration, r2, days = bd.empirical_duration(prices, yields)
        assert duration == pytest.approx(target, abs=0.15), target
        assert r2 > 0.99
        assert days > 300


def test_a_longer_fund_measures_a_longer_duration():
    """The ordering is the point. Verified live across the iShares
    ladder: SHY 1.43, IEI 4.10, IEF 6.84, TLH 11.24, TLT 13.21."""
    measured = []
    for target in (2.0, 5.0, 9.0, 15.0):
        prices, yields = _price_series(target, seed=1)
        measured.append(bd.empirical_duration(prices, yields)[0])
    assert measured == sorted(measured)


def test_too_few_days_yields_nothing_rather_than_a_noisy_slope():
    """A slope from a handful of days is noise wearing a number's
    clothes."""
    prices, yields = _price_series(6.0, days=20)
    duration, r2, days = bd.empirical_duration(prices, yields)
    assert duration is None and r2 is None


def test_a_flat_yield_series_has_no_measurable_duration():
    index = pd.date_range("2024-01-01", periods=200, freq="B")
    prices = pd.Series(np.linspace(100, 110, 200), index=index)
    flat = pd.Series(4.0, index=index)
    duration, r2, _ = bd.empirical_duration(prices, flat)
    assert duration is None


def test_missing_inputs_return_nothing():
    prices, yields = _price_series(5.0)
    assert bd.empirical_duration(None, yields)[0] is None
    assert bd.empirical_duration(prices, None)[0] is None
    assert bd.empirical_duration(None, None)[0] is None


def test_the_provider_duration_field_is_never_read():
    """A test against bond_holdings would lock in a number that ranks a
    twenty-year treasury fund below a one-to-three-year one."""
    tree = ast.parse(SOURCE)
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "bond_holdings"]
    assert not reads, "bond_holdings is documented as unusable"
    assert "REPORTED_DURATION_UNUSABLE" in SOURCE


# --- ratings ------------------------------------------------------------------

def test_the_letter_buckets_are_the_breakdown_and_government_is_separate():
    """Measured on six funds: the letters sum to exactly 1.0000 while
    us_government runs 0.9958 (TLT) to 0.0000 (LQD). A treasury is both
    AA-rated and government-issued, so folding them together
    double-counts — it reported TLT as 199.6% investment grade."""
    assert bd.GOVERNMENT_KEY not in dict(bd.RATING_ORDER)
    assert bd.GOVERNMENT_KEY not in bd.INVESTMENT_GRADE_KEYS
    assert set(bd.INVESTMENT_GRADE_KEYS) == {"aaa", "aa", "a", "bbb"}


def test_investment_grade_stops_at_bbb():
    fund = bd.BondFund(symbol="X", ratings_pct={
        "aaa": 10.0, "aa": 20.0, "a": 30.0, "bbb": 15.0,
        "bb": 20.0, "b": 5.0})
    assert fund.investment_grade_pct == pytest.approx(75.0)
    assert fund.ratings_total_pct == pytest.approx(100.0)


def test_a_treasury_fund_is_not_counted_twice():
    """TLT: 100% AA and 99.6% government. Investment grade is 100, not
    199.6."""
    fund = bd.BondFund(symbol="TLT", ratings_pct={"aa": 100.0},
                       government_pct=99.58)
    assert fund.investment_grade_pct == pytest.approx(100.0)
    assert fund.ratings_total_pct == pytest.approx(100.0)


def test_rating_rows_come_out_in_credit_order_and_skip_empties():
    fund = bd.BondFund(symbol="X", ratings_pct={
        "bb": 58.2, "b": 32.2, "bbb": 0.8, "aaa": 0.0})
    rows = bd.rating_rows(fund)
    assert [label for label, _ in rows] == ["BBB", "BB", "B"]
    assert rows[0][1] == pytest.approx(0.8)


def test_a_fund_with_no_ratings_reports_nothing_rather_than_zero():
    fund = bd.BondFund(symbol="X")
    assert fund.investment_grade_pct is None
    assert bd.rating_rows(fund) == []


# --- validation ---------------------------------------------------------------

def test_a_negative_yield_is_unusual_but_not_invalid():
    """German and Japanese government debt traded below zero for years.
    The task's "YTM should be positive" would reject real data."""
    assert bd.validate_yield(-0.4) is None
    assert bd.validate_yield(0.0) is None
    assert bd.validate_yield(-8.0) is not None, "but not absurd ones"


def test_the_yield_ceiling_catches_the_unit_error_it_is_for():
    assert bd.validate_yield(4.7) is None
    assert bd.validate_yield(470.0) is not None      # basis points
    problem = bd.validate_yield(470.0)
    assert "basis points" in problem


def test_duration_must_be_positive():
    assert bd.validate_duration(6.0, None) is None
    assert bd.validate_duration(0.0, None) is not None
    assert bd.validate_duration(-2.0, None) is not None


def test_duration_above_maturity_means_the_figures_disagree():
    """Arithmetically impossible for a coupon bond, so it means the two
    numbers came from different places — which is exactly what Yahoo's
    fields do."""
    assert bd.validate_duration(13.21, 7.60) is not None
    assert "disagree" in bd.validate_duration(13.21, 7.60)
    assert bd.validate_duration(5.0, 9.0) is None
    # A zero-coupon bond's duration equals its maturity; the tolerance
    # must not fail that.
    assert bd.validate_duration(9.0, 9.0) is None


def test_price_and_coupon_bounds_are_the_tasks_own():
    assert bd.validate_price(99.5) is None
    assert bd.validate_price(0.0) is not None
    assert bd.validate_price(250.0) is not None
    assert bd.validate_coupon(4.5) is None
    assert bd.validate_coupon(15.0) is not None
    assert bd.validate_coupon(-1.0) is not None


def test_validators_pass_over_what_was_never_reported():
    """None is "not reported", not "invalid" — every validator must say
    nothing about it."""
    for validator in (bd.validate_yield, bd.validate_price, bd.validate_coupon):
        assert validator(None) is None
    assert bd.validate_duration(None, 5.0) is None


def test_a_not_a_number_is_caught_by_every_validator():
    nan = float("nan")
    assert bd.validate_yield(nan) is not None
    assert bd.validate_price(nan) is not None
    assert bd.validate_coupon(nan) is not None
    assert bd.validate_duration(nan, None) is not None


def test_a_clean_curve_has_no_problems():
    assert bd.validate_curve(_curve((3, 3.7), (120, 4.7), (360, 5.2))) == []


def test_validation_catches_a_bad_point_and_a_duplicate():
    curve = bd.Curve((
        bd.CurvePoint(120, "10Y", 470.0, bd.SOURCE_YAHOO),
        bd.CurvePoint(120, "10Y", 4.7, bd.SOURCE_YAHOO),
    ))
    problems = bd.validate_curve(curve)
    assert len(problems) == 2
    assert any("basis points" in p for p in problems)
    assert any("duplicated" in p for p in problems)


def test_an_empty_curve_is_a_problem_in_itself():
    problems = bd.validate_curve(bd.Curve())
    assert problems and "Fewer than two" in problems[0]


# --- loaders, mocked ----------------------------------------------------------

def _install_yf(monkeypatch, download):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_the_yahoo_loader_builds_points_in_maturity_order(monkeypatch):
    index = pd.date_range("2026-08-20", periods=5, freq="B")
    close = pd.DataFrame({"^IRX": 3.70, "^FVX": 4.40,
                          "^TNX": 4.70, "^TYX": 5.23}, index=index)
    _install_yf(monkeypatch, lambda *a, **k: pd.concat({"Close": close}, axis=1))

    points, error = bd._load_yahoo_points.__wrapped__()
    assert error is None
    assert [p.label for p in points] == ["3M", "5Y", "10Y", "30Y"]
    assert [p.months for p in points] == sorted(p.months for p in points)
    assert points[0].yield_pct == pytest.approx(3.70)
    assert all(p.source == bd.SOURCE_YAHOO for p in points)


def test_a_failing_download_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")

    _install_yf(monkeypatch, _boom)
    points, error = bd._load_yahoo_points.__wrapped__()
    assert points == ()
    assert error and "network down" in error


def test_an_empty_download_is_reported(monkeypatch):
    _install_yf(monkeypatch, lambda *a, **k: pd.DataFrame())
    points, error = bd._load_yahoo_points.__wrapped__()
    assert points == ()
    assert error and "empty" in error


def test_a_maturity_with_no_data_is_skipped_not_zeroed(monkeypatch):
    index = pd.date_range("2026-08-20", periods=5, freq="B")
    close = pd.DataFrame({"^IRX": 3.70, "^FVX": np.nan,
                          "^TNX": 4.70, "^TYX": 5.23}, index=index)
    _install_yf(monkeypatch, lambda *a, **k: pd.concat({"Close": close}, axis=1))

    points, _ = bd._load_yahoo_points.__wrapped__()
    assert [p.label for p in points] == ["3M", "10Y", "30Y"]
    assert all(p.yield_pct != 0 for p in points)


def test_the_curve_records_which_maturities_it_is_missing():
    missing = bd._missing_labels([
        bd.CurvePoint(3, "3M", 3.7, bd.SOURCE_YAHOO),
        bd.CurvePoint(120, "10Y", 4.7, bd.SOURCE_YAHOO)])
    assert missing == ("6M", "1Y", "2Y", "3Y", "5Y", "7Y", "20Y", "30Y")


def test_a_fund_load_that_raises_returns_an_error(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    fake.download = lambda *a, **k: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    fund = bd.load_bond_fund.__wrapped__("TLT")
    assert not fund.ok
    assert "TLT" in fund.error and "RuntimeError" in fund.error


def test_an_empty_symbol_is_rejected_before_any_fetch():
    for blank in ("", "  ", None):
        fund = bd.load_bond_fund.__wrapped__(blank)
        assert not fund.ok and "No symbol" in fund.error


# --- live ---------------------------------------------------------------------

@pytest.mark.live
def test_the_real_curve_loads_and_validates():
    curve = bd.load_curve()
    assert curve.ok, curve.error
    assert len(curve.points) >= 4
    assert bd.validate_curve(curve) == []
    for point in curve.points:
        assert 0.0 < point.yield_pct < bd.YTM_MAX_PCT, point


@pytest.mark.live
def test_the_real_curve_has_a_readable_shape():
    shape = bd.curve_shape(bd.load_curve())
    assert shape.label in ("Inverted", "Flat", "Normal", "Steep")
    assert shape.spread_pp is not None
    assert shape.short_label and shape.long_label


@pytest.mark.live
def test_measured_duration_ranks_the_ishares_ladder_correctly():
    """THE regression test for this module. These five funds differ only
    in maturity band, so their durations must increase. The provider's
    own field does not: it reports 3.1, 3.49, 4.2, 3.54, 3.56."""
    ladder = ("SHY", "IEI", "IEF", "TLH", "TLT")
    measured = []
    for symbol in ladder:
        fund = bd.load_bond_fund(symbol)
        assert fund.ok, fund.error
        assert fund.empirical_duration is not None, symbol
        measured.append(fund.empirical_duration)
    assert measured == sorted(measured), dict(zip(ladder, measured))
    # And the spread between the ends is real, not noise.
    assert measured[-1] > measured[0] * 3


@pytest.mark.live
def test_a_real_funds_ratings_sum_to_a_hundred():
    for symbol in ("TLT", "AGG", "LQD", "HYG"):
        fund = bd.load_bond_fund(symbol)
        assert fund.ok, fund.error
        assert fund.ratings_total_pct == pytest.approx(100.0, abs=1.0), symbol


@pytest.mark.live
def test_a_high_yield_fund_is_mostly_below_investment_grade():
    """A sanity check that the buckets mean what they say."""
    hyg = bd.load_bond_fund("HYG")
    assert hyg.ok
    assert hyg.investment_grade_pct < 20.0, hyg.ratings_pct
    tlt = bd.load_bond_fund("TLT")
    assert tlt.investment_grade_pct > 90.0
    assert tlt.government_pct > 90.0


# --- the FRED path, mocked ----------------------------------------------------
# There is no key in this build, so this path cannot be exercised against
# the live service. That makes a mocked test the ONLY assurance it works
# at all — without it, the first person to add a key would be the one to
# discover whether it parses.

class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


def _install_requests(monkeypatch, get):
    import sys
    import types

    fake = types.ModuleType("requests")
    fake.get = get
    monkeypatch.setitem(sys.modules, "requests", fake)


def _fred_payload(series_id):
    values = {"DGS3MO": "3.70", "DGS6MO": "3.78", "DGS1": "3.85",
              "DGS2": "3.96", "DGS3": "4.05", "DGS5": "4.41",
              "DGS7": "4.55", "DGS10": "4.70", "DGS20": "5.05",
              "DGS30": "5.23"}
    return _Response({"observations": [
        {"date": "2026-08-25", "value": values[series_id]}]})


def test_a_configured_key_yields_all_ten_maturities(monkeypatch):
    calls = []

    def _get(url, params=None, timeout=None):
        calls.append(params)
        return _fred_payload(params["series_id"])

    _install_requests(monkeypatch, _get)
    points, error = bd._load_fred_points.__wrapped__("KEY")

    assert error is None
    assert len(points) == 10
    assert [p.label for p in points] == [m.label for m in bd.MATURITIES]
    assert all(p.source == bd.SOURCE_FRED for p in points)
    assert points[0].yield_pct == pytest.approx(3.70)
    # One request per series, well inside FRED's 120/minute.
    assert len(calls) == 10
    assert all(c["api_key"] == "KEY" for c in calls)
    # Newest observation only.
    assert all(c["sort_order"] == "desc" and c["limit"] == 1 for c in calls)


def test_the_fred_curve_fills_in_what_yahoo_cannot(monkeypatch):
    """The whole point of the key: 2Y and 7Y have no Yahoo symbol."""
    _install_requests(monkeypatch,
                      lambda url, params=None, timeout=None:
                      _fred_payload(params["series_id"]))
    points, _ = bd._load_fred_points.__wrapped__("KEY")
    curve = bd.Curve(points, bd.SOURCE_FRED)
    assert curve.at(24) == pytest.approx(3.96)
    assert curve.at(84) == pytest.approx(4.55)
    assert bd._missing_labels(points) == ()
    # ...and with ten points the fitted model becomes available.
    _, method = bd.interpolate_yield(curve, 30)
    assert method == "Nelson-Siegel"


def test_a_rejected_key_reports_rather_than_raising(monkeypatch):
    _install_requests(monkeypatch,
                      lambda url, params=None, timeout=None:
                      _Response({"error_message": "Bad Request"}, status=400))
    points, error = bd._load_fred_points.__wrapped__("BAD")
    assert points == ()
    assert error and "check the API key" in error


def test_one_failing_series_does_not_lose_the_other_nine(monkeypatch):
    def _get(url, params=None, timeout=None):
        if params["series_id"] == "DGS20":
            return _Response({"observations": []})
        return _fred_payload(params["series_id"])

    _install_requests(monkeypatch, _get)
    points, error = bd._load_fred_points.__wrapped__("KEY")
    assert len(points) == 9
    assert error and "20Y" in error


def test_a_raising_request_is_caught_per_series(monkeypatch):
    def _get(url, params=None, timeout=None):
        if params["series_id"] == "DGS7":
            raise RuntimeError("timeout")
        return _fred_payload(params["series_id"])

    _install_requests(monkeypatch, _get)
    points, error = bd._load_fred_points.__wrapped__("KEY")
    assert len(points) == 9
    assert error and "7Y" in error


def test_load_curve_prefers_fred_when_a_key_is_configured(monkeypatch):
    monkeypatch.setenv(bd.FRED_ENV_VAR, "KEY")
    monkeypatch.setattr(bd, "_load_fred_points",
                        lambda key: ((bd.CurvePoint(120, "10Y", 4.7,
                                                    bd.SOURCE_FRED),), None))
    monkeypatch.setattr(bd, "_load_yahoo_points",
                        lambda: ((bd.CurvePoint(3, "3M", 9.9,
                                                bd.SOURCE_YAHOO),), None))
    curve = bd.load_curve()
    assert curve.source == bd.SOURCE_FRED
    assert curve.at(120) == pytest.approx(4.7)


def test_a_dead_fred_key_falls_back_to_yahoo_rather_than_taking_the_curve_down(monkeypatch):
    """A missing or broken key must not cost the four maturities that
    need no key at all."""
    monkeypatch.setenv(bd.FRED_ENV_VAR, "KEY")
    monkeypatch.setattr(bd, "_load_fred_points",
                        lambda key: ((), "FRED returned nothing"))
    monkeypatch.setattr(bd, "_load_yahoo_points",
                        lambda: ((bd.CurvePoint(3, "3M", 3.7, bd.SOURCE_YAHOO),
                                  bd.CurvePoint(120, "10Y", 4.7,
                                                bd.SOURCE_YAHOO)), None))
    curve = bd.load_curve()
    assert curve.source == bd.SOURCE_YAHOO
    assert curve.ok
    assert curve.error and "FRED returned nothing" in curve.error
    assert "2Y" in curve.missing


def test_without_a_key_the_curve_is_yahoos_and_says_what_is_missing(monkeypatch):
    monkeypatch.delenv(bd.FRED_ENV_VAR, raising=False)
    monkeypatch.setattr(bd, "_load_yahoo_points",
                        lambda: ((bd.CurvePoint(3, "3M", 3.7, bd.SOURCE_YAHOO),
                                  bd.CurvePoint(360, "30Y", 5.2,
                                                bd.SOURCE_YAHOO)), None))
    curve = bd.load_curve()
    assert curve.source == bd.SOURCE_YAHOO
    assert "2Y" in curve.missing and "7Y" in curve.missing


def test_a_fund_loads_its_ratings_and_a_measured_duration(monkeypatch):
    import sys
    import types

    class _Funds:
        bond_ratings = {"aa": 1.0, "us_government": 0.9958, "aaa": 0.0}

    prices, yields = _price_series(13.0)
    close = pd.DataFrame({"TLT": prices, "^TNX": yields})

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda s: type("T", (), {"funds_data": _Funds()})()
    fake.download = lambda *a, **k: pd.concat({"Close": close}, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    fund = bd.load_bond_fund.__wrapped__("tlt")
    assert fund.ok and fund.symbol == "TLT"
    assert fund.empirical_duration == pytest.approx(13.0, abs=0.2)
    assert fund.duration_r_squared > 0.99
    # Fractions in, percent out — and government kept off the ladder.
    assert fund.ratings_pct == {"aa": 100.0}
    assert fund.government_pct == pytest.approx(99.58)
    assert fund.investment_grade_pct == pytest.approx(100.0)
