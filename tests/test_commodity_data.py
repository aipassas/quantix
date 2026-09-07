"""Contract symbols, the probed calendar, liquidity, and inventory.

The measurements this file defends, all taken live on 2026-08-26:

  - GCZ26 is a 404 and GCZ26.CMX is a quote, so the venue suffix is
    load-bearing.
  - Corn lists 7 contract months of 18 and gold lists all 18, so the
    calendar must be probed rather than assumed either way.
  - Gold quotes a May-27 contract with 19 lots of open interest, so
    "quotes" is not "trades".
"""
import io

import pandas as pd
import pytest

import commodity_data as cd


# --- symbols ------------------------------------------------------------------

def test_a_contract_symbol_carries_its_venue_suffix():
    """THE 404 THIS PREVENTS. Probed: GCZ26 returns "Quote not found" and
    GCZ26.CMX returns Gold Dec 26. A symbol built without the suffix
    looks right and resolves to nothing."""
    gold = cd.COMMODITIES_BY_KEY["gold"]
    assert cd.contract_symbol(gold, 2026, 12) == "GCZ26.CMX"
    assert cd.contract_symbol(gold, 2027, 2) == "GCG27.CMX"


def test_every_month_maps_to_the_right_cme_code():
    """F G H J K M N Q U V X Z is January to December. An off-by-one
    here silently fetches the wrong contract, which still quotes."""
    gold = cd.COMMODITIES_BY_KEY["gold"]
    expected = "FGHJKMNQUVXZ"
    for month in range(1, 13):
        assert cd.contract_symbol(gold, 2027, month)[2] == expected[month - 1]


def test_a_two_digit_year_is_zero_padded():
    gold = cd.COMMODITIES_BY_KEY["gold"]
    assert cd.contract_symbol(gold, 2030, 1) == "GCF30.CMX"
    assert cd.contract_symbol(gold, 2105, 1) == "GCF05.CMX"


@pytest.mark.parametrize("month", [0, 13, -1])
def test_an_impossible_month_is_refused(month):
    with pytest.raises(ValueError):
        cd.contract_symbol(cd.COMMODITIES_BY_KEY["gold"], 2027, month)


def test_candidate_months_roll_the_year_over():
    months = cd.candidate_months((2026, 11), 4)
    assert months == ((2026, 11), (2026, 12), (2027, 1), (2027, 2))


def test_the_horizon_reaches_beyond_a_year():
    """A curve that stops at twelve months cannot show a full seasonal
    cycle plus the contract after it."""
    assert cd.CURVE_HORIZON_MONTHS >= 13
    assert len(cd.candidate_months((2026, 9))) == cd.CURVE_HORIZON_MONTHS


# --- the commodity table ------------------------------------------------------

def test_every_commodity_is_reachable_by_key_and_by_symbol():
    for commodity in cd.COMMODITIES:
        assert cd.COMMODITIES_BY_KEY[commodity.key] is commodity
        assert cd.COMMODITIES_BY_SYMBOL[commodity.continuous.upper()] is commodity


def test_every_commodity_declares_a_unit():
    """Prices are not comparable across rows — dollars a barrel against
    cents a bushel — so a table without units invites the comparison."""
    for commodity in cd.COMMODITIES:
        assert commodity.unit, commodity.key
        assert commodity.sector in cd.SECTORS, commodity.key


def test_every_sector_has_at_least_one_commodity():
    covered = {c.sector for c in cd.COMMODITIES}
    assert covered == set(cd.SECTORS)


def test_the_continuous_symbol_uses_yahoos_futures_form():
    for commodity in cd.COMMODITIES:
        assert commodity.continuous.endswith("=F"), commodity.key


# --- liquidity ----------------------------------------------------------------

def _point(label_month, price, oi, year=2027):
    return cd.ContractPoint(symbol=f"X{label_month}", year=year,
                            month=label_month, price=price,
                            open_interest=oi)


def _curve(points, key="gold"):
    return cd.Curve(commodity=cd.COMMODITIES_BY_KEY[key],
                    points=tuple(points),
                    as_of=pd.Timestamp("2026-08-26"), probed=len(points))


def test_thin_contracts_are_excluded_from_the_shape_read():
    """THE MEASURED CASE. Gold's odd months quote with 8 to 657 lots
    against a front month holding 318,285. Their prices are real in the
    sense that someone posted them and meaningless in the sense that
    nobody trades them."""
    curve = _curve([
        _point(9, 4429.8, 461), _point(10, 4441.9, 47048),
        _point(11, 4459.9, 657), _point(12, 4476.6, 318285),
    ])
    kept = {p.month for p in curve.liquid_points}
    assert kept == {10, 12}
    assert {p.month for p in curve.thin_points} == {9, 11}


def test_a_liquid_curve_keeps_every_contract():
    """Crude's months run 20-100% of the maximum, so none is dropped."""
    curve = _curve([_point(m, 80 - m, 100_000) for m in range(1, 8)], "wti")
    assert len(curve.liquid_points) == len(curve.points)
    assert curve.thin_points == ()


def test_a_curve_reporting_no_open_interest_is_not_thrown_away():
    """An absent field is not evidence of an illiquid market. Discarding
    the curve over it is the error that had the data-quality badge grade
    an ETF on filings it never makes."""
    curve = _curve([_point(m, 100 + m, None) for m in range(1, 5)])
    assert curve.max_open_interest == 0
    assert len(curve.liquid_points) == 4


def test_the_filter_never_leaves_fewer_than_two_contracts():
    """A shape needs two points. If the floor would leave one, the whole
    strip is used and the caller can see the open interest itself."""
    curve = _curve([_point(1, 100.0, 1_000_000), _point(2, 101.0, 1),
                    _point(3, 102.0, 1)])
    assert len(curve.liquid_points) >= 2


def test_the_liquidity_floor_is_relative_not_absolute():
    """An absolute floor would discard a small commodity's entire curve.
    Palladium's front month holds about 3,000 lots — a hundredth of
    gold's — and its curve is not therefore meaningless."""
    small = _curve([_point(1, 1400.0, 3000), _point(2, 1410.0, 900),
                    _point(3, 1420.0, 2)])
    kept = {p.month for p in small.liquid_points}
    assert 1 in kept and 2 in kept and 3 not in kept


# --- curve shape helpers ------------------------------------------------------

def test_a_contract_point_reports_its_distance_in_years():
    point = cd.ContractPoint("GCZ27.CMX", 2027, 12, 4909.0, 537)
    reference = pd.Timestamp("2026-12-01")
    assert point.months_out(reference) == 12
    assert point.years_out(reference) == pytest.approx(1.0)


def test_a_contract_label_reads_as_a_delivery_month():
    assert cd.ContractPoint("GCZ26.CMX", 2026, 12).label == "Z26"
    assert cd.ContractPoint("ZCH27.CBT", 2027, 3).label == "H27"


def test_a_curve_with_one_contract_is_not_a_curve():
    curve = cd.Curve(points=(_point(1, 100.0, 5),))
    assert not curve.ok


# --- validation ---------------------------------------------------------------

def test_a_negative_futures_price_is_NOT_flagged():
    """WTI settled at -37.63 in April 2020. A bound that rejects negative
    prices fails on the single most famous print in the market's
    history, and teaches its reader to ignore the suite."""
    assert cd.validate_price(-37.63) is None


def test_a_zero_price_and_a_nan_are_flagged():
    assert cd.validate_price(0.0) is not None
    assert cd.validate_price(float("nan")) is not None
    assert cd.validate_price(None) is None


def test_prices_across_four_orders_of_magnitude_all_pass():
    """Natural gas near 2.86 and gold near 4,683 sit in the same
    universe, so any range bound fails on correct data."""
    for price in (2.859, 18.07, 512.0, 4683.2):
        assert cd.validate_price(price) is None


def test_negative_inventory_is_flagged():
    assert cd.validate_inventory(-1.0) is not None
    assert cd.validate_inventory(0.0) is None
    assert cd.validate_inventory(424460.0) is None


def test_a_curve_out_of_expiry_order_is_flagged():
    curve = _curve([_point(12, 100.0, 5000), _point(6, 101.0, 5000)])
    assert any("expiry order" in n for n in cd.validate_curve(curve))


def test_a_curve_with_duplicate_expiries_is_flagged():
    curve = cd.Curve(points=(_point(6, 100.0, 5000), _point(6, 101.0, 5000)),
                     commodity=cd.COMMODITIES_BY_KEY["gold"])
    assert any("duplicate" in n for n in cd.validate_curve(curve))


def test_a_clean_curve_validates_clean():
    assert cd.validate_curve(_curve([_point(6, 100.0, 5000),
                                     _point(12, 101.0, 5000)])) == []


# --- what is and is not sourced -----------------------------------------------

def test_inventory_is_crude_only_and_says_so():
    assert cd.inventory_available("wti") is True
    assert cd.inventory_available("gold") is False
    assert cd.inventory_note("wti") == ""
    assert "USDA" in cd.inventory_note("corn")
    assert "crude" in cd.inventory_note("gold")


def test_the_unavailable_notes_describe_an_absence_with_a_reason():
    for note in (cd.SPOT_UNAVAILABLE, cd.AGRICULTURAL_INVENTORY_UNAVAILABLE,
                 cd.GEOPOLITICAL_UNAVAILABLE, cd.WEATHER_UNAVAILABLE,
                 cd.POLYGON_UNCONFIGURED, cd.EIA_API_UNCONFIGURED):
        assert len(note) > 80
        assert note.strip().endswith(".")


def test_the_spot_note_names_the_symbols_that_were_probed():
    """A bare "no spot price" sends the reader looking for one."""
    assert "XAUUSD=X" in cd.SPOT_UNAVAILABLE
    assert "calendar spread" in cd.SPOT_UNAVAILABLE


# --- the inventory parser -----------------------------------------------------

_EIA_HTML = """
<html><body><table>
<tr><th>Year-Month</th><th>Week 1 End Date</th><th>Week 1 Value</th>
    <th>Week 2 End Date</th><th>Week 2 Value</th></tr>
<tr><td>2026-Aug</td><td>08/07</td><td>424410</td><td>08/14</td><td>428815</td></tr>
<tr><td>2026-Jul</td><td>07/10</td><td>409665</td><td>07/17</td><td>411675</td></tr>
</table></body></html>
"""


def _install(monkeypatch, text, status=200):
    class _Response:
        status_code = status
        @property
        def text(self):
            return text

    monkeypatch.setattr("requests.get", lambda *a, **k: _Response())


def test_the_inventory_parser_unpivots_weeks_into_a_series(monkeypatch):
    """EIA lays the history out as one row per month with several
    (End Date, Value) pairs, so it is not a time series until it is
    unpivoted."""
    _install(monkeypatch, _EIA_HTML)
    series, error = cd.load_crude_inventory.__wrapped__()
    assert error is None
    assert len(series) == 4
    assert series.index.is_monotonic_increasing
    assert series.loc[pd.Timestamp("2026-08-14")] == pytest.approx(428815)
    assert series.loc[pd.Timestamp("2026-07-10")] == pytest.approx(409665)


def test_the_year_comes_from_the_ROW_not_the_cell(monkeypatch):
    """The cells carry only month/day. Taking the year from anywhere but
    the row label puts every observation in the wrong year."""
    _install(monkeypatch, _EIA_HTML)
    series, _ = cd.load_crude_inventory.__wrapped__()
    assert {d.year for d in series.index} == {2026}


def test_a_december_week_inside_a_january_row_rolls_back_a_year(monkeypatch):
    """EIA's January rows can carry a week ending in late December. Read
    naively that observation lands twelve months in the future."""
    html = """
    <html><body><table>
    <tr><th>Year-Month</th><th>Week 1 End Date</th><th>Week 1 Value</th></tr>
    <tr><td>2026-Jan</td><td>12/26</td><td>400000</td></tr>
    </table></body></html>
    """
    _install(monkeypatch, html)
    series, error = cd.load_crude_inventory.__wrapped__()
    assert error is None
    assert series.index[0] == pd.Timestamp("2025-12-26")


def test_a_failed_fetch_returns_an_error_rather_than_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("requests.get", boom)
    series, error = cd.load_crude_inventory.__wrapped__()
    assert series is None and error


def test_a_non_200_response_is_an_error(monkeypatch):
    _install(monkeypatch, _EIA_HTML, status=503)
    series, error = cd.load_crude_inventory.__wrapped__()
    assert series is None and error


def test_a_page_without_the_expected_table_is_reported(monkeypatch):
    _install(monkeypatch, "<html><body><table><tr><th>Nope</th>"
                          "<th>Still nope</th></tr>"
                          "<tr><td>1</td><td>2</td></tr></table></body></html>")
    series, error = cd.load_crude_inventory.__wrapped__()
    assert series is None and error


def test_unreadable_cells_are_skipped_rather_than_guessed(monkeypatch):
    html = """
    <html><body><table>
    <tr><th>Year-Month</th><th>Week 1 End Date</th><th>Week 1 Value</th>
        <th>Week 2 End Date</th><th>Week 2 Value</th></tr>
    <tr><td>2026-Aug</td><td>08/07</td><td>424410</td><td>--</td><td>NA</td></tr>
    </table></body></html>
    """
    _install(monkeypatch, html)
    series, error = cd.load_crude_inventory.__wrapped__()
    assert error is None and len(series) == 1


# --- the curve loader, mocked -------------------------------------------------
# The only path that touches the network, and the one carrying the two
# findings that shaped this module: the calendar must be probed, and a
# contract that quotes is not necessarily one that trades.

def _install_yf(monkeypatch, quotes):
    """quotes: {symbol: {...info...}}; anything absent 404s."""
    import sys
    import types

    class _Ticker:
        def __init__(self, symbol):
            self._symbol = symbol

        @property
        def info(self):
            if self._symbol not in quotes:
                raise RuntimeError(f"Quote not found for {self._symbol}")
            return quotes[self._symbol]

    fake = types.ModuleType("yfinance")
    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _quote(price, oi=100_000, volume=1000):
    return {"regularMarketPrice": price, "openInterest": oi,
            "regularMarketVolume": volume}


def test_the_calendar_is_PROBED_not_assumed(monkeypatch):
    """THE MEASURED CASE. Corn lists Sep/Dec/Mar/May/Jul and nothing
    else. Assuming every month invents eleven phantom contracts;
    assuming a fixed cycle for gold discards twelve real ones. Only the
    months that answer are kept."""
    _install_yf(monkeypatch, {
        "ZCU26.CBT": _quote(512.0, 1_461),
        "ZCZ26.CBT": _quote(536.75, 992_940),
        "ZCH27.CBT": _quote(552.25, 369_270),
    })
    curve = cd.load_curve.__wrapped__("corn", start=(2026, 9), horizon=8)
    assert curve.ok
    assert curve.probed == 8
    assert [p.label for p in curve.points] == ["U26", "Z26", "H27"]


def test_contracts_come_back_in_expiry_order(monkeypatch):
    _install_yf(monkeypatch, {
        "GCV26.CMX": _quote(4441.9), "GCZ26.CMX": _quote(4476.6),
        "GCG27.CMX": _quote(4512.6)})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 10), horizon=6)
    expiries = [p.expiry for p in curve.points]
    assert expiries == sorted(expiries)
    assert cd.validate_curve(curve) == []


def test_a_contract_quoting_zero_or_nothing_is_skipped(monkeypatch):
    """A zero price is not a market. Keeping it would put a point at the
    origin and turn any curve into a cliff."""
    _install_yf(monkeypatch, {
        "GCV26.CMX": _quote(4441.9),
        "GCX26.CMX": _quote(0.0),
        "GCZ26.CMX": {"regularMarketPrice": None},
        "GCF27.CMX": _quote(4496.3)})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 10), horizon=4)
    assert [p.label for p in curve.points] == ["V26", "F27"]


def test_open_interest_and_volume_are_carried_through(monkeypatch):
    """The liquidity filter has nothing to work with otherwise, and the
    panel could not show a thin month as thin."""
    _install_yf(monkeypatch, {
        "GCZ26.CMX": _quote(4476.6, oi=318_285, volume=39_887),
        "GCG27.CMX": _quote(4512.6, oi=32_692, volume=486)})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 12), horizon=3)
    front = curve.points[0]
    assert front.open_interest == 318_285
    assert front.volume == 39_887


def test_a_single_quoting_contract_is_not_a_curve(monkeypatch):
    """One contract is a price, not a term structure, and saying so is
    better than drawing a line through one point."""
    _install_yf(monkeypatch, {"GCZ26.CMX": _quote(4476.6)})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 12), horizon=3)
    assert not curve.ok
    assert "term structure" in curve.error


def test_a_commodity_that_quotes_nothing_reports_an_error(monkeypatch):
    _install_yf(monkeypatch, {})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 12), horizon=3)
    assert not curve.ok and curve.error
    assert curve.points == ()


def test_an_unknown_commodity_is_refused_by_name(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not fetch for an unknown commodity")
    monkeypatch.setattr("commodity_data._int", explode, raising=False)
    curve = cd.load_curve.__wrapped__("unobtainium")
    assert not curve.ok and "unobtainium" in curve.error


def test_one_failing_contract_does_not_take_the_curve_down(monkeypatch):
    """A dead symbol among live ones degrades one point, never the
    panel."""
    _install_yf(monkeypatch, {
        "GCV26.CMX": _quote(4441.9), "GCZ26.CMX": _quote(4476.6)})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 10), horizon=6)
    assert curve.ok and len(curve.points) == 2


def test_the_liquidity_split_reproduces_the_measured_gold_strip(monkeypatch):
    """End to end on the numbers actually observed: the even months hold
    47,048 to 318,285 lots and the odd ones 19 to 657."""
    _install_yf(monkeypatch, {
        "GCU26.CMX": _quote(4429.8, 461), "GCV26.CMX": _quote(4441.9, 47_048),
        "GCX26.CMX": _quote(4459.9, 657), "GCZ26.CMX": _quote(4476.6, 318_285),
        "GCF27.CMX": _quote(4496.3, 543), "GCG27.CMX": _quote(4512.6, 32_692),
        "GCH27.CMX": _quote(4529.5, 212), "GCJ27.CMX": _quote(4547.3, 8_030),
        "GCK27.CMX": _quote(4565.7, 19), "GCM27.CMX": _quote(4583.5, 3_799)})
    curve = cd.load_curve.__wrapped__("gold", start=(2026, 9), horizon=10)
    assert len(curve.points) == 10
    assert {p.label for p in curve.liquid_points} == {
        "V26", "Z26", "G27", "J27", "M27"}
    assert {p.label for p in curve.thin_points} == {
        "U26", "X26", "F27", "H27", "K27"}
