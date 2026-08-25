"""NVT and stock-to-flow, tested against their definitions.

The rule this file follows, learned the hard way in the bond phase: pin
maths against its DEFINITION, not against a remembered number. NVT is
market cap over on-chain volume — so a doubling of volume must halve it,
exactly. Stock-to-flow is supply over annual issuance — so a series with
a known constant issuance must return that issuance, exactly.
"""
import numpy as np
import pandas as pd
import pytest

import crypto_data as cd
import crypto_valuation as cv


def _days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def _flat(n, value, start="2024-01-01"):
    return pd.Series([float(value)] * n, index=_days(n, start))


# --- NVT is a definition ------------------------------------------------------

def test_nvt_is_market_cap_over_on_chain_volume():
    index = _days(200)
    mc = pd.Series(1000.0, index=index)
    tv = pd.Series(10.0, index=index)
    raw, signal = cv.nvt_series(mc, tv)
    assert raw.iloc[-1] == pytest.approx(100.0)
    assert signal.iloc[-1] == pytest.approx(100.0)


def test_doubling_on_chain_volume_exactly_halves_nvt():
    """The identity. A poisoned implementation that used exchange volume,
    or that inverted the ratio, cannot satisfy this."""
    index = _days(200)
    mc = pd.Series(1000.0, index=index)
    base = cv.nvt_series(mc, pd.Series(10.0, index=index))[1].iloc[-1]
    doubled = cv.nvt_series(mc, pd.Series(20.0, index=index))[1].iloc[-1]
    assert doubled == pytest.approx(base / 2.0)


def test_the_signal_smooths_the_DENOMINATOR_not_the_ratio():
    """NVT Signal is market cap over a 90-day AVERAGE volume, which is
    not the same as a 90-day average of the daily ratio — the two differ
    whenever volume moves, because the mean of a reciprocal is not the
    reciprocal of the mean."""
    index = _days(200)
    mc = pd.Series(1000.0, index=index)
    tv = pd.Series(np.linspace(5.0, 25.0, 200), index=index)
    raw, signal = cv.nvt_series(mc, tv)
    average_of_ratio = raw.iloc[-90:].mean()
    assert signal.iloc[-1] != pytest.approx(average_of_ratio, rel=1e-3)
    assert signal.iloc[-1] == pytest.approx(1000.0 / tv.iloc[-90:].mean())


def test_the_signal_needs_a_full_window_and_says_so():
    index = _days(45)
    reading = cv.read_nvt(pd.Series(1000.0, index=index),
                          pd.Series(10.0, index=index))
    assert not reading.ok
    assert "90" in reading.error


def test_a_zero_volume_day_is_dropped_rather_than_dividing_by_zero():
    index = _days(200)
    tv = pd.Series(10.0, index=index)
    tv.iloc[50] = 0.0
    raw, signal = cv.nvt_series(pd.Series(1000.0, index=index), tv)
    assert np.isfinite(raw).all()
    assert len(raw) == 199


def test_the_two_series_are_joined_on_the_overlap_only():
    """Market cap runs longer than transaction volume in the live data
    (1370 vs 1455 points over four years, with different coverage). The
    join must take the intersection, not pad."""
    mc = pd.Series(1000.0, index=_days(300, "2024-01-01"))
    tv = pd.Series(10.0, index=_days(200, "2024-03-01"))
    raw, _ = cv.nvt_series(mc, tv)
    assert len(raw) == 200


# --- the percentile, and the threshold that cannot fire -----------------------

def test_the_percentile_places_a_value_in_its_own_history():
    series = pd.Series(range(101), dtype="float64")
    assert cv.percentile_of(series, 50) == pytest.approx(50.495, abs=0.01)
    assert cv.percentile_of(series, 0) == pytest.approx(0.990, abs=0.01)
    assert cv.percentile_of(series, 100) == pytest.approx(100.0)
    assert cv.percentile_of(series, None) is None
    assert cv.percentile_of(pd.Series(dtype="float64"), 5) is None


def test_a_high_reading_is_called_rich_and_a_low_one_cheap():
    index = _days(400)
    mc = pd.Series(np.linspace(1000, 5000, 400), index=index)
    reading = cv.read_nvt(mc, pd.Series(10.0, index=index))
    label, detail = cv.nvt_verdict(reading)
    assert label == "Richly valued"
    assert "percentile" in detail

    falling = pd.Series(np.linspace(5000, 1000, 400), index=index)
    label2, _ = cv.nvt_verdict(cv.read_nvt(falling, pd.Series(10.0, index=index)))
    assert label2 == "Cheaply valued"


def test_a_reading_with_too_little_history_is_unscored_not_neutral():
    """A percentile over three weeks is a ranking against noise. It is
    withheld, and the value is still shown."""
    index = _days(200)
    reading = cv.read_nvt(pd.Series(1000.0, index=index),
                          pd.Series(10.0, index=index))
    assert reading.ok
    assert reading.observations == 111
    assert not reading.scored          # below MIN_PERCENTILE_OBSERVATIONS
    label, detail = cv.nvt_verdict(reading)
    assert label == "Unscored"
    assert "too few" in detail


def test_the_module_refuses_the_specs_dead_threshold():
    """PHASE 3.2 asks for "NVT < 20 = undervalued". Bitcoin's NVT has not
    been below 20 on any of the last 1365 days — its floor is 28.9 — so
    that rule can only ever return "not undervalued". The module must not
    contain it."""
    import ast
    import inspect

    # Checked as a COMPARISON in the AST, not as a substring. A substring
    # search for "< 20" matches the legitimate `percentile <= 20` branch
    # — a percentile threshold, which is a different quantity entirely —
    # and a search over the raw source matches the explanatory comment
    # that documents why the rule was rejected. Both produced a test that
    # failed on correct code.
    tree = ast.parse(inspect.getsource(cv))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if not {"nvt", "nvt_signal"} & names:
            continue
        constants = {n.value for n in ast.walk(node)
                     if isinstance(n, ast.Constant)
                     and isinstance(n.value, (int, float))}
        if constants:
            offenders.append((ast.unparse(node), constants))
    assert not offenders, (
        f"NVT is compared against a hard-coded constant: {offenders}. "
        "It must be scored against its own measured history.")
    assert cv.NVT_SPEC_THRESHOLD_NOTE
    assert "28.9" in cv.NVT_SPEC_THRESHOLD_NOTE


def test_an_unavailable_reading_never_reports_a_number():
    reading = cv.read_nvt(None, None)
    assert not reading.ok and reading.nvt_signal is None
    assert cv.nvt_verdict(reading)[0] == "Unavailable"


# --- stock-to-flow is a definition too ----------------------------------------

def test_stock_to_flow_recovers_a_KNOWN_issuance_rate():
    """Construct a supply series issuing exactly 164,250 coins a year and
    the measurement must return it. This is the check that catches an
    implementation still assuming the pre-2024 subsidy: it would report
    half the flow and twice the ratio, and no remembered number would
    reveal it."""
    days = 730
    per_day = 164_250.0 / 365.25
    supply = pd.Series(
        [20_000_000.0 + per_day * i for i in range(days)], index=_days(days))
    scarcity = cv.stock_to_flow(supply)
    assert scarcity.ok
    assert scarcity.flow == pytest.approx(164_250.0, rel=1e-6)
    assert scarcity.ratio == pytest.approx(scarcity.stock / 164_250.0)
    assert scarcity.inflation_pct == pytest.approx(
        100.0 * scarcity.flow / scarcity.stock)


def test_halving_the_issuance_doubles_the_ratio_at_the_same_stock():
    """Held at the same ENDING supply, because stock-to-flow is stock
    over flow and both move when only the issuance rate is changed —
    the first version of this test varied both and compared the result
    to a number that was never the identity."""
    days = 730
    def series(annual):
        step = annual / 365.25
        end = 20_000_000.0
        return pd.Series([end - step * (days - 1 - i) for i in range(days)],
                         index=_days(days))
    fast = cv.stock_to_flow(series(328_500.0))
    slow = cv.stock_to_flow(series(164_250.0))
    assert fast.stock == pytest.approx(slow.stock)
    assert fast.flow == pytest.approx(328_500.0, rel=1e-6)
    assert slow.flow == pytest.approx(164_250.0, rel=1e-6)
    assert slow.ratio == pytest.approx(fast.ratio * 2, rel=1e-6)


def test_a_short_window_refuses_to_measure_issuance():
    supply = pd.Series([20e6 + i for i in range(60)], index=_days(60))
    scarcity = cv.stock_to_flow(supply)
    assert not scarcity.ok
    assert "180" in scarcity.error
    assert scarcity.stock is not None      # the stock is still known


def test_a_flat_supply_reports_no_issuance_rather_than_infinity():
    scarcity = cv.stock_to_flow(_flat(400, 21e6))
    assert not scarcity.ok
    assert scarcity.ratio is None
    assert "did not increase" in scarcity.error


def test_no_supply_history_is_an_error_not_a_zero():
    assert not cv.stock_to_flow(None).ok
    assert not cv.stock_to_flow(pd.Series(dtype="float64")).ok


def test_the_scarcity_description_says_the_flow_was_measured():
    days = 730
    supply = pd.Series([20e6 + (164_250.0 / 365.25) * i for i in range(days)],
                       index=_days(days))
    text = cv.describe_scarcity(cv.stock_to_flow(supply))
    assert "measured" in text and "halving" in text


# --- supply, and the uncapped answer ------------------------------------------

def _row(**kwargs):
    base = dict(coin_id="x", symbol="x", name="X")
    base.update(kwargs)
    return cd.CoinRow(**base)


def test_an_uncapped_coin_gets_an_answer_not_a_blank():
    picture = cv.supply_picture(_row(circulating_supply=1.2e8, max_supply=None))
    assert picture.uncapped is True
    assert picture.maximum is None
    assert picture.pct_of_max_mined is None
    assert "no supply cap" in picture.note.lower()
    # The note must say a zero would be WRONG, not merely absent.
    assert "opposite" in picture.note


def test_a_capped_coin_reports_how_much_is_mined():
    picture = cv.supply_picture(
        _row(circulating_supply=20_075_134.0, max_supply=21_000_000.0))
    assert not picture.uncapped
    assert picture.pct_of_max_mined == pytest.approx(95.6, abs=0.05)
    assert "95.6%" in picture.note


# --- the scorecard ------------------------------------------------------------

def _good_reading():
    index = _days(400)
    return cv.read_nvt(pd.Series(np.linspace(1000, 5000, 400), index=index),
                       pd.Series(10.0, index=index))


def _good_scarcity():
    days = 730
    return cv.stock_to_flow(pd.Series(
        [20e6 + (164_250.0 / 365.25) * i for i in range(days)],
        index=_days(days)))


def test_the_scorecard_counts_what_it_could_measure():
    card = cv.scorecard(
        _good_reading(), _good_scarcity(),
        cv.supply_picture(_row(circulating_supply=20e6, max_supply=21e6)))
    assert card.ok
    assert card.dimensions_scored == card.dimensions_possible == 3


def test_a_blind_dimension_lowers_the_COUNT_not_the_grade():
    """The data-quality badge gave every ETF 18/100 by scoring absent
    evidence as bad news. A dimension that could not be measured must be
    reported as unmeasured, and must not drag the others down."""
    card = cv.scorecard(cv.read_nvt(None, None), cv.stock_to_flow(None),
                        cv.supply_picture(_row(circulating_supply=1e8,
                                               max_supply=None)))
    assert card.dimensions_scored == 1          # only the uncapped answer
    assert card.dimensions_possible == 3
    verdicts = {line.key: line.verdict for line in card.lines}
    assert verdicts["nvt"] == "Unavailable"
    assert verdicts["stock_to_flow"] == "Unavailable"
    assert verdicts["supply_cap"] == "Neutral"
    assert "MVRV" in card.summary


def test_stock_to_flow_is_never_reported_as_a_valuation_verdict():
    """Scarcity is a property of the issuance schedule, not a statement
    that the price is low. Calling it "Cheap" would be a category error."""
    card = cv.scorecard(_good_reading(), _good_scarcity(),
                        cv.supply_picture(_row(circulating_supply=20e6,
                                               max_supply=21e6)))
    line = next(l for l in card.lines if l.key == "stock_to_flow")
    assert line.verdict == "Neutral"
    assert "not a valuation" in line.detail
