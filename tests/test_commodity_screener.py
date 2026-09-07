"""The commodity screener: filters, presets and the columns it refuses."""
import pandas as pd
import pytest

import commodity_curve as cc
import commodity_data as cd
import commodity_screener as cs


def _row(key, name, sector, **kwargs):
    base = dict(key=key, name=name, sector=sector, unit="unit",
                symbol=f"{key.upper()}=F")
    base.update(kwargs)
    return cs.CommodityRow(**base)


def _universe():
    """A miniature board: three sectors, one row with a curve, one
    without, and one missing a price entirely."""
    return (
        _row("gold", "Gold", cd.METALS, price=4477.0, return_1y_pct=23.0,
             volatility_pct=28.9, range_position_pct=50.0,
             curve_shape=cc.CONTANGO, carry_pct=4.73, roll_yield_pct=-4.67),
        _row("wti", "WTI Crude Oil", cd.ENERGY, price=91.5,
             return_1y_pct=46.9, volatility_pct=54.2,
             range_position_pct=63.0, curve_shape=cc.BACKWARDATION,
             carry_pct=-26.9, roll_yield_pct=38.8),
        _row("corn", "Corn", cd.AGRICULTURE, price=512.0,
             return_1y_pct=27.0, volatility_pct=20.7,
             range_position_pct=92.0),
        _row("cattle", "Live Cattle", cd.LIVESTOCK, price=212.9,
             return_1y_pct=-9.7, volatility_pct=18.0,
             range_position_pct=12.0),
        _row("mystery", "Unquoted", cd.METALS),
    )


# --- filters ------------------------------------------------------------------

def test_a_numeric_filter_selects_on_the_comparison():
    matches, _ = cs.run(_universe(),
                        [cs.Criterion("volatility_pct", ">", 25.0)])
    assert {m.name for m in matches} == {"Gold", "WTI Crude Oil"}


def test_a_text_filter_selects_on_the_sector():
    matches, _ = cs.run(_universe(),
                        [cs.Criterion("sector", "is", cd.AGRICULTURE)])
    assert {m.name for m in matches} == {"Corn"}


def test_is_not_is_the_complement_within_the_judged_rows():
    matches, unjudged = cs.run(
        _universe(), [cs.Criterion("sector", "is not", cd.METALS)])
    assert {m.name for m in matches} == {"WTI Crude Oil", "Corn",
                                         "Live Cattle"}
    assert unjudged == 0


def test_criteria_combine_with_AND():
    matches, _ = cs.run(_universe(), [
        cs.Criterion("volatility_pct", ">", 20.0),
        cs.Criterion("return_1y_pct", ">", 30.0)])
    assert {m.name for m in matches} == {"WTI Crude Oil"}


@pytest.mark.parametrize("operator,expected", [
    (">", {"WTI Crude Oil"}), (">=", {"Gold", "WTI Crude Oil"}),
    ("<", {"Corn", "Live Cattle"}),
])
def test_every_numeric_operator_is_implemented(operator, expected):
    matches, _ = cs.run(_universe(),
                        [cs.Criterion("volatility_pct", operator, 28.9)])
    assert {m.name for m in matches} == expected


def test_a_row_missing_the_field_is_unjudged_not_failed():
    """The commodity with no price was never examined, and saying so is
    the difference between "one match" and "one match of two looked
    at"."""
    matches, unjudged = cs.run(_universe(),
                               [cs.Criterion("price", ">", 1000.0)])
    assert {m.name for m in matches} == {"Gold"}
    assert unjudged == 1


def test_curve_filters_leave_rows_without_a_curve_UNJUDGED():
    """Curve columns resolve only once the dated contracts have been
    probed. Treating a missing shape as a failure would report a screen
    over five that examined two."""
    matches, unjudged = cs.run(
        _universe(), [cs.Criterion("curve_shape", "is", cc.BACKWARDATION)])
    assert {m.name for m in matches} == {"WTI Crude Oil"}
    assert unjudged == 3


def test_no_criteria_passes_everything():
    matches, unjudged = cs.run(_universe(), [])
    assert len(matches) == len(_universe()) and unjudged == 0


# --- operators are metric-dependent -------------------------------------------

def test_a_text_metric_takes_is_and_a_numeric_one_takes_comparisons():
    """This is why the operator widget carries no Streamlit key: a
    stored ">" would raise the moment the metric changed to Sector."""
    assert cs.operators_for("sector") == cs.TEXT_OPERATORS
    assert cs.operators_for("curve_shape") == cs.TEXT_OPERATORS
    assert cs.operators_for("volatility_pct") == cs.NUMERIC_OPERATORS


def test_an_unknown_metric_falls_back_to_numeric_operators():
    assert cs.operators_for("nonsense") == cs.NUMERIC_OPERATORS


def test_the_shape_choices_match_the_curve_modules_own_labels():
    assert set(cs.shapes()) == {cc.CONTANGO, cc.BACKWARDATION, cc.FLAT,
                                cc.HUMPED}


def test_a_text_threshold_is_not_coerced_to_a_float():
    """A threshold can be a string now that Sector and Curve are
    categorical. Formatting one with float() has crashed this app
    twice."""
    text = cs.describe(cs.Criterion("sector", "is", cd.METALS))
    assert text == "Sector is Metals"
    assert cs.describe(cs.Criterion("volatility_pct", ">", 30.0)) == \
        "Volatility > 30%"


# --- presets ------------------------------------------------------------------

def test_every_preset_names_a_real_metric_and_operator():
    for preset in cs.PRESETS:
        for criterion in preset.criteria:
            assert criterion.metric in cs.METRICS_BY_KEY, criterion.metric
            assert criterion.operator in cs.operators_for(criterion.metric)


def test_the_sector_presets_all_return_something():
    universe = _universe()
    for name in ("Inflation hedges", "Agricultural", "Energy"):
        matches, _ = cs.run(universe, cs.PRESETS_BY_NAME[name].criteria)
        assert matches, name


def test_the_backwardation_preset_matches_when_a_curve_is_loaded():
    """It returns nothing when no curve has been probed — correctly, and
    with every row reported unjudged rather than failed."""
    matches, unjudged = cs.run(
        _universe(), cs.PRESETS_BY_NAME["Backwardated"].criteria)
    assert {m.name for m in matches} == {"WTI Crude Oil"}
    assert unjudged == 3


def test_no_preset_claims_a_theme_the_data_cannot_support():
    """The task asks for an "Energy transition" preset. Lithium, cobalt
    and rare earths have no contract on this board, so such a preset
    would be copper alone wearing a thematic label."""
    assert "Energy transition" not in cs.PRESETS_BY_NAME
    for preset in cs.PRESETS:
        assert len(preset.description) > 40, preset.name


def test_every_preset_is_reachable_by_name():
    for preset in cs.PRESETS:
        assert cs.PRESETS_BY_NAME[preset.name] is preset


# --- filters deliberately absent ----------------------------------------------

def test_inventory_is_not_a_universe_wide_filter():
    """It exists for US crude only, so offering it would leave fifteen
    of sixteen rows unjudged against it."""
    keys = {m.key for m in cs.METRICS}
    assert not [k for k in keys if "inventor" in k.lower()]


def test_production_region_is_not_a_filter():
    """There is no feed. Tagging contracts by hand and calling the
    result a filter presents an assumption as data."""
    keys = " ".join(m.key + m.label for m in cs.METRICS).lower()
    assert "opec" not in keys and "region" not in keys


# --- the results table --------------------------------------------------------

def test_curve_columns_stay_numeric_when_almost_every_row_is_blank():
    """THE DTYPE TRAP. Curve columns are absent until a curve loads, so
    they are the ones at risk of coming back object-dtype and sorting as
    text."""
    matches, _ = cs.run(_universe(), [])
    frame = cs.results_frame(matches)
    for column in ("Carry %", "Roll %"):
        assert pd.api.types.is_numeric_dtype(frame[column]), column
    assert frame["Carry %"].isna().sum() >= 3


def test_a_row_without_a_curve_reads_not_loaded_rather_than_blank():
    """Blank would say "no curve"; the truth is "not probed yet"."""
    matches, _ = cs.run(_universe(), [])
    frame = cs.results_frame(matches)
    assert "Not loaded" in set(frame["Curve"])


def test_the_table_carries_the_unit_because_prices_are_not_comparable():
    """Dollars a barrel beside cents a bushel."""
    assert "Unit" in cs.TABLE_COLUMNS
    frame = cs.results_frame(cs.run(_universe(), [])[0])
    assert list(frame.columns) == list(cs.TABLE_COLUMNS)


def test_an_empty_result_still_has_every_column():
    assert list(cs.results_frame([]).columns) == list(cs.TABLE_COLUMNS)


def test_percent_columns_do_not_use_the_multiplying_format():
    """NumberColumn(format="percent") multiplies by 100, and every figure
    here is already percent-valued."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cs.column_config).strip())
    formats = [kw.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               for kw in node.keywords
               if kw.arg == "format" and isinstance(kw.value, ast.Constant)]
    assert formats and "percent" not in formats
    assert len([f for f in formats if "%%" in f]) >= 4


def test_every_declared_numeric_column_is_in_the_table():
    for column in cs.NUMERIC_COLUMNS:
        assert column in cs.TABLE_COLUMNS, column


# --- the universe loader, mocked ----------------------------------------------

def _install(monkeypatch, frame):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: frame
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _price_frame(symbols, rows=300):
    import numpy as np
    index = pd.date_range("2025-09-01", periods=rows, freq="B")
    rng = np.random.default_rng(5)
    closes = pd.DataFrame(
        {s: 100 * np.cumprod(1 + rng.normal(0, 0.01, rows)) for s in symbols},
        index=index)
    return pd.concat({"Close": closes}, axis=1)


def test_the_universe_computes_return_volatility_and_range(monkeypatch):
    symbols = [c.continuous for c in cd.COMMODITIES]
    _install(monkeypatch, _price_frame(symbols))
    rows, error = cs.load_universe.__wrapped__()
    assert error is None
    assert len(rows) == len(cd.COMMODITIES)
    for row in rows:
        assert row.price is not None
        assert row.volatility_pct is not None
        assert 0 <= row.range_position_pct <= 100


def test_curve_columns_are_attached_when_supplied(monkeypatch):
    symbols = [c.continuous for c in cd.COMMODITIES]
    _install(monkeypatch, _price_frame(symbols))
    rows, _ = cs.load_universe.__wrapped__(
        {"gold": {"curve_shape": cc.CONTANGO, "carry_pct": 4.7,
                  "roll_yield_pct": -4.6, "open_interest": 318285}})
    gold = next(r for r in rows if r.key == "gold")
    assert gold.has_curve and gold.curve_shape == cc.CONTANGO
    assert gold.open_interest == 318285
    others = [r for r in rows if r.key != "gold"]
    assert all(not r.has_curve for r in others)


def test_a_commodity_with_no_prices_still_appears(monkeypatch):
    """Dropping it would shorten the board silently."""
    symbols = [c.continuous for c in cd.COMMODITIES if c.key != "hogs"]
    _install(monkeypatch, _price_frame(symbols))
    rows, _ = cs.load_universe.__wrapped__()
    hogs = next(r for r in rows if r.key == "hogs")
    assert hogs.price is None
    assert len(rows) == len(cd.COMMODITIES)


def test_a_failed_download_returns_an_error_rather_than_raising(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    def boom(*a, **k):
        raise RuntimeError("network down")
    fake.download = boom
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    rows, error = cs.load_universe.__wrapped__()
    assert rows == () and error


def test_an_empty_download_is_an_error(monkeypatch):
    _install(monkeypatch, pd.DataFrame())
    rows, error = cs.load_universe.__wrapped__()
    assert rows == () and error
