"""The currency screener: filters, presets and the columns it refuses."""
import pandas as pd
import pytest

import forex_data as fd
import forex_screener as fs


def _row(code, base, quote, **kwargs):
    base_kwargs = dict(code=code, label=f"{base}/{quote}",
                       symbol=f"{base}{quote}=X", base=base, quote=quote)
    base_kwargs.update(kwargs)
    return fs.PairRow(**base_kwargs)


def _universe():
    """A miniature board: a paid yen cross, a negative-carry major, a
    calm cross, and one row missing its rate."""
    return (
        _row("AUDJPY", "AUD", "JPY", spot=111.54, rate_differential_pct=3.35,
             carry_ratio=0.38, volatility_pct=8.8, skew=-0.47,
             ppp_deviation_pct=60.7, range_position_pct=72.0,
             return_1y_pct=4.0, funded_in_haven=True),
        _row("EURUSD", "EUR", "USD", spot=1.1631, rate_differential_pct=-1.38,
             carry_ratio=-0.24, volatility_pct=5.7, skew=0.24,
             ppp_deviation_pct=-17.4, range_position_pct=50.0,
             return_1y_pct=2.0),
        _row("USDCHF", "USD", "CHF", spot=0.8093, rate_differential_pct=3.62,
             carry_ratio=0.49, volatility_pct=7.3, skew=-0.73,
             ppp_deviation_pct=-13.1, range_position_pct=31.0,
             return_1y_pct=-3.0, funded_in_haven=True),
        _row("EURGBP", "EUR", "GBP", spot=0.8589, rate_differential_pct=-1.50,
             carry_ratio=-0.43, volatility_pct=3.5, skew=0.00,
             ppp_deviation_pct=-9.9, range_position_pct=44.0,
             return_1y_pct=0.5),
        _row("NZDJPY", "NZD", "JPY", spot=90.85, volatility_pct=8.5,
             skew=0.01, range_position_pct=60.0, funded_in_haven=True),
    )


# --- filters ------------------------------------------------------------------

def test_a_numeric_filter_selects_on_the_comparison():
    matches, _ = fs.run(_universe(),
                        [fs.Criterion("rate_differential_pct", ">", 0.0)])
    assert {m.label for m in matches} == {"AUD/JPY", "USD/CHF"}


def test_results_come_back_highest_carry_first():
    matches, _ = fs.run(_universe(),
                        [fs.Criterion("volatility_pct", ">", 0.0)])
    carries = [m.row.rate_differential_pct for m in matches
               if m.row.rate_differential_pct is not None]
    assert carries == sorted(carries, reverse=True)


def test_a_text_filter_selects_on_a_currency_leg():
    matches, _ = fs.run(_universe(), [fs.Criterion("quote", "is", "JPY")])
    assert {m.label for m in matches} == {"AUD/JPY", "NZD/JPY"}


def test_is_not_is_the_complement():
    matches, _ = fs.run(_universe(), [fs.Criterion("base", "is not", "EUR")])
    assert {m.label for m in matches} == {"AUD/JPY", "USD/CHF", "NZD/JPY"}


def test_criteria_combine_with_AND():
    matches, _ = fs.run(_universe(), [
        fs.Criterion("rate_differential_pct", ">", 0.0),
        fs.Criterion("volatility_pct", "<", 8.0)])
    assert {m.label for m in matches} == {"USD/CHF"}


@pytest.mark.parametrize("operator,expected", [
    (">", {"USD/CHF"}), (">=", {"AUD/JPY", "USD/CHF"}),
    ("<", {"EUR/USD", "EUR/GBP"}),
])
def test_every_numeric_operator_is_implemented(operator, expected):
    matches, _ = fs.run(_universe(),
                        [fs.Criterion("rate_differential_pct", operator, 3.35)])
    assert {m.label for m in matches} == expected


def test_a_row_missing_the_field_is_unjudged_not_failed():
    """NZD/JPY has no rate differential here, so it was never examined —
    and saying so is the difference between "two matches" and "two of
    four looked at"."""
    matches, unjudged = fs.run(
        _universe(), [fs.Criterion("rate_differential_pct", ">", 0.0)])
    assert "NZD/JPY" not in {m.label for m in matches}
    assert unjudged == 1


def test_no_criteria_passes_everything():
    matches, unjudged = fs.run(_universe(), [])
    assert len(matches) == len(_universe()) and unjudged == 0


# --- operators are metric-dependent -------------------------------------------

def test_a_currency_metric_takes_is_and_a_number_takes_comparisons():
    """This is why the operator widget carries no Streamlit key."""
    assert fs.operators_for("base") == fs.TEXT_OPERATORS
    assert fs.operators_for("quote") == fs.TEXT_OPERATORS
    assert fs.operators_for("carry_ratio") == fs.NUMERIC_OPERATORS
    assert fs.operators_for("nonsense") == fs.NUMERIC_OPERATORS


def test_the_currency_choices_come_from_the_data_module():
    assert set(fs.currencies()) == {c.code for c in fd.CURRENCIES}


def test_a_text_threshold_is_not_coerced_to_a_float():
    """Formatting a string threshold with float() has crashed this app
    twice."""
    assert fs.describe(fs.Criterion("quote", "is", "JPY")) == \
        "Quote currency is JPY"
    assert fs.describe(fs.Criterion("volatility_pct", "<", 8.0)) == \
        "Volatility < 8%"
    assert fs.describe(fs.Criterion("carry_ratio", ">", 0.25)) == \
        "Carry per unit of vol > 0.25"


# --- presets ------------------------------------------------------------------

def test_every_preset_names_a_real_metric_and_a_valid_operator():
    for preset in fs.PRESETS:
        for criterion in preset.criteria:
            assert criterion.metric in fs.METRICS_BY_KEY, criterion.metric
            assert criterion.operator in fs.operators_for(criterion.metric)


def test_every_preset_returns_something_on_a_realistic_board():
    """A preset that matches nothing reads as "no such pairs exist"."""
    universe = _universe()
    for preset in fs.PRESETS:
        matches, _ = fs.run(universe, preset.criteria)
        assert matches, preset.name


def test_the_carry_preset_does_not_call_being_paid_being_right():
    text = fs.PRESETS_BY_NAME["Positive carry"].description
    assert "not the same as being" in text


def test_the_yen_funded_preset_is_a_currency_filter_not_a_theme():
    """A filter on a currency is a fact; a filter on a theme is an
    opinion. The task's "Risk-Off Pairs" and "Growth" baskets are
    expressed as the legs that define them."""
    preset = fs.PRESETS_BY_NAME["Yen-funded"]
    assert [c.metric for c in preset.criteria] == ["quote"]
    assert preset.criteria[0].threshold == "JPY"


def test_every_preset_explains_itself():
    for preset in fs.PRESETS:
        assert len(preset.description) > 40, preset.name
        assert fs.PRESETS_BY_NAME[preset.name] is preset


# --- filters deliberately absent ----------------------------------------------

def test_no_bid_ask_or_spread_column_is_offered():
    """Yahoo quotes an ask below the bid on four of fourteen majors."""
    text = " ".join(m.key + m.label for m in fs.METRICS).lower()
    assert "bid" not in text and "ask" not in text and "spread" not in text


def test_no_rate_PATH_filter_is_offered():
    """"Countries with rising rates" needs a path, and the BIS publishes
    a level. The calendar carries the forward-looking half instead."""
    keys = " ".join(m.key for m in fs.METRICS).lower()
    assert "rising" not in keys and "hiking" not in keys


# --- the results table --------------------------------------------------------

def test_a_column_almost_no_row_reports_is_still_numeric():
    """A column that comes back object dtype sorts as text — how the ETF
    screener's P/E broke."""
    rows = tuple(_row(f"X{i}", "AUD", "JPY", spot=1.0) for i in range(3))
    frame = fs.results_frame(fs.run(rows, [])[0])
    assert frame["Carry %"].isna().all()
    assert pd.api.types.is_numeric_dtype(frame["Carry %"])


def test_the_table_has_every_declared_column_even_when_empty():
    assert list(fs.results_frame([]).columns) == list(fs.TABLE_COLUMNS)


def test_every_declared_numeric_column_is_in_the_table():
    for column in fs.NUMERIC_COLUMNS:
        assert column in fs.TABLE_COLUMNS, column


def test_percent_columns_do_not_use_the_multiplying_format():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fs.column_config).strip())
    formats = [kw.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               for kw in node.keywords
               if kw.arg == "format" and isinstance(kw.value, ast.Constant)]
    assert formats and "percent" not in formats
    assert len([f for f in formats if "%%" in f]) >= 5


def test_the_skew_column_states_its_window():
    """The screener measures skew over a year and each pair's page over
    five, so the two figures differ and the reader has to be told."""
    config = fs.column_config()
    help_text = config["Skew"].help if hasattr(config["Skew"], "help") else ""
    source = fs.column_config.__doc__ or ""
    import inspect
    body = inspect.getsource(fs.column_config)
    assert "YEAR" in body and "five" in body


# --- the universe loader, mocked ----------------------------------------------

def _install(monkeypatch, frame, rates=None, factors=None):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: frame
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    monkeypatch.setattr(fd, "load_policy_rates",
                        lambda: rates if rates is not None
                        else fd.PolicyRates(error="none"))
    monkeypatch.setattr(fd, "load_ppp",
                        lambda: factors if factors is not None
                        else fd.PppFactors(error="none"))


def _price_frame(symbols, rows=300):
    import numpy as np
    index = pd.date_range("2025-09-01", periods=rows, freq="B")
    rng = np.random.default_rng(5)
    closes = pd.DataFrame(
        {s: 100 * np.cumprod(1 + rng.normal(0, 0.004, rows)) for s in symbols},
        index=index)
    return pd.concat({"Close": closes}, axis=1)


def test_the_universe_computes_return_volatility_skew_and_range(monkeypatch):
    symbols = [p.symbol for p in fd.PAIRS]
    _install(monkeypatch, _price_frame(symbols))
    rows, error = fs.load_universe.__wrapped__()
    assert error is None and len(rows) == len(fd.PAIRS)
    for row in rows:
        assert row.spot is not None
        assert row.volatility_pct is not None
        assert row.skew is not None
        assert 0 <= row.range_position_pct <= 100


def test_the_differential_is_attached_when_rates_resolve(monkeypatch):
    symbols = [p.symbol for p in fd.PAIRS]
    rates = fd.PolicyRates(rates={
        "AUD": fd.PolicyRate("AUD", 4.35, "2026-08"),
        "JPY": fd.PolicyRate("JPY", 1.0, "2026-08")})
    _install(monkeypatch, _price_frame(symbols), rates=rates)
    rows, _ = fs.load_universe.__wrapped__()
    aud = next(r for r in rows if r.code == "AUDJPY")
    assert aud.rate_differential_pct == pytest.approx(3.35)
    assert aud.carry_ratio == pytest.approx(3.35 / aud.volatility_pct)
    # A pair whose legs have no rate stays unjudged rather than zeroed.
    eur = next(r for r in rows if r.code == "EURUSD")
    assert eur.rate_differential_pct is None


def test_a_pair_with_no_prices_still_appears(monkeypatch):
    symbols = [p.symbol for p in fd.PAIRS if p.code != "CHFJPY"]
    _install(monkeypatch, _price_frame(symbols))
    rows, _ = fs.load_universe.__wrapped__()
    assert len(rows) == len(fd.PAIRS)
    assert next(r for r in rows if r.code == "CHFJPY").spot is None


def test_a_failed_download_returns_an_error_rather_than_raising(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    def boom(*a, **k):
        raise RuntimeError("down")
    fake.download = boom
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    rows, error = fs.load_universe.__wrapped__()
    assert rows == () and error


def test_an_empty_download_is_an_error(monkeypatch):
    _install(monkeypatch, pd.DataFrame())
    rows, error = fs.load_universe.__wrapped__()
    assert rows == () and error


def test_the_haven_flag_is_set_from_the_quote_leg(monkeypatch):
    symbols = [p.symbol for p in fd.PAIRS]
    _install(monkeypatch, _price_frame(symbols))
    rows, _ = fs.load_universe.__wrapped__()
    assert next(r for r in rows if r.code == "AUDJPY").funded_in_haven
    assert not next(r for r in rows if r.code == "EURGBP").funded_in_haven
