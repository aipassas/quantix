"""The coin screener: filters, presets, and the fields it refuses to use.

Two rules this file enforces, both learned from shipped bugs:

  - a preset that matches nothing reads as "no such coins exist", so
    every preset must return something on a realistic universe;
  - a coin that cannot be judged is not a coin that failed, and the
    count of unjudged coins must reach the reader.
"""
import numpy as np
import pandas as pd
import pytest

import crypto_data as cd
import crypto_screener as cs


def _row(symbol="x", **kwargs):
    base = dict(coin_id=symbol, symbol=symbol, name=symbol.upper())
    base.update(kwargs)
    return cd.CoinRow(**base)


def _universe():
    """A miniature of the real one: a capped major, an uncapped major, a
    stablecoin, a deep drawdown, and a coin missing a field."""
    return (
        _row("btc", name="Bitcoin", market_cap_rank=1, market_cap=1.57e12,
             price=78249.0, volume_24h=4.15e10, circulating_supply=20.07e6,
             max_supply=21e6, change_1y_pct=-29.8, ath_change_pct=-37.9),
        _row("eth", name="Ethereum", market_cap_rank=2, market_cap=2.94e11,
             price=2438.0, volume_24h=1.68e10, circulating_supply=1.2e8,
             max_supply=None, change_1y_pct=-40.0, ath_change_pct=-50.0),
        _row("usdt", name="Tether", market_cap_rank=3, market_cap=1.83e11,
             price=1.0, volume_24h=9.0e10, circulating_supply=1.83e11,
             max_supply=None, change_1y_pct=0.1, ath_change_pct=-0.5),
        _row("doge", name="Dogecoin", market_cap_rank=9, market_cap=1.34e10,
             price=0.086, volume_24h=1.07e9, circulating_supply=1.55e11,
             max_supply=None, change_1y_pct=-60.0, ath_change_pct=-88.0),
        _row("new", name="Newcoin", market_cap_rank=200, market_cap=2.0e8,
             price=1.5, volume_24h=1.0e7, circulating_supply=1e6,
             max_supply=2e6, change_1y_pct=None, ath_change_pct=-10.0),
    )


# --- the filters --------------------------------------------------------------

def test_a_numeric_filter_selects_on_the_stated_comparison():
    matches, _ = cs.run(_universe(),
                        [cs.Criterion("market_cap", ">", 1e11)])
    assert {m.symbol for m in matches} == {"BTC", "ETH", "USDT"}


def test_criteria_are_combined_with_AND():
    matches, _ = cs.run(_universe(), [
        cs.Criterion("market_cap", ">", 1e10),
        cs.Criterion("change_1y_pct", "<", -35.0)])
    assert {m.symbol for m in matches} == {"ETH", "DOGE"}


def test_results_come_back_largest_first():
    matches, _ = cs.run(_universe(), [cs.Criterion("market_cap", ">", 0)])
    caps = [m.row.market_cap for m in matches]
    assert caps == sorted(caps, reverse=True)


@pytest.mark.parametrize("operator,expected", [
    (">", {"BTC"}), (">=", {"BTC", "ETH"}),
    ("<", {"USDT", "DOGE", "NEW"}), ("<=", {"USDT", "DOGE", "NEW", "ETH"}),
])
def test_every_operator_is_implemented(operator, expected):
    matches, _ = cs.run(_universe(),
                        [cs.Criterion("market_cap", operator, 2.94e11)])
    assert {m.symbol for m in matches} == expected


def test_a_computed_property_is_filterable():
    """turnover and pct_of_max_mined are properties, not fields — a
    getattr-on-fields-only implementation would silently judge nothing."""
    matches, unjudged = cs.run(_universe(),
                               [cs.Criterion("turnover", ">", 0.4)])
    assert {m.symbol for m in matches} == {"USDT"}
    assert unjudged == 0


# --- unjudged is not failed ---------------------------------------------------

def test_a_coin_missing_the_field_is_unjudged_not_failed():
    """Silently dropping them would report "1 match" from a universe
    where most were never examined."""
    matches, unjudged = cs.run(_universe(),
                               [cs.Criterion("change_1y_pct", ">", -50.0)])
    assert "NEW" not in {m.symbol for m in matches}
    assert unjudged == 1


def test_an_uncapped_coin_is_unjudged_by_a_supply_screen_not_failed():
    """THE UNCAPPED TRAP, at screener level. Three of these five coins
    have no supply cap; a "percent mined" screen cannot judge them, and
    treating their None as 0 would fail all three."""
    matches, unjudged = cs.run(
        _universe(), [cs.Criterion("pct_of_max_mined", ">", 50.0)])
    # NEW is at exactly 50% mined, so a strict > excludes it — judged and
    # failed, which is a different outcome from the three uncapped coins.
    assert {m.symbol for m in matches} == {"BTC"}
    assert unjudged == 3
    inclusive, _ = cs.run(
        _universe(), [cs.Criterion("pct_of_max_mined", ">=", 50.0)])
    assert {m.symbol for m in inclusive} == {"BTC", "NEW"}


def test_no_criteria_passes_everything():
    matches, unjudged = cs.run(_universe(), [])
    assert len(matches) == len(_universe()) and unjudged == 0


# --- presets ------------------------------------------------------------------

def test_every_preset_returns_something_on_a_realistic_universe():
    """THE BUG THIS PREVENTS. The bond screener shipped a preset that
    matched nothing, which reads as "no such funds exist" rather than as
    a filter that is too tight."""
    universe = _universe()
    for preset in cs.PRESETS:
        matches, _ = cs.run(universe, preset.criteria)
        assert matches, f"{preset.name} matched nothing"


def test_every_preset_names_a_metric_that_exists():
    for preset in cs.PRESETS:
        for criterion in preset.criteria:
            assert criterion.metric in cs.METRICS_BY_KEY, criterion.metric
            assert criterion.operator in cs.OPERATORS


def test_every_preset_explains_itself():
    for preset in cs.PRESETS:
        assert len(preset.description) > 40, preset.name
        assert cs.PRESETS_BY_NAME[preset.name] is preset


def test_the_large_cap_preset_is_a_rank_cut_not_a_hand_written_list():
    """A hard-coded ["BTC", "ETH"] goes stale the day something overtakes
    one of them."""
    preset = cs.PRESETS_BY_NAME["Large caps"]
    assert [c.metric for c in preset.criteria] == ["market_cap_rank"]


def test_a_drawdown_preset_does_not_call_a_drawdown_a_discount():
    preset = cs.PRESETS_BY_NAME["Deeply discounted"]
    assert "not a verdict" in preset.description or "not a discount" in preset.description


# --- categories are refused ---------------------------------------------------

def test_category_tags_are_not_offered_as_a_filter():
    """Measured: Bitcoin, Ethereum, Solana AND Dogecoin all carry "Smart
    Contract Platform" as their first tag, and Bitcoin, Ethereum and
    Solana all carry "FTX Holdings". A "DeFi" screen built on that field
    would return Bitcoin."""
    assert not [m for m in cs.METRICS if "categor" in m.key.lower()]
    assert not [m for m in cs.METRICS if "categor" in m.label.lower()]


def test_every_metric_is_readable_off_a_coin_row():
    row = _row("btc", market_cap=1.0, price=1.0, volume_24h=1.0,
               circulating_supply=1.0, max_supply=2.0, market_cap_rank=1,
               change_24h_pct=1.0, change_7d_pct=1.0, change_30d_pct=1.0,
               change_1y_pct=1.0, ath_change_pct=-1.0)
    for metric in cs.METRICS:
        assert cs._value(row, metric.key) is not None, metric.key


# --- formatting ---------------------------------------------------------------

def test_a_sub_cent_price_keeps_its_significant_digits():
    """Crypto prices span eight orders of magnitude in one universe. A
    fixed two-decimal money format renders half of it as $0.00."""
    assert cs.crypto_compact(0.0000123) == "$0.0000123"
    assert cs.crypto_compact(0.086) == "$0.086"


def test_large_figures_are_compacted():
    assert cs.crypto_compact(1.57e12) == "$1.57T"
    assert cs.crypto_compact(2.94e11) == "$294.00B"
    assert cs.crypto_compact(1.0e7) == "$10.00M"
    assert cs.crypto_compact(78249.0) == "$78.25K"


def test_a_missing_price_is_never_rendered_as_zero():
    assert cs.crypto_compact(None) == "Not reported"
    assert cs.crypto_compact(float("nan")) == "Not reported"
    assert cs.crypto_compact("banana") == "Not reported"


def test_a_true_zero_is_distinguishable_from_a_missing_one():
    assert cs.crypto_compact(0) == "$0"


def test_describe_renders_a_threshold_in_its_own_units():
    assert cs.describe(cs.Criterion("market_cap", ">", 1e9)) == "Market cap > $1.00B"
    assert cs.describe(cs.Criterion("change_1y_pct", ">", 0)) == "1-year change > 0%"
    assert cs.describe(cs.Criterion("market_cap_rank", "<=", 10)) == "Rank <= 10"


# --- the results table --------------------------------------------------------

def test_a_column_no_row_reports_is_still_numeric():
    """THE DTYPE TRAP. pandas gives an all-None column object dtype, and
    Streamlit then sorts it as text — which is how the ETF screener's P/E
    column broke once a filter narrowed it to funds reporting none."""
    rows = tuple(_row(f"c{i}", market_cap=1e9, market_cap_rank=i,
                      max_supply=None, circulating_supply=1e6)
                 for i in range(3))
    matches, _ = cs.run(rows, [])
    frame = cs.results_frame(matches)
    assert frame["Mined %"].isna().all()
    assert pd.api.types.is_numeric_dtype(frame["Mined %"])


def test_the_table_has_every_declared_column_even_when_empty():
    frame = cs.results_frame([])
    assert list(frame.columns) == list(cs.TABLE_COLUMNS)


def test_percent_columns_do_not_use_the_multiplying_format():
    """NumberColumn(format="percent") multiplies the stored value by 100,
    and every figure in this app is already percent-valued."""
    import ast
    import inspect

    # Checked as a KEYWORD VALUE in the AST, not as a substring: this
    # function's own docstring explains the trap and therefore contains
    # the exact string a substring search looks for. The first version
    # of this test failed against correct code for that reason.
    tree = ast.parse(inspect.getsource(cs.column_config).strip())
    formats = [kw.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               for kw in node.keywords
               if kw.arg == "format" and isinstance(kw.value, ast.Constant)]
    assert formats, "no column formats were declared at all"
    assert "percent" not in formats, (
        f"format=\"percent\" multiplies by 100: {formats}")
    percent_columns = [f for f in formats if "%%" in f]
    assert len(percent_columns) >= 4, formats


def test_every_numeric_column_is_declared_numeric():
    for column in cs.NUMERIC_COLUMNS:
        assert column in cs.TABLE_COLUMNS, column
