"""Clearing the Stock Ticker box must not take the whole app down.

It used to. `load_ticker_bundle("")` raises ValueError("Empty ticker
name") from yfinance, and because finance.py is a script that exception
escaped before any tab rendered — so emptying one sidebar box replaced
the entire page with a traceback.

These are source-reading tests. The guard lives at module scope in a
Streamlit script, so there is no function to call; what can be checked,
and what actually matters, is that the guard EXISTS, that it stops the
script, and above all WHERE it sits relative to the sidebar and the
fetch. That placement is the design: too early and the user loses the
controls they need to recover, too late and the crash still happens.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
SOURCE = FINANCE.read_text()
LINES = SOURCE.splitlines()


def _line_of(needle: str) -> int:
    for index, line in enumerate(LINES, start=1):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} is not in finance.py")


def _last_line_of(needle: str) -> int:
    found = [i for i, line in enumerate(LINES, start=1) if needle in line]
    assert found, f"{needle!r} is not in finance.py"
    return found[-1]


# --- the guard exists ---------------------------------------------------------

def test_an_empty_ticker_is_guarded_before_the_data_fetch():
    guard = _line_of("if not ticker_symbol:")
    fetch = _line_of("load_ticker_bundle(ticker_symbol")
    assert guard < fetch, (
        "the guard must run before the fetch, or the fetch still raises")


def test_the_guard_stops_the_script_rather_than_falling_through():
    """Without st.stop() every section below would run against an empty
    ticker and fail one at a time instead of once, clearly."""
    guard = _line_of("if not ticker_symbol:")
    fetch = _line_of("load_ticker_bundle(ticker_symbol")
    block = "\n".join(LINES[guard - 1:fetch])
    assert "st.stop()" in block


def test_the_empty_state_names_the_ways_out():
    """A dead end that does not say how to leave is barely better than
    the traceback."""
    guard = _line_of("if not ticker_symbol:")
    block = "\n".join(LINES[guard - 1:guard + 30])
    lowered = block.lower()
    assert "stock ticker" in lowered
    assert "watchlist" in lowered
    assert "find a ticker" in lowered


def test_the_empty_state_says_nothing_was_lost():
    """Clearing a box and having the page vanish reads like data loss."""
    guard = _line_of("if not ticker_symbol:")
    block = "\n".join(LINES[guard - 1:guard + 30])
    assert "still saved" in block or "Nothing has been lost" in block


# --- placement is the design --------------------------------------------------

def test_the_guard_sits_after_the_whole_sidebar():
    """THE LOAD-BEARING ASSERTION. The watchlist and "Find a ticker" are
    exactly what someone needs to pick a new symbol; stopping at the
    input would take them off screen at the moment they became useful."""
    guard = _line_of("if not ticker_symbol:")
    last_sidebar = _last_line_of("st.sidebar.")
    assert last_sidebar < guard, (
        f"sidebar still rendering at line {last_sidebar}, after the guard at "
        f"{guard} — an empty ticker would hide the controls that fix it")


def test_the_ticker_input_is_far_above_the_guard():
    """Pins the reason the recovery button cannot assign the widget's own
    key: Streamlit forbids that once the widget has been instantiated."""
    widget = _line_of('"Stock Ticker", key="ticker_input"')
    guard = _line_of("if not ticker_symbol:")
    assert widget < guard


# --- whitespace is the same case ----------------------------------------------

def test_the_ticker_input_is_stripped():
    """A box holding only spaces is a DIFFERENT failure: "" raises
    ValueError("Empty ticker name") while "   " raises TypeError(
    "argument of type 'NoneType' is not iterable") after several failed
    HTTP round-trips. Stripping collapses both into one guarded case."""
    widget = _line_of('"Stock Ticker", key="ticker_input"')
    block = "\n".join(LINES[widget - 1:widget + 3])
    assert ".strip()" in block, "whitespace would reach yfinance unguarded"


@pytest.mark.parametrize("raw", ["", "   ", "\t", "\n", " \t "])
def test_every_blank_shape_reduces_to_the_guarded_case(raw):
    """What the app's own expression does to each of them."""
    assert raw.strip().upper() == ""


# --- recovery -----------------------------------------------------------------

def test_the_recovery_button_uses_the_deferred_switch():
    """Assigning "ticker_input" directly here would raise: Streamlit
    forbids writing a widget's key after that widget exists this run.
    The deferred "_pending_ticker" hand-off already exists for the
    sidebar watchlist, for exactly this reason."""
    guard = _line_of("if not ticker_symbol:")
    block = "\n".join(LINES[guard - 1:guard + 30])
    assert '"_pending_ticker"' in block
    assert 'st.session_state["ticker_input"]' not in block
    assert "st.rerun()" in block


def test_the_pending_switch_is_applied_before_the_widget_is_built():
    """The hand-off only works if the pop happens ABOVE the widget."""
    pop = _line_of('st.session_state.pop("_pending_ticker"')
    widget = _line_of('"Stock Ticker", key="ticker_input"')
    assert pop < widget


def test_the_recovery_offers_the_configured_default_not_a_literal():
    """Every place the default symbol is named must come from config —
    the button label, its help, and the value handed to the switch.

    Counting them matters: an earlier version of this test only looked
    for the token '"AAPL"', which a label reading "Analyse AAPL" does not
    contain, so a poison that hardcoded the label passed.
    """
    guard = _line_of("if not ticker_symbol:")
    block = "\n".join(LINES[guard - 1:guard + 30])
    assert block.count("CHART_DEFAULTS.default_ticker") >= 3, (
        "label, help text and the pending-switch value must all read the "
        "configured default")
    literal = re.search(r'"[^"]*\b(?:Analyse|Analyze)\s+[A-Z]{1,5}\b', block)
    assert literal is None, f"hardcoded symbol would drift from config: {literal}"


# --- the script still parses --------------------------------------------------

def test_finance_still_parses():
    ast.parse(SOURCE)


def test_there_is_exactly_one_empty_ticker_guard():
    """Two guards would mean one is unreachable, and the second would be
    the one nobody maintains."""
    assert len(re.findall(r"^if not ticker_symbol:$", SOURCE, re.M)) == 1
