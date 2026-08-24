"""The analysis date range: preset arithmetic and the range control.

Everything here takes `today` as an argument rather than calling
date.today(), so these tests are the same program in January as in
August — the class of test that passes for eleven months and then fails
on New Year's Day.

The load-bearing case is coerce(). st.date_input in range mode returns a
ONE-element tuple between the two clicks of a selection, and this control
feeds every fetch on the page, so unpacking it naively takes the whole
app down mid-click.
"""
import datetime
import re
from pathlib import Path

import pytest

import date_range as dr


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")

TODAY = datetime.date(2026, 8, 24)


# --- the half-made selection --------------------------------------------------

def test_a_one_element_tuple_holds_the_previous_range():
    """The intermediate state of every range selection. Unpacking it
    raises, and this control drives every fetch on the page."""
    fallback = (datetime.date(2025, 1, 1), TODAY)
    assert dr.coerce((datetime.date(2026, 3, 1),), fallback) == fallback


def test_an_empty_selection_holds_the_previous_range():
    fallback = (datetime.date(2025, 1, 1), TODAY)
    assert dr.coerce((), fallback) == fallback
    assert dr.coerce(None, fallback) == fallback


def test_a_bare_date_holds_the_previous_range():
    """date_input returns a plain date, not a tuple, when it is not in
    range mode — a shape this must survive rather than iterate."""
    fallback = (datetime.date(2025, 1, 1), TODAY)
    assert dr.coerce(datetime.date(2026, 3, 1), fallback) == fallback


def test_a_complete_selection_is_taken():
    picked = (datetime.date(2026, 1, 1), datetime.date(2026, 6, 30))
    assert dr.coerce(picked, (TODAY, TODAY)) == picked


def test_a_reversed_pair_is_ordered_rather_than_rejected():
    """A calendar can hand back end-before-start."""
    assert dr.coerce((datetime.date(2026, 6, 30), datetime.date(2026, 1, 1)),
                     (TODAY, TODAY)) == (datetime.date(2026, 1, 1),
                                         datetime.date(2026, 6, 30))


# --- preset arithmetic --------------------------------------------------------

@pytest.mark.parametrize("key,expected_start", [
    ("1M", datetime.date(2026, 7, 24)),
    ("3M", datetime.date(2026, 5, 24)),
    ("6M", datetime.date(2026, 2, 24)),
    ("YTD", datetime.date(2026, 1, 1)),
    ("1Y", datetime.date(2025, 8, 24)),
    ("5Y", datetime.date(2021, 8, 24)),
])
def test_each_preset_starts_where_it_says(key, expected_start):
    assert dr.resolve(key, TODAY) == (expected_start, TODAY)


def test_every_preset_ends_today():
    for key in dr.PRESET_KEYS:
        assert dr.resolve(key, TODAY)[1] == TODAY


def test_presets_are_calendar_months_not_thirty_day_blocks():
    """"3M" from 31 May is the end of February, not 2 March. Timedelta
    arithmetic drifts the anchor and makes two 1M clicks land on
    different days of the month."""
    assert dr._months_back(datetime.date(2026, 5, 31), 3) == datetime.date(2026, 2, 28)
    assert dr._months_back(datetime.date(2026, 3, 31), 1) == datetime.date(2026, 2, 28)


def test_a_leap_day_anchor_clamps_rather_than_raising():
    assert dr._months_back(datetime.date(2024, 2, 29), 12) == datetime.date(2023, 2, 28)
    assert dr._months_back(datetime.date(2024, 2, 29), 48) == datetime.date(2020, 2, 29)


def test_ytd_on_the_first_of_january_is_a_zero_length_window():
    """Not a crash, and problems() is what flags it as unusable."""
    jan_first = datetime.date(2026, 1, 1)
    start, end = dr.resolve("YTD", jan_first)
    assert start == end == jan_first
    assert dr.problems(start, end, jan_first)


def test_an_unknown_preset_returns_none_rather_than_a_default_window():
    """Substituting a different range for an unrecognised key would
    change what the whole page analyses without saying so."""
    assert dr.resolve("6Y", TODAY) is None
    assert dr.resolve("", TODAY) is None


def test_max_asks_for_more_than_any_listing_has():
    start, _ = dr.resolve("Max", TODAY)
    assert (TODAY - start).days > 365 * 25


def test_max_is_described_as_a_request_not_a_promise():
    """Nothing here knows a symbol's listing date without fetching it."""
    note = dr.PRESETS_BY_KEY["Max"].note
    assert "source has" in note or "data source" in note


# --- reflecting the current range ---------------------------------------------

def test_a_preset_shaped_range_is_recognised():
    """So the pill row shows the window in force rather than going blank
    on every reload."""
    start, end = dr.resolve("3M", TODAY)
    assert dr.matching_preset(start, end, TODAY) == "3M"


def test_a_hand_picked_range_matches_no_preset():
    assert dr.matching_preset(datetime.date(2026, 2, 3),
                              datetime.date(2026, 4, 7), TODAY) is None


# --- the summary --------------------------------------------------------------

def test_the_summary_states_the_span_not_just_the_dates():
    """Two dates a year apart look much like two dates three years apart
    at a glance; the day count is the part that is hard to see."""
    text = dr.describe(datetime.date(2025, 8, 24), TODAY)
    assert "24 Aug 2025" in text and "24 Aug 2026" in text
    assert "365 days" in text


def test_long_spans_also_state_years():
    text = dr.describe(datetime.date(2016, 8, 24), TODAY)
    assert "years" in text


def test_a_future_end_date_is_called_out():
    text = dr.describe(datetime.date(2026, 1, 1),
                       datetime.date(2027, 1, 1), TODAY)
    assert "future" in text


def test_problems_flags_a_window_too_short_to_analyse():
    assert dr.problems(TODAY, TODAY, TODAY)
    assert not dr.problems(datetime.date(2025, 8, 24), TODAY, TODAY)


# --- the control in the app ---------------------------------------------------

def _range_block() -> str:
    start = FINANCE.index("# --- Analysis date range")
    return FINANCE[start:FINANCE.index("st.sidebar.caption(date_range.describe", start)]


def test_there_is_one_range_control_not_two_date_inputs():
    block = _range_block()
    assert block.count("st.sidebar.date_input(") == 1
    assert 'st.sidebar.date_input("Start Date"' not in FINANCE
    assert 'st.sidebar.date_input("End Date"' not in FINANCE


def test_the_calendar_takes_a_computed_value_and_no_key():
    """Passing both value= and key= silently reverts the user's edit on
    the next rerun, and a keyed widget restores its old value over a
    just-applied preset. The range lives outside the widget instead."""
    block = _range_block()
    call = block[block.index("st.sidebar.date_input("):]
    call = call[:call.index("\n)")]
    assert "value=" in call
    assert "key=" not in call
    assert '"_dr_range"' in block


def test_the_pills_take_no_key_either():
    block = _range_block()
    call = block[block.index("st.sidebar.pills("):]
    call = call[:call.index("\n)")]
    assert "default=" in call and "key=" not in call


def test_the_preset_fires_on_a_changed_RANGE_not_a_stale_click():
    """Comparing against the last click instead would re-fire every run
    and drag the dates back over any manual edit — the bug the ticker
    autocomplete hit and documents."""
    block = _range_block()
    assert '_dr_resolved != st.session_state["_dr_range"]' in block


def test_the_app_routes_the_widget_through_coerce():
    """Without it, the app crashes between the two clicks of every range
    selection."""
    assert "date_range.coerce(_dr_picked" in FINANCE
    assert re.search(r"start_date, end_date = date_range\.coerce", FINANCE)


def test_the_range_still_feeds_the_loaders():
    """start_date/end_date are consumed by seven call sites; renaming
    them here would break the page silently."""
    assert "load_ticker_bundle(ticker_symbol, start_date, end_date" in FINANCE
    assert "load_macro_bundle(benchmark_symbol, start_date, end_date)" in FINANCE


def test_future_dates_cannot_be_selected():
    block = _range_block()
    assert "max_value=today" in block
    assert "min_value=date_range.EARLIEST_SELECTABLE" in block
