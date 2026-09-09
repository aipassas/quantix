"""Clicking a saved screener runs it, in one click.

Most of "Custom Screeners (Save & Re-Use)" was already built by the
earlier "Saved Screeners as Quick Templates" task: screener_templates.py
persists named filter sets WITH their universe, reorderable, seeded once
and then owned by the user. What was missing was the ticket's own words —
"re-run it later with one click". Clicking a template loaded the filters
and then said "Press Run Screen to execute it", which is two clicks, and
the second one is most of the friction the feature exists to remove.

Applying also left the PREVIOUS screen's results on display underneath
the new filters, with nothing saying they disagreed.

These are source-reading tests: the apply helper lives at module scope in
a Streamlit script, so what can be checked is that it sets the run flag,
clears the stale results, and that the flag is popped BELOW the builder —
the ordering is what makes the loaded criteria reach the run.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import screener_templates

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
SOURCE = FINANCE.read_text()
LINES = SOURCE.splitlines()


def _line_of(needle: str) -> int:
    for index, line in enumerate(LINES, start=1):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} is not in finance.py")


def _apply_body() -> str:
    start = _line_of("def _screener_apply_template(")
    end = _line_of("# --- Saved screeners ---")
    return "\n".join(LINES[start - 1:end])


# --- one click ----------------------------------------------------------------

def test_applying_a_template_triggers_the_run():
    """The ticket's own words: re-run it later with ONE click."""
    assert '"_screener_rerun"' in _apply_body()


def test_the_run_flag_is_popped_below_the_builder():
    """Ordering is the whole mechanism: the criteria and universe the
    apply helper writes must already be loaded when the run starts."""
    apply_at = _line_of("def _screener_apply_template(")
    builder = _line_of('key="screener_universe_text"')
    pop = _line_of('st.session_state.pop("_screener_rerun"')
    run = _line_of("_screener_results = run_screen(")
    assert apply_at < builder < pop < run


def test_the_message_does_not_ask_for_a_click_already_made():
    """It said "Press Run Screen to execute it" while having just run."""
    # The POP is the banner; the first mention of that key is the setter
    # inside the apply helper, which is not what this asserts about.
    banner = _line_of('st.session_state.pop("screener_applied_template"')
    block = "\n".join(LINES[banner - 1:banner + 8])
    assert "Press Run Screen" not in block
    assert "Ran" in block


def test_the_panel_caption_promises_what_it_now_does():
    caption = _line_of("One click loads a screen's filters")
    block = "\n".join(LINES[caption - 1:caption + 4])
    assert "runs it" in block


# --- stale results ------------------------------------------------------------

def test_applying_a_template_clears_the_previous_results():
    """Otherwise the builder shows THIS screen's filters while the table
    below shows the LAST one's results, with nothing saying so."""
    assert 'pop("screener_results_state", None)' in _apply_body()


def test_the_results_state_is_written_only_by_a_real_run():
    """If anything else wrote it, clearing on apply would not be enough."""
    writes = [line for line in LINES
              if 'st.session_state["screener_results_state"]' in line
              and "=" in line and "pop(" not in line]
    assert len(writes) == 1, f"expected one writer, found {writes}"


# --- one path for programmatic runs -------------------------------------------

def test_both_programmatic_callers_use_the_same_flag():
    """The empty-state's "remove this filter" action and a saved-screener
    click both start a screen. Two mechanisms would be two things to keep
    in step; a poison that changed one would leave the other working."""
    setters = re.findall(r'st\.session_state\["_screener_rerun"\]\s*=\s*True', SOURCE)
    popper = re.findall(r'st\.session_state\.pop\("_screener_rerun"', SOURCE)
    assert len(setters) >= 2, "both callers must set the shared flag"
    assert len(popper) == 1, "one place may consume it, or a run could be lost"


# --- what was already there stays there ---------------------------------------

def test_a_template_still_carries_its_universe():
    """The decision the whole module turns on: this screener filters a
    list you supply, so a saved screen that kept only the filters would
    give a different answer every time."""
    template = screener_templates.starter_templates()[0]
    assert template.universe, "a starter template must carry its tickers"


def test_the_apply_helper_loads_both_halves():
    body = _apply_body()
    assert '"screener_criteria"' in body
    assert '"screener_universe_text"' in body


def test_criteria_are_copied_not_aliased():
    """Editing the builder must not mutate the saved template in place."""
    assert "[dict(c) for c in" in _apply_body()


def test_a_template_with_no_universe_does_not_blank_the_box():
    """An older saved screen may predate universes; overwriting the box
    with an empty string would silently empty the screen."""
    body = _apply_body()
    assert "if _tpl.universe:" in body


def test_finance_still_parses():
    ast.parse(SOURCE)
