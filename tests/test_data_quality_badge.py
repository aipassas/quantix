"""The data-quality badge in the sticky header.

Two things are worth pinning here. First that the badge cannot silently
lie: an unrecognised grade must not fall through to green, and the badge
must not depend on grade_icon, which has returned "" for every grade
since the emoji removal and would render a colourless blank.

Second that the badge is a POPOVER. The symbol header is a single
st.markdown string, and custom HTML cannot call back into Streamlit — so
a badge rendered into that line could never satisfy "click to expand".
The quick-stats strip in the same block already documents this; a test
stops the badge being "simplified" into markup later.
"""
import ast
import re
from pathlib import Path

import pytest

import data_quality


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


# --- the grade scale ----------------------------------------------------------

def test_every_grade_the_module_can_produce_has_a_colour():
    """The task named three colours; the scale has four levels. A grade
    with no colour would fall through to the unknown case and read as
    Poor on perfectly good data."""
    produced = {data_quality._grade(score) for score in range(0, 101)}
    assert produced == set(data_quality.GRADE_COLOURS), produced


def test_the_four_grades_are_four_distinct_colours():
    """Collapsing Fair into a neighbour to fit "green/yellow/red" would
    misreport data that is readable but not dependable."""
    assert len(set(data_quality.GRADE_COLOURS.values())) == 4


def test_an_unknown_grade_is_never_green():
    """An unrecognised grade is not evidence of good data."""
    assert data_quality.grade_colour("Nonsense") == data_quality.GRADE_COLOURS["Poor"]
    assert data_quality.grade_colour("") == data_quality.GRADE_COLOURS["Poor"]
    assert "not recognised" in data_quality.grade_meaning("Nonsense")


def test_the_colours_run_in_the_right_direction():
    """Excellent must not be the red one."""
    assert data_quality.grade_colour("Excellent") != data_quality.grade_colour("Poor")
    assert data_quality.grade_colour("Excellent").lower() == "#00ea77"
    assert data_quality.grade_colour("Poor").lower() == "#ef4444"


@pytest.mark.parametrize("grade", ["Excellent", "Good", "Fair", "Poor"])
def test_every_grade_says_what_it_means_for_the_numbers(grade):
    """A score alone does not tell a reader whether to trust a ratio."""
    meaning = data_quality.grade_meaning(grade)
    assert len(meaning) > 40 and meaning.endswith(".")


def test_grade_boundaries_are_unchanged():
    """The badge's colours are pinned to these cutoffs; moving one
    silently recolours every ticker."""
    assert data_quality._grade(90) == "Excellent"
    assert data_quality._grade(89.9) == "Good"
    assert data_quality._grade(75) == "Good"
    assert data_quality._grade(74.9) == "Fair"
    assert data_quality._grade(55) == "Fair"
    assert data_quality._grade(54.9) == "Poor"


# --- the badge in the app -----------------------------------------------------

def _badge_block(code_only: bool = False) -> str:
    """The badge's source. `code_only` strips Python comments.

    Needed because the comments deliberately name what the code must NOT
    do — stBaseButton-secondary, grade_icon — so a raw substring scan
    finds the prose explaining the rule and fails on correct code."""
    start = FINANCE.index("# --- Data quality badge")
    block = FINANCE[start:FINANCE.index("with _pm_slot:", start)]
    if not code_only:
        return block
    return "\n".join(line for line in block.splitlines()
                      if not line.lstrip().startswith("#"))


def test_the_badge_is_a_popover_not_markup():
    """The symbol header is one st.markdown string and custom HTML cannot
    call back into Streamlit, so a badge rendered there could never
    expand. Same constraint the quick-stats strip documents."""
    block = _badge_block()
    assert "st.popover(" in block
    assert 'key="dq_badge"' in block


def test_the_badge_renders_inside_the_sticky_header():
    """"Prominent" means always on screen. Below the fold in a tab is
    where it already was."""
    header_start = FINANCE.index("with symbol_header_container:")
    tabs_start = FINANCE.index("= st.tabs([")
    badge_at = FINANCE.index("# --- Data quality badge")
    assert header_start < badge_at < tabs_start


def test_the_badge_colour_comes_from_the_grade():
    block = _badge_block()
    assert "data_quality.grade_colour(" in block
    assert "_dq_colour" in block
    # ...and is applied to the control, not just computed.
    assert re.search(r"st-key-dq_badge.*?\{\{", block, re.S)


def test_the_badge_rule_out_specifies_the_main_secondary_rule():
    """finance.py paints main-area secondaries at (0,2,1) with
    !important. The obvious [class*="st-key-dq_badge"] button is (0,1,1)
    and loses outright — measured live, the badge came back with the
    ordinary grey border and slate text.

    Note the attribute: a POPOVER trigger is data-testid="stPopoverButton"
    with kind="secondary", so button_roles' stBaseButton-secondary form
    would not have matched it either."""
    block = _badge_block(code_only=True)
    assert ('[data-testid="stMain"] [class*="st-key-dq_badge"] '
            'button[kind="secondary"]') in block
    assert "stBaseButton-secondary" not in block, (
        "a popover trigger does not carry that testid")


def test_the_badge_does_not_use_the_dead_grade_icon():
    """_grade_icon has returned "" for every grade since the emoji
    removal; a badge built on it would render blank."""
    assert all(data_quality._grade_icon(s) == "" for s in (95, 80, 60, 20))
    assert "grade_icon" not in _badge_block(code_only=True)


def test_the_report_is_assessed_once_and_reused():
    """Two assessments of the same bundle could not disagree, but they
    could drift if one call site gained an argument the other did not."""
    assert FINANCE.count("assess_data_quality(") == 1
    assert "quality = data_quality_report" in FINANCE


def test_the_detail_section_still_exists_for_the_full_lists():
    """The badge carries what a reader needs to act; the exhaustive
    field-by-field lists stay where there is room, and the badge says so."""
    assert "Data Quality Report —" in FINANCE
    assert "Overview → Data Quality Report" in _badge_block()


def test_the_badge_reports_missing_freshness_rather_than_implying_it():
    """staleness_days is Optional. Printing "0 days ago" for a ticker that
    reported no filing date would be a fabricated freshness."""
    block = _badge_block()
    assert "staleness_days is not None" in block
    assert "freshness could not be measured" in block


def test_there_is_no_fake_refresh_timer():
    """The report is recomputed from the same bundle the page is built
    from on every run, so it is never staler than the figures beside it.
    A fragment would re-render a value closed over from this run and only
    LOOK live."""
    block = _badge_block()
    assert "st.fragment" not in block
    assert "run_every" not in block
