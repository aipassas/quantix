"""Feature walkthroughs: the library, and the invariant that keeps it true.

The load-bearing test here is `test_every_named_control_still_exists`. A
how-to that quietly stops matching the app is worse than no how-to — it
costs the reader their trust in the rest of the documentation — so every
control a step names is checked against finance.py's own source. Rename
a button and this suite fails instead of the instructions rotting.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import support
import walkthroughs

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


# --- the anti-rot invariant ---------------------------------------------------

def test_every_named_control_still_exists_in_the_app():
    """Every on-screen label a step tells the reader to click must still
    be in finance.py. This is the whole point of recording `control`."""
    source = FINANCE.read_text()
    missing = [
        (w.id, control)
        for w in walkthroughs.all_walkthroughs()
        for control in w.controls
        if control not in source
    ]
    assert not missing, f"walkthroughs name controls the app no longer has: {missing}"


def test_the_invariant_actually_has_something_to_check():
    """A test that would pass over an empty list proves nothing. Pin a
    floor so deleting every `control=` cannot silently disarm it."""
    named = sum(len(w.controls) for w in walkthroughs.all_walkthroughs())
    assert named >= 20, f"only {named} controls named; the invariant is barely testing anything"


def test_controls_are_exact_labels_not_paraphrases():
    """A control must be a literal UI string, so a substring check is
    meaningful. Sentence-shaped values would match nothing useful."""
    for w in walkthroughs.all_walkthroughs():
        for control in w.controls:
            assert control == control.strip()
            assert not control.endswith("."), f"{w.id}: {control!r} looks like prose"


# --- coverage -----------------------------------------------------------------

def test_the_library_covers_the_major_features():
    """The ticket exists because ~3 of ~15 features had a how-to."""
    assert len(walkthroughs.all_walkthroughs()) >= 12


@pytest.mark.parametrize("topic", [
    "screen", "portfolio", "export", "alert", "journal",
    "earnings", "watchlist", "dcf", "keyboard",
])
def test_each_major_feature_is_findable_by_its_obvious_word(topic):
    assert walkthroughs.search(topic), f"nothing found for {topic!r}"


def test_ids_and_titles_are_unique():
    ids = [w.id for w in walkthroughs.all_walkthroughs()]
    titles = [w.title for w in walkthroughs.all_walkthroughs()]
    assert len(set(ids)) == len(ids)
    assert len(set(titles)) == len(titles), "the picker lists titles; duplicates are unpickable"


def test_every_walkthrough_has_steps_and_says_where_to_go():
    for w in walkthroughs.all_walkthroughs():
        assert w.steps, f"{w.id} has no steps"
        assert w.where.strip(), f"{w.id} does not say where the feature lives"
        assert w.summary.strip(), f"{w.id} has no summary"


def test_by_id_round_trips_and_misses_return_none():
    first = walkthroughs.all_walkthroughs()[0]
    assert walkthroughs.by_id(first.id) is first
    assert walkthroughs.by_id("no_such_walkthrough") is None
    assert walkthroughs.by_id("") is None


# --- the video slot -----------------------------------------------------------

def test_no_walkthrough_ships_a_video_url_in_this_build():
    """Not an aspiration — an assertion that nothing points at a URL that
    was never recorded. A dead embed is worse than written steps."""
    assert walkthroughs.with_video() == ()


def test_the_absence_of_video_is_stated_rather_than_left_to_be_discovered():
    note = walkthroughs.VIDEOS_NOT_RECORDED
    assert "written rather than filmed" in note
    # It must describe the absence AND the way out of it.
    assert "video" in note.lower()


def test_a_walkthrough_with_a_url_reports_that_it_has_one():
    """The path has to work for whoever records them later."""
    from dataclasses import replace
    original = walkthroughs.all_walkthroughs()[0]
    assert original.has_video is False
    with_url = replace(original, video_url="https://example.com/v.mp4")
    assert with_url.has_video is True


def test_whitespace_is_not_a_video():
    from dataclasses import replace
    blank = replace(walkthroughs.all_walkthroughs()[0], video_url="   ")
    assert blank.has_video is False


# --- body rendering -----------------------------------------------------------

def test_the_body_numbers_the_steps_and_carries_the_location():
    w = walkthroughs.by_id("wt_export")
    body = w.body
    assert "Where: " in body
    assert "1. " in body and "2. " in body


def test_the_body_includes_the_control_so_search_can_match_it():
    w = walkthroughs.by_id("wt_export")
    assert "Generate PDF Report" in w.body


# --- search -------------------------------------------------------------------

def test_search_ranks_the_more_relevant_walkthrough_first():
    hits = walkthroughs.search("decision journal conviction")
    assert hits and hits[0].id == "wt_journal"


def test_an_empty_search_returns_nothing_rather_than_everything():
    assert walkthroughs.search("") == ()
    assert walkthroughs.search("    ") == ()


def test_search_is_case_insensitive():
    assert walkthroughs.search("PORTFOLIO") == walkthroughs.search("portfolio")


# --- integration with the existing help corpus --------------------------------

def test_walkthroughs_join_the_one_help_index_rather_than_a_second_one():
    index = support.build_index()
    ids = {a.id for a in index}
    for w in walkthroughs.all_walkthroughs():
        assert w.id in ids, f"{w.id} is missing from the assembled help index"


def test_they_are_categorised_so_browse_can_separate_them():
    assert "How-to" in support.categories()
    assert len(support.browse("How-to")) == len(walkthroughs.all_walkthroughs())


def test_the_existing_corpus_is_not_displaced():
    """Adding how-tos must not evict the metric and chart explanations."""
    index = support.build_index()
    categories = {a.category for a in index}
    assert {"Metric", "Chart", "How-to"} <= categories
    assert len(index) > len(walkthroughs.all_walkthroughs()) + 10


def test_a_how_to_question_finds_the_walkthrough_in_the_shared_search():
    hits = support.search("how do I export a report")
    assert hits and hits[0].id == "wt_export"


def test_support_does_not_import_walkthroughs_at_module_scope():
    """The dependency must run support -> walkthroughs one way only;
    a module-scope import here would be a cycle waiting to happen."""
    source = (Path(support.__file__)).read_text()
    module_level = [
        line for line in source.splitlines()
        if line.startswith("import walkthroughs") or line.startswith("from walkthroughs")
    ]
    assert not module_level, "walkthroughs must be imported inside build_index()"


# --- UI wiring ----------------------------------------------------------------

def test_the_library_is_actually_rendered_by_the_app():
    source = FINANCE.read_text()
    assert "import walkthroughs" in source
    assert "all_walkthroughs()" in source


def test_the_app_shows_the_video_note_rather_than_reimplementing_it():
    source = FINANCE.read_text()
    assert "VIDEOS_NOT_RECORDED" in source


def test_the_app_renders_a_player_only_when_a_video_exists():
    """st.video must be guarded by has_video, or an entry with no
    recording renders an empty player."""
    source = FINANCE.read_text()
    assert "st.video(" in source, "the video path must be wired, not removed"
    for i, line in enumerate(source.splitlines()):
        if "st.video(" in line:
            window = "\n".join(source.splitlines()[max(0, i - 4):i])
            assert "has_video" in window, "st.video is not guarded by has_video"


def test_the_video_note_is_only_shown_while_nothing_is_recorded():
    """Once videos exist the caption must stop claiming they do not."""
    source = FINANCE.read_text()
    assert "if not walkthroughs.with_video():" in source
