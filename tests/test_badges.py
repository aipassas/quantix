"""Badges for what you did, never for what the market did.

THE TEST THAT MATTERS MOST is test_no_badge_is_awarded_for_a_return.
Measured over 73,457 observations: a random three-month position is
profitable 65.2% of the time, so "five consecutive profitable trades"
happens to one person in eight by chance — and it ran at 25.9% in 2017
against 2.3% in 2022, making the same badge eleven times easier in one
year than another. The ticket names that badge and "First 5-Bagger";
both are declined, and these tests are what keeps them declined.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import badges
from config import BADGES

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


class _Entry:
    def __init__(self, decided_on="2026-09-01", reviewed_at="", ticker="AAPL"):
        self.decided_on = decided_on
        self.reviewed_at = reviewed_at
        self.ticker = ticker
        self.reasoning = "because"


class _ContestEntry:
    def __init__(self, period):
        self.period = period


class _Streak:
    def __init__(self, longest=0, total_days=0):
        self.longest = longest
        self.total_days = total_days


def _badge(badge_id, badges_):
    return next(b for b in badges_ if b.spec.id == badge_id)


# --- no outcome badges --------------------------------------------------------

def test_no_badge_is_awarded_for_a_return():
    """The ticket asks for 'First 5-Bagger' and 'N consecutive profitable
    trades'. Neither exists, and nothing here counts money."""
    import re
    # WORD boundaries. A bare substring ban matched "again" inside
    # "against" and "returning" — the same declaration-not-substring trap
    # this suite records for CSS and for the share module's ban list.
    banned = ("profit", "profitable", "bagger", "gain", "gains", "return",
              "returns", "winner", "winning", "5x", "multibagger")
    pattern = re.compile(r"\b(" + "|".join(banned) + r")\b")
    for spec in badges.BADGE_SPECS:
        blob = f"{spec.id} {spec.name} {spec.blurb} {spec.unit}".lower()
        hit = pattern.search(blob)
        assert hit is None, f"{spec.id} looks outcome-based: {hit.group(0)!r}"
        assert "%" not in blob, f"{spec.id} quotes a percentage"


def test_no_evidence_field_is_a_return():
    """The counts a badge can read are all deliberate acts. A field
    holding a return would let an outcome badge be added later without
    anyone noticing."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(badges.Evidence)}
    assert fields == {"decisions", "reviews", "journal_months", "longest_streak",
                      "active_days", "contest_months", "saved_screens",
                      "watchlists"}
    for name in fields:
        assert not any(w in name for w in ("profit", "return", "gain", "pnl"))


def test_the_refusal_carries_the_measurement_that_justifies_it():
    """A refusal without its evidence is an opinion."""
    for figure in ("65.2%", "73,457", "one person in eight", "25.9%", "2.3%"):
        assert figure in badges.WHY_NO_OUTCOME_BADGES, figure


def test_the_refusal_says_it_is_a_measurement_not_a_preference():
    assert "measurement rather than a preference" in badges.WHY_NO_OUTCOME_BADGES


def test_every_badge_counts_a_deliberate_act():
    allowed_units = {"decisions", "reviews", "months", "days", "active days",
                     "months entered", "saved screens", "watchlists"}
    for spec in badges.BADGE_SPECS:
        assert spec.unit in allowed_units, f"{spec.id}: {spec.unit}"
        assert spec.target > 0


# --- the set ------------------------------------------------------------------

def test_the_badge_set_has_an_exact_size():
    """Adding one should break this and be extended deliberately.
    Relaxing it to >= would let a badge silently disappear — the same
    invariant quick_stats and support already carry."""
    assert len(badges.BADGE_SPECS) == 15


def test_every_badge_id_is_unique():
    ids = [s.id for s in badges.BADGE_SPECS]
    assert len(ids) == len(set(ids))


def test_every_badge_has_a_known_source():
    for spec in badges.BADGE_SPECS:
        assert spec.source in badges.SOURCES


def test_every_source_actually_has_badges():
    """A declared source with nothing in it is a dead branch in the
    panel."""
    grouped = badges.by_source(badges.evaluate(badges.Evidence()))
    for source in badges.SOURCES:
        assert grouped[source], f"{source} has no badges"


def test_every_badge_has_a_blurb_that_explains_what_earns_it():
    for spec in badges.BADGE_SPECS:
        assert len(spec.blurb) > 30, spec.id
        assert spec.name.strip()


def test_every_badge_is_reachable_from_the_evidence():
    """A spec whose id is not in count_for() would sit permanently at
    zero and could never be earned."""
    full = badges.Evidence(decisions=999, reviews=999, journal_months=999,
                           longest_streak=999, active_days=999,
                           contest_months=999, saved_screens=999, watchlists=999)
    for badge in badges.evaluate(full):
        assert badge.earned, f"{badge.spec.id} cannot be earned by any evidence"


# --- counting -----------------------------------------------------------------

def test_a_recorded_decision_counts():
    evidence = badges.evidence_from(journal_entries=[_Entry(), _Entry()])
    assert evidence.decisions == 2


def test_a_review_means_one_written_after_the_outcome():
    """Counting entries that merely have a reasoning field would award
    the hardest habit in investing for doing nothing."""
    evidence = badges.evidence_from(journal_entries=[
        _Entry(reviewed_at="2026-09-10T00:00:00"), _Entry(), _Entry()])
    assert evidence.decisions == 3
    assert evidence.reviews == 1


def test_a_blank_reviewed_at_is_not_a_review():
    evidence = badges.evidence_from(journal_entries=[_Entry(reviewed_at="   ")])
    assert evidence.reviews == 0


def test_journal_months_are_distinct_months_not_entries():
    """Otherwise 'a year of decisions' is earned in one afternoon."""
    evidence = badges.evidence_from(journal_entries=[
        _Entry(decided_on="2026-09-01"), _Entry(decided_on="2026-09-15"),
        _Entry(decided_on="2026-08-02")])
    assert evidence.decisions == 3
    assert evidence.journal_months == 2


def test_an_entry_with_no_date_does_not_invent_a_month():
    evidence = badges.evidence_from(journal_entries=[_Entry(decided_on="")])
    assert evidence.journal_months == 0


def test_contest_months_are_distinct_periods():
    evidence = badges.evidence_from(contest_entries=[
        _ContestEntry("2026-09"), _ContestEntry("2026-09"),
        _ContestEntry("2026-10")])
    assert evidence.contest_months == 2


def test_the_streak_counts_come_from_the_streak_record():
    evidence = badges.evidence_from(streak=_Streak(longest=12, total_days=40))
    assert evidence.longest_streak == 12 and evidence.active_days == 40


def test_a_missing_streak_is_zero_not_a_crash():
    assert badges.evidence_from(streak=None).longest_streak == 0


def test_a_negative_count_cannot_be_passed_in():
    evidence = badges.evidence_from(saved_screens=-5, watchlists=-1)
    assert evidence.saved_screens == 0 and evidence.watchlists == 0


def test_a_malformed_record_does_not_take_the_panel_down():
    class _Broken:
        pass
    evidence = badges.evidence_from(journal_entries=[_Broken(), _Entry()])
    assert evidence.decisions == 2 and evidence.reviews == 0


# --- earning ------------------------------------------------------------------

def test_a_badge_is_earned_at_its_target_not_above_it():
    week = _badge("week_streak", badges.evaluate(badges.Evidence(longest_streak=7)))
    assert week.earned and week.remaining == 0


def test_a_badge_one_short_is_not_earned():
    week = _badge("week_streak", badges.evaluate(badges.Evidence(longest_streak=6)))
    assert not week.earned and week.remaining == 1


def test_nothing_is_earned_from_an_empty_record():
    assert badges.earned(badges.evaluate(badges.Evidence())) == ()


def test_the_fraction_is_capped_at_one():
    """st.progress raises above 1.0."""
    badge = _badge("first_thesis", badges.evaluate(badges.Evidence(decisions=500)))
    assert badge.fraction == 1.0


def test_the_fraction_is_never_negative():
    for badge in badges.evaluate(badges.Evidence()):
        assert 0.0 <= badge.fraction <= 1.0


def test_earned_badges_are_listed_first():
    result = badges.evaluate(badges.Evidence(decisions=10, longest_streak=1))
    flags = [b.earned for b in result]
    assert flags == sorted(flags, reverse=True), "an unearned badge came first"


def test_unearned_badges_are_ordered_by_how_close_they_are():
    result = [b for b in badges.evaluate(badges.Evidence(decisions=9))
              if not b.earned]
    fractions = [b.fraction for b in result]
    assert fractions == sorted(fractions, reverse=True)


def test_the_order_is_stable_between_renders():
    evidence = badges.Evidence(decisions=3, longest_streak=3)
    first = [b.spec.id for b in badges.evaluate(evidence)]
    second = [b.spec.id for b in badges.evaluate(evidence)]
    assert first == second


# --- what is next -------------------------------------------------------------

def test_next_up_only_offers_badges_actually_started():
    """A badge at zero is not 'nearly there', and showing it as the next
    step invents momentum."""
    result = badges.next_up(badges.evaluate(badges.Evidence()))
    assert result == ()


def test_next_up_offers_the_closest_started_badges():
    result = badges.next_up(badges.evaluate(badges.Evidence(decisions=9)))
    assert result
    assert all(b.progress > 0 and not b.earned for b in result)
    assert result[0].spec.id == "ten_theses"


def test_next_up_is_capped():
    evidence = badges.Evidence(decisions=9, longest_streak=6, contest_months=5,
                               saved_screens=4, watchlists=2)
    assert len(badges.next_up(badges.evaluate(evidence), limit=2)) == 2


# --- the summary --------------------------------------------------------------

def test_an_empty_record_says_how_badges_are_earned():
    text = badges.summary(badges.evaluate(badges.Evidence()))
    assert text == badges.NOTHING_YET
    assert "never by" in text and "position going well" in text


def test_the_summary_counts_what_was_earned():
    text = badges.summary(badges.evaluate(badges.Evidence(decisions=1)))
    assert text.startswith("1 of 15 earned")
    assert "never for a position going well" in text


# --- no store, no consent -----------------------------------------------------

def test_the_module_persists_nothing():
    """Derived on every render, so there is no second copy to fall out of
    step and no new opt-in."""
    src = Path(badges.__file__).read_text()
    persistence = [ln.strip() for ln in src.splitlines()
                   if ln.strip().startswith(("import ", "from "))
                   and ("local_store" in ln or ln.strip() == "import json")]
    assert persistence == [], persistence
    for banned in ("shared_path", "store_path", "atomic_write_text",
                   "save_store", "write_text", "json.dumps"):
        assert banned not in src, banned


def test_badges_are_private_and_reach_no_shared_surface():
    """Nothing here goes to the feed, the profile or the leaderboard."""
    src = Path(badges.__file__).read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    for banned in ("following", "leaderboard", "peer_comparison", "peer_trending"):
        assert not any(banned in ln for ln in imports), banned


def test_evaluate_takes_counts_not_stores():
    """Passing the stores would let a badge reach into a record it was
    never meant to read."""
    import inspect
    assert list(inspect.signature(badges.evaluate).parameters) == ["evidence"]


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Badges ---")
    end = src.index("# --- end badges ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_states_why_no_badge_is_for_a_return():
    assert "BADGES_WHY_NO_OUTCOME" in _panel()


def test_the_panel_derives_through_the_module():
    panel = _panel()
    assert "bg_evidence_from(" in panel
    assert "bg_evaluate(" in panel
    assert "sorted(" not in panel, "the ordering belongs in the module"


def test_the_panel_feeds_every_source():
    panel = _panel()
    for argument in ("journal_entries=", "streak=", "contest_entries=",
                     "saved_screens=", "watchlists="):
        assert argument in panel, argument


def test_the_panel_shows_the_summary_and_what_is_next():
    panel = _panel()
    assert "bg_summary(" in panel
    assert "bg_next_up(" in panel


def test_the_panel_does_not_depend_on_another_tabs_local_name():
    """finance.py is a script, so name order is execution order.
    `_ij_store` is bound inside the Overview tab's journal expander, and
    reading it from the Portfolio tab is the fragility that already put a
    NameError in the leaderboard panel."""
    panel = _panel()
    assert "_ij_store" not in panel
    assert 'st.session_state.get("journal_store")' in panel


def test_the_panel_does_not_publish_anything():
    panel = _panel()
    for banned in ("following.publish", "leaderboard", "fl_publish"):
        assert banned not in panel
