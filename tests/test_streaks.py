"""Activity streaks: a day you chose, a run you can actually lose.

THE TEST THAT MATTERS MOST is test_rendering_the_page_is_not_an_action.
Merely rendering this app writes per-user files, so a login streak would
increment on a refresh, a reconnect or an automated hit — a number that
only goes up and can never honestly be lost. CLAUDE.md already records
the same trap for the first-sign-in adoption prompt.
"""
import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streaks
from config import STREAKS

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
TODAY = datetime.date(2026, 9, 23)


def _d(offset: int) -> datetime.date:
    return TODAY - datetime.timedelta(days=offset)


def _store(*offsets) -> streaks.StreakStore:
    return streaks.StreakStore(tuple(sorted(_d(o).isoformat() for o in offsets)))


# --- what counts --------------------------------------------------------------

def test_rendering_the_page_is_not_an_action():
    """The whole reason this is an ACTIVITY streak and not a login one."""
    store, changed = streaks.record(streaks.StreakStore(), "render", TODAY)
    assert changed is False and store.days == ()
    for banned in ("render", "login", "visit", "open", "page_view"):
        assert banned not in streaks.ACTION_KEYS


def test_the_action_list_is_closed():
    store, changed = streaks.record(streaks.StreakStore(), "anything", TODAY)
    assert changed is False and store.days == ()


def test_every_action_is_something_the_reader_chose_to_do():
    assert set(streaks.ACTION_KEYS) == {
        "journal", "screen", "watchlist", "contest", "note", "alert"}
    for key in streaks.ACTION_KEYS:
        assert streaks.ACTION_LABELS[key].strip()


def test_the_empty_state_explains_why_opening_the_page_is_not_counted():
    assert "Simply opening the page is not" in streaks.NO_ACTIVITY
    for hint in ("journal", "screen", "watchlist"):
        assert hint in streaks.NO_ACTIVITY


# --- recording ----------------------------------------------------------------

def test_a_valid_action_records_today():
    store, changed = streaks.record(streaks.StreakStore(), "journal", TODAY)
    assert changed is True
    assert store.days == (TODAY.isoformat(),)


def test_a_second_action_on_the_same_day_changes_nothing():
    """Called on every qualifying click — a caller that saved
    unconditionally would rewrite the file dozens of times a session."""
    store, _ = streaks.record(streaks.StreakStore(), "journal", TODAY)
    store2, changed = streaks.record(store, "screen", TODAY)
    assert changed is False and store2 == store


def test_a_corrupt_store_is_never_added_to():
    store, changed = streaks.record(streaks.StreakStore(corrupt=True), "journal", TODAY)
    assert changed is False and store.days == ()


def test_the_history_is_trimmed_to_its_cap():
    store = streaks.StreakStore(tuple(
        (TODAY - datetime.timedelta(days=i)).isoformat()
        for i in range(STREAKS.max_days_retained + 50)))
    store, _ = streaks.record(store, "journal", TODAY + datetime.timedelta(days=1))
    assert len(store.days) == STREAKS.max_days_retained


# --- the run ------------------------------------------------------------------

def test_no_activity_is_a_streak_of_zero_not_of_one():
    """Counting today before anything happened would show a streak to
    somebody who has done nothing."""
    result = streaks.summarise(streaks.StreakStore(), TODAY)
    assert result.current == 0 and result.longest == 0 and result.alive is False


def test_three_consecutive_days_ending_today():
    result = streaks.summarise(_store(0, 1, 2), TODAY)
    assert result.current == 3 and result.longest == 3
    assert result.active_today is True


def test_a_run_ending_yesterday_is_still_alive():
    """The day is not over. Dropping it to zero at midnight would punish
    somebody who simply has not opened the app yet."""
    result = streaks.summarise(_store(1, 2, 3), TODAY)
    assert result.current == 3 and result.active_today is False


def test_a_run_ending_two_days_ago_is_over():
    result = streaks.summarise(_store(2, 3, 4), TODAY)
    assert result.current == 0
    assert result.longest == 3, "the run that ended is still visible"


def test_a_missed_day_breaks_the_run_with_no_grace():
    """A streak that survives a missed day is not a count of consecutive
    days, and redefining the word to keep a number alive is flattery."""
    result = streaks.summarise(_store(0, 1, 3, 4, 5), TODAY)
    assert result.current == 2
    assert result.longest == 3


def test_the_longest_run_can_be_an_old_one():
    result = streaks.summarise(_store(0, 1, 10, 11, 12, 13), TODAY)
    assert result.current == 2 and result.longest == 4


def test_duplicate_or_unordered_days_cannot_inflate_a_run():
    """Trusting the file's order would silently corrupt every length."""
    messy = streaks.StreakStore((_d(1).isoformat(), _d(0).isoformat(),
                                 _d(1).isoformat(), _d(2).isoformat()))
    assert streaks.summarise(messy, TODAY).current == 3


def test_a_single_day_is_a_run_of_one():
    result = streaks.summarise(_store(0), TODAY)
    assert result.current == 1 and result.longest == 1


def test_the_total_counts_days_not_runs():
    result = streaks.summarise(_store(0, 1, 5, 9, 10), TODAY)
    assert result.total_days == 5


def test_an_unparseable_day_is_ignored_rather_than_crashing():
    store = streaks.StreakStore(("not-a-date", _d(0).isoformat()))
    assert streaks.summarise(store, TODAY).current == 1


# --- the calendar strip -------------------------------------------------------

def test_the_strip_is_oldest_first_and_the_right_length():
    strip = streaks.recent_days(_store(0, 2), TODAY, span=5)
    assert len(strip) == 5
    assert strip[0][0] == _d(4).isoformat()
    assert strip[-1][0] == TODAY.isoformat()


def test_the_strip_marks_exactly_the_active_days():
    strip = streaks.recent_days(_store(0, 2), TODAY, span=4)
    assert [active for _, active in strip] == [False, True, False, True]


def test_the_strip_is_built_from_the_same_days_as_the_counts():
    """A strip that disagrees with the number beside it is worse than no
    strip."""
    store = _store(0, 1, 2)
    strip = streaks.recent_days(store, TODAY, span=STREAKS.calendar_days)
    assert sum(1 for _, a in strip if a) == streaks.summarise(store, TODAY).current


# --- the sentence -------------------------------------------------------------

def test_the_sentence_uses_the_singular_for_one_day():
    assert "1 day in a row" in streaks.sentence(streaks.summarise(_store(0), TODAY))


def test_the_sentence_says_when_today_is_not_yet_recorded():
    text = streaks.sentence(streaks.summarise(_store(1, 2), TODAY))
    assert "continues only if you do something" in text


def test_the_sentence_does_not_nag_when_today_is_already_recorded():
    text = streaks.sentence(streaks.summarise(_store(0, 1), TODAY))
    assert "continues only if" not in text


def test_a_broken_run_reports_the_last_active_day_rather_than_a_stale_count():
    text = streaks.sentence(streaks.summarise(_store(5, 6, 7), TODAY))
    assert "No current streak" in text
    assert _d(5).isoformat() in text
    assert "Longest run so far: 3 days" in text


def test_the_empty_sentence_is_the_empty_state():
    assert streaks.sentence(streaks.summarise(streaks.StreakStore(), TODAY)) \
        == streaks.NO_ACTIVITY


# --- persistence --------------------------------------------------------------

def test_days_survive_a_round_trip(tmp_path):
    path = tmp_path / "s.json"
    streaks.save_store(_store(0, 1, 2), path)
    assert streaks.summarise(streaks.load_store(path), TODAY).current == 3


def test_a_missing_store_is_empty_not_corrupt(tmp_path):
    store = streaks.load_store(tmp_path / "nope.json")
    assert store.days == () and store.corrupt is False


def test_an_unreadable_store_is_corrupt_not_empty(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert streaks.load_store(path).corrupt is True


def test_a_corrupt_store_is_never_written_over(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert streaks.save_store(streaks.StreakStore(corrupt=True), path) is False
    assert path.read_text() == "{not json"


def test_junk_days_are_dropped_on_load(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"days": ["2026-09-23", "nonsense", "", 17,
                                         "2026-09-22", "2026-09-23"]}))
    store = streaks.load_store(path)
    assert store.days == ("2026-09-22", "2026-09-23")
    assert store.corrupt is False


def test_the_store_is_per_user(monkeypatch):
    """Resolved with a namespace active, not read off the source."""
    import local_store
    monkeypatch.setattr(local_store, "_namespace_provider", lambda: "somekey",
                        raising=False)
    assert local_store.current_namespace() == "somekey"
    resolved = streaks._path()
    assert resolved != local_store.app_dir() / STREAKS.store_filename
    assert resolved.parent != local_store.app_dir()


def test_the_store_is_declared_per_user_in_auth():
    import auth
    assert STREAKS.store_filename in auth.PER_USER_STORES
    assert STREAKS.store_filename not in auth.SHARED_STORES


def test_the_store_is_gitignored():
    ignored = (Path(streaks.__file__).resolve().parent / ".gitignore").read_text()
    assert STREAKS.store_filename in ignored


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Streak ---")
    end = src.index("# --- Monthly contest ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def _wired_actions() -> set:
    """The actions finance.py actually records, excluding the helper's
    own definition — counting that line is how an earlier version of
    this test passed a build with a hook removed."""
    import re
    src = FINANCE.read_text()
    return set(re.findall(r'_streak_record\(\s*"([a-z_]+)"', src))


def test_every_declared_action_a_user_can_reach_is_actually_wired():
    """A count alone is too weak: with the helper's own `def` line in
    the tally, deleting a hook still cleared a floor of three. Name the
    actions instead — a streak nobody can earn is not a streak."""
    wired = _wired_actions()
    for action in ("journal", "screen", "watchlist", "contest"):
        assert action in wired, f"nothing in finance.py records a '{action}' day"


def test_the_helper_definition_is_not_counted_as_a_hook():
    src = FINANCE.read_text()
    assert "def _streak_record(" in src
    assert "_streak_record" not in _wired_actions()


def test_every_recorded_action_is_a_declared_one():
    """A caller passing 'render' would reintroduce the login streak this
    module exists to avoid."""
    import re
    src = FINANCE.read_text()
    used = set(re.findall(r'_streak_record\(\s*"([a-z_]+)"', src))
    assert used, "no streak actions are recorded anywhere"
    assert used <= set(streaks.ACTION_KEYS), f"undeclared actions: {used}"


def test_the_panel_shows_the_run_and_its_explanation():
    panel = _panel()
    assert "streaks.summarise(" in panel
    assert "streaks.sentence(" in panel


def test_the_panel_saves_only_when_the_day_changed():
    src = FINANCE.read_text()
    helper = src[src.index("def _streak_record("):]
    helper = helper[:helper.index("\n\n\n")] if "\n\n\n" in helper else helper[:2000]
    assert "if _changed" in helper, "an unchanged day must not rewrite the file"
