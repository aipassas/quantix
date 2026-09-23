"""The named monthly leaderboard: a second consent, a floor, and a
metric chosen by measurement.

THE TEST THAT MATTERS MOST is test_the_peer_comparison_opt_in_does_not_
enrol_anyone. peer_comparison's opt-in buys an ANONYMOUS percentile and
its whole k-anonymity floor exists so that no individual return is ever
revealed. This ladder reveals every participant's return beside a name.
Reusing that consent would have been the cheap build and would have
silently upgraded what people agreed to.

The second is test_there_is_no_sharpe_ladder. Simulated 2026-09-23 over
20,000 runs, a one-month Sharpe of a portfolio whose TRUE Sharpe is 1.00
has a standard deviation of 3.68 and ranks two portfolios a full 1.0
apart correctly 57.7% of the time. Ranking people's money on that is the
fabricated-intelligence failure recommendations.py exists to refuse.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import leaderboard as lb
from config import LEADERBOARD, PEER_COMPARISON

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
PERIOD = "2026-09"


class _Profile:
    def __init__(self, user_key, name):
        self.user_key = user_key
        self.name = name


class _Profiles:
    def __init__(self, *profiles):
        self._by_key = {p.user_key: p for p in profiles}

    def get(self, user_key):
        return self._by_key.get(user_key)


class _Entry:
    def __init__(self, user_key, twr_pct, period=PERIOD):
        self.user_key = user_key
        self.twr_pct = twr_pct
        self.period = period


def _store(*keys):
    return lb.LeaderboardStore(tuple(lb.Participant(k, "2026-09-01T00:00:00")
                                     for k in keys))


def _cohort(n=5, base=1.0):
    """n participants, all joined, all sharing, all named — the minimum
    arrangement in which a ladder is drawn at all."""
    keys = [f"k{i}" for i in range(n)]
    entries = [_Entry(k, base + i) for i, k in enumerate(keys)]
    profiles = _Profiles(*[_Profile(k, f"Person {i}") for i, k in enumerate(keys)])
    return _store(*keys), entries, profiles


# --- the metric ---------------------------------------------------------------

def test_there_is_no_sharpe_ladder():
    """The ticket offers Sharpe. A monthly Sharpe cannot rank anything,
    and the module must say so rather than quietly omitting it."""
    src = Path(lb.__file__).read_text()
    assert "compute_sharpe_ratio" not in src, "nothing here computes one"
    assert not any(name for name in dir(lb)
                   if "sharpe" in name.lower() and callable(getattr(lb, name))), \
        "no callable ranks on a Sharpe ratio"
    # Every Row is a return, so there is no second metric to rank on.
    import dataclasses
    assert [f.name for f in dataclasses.fields(lb.Row)] == [
        "rank", "name", "twr_pct", "is_you"]


def test_the_refusal_carries_the_measurement_that_justifies_it():
    """A refusal without its evidence is an opinion. These figures are
    the simulation in the module docstring; if someone changes the
    prose without re-measuring, this fails."""
    for figure in ("3.68", "-4.91", "+7.17", "57.7%", "1.00"):
        assert figure in lb.SHARPE_UNAVAILABLE, figure
    assert "noise" in lb.SHARPE_UNAVAILABLE


def test_the_refusal_names_the_window_it_applies_to():
    """Sharpe is not useless everywhere — it is useless over a month.
    A blanket 'Sharpe is unreliable' would be wrong."""
    assert "single calendar month" in lb.SHARPE_UNAVAILABLE


# --- consent ------------------------------------------------------------------

def test_the_peer_comparison_opt_in_does_not_enrol_anyone():
    """Sharing an ANONYMOUS return must not put someone's NAME on a
    public ladder. The two stores are separate precisely so this cannot
    happen by accident."""
    entries = [_Entry(f"k{i}", 5.0) for i in range(8)]
    profiles = _Profiles(*[_Profile(f"k{i}", f"P{i}") for i in range(8)])
    result = lb.standings(entries, lb.LeaderboardStore(), profiles, PERIOD)
    assert result.rows == ()
    assert result.participants == 0


def test_joining_takes_a_key_and_nothing_else():
    """The signature IS the privacy boundary — there must be no
    parameter through which a return, a name or a portfolio arrives.
    Same assertion peer_comparison.publish carries."""
    import inspect
    params = list(inspect.signature(lb.join).parameters)
    assert params == ["store", "user_key"]


def test_a_participant_record_holds_only_consent():
    """Two fields. A return or a name here would be a second copy that
    could fall out of step with the one the ladder actually reads."""
    import dataclasses
    fields = [f.name for f in dataclasses.fields(lb.Participant)]
    assert fields == ["user_key", "joined_at"]


def test_joining_is_idempotent():
    store, _ = lb.join(_store("k1"), "k1")
    assert len(store.participants) == 1


def test_a_signed_out_reader_cannot_join():
    store, err = lb.join(lb.LeaderboardStore(), "")
    assert store.participants == () and "Sign in" in err


def test_leaving_removes_the_name_and_says_nothing_about_the_return():
    """Leaving is a narrower act than withdrawing from the comparison —
    it must not reach into peer_comparison's store."""
    # The real property is that this module has no CODE dependency on
    # peer_comparison at all — the returns arrive as a parameter — so it
    # has no route by which it could delete one. Checked as an import,
    # not as a substring: the docstring names the module repeatedly.
    src = Path(lb.__file__).read_text()
    imports = [ln for ln in src.splitlines()
               if ln.startswith(("import ", "from ")) and "peer_comparison" in ln]
    assert imports == [], f"leaderboard must not import peer_comparison: {imports}"
    store = lb.leave(_store("k1", "k2"), "k1")
    assert [p.user_key for p in store.participants] == ["k2"]


def test_leaving_an_account_that_never_joined_changes_nothing():
    before = _store("k1")
    assert lb.leave(before, "k9") == before


# --- the three conditions -----------------------------------------------------

def test_a_participant_without_a_shared_return_is_not_listed():
    store, entries, profiles = _cohort(5)
    store = lb.LeaderboardStore(store.participants + (lb.Participant("k9", "t"),))
    profiles_plus = _Profiles(*[_Profile(f"k{i}", f"Person {i}") for i in range(5)],
                              _Profile("k9", "Silent"))
    listed = lb.listed(entries, store, profiles_plus, PERIOD)
    assert "k9" not in [k for k, _, _ in listed]


def test_a_participant_without_a_profile_name_is_not_listed():
    """Nobody is ever named who did not choose the name."""
    store, entries, profiles = _cohort(5)
    store = lb.LeaderboardStore(store.participants + (lb.Participant("k9", "t"),))
    entries = entries + [_Entry("k9", 99.0)]
    listed = lb.listed(entries, store, profiles, PERIOD)
    assert "k9" not in [k for k, _, _ in listed], "a nameless account was listed"


def test_a_blank_profile_name_does_not_count_as_a_name():
    store, entries, profiles = _cohort(5)
    store = lb.LeaderboardStore(store.participants + (lb.Participant("k9", "t"),))
    entries = entries + [_Entry("k9", 99.0)]
    named = _Profiles(*[_Profile(f"k{i}", f"Person {i}") for i in range(5)],
                      _Profile("k9", "   "))
    assert "k9" not in [k for k, _, _ in lb.listed(entries, store, named, PERIOD)]


def test_deleting_a_profile_drops_the_row_on_the_next_read():
    """Re-checked on every render rather than trusted from join time —
    the same discipline following.feed_for uses."""
    store, entries, profiles = _cohort(5)
    assert len(lb.listed(entries, store, profiles, PERIOD)) == 5
    gone = _Profiles(*[_Profile(f"k{i}", f"Person {i}") for i in range(4)])
    assert len(lb.listed(entries, store, gone, PERIOD)) == 4


def test_withdrawing_the_shared_return_drops_the_row_too():
    """There is no second copy of the return here, so peer_comparison's
    withdraw removes someone from this ladder automatically."""
    store, entries, profiles = _cohort(5)
    remaining = [e for e in entries if e.user_key != "k0"]
    assert "k0" not in [k for k, _, _ in lb.listed(remaining, store, profiles, PERIOD)]


def test_another_months_entry_does_not_count_toward_this_month():
    store, entries, profiles = _cohort(5)
    stale = [_Entry(e.user_key, e.twr_pct, "2026-08") for e in entries]
    assert lb.listed(stale, store, profiles, PERIOD) == []


# --- the floor ----------------------------------------------------------------

def test_the_floor_is_peer_comparisons_own_number():
    """Deliberately the same 5, not a second number that can drift."""
    assert LEADERBOARD.min_participants == PEER_COMPARISON.min_cohort


def test_below_the_floor_no_ladder_is_drawn():
    store, entries, profiles = _cohort(LEADERBOARD.min_participants - 1)
    result = lb.standings(entries, store, profiles, PERIOD, "k0")
    assert result.rows == ()
    assert result.participants == LEADERBOARD.min_participants - 1
    assert str(LEADERBOARD.min_participants) in result.reason
    assert "disclosure" in result.reason


def test_at_the_floor_the_ladder_is_drawn():
    store, entries, profiles = _cohort(LEADERBOARD.min_participants)
    result = lb.standings(entries, store, profiles, PERIOD, "k0")
    assert result.has_result
    assert len(result.rows) == LEADERBOARD.min_participants


def test_the_refusal_reports_the_count_rather_than_a_ranking():
    store, entries, profiles = _cohort(2)
    result = lb.standings(entries, store, profiles, PERIOD, "k0")
    assert "2 account(s)" in result.reason
    assert result.your_rank is None


# --- the ranking --------------------------------------------------------------

def test_the_highest_return_ranks_first():
    store, entries, profiles = _cohort(5)          # k4 has the highest
    result = lb.standings(entries, store, profiles, PERIOD, "k0")
    assert result.rows[0].rank == 1
    assert result.rows[0].name == "Person 4"
    assert result.rows[0].twr_pct == 5.0


def test_a_negative_return_still_ranks():
    """A losing month is a real result, not missing data."""
    keys = [f"k{i}" for i in range(5)]
    entries = [_Entry(k, -float(i)) for i, k in enumerate(keys)]
    profiles = _Profiles(*[_Profile(k, f"P{i}") for i, k in enumerate(keys)])
    result = lb.standings(entries, _store(*keys), profiles, PERIOD)
    assert [r.twr_pct for r in result.rows] == [0.0, -1.0, -2.0, -3.0, -4.0]


def test_ties_share_a_rank_and_the_next_row_skips():
    """Standard competition ranking. Breaking the tie on anything else
    would assert a difference the returns do not contain."""
    keys = ["a", "b", "c", "d", "e"]
    values = [10.0, 5.0, 5.0, 5.0, 1.0]
    entries = [_Entry(k, v) for k, v in zip(keys, values)]
    profiles = _Profiles(*[_Profile(k, k.upper()) for k in keys])
    result = lb.standings(entries, _store(*keys), profiles, PERIOD)
    assert [r.rank for r in result.rows] == [1, 2, 2, 2, 5]


def test_a_whole_field_of_ties_is_all_rank_one():
    keys = [f"k{i}" for i in range(5)]
    entries = [_Entry(k, 3.0) for k in keys]
    profiles = _Profiles(*[_Profile(k, f"P{k}") for k in keys])
    result = lb.standings(entries, _store(*keys), profiles, PERIOD)
    assert {r.rank for r in result.rows} == {1}


def test_the_order_is_stable_whatever_order_people_joined_in():
    """Two readers loading the same month must see the same ladder.

    The rows are built by iterating the STORE, so reversing the entries
    list varies nothing — an earlier version of this test did exactly
    that and passed a build whose sort had no tiebreaker at all. Vary
    the join order, which is what actually feeds the sort.
    """
    keys = ["z", "a", "m", "q", "b"]
    entries = [_Entry(k, 4.0) for k in keys]          # a total tie
    profiles = _Profiles(*[_Profile(k, k.upper()) for k in keys])
    first = lb.standings(entries, _store(*keys), profiles, PERIOD)
    second = lb.standings(entries, _store(*reversed(keys)), profiles, PERIOD)
    assert [r.name for r in first.rows] == [r.name for r in second.rows]
    assert [r.name for r in first.rows] == ["A", "B", "M", "Q", "Z"]


def test_a_partial_tie_is_also_ordered_deterministically():
    keys = ["z", "a", "m"]
    entries = [_Entry("z", 9.0), _Entry("a", 4.0), _Entry("m", 4.0)]
    profiles = _Profiles(*[_Profile(k, k.upper()) for k in keys])
    first = lb.standings(entries, _store(*keys), profiles, PERIOD, max_rows=99)
    second = lb.standings(entries, _store(*reversed(keys)), profiles, PERIOD,
                          max_rows=99)
    assert [(r.rank, r.name) for r in first.rows] == [(r.rank, r.name)
                                                      for r in second.rows]


# --- the reader's own row -----------------------------------------------------

def test_your_own_row_is_marked():
    store, entries, profiles = _cohort(5)
    result = lb.standings(entries, store, profiles, PERIOD, "k2")
    yours = [r for r in result.rows if r.is_you]
    assert len(yours) == 1 and yours[0].name == "Person 2"
    assert result.your_rank == 3


def test_a_reader_outside_the_visible_top_still_sees_their_own_row():
    """A ladder that never shows you where you stand is a scoreboard for
    other people."""
    store, entries, profiles = _cohort(12)
    result = lb.standings(entries, store, profiles, PERIOD, "k0", max_rows=3)
    assert len(result.rows) == 4, "three visible rows plus the reader's own"
    assert result.rows[-1].is_you and result.rows[-1].rank == 12
    assert result.your_rank == 12


def test_the_readers_row_is_not_duplicated_when_already_visible():
    store, entries, profiles = _cohort(12)
    result = lb.standings(entries, store, profiles, PERIOD, "k11", max_rows=3)
    assert sum(1 for r in result.rows if r.is_you) == 1
    assert len(result.rows) == 3


def test_a_reader_who_is_not_participating_gets_no_rank():
    store, entries, profiles = _cohort(5)
    result = lb.standings(entries, store, profiles, PERIOD, "stranger")
    assert result.has_result
    assert result.your_rank is None
    assert not any(r.is_you for r in result.rows)


def test_a_row_carries_no_account_key():
    """The key is a hash of an identity; it has no business on screen."""
    import dataclasses
    fields = [f.name for f in dataclasses.fields(lb.Row)]
    assert fields == ["rank", "name", "twr_pct", "is_you"]


# --- the caption --------------------------------------------------------------

def test_the_caption_counts_through_the_same_filter_as_the_ladder():
    """A caption that disagrees with the table under it is worse than no
    caption.

    The cohort must contain someone who JOINED but does not qualify,
    or len(store.participants) and len(listed(...)) agree by accident
    and the test proves nothing — which is exactly what an earlier
    version did, passing a build that counted raw participants.
    """
    store, entries, profiles = _cohort(7)
    # Two extra joiners: one sharing no return, one with no profile name.
    store = lb.LeaderboardStore(store.participants
                                + (lb.Participant("silent", "t"),
                                   lb.Participant("nameless", "t")))
    entries = entries + [_Entry("nameless", 50.0)]
    assert len(store.participants) == 9, "the raw count must differ from the real one"

    note = lb.participation_note(store, entries, profiles, PERIOD)
    result = lb.standings(entries, store, profiles, PERIOD)
    assert result.participants == 7
    assert "7 accounts" in note
    # Not a bare "9": the period label "2026-09" contains one.
    assert "9 accounts" not in note, "the caption counted joiners, not the ladder"


def test_an_empty_instance_says_so_rather_than_showing_a_blank_table():
    note = lb.participation_note(lb.LeaderboardStore(), [], _Profiles(), PERIOD)
    assert "Nobody" in note and PERIOD in note


def test_the_caption_does_not_say_one_accounts():
    store, entries, profiles = _cohort(1)
    assert "1 account is" in lb.participation_note(store, entries, profiles, PERIOD)


# --- persistence --------------------------------------------------------------

def test_consent_survives_a_round_trip(tmp_path):
    path = tmp_path / "l.json"
    lb.save_store(_store("k1", "k2"), path)
    loaded = lb.load_store(path)
    assert [p.user_key for p in loaded.participants] == ["k1", "k2"]
    assert loaded.corrupt is False


def test_the_persisted_record_holds_no_return_and_no_name(tmp_path):
    path = tmp_path / "l.json"
    lb.save_store(_store("k1"), path)
    raw = json.loads(path.read_text())
    assert set(raw["participants"][0]) == {"user_key", "joined_at"}


def test_a_missing_store_is_empty_not_corrupt(tmp_path):
    store = lb.load_store(tmp_path / "nope.json")
    assert store.participants == () and store.corrupt is False


def test_an_unreadable_store_is_corrupt_not_empty(tmp_path):
    """Treating it as empty would let the next join drop everyone else's
    consent record."""
    path = tmp_path / "l.json"
    path.write_text("{not json")
    assert lb.load_store(path).corrupt is True


def test_a_corrupt_store_is_never_written_over(tmp_path):
    path = tmp_path / "l.json"
    path.write_text("{not json")
    assert lb.save_store(lb.LeaderboardStore(corrupt=True), path) is False
    assert path.read_text() == "{not json"


def test_a_duplicated_key_on_disk_is_collapsed(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"participants": [
        {"user_key": "k1", "joined_at": "a"}, {"user_key": "k1", "joined_at": "b"},
        {"user_key": ""}, "nonsense",
    ]}))
    store = lb.load_store(path)
    assert [p.user_key for p in store.participants] == ["k1"]
    assert store.corrupt is False


def test_the_store_is_shared_not_per_user(monkeypatch):
    """Resolved with a namespace ACTIVE, not read off the source.

    A textual check passes a build that merely imports store_path
    without calling it — proven by a poison. This signs a namespace in
    and asserts the file still lands in the app directory, because a
    ranking across accounts cannot live inside one account's folder.
    """
    import local_store
    monkeypatch.setattr(local_store, "_namespace_provider", lambda: "someones-key",
                        raising=False)
    assert local_store.current_namespace() == "someones-key"
    assert local_store.store_path("x.json").parent != local_store.app_dir(), \
        "the namespace must really be active, or this test proves nothing"

    resolved = lb._store_path()
    assert resolved == local_store.app_dir() / LEADERBOARD.store_filename
    assert "users" not in resolved.parts


def test_the_store_is_declared_shared_in_auth():
    import auth
    assert LEADERBOARD.store_filename in auth.SHARED_STORES
    assert LEADERBOARD.store_filename not in auth.PER_USER_STORES


def test_the_store_is_gitignored():
    """Enforced by nothing else — five stores were sitting committable
    before this was checked."""
    ignored = (Path(lb.__file__).resolve().parent / ".gitignore").read_text()
    assert LEADERBOARD.store_filename in ignored


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Leaderboard ---")
    end = src.index("if _pf_perf.mwr_pct is not None:", start)
    # Comments stripped: this block's own commentary names what it
    # avoids, so a substring check would match the explanation.
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_states_why_there_is_no_sharpe_ladder():
    assert "LEADERBOARD_SHARPE_UNAVAILABLE" in _panel()


def test_joining_is_its_own_control_not_the_comparison_switch():
    panel = _panel()
    assert "lb_join" in panel and "lb_leave" in panel
    assert "peer_publish" not in panel, "the ladder must not reuse the anonymous opt-in"


def test_the_panel_warns_what_joining_discloses_before_it_happens():
    panel = _panel()
    join_at = panel.index('key="lb_join"')
    assert "LEADERBOARD_NOT_PARTICIPATING" in panel[:join_at]


def test_the_panel_requires_a_profile_name():
    assert "LEADERBOARD_NEEDS_PROFILE" in _panel()


def test_the_panel_refuses_to_write_a_corrupt_store():
    panel = _panel()
    assert "_lb_store.corrupt" in panel


def test_the_panel_does_not_read_a_conditionally_assigned_name():
    """finance.py is a script, so name order is execution order.
    `_peer_in` is assigned inside the comparison expander's else-branch,
    so a reader whose peer store is corrupt would hit a NameError here —
    a crash that fires only for the people whose data is already broken.
    The panel asks peer_comparison directly instead."""
    panel = _panel()
    assert "_peer_in" not in panel
    assert "peer_comparison.is_participating(" in panel


def test_the_panel_renders_the_computed_standings_not_its_own_sort():
    panel = _panel()
    assert "lb_standings(" in panel
    assert "sorted(" not in panel, "the ranking belongs in the module"
