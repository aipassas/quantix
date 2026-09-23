"""The monthly contest: entered blind, scored as a binary, ranked over
many months.

THE TEST THAT MATTERS MOST is test_a_pick_cannot_be_made_inside_the_month
_it_predicts. A pick made on the 20th has already seen twenty days of the
return it is "predicting", and scoring from the entry date instead would
give every entrant a different window — the mismatched-span problem
peer_comparison was built to fix.

The second is test_the_ladder_does_not_rank_the_months_return. Measured
over 30 large caps and 60 real months, the month's best performer sits at
a median volatility rank of 6 of 30, 58% of months are won by a
top-quartile-volatility name against 25% by chance, and winning does not
repeat. A highest-return contest ranks volatility.
"""
import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import monthly_contest as mc
from config import CONTEST

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"

SEPT = datetime.date(2026, 9, 15)      # entries open for 2026-10
OCT = "2026-10"


class _Profile:
    def __init__(self, user_key, name):
        self.user_key = user_key
        self.name = name


class _Profiles:
    def __init__(self, *profiles):
        self._by_key = {p.user_key: p for p in profiles}

    def get(self, key):
        return self._by_key.get(key)


def _series(start: str, values):
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index)


def _entry(key="k1", period=OCT, ticker="AAPL", thesis="services margin"):
    return mc.Entry(key, period, ticker, thesis, "2026-09-15T00:00:00")


def _result(key, beat, period="2026-01"):
    return mc.Result(_entry(key, period), pick_pct=1.0, benchmark_pct=0.0,
                     beat=beat)


# --- the entry window ---------------------------------------------------------

def test_entries_are_open_for_the_next_month_not_this_one():
    assert mc.open_period(SEPT) == "2026-10"
    assert mc.open_period(datetime.date(2026, 9, 1)) == "2026-10"
    assert mc.open_period(datetime.date(2026, 9, 30)) == "2026-10"


def test_the_window_rolls_over_the_year_end():
    assert mc.open_period(datetime.date(2026, 12, 14)) == "2027-01"


def test_entries_close_the_day_before_the_month_starts():
    assert mc.entries_close("2026-10") == datetime.date(2026, 9, 30)


def test_a_pick_cannot_be_made_inside_the_month_it_predicts():
    """The anti-hindsight guarantee. On 15 October, October is closed."""
    store, err = mc.enter(mc.ContestStore(), "k1", "2026-10", "AAPL", "why",
                          day=datetime.date(2026, 10, 15))
    assert store.entries == ()
    assert "closed" in err and "2026-09-30" in err
    assert "already seen part of the answer" in err


def test_a_pick_cannot_be_made_for_a_month_two_ahead_either():
    """Otherwise someone could fill the whole year in advance and the
    'open period' would mean nothing."""
    _, err = mc.enter(mc.ContestStore(), "k1", "2026-12", "AAPL", "why", day=SEPT)
    assert err and "closed" in err


def test_a_past_month_cannot_be_entered():
    _, err = mc.enter(mc.ContestStore(), "k1", "2026-01", "AAPL", "why", day=SEPT)
    assert err and "closed" in err


def test_the_scored_month_is_the_one_in_progress():
    assert mc.scored_period(SEPT) == "2026-09"


# --- entering -----------------------------------------------------------------

def test_a_valid_entry_is_recorded():
    store, err = mc.enter(mc.ContestStore(), "k1", OCT, "aapl", "services margin",
                          day=SEPT)
    assert err == ""
    entry = store.entry_for("k1", OCT)
    assert entry.ticker == "AAPL", "the ticker is normalised"
    assert entry.thesis == "services margin"
    assert entry.entered_at


def test_an_entry_needs_a_thesis():
    _, err = mc.enter(mc.ContestStore(), "k1", OCT, "AAPL", "   ", day=SEPT)
    assert err == mc.NEEDS_THESIS
    assert "teaches nobody anything" in err


def test_an_entry_needs_a_ticker():
    _, err = mc.enter(mc.ContestStore(), "k1", OCT, "  ", "why", day=SEPT)
    assert err == mc.NEEDS_TICKER


def test_a_signed_out_reader_cannot_enter():
    _, err = mc.enter(mc.ContestStore(), "", OCT, "AAPL", "why", day=SEPT)
    assert err == mc.NOT_SIGNED_IN


def test_changing_your_pick_while_the_window_is_open_replaces_it():
    store, _ = mc.enter(mc.ContestStore(), "k1", OCT, "AAPL", "first", day=SEPT)
    store, err = mc.enter(store, "k1", OCT, "MSFT", "second", day=SEPT)
    assert err == ""
    assert len(store.for_period(OCT)) == 1
    assert store.entry_for("k1", OCT).ticker == "MSFT"


def test_a_long_thesis_is_truncated_not_refused():
    store, err = mc.enter(mc.ContestStore(), "k1", OCT, "AAPL", "x" * 5000, day=SEPT)
    assert err == ""
    assert len(store.entry_for("k1", OCT).thesis) == CONTEST.max_thesis_chars


def test_two_accounts_can_pick_the_same_stock():
    store, _ = mc.enter(mc.ContestStore(), "k1", OCT, "AAPL", "a", day=SEPT)
    store, err = mc.enter(store, "k2", OCT, "AAPL", "b", day=SEPT)
    assert err == "" and len(store.for_period(OCT)) == 2


def test_an_entry_carries_no_position_size_or_target():
    """None of those is scored, and each would read as advice."""
    import dataclasses
    assert [f.name for f in dataclasses.fields(mc.Entry)] == [
        "user_key", "period", "ticker", "thesis", "entered_at"]


# --- withdrawing --------------------------------------------------------------

def test_an_open_month_can_be_withdrawn_from():
    store, _ = mc.enter(mc.ContestStore(), "k1", OCT, "AAPL", "why", day=SEPT)
    store, err = mc.withdraw(store, "k1", OCT, day=SEPT)
    assert err == "" and store.entries == ()


def test_a_month_that_has_begun_cannot_be_withdrawn_from():
    """Deleting the months that went badly would make every hit rate on
    the ladder meaningless."""
    store = mc.ContestStore((_entry("k1", "2026-09"),))
    after, err = mc.withdraw(store, "k1", "2026-09", day=SEPT)
    assert after == store
    assert "meaningless" in err


# --- scoring ------------------------------------------------------------------

def test_a_pick_that_beats_the_benchmark_is_a_hit():
    entry = _entry(period="2026-01")
    pick = _series("2026-01-02", [100.0, 110.0])
    bench = _series("2026-01-02", [100.0, 102.0])
    result = mc.measure(entry, pick, bench)
    assert result.measured and result.beat is True
    assert result.pick_pct == pytest.approx(10.0)
    assert result.benchmark_pct == pytest.approx(2.0)


def test_a_pick_that_loses_to_the_benchmark_is_a_miss():
    entry = _entry(period="2026-01")
    result = mc.measure(entry, _series("2026-01-02", [100.0, 101.0]),
                        _series("2026-01-02", [100.0, 105.0]))
    assert result.beat is False


def test_a_rising_pick_can_still_be_a_miss():
    """The contest scores RELATIVE performance — a stock that went up
    less than the index did not beat it."""
    entry = _entry(period="2026-01")
    result = mc.measure(entry, _series("2026-01-02", [100.0, 103.0]),
                        _series("2026-01-02", [100.0, 108.0]))
    assert result.pick_pct > 0 and result.beat is False


def test_a_falling_pick_can_still_be_a_hit():
    entry = _entry(period="2026-01")
    result = mc.measure(entry, _series("2026-01-02", [100.0, 97.0]),
                        _series("2026-01-02", [100.0, 90.0]))
    assert result.pick_pct < 0 and result.beat is True


def test_prices_outside_the_month_are_not_counted():
    """Scoring a month partly on days belonging to another one would put
    a different window under each entrant."""
    entry = _entry(period="2026-01")
    # Twenty December bars at 50, then January's bars. The series must
    # really cross the boundary or this proves nothing — an earlier
    # version started in December and ENDED there, so January was empty
    # and the test passed against a build with no window filter at all.
    pick = _series("2025-12-04", [50.0] * 20 + [100.0, 101.0])
    bench = _series("2025-12-04", [50.0] * 20 + [100.0, 100.5])
    assert any(d.year == 2026 for d in pick.index), "the fixture must cross into January"
    assert any(d.year == 2025 for d in pick.index), "and start before it"
    result = mc.measure(entry, pick, bench)
    assert result.pick_pct == pytest.approx(1.0), "December must not leak in"


def test_an_unmeasurable_month_is_not_scored_as_a_loss():
    """A data gap is not a wrong call. False here would silently drag a
    record down for a missing price series."""
    result = mc.measure(_entry(period="2026-01"), None, None)
    assert result.beat is None and result.measured is False
    assert "Not enough price history" in result.reason


def test_a_single_bar_is_not_enough_to_score():
    entry = _entry(period="2026-01")
    result = mc.measure(entry, _series("2026-01-02", [100.0]),
                        _series("2026-01-02", [100.0, 101.0]))
    assert result.beat is None


# --- significance -------------------------------------------------------------

def test_the_null_is_the_measured_base_rate_not_fifty_percent():
    """50.7% of stock-months beat SPY, measured over 3,435 stock-months."""
    assert CONTEST.null_hit_rate == 0.507


def test_three_months_can_never_be_significant():
    """Not an edge case to work around — it is the correct answer. Even
    three for three says nothing."""
    assert mc.is_significant(3, 3) is False
    for hits in range(4):
        assert mc.is_significant(hits, 3) is False


def test_the_measured_thresholds_hold():
    """The exact figures the config and the UI quote. If the null or the
    alpha moves, these move with it and the prose must be re-measured."""
    assert mc.is_significant(6, 6) is True
    assert mc.is_significant(5, 6) is False
    assert mc.is_significant(10, 12) is True
    assert mc.is_significant(9, 12) is False
    assert mc.is_significant(17, 24) is True
    assert mc.is_significant(16, 24) is False


def test_a_losing_record_is_never_significant():
    assert mc.is_significant(0, 24) is False
    assert mc.is_significant(5, 24) is False


def test_an_impossible_record_is_refused_rather_than_crashing():
    assert mc.is_significant(5, 3) is False
    assert mc.is_significant(1, 0) is False


def test_the_short_record_note_carries_the_measurement():
    note = mc.short_record_note()
    assert "50.7%" in note
    assert "6 of 6" in note and "10 of 12" in note and "17 of 24" in note


# --- the ladder ---------------------------------------------------------------

def test_the_ladder_does_not_rank_the_months_return():
    """The whole design decision. Nothing here orders anyone by how much
    their pick made."""
    src = Path(mc.__file__).read_text()
    assert "WHY_NOT_RANKED_BY_RETURN" in src
    import dataclasses
    fields = [f.name for f in dataclasses.fields(mc.Record)]
    assert "return_pct" not in fields and "pick_pct" not in fields
    assert fields == ["user_key", "name", "months", "hits", "entered", "significant"]


def test_the_refusal_carries_the_measurement_that_justifies_it():
    for figure in ("6 out of 30", "58%", "25%", "56%"):
        assert figure in mc.WHY_NOT_RANKED_BY_RETURN, figure


def test_a_record_counts_only_measured_months():
    """An entry whose prices could not be read is carried separately, not
    counted as a loss."""
    results = [_result("k1", True), _result("k1", False),
               mc.Result(_entry("k1", "2026-03"))]
    (record,) = mc.records(results, _Profiles(_Profile("k1", "Ana")))
    assert record.months == 2 and record.hits == 1 and record.entered == 3
    assert record.hit_rate == pytest.approx(50.0)


def test_an_entrant_without_a_profile_name_is_not_listed():
    assert mc.records([_result("k1", True)], _Profiles()) == ()


def test_a_significant_record_outranks_a_higher_but_short_one():
    """A 2-of-2 must not outrank a 17-of-24. Ranking purely on the rate
    would crown whoever had the shortest record."""
    results = ([_result("short", True, f"2026-{m:02d}") for m in (1, 2)]
               + [_result("long", i < 17, f"2025-{m:02d}")
                  for i, m in enumerate(range(1, 13))]
               + [_result("long", True, f"2026-{m:02d}") for m in range(1, 13)])
    ranked = mc.records(results, _Profiles(_Profile("short", "Shorty"),
                                           _Profile("long", "Longy")))
    assert ranked[0].name == "Longy"
    assert ranked[0].significant and not ranked[1].significant


def test_among_equally_short_records_the_longer_one_ranks_higher():
    results = ([_result("a", True, f"2026-{m:02d}") for m in (1, 2)]
               + [_result("b", True, f"2026-{m:02d}") for m in (1, 2, 3, 4)])
    ranked = mc.records(results, _Profiles(_Profile("a", "A"), _Profile("b", "B")))
    assert [r.name for r in ranked] == ["B", "A"]


def test_the_order_is_stable_between_reads():
    results = [_result(k, True, "2026-01") for k in ("z", "a", "m")]
    profiles = _Profiles(*[_Profile(k, k.upper()) for k in ("z", "a", "m")])
    first = mc.records(results, profiles)
    second = mc.records(list(reversed(results)), profiles)
    assert [r.name for r in first] == [r.name for r in second] == ["A", "M", "Z"]


def test_an_entrant_with_no_measured_month_has_no_hit_rate():
    results = [mc.Result(_entry("k1", "2026-03"))]
    (record,) = mc.records(results, _Profiles(_Profile("k1", "Ana")))
    assert record.hit_rate is None and record.months == 0
    assert record.significant is False


# --- persistence --------------------------------------------------------------

def test_entries_survive_a_round_trip(tmp_path):
    path = tmp_path / "c.json"
    store, _ = mc.enter(mc.ContestStore(), "k1", OCT, "AAPL", "why", day=SEPT)
    mc.save_store(store, path)
    loaded = mc.load_store(path)
    assert loaded.entry_for("k1", OCT).ticker == "AAPL"
    assert loaded.entry_for("k1", OCT).thesis == "why"


def test_an_unreadable_store_is_corrupt_not_empty(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert mc.load_store(path).corrupt is True


def test_a_corrupt_store_is_never_written_over(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert mc.save_store(mc.ContestStore(corrupt=True), path) is False
    assert path.read_text() == "{not json"


def test_a_junk_entry_is_dropped_rather_than_taking_the_store_down(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"entries": [
        {"user_key": "k1", "period": OCT, "ticker": "AAPL", "thesis": "t"},
        {"user_key": "k2", "period": OCT}, {"ticker": "X"}, "nonsense",
    ]}))
    store = mc.load_store(path)
    assert [e.user_key for e in store.entries] == ["k1"]
    assert store.corrupt is False


def test_the_store_is_shared_not_per_user(monkeypatch):
    import local_store
    monkeypatch.setattr(local_store, "_namespace_provider", lambda: "somekey",
                        raising=False)
    assert local_store.current_namespace() == "somekey"
    assert mc._store_path() == local_store.app_dir() / CONTEST.store_filename


def test_the_store_is_declared_shared_in_auth():
    import auth
    assert CONTEST.store_filename in auth.SHARED_STORES
    assert CONTEST.store_filename not in auth.PER_USER_STORES


def test_the_store_is_gitignored():
    ignored = (Path(mc.__file__).resolve().parent / ".gitignore").read_text()
    assert CONTEST.store_filename in ignored


def test_the_month_label_agrees_with_peer_comparison():
    """Two definitions of 'this month' is how two panels start
    disagreeing about which month it is."""
    import peer_comparison
    for day in (datetime.date(2026, 1, 1), datetime.date(2026, 6, 30),
                datetime.date(2026, 12, 31)):
        assert mc.scored_period(day) == peer_comparison.period_for(day)


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Monthly contest ---")
    end = src.index("# --- Leaderboard ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_states_why_the_month_is_not_ranked():
    assert "CONTEST_WHY_NOT_RANKED" in _panel()


def test_the_panel_offers_the_open_month_not_the_current_one():
    panel = _panel()
    assert "mc_open_period(" in panel
    assert "contest_enter" in panel


def test_the_panel_shows_the_short_record_caveat():
    assert "mc_short_record_note(" in _panel()


def test_the_panel_ranks_through_the_module():
    panel = _panel()
    assert "mc_records(" in panel
    assert "sorted(" not in panel, "the ranking belongs in the module"


def test_entering_the_contest_counts_toward_the_streak():
    panel = _panel()
    assert '_streak_record("contest")' in panel


def test_the_panel_refuses_to_write_a_corrupt_store():
    assert "_mc_store.corrupt" in _panel()
