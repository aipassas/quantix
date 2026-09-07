"""The journal: what you thought, and what happened.

The rules this file defends, in order of how badly breaking them would
hurt:

  - a corrupt journal must never be overwritten; it is the one store
    whose contents cannot be re-fetched from anywhere;
  - conviction cannot be edited once the outcome has been seen, or
    hindsight rewrites the record the feature exists to test;
  - a Sell that was followed by a fall went the writer's way, and
    reporting it as a loss would misread their own record;
  - outcomes are recomputed, never stored.
"""
import datetime
import json

import pandas as pd
import pytest

import investment_journal as ij


def _dates(n, start="2026-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def _prices(values, start="2026-01-01"):
    return pd.Series(values, index=_dates(len(values), start), dtype="float64")


def _entry(**kwargs):
    base = dict(ticker="AAPL", action=ij.BUY, reasoning="thesis",
                conviction="Medium",
                decided_on=datetime.date(2026, 1, 1))
    base.update(kwargs)
    store, error = ij.add_entry(ij.JournalStore(), **base)
    assert error is None, error
    return store.entries[0]


# --- writing ------------------------------------------------------------------

def test_an_entry_records_the_reasoning_and_the_conviction():
    store, error = ij.add_entry(
        ij.JournalStore(), "AAPL", ij.BUY, "Services margin expanding.",
        conviction="High", decided_on=datetime.date(2026, 3, 2))
    assert error is None
    entry = store.entries[0]
    assert entry.ticker == "AAPL" and entry.action == ij.BUY
    assert entry.conviction == "High"
    assert entry.reasoning.startswith("Services margin")
    assert entry.decided_on == "2026-03-02"


def test_reasoning_is_REQUIRED_because_it_is_the_point():
    """An entry without it records that you acted but not why, which is
    the half worth coming back to."""
    store, error = ij.add_entry(ij.JournalStore(), "AAPL", ij.BUY, "   ")
    assert error and "reasoning" in error.lower()
    assert store.entries == ()


def test_a_ticker_is_required_and_normalised():
    store, error = ij.add_entry(ij.JournalStore(), "  aapl ", ij.BUY, "why")
    assert error is None and store.entries[0].ticker == "AAPL"
    _, error = ij.add_entry(ij.JournalStore(), "", ij.BUY, "why")
    assert error


@pytest.mark.parametrize("action", ij.ACTIONS)
def test_every_declared_action_is_accepted(action):
    _, error = ij.add_entry(ij.JournalStore(), "AAPL", action, "why")
    assert error is None


def test_an_unknown_action_is_refused():
    _, error = ij.add_entry(ij.JournalStore(), "AAPL", "Yolo", "why")
    assert error and "Buy" in error


def test_a_decision_cannot_be_dated_in_the_future():
    """A journal is a record of what you thought at a time. Post-dating
    one is either a typo or a fiction."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    _, error = ij.add_entry(ij.JournalStore(), "AAPL", ij.BUY, "why",
                            decided_on=tomorrow)
    assert error and "future" in error


def test_entries_come_back_newest_first(tmp_path):
    store = ij.JournalStore()
    for day in (1, 15, 8):
        store, _ = ij.add_entry(store, "AAPL", ij.BUY, "why",
                                decided_on=datetime.date(2026, 3, day))
    path = tmp_path / "j.json"
    ij.save_store(store, path)
    dates = [e.decided_on for e in ij.load_store(path).entries]
    assert dates == sorted(dates, reverse=True)


# --- conviction cannot be edited after the fact --------------------------------

def test_conviction_can_be_changed_while_the_outcome_is_unknown():
    store, _ = ij.add_entry(ij.JournalStore(), "AAPL", ij.BUY, "why",
                            conviction="Low")
    entry_id = store.entries[0].id
    store, error = ij.update_conviction(store, entry_id, "High")
    assert error is None and store.entries[0].conviction == "High"


def test_conviction_is_FROZEN_once_the_outcome_has_been_seen():
    """THE RULE THAT MAKES THE PATTERN MEAN ANYTHING. Whether a single
    call was right is mostly noise; whether you are more often right
    when you felt certain is a pattern about you — and only if the
    confidence was written down BEFORE the result was known."""
    store, _ = ij.add_entry(ij.JournalStore(), "AAPL", ij.BUY, "why",
                            conviction="Low")
    entry_id = store.entries[0].id
    store, changed = ij.mark_outcome_seen(store, [entry_id])
    assert changed and store.entries[0].locked

    store, error = ij.update_conviction(store, entry_id, "High")
    assert error and "hindsight" in error
    assert store.entries[0].conviction == "Low"


def test_marking_an_outcome_seen_twice_changes_nothing():
    """The freeze records WHEN it was first shown; re-stamping it would
    move the boundary the lock depends on."""
    store, _ = ij.add_entry(ij.JournalStore(), "AAPL", ij.BUY, "why")
    entry_id = store.entries[0].id
    store, first = ij.mark_outcome_seen(store, [entry_id])
    stamp = store.entries[0].outcome_seen_at
    store, second = ij.mark_outcome_seen(store, [entry_id])
    assert first is True and second is False
    assert store.entries[0].outcome_seen_at == stamp


def test_a_review_is_kept_SEPARATE_from_the_original_reasoning():
    """Editing the original would destroy exactly the record this keeps:
    what you thought at the time, not what you now wish you had."""
    store, _ = ij.add_entry(ij.JournalStore(), "AAPL", ij.BUY,
                            "Original thesis.")
    entry_id = store.entries[0].id
    store, error = ij.add_review(store, entry_id, "In hindsight, lucky.")
    assert error is None
    entry = store.entries[0]
    assert entry.reasoning == "Original thesis."
    assert entry.review_note == "In hindsight, lucky."
    assert entry.reviewed_at


def test_a_review_on_a_missing_entry_is_refused():
    _, error = ij.add_review(ij.JournalStore(), "nope", "note")
    assert error


# --- the store ----------------------------------------------------------------

def test_a_round_trip_preserves_every_field(tmp_path):
    store, _ = ij.add_entry(ij.JournalStore(), "MSFT", ij.SELL,
                            "Multiple too rich.", conviction="High",
                            decided_on=datetime.date(2026, 2, 10),
                            price_at_decision=412.5)
    store, _ = ij.add_review(store, store.entries[0].id, "Reviewed later.")
    store, _ = ij.mark_outcome_seen(store, [store.entries[0].id])
    path = tmp_path / "j.json"
    assert ij.save_store(store, path) is None

    loaded = ij.load_store(path).entries[0]
    original = store.entries[0]
    for field in ("id", "ticker", "action", "reasoning", "conviction",
                  "decided_on", "price_at_decision", "review_note",
                  "outcome_seen_at"):
        assert getattr(loaded, field) == getattr(original, field), field
    assert loaded.locked


def test_a_missing_file_is_an_empty_journal(tmp_path):
    store = ij.load_store(tmp_path / "absent.json")
    assert store.entries == () and not store.corrupt


def test_a_corrupt_journal_is_REPORTED_not_treated_as_empty(tmp_path):
    path = tmp_path / "j.json"
    path.write_text("{ this is not json")
    store = ij.load_store(path)
    assert store.corrupt and store.entries == ()


def test_a_corrupt_journal_is_NEVER_overwritten(tmp_path):
    """THE WORST FAILURE AVAILABLE HERE. Treating an unreadable journal
    as empty means the next entry erases everything the user wrote, and
    this is the one store whose contents exist nowhere else."""
    path = tmp_path / "j.json"
    original = "{ this is not json but it is somebody's writing"
    path.write_text(original)
    store = ij.load_store(path)
    error = ij.save_store(store, path)
    assert error and "overwrite" in error
    assert path.read_text() == original


def test_a_malformed_entry_is_dropped_without_losing_the_rest(tmp_path):
    path = tmp_path / "j.json"
    path.write_text(json.dumps({"entries": [
        {"ticker": "AAPL", "reasoning": "good", "action": "Buy",
         "decided_on": "2026-01-05"},
        {"ticker": "", "reasoning": "no ticker"},
        {"ticker": "MSFT", "reasoning": ""},
        "not even a dict",
    ]}))
    store = ij.load_store(path)
    assert [e.ticker for e in store.entries] == ["AAPL"]
    assert not store.corrupt


def test_an_unknown_conviction_in_the_file_falls_back_rather_than_crashing(tmp_path):
    path = tmp_path / "j.json"
    path.write_text(json.dumps({"entries": [
        {"ticker": "AAPL", "reasoning": "why", "conviction": "Absolute"}]}))
    assert ij.load_store(path).entries[0].conviction == "Medium"


def test_deleting_an_entry_leaves_the_others(tmp_path):
    store = ij.JournalStore()
    for ticker in ("AAPL", "MSFT"):
        store, _ = ij.add_entry(store, ticker, ij.BUY, "why")
    victim = store.entries[0].id
    store = ij.delete_entry(store, victim)
    assert len(store.entries) == 1 and store.entries[0].id != victim


def test_entries_can_be_filtered_by_ticker():
    store = ij.JournalStore()
    for ticker in ("AAPL", "MSFT", "AAPL"):
        store, _ = ij.add_entry(store, ticker, ij.BUY, "why")
    assert len(store.for_ticker("aapl")) == 2
    assert store.tickers == ("AAPL", "MSFT")


# --- outcomes -----------------------------------------------------------------

def test_the_outcome_is_measured_from_prices_not_stored():
    """A stored return is frozen at the moment of writing and quietly
    wrong forever after — and this is the feature whose value depends on
    the number being current."""
    entry = _entry(decided_on=datetime.date(2026, 1, 1))
    early = ij.measure_outcome(entry, _prices([100.0] * 30 + [110.0]))
    later = ij.measure_outcome(entry, _prices([100.0] * 30 + [150.0]))
    assert early.change_pct == pytest.approx(10.0)
    assert later.change_pct == pytest.approx(50.0)
    assert not hasattr(entry, "change_pct")


def test_the_recorded_price_is_preferred_as_the_starting_point():
    """It is what the writer actually saw."""
    entry = _entry(price_at_decision=50.0)
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [110.0]))
    assert outcome.price_then == pytest.approx(50.0)
    assert outcome.change_pct == pytest.approx(120.0)


def test_without_a_recorded_price_the_close_on_the_decision_date_is_used():
    entry = _entry()
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [110.0]))
    assert outcome.price_then == pytest.approx(100.0)


def test_a_SELL_that_fell_went_the_writers_way():
    """Reporting a short thesis that worked as "-12%" reads as a loss on
    their own record."""
    entry = _entry(action=ij.SELL)
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [88.0]))
    assert outcome.change_pct < 0
    assert outcome.went_as_expected is True
    assert "betting on" in ij.describe_outcome(outcome)


def test_a_BUY_that_fell_went_against_the_writer():
    entry = _entry(action=ij.BUY)
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [88.0]))
    assert outcome.went_as_expected is False
    assert "against the direction" in ij.describe_outcome(outcome)


def test_a_WATCH_is_not_a_bet_and_is_not_scored():
    entry = _entry(action=ij.WATCH)
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [88.0]))
    assert outcome.went_as_expected is None


def test_a_young_decision_is_not_given_an_outcome_to_read():
    """A week's move says nothing about a thesis, and showing it invites
    reading one anyway."""
    entry = _entry(decided_on=datetime.date.today() - datetime.timedelta(days=5))
    index = pd.date_range(datetime.date.today() - datetime.timedelta(days=5),
                          periods=5, freq="D")
    outcome = ij.measure_outcome(entry, pd.Series([100, 101, 102, 103, 104],
                                                  index=index, dtype="float64"))
    assert outcome.ok and not outcome.mature
    assert "too early" in ij.describe_outcome(outcome)


def test_the_benchmark_gap_is_reported_for_a_BUY():
    entry = _entry(action=ij.BUY)
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [120.0]),
                                 _prices([100.0] * 30 + [110.0]), "SPY")
    assert outcome.versus_benchmark_pct == pytest.approx(10.0)
    assert "gap is +10.0 points" in ij.describe_outcome(outcome)


def test_no_benchmark_GAP_is_computed_for_a_SELL():
    """Subtracting implies a counterfactual nobody supplied: the
    difference depends on what the proceeds did, and this app does not
    know whether they went into the index or into cash. Reporting -20
    points on a short thesis that WORKED reads as underperformance."""
    entry = _entry(action=ij.SELL)
    outcome = ij.measure_outcome(entry, _prices([100.0] * 30 + [90.0]),
                                 _prices([100.0] * 30 + [110.0]), "SPY")
    assert outcome.versus_benchmark_pct is None
    text = ij.describe_outcome(outcome)
    assert "depends on what the proceeds did" in text
    assert "gap is" not in text


def test_an_entry_with_no_price_history_says_so():
    entry = _entry()
    assert not ij.measure_outcome(entry, None).ok
    assert not ij.measure_outcome(entry, pd.Series(dtype="float64")).ok


def test_an_entry_written_TODAY_is_not_reported_as_missing_data():
    """Every entry written today sits after the last close, so the naive
    check calls it "the price history does not reach the decision date"
    — a data failure for what is simply a brand-new entry. Seen live."""
    today = datetime.date.today()
    entry = _entry(decided_on=today)
    index = pd.date_range(today - datetime.timedelta(days=10), periods=6,
                          freq="D")
    outcome = ij.measure_outcome(entry, pd.Series([100.0] * 6, index=index,
                                                  dtype="float64"))
    assert outcome.too_new
    assert not outcome.error
    assert "nothing to measure" in ij.describe_outcome(outcome)
    assert not outcome.mature


def test_history_that_stops_before_the_decision_is_reported():
    """A genuine gap — the decision predates nothing and the series
    stops years short — is still an error, not "too new"."""
    entry = _entry(decided_on=datetime.date(2020, 6, 1))
    outcome = ij.measure_outcome(entry, _prices([100.0] * 10, "2019-01-01"))
    assert not outcome.ok and not outcome.too_new
    assert "does not reach" in outcome.error


# --- the pattern --------------------------------------------------------------

def _matured(entry_id, went_well):
    return ij.Outcome(entry_id, "AAPL", ij.BUY, days_held=90,
                      price_then=100.0, price_now=110.0 if went_well else 90.0,
                      change_pct=10.0 if went_well else -10.0)


def test_a_hit_rate_is_WITHHELD_below_a_meaningful_sample():
    """A hit rate over three decisions is a coin flip with a percentage
    sign, and this panel's whole claim is that it shows you something
    real about your own judgement."""
    store = ij.JournalStore()
    for _ in range(3):
        store, _ = ij.add_entry(store, "AAPL", ij.BUY, "why",
                                conviction="High")
    outcomes = {e.id: _matured(e.id, True) for e in store.entries}
    rows = ij.conviction_pattern(store.entries, outcomes)
    high = next(r for r in rows if r.conviction == "High")
    assert high.matured == 3
    assert high.hit_rate_pct is None and not high.scored
    assert "at least" in ij.describe_pattern(rows)


def test_a_hit_rate_appears_once_the_sample_is_large_enough():
    store = ij.JournalStore()
    for _ in range(ij.MIN_DECISIONS_FOR_PATTERN):
        store, _ = ij.add_entry(store, "AAPL", ij.BUY, "why",
                                conviction="High")
    outcomes = {e.id: _matured(e.id, i % 2 == 0)
                for i, e in enumerate(store.entries)}
    rows = ij.conviction_pattern(store.entries, outcomes)
    high = next(r for r in rows if r.conviction == "High")
    assert high.scored
    assert high.hit_rate_pct == pytest.approx(50.0)


def test_immature_outcomes_do_not_count_toward_the_pattern():
    """Counting a five-day-old decision as a hit is how a hit rate
    becomes a measure of how recently you traded."""
    store = ij.JournalStore()
    for _ in range(ij.MIN_DECISIONS_FOR_PATTERN):
        store, _ = ij.add_entry(store, "AAPL", ij.BUY, "why",
                                conviction="High")
    young = {e.id: ij.Outcome(e.id, "AAPL", ij.BUY, days_held=3,
                              change_pct=10.0) for e in store.entries}
    rows = ij.conviction_pattern(store.entries, young)
    assert all(not r.scored for r in rows)


def test_the_pattern_names_which_conviction_did_better():
    store = ij.JournalStore()
    outcomes = {}
    for level, wins in (("High", 9), ("Low", 2)):
        for i in range(ij.MIN_DECISIONS_FOR_PATTERN):
            store, _ = ij.add_entry(store, "AAPL", ij.BUY, "why",
                                    conviction=level)
            outcomes[store.entries[0].id] = _matured(store.entries[0].id,
                                                     i < wins)
    rows = ij.conviction_pattern(store.entries, outcomes)
    text = ij.describe_pattern(rows)
    assert "high-conviction" in text and "gone your way more often" in text


# --- what it refuses to do ----------------------------------------------------

def test_it_does_not_score_a_decision_as_good_or_bad():
    """A buy that fell is not a mistake and a buy that rose is not a
    skill: over one holding period those are mostly the market."""
    names = [n for n in dir(ij) if not n.startswith("_")]
    assert not [n for n in names
                if any(w in n.lower() for w in ("grade", "score_entry",
                                                "rating", "verdict"))]
    assert "scoreboard" in ij.NOT_A_SCOREBOARD


def test_the_two_standing_notes_explain_themselves():
    for note in (ij.NOT_A_SCOREBOARD, ij.CONVICTION_IS_THE_LEARNABLE_PART):
        assert len(note) > 80 and note.strip().endswith(".")


# --- the UI wiring ------------------------------------------------------------

import pathlib

FINANCE = (pathlib.Path(__file__).resolve().parent.parent
           / "finance.py").read_text(encoding="utf-8")


def _panel():
    start = FINANCE.index("# INVESTMENT JOURNAL (why you decided")
    return FINANCE[start:FINANCE.index("# TEAM NOTES (per-ticker thread")]


def test_the_panel_refuses_to_write_over_a_corrupt_journal():
    """The module refuses, and the panel has to surface that rather than
    showing an empty form the user would type into."""
    panel = _panel()
    assert "_ij_store.corrupt" in panel
    assert "overwrite entries still in the file" in panel


def test_seeing_a_mature_outcome_freezes_that_entrys_conviction():
    """The lock is what makes the pattern mean anything, and it has to
    fire from the panel — the module cannot know when a number was
    shown."""
    panel = _panel()
    assert "mark_outcome_seen" in panel
    assert "_ij_o.mature" in panel


def test_only_MATURE_outcomes_trigger_the_freeze():
    """Freezing on a five-day-old entry would lock conviction before the
    reader has seen anything worth calling an outcome."""
    panel = _panel()
    freeze = panel.index("mark_outcome_seen")
    guard = panel.index("if _ij_o.mature:")
    assert guard < freeze


def test_the_reasoning_box_is_cleared_with_a_DEFERRED_flag():
    """Popping a widget's own key inside the handler does nothing —
    Streamlit restores it from its widget-state layer on the next run,
    and a box still holding what you just saved reads as 'nothing
    happened'."""
    panel = _panel()
    assert 'st.session_state.pop("journal_clear", False)' in panel
    assert 'st.session_state["journal_reasoning"] = ""' in panel
    assert 'st.session_state["journal_clear"] = True' in panel


def test_the_delete_button_is_marked_destructive():
    import button_roles

    assert "journal_remove_" in button_roles.DANGER_PREFIXES


def test_per_entry_widget_keys_carry_the_entry_id():
    """The panel loops over entries; a key that did not vary would
    collide the moment a ticker had two — the alert bell's crash."""
    panel = _panel()
    for prefix in ("journal_review_", "journal_rev_save_", "journal_remove_"):
        assert f'{prefix}{{_ij_e.id}}' in panel, prefix


def test_the_outcome_is_recomputed_in_the_panel_not_read_from_the_entry():
    panel = _panel()
    assert "measure_outcome" in panel
    assert "load_price_history_only" in panel


def test_the_panel_uses_the_sidebars_own_benchmark():
    """Rather than hard-coding one, so the comparison matches whatever
    the reader is measuring everything else against."""
    assert "benchmark_symbol" in _panel()


def test_the_two_standing_notes_reach_the_page():
    panel = _panel()
    assert "NOT_A_SCOREBOARD" in panel
    assert "CONVICTION_IS_THE_LEARNABLE_PART" in panel


def test_the_date_input_cannot_be_set_to_the_future():
    assert "max_value=datetime.date.today()" in _panel()
