"""The weekly highlight: the gate, the rotation, the freeze, and a
writeup that predicts nothing.

THE TEST THAT MATTERS MOST is test_the_same_name_is_not_featured_twice_
until_the_cohort_is_exhausted. The naive build — rank by alignment score,
take the top — was measured against the live basket on 2026-09-23 and
produced a four-way tie at 100 (MSFT/GOOGL/NVDA/META) with seven more at
75. Broken alphabetically that features GOOGL every week forever, because
these are statement figures that move once a quarter. A "Stock of the
Week" that never changes is the whole feature failing silently, and no
unit test of the scorer would have caught it.
"""
import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stock_of_the_week as sotw
from config import STOCK_OF_THE_WEEK
from fundamental_analysis import WatchlistCheck

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def _check(label="Net margin", passed=True, evaluable=True,
           display="40.30%", benchmark="at least 10%"):
    return WatchlistCheck(label, passed, evaluable, display, benchmark)


def _cand(ticker, score=100.0, status="High", checks=None, figures=None,
          sector="Technology"):
    return sotw.Candidate(
        ticker=ticker, score=score, status=status,
        checks=checks if checks is not None else (_check(),) * 4,
        figures=figures or {}, sector=sector,
    )


def _pick(week, ticker, score=100.0):
    return sotw.Pick(week=week, ticker=ticker, score=score, status="High",
                     chosen_at="2026-09-23T00:00:00")


def _store(*picks):
    return sotw.PickStore(tuple(picks))


# --- the week label -----------------------------------------------------------

def test_the_week_label_is_the_iso_week():
    assert sotw.week_for(datetime.date(2026, 9, 23)) == "2026-W39"


def test_every_day_of_one_iso_week_gets_the_same_label():
    """Monday to Sunday must agree, or the card changes mid-week."""
    monday = datetime.date(2026, 9, 21)
    labels = {sotw.week_for(monday + datetime.timedelta(days=d)) for d in range(7)}
    assert len(labels) == 1


def test_the_label_follows_the_iso_year_not_the_calendar_year():
    """1 Jan 2027 is a Friday and belongs to ISO week 53 of 2026. A label
    built from `day.year` would file it under 2027-W53, which does not
    exist, and week_bounds would then refuse to parse its own output."""
    label = sotw.week_for(datetime.date(2027, 1, 1))
    assert label == "2026-W53"
    start, end = sotw.week_bounds(label)
    assert start == datetime.date(2026, 12, 28)
    assert end == datetime.date(2027, 1, 3)


def test_week_bounds_are_monday_to_sunday():
    start, end = sotw.week_bounds("2026-W39")
    assert start == datetime.date(2026, 9, 21) and start.weekday() == 0
    assert end == datetime.date(2026, 9, 27)


def test_week_bounds_never_raises_on_junk():
    assert sotw.week_bounds("not-a-week") == (None, None)


# --- the quality gate ---------------------------------------------------------

def test_only_names_at_or_above_the_gate_are_eligible():
    cands = [_cand("AAA", 100.0), _cand("BBB", 75.0), _cand("CCC", 50.0),
             _cand("DDD", 25.0)]
    assert [c.ticker for c in sotw.eligible(cands)] == ["AAA", "BBB"]


def test_the_gate_is_inclusive_at_its_own_boundary():
    assert sotw.eligible([_cand("X", STOCK_OF_THE_WEEK.min_score)])


def test_nothing_is_featured_when_nothing_clears_the_gate():
    """Featuring the least-bad name would attach 'noteworthy' to something
    the app's own pre-screen just rejected."""
    pick, reason = sotw.choose([_cand("AAA", 50.0), _cand("BBB", 25.0)],
                               sotw.PickStore(), "2026-W39")
    assert pick is None
    assert "AAA" in reason and "50" in reason
    assert "75" in reason


def test_an_unscreenable_basket_says_so_rather_than_inventing_a_pick():
    pick, reason = sotw.choose([], sotw.PickStore(), "2026-W39")
    assert pick is None and reason == sotw.NO_CANDIDATES


# --- the rotation -------------------------------------------------------------

def test_a_name_never_featured_goes_before_every_name_that_has_been():
    store = _store(_pick("2026-W38", "AAA"))
    order = sotw.rotation_order([_cand("AAA"), _cand("BBB")], store)
    assert [c.ticker for c in order] == ["BBB", "AAA"]


def test_the_longest_unfeatured_name_goes_first():
    store = _store(_pick("2026-W36", "CCC"), _pick("2026-W37", "AAA"),
                   _pick("2026-W38", "BBB"))
    order = sotw.rotation_order([_cand("AAA"), _cand("BBB"), _cand("CCC")], store)
    assert [c.ticker for c in order] == ["CCC", "AAA", "BBB"]


def test_among_equally_unfeatured_names_the_higher_score_goes_first():
    order = sotw.rotation_order([_cand("BBB", 75.0), _cand("AAA", 100.0)],
                                sotw.PickStore())
    assert [c.ticker for c in order] == ["AAA", "BBB"]


def test_a_remaining_tie_breaks_alphabetically_so_the_order_is_stable():
    """Two readers loading the same week must compute the same card."""
    order = sotw.rotation_order([_cand("ZZZ"), _cand("AAA"), _cand("MMM")],
                                sotw.PickStore())
    assert [c.ticker for c in order] == ["AAA", "MMM", "ZZZ"]


def test_the_same_name_is_not_featured_twice_until_the_cohort_is_exhausted():
    """The defect this feature's design exists to avoid. Eleven names tied
    at or above the gate — as the live basket actually was — must produce
    eleven distinct weeks before any repeat."""
    tickers = ["MSFT", "GOOGL", "NVDA", "META", "AAPL", "AMZN", "V", "LLY",
               "AVGO", "ASML", "CAT"]
    cands = [_cand(t, 100.0 if t in ("MSFT", "GOOGL", "NVDA", "META") else 75.0)
             for t in tickers]

    store = sotw.PickStore()
    featured = []
    for week_number in range(1, len(tickers) + 1):
        pick, reason = sotw.choose(cands, store, f"2026-W{week_number:02d}")
        assert pick is not None, reason
        featured.append(pick.ticker)
        store = sotw.record(store, pick)

    assert sorted(featured) == sorted(tickers)
    assert len(set(featured)) == len(tickers), "a name repeated before the cycle ended"


def test_the_cycle_restarts_in_the_same_order_after_it_completes():
    tickers = ["AAA", "BBB", "CCC"]
    cands = [_cand(t) for t in tickers]
    store = sotw.PickStore()
    seen = []
    for week_number in range(1, 7):
        pick, _ = sotw.choose(cands, store, f"2026-W{week_number:02d}")
        seen.append(pick.ticker)
        store = sotw.record(store, pick)
    assert seen == ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"]


def test_a_name_that_drops_below_the_gate_is_skipped_without_breaking_rotation():
    store = _store(_pick("2026-W38", "AAA"))
    order = sotw.rotation_order([_cand("AAA"), _cand("BBB", 50.0), _cand("CCC")],
                                store)
    assert [c.ticker for c in order] == ["CCC", "AAA"]


# --- the freeze ---------------------------------------------------------------

def test_an_already_recorded_week_is_returned_unchanged():
    store = _store(_pick("2026-W39", "AAA"))
    pick, reason = sotw.choose([_cand("BBB")], store, "2026-W39")
    assert pick.ticker == "AAA" and reason == ""


def test_the_freeze_holds_even_when_the_recorded_name_no_longer_qualifies():
    """A mid-week change in the underlying figures must not swap the card
    out from under someone who already read it."""
    store = _store(_pick("2026-W39", "AAA"))
    pick, _ = sotw.choose([_cand("AAA", 25.0), _cand("BBB", 100.0)], store,
                          "2026-W39")
    assert pick.ticker == "AAA"


def test_the_freeze_is_checked_before_the_candidates_are():
    """An empty candidate list must not blank a week that is already
    decided — the basket scan can fail transiently."""
    store = _store(_pick("2026-W39", "AAA"))
    pick, reason = sotw.choose([], store, "2026-W39")
    assert pick is not None and pick.ticker == "AAA" and reason == ""


def test_a_new_week_gets_a_new_pick():
    store = _store(_pick("2026-W39", "AAA"))
    pick, _ = sotw.choose([_cand("AAA"), _cand("BBB")], store, "2026-W40")
    assert pick.ticker == "BBB" and pick.week == "2026-W40"


def test_the_recorded_pick_carries_the_score_that_qualified_it():
    pick, _ = sotw.choose([_cand("AAA", 75.0, "High")], sotw.PickStore(), "2026-W39")
    assert pick.score == 75.0 and pick.status == "High" and pick.chosen_at


# --- the history --------------------------------------------------------------

def test_recording_the_same_week_twice_replaces_rather_than_duplicates():
    store = sotw.record(_store(_pick("2026-W39", "AAA")), _pick("2026-W39", "BBB"))
    assert len(store.picks) == 1 and store.picks[0].ticker == "BBB"


def test_the_history_stays_in_week_order():
    store = sotw.record(_store(_pick("2026-W40", "BBB")), _pick("2026-W39", "AAA"))
    assert [p.week for p in store.picks] == ["2026-W39", "2026-W40"]


def test_the_history_is_trimmed_to_its_cap():
    store = sotw.PickStore()
    for i in range(1, STOCK_OF_THE_WEEK.max_history + 6):
        store = sotw.record(store, _pick(f"2026-W{i:03d}", f"T{i}"))
    assert len(store.picks) == STOCK_OF_THE_WEEK.max_history


# --- persistence --------------------------------------------------------------

def test_a_pick_survives_a_round_trip(tmp_path):
    path = tmp_path / "s.json"
    sotw.save_store(_store(_pick("2026-W39", "MSFT", 100.0)), path)
    loaded = sotw.load_store(path)
    assert loaded.for_week("2026-W39").ticker == "MSFT"
    assert loaded.for_week("2026-W39").score == 100.0
    assert loaded.corrupt is False


def test_a_missing_store_is_empty_not_corrupt(tmp_path):
    store = sotw.load_store(tmp_path / "nope.json")
    assert store.picks == () and store.corrupt is False


def test_an_unreadable_store_is_corrupt_not_empty(tmp_path):
    """Treating it as empty would let the next save replace the rotation
    history — and the rotation IS that history."""
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert sotw.load_store(path).corrupt is True


def test_a_corrupt_store_is_never_written_over(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert sotw.save_store(sotw.PickStore(corrupt=True), path) is False
    assert path.read_text() == "{not json"


def test_a_junk_entry_is_dropped_rather_than_taking_the_store_down(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"picks": [
        {"week": "2026-W39", "ticker": "MSFT", "score": 100.0},
        {"ticker": "NOWEEK"}, {"week": "2026-W40"}, "nonsense",
    ]}))
    store = sotw.load_store(path)
    assert [p.ticker for p in store.picks] == ["MSFT"]
    assert store.corrupt is False


def test_the_store_path_is_shared_not_per_user():
    """One Stock of the Week for the instance. A per-user path would let
    two people on the same team see different ones."""
    import re
    src = Path(sotw.__file__).read_text()
    assert "shared_path(" in src
    # A bare `store_path(` only — "_store_path(" is this module's own
    # private helper and would match a naive substring check.
    assert re.search(r"(?<![_\w])store_path\(", src) is None
    imports = [ln for ln in src.splitlines() if ln.startswith("from local_store import")]
    assert imports and not any("store_path" in ln.replace("shared_path", "")
                               for ln in imports)


def test_the_store_is_declared_shared_in_auth():
    import auth
    assert STOCK_OF_THE_WEEK.store_filename in auth.SHARED_STORES
    assert STOCK_OF_THE_WEEK.store_filename not in auth.PER_USER_STORES


# --- the writeup --------------------------------------------------------------

def test_the_writeup_names_every_check_that_was_passed():
    cand = _cand("MSFT", checks=(
        _check("Net margin", True, True, "40.30%", "at least 10%"),
        _check("Debt-to-equity", True, True, "0.33", "under 1.00"),
        _check("Current ratio", True, True, "1.37", "above 1.20"),
        _check("P/E ratio", True, True, "27.73", "15-45"),
    ))
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    for figure in ("40.30%", "0.33", "1.37", "27.73"):
        assert figure in text
    assert "all 4" in text


def test_an_acronym_label_is_not_lowercased_mid_sentence():
    """A blunt .lower() rendered "P/E ratio" as "p/e ratio" on the live
    card — caught by reading the output, not by any unit assertion that
    existed at the time."""
    cand = _cand("MSFT", checks=(
        _check("Net margin", True, True, "40.30%", "at least 10%"),
        _check("P/E ratio", True, True, "27.73", "15-45"),
    ))
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    assert "P/E ratio 27.73" in text
    assert "p/e" not in text
    assert "net margin 40.30%" in text, "an ordinary word still reads lowercase"


def test_the_sector_is_parenthesised_not_run_into_the_sentence():
    cand = _cand("GOOGL", figures={"return_on_equity": 0.4868},
                 sector="Communication Services")
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    assert "Also reported (Communication Services):" in text


def test_a_failed_check_is_reported_with_its_threshold():
    cand = _cand("AVGO", 75.0, checks=(
        _check("Net margin", True, True, "42.94%", "at least 10%"),
        _check("Debt-to-equity", True, True, "0.90", "under 1.00"),
        _check("Current ratio", True, True, "1.30", "above 1.20"),
        _check("P/E ratio", False, True, "46.56", "15-45"),
    ))
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    assert "clears 3 of the 4" in text
    assert "misses" in text and "46.56" in text and "15-45" in text


def test_an_unreported_figure_is_not_described_as_a_failed_check():
    """A rule that could not be evaluated is not a rule that did not fire.
    The score counts it as a miss — that is pre-existing and deliberate —
    but the prose must not tell a reader the company failed a test that
    was never run on it."""
    cand = _cand("XXX", 75.0, checks=(
        _check("Net margin", True, True, "20.00%", "at least 10%"),
        _check("Debt-to-equity", False, False, "Not reported", "under 1.00"),
        _check("Current ratio", True, True, "1.50", "above 1.20"),
        _check("P/E ratio", True, True, "22.00", "15-45"),
    ))
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    assert "not reported" in text.lower()
    assert "misses Debt-to-equity" not in text
    assert "misses debt-to-equity" not in text


def test_the_writeup_says_it_rotates_rather_than_ranks():
    """A reader who thinks the app picked a WINNER has been told
    something false — the score cannot order the cohort."""
    text = " ".join(sotw.writeup(_cand("MSFT"), sotw.PickStore(), 11))
    assert "rotates" in text
    assert "11 names" in text


def test_a_cohort_of_one_is_not_described_as_a_rotation():
    text = " ".join(sotw.writeup(_cand("MSFT"), sotw.PickStore(), 1))
    assert "only name" in text
    assert "rotates" not in text


def test_a_returning_name_says_when_it_was_last_featured():
    store = _store(_pick("2026-W28", "MSFT"), _pick("2026-W39", "MSFT"))
    text = " ".join(sotw.writeup(_cand("MSFT"), store, 4))
    assert "2026-W28" in text


def test_a_first_time_name_claims_no_previous_appearance():
    text = " ".join(sotw.writeup(_cand("MSFT"), _store(_pick("2026-W39", "MSFT")), 4))
    assert "last featured" not in text


def test_the_writeup_makes_no_prediction():
    """recommendations.py refuses to claim a stock will go up; this card
    sits on the same page and must not undo it."""
    move = sotw.weekly_move([100, 101, 102, 103, 104, 105],
                            ["09-15", "09-16", "09-17", "09-18", "09-19", "09-22"])
    text = " ".join(sotw.writeup(_cand("MSFT"), sotw.PickStore(), 4, move)).lower()
    for banned in ("buy", "sell", "will rise", "expected to", "target price",
                   "outperform", "undervalued", "recommend"):
        assert banned not in text, f"the writeup implies a call: {banned!r}"


def test_the_price_move_is_labelled_as_a_fact_not_a_reason():
    move = sotw.weekly_move([100, 101, 102, 103, 104, 105],
                            ["a", "b", "c", "d", "e", "f"])
    text = " ".join(sotw.writeup(_cand("MSFT"), sotw.PickStore(), 4, move))
    assert "not a reason it was chosen" in text


def test_a_figure_that_is_not_reported_is_omitted_rather_than_shown_as_zero():
    cand = _cand("XXX", figures={"return_on_equity": None, "revenue_growth": 0.12})
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    assert "ROE" not in text
    assert "revenue growth 12.00%" in text


def test_fraction_valued_figures_are_not_shown_as_raw_fractions():
    """StandardizedFinancials mixes fractions with percent-valued fields;
    this app has shipped that confusion more than once."""
    cand = _cand("XXX", figures={"return_on_equity": 0.35})
    text = " ".join(sotw.writeup(cand, sotw.PickStore(), 4))
    assert "ROE 35.00%" in text
    assert "ROE 0.35" not in text


def test_a_company_with_no_reported_extras_gets_a_shorter_writeup_not_a_padded_one():
    cand = _cand("XXX", figures={})
    sentences = sotw.writeup(cand, sotw.PickStore(), 4)
    assert all(s.strip() for s in sentences)
    assert not any("Also reported" in s for s in sentences)


def test_the_disclosure_refuses_the_prediction_reading():
    assert "not that it will rise" in sotw.DISCLOSURE
    assert "rotates" in sotw.DISCLOSURE


# --- the week's price move ----------------------------------------------------

def test_the_move_spans_sessions_plus_one_bars():
    """A five-session move is measured against the close BEFORE those
    five. Using the first of the five silently reports a four-session
    move, and every figure on the card would be wrong by one day."""
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    move = sotw.weekly_move(closes, ["a", "b", "c", "d", "e", "f"], sessions=5)
    assert move.available
    assert move.pct == pytest.approx(50.0)  # 15 from 10, not 15 from 11


def test_the_move_is_measured_over_the_most_recent_window():
    closes = [1.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    move = sotw.weekly_move(closes, list("abcdefgh"), sessions=5)
    assert move.pct == pytest.approx(5.0)
    assert move.start_date == "c" and move.end_date == "h"


def test_a_fall_is_reported_as_a_fall():
    move = sotw.weekly_move([100.0, 99, 98, 97, 96, 95], list("abcdef"), sessions=5)
    assert move.pct == pytest.approx(-5.0)
    text = sotw._move_sentence(move)
    assert "down 5.00%" in text


def test_too_few_bars_reports_the_shortfall_rather_than_a_number():
    move = sotw.weekly_move([100.0, 101.0, 102.0], ["a", "b", "c"], sessions=5)
    assert not move.available
    assert "3 usable closes" in move.reason and "6 are needed" in move.reason


def test_exactly_sessions_bars_is_still_too_few():
    """The boundary, not a number comfortably short of it. Five closes
    describe FOUR sessions; accepting them reports a four-session move
    under a five-session label, and a shortfall test using three bars
    passes against that build — this one was added after a poison run
    proved it."""
    move = sotw.weekly_move([100.0, 101.0, 102.0, 103.0, 104.0],
                            list("abcde"), sessions=5)
    assert not move.available, "five closes span four sessions, not five"
    assert "5 usable closes" in move.reason


def test_exactly_sessions_plus_one_bars_is_enough():
    """The other side of the same boundary — the guard must not be so
    tight that a complete week is refused."""
    move = sotw.weekly_move([100.0, 101.0, 102.0, 103.0, 104.0, 110.0],
                            list("abcdef"), sessions=5)
    assert move.available and move.pct == pytest.approx(10.0)


def test_nan_closes_are_not_counted_as_usable_bars():
    move = sotw.weekly_move([float("nan")] * 4 + [100.0, 105.0], list("abcdef"),
                            sessions=5)
    assert not move.available


def test_a_zero_first_close_is_refused_rather_than_dividing():
    move = sotw.weekly_move([0.0, 1, 2, 3, 4, 5], list("abcdef"), sessions=5)
    assert not move.available


# --- the basket ---------------------------------------------------------------

def test_the_universe_is_the_same_basket_the_alignment_cards_scan():
    """A second universe here would mean two lists to keep in step and a
    second hourly fetch of the same names."""
    from config import WATCHLIST
    assert sotw.basket() == tuple(WATCHLIST.tech_basket) + tuple(WATCHLIST.diversified_basket)
    assert len(sotw.basket()) == len(set(sotw.basket())), "a duplicated ticker would be featured twice"


# --- the pre-screen breakdown it depends on -----------------------------------

def test_the_checks_agree_with_the_score_they_came_from():
    """The breakdown and the score must not be two computations. Built
    from a live-shaped standardized record rather than mocked, so a
    threshold change moves both together or this fails."""
    from dataclasses import dataclass as _dc
    from fundamental_analysis import FundamentalAnalysisEngine

    class _Std:
        ticker = "TEST"
        pe_ratio = 27.73
        net_margin = 0.4030
        debt_to_equity = 0.33
        current_ratio = 1.37

    score = FundamentalAnalysisEngine(_Std()).screen_watchlist()
    assert len(score.checks) == 4
    passed = sum(1 for c in score.checks if c.passed)
    assert score.score == (passed / 4) * 100


def test_an_unreported_input_is_marked_unevaluable_but_still_costs_the_score():
    from fundamental_analysis import FundamentalAnalysisEngine

    class _Std:
        ticker = "TEST"
        pe_ratio = 20.0
        net_margin = 0.20
        debt_to_equity = None
        current_ratio = 1.50

    score = FundamentalAnalysisEngine(_Std()).screen_watchlist()
    de = next(c for c in score.checks if c.label == "Debt-to-equity")
    assert de.evaluable is False and de.passed is False
    assert de.display == "Not reported"
    assert score.score == 75.0, "the pre-existing scoring must not have moved"


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# STOCK OF THE WEEK")
    end = src.index("# CUSTOM THRESHOLDS", start)
    # Comments stripped: this block's own commentary names the things it
    # avoids, and a substring check would match the explanation rather
    # than the code. Same trap as the CSS declaration tests.
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_card_renders_above_the_analysis_tabs():
    """'Surfaced on login' means the reader meets it without scrolling
    past the whole page."""
    src = FINANCE.read_text()
    assert src.index("# STOCK OF THE WEEK") < src.index('st.header("Stock Screener")')


def test_the_panel_freezes_through_the_store_rather_than_recomputing():
    panel = _panel()
    assert "sotw_choose(" in panel
    assert "sotw_load_store(" in panel
    assert "sotw_record(" in panel
    assert "sotw_save_store(" in panel


def test_the_panel_shows_the_reason_when_there_is_no_pick():
    panel = _panel()
    assert "_sotw_reason" in panel


def test_the_panel_renders_the_derived_writeup_not_its_own_prose():
    panel = _panel()
    assert "sotw_writeup(" in panel


def test_the_panel_carries_the_disclosure():
    panel = _panel()
    assert "SOTW_DISCLOSURE" in panel


def test_the_panel_can_load_the_featured_ticker():
    """A highlight you cannot act on is a poster."""
    panel = _panel()
    assert "_pending_ticker" in panel
    assert "sotw_open" in panel


def test_the_panel_never_writes_the_store_for_a_signed_out_reader():
    """require_sign_in stops the script above this point, so reaching the
    panel already implies a session — but the store write must still be
    gated on a pick existing rather than firing on every rerun."""
    panel = _panel()
    save = panel.index("sotw_save_store(")
    guard = panel.index("if _sotw_pick is not None")
    assert guard < save
