"""Empty states, and the screener's "which filter is blocking" readout.

The interesting assertions are about the screener. "No stocks match. Try
adjusting filters" is advice with no information in it, and this screener
can do better: every ScreenResult carries one pass/fail per criterion, so
the filter that actually did the rejecting is countable. These tests pin
that arithmetic — particularly the distinction between a criterion a
ticker FAILED and one that could not be evaluated for it, which is the
same never-fabricate rule the rest of the app follows.
"""
import ast
import re
from pathlib import Path

import pytest

import empty_states as es
from screener import ScreenCriterion, criterion_text


FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


class FakeResult:
    def __init__(self, ticker, passes, status="ok"):
        self.ticker = ticker
        self.criteria_passes = passes
        self.status = status


PE = ScreenCriterion("pe_ratio", "<", 15.0)
ROE = ScreenCriterion("return_on_equity", ">", 20.0)
SECTOR = ScreenCriterion("sector", "is", "Technology")


# --- the arithmetic -----------------------------------------------------------

def test_the_worst_blocker_is_the_one_that_rejected_most():
    results = [FakeResult("A", [False, True]), FakeResult("B", [False, True]),
               FakeResult("C", [False, True]), FakeResult("D", [True, False])]
    blockers = es.blocking_criteria(results, [PE, ROE])
    assert blockers[0].index == 0 and blockers[0].failed == 3
    assert blockers[1].index == 1 and blockers[1].failed == 1


def test_an_unmeasurable_metric_is_not_counted_as_a_rejection():
    """criteria_passes[i] is None when the metric could not be computed.
    Folding that into "failed" would report a filter as turning away
    companies it never got to judge."""
    results = [FakeResult("A", [None]), FakeResult("B", [None]),
               FakeResult("C", [False])]
    only = es.blocking_criteria(results, [PE])[0]
    assert only.failed == 1
    assert only.unavailable == 2
    assert only.considered == 3


def test_a_short_passes_list_contributes_nothing_past_its_end():
    """A fetch error can leave fewer passes than criteria. That ticker must
    not be counted as failing the criteria it never reached."""
    results = [FakeResult("OK", [False, False]),
               FakeResult("ERR", [], status="fetch_error")]
    first, second = sorted(es.blocking_criteria(results, [PE, ROE]),
                           key=lambda b: b.index)
    assert first.considered == 1 and second.considered == 1
    assert first.failed == 1 and second.failed == 1


def test_ties_keep_their_original_order():
    """So the readout does not reshuffle between two runs of the same
    screen, which would read as the answer changing."""
    results = [FakeResult("A", [False, False])]
    blockers = es.blocking_criteria(results, [PE, ROE])
    assert [b.index for b in blockers] == [0, 1]


def test_every_criterion_appears_even_when_it_rejected_nobody():
    results = [FakeResult("A", [False, True])]
    assert len(es.blocking_criteria(results, [PE, ROE])) == 2


def test_a_categorical_threshold_does_not_crash_the_readout():
    """"{:g}" and round() both raise on "Technology" — two crashes have
    already shipped from that assumption."""
    blockers = es.blocking_criteria([FakeResult("A", [False])], [SECTOR])
    assert blockers[0].text == "Sector is Technology"


def test_the_blocker_text_is_the_shared_formatter():
    """Not a second copy that can drift from the saved-screener summary."""
    blocker = es.blocking_criteria([FakeResult("A", [False])], [PE])[0]
    assert blocker.text == criterion_text("pe_ratio", "<", 15.0)


def test_sentence_mentions_unmeasured_only_when_there_are_some():
    plain = es.Blocker(0, "P/E < 15", failed=3, unavailable=0, considered=4)
    assert "could not be measured" not in plain.sentence()
    mixed = es.Blocker(0, "P/E < 15", failed=3, unavailable=1, considered=4)
    assert "1 could not be measured" in mixed.sentence()


# --- the guidance the panel shows ---------------------------------------------

def test_guidance_offers_the_worst_filter_for_removal():
    results = [FakeResult("A", [False, True]), FakeResult("B", [False, True])]
    sentence, drop = es.screener_guidance(results, [PE, ROE])
    assert drop is not None and drop.index == 0
    assert "P/E Ratio < 15" in sentence


def test_the_only_filter_is_never_offered_for_removal():
    """Dropping it leaves a screen that screens nothing, which is not a
    useful next step."""
    sentence, drop = es.screener_guidance([FakeResult("A", [False])], [PE])
    assert drop is None
    assert sentence


def test_all_missing_data_is_reported_as_such_not_as_a_strict_filter():
    results = [FakeResult("A", [None, None]), FakeResult("B", [None, None])]
    sentence, drop = es.screener_guidance(results, [PE, ROE])
    assert drop is None
    assert "could not be computed" in sentence


def test_no_criteria_and_no_results_each_say_so():
    assert es.screener_guidance([FakeResult("A", [])], [])[1] is None
    sentence, drop = es.screener_guidance([], [PE, ROE])
    assert drop is None and "universe" in sentence


def test_guidance_never_claims_a_count_it_did_not_measure():
    """Every number in the sentence must be traceable to the results."""
    results = [FakeResult("A", [False, True]), FakeResult("B", [True, True])]
    sentence, _ = es.screener_guidance(results, [PE, ROE])
    for number in re.findall(r"\d+", sentence):
        assert int(number) <= len(results) or number == "15", sentence


# --- the call sites -----------------------------------------------------------

@pytest.fixture(scope="module")
def source() -> str:
    return FINANCE.read_text(encoding="utf-8")


def _call(source: str, key: str) -> str:
    """The empty_states.render(...) call carrying this widget key."""
    match = re.search(
        r"empty_states\.render\((?:[^()]|\([^()]*\))*?key=\"" + key + r"\"(?:[^()]|\([^()]*\))*?\)",
        source, re.S)
    assert match, f"no empty_states.render call with key={key!r}"
    return match.group(0)


@pytest.mark.parametrize("key", [
    "empty_watchlist_add", "empty_screener_relax", "empty_rt_first_alert",
])
def test_each_of_the_three_states_offers_an_action(source, key):
    """"Each with action button" — a headline and guidance alone is what
    the app already had."""
    call = _call(source, key)
    assert "action_label=" in call


def test_the_watchlist_action_adds_the_symbol_on_screen(source):
    """Streamlit cannot focus the "Add ticker" box, so a button that only
    pointed at it would appear to do nothing."""
    start = source.index('key="empty_watchlist_add"')
    body = source[start:start + 1200]
    assert "add_ticker(" in body and "save_watchlist_store" in body


def test_the_first_alert_needs_no_invented_threshold(source):
    """A price target requires a number, and choosing one for someone is a
    recommendation wearing a default."""
    import realtime_alerts

    assert realtime_alerts.FIRST_ALERT_TRIGGER == "sma_cross_bullish"
    start = source.index('key="empty_rt_first_alert"')
    body = source[start:start + 1400]
    assert "RT_FIRST_ALERT_TRIGGER" in body
    assert "threshold=" not in body, "the seeded rule must not set a threshold"


def test_nothing_references_ticker_symbol_before_it_exists(source):
    """finance.py is a script: ticker_symbol is assigned by the sidebar
    around line 2400, and the real-time alerts fragment runs several
    hundred lines earlier. Using it there raised NameError — and only for
    a user with NO alert rules yet, which is to say every new user, so it
    was invisible to a signed-in developer with rules already saved.
    """
    tree = ast.parse(source)
    assigned_at = next(
        node.lineno for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ticker_symbol"
                for t in node.targets))
    early = sorted({node.lineno for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and node.id == "ticker_symbol"
                    and node.lineno < assigned_at})
    assert not early, (
        f"ticker_symbol is assigned at line {assigned_at} but referenced "
        f"earlier at {early}")


def test_the_first_alert_falls_back_to_session_state_not_script_locals(source):
    """A fragment with run_every re-executes ALONE on its timer, so
    script-level locals are whatever the last full run happened to leave
    behind. session_state is the durable read."""
    start = source.index('key="empty_rt_first_alert"')
    body = source[source.rindex("_rt_seed_ticker = ", 0, start):start]
    assert 'st.session_state.get("rt_new_ticker")' in body
    assert 'st.session_state.get("ticker_input")' in body
    # ...and ticker_input itself is not seeded until far below this
    # fragment, so the first run of a session has neither key.
    assert "CHART_DEFAULTS.default_ticker" in body


def test_no_ticker_means_no_action_button(source):
    """An action that cannot complete is worse than no action."""
    call = _call(source, "empty_rt_first_alert")
    assert "if _rt_seed_ticker else None" in call


def test_the_first_alert_reruns_the_whole_app_not_the_fragment(source):
    """The Active Rules list is rendered outside this fragment, so a
    fragment-scoped rerun would redraw the empty state and leave the new
    rule invisible."""
    start = source.index('key="empty_rt_first_alert"')
    body = source[start:start + 1400]
    assert 'st.rerun(scope="app")' in body


def test_the_screener_action_drops_only_the_named_filter(source):
    start = source.index('key="empty_screener_relax"')
    body = source[start:start + 1500]
    assert "!= _screener_drop.index" in body
    assert '_screener_rerun' in body


def test_the_dropped_filter_triggers_a_fresh_run(source):
    """Otherwise the user sees the same stale table and has to find the
    Run button again."""
    assert re.search(r'st\.session_state\.pop\("_screener_rerun", False\)', source)
    assert "screener_run_clicked = True" in source


def test_the_screener_state_does_not_claim_the_table_is_empty(source):
    """Zero passing is not zero results — every ticker is still listed
    with its own pass/fail, and that evidence stays on screen."""
    call = _call(source, "empty_screener_relax")
    assert "No stocks passed every filter" in call
    assert "No stocks match" not in call


def test_screener_criteria_rows_get_distinct_widget_identities(source):
    """Pre-existing crash found while verifying this task's screener state.

    Streamlit hashes (label, options, index, help) to identify an UNKEYED
    widget, and does not include label_visibility — so two criteria rows
    whose operator box offered the same options at the same index shared
    one auto-ID and raised StreamlitDuplicateElementId, taking the whole
    screener down. "+ Add Filter" appends rsi/"<"/30.0, which collides
    with any existing numeric "<" row, i.e. the default P/E filter.

    key= is the obvious fix and is exactly the thing that must not be used
    on these four widgets — see the comment above them. Unique labels are
    the fix instead, and every row past the first collapses its label so
    nothing changes on screen.
    """
    start = source.index("_row_suffix = ")
    block = source[start:source.index("_screener_remove_index = _i", start)]
    for widget in ("Metric", "Op", "Value", "Threshold"):
        assert f'f"{widget}{{_row_suffix}}"' in block, (
            f"the {widget} widget does not vary its label per row")
    assert 'key=' not in block.replace('key=f"screener_remove_', ''), (
        "the criteria widgets must stay unkeyed")


def test_the_row_suffix_is_empty_for_the_first_row(source):
    """Row 0's label is the one actually displayed, so it must read
    "Metric", not "Metric 1"."""
    assert '_row_suffix = "" if _i == 0 else f" {_i + 1}"' in source


def test_the_risk_alert_panel_did_not_get_a_dead_empty_state(source):
    """Its rules are SEEDED with defaults on first use, so a "you have no
    rules" state there would never render — the same "gated on empty,
    fires never" trap the first-sign-in prompt fell into."""
    assert '"risk_alert_rules"] = [' in source        # still seeded
    assert 'key="empty_risk_alert' not in source
