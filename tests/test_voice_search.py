"""Natural-language queries into screener criteria.

The failure this file mostly guards is the SILENT one: a query that
half-parses and screens on less than the reader asked for. A wrong
criterion is visible; a missing one is not, and the results still look
like an answer.
"""
import pytest

import screener
import voice_search as vs


def _texts(result):
    return [c.text for c in result.criteria]


def _spans(result):
    return [u.span for u in result.unparsed]


# --- the task's own example ---------------------------------------------------

def test_the_example_from_the_task_parses_exactly():
    """"Show me tech stocks with ROE > 15%" — the sentence the task
    names. Sector and metric both, nothing left over."""
    result = vs.parse("Show me tech stocks with ROE > 15%")
    assert result.ok and result.complete
    assert _texts(result) == ["Sector is Technology",
                              "Return on Equity > 15%"]


@pytest.mark.parametrize("query", [
    "Show me tech stocks with ROE > 15%",
    "show me tech stocks with roe above 15%",
    "SHOW ME TECH STOCKS WITH ROE OVER 15 PERCENT",
    "tech stocks, ROE greater than 15%.",
    "find technology companies with return on equity above 15",
])
def test_the_same_screen_survives_rephrasing(query):
    """Dictation and typing produce different casing and punctuation for
    the same request; they have to land in the same place."""
    result = vs.parse(query)
    assert result.ok, query
    assert set(_texts(result)) == {"Sector is Technology",
                                   "Return on Equity > 15%"}, query


# --- every example we advertise must work -------------------------------------

@pytest.mark.parametrize("query", vs.EXAMPLE_QUERIES)
def test_every_advertised_example_parses_cleanly(query):
    """An example in the empty state that does not work is worse than no
    example — it teaches the reader the feature is broken."""
    result = vs.parse(query)
    assert result.ok, query
    assert not result.unparsed, (query, _spans(result))


# --- operators ----------------------------------------------------------------

@pytest.mark.parametrize("phrase,operator", [
    ("above", ">"), ("over", ">"), ("greater than", ">"), ("more than", ">"),
    ("higher than", ">"), ("exceeds", ">"),
    ("below", "<"), ("under", "<"), ("less than", "<"), ("lower than", "<"),
    ("at least", ">="), ("no less than", ">="), ("minimum", ">="),
    ("at most", "<="), ("no more than", "<="), ("up to", "<="),
])
def test_each_comparison_word_maps_to_its_operator(phrase, operator):
    result = vs.parse(f"P/E {phrase} 20")
    assert _texts(result) == [f"P/E Ratio {operator} 20"], phrase


def test_no_more_than_is_not_read_as_more_than():
    """The longest phrase has to win, or a cap becomes a floor — the
    exact inversion that would screen for the opposite of the request."""
    assert _texts(vs.parse("P/E no more than 20")) == ["P/E Ratio <= 20"]
    assert _texts(vs.parse("P/E more than 20")) == ["P/E Ratio > 20"]


def test_symbols_work_as_well_as_words():
    assert _texts(vs.parse("P/E < 20")) == ["P/E Ratio < 20"]
    assert _texts(vs.parse("roe >= 15")) == ["Return on Equity >= 15%"]


def test_a_bare_number_reads_as_a_floor():
    """"ROE 15%" is how people say "at least 15" — but the assumption is
    worth pinning, because it is an assumption."""
    assert _texts(vs.parse("ROE 15%")) == ["Return on Equity > 15%"]


# --- numbers ------------------------------------------------------------------

def test_a_percent_sign_does_NOT_scale_the_number():
    """Every percent-valued metric in this app is stored percent-valued,
    so 15% is 15.0. Dividing by a hundred here is the same 100x error
    the Excel and Streamlit percent formats make, from the other side."""
    result = vs.parse("ROE above 15%")
    assert result.criteria[0].threshold == pytest.approx(15.0)


def test_percent_spelled_out_is_the_same_number():
    assert vs.parse("ROE above 15 percent").criteria[0].threshold == pytest.approx(15.0)


def test_a_dollar_amount_keeps_its_value():
    result = vs.parse("price under $100")
    assert result.criteria[0].threshold == pytest.approx(100.0)
    assert _texts(result) == ["Share Price < $100"]


def test_decimals_survive():
    assert vs.parse("debt to equity below 1.5").criteria[0].threshold == pytest.approx(1.5)


def test_a_spoken_number_word_is_understood():
    """Recognisers return "fifteen" as often as "15"."""
    assert vs.parse("ROE above fifteen").criteria[0].threshold == pytest.approx(15.0)


def test_a_negative_threshold_is_allowed():
    """Max drawdown and revenue growth are legitimately negative — a
    parser that dropped the sign would screen for the opposite."""
    result = vs.parse("max drawdown above -30%")
    assert result.criteria[0].threshold == pytest.approx(-30.0)


# --- several criteria in one sentence -----------------------------------------

def test_two_metrics_each_keep_their_own_number():
    """Without a window per metric the first one grabs the last number,
    which is a wrong screen that still looks like an answer."""
    result = vs.parse("P/E under 20 and ROE above 15")
    assert _texts(result) == ["P/E Ratio < 20", "Return on Equity > 15%"]


def test_three_criteria_including_a_sector():
    result = vs.parse(
        "Financials with debt to equity below 1 and price under $100")
    assert set(_texts(result)) == {"Sector is Financial Services",
                                   "Debt/Equity < 1", "Share Price < $100"}


def test_criteria_convert_to_the_screeners_own_type():
    """The whole point of the task: reuse the existing filter engine
    rather than build new query logic."""
    result = vs.parse("tech stocks with ROE above 15%")
    converted = result.screen_criteria()
    assert all(isinstance(c, screener.ScreenCriterion) for c in converted)
    assert {c.metric for c in converted} == {"sector", "roe_pct"}
    for criterion in converted:
        assert criterion.metric in screener.METRICS_BY_KEY
        assert criterion.operator in screener.operators_for(criterion.metric)


# --- sectors ------------------------------------------------------------------

@pytest.mark.parametrize("said,expected", [
    ("tech", "Technology"), ("software", "Technology"),
    ("healthcare", "Healthcare"), ("pharma", "Healthcare"),
    ("banks", "Financial Services"), ("financials", "Financial Services"),
    ("energy", "Energy"), ("utilities", "Utilities"),
    ("real estate", "Real Estate"), ("reits", "Real Estate"),
    ("telecom", "Communication Services"),
    ("consumer staples", "Consumer Defensive"),
    ("consumer discretionary", "Consumer Cyclical"),
    ("materials", "Basic Materials"), ("industrials", "Industrials"),
])
def test_spoken_sector_names_map_to_the_providers_labels(said, expected):
    result = vs.parse(f"{said} stocks")
    assert _texts(result) == [f"Sector is {expected}"], said


def test_every_sector_the_screener_offers_can_be_asked_for():
    """A sector in the picker with no spoken form is one a voice user
    simply cannot reach."""
    reachable = set(vs.SECTOR_SYNONYMS)
    assert set(screener.SECTORS) <= reachable


@pytest.mark.parametrize("phrase", ["excluding energy", "not energy",
                                    "other than energy", "except energy"])
def test_negation_flips_a_sector_to_is_not(phrase):
    result = vs.parse(f"stocks {phrase}")
    assert _texts(result) == ["Sector is not Energy"], phrase


def test_a_plain_sector_is_not_negated():
    assert _texts(vs.parse("energy stocks")) == ["Sector is Energy"]


# --- the silent-failure guards ------------------------------------------------

def test_a_comparative_without_a_number_is_REPORTED_not_dropped():
    """THE FAILURE THAT MATTERS. "cheap tech stocks with high ROE" can
    only screen on Technology. Returning that alone would run a screen
    the reader believes is narrower, and hand back a list they would
    read as cheap and high-return."""
    result = vs.parse("show me cheap tech stocks with high ROE")
    assert _texts(result) == ["Sector is Technology"]
    assert not result.complete
    joined = " ".join(_spans(result))
    assert "roe" in joined.lower() and "cheap" in joined.lower()


def test_the_hint_for_a_missing_threshold_invents_no_number():
    """Suggesting "Debt/Equity above 15" would be nonsense for a ratio
    that runs under 2, and picking a sensible number per metric means
    inventing the bounds this app refuses to invent."""
    result = vs.parse("banks with low debt to equity")
    reason = result.unparsed[0].reason
    assert "<n>" in reason
    assert not any(ch.isdigit() for ch in reason)


def test_a_query_with_nothing_screenable_says_so():
    result = vs.parse("find me some good stocks")
    assert not result.ok
    assert result.summary == vs.NO_CRITERIA_FOUND
    assert result.screen_criteria() == []


def test_an_empty_query_is_not_an_error():
    for query in ("", "   ", None):
        result = vs.parse(query)
        assert not result.ok and not result.unparsed


def test_the_summary_names_what_was_not_used():
    result = vs.parse("cheap tech stocks")
    assert "Sector is Technology" in result.summary
    assert "Not used" in result.summary
    assert "cheap" in result.summary


def test_a_complete_parse_reports_nothing_unused():
    result = vs.parse("tech stocks with ROE above 15%")
    assert result.complete
    assert "Not used" not in result.summary


# --- vocabulary integrity -----------------------------------------------------

def test_every_metric_synonym_points_at_a_real_metric():
    """A synonym for a renamed metric stops matching silently rather
    than failing."""
    for key in vs.METRIC_SYNONYMS:
        assert key in screener.METRICS_BY_KEY, key


def test_every_sector_synonym_points_at_a_real_sector():
    for name in vs.SECTOR_SYNONYMS:
        assert name in screener.SECTORS, name


def test_every_numeric_metric_can_be_asked_for_by_name():
    """A metric with no spoken form is one voice search cannot reach.
    Sector is excluded: it is asked for by naming the sector itself."""
    for key, spec in screener.METRICS_BY_KEY.items():
        if spec.kind == "categorical":
            continue
        assert key in vs.METRIC_SYNONYMS, f"{key} has no spoken form"


def test_each_metric_is_reachable_through_its_own_synonyms():
    """Declaring a synonym is not the same as it winning the match —
    "vol" must not be eaten by "volatility", and "pe" must not lose to
    "price to earnings"."""
    for key, synonyms in vs.METRIC_SYNONYMS.items():
        if key == "sector":
            continue
        for synonym in synonyms:
            result = vs.parse(f"{synonym} above 5")
            assert result.criteria, f"{synonym!r} matched nothing"
            assert result.criteria[0].metric == key, (
                f"{synonym!r} matched {result.criteria[0].metric}, not {key}")


def test_a_longer_metric_name_beats_a_shorter_one_inside_it():
    assert vs.parse("price to book below 3").criteria[0].metric == "price_to_book"
    assert vs.parse("price below 3").criteria[0].metric == "price"
    assert vs.parse("dividend yield above 2").criteria[0].metric == "dividend_yield_pct"


def test_a_metric_word_inside_another_word_does_not_fire():
    """Word boundaries: "it" sits inside "with", and an unanchored match
    turned every query containing "with" into a Technology screen."""
    result = vs.parse("with P/E under 20")
    assert _texts(result) == ["P/E Ratio < 20"]


# --- the disclosure -----------------------------------------------------------

def test_the_off_device_audio_disclosure_exists_and_is_specific():
    """The browser's speech API ships captured audio to the vendor. This
    app tells its reader that analysis runs on their machine, so the one
    step that leaves has to say so."""
    note = vs.SPEECH_SENDS_AUDIO_OFF_DEVICE
    assert "browser" in note and "servers" in note
    assert "Typing" in note


def test_the_vocabulary_description_lists_rather_than_summarises():
    """A user guessing at the vocabulary is the main way this feature
    disappoints: they say something reasonable, nothing matches, and the
    panel looks broken."""
    text = vs.describe_vocabulary()
    assert "Return on Equity" in text and "Technology" in text
    assert "above" in text and "below" in text


# --- the UI wiring ------------------------------------------------------------

import ast
import pathlib

FINANCE = (pathlib.Path(__file__).resolve().parent.parent
           / "finance.py").read_text(encoding="utf-8")


def _voice_block():
    start = FINANCE.index("# --- Ask in words (Voice Search)")
    return FINANCE[start:FINANCE.index("def _screener_apply_template")]


def test_the_parse_feeds_the_screeners_OWN_criteria_state():
    """The task asks to reuse the existing filter engine rather than
    build new query logic. Writing into screener_criteria is what makes
    Run Screen behave identically to a hand-built screen."""
    block = _voice_block()
    assert 'st.session_state["screener_criteria"]' in block
    assert "voice_search.parse(" in block


def test_no_second_screening_engine_was_written():
    """A parallel path would drift from the builder's behaviour."""
    import voice_search

    names = [n for n in dir(voice_search) if not n.startswith("_")]
    assert not [n for n in names
                if "run" in n.lower() or "screen_one" in n.lower()
                or "execute" in n.lower()]


def test_the_component_keys_match_the_widgets_it_drives():
    """The iframe hands the transcript back by finding these widgets in
    the parent document. A key renamed on one side and not the other
    leaves a mic button that transcribes into nothing."""
    import voice_search

    block = _voice_block()
    assert 'key="voice_query"' in block
    assert 'key="voice_apply"' in block
    html = voice_search.listener_html(query_key="voice_query",
                                      apply_key="voice_apply")
    assert '"voice_query"' in html and '"voice_apply"' in html
    assert 'query_key="voice_query"' in block
    assert 'apply_key="voice_apply"' in block


def test_the_transcript_is_delivered_with_the_native_setter():
    """React tracks an input's value on the node, so a plain assignment
    is invisible to Streamlit — the setter plus an input event is what
    it actually notices."""
    import voice_search

    html = voice_search.listener_html()
    assert "getOwnPropertyDescriptor" in html
    assert 'dispatchEvent' in html and '"input"' in html


def test_dictation_availability_is_DETECTED_not_assumed():
    """A button that is advertised and never fires is worse than none:
    Firefox has no webkitSpeechRecognition, a Chromium build may have no
    speech backend, and the microphone may simply be refused."""
    import voice_search

    html = voice_search.listener_html()
    assert "if (!Recognition)" in html
    assert "isSecureContext" in html
    for failure in ("not-allowed", "service-not-allowed", "audio-capture",
                    "network", "no-speech"):
        assert failure in html, failure


def test_a_blocked_microphone_is_reported_BEFORE_the_button_is_pressed():
    """Making the reader press a dead button to discover it is refused
    is the same shape as advertising a shortcut that never fires."""
    import voice_search

    html = voice_search.listener_html()
    assert 'navigator.permissions.query({ name: "microphone" })' in html
    assert '"denied"' in html
    # Safari does not implement the query for microphone, so the click
    # path must still be the backstop.
    assert ".catch(" in html


def test_every_dictation_failure_points_at_typing_instead():
    """The typed box is the fallback, so every dead end has to name it."""
    import voice_search

    html = voice_search.listener_html()
    # Each error branch and each hard-disable path mentions typing.
    assert html.count("type the query") >= 5


def test_the_off_device_audio_disclosure_is_rendered_beside_the_button():
    block = _voice_block()
    assert "SPEECH_SENDS_AUDIO_OFF_DEVICE" in block


def test_the_component_renders_at_a_visible_height():
    """A zero-height iframe is the convention for an invisible listener;
    this one has a button in it and must be seen."""
    import voice_search

    assert voice_search.component_height() > 0
    assert "height=voice_search.component_height()" in _voice_block()


def test_an_empty_query_does_not_clear_the_builder():
    """Pressing Search with nothing typed must not wipe the filters the
    user has already set up by hand."""
    block = _voice_block()
    assert '_vs_apply and (_vs_query or "").strip()' in block


def test_a_query_that_parsed_nothing_leaves_the_builder_alone():
    """Only a successful parse writes criteria; a failed one warns."""
    block = _voice_block()
    apply_index = block.index("if _vs_result.ok:")
    write_index = block.index('st.session_state["screener_criteria"] = [')
    assert write_index > apply_index, (
        "criteria are written outside the ok branch")


def test_the_query_box_takes_no_value_and_key_together():
    """Passing both makes Streamlit restore the old value on the next
    run, silently undoing what was just dictated."""
    block = _voice_block()
    start = block.index("_vs_query = st.text_input(")
    call = block[start:block.index(")", block.index("help=", start))]
    assert "key=" in call
    assert "value=" not in call
