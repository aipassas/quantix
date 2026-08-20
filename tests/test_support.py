"""Tests for support.py — searchable help and outbound reports.

Two things carry real weight here:

1. THE CORPUS IS ASSEMBLED, NOT COPIED. If a metric definition were ever
   duplicated into the FAQ, the tooltip and the help article for the same
   metric could drift apart and both would look authoritative. The index
   tests pin that down.

2. DIAGNOSTICS ARE OPT-IN AND INSPECTABLE. Log lines name the tickers
   someone has been researching. A report must never carry them unless
   they were asked for, and the exact text must be obtainable before
   sending so the UI can show it.
"""
import support
from config import SUPPORT
from metric_help import CHART_HELP, GLOSSARY
from support import (
    FAQ,
    HelpArticle,
    browse,
    build_index,
    categories,
    compose_report,
    diagnostics_snapshot,
    search,
    send_report,
    title_for,
)


class _FakeSender:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def __call__(self, to_address, subject, body):
        if self.fail:
            return False, "SMTP exploded"
        self.sent.append((to_address, subject, body))
        return True, None


# --- the corpus ---------------------------------------------------------------

def test_index_covers_the_faq_and_every_existing_help_entry():
    """Assembled from metric_help rather than re-typed — one fact, one
    source, so a tooltip and its help article cannot disagree."""
    index = build_index()
    assert len(index) == len(FAQ) + len(GLOSSARY) + len(CHART_HELP)


def test_metric_articles_reuse_the_tooltip_text_verbatim():
    index = {a.id: a for a in build_index()}
    for key, text in GLOSSARY.items():
        assert index[f"metric_{key}"].body == text


def test_chart_articles_reuse_the_chart_help_text_verbatim():
    index = {a.id: a for a in build_index()}
    for key, text in CHART_HELP.items():
        assert index[f"chart_{key}"].body == text


def test_no_faq_answer_duplicates_a_glossary_definition():
    """A second copy of a definition is a second thing to keep correct."""
    definitions = {t.strip() for t in GLOSSARY.values()}
    for article in FAQ:
        assert article.body.strip() not in definitions


def test_article_ids_are_unique():
    ids = [a.id for a in build_index()]
    assert len(ids) == len(set(ids))


def test_every_article_has_a_title_and_a_body():
    for article in build_index():
        assert article.title.strip(), article.id
        assert len(article.body.strip()) > 40, article.id


def test_no_title_leaks_a_raw_slug():
    """Search results are read by a person. `var_historical` is a lookup
    key, not a name, and showing it would look like a bug."""
    for article in build_index():
        assert "_" not in article.title, f"{article.id} -> {article.title}"


def test_every_glossary_key_gets_a_readable_title():
    for key in list(GLOSSARY) + list(CHART_HELP):
        title = title_for(key)
        assert title and "_" not in title, key


def test_faq_ids_are_prefixed_so_they_cannot_collide_with_metric_ids():
    assert all(a.id.startswith("faq_") for a in FAQ)


def test_the_ui_starter_questions_all_exist():
    """finance.py shows a fixed list of starter questions by id. A typo or
    a renamed FAQ entry would silently show nothing at all."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "finance.py").read_text()
    block = re.search(r"SUPPORT_STARTERS = \((.*?)\)", source, re.S)
    assert block, "SUPPORT_STARTERS not found in finance.py"
    starters = re.findall(r'"([a-z_]+)"', block.group(1))
    assert starters
    known = {a.id for a in build_index()}
    assert not (set(starters) - known), f"unknown starter ids: {set(starters) - known}"


# --- search -------------------------------------------------------------------

def test_empty_query_returns_nothing():
    assert search("") == ()
    assert search("   ") == ()


def test_a_metric_name_finds_its_definition():
    hits = search("sharpe")
    assert hits and hits[0].title == "Sharpe Ratio"


def test_a_question_finds_its_faq_entry():
    hits = search("settings disappeared")
    assert hits and hits[0].id == "faq_settings_vanished"


def test_search_is_case_insensitive():
    assert search("SHARPE")[0].id == search("sharpe")[0].id


def test_every_term_must_match():
    """Requiring all terms is what keeps a small corpus precise: without
    it, "dividend yield" would return everything mentioning yield."""
    hits = search("dividend yield")
    assert all("dividend" in a.haystack and "yield" in a.haystack for a in hits)


def test_a_term_matching_nothing_returns_nothing():
    assert search("sharpe zzzzznotaword") == ()
    assert search("zzzzznotaword") == ()


def test_title_matches_outrank_body_mentions():
    hits = search("drawdown")
    assert "drawdown" in hits[0].title.lower()


def test_results_are_capped():
    hits = search("the", limit=3)
    assert len(hits) <= 3


def test_default_limit_comes_from_config():
    assert len(search("a")) <= SUPPORT.search_results_shown


def test_search_accepts_an_injected_index():
    tiny = (HelpArticle(id="x", title="Only One", body="a body about kangaroos", category="Test"),)
    assert search("kangaroos", tiny)[0].id == "x"
    assert search("sharpe", tiny) == ()


def test_punctuation_in_the_query_is_ignored():
    assert search("sharpe?!")[0].title == "Sharpe Ratio"


# --- browsing -----------------------------------------------------------------

def test_categories_are_listed_in_corpus_order_without_duplicates():
    names = categories()
    assert len(names) == len(set(names))
    assert "Metric" in names and "Chart" in names


def test_browse_filters_to_one_category():
    metrics = browse("Metric")
    assert len(metrics) == len(GLOSSARY)
    assert all(a.category == "Metric" for a in metrics)


# --- diagnostics --------------------------------------------------------------

def test_diagnostics_include_environment_and_log_lines():
    text = diagnostics_snapshot(log_lines=3, log_reader=lambda n: [f"line {i}" for i in range(n)])
    assert "Python:" in text and "Platform:" in text
    assert "line 0" in text and "line 2" in text


def test_diagnostics_accept_extra_context():
    text = diagnostics_snapshot(log_reader=lambda n: [], extra={"Current ticker": "AAPL"})
    assert "Current ticker: AAPL" in text


def test_diagnostics_survive_an_unreadable_log():
    """A missing or locked log file must not stop someone reporting a bug
    — the environment details alone are still worth having."""
    def boom(n):
        raise OSError("no such file")
    text = diagnostics_snapshot(log_reader=boom)
    assert "Python:" in text
    assert "could not read the log" in text


def test_diagnostics_are_returned_as_text_so_the_ui_can_show_them_first():
    """The privacy property: the exact string that would be sent has to be
    obtainable before sending, or 'review what will be attached' is not
    something the UI can honestly offer."""
    text = diagnostics_snapshot(log_reader=lambda n: ["ticker=SECRETCO"])
    assert isinstance(text, str)
    assert "SECRETCO" in text


# --- composing ----------------------------------------------------------------

def test_a_valid_report_composes():
    report, err = compose_report("Bug report", "Chart is blank", "It renders empty for AAPL.")
    assert err is None
    assert report.category == "Bug report" and report.subject == "Chart is blank"


def test_subject_and_body_are_required():
    assert compose_report("Question", "", "body")[0] is None
    assert compose_report("Question", "subject", "   ")[0] is None


def test_lengths_are_capped():
    assert compose_report("Question", "x" * (SUPPORT.max_subject_chars + 1), "body")[0] is None
    assert compose_report("Question", "subject", "x" * (SUPPORT.max_body_chars + 1))[0] is None


def test_a_bad_reply_address_is_rejected():
    report, err = compose_report("Question", "s", "b", reply_to="not-an-email")
    assert report is None and err


def test_reply_address_is_optional():
    report, err = compose_report("Question", "s", "b", reply_to="")
    assert err is None and report.reply_to == ""


def test_category_defaults_when_blank():
    report, _ = compose_report("", "s", "b")
    assert report.category == SUPPORT.categories[0]


def test_report_carries_no_diagnostics_unless_supplied():
    """The opt-in property at the data layer: composing without
    diagnostics must produce a body with none in it."""
    report, _ = compose_report("Question", "s", "b")
    assert report.diagnostics == ""
    assert "diagnostics" not in report.as_email_body().lower()


def test_supplied_diagnostics_are_clearly_labelled_in_the_body():
    report, _ = compose_report("Bug report", "s", "b", diagnostics="Python: 3.12")
    body = report.as_email_body()
    assert "diagnostics" in body.lower() and "Python: 3.12" in body


def test_the_email_body_carries_the_category_and_reply_address():
    report, _ = compose_report("Bug report", "s", "the description", reply_to="me@example.com")
    body = report.as_email_body()
    assert "Bug report" in body and "me@example.com" in body and "the description" in body


def test_a_missing_reply_address_is_stated_rather_than_left_blank():
    report, _ = compose_report("Question", "s", "b")
    assert "not supplied" in report.as_email_body()


# --- sending ------------------------------------------------------------------

def test_sending_uses_the_configured_destination():
    report, _ = compose_report("Question", "How do I", "please help")
    sender = _FakeSender()
    ok, err = send_report(report, sender, to_address="support@example.com")
    assert ok and err is None
    to_address, subject, body = sender.sent[0]
    assert to_address == "support@example.com"
    assert "[Quantix Question]" in subject and "How do I" in subject
    assert "please help" in body


def test_no_destination_is_explained_rather_than_treated_as_a_failure():
    """A stock install genuinely has nowhere to send. The message must
    tell the user what to do with their words, not imply a malfunction."""
    report, _ = compose_report("Question", "s", "b")
    ok, err = send_report(report, _FakeSender(), to_address="")
    assert ok is False
    assert "copy the report" in err.lower()


def test_a_send_failure_is_returned_not_raised():
    report, _ = compose_report("Question", "s", "b")
    ok, err = send_report(report, _FakeSender(fail=True), to_address="support@example.com")
    assert ok is False and "SMTP" in err


def test_destination_configured_reflects_config(monkeypatch):
    """SupportConfig is frozen, so the config object can't be mutated —
    replace() a copy and rebind the module-level name, the same pattern
    the effective-threshold accessors use."""
    import dataclasses

    assert support.is_destination_configured() is False
    monkeypatch.setattr(
        support, "SUPPORT", dataclasses.replace(SUPPORT, support_address="help@example.com"))
    assert support.is_destination_configured() is True


def test_a_configured_address_is_used_when_none_is_passed():
    import dataclasses

    report, _ = compose_report("Question", "s", "b")
    sender = _FakeSender()
    configured = dataclasses.replace(SUPPORT, support_address="desk@example.com")
    original = support.SUPPORT
    support.SUPPORT = configured
    try:
        ok, err = send_report(report, sender)
    finally:
        support.SUPPORT = original
    assert ok and sender.sent[0][0] == "desk@example.com"


def test_the_shipped_default_has_no_support_address():
    """Shipping a real address would send strangers' bug reports somewhere
    nobody reads. The honest default is none."""
    assert SUPPORT.support_address == ""
