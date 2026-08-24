"""The data-quality badge in the sticky header.

Two things are worth pinning here. First that the badge cannot silently
lie: an unrecognised grade must not fall through to green, and the badge
must not depend on grade_icon, which has returned "" for every grade
since the emoji removal and would render a colourless blank.

Second that the badge is a POPOVER. The symbol header is a single
st.markdown string, and custom HTML cannot call back into Streamlit — so
a badge rendered into that line could never satisfy "click to expand".
The quick-stats strip in the same block already documents this; a test
stops the badge being "simplified" into markup later.
"""
import ast
import re
from pathlib import Path

import pytest

import data_quality


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


# --- the grade scale ----------------------------------------------------------

def test_every_grade_the_module_can_produce_has_a_colour():
    """The task named three colours; the scale has four levels. A grade
    with no colour would fall through to the unknown case and read as
    Poor on perfectly good data."""
    produced = {data_quality._grade(score) for score in range(0, 101)}
    assert produced == set(data_quality.GRADE_COLOURS), produced


def test_the_four_grades_are_four_distinct_colours():
    """Collapsing Fair into a neighbour to fit "green/yellow/red" would
    misreport data that is readable but not dependable."""
    assert len(set(data_quality.GRADE_COLOURS.values())) == 4


def test_an_unknown_grade_is_never_green():
    """An unrecognised grade is not evidence of good data."""
    assert data_quality.grade_colour("Nonsense") == data_quality.GRADE_COLOURS["Poor"]
    assert data_quality.grade_colour("") == data_quality.GRADE_COLOURS["Poor"]
    assert "not recognised" in data_quality.grade_meaning("Nonsense")


def test_the_colours_run_in_the_right_direction():
    """Excellent must not be the red one."""
    assert data_quality.grade_colour("Excellent") != data_quality.grade_colour("Poor")
    assert data_quality.grade_colour("Excellent").lower() == "#00ea77"
    assert data_quality.grade_colour("Poor").lower() == "#ef4444"


@pytest.mark.parametrize("grade", ["Excellent", "Good", "Fair", "Poor"])
def test_every_grade_says_what_it_means_for_the_numbers(grade):
    """A score alone does not tell a reader whether to trust a ratio."""
    meaning = data_quality.grade_meaning(grade)
    assert len(meaning) > 40 and meaning.endswith(".")


def test_grade_boundaries_are_unchanged():
    """The badge's colours are pinned to these cutoffs; moving one
    silently recolours every ticker."""
    assert data_quality._grade(90) == "Excellent"
    assert data_quality._grade(89.9) == "Good"
    assert data_quality._grade(75) == "Good"
    assert data_quality._grade(74.9) == "Fair"
    assert data_quality._grade(55) == "Fair"
    assert data_quality._grade(54.9) == "Poor"


# --- the badge in the app -----------------------------------------------------

def _badge_block(code_only: bool = False) -> str:
    """The badge's source. `code_only` strips Python comments.

    Needed because the comments deliberately name what the code must NOT
    do — stBaseButton-secondary, grade_icon — so a raw substring scan
    finds the prose explaining the rule and fails on correct code."""
    start = FINANCE.index("# --- Data quality badge")
    block = FINANCE[start:FINANCE.index("with _pm_slot:", start)]
    if not code_only:
        return block
    return "\n".join(line for line in block.splitlines()
                      if not line.lstrip().startswith("#"))


def test_the_badge_is_a_popover_not_markup():
    """The symbol header is one st.markdown string and custom HTML cannot
    call back into Streamlit, so a badge rendered there could never
    expand. Same constraint the quick-stats strip documents."""
    block = _badge_block()
    assert "st.popover(" in block
    assert 'key="dq_badge"' in block


def test_the_badge_renders_inside_the_sticky_header():
    """"Prominent" means always on screen. Below the fold in a tab is
    where it already was."""
    header_start = FINANCE.index("with symbol_header_container.container():")
    tabs_start = FINANCE.index("= st.tabs([")
    badge_at = FINANCE.index("# --- Data quality badge")
    assert header_start < badge_at < tabs_start


def test_the_badge_colour_comes_from_the_grade():
    block = _badge_block()
    assert "data_quality.grade_colour(" in block
    assert "_dq_colour" in block
    # ...and is applied to the control, not just computed.
    assert re.search(r"st-key-dq_badge.*?\{\{", block, re.S)


def test_the_badge_rule_out_specifies_the_main_secondary_rule():
    """finance.py paints main-area secondaries at (0,2,1) with
    !important. The obvious [class*="st-key-dq_badge"] button is (0,1,1)
    and loses outright — measured live, the badge came back with the
    ordinary grey border and slate text.

    Note the attribute: a POPOVER trigger is data-testid="stPopoverButton"
    with kind="secondary", so button_roles' stBaseButton-secondary form
    would not have matched it either."""
    block = _badge_block(code_only=True)
    assert ('[data-testid="stMain"] [class*="st-key-dq_badge"] '
            'button[kind="secondary"]') in block
    assert "stBaseButton-secondary" not in block, (
        "a popover trigger does not carry that testid")


def test_the_badge_does_not_use_the_dead_grade_icon():
    """_grade_icon has returned "" for every grade since the emoji
    removal; a badge built on it would render blank."""
    assert all(data_quality._grade_icon(s) == "" for s in (95, 80, 60, 20))
    assert "grade_icon" not in _badge_block(code_only=True)


def test_the_report_is_assessed_once_and_reused():
    """Two assessments of the same bundle could not disagree, but they
    could drift if one call site gained an argument the other did not."""
    assert FINANCE.count("assess_data_quality(") == 1
    assert "quality = data_quality_report" in FINANCE


def test_the_detail_section_still_exists_for_the_full_lists():
    """The badge carries what a reader needs to act; the exhaustive
    field-by-field lists stay where there is room, and the badge says so."""
    assert "Data Quality Report —" in FINANCE
    assert "Overview → Data Quality Report" in _badge_block()


def test_the_badge_reports_missing_freshness_rather_than_implying_it():
    """staleness_days is Optional. Printing "0 days ago" for a ticker that
    reported no filing date would be a fabricated freshness."""
    block = _badge_block()
    assert "staleness_days is not None" in block
    assert "freshness could not be measured" in block


def test_there_is_no_fake_refresh_timer():
    """The report is recomputed from the same bundle the page is built
    from on every run, so it is never staler than the figures beside it.
    A fragment would re-render a value closed over from this run and only
    LOOK live."""
    block = _badge_block()
    assert "st.fragment" not in block
    assert "run_every" not in block


# --- what the score MEANS per asset class -------------------------------------
#
# The badge read "18/100 · Poor" for every ETF. That was not a low score,
# it was a category error, and it was the same non-fact deducted three
# separate times. These pin the shape of the fix.

import copy
import datetime

import pandas as pd

import asset_class
from data_loader import STATEMENT_LABELS, MacroBundle, TickerBundle


def _bundle(ticker="SPY", quote_type="ETF", *, info=None, price_days_old=0,
            warnings=(), errors=()):
    """A bundle with a current price series and a complete fund profile,
    which individual tests then damage one field at a time."""
    base = {"quoteType": quote_type}
    base.update({f: 1.0 for f in data_quality.FUND_PROFILE_FIELDS})
    if info is not None:
        base = dict(base, **info)
    last = datetime.date.today() - datetime.timedelta(days=price_days_old)
    history = pd.DataFrame(
        {"Close": [1.0, 2.0]},
        index=pd.to_datetime([last - datetime.timedelta(days=1), last]))
    return TickerBundle(ticker=ticker, info=base, price_history=history,
                        warnings=list(warnings), errors=list(errors))


class _Check:
    def __init__(self, name, required, present):
        self.name, self.required, self.present = name, required, present


class _Stmt:
    """A statement whose fields are all ABSENT — which is exactly the
    state a fund is in, and the state that scored 0% completeness. A stub
    with no checks at all would score 100% and quietly prove nothing."""
    def __init__(self):
        self.statement_name = "Balance Sheet"
        self.checks = [_Check("Total Assets", True, False),
                       _Check("Current Assets", True, False),
                       _Check("Retained Earnings", False, False)]
        self.missing_required = ["Total Assets", "Current Assets"]
        self.missing_optional = ["Retained Earnings"]
        self.is_valid = False


class _Validation:
    statements = [_Stmt()]


class _Std:
    """Stands in for a fund's standardized financials: no statements at
    all, which is the state that produced 0% completeness."""
    ticker = "SPY"
    validation = _Validation()
    most_recent_quarter = None


def _assess(bundle, klass=None):
    return data_quality.assess_data_quality(
        _Std(), bundle, MacroBundle(), klass=klass)


def test_a_fund_is_not_graded_poor_for_lacking_filings_it_never_makes():
    """THE BUG. Every ETF scored 17.5/100 "Poor" — SPY and TLT alike —
    because the module measured a fund against corporate filings. A fund
    has no income statement; that absence is a fact about the instrument,
    not evidence its data is untrustworthy."""
    report = _assess(_bundle(warnings=[
        f"SPY: {label} data unavailable." for label in STATEMENT_LABELS]))

    assert report.asset_class == asset_class.ETF
    assert report.grade == "Excellent"
    assert report.score >= 90, report.score
    # The statement-shaped figures are still computed for the detail
    # panel, but must carry no weight.
    assert report.required_completeness_pct == 0.0
    assert "required_completeness_pct" not in [d.key for d in report.dimensions]


def test_the_same_absence_is_not_deducted_for_three_times():
    """Required completeness, optional completeness AND fetch reliability
    were each docked for the one fact that a fund files nothing. The
    statement warnings must not touch fetch reliability for a class with
    no filings."""
    clean = _assess(_bundle())
    with_statement_warnings = _assess(_bundle(warnings=[
        f"SPY: {label} data unavailable." for label in STATEMENT_LABELS]))

    assert with_statement_warnings.fetch_reliability_score == 100.0
    assert with_statement_warnings.score == clean.score


def test_an_equity_is_still_charged_for_a_missing_statement():
    """The exemption is per class, not global — a stock that failed to
    return its balance sheet has a real data problem, and hiding that
    would be the opposite bug."""
    report = _assess(_bundle(quote_type="EQUITY", warnings=[
        f"AAPL: {label} data unavailable." for label in STATEMENT_LABELS]))

    assert report.asset_class == asset_class.EQUITY
    assert report.fetch_reliability_score == 70.0


def test_a_statement_warning_is_recognised_from_data_loaders_own_labels():
    """Matched against STATEMENT_LABELS rather than a literal copied into
    this module, so rewording the message in data_loader cannot silently
    turn a fund's non-filing back into a fetch failure."""
    for label in STATEMENT_LABELS:
        assert data_quality._is_statement_warning(f"SPY: {label} data unavailable.")
    assert not data_quality._is_statement_warning(
        "SPY: price history failed after 3 attempt(s)")
    # And the loader must actually emit those labels.
    loader = (ROOT / "data_loader.py").read_text(encoding="utf-8")
    assert "STATEMENT_LABELS" in loader
    assert '_load_statement_field(stock, "financials", ticker, _income_label' in loader


def test_every_classes_weights_sum_to_one():
    """Otherwise a class is quietly graded out of less than 100 and can
    never reach Excellent no matter how good its data is."""
    for klass in list(data_quality.DIMENSIONS) + ["not-a-real-class"]:
        dims = data_quality.dimensions_for(klass)
        total = sum(d.weight for d in dims)
        assert abs(total - 1.0) < 1e-9, (klass, total)
        assert dims, klass


def test_an_unrecognised_class_is_not_graded_as_an_equity():
    """Assuming filings exist for something we could not identify is what
    produced the original bug."""
    dims = data_quality.dimensions_for("not-a-real-class")
    assert "required_completeness_pct" not in [d.key for d in dims]
    assert data_quality.dimensions_for(asset_class.UNKNOWN) == dims


def test_every_dimension_key_is_a_real_field_on_the_report():
    """A dimension whose key does not name a field would raise at render
    time, in the sticky header, on every run."""
    report = _assess(_bundle())
    for klass in data_quality.DIMENSIONS:
        for dim in data_quality.dimensions_for(klass):
            assert hasattr(report, dim.key), (klass, dim.key)
            assert isinstance(getattr(report, dim.key), float)


def test_the_fund_score_still_falls_when_fund_data_is_actually_missing():
    """A badge that always reads Excellent is as useless as one that
    always read Poor. The dimensions a fund DOES have must bite."""
    full = _assess(_bundle())
    assert full.score == 100.0

    thin = _assess(_bundle(info={"netExpenseRatio": None, "totalAssets": None}))
    assert thin.score < full.score
    assert set(thin.missing_fund_fields) == {"netExpenseRatio", "totalAssets"}

    stale = _assess(_bundle(price_days_old=21))
    assert stale.price_history_score < 50
    assert stale.score < full.score

    broken = _assess(_bundle(errors=["SPY: price history failed"]))
    assert broken.fetch_reliability_score == 70.0
    assert broken.score < full.score


def test_price_history_is_scored_on_currency_not_on_length():
    """The user picks the date range, so a short series is a choice, not a
    defect — but a series whose last bar is weeks old means the
    technicals and risk panels are drawing stale conclusions."""
    short_and_current = _bundle(price_days_old=0)
    assert len(short_and_current.price_history) == 2
    assert _assess(short_and_current).price_history_score == 100.0

    long_and_stale = _bundle(price_days_old=21)
    long_and_stale.price_history = pd.DataFrame(
        {"Close": range(500)},
        index=pd.to_datetime([datetime.date.today() - datetime.timedelta(days=21 + i)
                              for i in range(499, -1, -1)]))
    assert _assess(long_and_stale).price_history_score < 50


def test_an_empty_price_series_scores_zero_not_a_default():
    bundle = _bundle()
    bundle.price_history = pd.DataFrame()
    report = _assess(bundle)
    assert report.price_history_score == 0.0
    assert report.price_age_days is None


def test_a_weekend_does_not_make_the_price_series_look_stale():
    """Equities and ETFs skip weekends while crypto does not, so the
    threshold is tolerant rather than counting expected trading days."""
    for days in (0, 1, 2, 3, 4):
        assert _assess(_bundle(price_days_old=days)).price_history_score == 100.0


# --- the badge and the panel must explain what they measured -------------------

def test_the_badge_lists_the_reports_own_dimensions():
    """Hardcoding four rows under the score is what let a fund's badge
    explain its number with "Required fields: 0%" — a figure that carried
    no weight in it."""
    assert "for _dq_dim in data_quality_report.dimensions:" in FINANCE
    assert "getattr(data_quality_report, _dq_dim.key)" in FINANCE
    # The hardcoded row dict must not come back.
    assert '"Required fields": f"{data_quality_report.required_completeness_pct' not in FINANCE


def test_a_non_equity_is_told_what_it_was_scored_on():
    """A reader who assumes the score means the same thing for every
    symbol will compare a fund's against a stock's."""
    assert "quality.scored_on" in FINANCE
    assert "data_quality_report.scored_on" in FINANCE


def test_the_badge_is_given_the_class_the_rest_of_the_page_uses():
    """Classifying twice invites the badge and the panels disagreeing
    about what the symbol is."""
    assert "macro_bundle, klass=asset_kind)" in FINANCE


def test_scored_on_names_every_weighted_dimension():
    report = _assess(_bundle())
    for dim in report.dimensions:
        assert dim.label.lower() in report.scored_on


# --- the words under the score --------------------------------------------

def test_a_fund_is_not_told_its_required_fields_are_present():
    """The Excellent line read "Every required field is present and
    current" under a score that had not looked at a single filing — a
    sentence about evidence that was never weighed."""
    fund = data_quality.grade_meaning("Excellent", asset_class.ETF)
    assert "required field" not in fund
    assert "filing" not in fund.lower()
    # ...while an equity keeps the filing wording, which is accurate there.
    equity = data_quality.grade_meaning("Excellent", asset_class.EQUITY)
    assert "required field" in equity
    assert fund != equity


def test_every_grade_has_wording_for_both_kinds_of_instrument():
    grades = ["Excellent", "Good", "Fair", "Poor"]
    for grade in grades:
        for klass in (asset_class.EQUITY, asset_class.ETF, asset_class.CRYPTO):
            text = data_quality.grade_meaning(grade, klass)
            assert text and "not recognised" not in text, (grade, klass)
    assert "not recognised" in data_quality.grade_meaning("Nonsense", asset_class.ETF)


def test_the_badge_defaults_to_the_equity_wording():
    """Called with one argument it must behave exactly as before."""
    for grade in ("Excellent", "Good", "Fair", "Poor"):
        assert (data_quality.grade_meaning(grade)
                == data_quality.grade_meaning(grade, asset_class.EQUITY))


def test_the_class_is_named_with_a_grammatical_article():
    """Lowercasing the label produced "a etf / fund" on the live badge —
    wrong article, and an acronym written as a word."""
    assert asset_class.with_article(asset_class.ETF) == "an ETF / fund"
    assert asset_class.with_article(asset_class.EQUITY) == "a stock"
    assert asset_class.with_article(asset_class.INDEX) == "an index"
    assert asset_class.with_article(asset_class.CRYPTO) == "a cryptocurrency"
    assert asset_class.with_article("nonsense") == "an unrecognised instrument"
    for klass in [s.key for s in asset_class.SPECS]:
        assert asset_class.with_article(klass).startswith(("a ", "an "))


def test_no_render_site_lowercases_a_class_label_by_hand():
    """The article helper exists so this cannot come back in a third place."""
    for name in ("finance.py", "asset_class.py", "data_quality.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "label(quality.asset_class).lower()" not in source, name
        assert "label(data_quality_report.asset_class).lower()" not in source, name
    assert "with_article" in FINANCE


def test_the_not_applicable_note_reads_grammatically_too():
    note = asset_class.unavailable_note(asset_class.ETF, asset_class.FUNDAMENTALS)
    assert note.startswith("Not applicable to an ETF / fund.")
    assert "a etf" not in note


def test_uncounted_warnings_are_labelled_as_uncounted():
    """The detail header reads "0 issue(s)" above a list of three
    statement warnings. Without a word of explanation the reader cannot
    tell whether they counted."""
    assert "did NOT count against this score" in FINANCE
    assert "data_quality._is_statement_warning(w)" in FINANCE


def test_a_crypto_is_not_charged_with_missing_fund_fields():
    """The badge read "6 field-level issue(s)" for BTC-USD — one per fund
    profile field, on an instrument with no expense ratio or fund family
    to report. The original category error, one level down."""
    report = _assess(_bundle(ticker="BTC-USD", quote_type="CRYPTOCURRENCY",
                             info={f: None for f in data_quality.FUND_PROFILE_FIELDS}))
    assert report.asset_class == asset_class.CRYPTO
    assert report.missing_fund_fields          # still recorded...
    assert report.issue_count == 0             # ...but not counted


def test_a_fund_IS_charged_with_missing_fund_fields():
    """The exemption is per class, not blanket — a fund that fails to
    report its expense ratio has a real gap."""
    report = _assess(_bundle(info={"netExpenseRatio": None}))
    assert report.issue_count == 1


def test_an_equity_counts_its_statement_gaps_as_before():
    report = _assess(_bundle(quote_type="EQUITY"))
    # The stub statement has 2 missing required + 1 missing optional.
    assert report.issue_count == 3


def test_a_real_fetch_error_counts_for_every_class():
    for quote_type in ("EQUITY", "ETF", "CRYPTOCURRENCY"):
        report = _assess(_bundle(quote_type=quote_type, errors=["boom"]))
        assert report.issue_count >= 1, quote_type


def test_both_render_sites_use_the_one_issue_count():
    """Two hand-rolled sums drift apart; one of them already had the
    crypto bug in it."""
    assert "data_quality_report.issue_count" in FINANCE
    assert "detail_issue_count = quality.issue_count" in FINANCE
