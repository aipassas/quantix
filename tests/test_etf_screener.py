"""The ETF screener.

The judgement worth pinning is what this refuses to offer. The task asks
for tracking error, down-capture, beta and Sharpe filters; none of those
figures is on a screener row, and a filter labelled "Sharpe > 1" that
quietly means something else is worse than no filter at all. It also asks
for PostgreSQL, Elasticsearch and a nightly job to search 250 rows, in an
app whose entire persistence is local JSON and which has no worker.

The other load-bearing distinction is between a fund that FAILED a filter
and one that could not be JUDGED by it. Counting the second as the first
would silently shrink every result set by whatever the source omitted —
78 of the funds carry no P/E.
"""
import pathlib

import pytest

import etf_screener as es
from screener import CATEGORICAL_OPERATORS, OPERATORS


ROOT = pathlib.Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


def row(symbol="AAA", **kw):
    base = dict(name=f"{symbol} Fund", price=100.0, expense_ratio_pct=0.1,
                assets=1e9, pe_ratio=20.0, dividend_yield_pct=2.0,
                return_1y_pct=10.0, return_3y_pct=8.0, return_ytd_pct=5.0)
    base.update(kw)
    return es.EtfRow(symbol=symbol, **base)


# --- missing data is not failure ----------------------------------------------

def test_a_fund_that_does_not_report_a_metric_is_set_aside_not_failed():
    """78 of the 250 funds carry no P/E. Counting those as failures would
    shrink every result set by whatever the source happened to omit."""
    passed, unjudged = es.run(
        [row("HASPE", pe_ratio=10.0), row("NOPE", pe_ratio=None)],
        [es.EtfCriterion("pe_ratio", "<", 15.0)])
    assert [m.row.symbol for m in passed] == ["HASPE"]
    assert [m.row.symbol for m in unjudged] == ["NOPE"]


def test_the_unjudged_fund_names_what_it_could_not_be_judged_on():
    _, unjudged = es.run([row(pe_ratio=None)],
                         [es.EtfCriterion("pe_ratio", "<", 15.0)])
    assert unjudged[0].unmeasured == ("Price / Earnings",)


def test_a_genuine_failure_is_simply_absent():
    passed, unjudged = es.run([row(pe_ratio=99.0)],
                              [es.EtfCriterion("pe_ratio", "<", 15.0)])
    assert passed == [] and unjudged == []


def test_a_failure_beats_a_gap_when_both_apply():
    """Failing one filter excludes the fund even if another was
    unmeasurable — it is out on the evidence available."""
    passed, unjudged = es.run(
        [row(pe_ratio=None, expense_ratio_pct=5.0)],
        [es.EtfCriterion("pe_ratio", "<", 15.0),
         es.EtfCriterion("expense_ratio_pct", "<", 0.2)])
    assert passed == [] and unjudged == []


# --- operators ----------------------------------------------------------------

def test_the_operator_vocabulary_is_shared_with_the_equity_screener():
    """Two definitions of "<" is how they come to mean different things."""
    assert set(es.operators_for("pe_ratio")) == set(OPERATORS)
    # Text gets its OWN pair: the equity screener's categorical operators
    # mean equality, and this metric is a substring search.
    assert set(es.operators_for("name")) == set(es.TEXT_OPERATORS)
    assert set(es.operators_for("name")) != set(CATEGORICAL_OPERATORS)


def test_a_text_metric_gets_text_operators():
    assert "contains" in es.operators_for("name")
    assert "<" not in es.operators_for("name")


def test_name_matching_is_case_insensitive():
    passed, _ = es.run([row("VOO", name="Vanguard S&P 500 ETF")],
                       [es.EtfCriterion("name", "contains", "vanguard")])
    assert len(passed) == 1


def test_an_unknown_metric_cannot_silently_pass_everything():
    """It returned None, which was not counted as unmeasured because the
    key has no label — so the fund fell through to `passed` and a typo'd
    filter matched EVERYTHING instead of nothing."""
    passed, unjudged = es.run([row()], [es.EtfCriterion("not_a_metric", "<", 1)])
    assert passed == []
    assert len(unjudged) == 1
    assert "unknown filter" in unjudged[0].unmeasured[0]


# --- what is deliberately not offered -----------------------------------------

@pytest.mark.parametrize("absent", ["tracking", "sharpe", "beta", "down_capture",
                                    "down-capture", "inception"])
def test_unavailable_filters_are_not_offered(absent):
    """A filter that silently means something else is worse than none."""
    for metric in es.METRICS:
        assert absent not in metric.key.lower()
        assert absent not in metric.label.lower()


def test_the_gaps_are_written_down_for_the_reader():
    text = " ".join(es.UNSUPPORTED_FILTERS).lower()
    for topic in ("tracking error", "beta", "sharpe", "family", "inception"):
        assert topic in text, topic


def test_the_app_shows_those_gaps():
    assert "etf_screener.UNSUPPORTED_FILTERS" in FINANCE
    assert "cannot filter on" in FINANCE


def test_no_database_was_introduced():
    """The task specifies PostgreSQL, Elasticsearch and a nightly job to
    search 250 rows. This app's whole persistence is local JSON and it has
    no worker; the table is cached in memory instead."""
    import ast

    source = (ROOT / "etf_screener.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for engine in ("sqlite3", "psycopg", "psycopg2", "sqlalchemy", "elasticsearch"):
        assert engine not in imported, engine


# --- presets ------------------------------------------------------------------

def test_the_presets_only_use_metrics_that_exist():
    for preset in es.PRESETS:
        for criterion in preset.criteria:
            assert criterion.metric in es.METRICS_BY_KEY, criterion
            assert criterion.operator in es.operators_for(criterion.metric)


def test_the_two_presets_needing_absent_data_are_handled_honestly():
    """"Sector Leaders" (beta, Sharpe) is absent entirely rather than
    approximated; "High Dividend" drops its tracking-error clause and
    says so."""
    assert "Sector Leaders" not in es.PRESETS_BY_NAME
    high = es.PRESETS_BY_NAME["High Dividend"]
    assert len(high.criteria) == 1
    assert "tracking error" in high.description.lower()


def test_the_faithful_presets_match_the_spec():
    low = es.PRESETS_BY_NAME["Low-Cost Index"]
    assert es.describe(low.criteria[0]) == "Expense ratio < 0.2%"
    assert "100.0M" in es.describe(low.criteria[1])
    value = es.PRESETS_BY_NAME["Value Plays"]
    assert {c.metric for c in value.criteria} == {"pe_ratio", "dividend_yield_pct"}


# --- search -------------------------------------------------------------------

def test_a_symbol_match_outranks_a_name_match():
    """Someone typing letters that form a ticker almost always means it."""
    rows = [row("XYZ", name="Vanguard Growth"), row("VUG", name="Something Else")]
    assert es.search(rows, "vu")[0].symbol == "VUG"


def test_search_finds_by_name_too():
    rows = [row("VOO", name="Vanguard S&P 500")]
    assert es.search(rows, "vanguard")[0].symbol == "VOO"


def test_an_empty_query_returns_nothing_rather_than_everything():
    assert es.search([row()], "") == ()
    assert es.search([row()], "   ") == ()


def test_search_respects_its_limit():
    rows = [row(f"AA{i}") for i in range(30)]
    assert len(es.search(rows, "aa", limit=5)) == 5


# --- formatting ---------------------------------------------------------------

def test_a_missing_cell_is_stated_never_zero():
    assert es.format_value("pe_ratio", None) == "Not reported"
    assert es.format_value("expense_ratio_pct", float("nan")) == "Not reported"


def test_money_is_compacted():
    assert es.format_value("assets", 1_500_000_000) == "$1.5B"
    assert es.format_value("assets", 250_000_000) == "$250.0M"


def test_percentages_carry_their_unit():
    assert es.format_value("expense_ratio_pct", 0.095) == "0.10%"
    assert es.format_value("return_1y_pct", 12.345) == "12.3%"


def test_describe_reads_as_a_sentence():
    assert es.describe(es.EtfCriterion("dividend_yield_pct", ">", 3.0)) == \
        "Dividend yield > 3%"


# --- the universe -------------------------------------------------------------

def test_a_row_without_a_symbol_is_dropped():
    assert es._to_row({"longName": "Nameless"}) is None


def test_the_row_reads_percentages_from_the_screener_not_the_info_dict():
    """The per-fund info dict mixes conventions — its ytdReturn is a
    percent while threeYearAverageReturn is a fraction. The screener row
    is consistent, which is why this module reads only that."""
    import re

    source = (ROOT / "etf_screener.py").read_text(encoding="utf-8")
    # Comments explain why the info dict is NOT used and name its fields,
    # so scan the code only.
    code = "\n".join(l for l in source.splitlines()
                      if not l.lstrip().startswith("#"))
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    assert "fiftyTwoWeekChangePercent" in code
    assert "threeYearAverageReturn" not in code


def test_the_universe_is_capped_and_cached():
    assert es.UNIVERSE_SIZE <= 250, "one request returns at most 250"
    assert es.CACHE_TTL_SECONDS == 300, "the task asks for a five-minute cache"


def test_the_etf_criteria_widgets_cannot_collide_with_the_stock_screener():
    """Streamlit identifies an unkeyed widget by hashing (label, options,
    index, help). The ETF screener's row 0 was labelled "Op" with the
    same options and index as the STOCK screener's row 0 "Op", so the two
    collided across sections and took the page down. Numbering rows was
    not enough; the labels are prefixed."""
    for label in ("ETF metric", "ETF op", "ETF threshold", "ETF value"):
        assert f'f"{label}{{_etfs_suffix}}"' in FINANCE, label
    # ...and the bare forms must not come back.
    for bare in ('f"Metric{_etfs_suffix}"', 'f"Op{_etfs_suffix}"'):
        assert bare not in FINANCE, bare


def _row(symbol, **over):
    base = dict(name=f"{symbol} Fund", price=58.4, expense_ratio_pct=0.03,
                assets=5.1e10, pe_ratio=26.8, dividend_yield_pct=3.89,
                return_1y_pct=-2.1, return_3y_pct=0.4)
    base.update(over)
    return es.EtfRow(symbol=symbol, **base)


def test_a_column_no_fund_in_the_result_set_reports_stays_blank():
    """THE TRIGGER IS THE RESULT SET, NOT THE ROW. pandas already turns a
    None among floats into NaN, so a column stays float64 as long as ONE
    fund reports it — which is why this looked fine until a filter
    narrowed the table to bond funds, none of which report a P/E. An
    all-None column is object dtype, and Streamlit draws the literal
    string "None" into every cell of it. That shipped, and a reader would
    have copied "None" out into a spreadsheet."""
    frame = es.results_frame([
        es.EtfMatch(row=_row("VGIT", pe_ratio=None), unmeasured=()),
        es.EtfMatch(row=_row("VCSH", pe_ratio=None), unmeasured=()),
    ])

    assert str(frame["P/E"].dtype).startswith("float"), frame.dtypes.to_dict()
    assert frame["P/E"].isna().all()
    # Never the string "None", on screen or in the file.
    assert "None" not in frame.to_csv(index=False)


def test_a_gap_is_blank_and_never_a_zero():
    """A zero is silently absorbed into whatever average the reader builds
    over the CSV; a blank is skipped by both AVERAGE and SUM."""
    import math

    frame = es.results_frame([
        es.EtfMatch(row=_row("VGIT", pe_ratio=None), unmeasured=()),
        es.EtfMatch(row=_row("VOO", pe_ratio=26.8), unmeasured=()),
    ])
    assert math.isnan(frame["P/E"].iloc[0])
    assert frame["P/E"].iloc[1] == 26.8
    assert (frame["P/E"] == 0).sum() == 0
    assert frame.to_csv(index=False).splitlines()[1].split(",")[5] == ""


def test_the_results_frame_keeps_every_column_and_its_order():
    """Named explicitly, not derived from TABLE_COLUMNS — a test that
    compares the frame to the same tuple it was built from cannot notice
    a column going missing."""
    expected = ["Symbol", "Name", "Price", "ER %", "AUM", "P/E",
                "Yield %", "1Y %", "3Y %"]
    assert list(es.results_frame([]).columns) == expected
    assert [c[0] for c in es.TABLE_COLUMNS] == expected
    # An empty screen still yields the full header, so the CSV of a
    # no-match result is still readable.
    assert es.results_frame([]).to_csv(index=False).strip() == ",".join(expected)


def test_percent_columns_are_not_given_streamlits_percent_preset():
    """Streamlit's "percent" format multiplies the stored number by 100,
    exactly as an Excel percent format does. Every figure in this app is
    already percent-valued, so the preset would render an 0.03% expense
    ratio as 3.00%."""
    formats = {label: fmt for label, _a, fmt, _h in es.TABLE_COLUMNS}
    for label in ("ER %", "Yield %", "1Y %", "3Y %"):
        assert formats[label] in ("%.2f%%", "%.1f%%"), label
    assert "percent" not in [f for f in formats.values() if f]


def test_the_numeric_columns_are_the_ones_that_get_a_format():
    formats = {label: fmt for label, _a, fmt, _h in es.TABLE_COLUMNS}
    assert set(es.NUMERIC_COLUMNS) == {l for l, f in formats.items() if f}
    assert "Symbol" not in es.NUMERIC_COLUMNS
    assert "Name" not in es.NUMERIC_COLUMNS


def test_every_column_that_can_be_missing_says_a_blank_is_not_a_zero():
    """Streamlit draws its own muted "None" into a null numeric cell and
    gives no API to reword it — measured: it does that for any null in a
    numeric column, with or without a column_config. So the tooltip has
    to carry the meaning, or the reader is left to guess whether a fund
    with no P/E has a P/E of zero."""
    helps = {label: h for label, _a, _f, h in es.TABLE_COLUMNS}
    for label in ("ER %", "AUM", "P/E", "Yield %", "1Y %", "3Y %"):
        assert "not zero" in helps[label], label
    # And the tooltips must actually reach the widget, along with the
    # format — a column_config that silently dropped either would leave
    # the table looking exactly as it did before.
    config = es.column_config()
    assert set(config) == set(es.NUMERIC_COLUMNS)
    for label, col in config.items():
        assert col["help"], label
        assert col["type_config"]["format"], label


def test_the_table_is_built_once_for_both_the_screen_and_the_csv():
    """The CSV must be the table the reader is looking at. Building the
    frame twice invites the two drifting apart."""
    assert "etf_screener.results_frame(" in FINANCE
    assert "column_config=etf_screener.column_config()" in FINANCE
    # The hand-rolled DataFrame it replaced must not come back.
    assert '"Yield %": m.row.dividend_yield_pct' not in FINANCE
