"""Importing holdings or a watchlist from a spreadsheet or pasted text.

The fixtures here are shaped like real broker exports rather than like
tidy CSVs, because that is where every failure lives: an account
preamble above the header, "$1,234.56" in a numeric column, a cash line
and a totals line at the bottom, and a cost column that might be a
per-share price or a position total.

Two of these tests exist for ambiguities that are SILENT when wrong —
they produce a plausible number, not an exception — so a weak test here
would let a 10x cost error or a month-shifted purchase date through.
"""
import datetime
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spreadsheet_import as si


SCHWAB_EXPORT = '''"Positions for account Individual ...123 as of 09/01/2026"
""

"Symbol","Description","Quantity","Price","Cost Basis","Market Value"
"AAPL","APPLE INC","10","$319.97","$1,502.50","$3,199.70"
"BRK.B","BERKSHIRE HATHAWAY CL B","2","$450.00","$800.00","$900.00"
"Cash & Cash Investments","","--","--","--","$1,204.11"
"Account Total","","","","$2,302.50","$5,303.81"
'''


# --- reading a real export ----------------------------------------------------

def test_a_broker_preamble_does_not_crash_the_parser():
    """Measured: pd.read_csv on this raises ParserError "Expected 1
    fields in line 4, saw 3". Not a bad result — a traceback."""
    with pytest.raises(Exception):
        pd.read_csv(io.StringIO(SCHWAB_EXPORT))

    frame, error = si.read_table(SCHWAB_EXPORT.encode(), "positions.csv")
    assert error == ""
    assert list(frame.columns)[:3] == ["Symbol", "Description", "Quantity"]
    assert len(frame) == 4


def test_the_header_row_is_found_past_the_preamble():
    assert si.find_header_row(SCHWAB_EXPORT) == 3


def test_a_file_with_no_preamble_starts_at_row_zero():
    assert si.find_header_row("Symbol,Shares\nAAPL,10\n") == 0


def test_an_empty_file_says_so():
    frame, error = si.read_table(b"", "x.csv")
    assert frame is None and "empty" in error.lower()


def test_an_oversized_file_is_refused_with_the_limit(monkeypatch):
    import dataclasses
    monkeypatch.setattr(si, "SPREADSHEET_IMPORT",
                        dataclasses.replace(si.SPREADSHEET_IMPORT,
                                            max_upload_bytes=10))
    frame, error = si.read_table(b"x" * 50, "big.csv")
    assert frame is None and "larger than" in error


def test_legacy_xls_is_refused_with_the_fix_not_a_traceback():
    """Reading .xls needs xlrd, which is not installed — measured, pandas
    raises ImportError. An unsupported format should teach, not crash."""
    frame, error = si.read_table(b"\xd0\xcf\x11\xe0", "old.xls")
    assert frame is None
    assert "xlrd" in error and ".xlsx" in error


def test_an_unreadable_file_returns_a_sentence_not_an_exception():
    frame, error = si.read_table(b"\x00\x01\x02\x03", "junk.xlsx")
    assert frame is None
    assert error and "ParserError" not in error


def test_excel_round_trips_including_real_date_cells(tmp_path):
    path = tmp_path / "h.xlsx"
    pd.DataFrame({
        "Symbol": ["AAPL"], "Quantity": [10], "Cost/Share": [150.25],
        "Date Acquired": [datetime.date(2024, 1, 15)],
    }).to_excel(path, index=False)

    frame, error = si.read_table(path.read_bytes(), "h.xlsx")
    assert error == ""
    mapping = si.guess_mapping(list(frame.columns), frame)
    row = si.build_preview(frame, mapping, need_position=True).importable[0]
    assert row.purchase_date == datetime.date(2024, 1, 15)


def test_the_utf8_bom_needs_no_special_handling():
    """Checked before writing code for it: pandas 3.0.3 already strips
    Excel's BOM. Pinned so nobody re-adds a workaround, and so the
    assumption is retested if pandas changes."""
    frame, error = si.read_table('﻿Symbol,Shares\nAAPL,10\n'.encode(), "b.csv")
    assert error == ""
    assert list(frame.columns) == ["Symbol", "Shares"]


# --- numbers ------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("$1,234.56", 1234.56),
    ("1,234.56", 1234.56),
    ("(123.45)", -123.45),        # accounting negative
    ("$0.00", 0.0),
    ("10", 10.0),
    (10, 10.0),
    (10.5, 10.5),
])
def test_broker_number_formats_parse(raw, expected):
    assert si.parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["--", "N/A", "n/a", "", "   ", "none", "—", None])
def test_an_absent_number_is_none_not_zero(raw):
    """A zero is silently absorbed into any total the reader builds;
    None is skipped. Same rule the Excel export follows for blanks."""
    assert si.parse_number(raw) is None


def test_a_boolean_is_not_a_number():
    assert si.parse_number(True) is None


def test_nan_is_none():
    assert si.parse_number(float("nan")) is None


# --- tickers ------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("AAPL", "AAPL"), ("aapl", "AAPL"), ("  AAPL  ", "AAPL"),
    ("$AAPL", "AAPL"), ('"AAPL"', "AAPL"), ("AAPL*", "AAPL"),
    ("AAPL.", "AAPL"), ("AAPL (Apple Inc.)", "AAPL"),
])
def test_decoration_comes_off_a_symbol(raw, expected):
    assert si.normalise_ticker(raw) == expected


@pytest.mark.parametrize("symbol", ["BRK.B", "VWCE.DE", "BF-B", "^GSPC", "EURUSD=X"])
def test_the_interior_of_a_symbol_is_never_touched(symbol):
    """A dot or hyphen inside a symbol is load-bearing: BRK.B is a share
    class, VWCE.DE is a listing venue. This codebase already learned that
    dropping a venue suffix turns a working symbol into a 404."""
    assert si.normalise_ticker(symbol) == symbol
    assert si.looks_like_ticker(symbol) is True


@pytest.mark.parametrize("raw", [
    "Cash & Cash Investments", "Cash and Cash Investments", "Account Total",
    "Total", "TOTALS", "Subtotal", "Pending Activity", "--", "",
])
def test_a_non_position_row_is_recognised(raw):
    assert si.is_non_position_row(raw) is True


def test_a_real_ticker_is_not_mistaken_for_a_total_row():
    """Matched on the whole cell, not as a substring."""
    assert si.is_non_position_row("TOT") is False   # TotalEnergies
    assert si.is_non_position_row("CASHX") is False


@pytest.mark.parametrize("bad", ["Some Company Name Ltd", "12345678901234567890"])
def test_something_that_is_not_a_ticker_is_rejected(bad):
    assert si.looks_like_ticker(si.normalise_ticker(bad)) is False


# --- the two silent ambiguities -----------------------------------------------

def test_a_cost_total_is_divided_by_the_share_count():
    """The 10x error. 10 shares costing $1,502.50 in total is $150.25 a
    share; reading the total as a per-share price would record a
    $15,025 position."""
    frame, _ = si.read_table(SCHWAB_EXPORT.encode(), "p.csv")
    mapping = si.guess_mapping(list(frame.columns), frame)
    assert mapping.cost_is_total is True
    preview = si.build_preview(
        frame, si.__dict__["ColumnMapping"](
            ticker="Symbol", shares="Quantity", cost_basis="Cost Basis",
            cost_is_total=True, fallback_date=datetime.date(2024, 1, 15)),
        need_position=True)
    apple = preview.importable[0]
    assert apple.ticker == "AAPL"
    assert apple.cost_basis == pytest.approx(150.25)


def test_a_per_share_cost_is_left_alone():
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [10],
                          "Cost/Share": ["$150.25"]})
    mapping = si.ColumnMapping(ticker="Symbol", shares="Quantity",
                               cost_basis="Cost/Share", cost_is_total=False,
                               fallback_date=datetime.date(2024, 1, 15))
    assert si.build_preview(frame, mapping, need_position=True
                            ).importable[0].cost_basis == pytest.approx(150.25)


@pytest.mark.parametrize("header,expected", [
    ("Cost Basis", True), ("Cost Basis Total", True), ("Market Value", True),
    ("Cost/Share", False), ("Cost Per Share", False), ("Average Cost", False),
    ("Purchase Price", False), ("Price Paid", False),
])
def test_the_cost_column_heading_is_read_for_per_share_or_total(header, expected):
    assert si.cost_is_total_guess(header) is expected


def test_a_per_share_marker_beats_a_total_marker_in_the_same_heading():
    """"Total cost basis per share" would otherwise match "total" and
    invert the reading."""
    assert si.cost_is_total_guess("Total Cost Basis Per Share") is False


def test_one_unambiguous_row_settles_the_whole_date_column():
    assert si.date_convention(["01/02/2024", "15/03/2024"]) == si.DATE_DAY_FIRST
    assert si.date_convention(["01/02/2024", "12/25/2024"]) == si.DATE_MONTH_FIRST


def test_a_column_that_nothing_settles_is_undecidable_not_guessed():
    """01/02/2024 really is two different dates, and picking one silently
    moves a purchase by a month."""
    assert si.date_convention(["01/02/2024", "03/04/2024"]) == si.DATE_UNDECIDABLE


def test_a_contradictory_column_is_undecidable_too():
    assert si.date_convention(["15/01/2024", "12/25/2024"]) == si.DATE_UNDECIDABLE


def test_an_iso_date_column_needs_no_decision():
    assert si.date_convention(["2024-01-15", "2024-03-02"]) == si.DATE_UNDECIDABLE
    assert si.parse_date("2024-01-15") == datetime.date(2024, 1, 15)


def test_an_undecidable_column_warns_the_reader():
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [1],
                          "Cost/Share": [1.0], "Date": ["01/02/2024"]})
    mapping = si.ColumnMapping(ticker="Symbol", shares="Quantity",
                               cost_basis="Cost/Share", purchase_date="Date")
    preview = si.build_preview(frame, mapping, need_position=True)
    assert any("either way round" in w for w in preview.warnings)


@pytest.mark.parametrize("raw,day_first,expected", [
    ("01/02/2024", False, datetime.date(2024, 1, 2)),
    ("01/02/2024", True, datetime.date(2024, 2, 1)),
    ("2024-01-15", False, datetime.date(2024, 1, 15)),
    ("Jan 15, 2024", False, datetime.date(2024, 1, 15)),
])
def test_a_date_is_read_the_way_the_caller_says(raw, day_first, expected):
    assert si.parse_date(raw, day_first) == expected


def test_an_excel_serial_date_is_converted():
    """Excel's epoch is 1899-12-30, thanks to its deliberate 1900
    leap-year bug."""
    assert si.parse_date(45000) == datetime.date(2023, 3, 15)


def test_a_number_that_is_not_a_plausible_serial_date_is_not_one():
    assert si.parse_date(10) is None
    assert si.parse_date(999999) is None


def test_unreadable_date_text_is_none_not_today():
    assert si.parse_date("sometime last year") is None
    assert si.parse_date("") is None


# --- mapping ------------------------------------------------------------------

def test_a_broker_header_maps_itself():
    frame, _ = si.read_table(SCHWAB_EXPORT.encode(), "p.csv")
    mapping = si.guess_mapping(list(frame.columns), frame)
    assert mapping.ticker == "Symbol"
    assert mapping.shares == "Quantity"
    assert mapping.cost_basis == "Cost Basis"


def test_a_column_is_claimed_by_only_one_role():
    frame = pd.DataFrame(columns=["Symbol", "Quantity", "Price", "Cost Basis"])
    mapping = si.guess_mapping(list(frame.columns))
    chosen = [mapping.ticker, mapping.shares, mapping.cost_basis]
    assert len(set(chosen)) == len(chosen)


def test_an_unrecognisable_header_maps_to_nothing_rather_than_guessing_wrong():
    mapping = si.guess_mapping(["col_a", "col_b"])
    assert mapping.ticker == ""


# --- the preview --------------------------------------------------------------

def _schwab_preview(**overrides):
    frame, _ = si.read_table(SCHWAB_EXPORT.encode(), "p.csv")
    base = dict(ticker="Symbol", shares="Quantity", cost_basis="Cost Basis",
                cost_is_total=True, fallback_date=datetime.date(2024, 1, 15))
    base.update(overrides)
    return si.build_preview(frame, si.ColumnMapping(**base), need_position=True)


def test_the_preview_separates_problems_from_expected_noise():
    """A cash line is expected noise; a row that looks like a position
    and could not be read is a real loss. Collapsing them would hide the
    second inside the first."""
    preview = _schwab_preview()
    assert [r.ticker for r in preview.importable] == ["AAPL", "BRK.B"]
    assert len(preview.skipped) == 2
    assert preview.problems == ()


def test_the_summary_counts_each_category():
    summary = _schwab_preview().summary()
    assert "2 row(s) ready" in summary and "2 skipped" in summary


def test_a_dateless_export_reports_why_rather_than_importing_nothing():
    """A broker's positions export lists what you hold, not when you
    bought it — the most ordinary file there is."""
    preview = _schwab_preview(fallback_date=None)
    assert preview.importable == ()
    assert all("no date column" in r.reason for r in preview.problems)


def test_a_supplied_fallback_date_applies_to_every_row():
    preview = _schwab_preview(fallback_date=datetime.date(2020, 6, 1))
    assert all(r.purchase_date == datetime.date(2020, 6, 1)
               for r in preview.importable)


def test_an_unreadable_cell_in_a_real_date_column_stays_a_problem():
    """Different from having no date column at all: here the file claims
    a date and we failed to read it, which is worth surfacing."""
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [1],
                          "Cost/Share": [1.0], "Date": ["whenever"]})
    mapping = si.ColumnMapping(ticker="Symbol", shares="Quantity",
                               cost_basis="Cost/Share", purchase_date="Date",
                               fallback_date=datetime.date(2024, 1, 1))
    preview = si.build_preview(frame, mapping, need_position=True)
    assert preview.importable == ()
    assert "No purchase date could be read" in preview.problems[0].reason


def test_a_short_position_is_reported_not_silently_dropped():
    """add_holding refuses shares <= 0, so importing it would fail at
    apply time with nothing said here."""
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [-10],
                          "Cost/Share": [1.0]})
    preview = si.build_preview(
        frame, si.ColumnMapping(ticker="Symbol", shares="Quantity",
                                cost_basis="Cost/Share",
                                fallback_date=datetime.date(2024, 1, 1)),
        need_position=True)
    assert preview.importable == ()
    assert "Short position" in preview.problems[0].reason


def test_a_closed_position_is_skipped_not_flagged_as_broken():
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [0],
                          "Cost/Share": [1.0]})
    preview = si.build_preview(
        frame, si.ColumnMapping(ticker="Symbol", shares="Quantity",
                                cost_basis="Cost/Share",
                                fallback_date=datetime.date(2024, 1, 1)),
        need_position=True)
    assert preview.skipped and "closed position" in preview.skipped[0].reason


def test_a_future_purchase_date_is_refused():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [1],
                          "Cost/Share": [1.0], "Date": [tomorrow.isoformat()]})
    preview = si.build_preview(
        frame, si.ColumnMapping(ticker="Symbol", shares="Quantity",
                                cost_basis="Cost/Share", purchase_date="Date"),
        need_position=True)
    assert "future" in preview.problems[0].reason


def test_re_importing_the_same_file_does_not_double_a_position():
    preview = _schwab_preview()
    again = si.build_preview(
        si.read_table(SCHWAB_EXPORT.encode(), "p.csv")[0],
        si.ColumnMapping(ticker="Symbol", shares="Quantity",
                         cost_basis="Cost Basis", cost_is_total=True,
                         fallback_date=datetime.date(2024, 1, 15)),
        need_position=True, existing=[r.ticker for r in preview.importable])
    assert again.importable == ()
    assert all("Already present" in r.reason for r in again.skipped
               if r.ticker in ("AAPL", "BRK.B"))


def test_a_duplicate_inside_one_file_is_imported_once():
    frame = pd.DataFrame({"Symbol": ["AAPL", "AAPL"]})
    preview = si.build_preview(frame, si.ColumnMapping(ticker="Symbol"))
    assert len(preview.importable) == 1


def test_a_watchlist_import_needs_only_a_ticker():
    frame = pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})
    preview = si.build_preview(frame, si.ColumnMapping(ticker="Symbol"))
    assert [r.ticker for r in preview.importable] == ["AAPL", "MSFT"]


def test_no_ticker_column_is_an_explanation_not_an_empty_result():
    preview = si.build_preview(pd.DataFrame({"A": [1]}), si.ColumnMapping())
    assert preview.rows == ()
    assert "ticker" in preview.warnings[0].lower()


def test_a_missing_named_column_is_reported():
    preview = si.build_preview(pd.DataFrame({"A": [1]}),
                               si.ColumnMapping(ticker="Nope"))
    assert "not in this file" in preview.warnings[0]


def test_a_very_long_file_is_truncated_and_says_so(monkeypatch):
    import dataclasses
    monkeypatch.setattr(si, "SPREADSHEET_IMPORT",
                        dataclasses.replace(si.SPREADSHEET_IMPORT, max_rows=5))
    frame = pd.DataFrame({"Symbol": [f"T{i}" for i in range(20)]})
    preview = si.build_preview(frame, si.ColumnMapping(ticker="Symbol"))
    assert preview.truncated is True
    assert len(preview.rows) == 5
    assert any("only the first" in w for w in preview.warnings)


# --- capacity -----------------------------------------------------------------

def test_what_does_not_fit_is_named_never_silently_dropped():
    """Someone importing forty tickers into a ten-slot watchlist must be
    told which thirty did not go."""
    frame = pd.DataFrame({"Symbol": [f"T{i}" for i in range(13)]})
    preview = si.build_preview(frame, si.ColumnMapping(ticker="Symbol"))
    kept, note = si.fit_to_capacity(preview, room=10)
    assert len(kept) == 10
    assert "3 were left out" in note
    assert "T10" in note


def test_a_full_list_says_so_rather_than_importing_nothing_quietly():
    frame = pd.DataFrame({"Symbol": ["AAPL"]})
    preview = si.build_preview(frame, si.ColumnMapping(ticker="Symbol"))
    kept, note = si.fit_to_capacity(preview, room=0)
    assert kept == ()
    assert "no room" in note.lower()


def test_everything_fits_when_there_is_room():
    frame = pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})
    preview = si.build_preview(frame, si.ColumnMapping(ticker="Symbol"))
    kept, note = si.fit_to_capacity(preview, room=10)
    assert len(kept) == 2 and note == ""


# --- pasted text --------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "AAPL, MSFT, NVDA", "AAPL\nMSFT\nNVDA", "aapl; msft; nvda",
    "AAPL MSFT NVDA",
])
def test_a_pasted_ticker_list_parses_however_it_is_separated(text):
    frame, error = si.read_pasted(text)
    assert error == ""
    preview = si.build_preview(frame, si.guess_mapping(list(frame.columns)))
    assert [r.ticker for r in preview.importable][:3] == ["AAPL", "MSFT", "NVDA"]


def test_a_block_copied_out_of_a_spreadsheet_keeps_its_columns():
    """Copying cells out of Excel yields TAB-separated text; treating it
    as a ticker list would land the whole row as one symbol."""
    frame, error = si.read_pasted("Symbol\tShares\nAAPL\t10\nMSFT\t5")
    assert error == ""
    assert list(frame.columns) == ["Symbol", "Shares"]
    assert len(frame) == 2


def test_empty_paste_says_so():
    assert si.read_pasted("")[1] == "Paste something first."
    assert si.read_pasted("   \n  ")[1]


# --- UI wiring ----------------------------------------------------------------

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def test_the_panel_is_wired_into_the_app():
    source = FINANCE.read_text()
    assert "import spreadsheet_import" in source
    assert "spreadsheet_import.build_preview(" in source


def test_both_a_file_and_pasted_text_reach_the_same_pipeline():
    """The two backlog items are one feature; the second says so."""
    source = FINANCE.read_text()
    assert "spreadsheet_import.read_table(" in source
    assert "spreadsheet_import.read_pasted(" in source


def test_nothing_is_written_before_the_preview_is_confirmed():
    """The apply calls must sit inside the Import button's branch."""
    source = FINANCE.read_text()
    button = source.index('key="si_apply"')
    for call in ("pf_add_holding(\n", "update_active_tickers(\n"):
        # every apply-side call appears after the confirm button
        assert source.index("si_apply") < source.rindex(call.strip())


def test_the_two_silent_ambiguities_are_surfaced_as_controls():
    source = FINANCE.read_text()
    assert "si_cost_total" in source
    assert "si_day_first" in source or "si_fallback_date" in source


def test_the_mapping_selectboxes_take_no_key():
    """Their options change with the uploaded file, and a keyed
    selectbox whose stored value falls outside the new options raises —
    the same reason the screener's criteria widgets are unkeyed."""
    source = FINANCE.read_text()
    start = source.index("Which column is which")
    end = source.index('key="si_apply"')
    block = source[start:end]
    for line in block.splitlines():
        if "selectbox(" in line or "_si_options," in line:
            assert "key=" not in line, line
