"""Tests for the PowerPoint and Excel exports.

The properties worth defending here are the ones that make an export
trustworthy rather than merely present, and each has a specific failure
mode that looks fine on screen:

1. THE DECK IS NATIVE, NOT A PICTURE. The selling point of exporting a
   deck rather than a PDF is that the recipient can edit it. A generator
   that rendered each slide to an image would produce a file that opens,
   looks correct, and is useless — so the tests assert real text frames
   and real table cells, not just that a .pptx was produced.

2. NOTHING IS INVENTED. A metric that could not be computed must not
   appear as 0.00 anywhere. This is the rule the tear sheet actually
   broke in production — it printed a $0.00 intrinsic value whenever the
   DCF failed — so it is pinned here in both formats.

3. EXCEL PERCENTAGES ARE STORED THE WAY EXCEL MEANS THEM. Excel's
   percent formats multiply the stored number by 100 to display it, so
   storing 27.62 under "0.00%" renders as 2762.00%. Every figure in this
   app is percent-valued, which makes this an easy and invisible bug: the
   number is right in Python and wrong in the file.

4. MISSING NUMBERS LEAVE THE CELL EMPTY. A 0 would be silently absorbed
   into any SUM or AVERAGE the recipient builds. A blank is skipped by
   both, so the arithmetic stays honest even when the reader ignores the
   status column.

5. THE DISCLOSURE SURVIVES WHITE-LABELLING, matching the guarantee
   branding.py already makes for the on-screen app.
"""
import datetime
import io

import pytest

import export_deck
import export_workbook
from export_deck import DeckData, Metric
from export_workbook import Row, Sheet, WorkbookData

pptx = pytest.importorskip("pptx")
openpyxl = pytest.importorskip("openpyxl")


# --- fixtures -----------------------------------------------------------------

def _deck_data(**overrides):
    base = dict(
        ticker="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        alignment_verdict="high",
        alignment_pct=100.0,
        strengths=("Net margin is well above the floor.",),
        concerns=("P/E sits above the sector band.",),
        current_price=316.83,
        intrinsic_price=248.10,
        margin_of_safety_pct=-27.7,
        dcf_status="Trading above intrinsic value",
        wacc=0.0842,
        altman_z=11.94,
        altman_verdict="safe zone",
        risk_grade="B",
        metrics=(Metric("Net margin", 27.62, "%"),
                 Metric("Interest coverage", None)),
    )
    base.update(overrides)
    return DeckData(**base)


def _slides(payload):
    return pptx.Presentation(io.BytesIO(payload)).slides


def _all_text(payload):
    out = []
    for slide in _slides(payload):
        for shape in slide.shapes:
            if shape.has_text_frame:
                out.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    out.extend(cell.text for cell in row.cells)
    return "\n".join(out)


# --- the deck is native and editable ------------------------------------------

def test_deck_builds():
    payload, error = export_deck.build_deck(_deck_data())
    assert error is None
    assert payload and payload[:2] == b"PK"          # a real OOXML zip


def test_deck_is_sixteen_by_nine():
    payload, _ = export_deck.build_deck(_deck_data())
    prs = pptx.Presentation(io.BytesIO(payload))
    ratio = prs.slide_width / prs.slide_height
    assert abs(ratio - 16 / 9) < 0.01


def test_every_slide_carries_real_editable_text():
    """Not a picture of a report — the recipient must be able to retype it."""
    payload, _ = export_deck.build_deck(_deck_data())
    for index, slide in enumerate(_slides(payload), start=1):
        has_text = any(
            shape.has_text_frame and shape.text_frame.text.strip()
            for shape in slide.shapes
        )
        assert has_text, f"slide {index} has no editable text"


def test_health_metrics_are_a_real_table_not_an_image():
    payload, _ = export_deck.build_deck(_deck_data())
    tables = [shape.table for slide in _slides(payload)
              for shape in slide.shapes if shape.has_table]
    assert tables, "metrics should be a native table"
    cells = [c.text for row in tables[0].rows for c in row.cells]
    assert "Net margin" in cells


def test_narrative_comes_through_verbatim():
    """Slide 2 must show the digest's own wording, not a paraphrase."""
    data = _deck_data(strengths=("A distinctive strength sentence.",),
                      concerns=("A distinctive concern sentence.",))
    payload, _ = export_deck.build_deck(data)
    text = _all_text(payload)
    assert "A distinctive strength sentence." in text
    assert "A distinctive concern sentence." in text


# --- nothing is invented ------------------------------------------------------

def test_missing_metric_reads_as_not_reported_never_zero():
    payload, _ = export_deck.build_deck(_deck_data())
    text = _all_text(payload)
    assert "not reported" in text
    # The unavailable metric must not have become a plausible-looking 0.
    assert "0.00" not in text.replace("27.62", "")


def test_metric_display_never_fabricates():
    assert Metric("x", None).display == "not reported"
    assert Metric("x", 0.0).display == "0.00"       # a real zero still shows


def test_deck_without_a_dcf_says_so_rather_than_showing_zero():
    data = _deck_data(intrinsic_price=None, margin_of_safety_pct=None,
                      dcf_status="", wacc=None,
                      dcf_unavailable_reason="Negative free cash flow.")
    payload, error = export_deck.build_deck(data)
    assert error is None
    text = _all_text(payload)
    assert "Negative free cash flow." in text
    assert "$0.00" not in text


# --- the disclosure travels ---------------------------------------------------

def test_disclosure_is_in_the_deck():
    payload, _ = export_deck.build_deck(_deck_data())
    assert "Not investment advice" in _all_text(payload)


def test_disclosure_survives_rebranding(monkeypatch):
    """A licensee may change every name; they may not drop the notice."""
    import branding

    monkeypatch.setattr(branding, "brand",
                        lambda: branding.Brand(name="Meridian Capital",
                                               accent_color="#AA3366"))
    payload, _ = export_deck.build_deck(_deck_data())
    text = _all_text(payload)
    assert "Not investment advice" in text
    assert "Meridian Capital" in text


def test_filename_follows_the_brand(monkeypatch):
    import branding

    monkeypatch.setattr(branding, "brand",
                        lambda: branding.Brand(name="Meridian Capital"))
    name = export_deck.filename_for("AAPL", datetime.date(2026, 8, 22))
    assert name == "meridian-capital-aapl-20260822.pptx"


# --- charts -------------------------------------------------------------------

def test_a_date_axis_chart_still_renders():
    """Regression: every price chart in this app carries pandas
    Timestamps, and kaleido's orjson serializer rejects them outright
    ("Type is not JSON serializable: Timestamp"). This shipped as a deck
    whose numeric-only gauge slide rendered while the main price chart
    silently vanished — the confusing half-failure, not a clean one."""
    pd = pytest.importorskip("pandas")
    go = pytest.importorskip("plotly.graph_objects")
    # importorskip rather than charts_available(): that helper renders a
    # figure of its own, and kaleido's browser cold-start is the expensive
    # part, so probing with it would double the cost of this test.
    pytest.importorskip("kaleido")

    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(120))))
    fig.add_shape(type="line", x0=dates[60], x1=dates[60], y0=0, y1=119)
    fig.add_annotation(x=dates[60], y=119, text="signal")

    png = export_deck.chart_png(fig, 600, 300)
    assert png, "a date-axis chart must survive export"
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_render_failure_loses_the_chart_not_the_deck():
    class Broken:
        def to_image(self, **kwargs):
            raise RuntimeError("no browser binary")

    assert export_deck.chart_png(Broken()) is None
    payload, error = export_deck.build_deck(_deck_data(charts=()))
    assert error is None and payload


# --- the workbook holds real numbers ------------------------------------------

def _sheet(payload, title):
    return openpyxl.load_workbook(io.BytesIO(payload))[title]


def _workbook(rows):
    data = WorkbookData(ticker="AAPL", sheets=(Sheet("Data", rows=rows),))
    payload, error = export_workbook.build_workbook(data)
    assert error is None, error
    return _sheet(payload, "Data")


def test_values_are_numbers_not_preformatted_strings():
    """A string looks identical on screen and dies in a formula."""
    ws = _workbook((Row("Current ratio", 1.25, "x"),))
    cell = ws.cell(row=2, column=2)
    assert isinstance(cell.value, float)
    assert cell.value == pytest.approx(1.25)


def test_percentages_are_stored_as_excel_means_them():
    """27.62% must store 0.2762 — Excel multiplies by 100 to display."""
    ws = _workbook((Row("Net margin", 27.62, "%", percent=True),))
    cell = ws.cell(row=2, column=2)
    assert cell.value == pytest.approx(0.2762)
    assert cell.number_format == "0.00%"
    # what the user actually sees
    assert f"{cell.value * 100:.2f}%" == "27.62%"


def test_a_non_percent_row_is_left_alone():
    ws = _workbook((Row("Altman Z", 11.94),))
    assert ws.cell(row=2, column=2).value == pytest.approx(11.94)


def test_missing_value_leaves_the_cell_blank_not_zero():
    """A 0 would be absorbed into the recipient's SUM; a blank is skipped."""
    ws = _workbook((Row("Interest coverage", None, "x"),))
    assert ws.cell(row=2, column=2).value is None
    assert ws.cell(row=2, column=4).value == "not reported"


def test_available_value_is_flagged_ok():
    ws = _workbook((Row("Net margin", 27.62, "%", percent=True),))
    assert ws.cell(row=2, column=4).value == "ok"


def test_a_real_zero_is_still_written_as_zero():
    """Honesty runs both ways: 0 that was measured must not become blank."""
    ws = _workbook((Row("Dividend yield", 0.0, "%", percent=True),))
    assert ws.cell(row=2, column=2).value == 0.0
    assert ws.cell(row=2, column=4).value == "ok"


def test_thresholds_travel_with_the_values():
    ws = _workbook((Row("Net margin", 27.62, "%", detail="passes — benchmark: > 10%"),))
    assert "benchmark: > 10%" in ws.cell(row=2, column=5).value


def test_headers_and_frozen_panes():
    ws = _workbook((Row("Net margin", 1.0),))
    assert [ws.cell(row=1, column=c).value for c in range(1, 6)] == [
        "Metric", "Value", "Unit", "Status", "Detail"]
    assert ws.freeze_panes == "A2"


# --- Excel's own constraints --------------------------------------------------

def test_illegal_sheet_names_do_not_break_the_save():
    """Excel rejects : \\ / ? * [ ] and caps names at 31 characters."""
    data = WorkbookData(
        ticker="AAPL",
        sheets=(Sheet("Valuation: DCF / WACC [core] " + "x" * 40,
                      rows=(Row("a", 1.0),)),))
    payload, error = export_workbook.build_workbook(data)
    assert error is None
    names = openpyxl.load_workbook(io.BytesIO(payload)).sheetnames
    made = names[1]
    assert len(made) <= 31
    assert not set(made) & set(':\\/?*[]')


def test_workbook_carries_the_disclosure():
    data = WorkbookData(ticker="AAPL", sheets=(Sheet("Data", rows=(Row("a", 1.0),)),))
    payload, _ = export_workbook.build_workbook(data)
    overview = _sheet(payload, "Overview")
    text = "\n".join(str(c.value) for row in overview.iter_rows() for c in row
                     if c.value is not None)
    assert "Not investment advice" in text
    assert "not a zero" in text


def test_workbook_filename_follows_the_brand():
    name = export_workbook.filename_for("AAPL", datetime.date(2026, 8, 22))
    assert name.endswith("-aapl-20260822.xlsx")
