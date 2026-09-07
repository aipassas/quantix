"""Earnings materials: fetching, classification and cross-quarter search.

The classification tests are built from documents that were actually
measured, not invented. Two of them exist because a naive implementation
got them wrong on live data: CSCO and CVX both ANNOUNCE prepared remarks
without filing any, and a substring test called both "management
commentary". JPM files its press release as `…narrative.htm`, which read
as commentary until its description was given the deciding vote.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import earnings_materials as em


# --- html to text -------------------------------------------------------------

def test_block_tags_separate_text_so_cells_do_not_weld_together():
    # Without a separator this is "$1,234Revenue", one unsearchable token.
    html = "<table><tr><td>$1,234</td><td>Revenue</td></tr></table>"
    text = em.html_to_text(html)
    assert "$1,234 Revenue" in text
    assert "1,234Revenue" not in text


def test_script_and_style_bodies_are_not_treated_as_document_text():
    html = "<p>Real text</p><script>var hidden='secret';</script><style>.a{color:red}</style>"
    text = em.html_to_text(html)
    assert "Real text" in text
    assert "secret" not in text
    assert "color:red" not in text


def test_entities_are_decoded_rather_than_left_as_markup():
    assert "AT&T" in em.html_to_text("<p>AT&amp;T</p>")


def test_malformed_html_degrades_instead_of_raising():
    # One bad quarter must not remove the other eleven from the search.
    text = em.html_to_text("<p>Revenue rose <b>12%<p> unterminated")
    assert "Revenue rose" in text


def test_whitespace_is_collapsed_so_snippets_read_as_prose():
    assert em.html_to_text("<p>a\n\n\t  b</p>") == "a b"


# --- the announcement guard ---------------------------------------------------

CSCO_ANNOUNCEMENT = (
    "Text of the conference call's prepared remarks will be available "
    "within 24 hours of completion of the call. The webcast will include "
    "both the prepared remarks and the question-and-answer session."
)
CVX_ANNOUNCEMENT = (
    "Prepared remarks for today's call, additional financial and operating "
    "information and other complementary materials are available on "
    "Chevron's website under the Investors section."
)
HUM_SELF_REFERENTIAL = (
    "Certain of the matters discussed in these prepared remarks are "
    "forward-looking and are subject to a number of risks and uncertainties."
)


def test_a_company_announcing_remarks_has_not_filed_them():
    assert em._is_announcement_only(CSCO_ANNOUNCEMENT) is True
    assert em._is_announcement_only(CVX_ANNOUNCEMENT) is True


def test_a_document_referring_to_itself_is_the_remarks():
    assert em._is_announcement_only(HUM_SELF_REFERENTIAL) is False


def test_the_two_measured_false_positives_classify_as_releases():
    # This is the regression these tests exist for.
    assert em.classify_document(CSCO_ANNOUNCEMENT) == em.RESULTS_RELEASE
    assert em.classify_document(CVX_ANNOUNCEMENT) == em.RESULTS_RELEASE


def test_self_referential_remarks_classify_as_commentary():
    assert em.classify_document(HUM_SELF_REFERENTIAL) == em.MANAGEMENT_COMMENTARY


def test_self_reference_wins_even_when_an_announcement_is_also_present():
    # HUM's remarks document also announces where the call materials live.
    body = HUM_SELF_REFERENTIAL + " " + CVX_ANNOUNCEMENT
    assert em.classify_document(body) == em.MANAGEMENT_COMMENTARY


# --- classification order -----------------------------------------------------

def test_a_cfo_commentary_exhibit_is_commentary_without_the_phrase():
    # NVDA's CFO Commentary never says "prepared remarks"; the filename is
    # the whole signal.
    assert em.classify_document("Revenue was up.", "q2fy27cfocommentary.htm") \
        == em.MANAGEMENT_COMMENTARY


def test_posted_remarks_filename_is_recognised():
    assert em.classify_document("x", "a2q2026humanaincpostedrema.htm") \
        == em.MANAGEMENT_COMMENTARY


def test_supplement_beats_release_when_the_description_says_both():
    # JPM: "EARNINGS RELEASE FINANCIAL SUPPLEMENT - SECOND QUARTER 2026".
    kind = em.classify_document(
        "tables", "a2q26erfex992supplement.htm",
        "JPMORGAN CHASE & CO. EARNINGS RELEASE FINANCIAL SUPPLEMENT")
    assert kind == em.SUPPLEMENT


def test_a_release_named_narrative_is_a_release_not_commentary():
    # The defect this ordering fixes: the filename said "narrative", the
    # description said "EARNINGS RELEASE".
    kind = em.classify_document(
        "Dimon said the economy was resilient.",
        "a2q26erfexhibit991narrative.htm",
        "JPMORGAN CHASE & CO. EARNINGS RELEASE - SECOND QUARTER 2026 RESULTS")
    assert kind == em.RESULTS_RELEASE


def test_a_presentation_exhibit_is_a_supplement():
    assert em.classify_document("x", "earningspresentationfy27.htm") == em.SUPPLEMENT


def test_supplement_beats_commentary_when_a_document_matches_both():
    """A deck of CFO commentary slides matches both hint lists. Slides are
    a supplement — the reader gets tables and bullet fragments, not the
    prose that makes commentary worth searching — so supplement wins, and
    this pins that precedence rather than leaving it to hint order."""
    assert em.classify_document("x", "q2fy27cfocommentarypresentation.htm") \
        == em.SUPPLEMENT


def test_a_table_heavy_document_with_no_naming_hint_is_a_supplement():
    numbers = " ".join("1234567890" for _ in range(200))
    assert em._digit_share(numbers) > em.classify_document.__globals__[
        "EARNINGS_MATERIALS"].supplement_digit_share
    assert em.classify_document(numbers, "ex99x2.htm") == em.SUPPLEMENT


def test_prose_with_no_naming_hint_is_a_release():
    prose = ("The company reported strong demand across every segment and "
             "expects continued momentum into the next fiscal year. ") * 20
    assert em.classify_document(prose, "ex99x1.htm") == em.RESULTS_RELEASE


def test_digit_share_of_empty_text_is_zero_not_an_error():
    assert em._digit_share("") == 0.0
    assert em._digit_share("   ") == 0.0


# --- the filing index ---------------------------------------------------------

INDEX_HTML = """
<table>
 <tr><td>1</td><td>8-K cover</td><td>msft-20260729.htm</td><td>8-K</td>
     <td><a href="/Archives/edgar/data/789019/x/msft-20260729.htm">doc</a></td></tr>
 <tr><td>2</td><td>EX-99.1</td><td>msft-ex99_1.htm</td><td>EX-99.1</td>
     <td><a href="/Archives/edgar/data/789019/x/msft-ex99_1.htm">doc</a></td></tr>
 <tr><td>3</td><td>EX-99.1</td><td>exhibit991015.jpg</td><td>GRAPHIC</td>
     <td><a href="/Archives/edgar/data/789019/x/exhibit991015.jpg">doc</a></td></tr>
 <tr><td>4</td><td>EXHIBIT 99.2</td><td>tm_ex99-2.htm</td><td>EXHIBIT 99.2</td>
     <td><a href="/Archives/edgar/data/789019/x/tm_ex99-2.htm">doc</a></td></tr>
 <tr><td>5</td><td>schema</td><td>msft.xsd</td><td>EX-101.SCH</td>
     <td><a href="/Archives/edgar/data/789019/x/msft.xsd">doc</a></td></tr>
 <tr><td>6</td><td>EX-99.4</td><td>ex994deck.pdf</td><td>EX-99.4</td>
     <td><a href="/Archives/edgar/data/789019/x/ex994deck.pdf">doc</a></td></tr>
</table>
"""


def test_the_index_yields_only_ex99_text_documents():
    entries = em.parse_filing_index(INDEX_HTML)
    hrefs = [e[2] for e in entries]
    assert any("msft-ex99_1.htm" in h for h in hrefs)
    assert not any(h.endswith(".jpg") for h in hrefs), "image exhibits are unsearchable"
    assert not any(h.endswith(".xsd") for h in hrefs), "XBRL schema is not a document"


def test_an_ex99_exhibit_in_a_binary_format_is_excluded_by_extension():
    """The JPEG rows above are filtered by their GRAPHIC *type*, so they
    never exercise the extension check. A PDF exhibit declared as EX-99.4
    does: it is a real EX-99 and still carries no extractable text."""
    hrefs = [e[2] for e in em.parse_filing_index(INDEX_HTML)]
    assert not any(h.endswith(".pdf") for h in hrefs)


def test_the_spelled_out_exhibit_type_is_recognised():
    # MRK declares "EXHIBIT 99.2" where MSFT declares "EX-99.2".
    hrefs = [e[2] for e in em.parse_filing_index(INDEX_HTML)]
    assert any("tm_ex99-2.htm" in h for h in hrefs)


def test_an_index_with_no_exhibits_returns_empty_rather_than_raising():
    assert em.parse_filing_index("<table><tr><td>nothing</td></tr></table>") == ()
    assert em.parse_filing_index("") == ()


# --- filings ------------------------------------------------------------------

def _submissions(forms, items, dates, accessions):
    return json.dumps({"filings": {"recent": {
        "form": forms, "items": items, "filingDate": dates,
        "accessionNumber": accessions}}})


@pytest.fixture
def sec(monkeypatch):
    """A fake SEC. Keys are matched as substrings of the requested URL."""
    calls = []
    payloads = {}

    def fake_get(url):
        calls.append(url)
        for key, value in payloads.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise RuntimeError(f"HTTP 404 from {url}")

    monkeypatch.setattr(em, "_sec_get", fake_get)
    monkeypatch.setattr(em, "load_ticker_map", lambda: ({"MSFT": "0000789019"}, None))
    return type("SEC", (), {"payloads": payloads, "calls": calls})()


def test_only_item_202_filings_count_as_earnings(sec):
    sec.payloads["submissions"] = _submissions(
        ["8-K", "8-K", "10-Q", "8-K"],
        ["5.02", "2.02,9.01", "", "2.02"],
        ["2026-08-01", "2026-07-29", "2026-07-01", "2026-04-29"],
        ["a-1", "a-2", "a-3", "a-4"],
    )
    filings, error = em.load_filings.__wrapped__("MSFT")
    assert error is None
    assert [f.accession for f in filings] == ["a-2", "a-4"]


def test_a_company_with_no_earnings_8ks_says_so_rather_than_returning_empty(sec):
    sec.payloads["submissions"] = _submissions(["10-K"], [""], ["2026-01-01"], ["a-1"])
    filings, error = em.load_filings.__wrapped__("MSFT")
    assert filings == ()
    assert "no 8-K filings reporting results" in error


def test_a_dead_sec_returns_a_message_rather_than_raising(sec):
    sec.payloads["submissions"] = RuntimeError("HTTP 429")
    filings, error = em.load_filings.__wrapped__("MSFT")
    assert filings == ()
    assert "Could not reach" in error


def test_an_unknown_ticker_explains_that_the_index_is_us_registrants_only(monkeypatch):
    monkeypatch.setattr(em, "load_ticker_map", lambda: ({"MSFT": "0000789019"}, None))
    cik, error = em.resolve_cik("VWCE.DE")
    assert cik is None
    assert "not in the SEC's registrant index" in error
    assert "non-US listings" in error


def test_an_empty_ticker_is_refused_without_a_network_call(monkeypatch):
    def explode():
        raise AssertionError("should not have been called")
    monkeypatch.setattr(em, "load_ticker_map", explode)
    cik, error = em.resolve_cik("")
    assert cik is None and error


def test_the_index_url_drops_the_dashes_from_the_accession():
    filing = em.Filing("MSFT", "0000789019", "0001193125-26-323632", "2026-07-29", "2.02")
    assert "/000119312526323632/" in filing.index_url
    assert filing.index_url.endswith("0001193125-26-323632-index.htm")
    # The CIK is unpadded in an archive path; a padded one 404s.
    assert "/data/789019/" in filing.index_url


# --- documents ----------------------------------------------------------------

def test_documents_are_fetched_classified_and_stubs_dropped(sec):
    sec.payloads["submissions"] = _submissions(
        ["8-K"], ["2.02"], ["2026-07-29"], ["0001193125-26-323632"])
    sec.payloads["-index.htm"] = INDEX_HTML
    sec.payloads["msft-ex99_1.htm"] = "<p>" + ("Revenue grew strongly. " * 200) + "</p>"
    sec.payloads["tm_ex99-2.htm"] = "<p>too short</p>"

    documents, warnings = em.load_documents.__wrapped__("MSFT", quarters=1)
    assert len(documents) == 1, "the stub exhibit is below the length floor"
    assert documents[0].exhibit == "EX-99.1"
    assert documents[0].kind == em.RESULTS_RELEASE
    assert documents[0].word_count > 100
    assert warnings == ()


def test_one_unreadable_exhibit_warns_and_keeps_the_others(sec):
    sec.payloads["submissions"] = _submissions(
        ["8-K"], ["2.02"], ["2026-07-29"], ["0001193125-26-323632"])
    sec.payloads["-index.htm"] = INDEX_HTML
    sec.payloads["msft-ex99_1.htm"] = "<p>" + ("Revenue grew. " * 200) + "</p>"
    sec.payloads["tm_ex99-2.htm"] = RuntimeError("HTTP 500")

    documents, warnings = em.load_documents.__wrapped__("MSFT", quarters=1)
    assert len(documents) == 1
    assert any("could not read the exhibit" in w for w in warnings)


def test_an_image_only_filing_says_why_it_has_no_text(sec):
    sec.payloads["submissions"] = _submissions(
        ["8-K"], ["2.02"], ["2026-07-22"], ["0001628280-26-049213"])
    sec.payloads["-index.htm"] = """<table><tr>
        <td>1</td><td>EX-99.1</td><td>exhibit991015.jpg</td><td>GRAPHIC</td>
        <td><a href="/Archives/x/exhibit991015.jpg">doc</a></td></tr></table>"""
    documents, warnings = em.load_documents.__wrapped__("MSFT", quarters=1)
    assert documents == ()
    assert any("images or data only" in w for w in warnings)


def test_quarters_is_clamped_to_the_configured_maximum(sec):
    sec.payloads["submissions"] = _submissions(
        ["8-K"] * 40, ["2.02"] * 40,
        [f"2026-01-{(i % 28) + 1:02d}" for i in range(40)],
        [f"a-{i}" for i in range(40)])
    sec.payloads["-index.htm"] = "<table></table>"
    em.load_documents.__wrapped__("MSFT", quarters=9999)
    index_calls = [c for c in sec.calls if "-index.htm" in c]
    assert len(index_calls) <= em.EARNINGS_MATERIALS.max_quarters


# --- search -------------------------------------------------------------------

def _doc(text, filed_on="2026-07-29", exhibit="EX-99.1"):
    return em.Document("MSFT", "a-1", filed_on, exhibit, "f.htm", "u",
                       "", text, em.RESULTS_RELEASE)


def test_search_matches_whole_words_only():
    # The defect a plain substring search has: "AI" inside "said".
    doc = _doc("The company said AI adoption rose. Repairs are not AI.")
    assert em.search([doc], "AI").total_hits == 2


def test_search_is_case_insensitive():
    doc = _doc("Tariffs rose. TARIFF pressure eased. tariff talk continued.")
    assert em.search([doc], "tariff").total_hits == 2  # "Tariffs" is a different word


def test_a_multi_word_query_matches_across_the_phrase():
    doc = _doc("Net interest income was $25.6 billion, up 10%.")
    assert em.search([doc], "net interest income").total_hits == 1


def test_a_query_with_punctuation_still_works():
    # A word-boundary pattern would never match a leading "$".
    doc = _doc("Revenue of $1.2 billion was reported.")
    assert em.search([doc], "$1.2 billion").total_hits == 1


def test_an_empty_query_returns_nothing_and_does_not_match_everything():
    result = em.search([_doc("anything at all")], "   ")
    assert result.total_hits == 0
    assert result.documents_searched == 1


def test_hits_per_document_are_capped_and_the_document_count_stays_right():
    doc = _doc("revenue " * 500)
    result = em.search([doc], "revenue")
    assert result.total_hits == em.EARNINGS_MATERIALS.max_hits_per_document
    assert result.documents_matched == 1


def test_documents_matched_counts_documents_not_hits():
    docs = [_doc("growth growth growth"), _doc("nothing here"), _doc("growth")]
    result = em.search(docs, "growth")
    assert result.documents_matched == 2
    assert result.documents_searched == 3
    assert result.total_hits == 4


def test_a_snippet_carries_context_around_the_match():
    doc = _doc("A " * 300 + "tariff" + " B" * 300)
    hit = em.search([doc], "tariff").hits[0]
    assert "tariff" in hit.snippet
    assert len(hit.snippet) < len(doc.text)
    assert hit.snippet.startswith("…") and hit.snippet.endswith("…")


def test_a_snippet_at_the_start_has_no_leading_ellipsis():
    doc = _doc("Tariff pressure eased through the quarter as volumes recovered.")
    hit = em.search([doc], "tariff").hits[0]
    assert not hit.snippet.startswith("…")


# --- disclosure and coverage --------------------------------------------------

def test_the_unavailability_note_describes_an_absence():
    note = em.TRANSCRIPTS_UNAVAILABLE
    assert "not call transcripts" in note
    assert "analyst Q&A" in note or "Q&A" in note
    # It must point somewhere useful rather than only saying no.
    assert "investor-relations" in note or "investor relations" in note


def test_the_unavailability_note_quotes_the_measurement():
    assert "25" in em.TRANSCRIPTS_UNAVAILABLE


def test_coverage_note_says_when_nothing_is_management_commentary():
    docs = [_doc("a"), _doc("b")]
    note = em.coverage_note(docs)
    assert "None of them is management commentary" in note


def test_coverage_note_counts_commentary_and_agrees_with_itself():
    commentary = em.Document("MSFT", "a", "2026-07-29", "EX-99.2", "f", "u", "",
                             "text", em.MANAGEMENT_COMMENTARY)
    note = em.coverage_note([_doc("a"), commentary])
    assert "1 of them is management commentary" in note


def test_coverage_note_on_an_empty_corpus_does_not_claim_a_range():
    assert em.coverage_note([]) == "No earnings documents were retrieved."


def test_coverage_note_reports_the_real_date_span():
    docs = [_doc("a", filed_on="2026-07-29"), _doc("b", filed_on="2025-01-30")]
    note = em.coverage_note(docs)
    assert "2025-01-30 to 2026-07-29" in note


def test_earliest_filing_date_ignores_unparseable_dates():
    filings = [
        em.Filing("MSFT", "1", "a", "2026-07-29", "2.02"),
        em.Filing("MSFT", "1", "b", "", "2.02"),
        em.Filing("MSFT", "1", "c", "2020-06-08", "2.02"),
    ]
    assert em.earliest_filing_date(filings).isoformat() == "2020-06-08"
    assert em.earliest_filing_date([]) is None


def test_the_document_label_names_the_exhibit_so_two_of_a_kind_differ():
    a = _doc("x", exhibit="EX-99.1")
    b = em.Document("MSFT", "a-1", "2026-07-29", "EX-99.2", "f", "u", "", "x",
                    em.RESULTS_RELEASE)
    assert a.label != b.label


# --- SEC access rules ---------------------------------------------------------

def test_the_user_agent_declares_a_contact_as_the_sec_requires():
    agent = em._headers()["User-Agent"]
    assert "@" in agent, "SEC refuses a User-Agent with no contact address"
    assert agent == em.EARNINGS_MATERIALS.user_agent


def test_requests_are_throttled_inside_the_sec_rate_limit():
    # The SEC asks for no more than ten a second.
    assert em.EARNINGS_MATERIALS.request_interval_seconds >= 0.1


def test_every_fetch_goes_through_the_throttled_helper():
    """No module may call requests.get directly and skip the rate limit."""
    source = Path(em.__file__).read_text()
    body = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    direct = [line for line in body.splitlines()
              if "requests.get(" in line and "_sec_get" not in line]
    assert len(direct) == 1, f"expected only the helper's own call, got {direct}"


# --- UI wiring ----------------------------------------------------------------

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def test_the_panel_is_actually_wired_into_the_app():
    """Declaring a module is not the same as sourcing it — this project
    has shipped a panel whose data was never handed to it."""
    source = FINANCE.read_text()
    assert "import earnings_materials" in source
    assert "load_documents" in source


def test_the_app_shows_the_transcript_disclosure_rather_than_reimplementing_it():
    source = FINANCE.read_text()
    assert "TRANSCRIPTS_UNAVAILABLE" in source


def test_the_app_does_not_call_the_panel_a_transcript_archive():
    """Naming it "Transcripts" would promise the analyst Q&A that no free
    source has."""
    source = FINANCE.read_text()
    for line in source.splitlines():
        if "st.expander(" in line and "arnings" in line:
            assert "ranscript" not in line, line
