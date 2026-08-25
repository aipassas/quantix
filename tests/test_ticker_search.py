"""Tests for ticker_search.py — the autocomplete dropdown's option list
and the Yahoo company-name search behind it.

The searcher is injected throughout so nothing here touches the network.
The failure-path tests matter most: this control lives in the sidebar, so
a search that raised would take down the whole page rather than just
itself.
"""
import ticker_search as ts
from ticker_search import (
    SymbolMatch,
    build_universe,
    search_symbols,
    symbol_from_label,
)


def _quotes(*rows):
    """Yahoo-shaped search payload."""
    return [
        {"symbol": s, "shortname": n, "quoteType": qt, "exchange": ex}
        for s, n, qt, ex in rows
    ]


# --- labels -------------------------------------------------------------------

def test_label_pairs_symbol_with_name():
    assert SymbolMatch("AAPL", "Apple Inc.").label == "AAPL — Apple Inc."


def test_label_falls_back_to_the_bare_symbol_without_a_name():
    """A dangling separator would look like a rendering bug."""
    assert SymbolMatch("AAPL", "").label == "AAPL"


def test_detail_shows_type_and_exchange():
    assert SymbolMatch("AAPL", "Apple Inc.", "EQUITY", "NMS").detail == "EQUITY · NMS"


def test_detail_is_empty_when_nothing_is_known():
    assert SymbolMatch("AAPL", "Apple Inc.").detail == ""


# --- label -> symbol round trip -------------------------------------------------

def test_symbol_is_recovered_from_a_generated_label():
    assert symbol_from_label("AAPL — Apple Inc.") == "AAPL"


def test_freehand_input_is_normalised():
    """accept_new_options means the value may be whatever was typed."""
    assert symbol_from_label("  aapl ") == "AAPL"


def test_a_symbol_containing_a_hyphen_survives():
    """Splitting on the wrong character would mangle BRK-B or BTC-USD."""
    assert symbol_from_label("BRK-B") == "BRK-B"
    assert symbol_from_label("BTC-USD — Bitcoin USD") == "BTC-USD"


def test_empty_label_is_empty():
    assert symbol_from_label("") == ""
    assert symbol_from_label(None) == ""


# --- universe -------------------------------------------------------------------

def test_universe_deduplicates_and_sorts(monkeypatch):
    monkeypatch.setattr(ts, "name_for", lambda t: "")
    got = [m.symbol for m in build_universe(["MSFT", "aapl", "AAPL", "  ", "nvda"])]
    assert got == ["AAPL", "MSFT", "NVDA"]


def test_universe_is_alphabetical_not_insertion_ordered(monkeypatch):
    """This list is typed into, not scanned — a symbol changing position
    between renders would actively get in the way."""
    monkeypatch.setattr(ts, "name_for", lambda t: "")
    assert [m.symbol for m in build_universe(["ZZZ", "AAA"])] == ["AAA", "ZZZ"]


def test_universe_attaches_names(monkeypatch):
    monkeypatch.setattr(ts, "name_for", lambda t: {"AAPL": "Apple Inc."}.get(t, ""))
    labels = [m.label for m in build_universe(["AAPL", "XYZ"])]
    assert labels == ["AAPL — Apple Inc.", "XYZ"]


def test_universe_can_skip_name_resolution():
    assert build_universe(["AAPL"], resolve_names=False)[0].name == ""


def test_empty_universe_is_empty(monkeypatch):
    monkeypatch.setattr(ts, "name_for", lambda t: "")
    assert build_universe([]) == ()
    assert build_universe(["", "   "]) == ()


# --- search ---------------------------------------------------------------------

def test_search_maps_yahoo_rows_to_matches():
    def fake(q, n):
        return _quotes(("AAPL", "Apple Inc.", "EQUITY", "NMS"))
    matches, err = search_symbols("apple", searcher=fake)
    assert err is None
    assert matches[0] == SymbolMatch("AAPL", "Apple Inc.", "EQUITY", "NMS")


def test_search_requires_two_characters():
    """Guards against firing a request on the first keystroke."""
    matches, err = search_symbols("a", searcher=lambda q, n: _quotes(("X", "X", "", "")))
    assert matches == () and "two characters" in err


def test_search_deduplicates_repeated_symbols():
    def fake(q, n):
        return _quotes(("AAPL", "Apple Inc.", "EQUITY", "NMS"),
                       ("AAPL", "Apple Inc.", "EQUITY", "NMS"))
    matches, _ = search_symbols("apple", searcher=fake)
    assert len(matches) == 1


def test_search_skips_rows_with_no_symbol():
    def fake(q, n):
        return [{"shortname": "No Symbol"}, *_quotes(("AAPL", "Apple Inc.", "EQUITY", "NMS"))]
    matches, _ = search_symbols("apple", searcher=fake)
    assert [m.symbol for m in matches] == ["AAPL"]


def test_search_tolerates_non_dict_rows():
    """A yfinance shape change must not raise inside the sidebar."""
    def fake(q, n):
        return ["garbage", None, *_quotes(("AAPL", "Apple Inc.", "", ""))]
    matches, err = search_symbols("apple", searcher=fake)
    assert err is None and [m.symbol for m in matches] == ["AAPL"]


def test_search_respects_max_results():
    def fake(q, n):
        return _quotes(*[(f"T{i}", f"Name {i}", "EQUITY", "NMS") for i in range(20)])
    matches, _ = search_symbols("test", max_results=3, searcher=fake)
    assert len(matches) == 3


def test_no_results_reports_the_query_back():
    matches, err = search_symbols("zzzzz", searcher=lambda q, n: [])
    assert matches == () and "zzzzz" in err


def test_a_raising_searcher_returns_an_error_rather_than_propagating():
    """The whole point: this renders in the sidebar, so an exception here
    would take down the entire page instead of one control."""
    def boom(q, n):
        raise ConnectionError("network down")
    matches, err = search_symbols("apple", searcher=boom)
    assert matches == ()
    assert "unavailable" in err and "ConnectionError" in err


def test_search_handles_a_none_payload():
    matches, err = search_symbols("apple", searcher=lambda q, n: None)
    assert matches == () and err is not None


# --- recovering a symbol that needs an exchange suffix ------------------------
# Reported live: typing VWCE gave "No reliable data found … check the
# ticker symbol", which is unhelpful because the symbol is RIGHT. A
# European fund is quoted by its bare ticker on every platform except
# this data source, which wants the venue: VWCE 404s, VWCE.DE / .MI / .AS
# all resolve. Yahoo's own search already maps one to the other.

def _fake_searcher(rows):
    return lambda query, max_results: rows[:max_results]


def test_a_bare_european_ticker_is_matched_to_its_listings():
    rows = [
        {"symbol": "VWCE.DE", "shortname": "Vanguard FTSE All-World",
         "quoteType": "ETF", "exchange": "GER"},
        {"symbol": "VWCE.MI", "shortname": "Vanguard Ftse All-World",
         "quoteType": "ETF", "exchange": "MIL"},
        {"symbol": "VWCE.AS", "shortname": "Vanguard Ftse All-World",
         "quoteType": "ETF", "exchange": "AMS"},
    ]
    # search_symbols carries the injectable searcher and
    # suggest_alternatives goes through it, so drive the seam that exists
    # rather than reaching past it to the network.
    matches, error = ts.search_symbols("VWCE", searcher=_fake_searcher(rows))
    assert error is None
    assert [m.symbol for m in matches] == ["VWCE.DE", "VWCE.MI", "VWCE.AS"]
    # The venue is shown, because the same fund on two exchanges is two
    # different prices in two different currencies.
    assert "GER" in matches[0].detail


def test_the_symbol_that_just_failed_is_not_offered_back():
    """Handing someone the exact thing that did not work is noise."""
    rows = [
        {"symbol": "VWCE", "shortname": "stale row", "quoteType": "ETF"},
        {"symbol": "VWCE.DE", "shortname": "Vanguard FTSE All-World",
         "quoteType": "ETF", "exchange": "GER"},
    ]
    matches, _ = ts.suggest_alternatives.__wrapped__(
        "VWCE", searcher=_fake_searcher(rows))
    assert [m.symbol for m in matches] == ["VWCE.DE"]
    # Case-insensitively, since the box upper-cases what is typed.
    matches, _ = ts.suggest_alternatives.__wrapped__(
        "vwce", searcher=_fake_searcher(rows))
    assert [m.symbol for m in matches] == ["VWCE.DE"]


def test_suggestions_are_capped():
    rows = [{"symbol": f"X{i}.DE", "shortname": f"Fund {i}", "quoteType": "ETF"}
            for i in range(20)]
    matches, _ = ts.suggest_alternatives.__wrapped__(
        "XX", limit=4, searcher=_fake_searcher(rows))
    assert len(matches) == 4


def test_a_too_short_symbol_is_not_searched():
    """Yahoo's search needs two characters, and a one-letter query returns
    noise rather than the listing someone meant."""
    matches, error = ts.suggest_alternatives.__wrapped__("V")
    assert matches == ()
    assert error is None
    matches, error = ts.suggest_alternatives.__wrapped__("")
    assert matches == ()


def test_the_suggestion_lookup_is_cached():
    """The failure path calls this on EVERY rerun while an unresolvable
    ticker is loaded, and this page reruns constantly — an uncached call
    would be several Yahoo requests a minute for an unchanged symbol."""
    assert hasattr(ts.suggest_alternatives, "__wrapped__"), (
        "suggest_alternatives must be wrapped in st.cache_data")


def test_the_failure_path_offers_the_alternatives_it_finds():
    from pathlib import Path

    finance = (Path(__file__).resolve().parent.parent
               / "finance.py").read_text(encoding="utf-8")
    assert "ts_suggest_alternatives(ticker_symbol)" in finance
    # ...and switches to the pick through the deferred route, like every
    # other symbol switch on the page.
    assert 'st.session_state["_pending_ticker"] = _fail_m.symbol' in finance
    # The unhelpful advice must not be the whole message any more.
    assert "listed under a venue-specific symbol" in finance


def test_the_failure_message_still_says_what_went_wrong():
    from pathlib import Path

    finance = (Path(__file__).resolve().parent.parent
               / "finance.py").read_text(encoding="utf-8")
    assert "No reliable data found for" in finance
    # And a genuinely unknown symbol still gets the original advice.
    assert "check the ticker symbol." in finance
