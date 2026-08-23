"""Discovery: live movers, sector browsing, and the description box.

Nothing here touches the network. Every screen goes through yf.screen,
so the tests inject a fake in its place and assert on what the module
does with the rows — which is where the real decisions live: dropping OTC
shadows, refusing to invent a figure, and resolving "biotech" to
Biotechnology rather than to Technology.
"""
import ast
import re
from pathlib import Path

import pytest

import ticker_discovery as td
from screener import SECTORS


FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def row(symbol, name="", exchange="NasdaqGS", price=10.0, change=1.5,
        volume=1000, cap=1e9, code="NMS"):
    return {
        "symbol": symbol, "shortName": name, "exchange": code,
        "fullExchangeName": exchange, "regularMarketPrice": price,
        "regularMarketChangePercent": change, "regularMarketVolume": volume,
        "marketCap": cap,
    }


@pytest.fixture(autouse=True)
def no_cache():
    """st.cache_data would serve one test's fake rows to the next."""
    td._trending_cached.clear()
    td._by_field_cached.clear()
    yield
    td._trending_cached.clear()
    td._by_field_cached.clear()


@pytest.fixture
def fake_screen(monkeypatch):
    """Replace yf.screen. Returns a recorder so tests can assert the query."""
    import yfinance

    calls = []

    def install(rows=None, error=None):
        def screen(body, **kwargs):
            calls.append({"body": body, **kwargs})
            if error:
                raise error
            return {"quotes": list(rows or [])}
        monkeypatch.setattr(yfinance, "screen", screen)
        return calls

    return install


# --- OTC shadows --------------------------------------------------------------

def test_otc_shadows_of_a_primary_listing_are_dropped(fake_screen):
    """A sector screen returns argenx as both ARGX (NasdaqGS) and ARGNF
    (OTC Pink). They are one company, and the duplicate pushes a real
    result off an eight-row list."""
    fake_screen([
        row("ARGX", "argenx SE"),
        row("ARGNF", "ARGENX SE", exchange="OTC Markets OTCPK", code="PNK"),
        row("CSLLY", "CSL Ltd.", exchange="OTC Markets OTCQX", code="OQX"),
        row("VRTX", "Vertex Pharmaceuticals"),
    ])
    listings, error = td.by_industry("Biotechnology")
    assert error is None
    assert [l.symbol for l in listings] == ["ARGX", "VRTX"]


@pytest.mark.parametrize("code,full", [
    ("PNK", "OTC Markets OTCPK"), ("OQX", "OTC Markets OTCQX"),
    ("NMS", "OTC Markets Something"),      # name alone is enough
    ("OTC", "Whatever"),                   # code alone is enough
])
def test_every_otc_spelling_is_caught(code, full):
    assert td._is_otc({"exchange": code, "fullExchangeName": full})


def test_a_primary_listing_is_never_mistaken_for_otc():
    for code, full in (("NMS", "NasdaqGS"), ("NYQ", "NYSE"), ("NGM", "NasdaqGM")):
        assert not td._is_otc({"exchange": code, "fullExchangeName": full})


def test_duplicate_symbols_are_collapsed(fake_screen):
    fake_screen([row("AAPL"), row("AAPL"), row("MSFT")])
    listings, _ = td.trending("most_actives")
    assert [l.symbol for l in listings] == ["AAPL", "MSFT"]


# --- never inventing a figure -------------------------------------------------

def test_a_missing_change_is_stated_not_zeroed(fake_screen):
    """0.00% reads as "flat today", which is a claim. The figure being
    absent is a different fact and has to look different."""
    fake_screen([{"symbol": "AAPL", "shortName": "Apple"}])
    listings, _ = td.trending("most_actives")
    assert listings[0].change_pct is None
    assert listings[0].change_text() == "not reported"


def test_a_real_zero_still_reads_as_zero(fake_screen):
    fake_screen([row("AAPL", change=0.0)])
    listings, _ = td.trending("most_actives")
    assert listings[0].change_text() == "+0.00%"


def test_unparseable_numbers_become_none_rather_than_raising(fake_screen):
    fake_screen([row("AAPL", price="n/a", change="", volume="lots", cap=None)])
    listings, error = td.trending("most_actives")
    assert error is None
    only = listings[0]
    assert (only.price, only.change_pct, only.volume, only.market_cap) == (None,) * 4


def test_a_row_with_no_symbol_is_skipped(fake_screen):
    fake_screen([{"shortName": "Nameless"}, row("AAPL")])
    listings, _ = td.trending("most_actives")
    assert [l.symbol for l in listings] == ["AAPL"]


# --- failure is reported, never raised ----------------------------------------

def test_a_network_failure_comes_back_as_a_message(fake_screen):
    fake_screen(error=ConnectionError("boom"))
    listings, error = td.trending("most_actives")
    assert listings == ()
    assert "unavailable" in error.lower() and "ConnectionError" in error


def test_an_empty_response_is_reported(fake_screen):
    fake_screen([])
    listings, error = td.trending("most_actives")
    assert listings == () and error


def test_an_unknown_screen_is_refused_without_a_request(fake_screen):
    calls = fake_screen([row("AAPL")])
    listings, error = td.trending("no_such_screen")
    assert listings == () and "Unknown screen" in error
    assert calls == [], "a bad screen name must not reach the network"


def test_an_unknown_sector_is_refused_without_a_request(fake_screen):
    calls = fake_screen([row("AAPL")])
    listings, error = td.by_sector("Nonsense")
    assert listings == () and "not one of Yahoo" in error
    assert calls == []


# --- the screens themselves ---------------------------------------------------

def test_every_trending_screen_names_what_it_ranks_by():
    """"Trending" on its own is a claim the data does not support. Each
    screen states the quantity it is ordered by."""
    for key, label, basis in td.TRENDING_SCREENS:
        assert label and basis
        assert basis.startswith("by "), basis
    assert set(td.TRENDING_LABELS) == {k for k, _, _ in td.TRENDING_SCREENS}


def test_sector_screens_are_scoped_to_us_and_sorted_by_size(fake_screen):
    calls = fake_screen([row("AAPL")])
    td.by_sector("Technology")
    assert calls[0]["sortField"] == "intradaymarketcap"
    assert calls[0]["sortAsc"] is False
    rendered = str(calls[0]["body"])
    assert "region" in rendered and "us" in rendered
    assert "Technology" in rendered


def test_the_sector_list_is_the_apps_own(fake_screen):
    """A second hard-coded list of Yahoo's eleven sectors would drift from
    the screener's."""
    fake_screen([row("AAPL")])
    for sector in SECTORS:
        _, error = td.by_sector(sector)
        assert error is None, (sector, error)


def test_results_are_capped(fake_screen):
    fake_screen([row(f"T{i}") for i in range(40)])
    listings, _ = td.trending("most_actives", limit=td.TRENDING_LIMIT)
    assert len(listings) == td.TRENDING_LIMIT


# --- the description box ------------------------------------------------------

def test_biotech_beats_tech():
    """The collision the whole ordered-table design exists for: a plain
    substring scan resolves "biotech stocks" to Technology."""
    intent = td.interpret("Show me biotech stocks")
    assert (intent.field, intent.value) == ("industry", "Biotechnology")


@pytest.mark.parametrize("phrase,expected", [
    ("Show me biotech stocks", "Biotechnology"),
    ("tech companies", "Technology"),
    ("I want banks", "Banks - Diversified"),
    ("semiconductor names", "Semiconductors"),
    ("REITs please", "Real Estate"),
    ("healthcare", "Healthcare"),
    ("energy", "Energy"),
])
def test_recognised_phrases_resolve(phrase, expected):
    assert td.interpret(phrase).value == expected


@pytest.mark.parametrize("phrase", [
    "", "   ", "companies that benefit from rate cuts",
    "something with good vibes", "AAPL",
])
def test_an_unrecognised_phrase_returns_none_rather_than_guessing(phrase):
    """Silently answering a question you did not understand is worse than
    declining it — the caller shows a message saying so."""
    assert td.interpret(phrase) is None


def test_for_description_reports_no_match_distinctly(fake_screen):
    fake_screen([row("AAPL")])
    listings, error, intent = td.for_description("rate cuts")
    assert (listings, error, intent) == ((), None, None)


def test_for_description_quotes_back_the_word_it_matched(fake_screen):
    """So the reader can see WHY they got these results, and correct it."""
    fake_screen([row("VRTX")])
    _, _, intent = td.for_description("any good biotech ideas?")
    assert intent.matched == "biotech"
    assert intent.label == "Biotechnology"


def test_every_description_term_points_at_a_real_field():
    for term, field, value, label in td.DESCRIPTION_TERMS:
        assert field in ("sector", "industry"), term
        assert term == term.lower(), term
        assert label
        if field == "sector":
            assert value in SECTORS, f"{term} -> {value} is not a Yahoo sector"


# --- the panel ----------------------------------------------------------------

@pytest.fixture(scope="module")
def finance_src() -> str:
    return FINANCE.read_text(encoding="utf-8")


def test_discovery_buttons_do_not_use_primary(finance_src):
    """The active-ticker styling is scoped by widget-key prefix in the CSS
    block; a fourth surface using type="primary" without being added to
    that scope renders in Streamlit's stock red. The login page shipped
    exactly that bug."""
    start = finance_src.index('def _ts_pick_button')
    body = finance_src[start:finance_src.index("# Recently viewed", start)]
    assert 'type="primary"' not in body
    assert "· current" in body


def test_the_current_ticker_is_dropped_from_recently_viewed(finance_src):
    """It is already in the header in large type; spending one of five
    slots to repeat it is the waste the quick-stats defaults were trimmed
    for."""
    start = finance_src.index("# Recently viewed")
    body = finance_src[start:finance_src.index("# Live movers", start)]
    assert "!= ticker_symbol" in body
    assert "td.RECENTS_LIMIT" in body


def test_each_discovery_list_uses_its_own_key_prefix(finance_src):
    """One symbol can appear in the movers AND in a sector list in the
    same run — MRNA did, live. Shared keys would raise DuplicateWidgetID,
    which bare-mode tests cannot catch."""
    # The shared helper takes the prefix as an argument, so these appear
    # at the call sites rather than inside a key= expression.
    prefixes = set(re.findall(r'"(ts_(?:trend|sect|desc|recent)_)"', finance_src))
    assert prefixes == {"ts_trend_", "ts_sect_", "ts_desc_", "ts_recent_"}
    # ...and the helper really does interpolate it into the key.
    assert 'key=f"{_key_prefix}{_listing.symbol}"' in finance_src


def test_the_panel_never_calls_the_movers_trending(finance_src):
    """The word claims something the ranking does not establish."""
    # Only what the reader SEES. td.trending() is the function's name and
    # says nothing to the user; the radio label and captions do.
    start = finance_src.index("# Live movers")
    end = finance_src.index("# Browse by sector", start)
    shown = [n.value for n in ast.walk(ast.parse(finance_src))
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and getattr(n, "col_offset", None) is not None
             and start <= _offset_of(finance_src, n) < end]
    # Widget keys and key prefixes are identifiers, not prose — "ts_trend_"
    # is never read by anyone.
    shown = [t for t in shown if not re.fullmatch(r"ts_[a-z_]+", t)]
    assert shown, "no user-facing strings found in the movers block"
    for text in shown:
        assert "trend" not in text.lower(), f"user-facing string says trending: {text!r}"


def _offset_of(source: str, node) -> int:
    """Character offset of an ast node, for slicing against a text index."""
    lines = source.splitlines(keepends=True)
    return sum(len(l) for l in lines[:node.lineno - 1]) + node.col_offset
