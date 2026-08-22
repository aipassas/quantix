"""Tests for watchlist_panel.py — the sidebar quick-switch watchlist's
ticker-list editing, multi-list persistence, and quote-snapshot shaping.

The list-editing helpers are pure functions over tuples, so they're tested
directly. _load_quote() is exercised through a stubbed load_ticker_bundle
rather than the network, so the day-change math and the never-fabricate
fallbacks are verified deterministically.
"""
import pytest

import watchlist_panel
from config import WATCHLIST_PANEL
from watchlist_panel import (
    QuoteSnapshot,
    SavedWatchlist,
    WatchlistStore,
    add_ticker,
    create_watchlist,
    delete_watchlist,
    load_watchlist_store,
    parse_tickers,
    record_recent,
    remove_ticker,
    rename_watchlist,
    save_watchlist_store,
    set_active_watchlist,
    update_active_tickers,
)


# --- parse_tickers ----------------------------------------------------------

def test_parse_tickers_handles_commas_spaces_and_case():
    assert parse_tickers("aapl, msft  nvda") == ("AAPL", "MSFT", "NVDA")


def test_parse_tickers_dedupes_preserving_first_position():
    assert parse_tickers("AAPL, MSFT, AAPL") == ("AAPL", "MSFT")


def test_parse_tickers_empty_input_is_empty_tuple():
    assert parse_tickers("   ,  , ") == ()


# --- add_ticker -------------------------------------------------------------

def test_add_ticker_appends_to_end():
    new, err = add_ticker(("AAPL",), "MSFT", max_tickers=10)
    assert new == ("AAPL", "MSFT")
    assert err is None


def test_add_ticker_rejects_empty_input_with_reason():
    current = ("AAPL",)
    new, err = add_ticker(current, "  ", max_tickers=10)
    assert new == current and err  # unchanged, and an explicit reason given


def test_add_ticker_rejects_duplicate_with_reason():
    current = ("AAPL", "MSFT")
    new, err = add_ticker(current, "msft", max_tickers=10)
    assert new == current
    assert "MSFT" in err


def test_add_ticker_rejects_when_full():
    current = ("A", "B", "C")
    new, err = add_ticker(current, "D", max_tickers=3)
    assert new == current
    assert "full" in err.lower()


def test_add_ticker_rejects_batch_that_would_overflow_rather_than_truncating():
    """A partial add would silently drop tickers the user asked for —
    reject the whole batch with a reason instead."""
    current = ("A", "B")
    new, err = add_ticker(current, "C, D, E", max_tickers=4)
    assert new == current
    assert "room for 2" in err


def test_add_ticker_accepts_batch_that_exactly_fills():
    new, err = add_ticker(("A",), "B, C", max_tickers=3)
    assert new == ("A", "B", "C")
    assert err is None


# --- remove_ticker ----------------------------------------------------------

def test_remove_ticker_removes_only_the_named_one():
    assert remove_ticker(("AAPL", "MSFT", "NVDA"), "MSFT") == ("AAPL", "NVDA")


def test_remove_ticker_absent_is_a_no_op():
    current = ("AAPL",)
    assert remove_ticker(current, "TSLA") == current


# --- direction_icon ---------------------------------------------------------

@pytest.mark.parametrize("change_pct,expected", [
    (1.5, "▲"), (-1.5, "▼"), (0.0, ""), (None, ""),
])
def test_direction_icon_matches_sign(change_pct, expected):
    assert QuoteSnapshot(ticker="X", change_pct=change_pct).direction_icon == expected


# --- _load_quote (stubbed bundle, no network) -------------------------------

class _StubBundle:
    def __init__(self, info=None, errors=None):
        self.info = info or {}
        self.errors = errors or []


def _patch_bundle(monkeypatch, bundle):
    monkeypatch.setattr(watchlist_panel, "load_ticker_bundle", lambda t, deep=False: bundle)


def _uncached_load_quote(ticker):
    """Call the undecorated function so st.cache_data doesn't serve a
    previous test's stubbed result for the same ticker argument."""
    return watchlist_panel._load_quote.__wrapped__(ticker)


def test_load_quote_computes_change_from_price_and_previous_close(monkeypatch):
    """Day change is derived, never read from Yahoo's own percent field —
    here that field is deliberately WRONG to prove it isn't consulted."""
    _patch_bundle(monkeypatch, _StubBundle({
        "currentPrice": 110.0, "previousClose": 100.0,
        "trailingPE": 20.0, "regularMarketChangePercent": -99.0,
    }))
    q = _uncached_load_quote("TEST")
    assert q.status == "ok"
    assert q.change_pct == pytest.approx(10.0)
    assert q.pe_ratio == 20.0
    assert q.direction_icon == "▲"


def test_load_quote_negative_change(monkeypatch):
    _patch_bundle(monkeypatch, _StubBundle({"currentPrice": 90.0, "previousClose": 100.0}))
    q = _uncached_load_quote("TEST")
    assert q.change_pct == pytest.approx(-10.0)
    assert q.direction_icon == "▼"


def test_load_quote_falls_back_to_regular_market_fields(monkeypatch):
    _patch_bundle(monkeypatch, _StubBundle({
        "regularMarketPrice": 50.0, "regularMarketPreviousClose": 40.0,
    }))
    q = _uncached_load_quote("TEST")
    assert q.status == "ok"
    assert q.change_pct == pytest.approx(25.0)


def test_load_quote_unavailable_when_info_empty(monkeypatch):
    _patch_bundle(monkeypatch, _StubBundle({}, errors=["not found"]))
    q = _uncached_load_quote("BADTICKER")
    assert q.status == "unavailable"
    assert q.change_pct is None
    assert q.detail  # a real, disclosed reason
    assert q.direction_icon == ""


def test_load_quote_unavailable_when_previous_close_missing(monkeypatch):
    """Price alone can't produce a day change — must not fabricate 0.00%."""
    _patch_bundle(monkeypatch, _StubBundle({"currentPrice": 100.0}))
    q = _uncached_load_quote("TEST")
    assert q.status == "unavailable"
    assert q.change_pct is None
    assert q.price == 100.0  # what IS known is still reported


def test_load_quote_unavailable_when_previous_close_is_zero(monkeypatch):
    """A zero previous close would be a divide-by-zero, not a 0% day."""
    _patch_bundle(monkeypatch, _StubBundle({"currentPrice": 100.0, "previousClose": 0}))
    q = _uncached_load_quote("TEST")
    assert q.status == "unavailable"
    assert q.change_pct is None


def test_load_quote_never_raises_on_unexpected_error(monkeypatch):
    def _boom(t, deep=False):
        raise RuntimeError("network exploded")
    monkeypatch.setattr(watchlist_panel, "load_ticker_bundle", _boom)
    q = _uncached_load_quote("TEST")
    assert q.status == "unavailable"
    assert q.detail


# --- record_recent (recently-viewed strip) ----------------------------------

def test_record_recent_puts_new_ticker_at_front():
    assert record_recent(("MSFT", "NVDA"), "AAPL", max_recent=8) == ("AAPL", "MSFT", "NVDA")


def test_record_recent_is_idempotent_for_the_current_ticker():
    """Streamlit re-runs the script on every widget interaction, so this is
    called repeatedly with the same ticker — it must not duplicate or churn."""
    seq = ("AAPL", "MSFT")
    once = record_recent(seq, "AAPL", max_recent=8)
    twice = record_recent(once, "AAPL", max_recent=8)
    assert once == twice == ("AAPL", "MSFT")


def test_record_recent_moves_an_existing_ticker_to_front_without_duplicating():
    assert record_recent(("AAPL", "MSFT", "NVDA"), "NVDA", max_recent=8) == ("NVDA", "AAPL", "MSFT")


def test_record_recent_caps_length_dropping_the_oldest():
    result = record_recent(("B", "C", "D"), "A", max_recent=3)
    assert result == ("A", "B", "C")  # "D", the least recent, falls off


def test_record_recent_cap_of_one_keeps_only_current():
    assert record_recent(("B", "C"), "A", max_recent=1) == ("A",)


def test_record_recent_empty_ticker_leaves_list_untouched():
    current = ("AAPL",)
    assert record_recent(current, "", max_recent=8) == current


def test_record_recent_non_positive_cap_yields_empty():
    assert record_recent(("AAPL",), "MSFT", max_recent=0) == ()


# --- Multiple saved watchlists (persistence) ---------------------------------

def test_fresh_store_seeds_one_default_watchlist(tmp_path):
    store = load_watchlist_store(tmp_path / "nope.json")
    assert store.active == WATCHLIST_PANEL.default_watchlist_name
    assert list(store.lists.keys()) == [WATCHLIST_PANEL.default_watchlist_name]
    assert store.lists[store.active].tickers == tuple(WATCHLIST_PANEL.default_tickers)


def test_load_store_corrupt_file_degrades_to_default_not_raise(tmp_path):
    path = tmp_path / "store.json"
    path.write_text("{not valid json")
    store = load_watchlist_store(path)
    assert store.active == WATCHLIST_PANEL.default_watchlist_name


def test_save_and_load_round_trip_multiple_lists(tmp_path):
    path = tmp_path / "store.json"
    store = WatchlistStore(active="B", lists={
        "A": SavedWatchlist(name="A", tickers=("AAPL",), created_at="t1"),
        "B": SavedWatchlist(name="B", tickers=("KO", "JNJ"), created_at="t2"),
    })
    save_watchlist_store(store, path)
    reloaded = load_watchlist_store(path)
    assert reloaded.active == "B"
    assert reloaded.lists["A"].tickers == ("AAPL",)
    assert reloaded.lists["B"].tickers == ("KO", "JNJ")


def test_load_store_falls_back_when_persisted_active_no_longer_exists(tmp_path):
    """A store file could in principle name an active list that isn't in
    `lists` (e.g. hand-edited) — must recover to SOME real list, never
    crash or point at a list that doesn't exist."""
    path = tmp_path / "store.json"
    path.write_text('{"active": "Ghost", "lists": {"Real": {"name": "Real", "tickers": ["AAPL"], "created_at": "t"}}}')
    store = load_watchlist_store(path)
    assert store.active == "Real"


# --- create_watchlist ---------------------------------------------------------

def test_create_watchlist_becomes_active():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = create_watchlist(store, "B")
    assert err is None
    assert new_store.active == "B"
    assert "B" in new_store.lists
    assert new_store.lists["B"].tickers == ()


def test_create_watchlist_rejects_empty_name():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = create_watchlist(store, "   ")
    assert err
    assert new_store is store


def test_create_watchlist_rejects_duplicate_name():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = create_watchlist(store, "A")
    assert "already exists" in err
    assert new_store is store


def test_create_watchlist_rejects_beyond_max():
    lists = {f"L{i}": SavedWatchlist(name=f"L{i}") for i in range(WATCHLIST_PANEL.max_watchlists)}
    store = WatchlistStore(active="L0", lists=lists)
    new_store, err = create_watchlist(store, "One More")
    assert "limit reached" in err
    assert new_store is store


# --- rename_watchlist ----------------------------------------------------------

def test_rename_watchlist_preserves_tickers():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A", tickers=("AAPL", "MSFT"))})
    new_store, err = rename_watchlist(store, "A", "Tech")
    assert err is None
    assert "Tech" in new_store.lists and "A" not in new_store.lists
    assert new_store.lists["Tech"].tickers == ("AAPL", "MSFT")


def test_rename_watchlist_updates_active_when_renaming_the_active_one():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = rename_watchlist(store, "A", "B")
    assert err is None
    assert new_store.active == "B"


def test_rename_watchlist_does_not_change_active_when_renaming_a_different_list():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A"), "B": SavedWatchlist(name="B")})
    new_store, err = rename_watchlist(store, "B", "C")
    assert err is None
    assert new_store.active == "A"


def test_rename_watchlist_rejects_collision_with_another_list():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A"), "B": SavedWatchlist(name="B")})
    new_store, err = rename_watchlist(store, "A", "B")
    assert "already exists" in err
    assert new_store is store


def test_rename_watchlist_missing_source_is_a_clean_error():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = rename_watchlist(store, "Ghost", "B")
    assert "does not exist" in err


# --- delete_watchlist ------------------------------------------------------------

def test_delete_watchlist_removes_it():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A"), "B": SavedWatchlist(name="B")})
    new_store, err = delete_watchlist(store, "B")
    assert err is None
    assert list(new_store.lists.keys()) == ["A"]


def test_delete_watchlist_refuses_to_delete_the_last_one():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = delete_watchlist(store, "A")
    assert "last watchlist" in err
    assert new_store is store


def test_delete_active_watchlist_reassigns_active_to_a_survivor():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A"), "B": SavedWatchlist(name="B")})
    new_store, err = delete_watchlist(store, "A")
    assert err is None
    assert new_store.active == "B"


def test_delete_inactive_watchlist_leaves_active_unchanged():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A"), "B": SavedWatchlist(name="B")})
    new_store, err = delete_watchlist(store, "B")
    assert new_store.active == "A"


# --- set_active_watchlist -------------------------------------------------------

def test_set_active_watchlist_switches():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A"), "B": SavedWatchlist(name="B")})
    new_store, err = set_active_watchlist(store, "B")
    assert err is None
    assert new_store.active == "B"


def test_set_active_watchlist_missing_name_is_a_clean_error():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A")})
    new_store, err = set_active_watchlist(store, "Ghost")
    assert "does not exist" in err
    assert new_store is store


# --- update_active_tickers -------------------------------------------------------

def test_update_active_tickers_only_touches_the_active_list():
    store = WatchlistStore(active="A", lists={"A": SavedWatchlist(name="A", tickers=("AAPL",)), "B": SavedWatchlist(name="B", tickers=("KO",))})
    new_store = update_active_tickers(store, ("AAPL", "MSFT"))
    assert new_store.lists["A"].tickers == ("AAPL", "MSFT")
    assert new_store.lists["B"].tickers == ("KO",)  # untouched
