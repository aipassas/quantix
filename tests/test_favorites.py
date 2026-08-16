"""Tests for favorites.py — the starred-ticker set and the persisted
recently-viewed list behind the quick-access strip.
"""
from config import FAVORITES
from favorites import (
    QuickAccessStore,
    is_favorite,
    load_store,
    quick_access_chips,
    save_store,
    toggle_favorite,
)


# --- persistence -----------------------------------------------------------------

def test_fresh_instance_has_empty_store(tmp_path):
    store = load_store(tmp_path / "nope.json")
    assert store.favorites == ()
    assert store.recents == ()


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "favorites.json"
    save_store(QuickAccessStore(favorites=("AAPL", "MSFT"), recents=("NVDA", "AAPL")), path)
    loaded = load_store(path)
    assert loaded.favorites == ("AAPL", "MSFT")
    assert loaded.recents == ("NVDA", "AAPL")


def test_load_corrupt_file_degrades_to_empty_not_raise(tmp_path):
    path = tmp_path / "favorites.json"
    path.write_text("{not valid json")
    assert load_store(path) == QuickAccessStore()


def test_load_tolerates_a_partial_store_file(tmp_path):
    """A file written by an older/newer version that only carries one of
    the two lists must still load, with the missing half empty."""
    path = tmp_path / "favorites.json"
    path.write_text('{"favorites": ["AAPL"]}')
    loaded = load_store(path)
    assert loaded.favorites == ("AAPL",)
    assert loaded.recents == ()


def test_save_leaves_no_leftover_temp_file(tmp_path):
    path = tmp_path / "favorites.json"
    save_store(QuickAccessStore(favorites=("AAPL",)), path)
    assert [p.name for p in tmp_path.iterdir()] == ["favorites.json"]


# --- favorites mutations -----------------------------------------------------------------

def test_toggle_adds_then_removes(tmp_path):
    store = QuickAccessStore()
    store, error = toggle_favorite(store, "AAPL")
    assert error is None
    assert store.favorites == ("AAPL",)
    assert is_favorite(store, "AAPL")

    store, error = toggle_favorite(store, "AAPL")
    assert error is None
    assert store.favorites == ()
    assert not is_favorite(store, "AAPL")


def test_new_favorites_append_so_existing_chips_do_not_move():
    """Stable order is the point: starring something new must not
    reshuffle the chip the user was already reaching for."""
    store = QuickAccessStore(favorites=("AAPL", "MSFT"))
    store, _ = toggle_favorite(store, "NVDA")
    assert store.favorites == ("AAPL", "MSFT", "NVDA")


def test_toggle_is_capped_and_reports_why():
    store = QuickAccessStore(favorites=tuple(f"T{i}" for i in range(FAVORITES.max_favorites)))
    store_after, error = toggle_favorite(store, "NEW")
    assert store_after == store  # unchanged
    assert error is not None and "full" in error.lower()


def test_unstarring_still_works_when_at_or_over_the_cap():
    """Only ADDING is capped — someone already at/over the limit (hand-edited
    store, or a cap lowered by a later version) must still be able to dig out."""
    over = tuple(f"T{i}" for i in range(FAVORITES.max_favorites + 3))
    store, error = toggle_favorite(QuickAccessStore(favorites=over), "T0")
    assert error is None
    assert "T0" not in store.favorites
    assert len(store.favorites) == len(over) - 1


def test_toggle_rejects_an_empty_ticker():
    store, error = toggle_favorite(QuickAccessStore(), "")
    assert store == QuickAccessStore()
    assert error is not None


def test_is_favorite_is_false_for_empty_ticker():
    assert is_favorite(QuickAccessStore(favorites=("AAPL",)), "") is False


# --- the merged chip row -----------------------------------------------------------------

def test_favorites_lead_and_recents_follow():
    store = QuickAccessStore(favorites=("AAPL", "MSFT"), recents=("NVDA", "GOOGL"))
    assert quick_access_chips(store) == (
        ("AAPL", True), ("MSFT", True), ("NVDA", False), ("GOOGL", False),
    )


def test_a_starred_ticker_is_not_also_shown_as_a_recent():
    """AAPL being both starred and recently viewed is the common case —
    it must render once, as a favorite, not twice."""
    store = QuickAccessStore(favorites=("AAPL",), recents=("NVDA", "AAPL", "GOOGL"))
    chips = quick_access_chips(store)
    assert [t for t, _ in chips] == ["AAPL", "NVDA", "GOOGL"]
    assert chips[0] == ("AAPL", True)


def test_chip_row_is_capped():
    store = QuickAccessStore(
        favorites=tuple(f"F{i}" for i in range(FAVORITES.max_favorites)),
        recents=tuple(f"R{i}" for i in range(20)),
    )
    assert len(quick_access_chips(store)) == FAVORITES.max_chips


def test_favorites_are_never_crowded_out_of_the_capped_row():
    """The cap trims recents first — a starred ticker must not vanish
    from the strip just because a lot of symbols were visited."""
    store = QuickAccessStore(
        favorites=("AAPL", "MSFT"),
        recents=tuple(f"R{i}" for i in range(20)),
    )
    chips = quick_access_chips(store, max_chips=4)
    assert chips[0] == ("AAPL", True)
    assert chips[1] == ("MSFT", True)
    assert len(chips) == 4


def test_empty_store_renders_no_chips():
    assert quick_access_chips(QuickAccessStore()) == ()


def test_non_positive_cap_renders_no_chips():
    store = QuickAccessStore(favorites=("AAPL",), recents=("MSFT",))
    assert quick_access_chips(store, max_chips=0) == ()
