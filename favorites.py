"""Favorites (starred tickers) and recently-viewed symbols — the two
halves of the quick-access strip under the symbol header.

WHY THIS IS SEPARATE FROM THE SIDEBAR WATCHLISTS: watchlist_panel.py's
lists are multiple, named and curated ("Dividend Payers", "Tech"), and
exactly one is active at a time. A pin action against "whichever list is
active" would therefore mean something different depending on where you
happened to be, and would quietly dirty a basket the user built on
purpose. Favorites are instead one flat set meaning "always offer me
these, whichever watchlist I'm in." That split was settled with the user
before this module was written rather than assumed.

WHAT CHANGED ABOUT RECENTS: the recently-viewed strip used to be
deliberately session-only, on the reasoning that it's a byproduct of
navigating rather than a list anyone asked to keep ("why do old tickers
I glanced at keep coming back"). The originating task for this module
explicitly requires recents to persist across sessions, so that earlier
call is deliberately reversed here — and the original concern is
answered instead by the "Clear recents" control in the UI, so the list
is durable but never something you're stuck with.

The MRU ordering itself still lives in watchlist_panel.record_recent():
that's a pure move-to-front/dedupe/cap helper with its own tests, and
this module owns only the persistence of the result. Both stores live in
one file, since they're written together on the same interactions.

Persisted with the same atomic-write, gitignored-local-file pattern
every other piece of cross-restart state in this app uses (see
local_store.py). Quantix has no accounts, so this is a single shared
store for whoever runs this instance, not per-user.
"""
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

from config import FAVORITES
from local_store import atomic_write_text
from logging_setup import get_logger, log_exception

logger = get_logger("favorites")


@dataclass(frozen=True)
class QuickAccessStore:
    favorites: Tuple[str, ...] = ()
    recents: Tuple[str, ...] = ()


def _store_path() -> Path:
    return Path(__file__).resolve().parent / FAVORITES.store_filename


def load_store(path: Optional[Path] = None) -> QuickAccessStore:
    """Never raises: a missing file (fresh install) or a corrupt/
    unreadable one both degrade to an empty store rather than crashing
    the app on load — the same graceful-degradation contract every other
    local store in this app honours."""
    path = path or _store_path()
    if not path.exists():
        return QuickAccessStore()
    try:
        raw = json.loads(path.read_text())
        return QuickAccessStore(
            favorites=tuple(raw.get("favorites", ())),
            recents=tuple(raw.get("recents", ())),
        )
    except Exception:
        log_exception(logger, "favorites.store_corrupt", section="favorites")
        return QuickAccessStore()


def save_store(store: QuickAccessStore, path: Optional[Path] = None) -> None:
    path = path or _store_path()
    payload = {"favorites": list(store.favorites), "recents": list(store.recents)}
    atomic_write_text(path, json.dumps(payload, indent=2))


def is_favorite(store: QuickAccessStore, ticker: str) -> bool:
    return bool(ticker) and ticker in store.favorites


def toggle_favorite(
    store: QuickAccessStore, ticker: str, max_favorites: Optional[int] = None,
) -> Tuple[QuickAccessStore, Optional[str]]:
    """Star an unstarred ticker, or unstar a starred one. Returns
    (updated_store, error_message); error is None on success.

    Unstarring is always allowed even at/over the cap — only ADDING is
    capped, so a user who somehow ends up over the limit (a hand-edited
    store file, or a lowered cap in a later version) can still dig
    themselves out rather than being stuck.

    New favorites are appended, not prepended: this is a curated set, so
    a stable order means the chip you reach for doesn't move under your
    cursor every time you star something else.
    """
    max_favorites = FAVORITES.max_favorites if max_favorites is None else max_favorites
    if not ticker:
        return store, "No ticker to favorite."
    if ticker in store.favorites:
        return replace(store, favorites=tuple(t for t in store.favorites if t != ticker)), None
    if len(store.favorites) >= max_favorites:
        return store, f"Favorites are full ({max_favorites}). Unstar one first."
    return replace(store, favorites=store.favorites + (ticker,)), None


def quick_access_chips(store: QuickAccessStore, max_chips: Optional[int] = None) -> Tuple[Tuple[str, bool], ...]:
    """The one merged row rendered under the symbol header, as
    (ticker, is_favorite) pairs: every favorite first (in their stable
    starred order), then recents not already starred.

    Merged into a single row rather than two stacked ones on purpose —
    this block sits inside the STICKY symbol header, so every extra row
    permanently costs vertical space on every screen. One row also
    matches what the strip is actually for: "everywhere I can jump to
    from here", with the pinned ones leading.
    """
    max_chips = FAVORITES.max_chips if max_chips is None else max_chips
    if max_chips <= 0:
        return ()
    chips = [(t, True) for t in store.favorites]
    chips += [(t, False) for t in store.recents if t not in store.favorites]
    return tuple(chips[:max_chips])
