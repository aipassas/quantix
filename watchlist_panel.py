"""Symbol navigation aids — ways to change the analyzed ticker without
retyping it. (Starred favorites are a third, and live in favorites.py.)

1. The sidebar watchlist: one or more NAMED, user-maintained lists of
   tickers (create/rename/delete, one "active" at a time), each with a
   live quote line per ticker, rendered so it stays visible alongside
   whichever analysis panel is open. Persisted to a local JSON file —
   the same atomic-write, gitignored-local-file pattern every other
   piece of cross-restart state in this app already uses (see
   realtime_alerts.py / ml_pipeline.py / scenario_modeling.py /
   onboarding.py) — so lists and which one you were looking at survive
   an app restart. Quantix has no accounts, so this is a single shared
   store for whoever runs this instance, not per-user.
2. The recently-viewed half of the quick-access strip: an automatic
   most-recently-used list of visited symbols, rendered as chips under
   the symbol header alongside starred favorites. record_recent() below
   is the pure move-to-front/dedupe/cap helper that computes the
   ordering; favorites.py owns PERSISTING the result (and the favorites
   set it renders next to). That split is deliberate — the ordering rule
   is generic list arithmetic with its own tests here, while the storage
   concern belongs with the other half of the strip it's stored beside.

   Note this list used to be session-only, on the reasoning that it's a
   byproduct of navigating rather than something anyone asked to keep.
   The Favorites & Quick Access task explicitly required it to survive
   restarts, so that call was reversed; see favorites.py for the
   reversal and for the "Clear recents" control that answers the
   original concern.

Deliberately reuses data_loader.load_ticker_bundle(deep=False), the exact
same shallow (info-only) fetch the Institutional Watchlist scan and the
Peer Comparison already use — so a ticker already loaded elsewhere this
session is served from that fetch's own 30-minute cache rather than
costing a new Yahoo call.

Day-change convention: computed here as (price - previous_close) /
previous_close, NOT read from Yahoo's own `regularMarketChangePercent`.
Yahoo's percent fields have historically been inconsistent about whether
they're a fraction or already-multiplied percent (the same class of
scaling ambiguity financial_standardization.py documents for
`debtToEquity`), and both inputs needed for the honest calculation are
right there in the same payload — so this computes it rather than trusting
a field whose scale can't be verified per-ticker.

Never-fabricate convention: a ticker Yahoo can't resolve, or one missing
either price leg, comes back with status != "ok" and None values, and the
panel renders it as an explicit unavailable row — never silently dropped
and never shown with a made-up 0.00.
"""
import datetime
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from config import WATCHLIST_PANEL
from data_loader import load_ticker_bundle
from local_store import atomic_write_text
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("watchlist_panel")


@dataclass(frozen=True)
class QuoteSnapshot:
    """One watchlist row. `status` is "ok" only when price AND
    previous_close both resolved, since day-change needs both."""
    ticker: str
    price: Optional[float] = None
    previous_close: Optional[float] = None
    change_pct: Optional[float] = None
    pe_ratio: Optional[float] = None
    status: str = "ok"          # "ok" | "unavailable"
    detail: str = ""

    @property
    def direction_icon(self) -> str:
        """Green/red/white ball matching the app's status-colour vocabulary.
        White for an unavailable quote or a genuinely flat day — never a
        directional colour for a direction that isn't there."""
        if self.change_pct is None:
            return "⚪"
        if self.change_pct > 0:
            return "🟢"
        if self.change_pct < 0:
            return "🔴"
        return "⚪"


def parse_tickers(raw: str) -> Tuple[str, ...]:
    """Split a comma/space-separated string into deduplicated uppercase
    ticker symbols, preserving the order they were typed in."""
    parts = [p.strip().upper() for p in raw.replace(",", " ").split()]
    return tuple(dict.fromkeys(p for p in parts if p))


def add_ticker(current: Tuple[str, ...], raw: str, max_tickers: int) -> Tuple[Tuple[str, ...], Optional[str]]:
    """Append parsed ticker(s) to `current`, returning (new_list, error).

    Returns the list UNCHANGED with an error message when the input is
    empty, already present, or would exceed `max_tickers` — the caller
    surfaces that message rather than silently no-op'ing, so a click that
    appears to do nothing always has a stated reason.
    """
    incoming = parse_tickers(raw)
    if not incoming:
        return current, "Enter a ticker symbol first."

    already = [t for t in incoming if t in current]
    if already:
        return current, f"Already on the watchlist: {', '.join(already)}."

    room = max_tickers - len(current)
    if room <= 0:
        return current, f"Watchlist is full ({max_tickers} max) — remove one first."
    if len(incoming) > room:
        return current, f"Only room for {room} more ticker(s) ({max_tickers} max)."

    return current + incoming, None


def remove_ticker(current: Tuple[str, ...], ticker: str) -> Tuple[str, ...]:
    return tuple(t for t in current if t != ticker)


def record_recent(current: Tuple[str, ...], ticker: str, max_recent: int) -> Tuple[str, ...]:
    """Most-recently-used list with `ticker` moved to the front.

    Idempotent by design: Streamlit re-runs the whole script on every
    widget interaction (a slider nudge, a tab click), so this is called
    constantly with the SAME ticker. Re-recording the current symbol must
    therefore be a no-op rather than duplicating it or churning the order
    — which is why this moves-to-front and dedupes instead of appending.

    Returns `current` unchanged for an empty ticker, and an empty tuple
    for a non-positive `max_recent`, rather than raising.
    """
    if max_recent <= 0:
        return ()
    if not ticker:
        return current
    rest = tuple(t for t in current if t != ticker)
    return (ticker,) + rest[:max_recent - 1]


# --- Multiple saved, named watchlists (persisted) -----------------------------

@dataclass(frozen=True)
class SavedWatchlist:
    name: str
    tickers: Tuple[str, ...] = ()
    created_at: str = ""


@dataclass(frozen=True)
class WatchlistStore:
    """`active` names whichever list is currently shown/quick-switchable
    in the sidebar — persisted alongside the lists themselves, so which
    watchlist you were looking at survives a restart too, not just the
    lists' contents."""
    active: str
    lists: Dict[str, SavedWatchlist] = field(default_factory=dict)


def _watchlist_store_path() -> Path:
    return Path(__file__).resolve().parent / WATCHLIST_PANEL.store_filename


def _seed_default_store() -> WatchlistStore:
    """The FIRST-EVER store, when no save file exists yet — one named
    list seeded from WATCHLIST_PANEL.default_tickers, preserving exactly
    what a fresh install showed before this task added persistence."""
    name = WATCHLIST_PANEL.default_watchlist_name
    return WatchlistStore(active=name, lists={
        name: SavedWatchlist(name=name, tickers=tuple(WATCHLIST_PANEL.default_tickers), created_at=datetime.datetime.now().isoformat(timespec="seconds")),
    })


def load_watchlist_store(path: Optional[Path] = None) -> WatchlistStore:
    """Never raises: a missing file seeds the default store (a fresh
    install); a corrupt one degrades to the same default rather than
    crashing the app on load."""
    path = path or _watchlist_store_path()
    if not path.exists():
        return _seed_default_store()
    try:
        raw = json.loads(path.read_text())
        lists = {name: SavedWatchlist(name=v["name"], tickers=tuple(v["tickers"]), created_at=v.get("created_at", "")) for name, v in raw["lists"].items()}
        active = raw["active"] if raw.get("active") in lists else next(iter(lists), None)
        if active is None:
            return _seed_default_store()
        return WatchlistStore(active=active, lists=lists)
    except Exception:
        log_exception(logger, "watchlist_store.corrupt", section="watchlist_panel")
        return _seed_default_store()


def save_watchlist_store(store: WatchlistStore, path: Optional[Path] = None) -> None:
    """Atomic write (temp file + rename), same pattern as every other
    local store in this app."""
    path = path or _watchlist_store_path()
    payload = {
        "active": store.active,
        "lists": {name: {"name": wl.name, "tickers": list(wl.tickers), "created_at": wl.created_at} for name, wl in store.lists.items()},
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


def create_watchlist(store: WatchlistStore, name: str) -> Tuple[WatchlistStore, Optional[str]]:
    """Returns (store, reason) UNCHANGED with a stated reason for an
    empty/duplicate name or the max-lists cap — the caller surfaces that
    message rather than the click silently doing nothing. The new list
    becomes active immediately (matches create_watchlist's obvious intent
    — you just made it to look at it)."""
    name = name.strip()
    if not name:
        return store, "Enter a name first."
    if name in store.lists:
        return store, f'A watchlist named "{name}" already exists.'
    if len(store.lists) >= WATCHLIST_PANEL.max_watchlists:
        return store, f"Watchlist limit reached ({WATCHLIST_PANEL.max_watchlists} max) — delete one first."
    new_lists = dict(store.lists)
    new_lists[name] = SavedWatchlist(name=name, tickers=(), created_at=datetime.datetime.now().isoformat(timespec="seconds"))
    return WatchlistStore(active=name, lists=new_lists), None


def rename_watchlist(store: WatchlistStore, old_name: str, new_name: str) -> Tuple[WatchlistStore, Optional[str]]:
    new_name = new_name.strip()
    if not new_name:
        return store, "Enter a name first."
    if old_name not in store.lists:
        return store, f'"{old_name}" does not exist.'
    if new_name != old_name and new_name in store.lists:
        return store, f'A watchlist named "{new_name}" already exists.'
    new_lists = {}
    for key, wl in store.lists.items():
        if key == old_name:
            new_lists[new_name] = replace(wl, name=new_name)
        else:
            new_lists[key] = wl
    new_active = new_name if store.active == old_name else store.active
    return WatchlistStore(active=new_active, lists=new_lists), None


def delete_watchlist(store: WatchlistStore, name: str) -> Tuple[WatchlistStore, Optional[str]]:
    """Refuses to delete the LAST remaining watchlist — there must always
    be at least one to be "active," and silently recreating a default one
    afterward would be a confusing surprise."""
    if name not in store.lists:
        return store, f'"{name}" does not exist.'
    if len(store.lists) <= 1:
        return store, "Can't delete your last watchlist."
    new_lists = {k: v for k, v in store.lists.items() if k != name}
    new_active = store.active if store.active != name else next(iter(new_lists))
    return WatchlistStore(active=new_active, lists=new_lists), None


def set_active_watchlist(store: WatchlistStore, name: str) -> Tuple[WatchlistStore, Optional[str]]:
    if name not in store.lists:
        return store, f'"{name}" does not exist.'
    return WatchlistStore(active=name, lists=store.lists), None


def update_active_tickers(store: WatchlistStore, tickers: Tuple[str, ...]) -> WatchlistStore:
    """Writes a new ticker tuple back into whichever list is currently
    active — used after add_ticker()/remove_ticker() mutate it."""
    active_wl = store.lists[store.active]
    new_lists = dict(store.lists)
    new_lists[store.active] = replace(active_wl, tickers=tickers)
    return WatchlistStore(active=store.active, lists=new_lists)


@st.cache_data(ttl=300, show_spinner=False)
def _load_quote(ticker: str) -> QuoteSnapshot:
    """One ticker's quote line, from the shared shallow bundle.

    Cached for 5 minutes rather than the underlying info fetch's own 30
    minutes: this is the shortest-lived, most quote-like view of that
    data, and a cheap cache hit on the already-cached bundle underneath
    costs nothing when it does re-run.
    """
    try:
        bundle = load_ticker_bundle(ticker, deep=False)
    except Exception:
        # load_ticker_bundle handles routine failures internally via
        # bundle.errors, so anything raised here is genuinely unexpected.
        log_event(logger, logging.WARNING, "quote.error", ticker=ticker)
        return QuoteSnapshot(ticker=ticker, status="unavailable", detail="unexpected error loading quote")

    if not bundle.info:
        return QuoteSnapshot(ticker=ticker, status="unavailable", detail="; ".join(bundle.errors) or "no quote data returned")

    info = bundle.info
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    pe_ratio = info.get("trailingPE")

    if price is None or not previous_close:
        return QuoteSnapshot(
            ticker=ticker, price=price, previous_close=previous_close, pe_ratio=pe_ratio,
            status="unavailable", detail="quote missing price or previous close",
        )

    change_pct = (price - previous_close) / previous_close * 100
    return QuoteSnapshot(
        ticker=ticker, price=price, previous_close=previous_close,
        change_pct=change_pct, pe_ratio=pe_ratio, status="ok",
    )


def load_quote_snapshots(tickers: Tuple[str, ...]) -> Tuple[QuoteSnapshot, ...]:
    """Quote lines for every watchlist ticker, in the caller's order."""
    return tuple(_load_quote(t) for t in tickers)
