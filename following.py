"""Profiles, following, and a feed of what colleagues are doing.

TWO TICKETS, ONE FEATURE. "Follow Expert Analysts" says in its own notes
that it depends on "User Profiles & Follow System"; the first cannot
exist without the second, so both are built here and closed together.

THERE ARE NO EXPERTS, AND THE WORD IS DROPPED ON PURPOSE. The only
population on this instance is the other ACCOUNTS on it — colleagues, on
a licensee's team deployment; nobody at all on a laptop. There is no
grantor for an "expert" badge and no role system to hang one on (RBAC is
its own ticket). A profile carries an optional TITLE the account writes
itself, shown as self-declared, which is what a colleague's title is.

THE PRIVACY MODEL IS ONE RULE: A FOLLOWER'S SESSION NEVER OPENS ANOTHER
ACCOUNT'S FILES. Everything a feed could show — watchlist changes, alert
triggers, journal entries, recently viewed tickers — lives in per-user
stores that are private by design (auth.PER_USER_STORES). The feed does
not read them. Instead, the SHARING account publishes an event into a
shared feed store from its own session, and only when it has switched
that stream on. A reader sees what was published, filtered to the
accounts it follows. The reading side has no code path to a private
store, so a bug in the reader cannot become a leak.

NOTHING IS SHARED BY DEFAULT, AND SWITCHING A STREAM OFF IS RETROACTIVE.
Every stream starts disabled. Turning one off removes that stream's past
events from the feed, not just future ones — opting out has to mean
gone, or the choice is cosmetic (the same rule peer_comparison follows).
Deleting a profile removes every event the account ever published. And
the reader re-checks the publisher's CURRENT switches at read time, so a
stream turned off is invisible immediately, before the cleanup runs.

FOLLOWS ARE PRIVATE; PROFILES ARE SHARED. Whom you follow is your own
business and lives in your namespace. A profile is by definition the
thing other accounts can see, so it lives in the shared store — and an
account WITHOUT a profile is not followable at all. That is the
"public/followable profile" the ticket asked for: you become visible by
creating one, not by existing.

THE FEED IS BOUNDED AND DEDUPLICATED. "Recently viewed" fires on every
ticker visit and would drown the feed; a repeat view of the same ticker
by the same account within a day is collapsed. The store keeps a fixed
number of events, oldest dropped first.
"""
import datetime
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import FOLLOWING
from local_store import atomic_write_text, shared_path, store_path
from logging_setup import get_logger, log_exception

logger = get_logger("following")


# The streams an account can choose to share. A closed set: a stream
# added later starts disabled for everyone, because the default for any
# stream is off.
STREAM_WATCHLIST = "watchlist"
STREAM_ALERT = "alert"
STREAM_JOURNAL = "journal"
STREAM_VIEWED = "viewed"

STREAMS: Dict[str, str] = {
    STREAM_WATCHLIST: "Watchlist changes — tickers you add to a watchlist.",
    STREAM_ALERT: "Alert triggers — when one of your alert rules fires. Shares the "
                  "rule's ticker and condition, never your portfolio.",
    STREAM_JOURNAL: "Journal entries — your Buy/Sell/Hold decisions with the "
                    "reasoning. The most useful stream and the most personal: "
                    "it is your actual calls.",
    STREAM_VIEWED: "Recently viewed — tickers you open. A passive research "
                   "trail; easy to over-share.",
}

STREAM_LABELS: Dict[str, str] = {
    STREAM_WATCHLIST: "added to a watchlist",
    STREAM_ALERT: "alert fired",
    STREAM_JOURNAL: "journal entry",
    STREAM_VIEWED: "looked at",
}


@dataclass(frozen=True)
class Profile:
    """An account that has chosen to be followable."""
    user_key: str
    name: str
    title: str = ""                      # self-described; shown as such
    streams: Tuple[str, ...] = ()        # enabled streams; empty by default
    created_at: str = ""

    def shares(self, stream: str) -> bool:
        return stream in self.streams


@dataclass(frozen=True)
class FeedEvent:
    """One thing a colleague did, as they chose to publish it.

    Deliberately flat and small. `summary` is one line the publisher's
    own session wrote — the reader never derives anything from the
    publisher's stores.
    """
    id: str
    actor_key: str
    stream: str
    ticker: str
    summary: str
    at: str


@dataclass(frozen=True)
class ProfilesStore:
    profiles: Tuple[Profile, ...] = ()
    corrupt: bool = False

    def get(self, user_key: str) -> Optional[Profile]:
        return next((p for p in self.profiles if p.user_key == user_key), None)


@dataclass(frozen=True)
class FeedStore:
    events: Tuple[FeedEvent, ...] = ()
    corrupt: bool = False


@dataclass(frozen=True)
class FollowStore:
    """Whom THIS account follows. Per-user, because it is nobody else's
    business."""
    following: Tuple[str, ...] = ()
    corrupt: bool = False


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _profiles_path() -> Path:
    return shared_path(FOLLOWING.profiles_filename)


def _feed_path() -> Path:
    return shared_path(FOLLOWING.feed_filename)


def _follows_path() -> Path:
    return store_path(FOLLOWING.follows_filename)


# --- persistence --------------------------------------------------------------

def _read_json(path: Path):
    """(data, corrupt). Missing is an empty dict; unreadable is flagged
    so a save cannot overwrite what could not be parsed."""
    if not path.exists():
        return {}, False
    try:
        raw = json.loads(path.read_text())
        return (raw if isinstance(raw, dict) else {}), not isinstance(raw, dict)
    except Exception:
        log_exception(logger, "following.store_corrupt", section="following")
        return {}, True


def load_profiles(path: Optional[Path] = None) -> ProfilesStore:
    raw, corrupt = _read_json(path or _profiles_path())
    if corrupt:
        return ProfilesStore(corrupt=True)
    profiles: List[Profile] = []
    for item in raw.get("profiles", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("user_key") or "").strip()
        if not key:
            continue
        streams = tuple(s for s in (item.get("streams") or ()) if s in STREAMS)
        profiles.append(Profile(
            user_key=key,
            name=str(item.get("name") or "").strip() or "Colleague",
            title=str(item.get("title") or "").strip()[:FOLLOWING.max_title_chars],
            streams=streams,
            created_at=str(item.get("created_at") or ""),
        ))
    return ProfilesStore(tuple(profiles))


def save_profiles(store: ProfilesStore, path: Optional[Path] = None) -> bool:
    if store.corrupt:
        return False
    payload = {"profiles": [{
        "user_key": p.user_key, "name": p.name, "title": p.title,
        "streams": list(p.streams), "created_at": p.created_at,
    } for p in store.profiles]}
    atomic_write_text(path or _profiles_path(), json.dumps(payload, indent=2))
    return True


def load_feed(path: Optional[Path] = None) -> FeedStore:
    raw, corrupt = _read_json(path or _feed_path())
    if corrupt:
        return FeedStore(corrupt=True)
    events: List[FeedEvent] = []
    for item in raw.get("events", []) or []:
        if not isinstance(item, dict):
            continue
        actor = str(item.get("actor_key") or "").strip()
        stream = str(item.get("stream") or "")
        if not actor or stream not in STREAMS:
            continue
        events.append(FeedEvent(
            id=str(item.get("id") or ""),
            actor_key=actor, stream=stream,
            ticker=str(item.get("ticker") or "").upper(),
            summary=str(item.get("summary") or "")[:FOLLOWING.max_summary_chars],
            at=str(item.get("at") or ""),
        ))
    return FeedStore(tuple(events))


def save_feed(store: FeedStore, path: Optional[Path] = None) -> bool:
    if store.corrupt:
        return False
    payload = {"events": [{
        "id": e.id, "actor_key": e.actor_key, "stream": e.stream,
        "ticker": e.ticker, "summary": e.summary, "at": e.at,
    } for e in store.events]}
    atomic_write_text(path or _feed_path(), json.dumps(payload, indent=2))
    return True


def load_follows(path: Optional[Path] = None) -> FollowStore:
    raw, corrupt = _read_json(path or _follows_path())
    if corrupt:
        return FollowStore(corrupt=True)
    keys = tuple(str(k) for k in (raw.get("following") or ()) if str(k).strip())
    return FollowStore(tuple(dict.fromkeys(keys)))


def save_follows(store: FollowStore, path: Optional[Path] = None) -> bool:
    if store.corrupt:
        return False
    atomic_write_text(path or _follows_path(),
                      json.dumps({"following": list(store.following)}, indent=2))
    return True


# --- profiles -----------------------------------------------------------------

def upsert_profile(store: ProfilesStore, user_key: str, name: str,
                   title: str = "", streams: Sequence[str] = ()
                   ) -> Tuple[ProfilesStore, str]:
    """Create or update this account's profile. Returns (store, error).

    Creating a profile is the act of becoming followable. The stream set
    passed here is the whole set: anything not listed is OFF.
    """
    user_key = (user_key or "").strip()
    if not user_key:
        return store, "Sign in to create a profile."
    name = (name or "").strip()
    if not name:
        return store, "A profile needs a name."
    title = (title or "").strip()
    if len(title) > FOLLOWING.max_title_chars:
        return store, f"Keep the title under {FOLLOWING.max_title_chars} characters."
    chosen = tuple(dict.fromkeys(s for s in streams if s in STREAMS))

    existing = store.get(user_key)
    profile = Profile(
        user_key=user_key, name=name, title=title, streams=chosen,
        created_at=existing.created_at if existing else _now_iso(),
    )
    others = tuple(p for p in store.profiles if p.user_key != user_key)
    return replace(store, profiles=others + (profile,)), ""


def delete_profile(store: ProfilesStore, user_key: str) -> ProfilesStore:
    return replace(store, profiles=tuple(p for p in store.profiles
                                         if p.user_key != user_key))


def followable(store: ProfilesStore, viewer_key: str = "") -> Tuple[Profile, ...]:
    """Everyone with a profile except the viewer, by name."""
    return tuple(sorted((p for p in store.profiles if p.user_key != viewer_key),
                        key=lambda p: p.name.lower()))


# --- following ----------------------------------------------------------------

def follow(store: FollowStore, target_key: str) -> FollowStore:
    target_key = (target_key or "").strip()
    if not target_key or target_key in store.following:
        return store
    return replace(store, following=store.following + (target_key,))


def unfollow(store: FollowStore, target_key: str) -> FollowStore:
    return replace(store, following=tuple(k for k in store.following
                                          if k != target_key))


# --- publishing ---------------------------------------------------------------

def publish(feed: FeedStore, profiles: ProfilesStore, actor_key: str,
            stream: str, ticker: str, summary: str,
            now: Optional[datetime.datetime] = None) -> FeedStore:
    """Record one event, if and only if the actor shares that stream.

    THIS IS THE BOUNDARY. It takes a stream name, a ticker and a one-line
    summary — never a store, a holding, a rule object or an entry. The
    actor's own session composes the summary from data it already has;
    the feed learns nothing else. And it checks the actor's profile
    switch here rather than trusting the caller, so a hook that forgets
    to check shares nothing.

    Returns the feed unchanged, silently, when the stream is off. A
    hook fires on every watchlist add whether or not anyone is sharing,
    and "not shared" is the normal case, not an error.
    """
    actor_key = (actor_key or "").strip()
    profile = profiles.get(actor_key) if actor_key else None
    if profile is None or not profile.shares(stream):
        return feed
    ticker = (ticker or "").strip().upper()
    summary = (summary or "").strip()[:FOLLOWING.max_summary_chars]
    if not summary:
        return feed
    now = now or datetime.datetime.now()

    # Collapse repeat views. Opening the same ticker four times in an
    # afternoon is one fact, not four events.
    if stream == STREAM_VIEWED:
        cutoff = (now - datetime.timedelta(hours=FOLLOWING.viewed_dedupe_hours)
                  ).isoformat(timespec="seconds")
        for event in feed.events:
            if (event.actor_key == actor_key and event.stream == STREAM_VIEWED
                    and event.ticker == ticker and event.at >= cutoff):
                return feed

    import uuid
    event = FeedEvent(uuid.uuid4().hex[:12], actor_key, stream, ticker, summary,
                      now.isoformat(timespec="seconds"))
    events = (feed.events + (event,))[-FOLLOWING.max_events:]
    return replace(feed, events=events)


def purge(feed: FeedStore, actor_key: str,
          streams: Optional[Sequence[str]] = None) -> FeedStore:
    """Remove an actor's events — all of them, or only the named streams.

    Called when a stream is switched off (that stream's history goes) and
    when a profile is deleted (everything goes). Opting out is
    retroactive or it is cosmetic.
    """
    if streams is None:
        keep = tuple(e for e in feed.events if e.actor_key != actor_key)
    else:
        gone = set(streams)
        keep = tuple(e for e in feed.events
                     if not (e.actor_key == actor_key and e.stream in gone))
    return replace(feed, events=keep)


def apply_stream_change(feed: FeedStore, before: Optional[Profile],
                        after: Profile) -> FeedStore:
    """When a profile is saved, drop history for any stream that was
    just turned off."""
    was = set(before.streams) if before else set()
    now = set(after.streams)
    switched_off = tuple(was - now)
    return purge(feed, after.user_key, switched_off) if switched_off else feed


# --- reading ------------------------------------------------------------------

@dataclass(frozen=True)
class FeedItem:
    event: FeedEvent
    actor: Profile

    @property
    def line(self) -> str:
        label = STREAM_LABELS.get(self.event.stream, self.event.stream)
        who = self.actor.name + (f" ({self.actor.title})" if self.actor.title else "")
        when = self.event.at.replace("T", " ")
        return f"**{who}** · {label} **{self.event.ticker}** · {when}"


def feed_for(follows: FollowStore, feed: FeedStore, profiles: ProfilesStore,
             limit: int = 0) -> Tuple[FeedItem, ...]:
    """What the accounts you follow have published, newest first.

    Re-checks each publisher's CURRENT switches. An event whose stream
    has since been turned off is not shown even if the cleanup has not
    run yet — the publisher's choice takes effect on the next read, not
    the next save.
    """
    limit = limit or FOLLOWING.feed_limit
    wanted = set(follows.following)
    items: List[FeedItem] = []
    for event in reversed(feed.events):
        if event.actor_key not in wanted:
            continue
        actor = profiles.get(event.actor_key)
        if actor is None or not actor.shares(event.stream):
            continue
        items.append(FeedItem(event, actor))
        if len(items) >= limit:
            break
    return tuple(items)


def sharing_summary(profile: Optional[Profile]) -> str:
    if profile is None:
        return "You have no profile, so nobody can follow you and nothing you do is shared."
    if not profile.streams:
        return ("Your profile is visible, but every stream is off — followers see "
                "your name and nothing else.")
    names = ", ".join(STREAM_LABELS[s] for s in profile.streams)
    return f"Sharing: {names}. Everything else stays private."
