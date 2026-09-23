"""The opt-in monthly ranking of accounts on this instance.

IT RANKS ON RETURN, AND REFUSING SHARPE IS A MEASUREMENT. The ticket
offers "portfolio return, Sharpe ratio, or another performance metric".
Simulated before anything was designed — 20,000 runs per window, over
portfolios whose TRUE Sharpe is exactly 1.00:

    window      measured mean    sd      90% range
    1 month        1.06         3.68    [-4.91, +7.17]
    1 quarter      1.01         2.04    [-2.34, +4.37]
    1 year         1.00         1.00    [-0.65, +2.65]

Over a calendar month the sampling error is nearly four times the
quantity being estimated, and a monthly Sharpe asked to order two
portfolios whose true Sharpes differ by a FULL 1.0 gets it right 57.7% of
the time. Ranking people on that would be presenting noise as skill on a
screen about their money, which is the exact thing recommendations.py
was written to refuse. A realised monthly return has no such error — it
is not an estimate of a parameter, it is what happened. So the ladder is
return only and SHARPE_UNAVAILABLE says so on screen, the way
etf_technicals.NAV_PREMIUM_UNAVAILABLE does.

THIS IS A SECOND CONSENT, NOT A REUSE OF peer_comparison's. That module
buys an ANONYMOUS percentile, and its entire k-anonymity floor exists so
that no individual return is ever revealed. A leaderboard reveals every
participant's return beside a name — the opposite disclosure. Treating
one opt-in as covering the other would silently upgrade what people
agreed to, so joining has its own switch and its own store, and nobody is
enrolled by being in the other.

THE RETURN ITSELF IS NOT STORED HERE. `leaderboard_store.json` holds
consent records and nothing else: a user key and when they joined. The
figures come from peer_comparison's entries, which means withdrawing
there removes someone from this ladder too, automatically, and there is
no second copy of anyone's return to fall out of step. `join()` takes no
return and no name, for the same reason peer_comparison.publish() takes
no portfolio: the signature is the privacy boundary.

A PARTICIPANT MUST HAVE CHOSEN A DISPLAY NAME. Names come from
following.Profile, so nobody is ever listed who did not pick the name
they are listed under — a raw account key is a hash, and an email is a
disclosure nobody agreed to. The name is re-read on every render rather
than copied in at join time, so deleting a profile drops you off
immediately, exactly as following.feed_for re-checks its publishers'
current switches.

THE FLOOR IS peer_comparison's OWN NUMBER. Below it the ladder is not
drawn and the count is reported instead. Note the sense differs and the
config records it: min_cohort counts the OTHER participants, this counts
everyone on the ladder including the reader.

TIES SHARE A RANK. Two identical returns are joint second and the next
row is fourth — standard competition ranking. Ordering them by anything
else (alphabetically, by who joined first) would invent a difference the
figures do not contain.
"""
import datetime
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import LEADERBOARD
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_exception

logger = get_logger("leaderboard")

# Why there is no Sharpe ladder, in the UI's own words. The figures are
# the simulation in the module docstring; a test pins them so the prose
# cannot drift away from the measurement it reports.
SHARPE_UNAVAILABLE = (
    "There is no Sharpe ratio ladder. Over a single calendar month a Sharpe ratio "
    "is almost entirely noise: simulated across portfolios whose true Sharpe is "
    "1.00, the one-month estimate has a standard deviation of 3.68 and lands "
    "between -4.91 and +7.17 nine times in ten. Asked to rank two portfolios whose "
    "true Sharpes differ by a full 1.0, it gets the order right 57.7% of the time. "
    "A realised return carries no such error, so that is what this ranks on."
)

NEEDS_PROFILE = (
    "Set a display name in your profile first — the leaderboard lists people by "
    "the name they chose, never by an account key."
)

NEEDS_RETURN = (
    "You are not sharing a return for this month yet. The leaderboard ranks the "
    "figure you share under \"Compare with others\" above, so that has to be on "
    "first."
)

NOT_PARTICIPATING = (
    "You are not on the leaderboard. Joining shows your display name and your "
    "monthly return to every account on this instance — a wider disclosure than "
    "the anonymous comparison above, which is why it is a separate choice."
)


@dataclass(frozen=True)
class Participant:
    """A consent record. Deliberately two fields.

    No return, no name, no period. The return lives in peer_comparison's
    store and the name in the profile, so there is exactly one copy of
    each and nothing here to fall out of step with them.
    """
    user_key: str
    joined_at: str


@dataclass(frozen=True)
class LeaderboardStore:
    participants: Tuple[Participant, ...] = ()
    corrupt: bool = False

    def has(self, user_key: str) -> bool:
        user_key = (user_key or "").strip()
        return bool(user_key) and any(p.user_key == user_key
                                      for p in self.participants)


@dataclass(frozen=True)
class Row:
    rank: int
    name: str
    twr_pct: float
    is_you: bool = False


@dataclass(frozen=True)
class Standings:
    """The ladder, or the reason there isn't one."""
    period: str
    rows: Tuple[Row, ...] = ()
    participants: int = 0
    your_rank: Optional[int] = None
    reason: str = ""

    @property
    def has_result(self) -> bool:
        return bool(self.rows)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _store_path() -> Path:
    # Shared by definition: a ranking across accounts cannot live inside
    # one account's namespace.
    return shared_path(LEADERBOARD.store_filename)


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None) -> LeaderboardStore:
    """Never raises. A file that exists but cannot be read is flagged
    corrupt rather than reported empty — treating it as empty would let
    the next join drop everyone else's consent record, which is the one
    thing in this file that must not be lost silently."""
    path = path or _store_path()
    if not path.exists():
        return LeaderboardStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "leaderboard.store_corrupt", section="leaderboard")
        return LeaderboardStore(corrupt=True)
    if not isinstance(raw, dict):
        return LeaderboardStore(corrupt=True)

    seen: set = set()
    kept: List[Participant] = []
    for item in raw.get("participants", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("user_key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(Participant(key, str(item.get("joined_at") or "")))
    return LeaderboardStore(tuple(kept))


def save_store(store: LeaderboardStore, path: Optional[Path] = None) -> bool:
    """Persist. Refuses to write over a store that could not be read —
    it holds other people's consent, not just yours."""
    if store.corrupt:
        return False
    path = path or _store_path()
    # Field by field rather than asdict(), so a field added to
    # Participant later cannot silently start being shared.
    payload = {"participants": [{"user_key": p.user_key, "joined_at": p.joined_at}
                                for p in store.participants]}
    atomic_write_text(path, json.dumps(payload, indent=2))
    return True


# --- opting in and out --------------------------------------------------------

def join(store: LeaderboardStore, user_key: str) -> Tuple[LeaderboardStore, str]:
    """Record this account's consent to be listed BY NAME. (store, error).

    Takes a key and nothing else. There is deliberately no parameter here
    through which a return, a name or a portfolio could arrive — the same
    reasoning as peer_comparison.publish(), where the signature is what
    makes the privacy claim checkable rather than merely stated.
    """
    user_key = (user_key or "").strip()
    if not user_key:
        return store, "Sign in to join the leaderboard."
    if store.has(user_key):
        return store, ""
    return replace(store, participants=store.participants
                   + (Participant(user_key, _now_iso()),)), ""


def leave(store: LeaderboardStore, user_key: str) -> LeaderboardStore:
    """Remove this account's consent.

    Leaving removes the NAME from the ladder and nothing else: the
    monthly return stays wherever the reader put it, under
    peer_comparison's own opt-in, because that is a separate choice made
    separately. Withdrawing there removes the figure — and therefore the
    row — as well.
    """
    user_key = (user_key or "").strip()
    return replace(store, participants=tuple(p for p in store.participants
                                             if p.user_key != user_key))


def is_participating(store: LeaderboardStore, user_key: str) -> bool:
    return store.has(user_key)


# --- the ladder ---------------------------------------------------------------

def _ranked(pairs: Sequence[Tuple[str, str, float]]) -> List[Tuple[int, str, str, float]]:
    """(rank, key, name, return) with ties SHARING a rank.

    Standard competition ranking: two joint seconds are followed by a
    fourth. Breaking the tie on anything else would assert a difference
    the returns do not contain.
    """
    ordered = sorted(pairs, key=lambda p: (-p[2], p[1].lower(), p[0]))
    out: List[Tuple[int, str, str, float]] = []
    for position, (key, name, value) in enumerate(ordered, start=1):
        if out and out[-1][3] == value:
            rank = out[-1][0]
        else:
            rank = position
        out.append((rank, key, name, value))
    return out


def listed(entries: Sequence, store: LeaderboardStore, profiles,
           period: str) -> List[Tuple[str, str, float]]:
    """(key, name, return) for everyone who actually belongs on the ladder.

    THE THREE CONDITIONS, IN ONE PLACE. A participant appears only when
    they joined, they are sharing a return for this period, and they
    currently have a profile with a name. The last is re-checked on every
    read rather than trusted from join time, so deleting a profile drops
    the row immediately — the same discipline following.feed_for uses on
    its publishers' current switches.
    """
    by_key: Dict[str, float] = {}
    for entry in entries:
        if getattr(entry, "period", "") == period:
            try:
                by_key[entry.user_key] = float(entry.twr_pct)
            except (TypeError, ValueError):
                continue

    out: List[Tuple[str, str, float]] = []
    for participant in store.participants:
        if participant.user_key not in by_key:
            continue
        profile = profiles.get(participant.user_key) if profiles else None
        name = (getattr(profile, "name", "") or "").strip()
        if not name:
            continue
        out.append((participant.user_key, name, by_key[participant.user_key]))
    return out


def standings(entries: Sequence, store: LeaderboardStore, profiles,
              period: str, viewer_key: str = "",
              max_rows: Optional[int] = None) -> Standings:
    """The ladder for `period`, or the reason it cannot be drawn.

    `entries` are peer_comparison.PeerEntry records for the period —
    passed in rather than loaded, so this is testable without touching
    either store and so the caller cannot be surprised by a second read
    of a file it already has.

    A participant appears only when ALL THREE hold: they joined, they are
    sharing a return for this period, and they currently have a profile
    with a name. The last is re-checked here rather than trusted from
    join time, so deleting a profile drops the row on the next render.
    """
    max_rows = max_rows if max_rows is not None else LEADERBOARD.max_rows
    viewer_key = (viewer_key or "").strip()

    pairs = listed(entries, store, profiles, period)
    total = len(pairs)
    if total < LEADERBOARD.min_participants:
        return Standings(period, participants=total, reason=(
            f"{total} account(s) on this instance are on the leaderboard for "
            f"{period}. The ladder is drawn once {LEADERBOARD.min_participants} are: "
            "below that it ranks a handful of people while stating each of their "
            "returns exactly, which is a disclosure rather than a competition."))

    ranked = _ranked(pairs)
    your_rank = next((r for r, key, _, _ in ranked if key == viewer_key), None)

    rows = [Row(rank, name, value, key == viewer_key)
            for rank, key, name, value in ranked[:max_rows]]
    # The reader's own row is appended when they placed outside the
    # visible top — a ladder that never shows you where you stand is a
    # scoreboard for other people.
    if your_rank is not None and not any(r.is_you for r in rows):
        rank, key, name, value = next(r for r in ranked if r[1] == viewer_key)
        rows.append(Row(rank, name, value, True))

    return Standings(period, tuple(rows), total, your_rank)


def participation_note(store: LeaderboardStore, entries: Sequence,
                       profiles, period: str) -> str:
    """One line on how many are on the ladder, for the panel's caption.

    Counts through listed() rather than re-deriving the three conditions
    inline. A caption that disagrees with the table under it is worse
    than no caption, and duplicated filtering is exactly how the two
    would drift apart.
    """
    count = len(listed(entries, store, profiles, period))
    if count == 0:
        return (f"Nobody on this instance is on the leaderboard for {period} yet. "
                "It lists the accounts that opted in, by the display name they chose.")
    if count == 1:
        return f"1 account is on the leaderboard for {period}."
    return f"{count} accounts are on the leaderboard for {period}."
