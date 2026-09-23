"""Daily activity streaks — consecutive days on which this account did
something deliberate.

A LOGIN STREAK WOULD MEASURE NOTHING. Merely rendering this app writes
per-user files: a visited ticker becomes a recent, the tour writes its
flag, the risk panel seeds default rules. So "opened the page" is
satisfied by a refresh, a reconnect, or anything that hits the URL, and a
streak that cannot honestly be lost is not a streak — it is a number that
only goes up. CLAUDE.md records the same trap for the first-sign-in
adoption prompt, which was gated on "is this namespace empty" and was
already suppressed by the time the sidebar drew. A day therefore counts
only when the account took one of the ACTIONS below.

THE GRACE IS ZERO. Most habit products forgive a missed day to keep the
number alive. That redefines "consecutive" to flatter the reader, which
is the opposite of how every other figure in this app is handled. A
missed day ends the run — and `longest` is kept beside `current` so the
run that ended is still visible as something that happened.

TODAY IS NOT ASSUMED. A run whose last active day was yesterday is still
alive: the day is not over. A run whose last active day was earlier than
that is over, and `current_streak` returns 0 rather than the stale count.
Getting this wrong in the other direction — counting today before
anything happened — would show a streak of 1 to someone who has done
nothing, which is the same fabrication as a figure with no evidence.

DAYS ARE STORED, NOT A COUNTER. A counter cannot be recomputed, cannot be
audited, and silently keeps whatever a bug put in it. The store holds the
set of ISO dates on which something happened and every figure is derived
from it, so a wrong number can always be traced to a wrong day.

WHAT IS NOT HERE. No badges and no rewards — that is the Achievement
Badges ticket. Nothing is shared: this store is per-user, because how
often somebody opens a research tool is nobody else's business unless
they choose to say so.
"""
import datetime
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from config import STREAKS
from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_exception

logger = get_logger("streaks")

# What counts as a deliberate act. Each is something the reader chose to
# do, not something the page did on their behalf while loading.
ACTIONS: Tuple[Tuple[str, str], ...] = (
    ("journal", "recorded a decision in the journal"),
    ("screen", "ran a screen"),
    ("watchlist", "added a ticker to a watchlist"),
    ("contest", "entered the monthly contest"),
    ("note", "posted a team note"),
    ("alert", "created an alert rule"),
)
ACTION_KEYS: Tuple[str, ...] = tuple(key for key, _ in ACTIONS)
ACTION_LABELS = dict(ACTIONS)

NO_ACTIVITY = (
    "No activity recorded yet. A day counts once you do something deliberate — "
    "record a journal decision, run a screen, add to a watchlist, enter the "
    "contest, post a note or create an alert. Simply opening the page is not "
    "counted, because this app writes files on every render and a streak nobody "
    "could lose would not mean anything."
)


@dataclass(frozen=True)
class StreakStore:
    """The ISO dates on which something happened, newest last."""
    days: Tuple[str, ...] = ()
    corrupt: bool = False

    def has(self, day: str) -> bool:
        return day in self.days


@dataclass(frozen=True)
class Streak:
    current: int = 0
    longest: int = 0
    total_days: int = 0
    last_active: str = ""
    active_today: bool = False

    @property
    def alive(self) -> bool:
        return self.current > 0


def _today() -> datetime.date:
    return datetime.date.today()


def _iso(day: datetime.date) -> str:
    return day.isoformat()


def _parse(value: str) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(value)
    except Exception:
        return None


def _path() -> Path:
    # Per-user: how often somebody opens a research tool is nobody else's
    # business unless they choose to say so.
    return store_path(STREAKS.store_filename)


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None) -> StreakStore:
    """Never raises. A file that exists but cannot be read is flagged
    corrupt rather than reported empty, so the next write cannot silently
    replace someone's whole history with today."""
    path = path or _path()
    if not path.exists():
        return StreakStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "streaks.store_corrupt", section="streaks")
        return StreakStore(corrupt=True)
    if not isinstance(raw, dict):
        return StreakStore(corrupt=True)

    days = []
    for value in raw.get("days", []) or []:
        parsed = _parse(str(value))
        if parsed is not None:
            days.append(_iso(parsed))
    # Sorted and de-duplicated here rather than trusted from the file: an
    # unordered or repeated day would silently corrupt every run length.
    return StreakStore(tuple(sorted(set(days))))


def save_store(store: StreakStore, path: Optional[Path] = None) -> bool:
    if store.corrupt:
        return False
    path = path or _path()
    atomic_write_text(path, json.dumps({"days": list(store.days)}, indent=2))
    return True


# --- recording ----------------------------------------------------------------

def record(store: StreakStore, action: str,
           day: Optional[datetime.date] = None) -> Tuple[StreakStore, bool]:
    """Mark today active. Returns (store, changed).

    `changed` is False when the day was already recorded, which is the
    common case — this is called from the app on every qualifying click,
    and a caller that saved unconditionally would rewrite the file dozens
    of times a session for no change at all.

    An unknown action is refused rather than counted. The list is closed
    on purpose: a caller passing "render" would quietly reintroduce the
    login streak this module exists to avoid.
    """
    if action not in ACTION_KEYS:
        return store, False
    if store.corrupt:
        return store, False
    today = _iso(day or _today())
    if store.has(today):
        return store, False
    days = tuple(sorted(set(store.days) | {today}))
    return replace(store, days=days[-STREAKS.max_days_retained:]), True


# --- reading ------------------------------------------------------------------

def _runs(days: Sequence[datetime.date]) -> List[int]:
    """Lengths of every consecutive run in an ascending, unique list."""
    runs: List[int] = []
    length = 0
    previous: Optional[datetime.date] = None
    for day in days:
        if previous is not None and (day - previous).days == 1:
            length += 1
        else:
            if length:
                runs.append(length)
            length = 1
        previous = day
    if length:
        runs.append(length)
    return runs


def summarise(store: StreakStore, today: Optional[datetime.date] = None) -> Streak:
    """Current run, longest run, and what the store actually holds.

    The current run counts back from the most recent active day, and is
    reported only when that day is today or yesterday. Yesterday still
    counts because the day is not over; anything earlier means the run
    ended, and returning its old length would be reporting a streak the
    reader no longer has.
    """
    today = today or _today()
    parsed = sorted({d for d in (_parse(v) for v in store.days) if d is not None})
    if not parsed:
        return Streak()

    runs = _runs(parsed)
    longest = max(runs) if runs else 0
    last = parsed[-1]
    gap = (today - last).days
    current = runs[-1] if 0 <= gap <= 1 else 0
    return Streak(current=current, longest=longest, total_days=len(parsed),
                  last_active=_iso(last), active_today=(gap == 0))


def recent_days(store: StreakStore, today: Optional[datetime.date] = None,
                span: Optional[int] = None) -> Tuple[Tuple[str, bool], ...]:
    """(iso date, was active) for the last `span` days, oldest first —
    the strip the panel draws. Built from the stored days so it cannot
    disagree with the counts beside it."""
    today = today or _today()
    span = span or STREAKS.calendar_days
    active = set(store.days)
    return tuple(
        (_iso(today - datetime.timedelta(days=offset)),
         _iso(today - datetime.timedelta(days=offset)) in active)
        for offset in range(span - 1, -1, -1)
    )


def sentence(streak: Streak, today: Optional[datetime.date] = None) -> str:
    """One line describing the run, or why there isn't one."""
    if streak.total_days == 0:
        return NO_ACTIVITY
    if streak.current == 0:
        return (f"No current streak — the last active day was {streak.last_active}. "
                f"Longest run so far: {streak.longest} day"
                f"{'s' if streak.longest != 1 else ''}.")
    day_word = "day" if streak.current == 1 else "days"
    tail = (" Nothing recorded today yet, so this run continues only if you do "
            "something before midnight." if not streak.active_today else "")
    best = (f" Your longest is {streak.longest}."
            if streak.longest > streak.current else " That is your longest run.")
    return f"{streak.current} {day_word} in a row.{best}{tail}"
