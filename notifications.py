"""The notification bell: what is unread, and which rules are snoozed.

WHAT FEEDS IT, and what deliberately does not. The Real-Time Alert Engine
already persists a timestamped TriggerEvent every time a rule fires, so
"unread since you last looked" is a real count over real events. The
Smart Risk-Aware Alerts are a different animal — the app is stateless
with no background worker, so that panel is explicitly an on-demand
SNAPSHOT of what is true right now, with no event log and no timestamps.
Folding it in would mean inventing an occurrence time and a dedup key so
a standing breach did not re-notify on every rerun. Settled with the user
before building: the bell counts real events only.

UNREAD IS A WATERMARK, NOT A FLAG PER EVENT. One ISO timestamp per user
— everything newer is unread. That survives the history being trimmed
(it is capped at 100 and rewritten on every save), which a per-event
read-flag would not: the flags would be dropped along with the events
they described and old alerts would silently become unread again.

SNOOZE ACTS ON THE RULE, NOT THE NOTIFICATION. The event has already
happened; hiding it changes nothing, and the rule fires again on the next
poll — which reads as the button not working. Snoozing mutes the rule for
a period, and while muted the engine records no new events for it. Also
settled with the user.

A MUTE THAT HAS EXPIRED IS NOT A MUTE. Nothing here writes "unmute"
events; is_muted() compares against the clock every time, so a snooze
lapses on its own and a store carried across a restart cannot leave a
rule permanently silent.
"""
import datetime
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("notifications")

STORE_FILENAME = "notifications.json"

# How many entries the dropdown shows before pointing at the full list.
DROPDOWN_LIMIT = 6

# Offered snooze durations, in hours.
SNOOZE_CHOICES: Tuple[Tuple[str, int], ...] = (
    ("1 hour", 1),
    ("4 hours", 4),
    ("24 hours", 24),
)

# Beyond this the badge reads "9+" rather than growing the control.
BADGE_CAP = 9


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _parse(stamp: Optional[str]) -> Optional[datetime.datetime]:
    """An ISO string as a datetime, or None. Never raises: a store written
    by a future version, or hand-edited, must not take the page down."""
    if not stamp:
        return None
    try:
        return datetime.datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None


# --- storage ------------------------------------------------------------------

def _path():
    return store_path(STORE_FILENAME)


def _read() -> Tuple[Dict, bool]:
    """(data, corrupt) — the distinction this codebase makes everywhere.
    Treating an unreadable store as empty means the next write destroys
    it; here that would silently un-snooze every muted rule."""
    path = _path()
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log_exception(logger, "notifications.unreadable", section="notifications")
        return {}, True
    if not isinstance(data, dict):
        return {}, True
    return data, False


def _write(data: Dict) -> None:
    atomic_write_text(_path(), json.dumps(data, indent=2))


def store_is_corrupt() -> bool:
    return _read()[1]


# --- unread watermark ---------------------------------------------------------

def last_seen() -> Optional[datetime.datetime]:
    data, corrupt = _read()
    if corrupt:
        return None
    return _parse(data.get("last_seen"))


def mark_all_read(now: Optional[datetime.datetime] = None) -> bool:
    """Move the watermark to `now`. Returns False if the store is
    unreadable, rather than overwriting it."""
    data, corrupt = _read()
    if corrupt:
        return False
    data["last_seen"] = (now or _now()).isoformat(timespec="seconds")
    _write(data)
    log_event(logger, logging.INFO, "notifications.marked_read")
    return True


def unread(history: Sequence, seen: Optional[datetime.datetime]) -> List:
    """Events newer than the watermark, newest first.

    With no watermark every event is unread — a first-time user genuinely
    has not seen any of them, and defaulting to "all read" would hide the
    feature the first time it mattered.
    """
    out = []
    for event in reversed(list(history or [])):
        when = _parse(getattr(event, "triggered_at", None))
        if seen is None or (when is not None and when > seen):
            out.append(event)
    return out


def badge_text(count: int) -> str:
    """"", "3", or "9+". Empty when there is nothing to report, so the
    control can say "Alerts" rather than "Alerts 0"."""
    if count <= 0:
        return ""
    return f"{count}" if count <= BADGE_CAP else f"{BADGE_CAP}+"


# --- snoozing a rule ----------------------------------------------------------

def mutes(now: Optional[datetime.datetime] = None) -> Dict[str, datetime.datetime]:
    """Rule id -> when its mute expires, expired ones already dropped."""
    data, corrupt = _read()
    if corrupt:
        return {}
    now = now or _now()
    live = {}
    for rule_id, until in (data.get("mutes") or {}).items():
        expires = _parse(until)
        if expires is not None and expires > now:
            live[str(rule_id)] = expires
    return live


def is_muted(rule_id: str, now: Optional[datetime.datetime] = None) -> bool:
    return str(rule_id) in mutes(now)


def snooze(rule_id: str, hours: int,
           now: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
    """Mute a rule for `hours`. Returns the expiry, or None if refused."""
    if not rule_id or hours <= 0:
        return None
    data, corrupt = _read()
    if corrupt:
        return None
    now = now or _now()
    until = now + datetime.timedelta(hours=hours)
    data.setdefault("mutes", {})[str(rule_id)] = until.isoformat(timespec="seconds")
    _write(data)
    log_event(logger, logging.INFO, "notifications.snoozed", hours=hours)
    return until


def unsnooze(rule_id: str) -> bool:
    data, corrupt = _read()
    if corrupt:
        return False
    existing = data.get("mutes") or {}
    if str(rule_id) not in existing:
        return False
    existing.pop(str(rule_id))
    data["mutes"] = existing
    _write(data)
    log_event(logger, logging.INFO, "notifications.unsnoozed")
    return True


def describe_mute(until: datetime.datetime,
                  now: Optional[datetime.datetime] = None) -> str:
    """"muted for another 47 min" — a duration, because an absolute time
    makes the reader do the subtraction."""
    now = now or _now()
    minutes = max(0, int((until - now).total_seconds() // 60))
    if minutes >= 60:
        return f"muted for another {minutes // 60}h {minutes % 60:02d}m"
    return f"muted for another {minutes} min"


# --- rendering helpers --------------------------------------------------------

def describe_age(stamp: Optional[str],
                 now: Optional[datetime.datetime] = None) -> str:
    """"4 min ago". Returns "time unknown" rather than guessing when the
    timestamp is missing or unparseable."""
    when = _parse(stamp)
    if when is None:
        return "time unknown"
    seconds = int(((now or _now()) - when).total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def clear_history() -> bool:
    """Caller empties the history itself; this only moves the watermark so
    a later restore does not resurrect the badge count."""
    return mark_all_read()
