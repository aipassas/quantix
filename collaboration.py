"""Per-ticker notes with @-mentions — the app's collaboration layer.

Covers two backlog items at once, deliberately: a personal "Stock Notes &
Annotations" entry and a team "Collaboration Features" one are the same
feature with an author name attached, and building them separately would
leave the app with two overlapping notes systems.

IDENTITY IS SELF-DECLARED, NOT AUTHENTICATED. Quantix has no accounts —
six other modules already document this — so there is nothing to hang a
real user off. The honest model here is a shared team whiteboard: you say
who you are, that name is stored with the note, and anyone using this
instance can see and edit the thread. This is disclosed in the UI, not
just here, because a note attributed to a name nobody proved they own is
a materially weaker claim than a signed comment and the reader deserves
to know that.

@-MENTIONS RESOLVE ONLY AGAINST THE ROSTER. A mention is matched against
the teammates explicitly added in the panel, never parsed as a free-form
address. That is a safety boundary, not a convenience: the app emails
people on mention, so bounding it to a list the user deliberately curated
means a typo — or someone typing "@ceo@bigcorp.com" — can never cause
mail to reach a stranger. Unmatched mentions are simply left as plain
text.

NOTIFICATION IS BEST-EFFORT AND NEVER BLOCKS THE NOTE. Sending is
attempted once, at creation. If SMTP isn't configured or the send fails,
the note is still saved and the UI says the notice didn't go out —
losing someone's written thinking because a mail server was down would be
a much worse failure than a missed notification.

Persisted with the same atomic-write, gitignored-local-file pattern every
other piece of cross-restart state here uses (see local_store.py).
"""
import datetime
import json
import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import COLLABORATION
from local_store import atomic_write_text
from logging_setup import get_logger, log_exception

logger = get_logger("collaboration")

# A mention is @ followed by word characters, dots or hyphens. Matched
# case-insensitively against roster names with spaces stripped, so a
# teammate stored as "Ana Silva" is mentioned as @AnaSilva or @anasilva.
_MENTION_RE = re.compile(r"@([A-Za-z0-9._-]+)")


@dataclass(frozen=True)
class TeamMember:
    name: str
    email: str

    @property
    def handle(self) -> str:
        """The @-handle for this member: their name with spaces removed,
        lowercased. Derived rather than stored so renaming a member can
        never leave a stale handle behind."""
        return self.name.replace(" ", "").lower()


@dataclass(frozen=True)
class Note:
    id: str
    ticker: str
    author: str
    body: str
    created_at: str
    mentions: Tuple[str, ...] = ()      # roster NAMES that were matched
    notified: Tuple[str, ...] = ()      # names an email actually reached


@dataclass(frozen=True)
class CollaborationStore:
    members: Tuple[TeamMember, ...] = ()
    notes: Dict[str, Tuple[Note, ...]] = field(default_factory=dict)


def _store_path() -> Path:
    return Path(__file__).resolve().parent / COLLABORATION.store_filename


def load_store(path: Optional[Path] = None) -> CollaborationStore:
    """Never raises: a missing file is an empty store, and a corrupt one
    degrades to empty rather than crashing the app on load. Individual
    malformed notes/members are dropped rather than discarding the whole
    file — losing one bad row beats losing everyone's notes."""
    path = path or _store_path()
    if not path.exists():
        return CollaborationStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "collaboration.store_corrupt", section="collaboration")
        return CollaborationStore()
    if not isinstance(raw, dict):
        return CollaborationStore()

    members = []
    for m in raw.get("members", []):
        if isinstance(m, dict) and str(m.get("name", "")).strip() and "@" in str(m.get("email", "")):
            members.append(TeamMember(name=str(m["name"]).strip(), email=str(m["email"]).strip()))

    notes: Dict[str, Tuple[Note, ...]] = {}
    for ticker, items in (raw.get("notes") or {}).items():
        if not isinstance(ticker, str) or not isinstance(items, list):
            continue
        good = []
        for n in items:
            if not isinstance(n, dict) or not str(n.get("body", "")).strip():
                continue
            good.append(Note(
                id=str(n.get("id") or uuid.uuid4().hex),
                ticker=ticker,
                author=str(n.get("author") or "Unknown"),
                body=str(n["body"]),
                created_at=str(n.get("created_at") or ""),
                mentions=tuple(str(x) for x in (n.get("mentions") or [])),
                notified=tuple(str(x) for x in (n.get("notified") or [])),
            ))
        if good:
            notes[ticker] = tuple(good)
    return CollaborationStore(members=tuple(members), notes=notes)


def save_store(store: CollaborationStore, path: Optional[Path] = None) -> None:
    path = path or _store_path()
    payload = {
        "members": [{"name": m.name, "email": m.email} for m in store.members],
        "notes": {
            t: [{"id": n.id, "author": n.author, "body": n.body, "created_at": n.created_at,
                 "mentions": list(n.mentions), "notified": list(n.notified)} for n in items]
            for t, items in store.notes.items()
        },
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


# --- roster -------------------------------------------------------------------

def add_member(store: CollaborationStore, name: str, email: str) -> Tuple[CollaborationStore, Optional[str]]:
    """Add a teammate. Returns (store, error); error is None on success."""
    name, email = (name or "").strip(), (email or "").strip()
    if not name:
        return store, "Enter a name."
    if "@" not in email:
        return store, "Enter a valid email address."
    if len(store.members) >= COLLABORATION.max_members:
        return store, f"Team is full ({COLLABORATION.max_members} members)."
    candidate = TeamMember(name=name, email=email)
    if any(m.handle == candidate.handle for m in store.members):
        return store, f'"{name}" is already on the team (handles must be unique).'
    return replace(store, members=store.members + (candidate,)), None


def remove_member(store: CollaborationStore, name: str) -> CollaborationStore:
    return replace(store, members=tuple(m for m in store.members if m.name != name))


def member_by_handle(store: CollaborationStore, handle: str) -> Optional[TeamMember]:
    handle = handle.replace(" ", "").lower()
    return next((m for m in store.members if m.handle == handle), None)


# --- mentions -----------------------------------------------------------------

def parse_mentions(body: str, members: Tuple[TeamMember, ...]) -> Tuple[str, ...]:
    """Roster NAMES mentioned in `body`, deduplicated, in first-appearance
    order.

    Only matches people already on the roster — an @handle that doesn't
    correspond to a member is deliberately ignored rather than treated as
    an address, which is what keeps this app from ever mailing someone
    the user didn't add on purpose.
    """
    by_handle = {m.handle: m.name for m in members}
    found: List[str] = []
    for raw in _MENTION_RE.findall(body or ""):
        name = by_handle.get(raw.replace(".", "").lower()) or by_handle.get(raw.lower())
        if name and name not in found:
            found.append(name)
    return tuple(found)


# --- notes --------------------------------------------------------------------

def add_note(store: CollaborationStore, ticker: str, author: str, body: str) -> Tuple[CollaborationStore, Optional[Note], Optional[str]]:
    """Append a note to a ticker's thread. Returns (store, note, error).

    Newest-last, so the thread reads chronologically like a conversation.
    """
    ticker = (ticker or "").strip().upper()
    author = (author or "").strip()
    body = (body or "").strip()
    if not ticker:
        return store, None, "No ticker to attach this note to."
    if not author:
        return store, None, "Enter your name so the note has an author."
    if not body:
        return store, None, "Write something first."
    if len(body) > COLLABORATION.max_note_chars:
        return store, None, f"Notes are capped at {COLLABORATION.max_note_chars} characters."

    note = Note(
        id=uuid.uuid4().hex,
        ticker=ticker,
        author=author,
        body=body,
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        mentions=parse_mentions(body, store.members),
    )
    thread = store.notes.get(ticker, ()) + (note,)
    return replace(store, notes={**store.notes, ticker: thread}), note, None


def delete_note(store: CollaborationStore, ticker: str, note_id: str) -> CollaborationStore:
    thread = tuple(n for n in store.notes.get(ticker, ()) if n.id != note_id)
    notes = {**store.notes}
    if thread:
        notes[ticker] = thread
    else:
        notes.pop(ticker, None)
    return replace(store, notes=notes)


def notes_for(store: CollaborationStore, ticker: str) -> Tuple[Note, ...]:
    return store.notes.get((ticker or "").strip().upper(), ())


def mark_notified(store: CollaborationStore, ticker: str, note_id: str, names: Tuple[str, ...]) -> CollaborationStore:
    """Record which mentioned people an email actually reached, so the UI
    can be honest about partial delivery rather than implying everyone was
    notified."""
    thread = tuple(replace(n, notified=names) if n.id == note_id else n
                   for n in store.notes.get(ticker, ()))
    return replace(store, notes={**store.notes, ticker: thread})


def notify_mentions(store: CollaborationStore, note: Note, sender) -> Tuple[Tuple[str, ...], List[str]]:
    """Email everyone mentioned in `note`. Returns (names_notified, errors).

    `sender` is injected (email_report.send_notification_email in the app,
    a fake in tests) so this is testable without touching a mail server.
    Best-effort by contract: one failed recipient never stops the others,
    and no failure here ever unsaves the note.
    """
    notified: List[str] = []
    errors: List[str] = []
    for name in note.mentions:
        member = next((m for m in store.members if m.name == name), None)
        if member is None:
            continue
        subject = COLLABORATION.mention_subject_template.format(author=note.author, ticker=note.ticker)
        body = COLLABORATION.mention_body_template.format(
            name=member.name, author=note.author, ticker=note.ticker, body=note.body,
        )
        ok, err = sender(member.email, subject, body)
        if ok:
            notified.append(name)
        else:
            errors.append(f"{name}: {err}")
    if notified:
        logger.log(logging.INFO, "collaboration.mentions_notified count=%d", len(notified))
    return tuple(notified), errors
