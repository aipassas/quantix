"""Per-ticker notes with @-mentions — the app's collaboration layer.

Covers two backlog items at once, deliberately: a personal "Stock Notes &
Annotations" entry and a team "Collaboration Features" one are the same
feature with an author name attached, and building them separately would
leave the app with two overlapping notes systems.

IDENTITY MAY BE VERIFIED OR SELF-DECLARED, AND EACH NOTE RECORDS WHICH.
When auth.py has a provider configured and the writer is signed in, the
author name comes from their OIDC identity and the note is stored with
authenticated=True plus the issuer that vouched for it. Otherwise the
name is typed into a box and proves nothing. Both remain possible, so the
distinction is stored per-note rather than inferred later: a note written
while signed out does not retroactively become verified when its author
signs in afterwards. The difference is surfaced in the thread, in the
notification email, and here, because "Ana says sell" carries very
different weight depending on whether anyone proved she wrote it.

THESE NOTES STAY SHARED EVEN WHEN EVERYTHING ELSE GOES PER-USER. auth.py
scopes watchlists, favourites, themes, thresholds, alert rules and
scenarios to the signed-in user; this store is deliberately excluded. A
thread on AAPL exists so teammates can read each other — namespacing it
would quietly convert the collaboration feature into private diaries.

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
from local_store import atomic_write_text, shared_path
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
    # Whether `author` came from a signed-in OIDC identity or was typed
    # into a box. Stored per-note rather than derived at read time,
    # because it's a fact about the moment the note was written: a note
    # written signed-out doesn't retroactively become verified when its
    # author later signs in.
    authenticated: bool = False
    issuer: str = ""                    # who verified it, when authenticated
    # OWNERSHIP NEEDS AN IDENTITY, AND A DISPLAY NAME IS NOT ONE. Two
    # accounts can both be "Ana"; the account KEY (the same hash that
    # names the user's directory) is what says whether the person now
    # signed in is the person who wrote this. Empty on notes written
    # signed-out, which is exactly why those can be owned by nobody.
    author_key: str = ""
    # A reply points at the note it answers. One level deep on purpose:
    # a reply-to-a-reply makes a thread a tree, and a tree on a ticker
    # page is a forum, which this is not trying to be.
    parent_id: str = ""
    # SOFT DELETE, RECORDED. A hidden note leaves everyone's view but
    # stays in the store with who hid it and when, so a removal is an
    # auditable act rather than a vanishing. The author can restore it.
    hidden: bool = False
    hidden_by: str = ""
    hidden_at: str = ""

    @property
    def is_reply(self) -> bool:
        return bool(self.parent_id)


@dataclass(frozen=True)
class CollaborationStore:
    members: Tuple[TeamMember, ...] = ()
    notes: Dict[str, Tuple[Note, ...]] = field(default_factory=dict)


def _store_path() -> Path:
    return shared_path(COLLABORATION.store_filename)


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
                authenticated=bool(n.get("authenticated", False)),
                issuer=str(n.get("issuer") or ""),
                author_key=str(n.get("author_key") or ""),
                parent_id=str(n.get("parent_id") or ""),
                hidden=bool(n.get("hidden", False)),
                hidden_by=str(n.get("hidden_by") or ""),
                hidden_at=str(n.get("hidden_at") or ""),
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
                 "mentions": list(n.mentions), "notified": list(n.notified),
                 "authenticated": n.authenticated, "issuer": n.issuer,
                 "author_key": n.author_key, "parent_id": n.parent_id,
                 "hidden": n.hidden, "hidden_by": n.hidden_by,
                 "hidden_at": n.hidden_at} for n in items]
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

SIGN_IN_TO_POST = (
    "Sign in to post. Reading stays open to everyone on this instance, but "
    "a note needs a verified author so that it can be replied to and so "
    "that only its author can remove it — a typed name could be anyone."
)


def add_note(store: CollaborationStore, ticker: str, author: str, body: str,
             authenticated: bool = False, issuer: str = "",
             author_key: str = "", parent_id: str = ""
             ) -> Tuple[CollaborationStore, Optional[Note], Optional[str]]:
    """Append a note, or a reply, to a ticker's thread. Returns
    (store, note, error).

    POSTING REQUIRES A VERIFIED AUTHOR. This is the moderation model in
    one rule: a note that nobody provably wrote can be owned by nobody,
    and a thread where anyone can claim any name has no accountability
    to moderate with. Notes written under the earlier rule, with a typed
    name, are still loaded and shown — they are simply not removable by
    anyone, which the panel says.

    Newest-last, so the thread reads chronologically like a conversation.
    """
    ticker = (ticker or "").strip().upper()
    author = (author or "").strip()
    body = (body or "").strip()
    if not ticker:
        return store, None, "No ticker to attach this note to."
    if not authenticated or not (author_key or "").strip():
        return store, None, SIGN_IN_TO_POST
    if not author:
        return store, None, "Enter your name so the note has an author."
    if not body:
        return store, None, "Write something first."
    if len(body) > COLLABORATION.max_note_chars:
        return store, None, f"Notes are capped at {COLLABORATION.max_note_chars} characters."

    parent_id = (parent_id or "").strip()
    if parent_id:
        parent = next((n for n in store.notes.get(ticker, ()) if n.id == parent_id), None)
        if parent is None:
            return store, None, "The note you are replying to is no longer here."
        if parent.is_reply:
            # One level only — see Note.parent_id.
            return store, None, "Reply to the original note rather than to a reply."
        if parent.hidden:
            return store, None, "That note has been removed, so it cannot be replied to."

    note = Note(
        id=uuid.uuid4().hex,
        ticker=ticker,
        author=author,
        body=body,
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        mentions=parse_mentions(body, store.members),
        authenticated=authenticated,
        issuer=issuer,
        author_key=author_key.strip(),
        parent_id=parent_id,
    )
    thread = store.notes.get(ticker, ()) + (note,)
    return replace(store, notes={**store.notes, ticker: thread}), note, None


# --- ownership and removal ----------------------------------------------------

def can_moderate(note: Note, user_key: str) -> bool:
    """Whether the signed-in account may hide or restore this note.

    ONLY THE VERIFIED AUTHOR. There is no moderator role in this build —
    that is the RBAC ticket's territory — so the rule is ownership, and
    ownership is the account key, not the display name. A note written
    signed-out has no key and therefore no owner: it can be removed by
    nobody, which is stated on screen rather than left as a surprise.
    """
    user_key = (user_key or "").strip()
    return bool(user_key and note.authenticated and note.author_key == user_key)


def _update_note(store: CollaborationStore, ticker: str, note_id: str, **changes) -> CollaborationStore:
    thread = tuple(replace(n, **changes) if n.id == note_id else n
                   for n in store.notes.get(ticker, ()))
    return replace(store, notes={**store.notes, ticker: thread})


def hide_note(store: CollaborationStore, ticker: str, note_id: str,
              user_key: str) -> Tuple[CollaborationStore, Optional[str]]:
    """Soft-delete. Returns (store, error).

    The note stays in the store with who hid it and when. Its replies
    stay too, attached to a hidden parent, so restoring the note brings
    the conversation back whole rather than leaving orphans.
    """
    note = next((n for n in store.notes.get(ticker, ()) if n.id == note_id), None)
    if note is None:
        return store, "That note is no longer here."
    if not can_moderate(note, user_key):
        if not note.authenticated:
            return store, ("This note was posted without signing in, so it has "
                           "no owner and cannot be removed.")
        return store, "Only the person who wrote a note can remove it."
    return _update_note(
        store, ticker, note_id, hidden=True, hidden_by=(user_key or "").strip(),
        hidden_at=datetime.datetime.now().isoformat(timespec="seconds"),
    ), None


def restore_note(store: CollaborationStore, ticker: str, note_id: str,
                 user_key: str) -> Tuple[CollaborationStore, Optional[str]]:
    """Undo a hide. Same rule as hiding: the author, and nobody else."""
    note = next((n for n in store.notes.get(ticker, ()) if n.id == note_id), None)
    if note is None:
        return store, "That note is no longer here."
    if not can_moderate(note, user_key):
        return store, "Only the person who wrote a note can restore it."
    return _update_note(store, ticker, note_id, hidden=False, hidden_by="",
                        hidden_at=""), None


def visible_notes_for(store: CollaborationStore, ticker: str) -> Tuple[Note, ...]:
    """The thread as readers see it: hidden notes gone, and a reply to a
    hidden note gone with it — a reply with no visible parent reads as a
    non sequitur."""
    thread = notes_for(store, ticker)
    hidden_ids = {n.id for n in thread if n.hidden}
    return tuple(n for n in thread
                 if not n.hidden and n.parent_id not in hidden_ids)


def top_level(store: CollaborationStore, ticker: str) -> Tuple[Note, ...]:
    return tuple(n for n in visible_notes_for(store, ticker) if not n.is_reply)


def replies_for(store: CollaborationStore, ticker: str, parent_id: str) -> Tuple[Note, ...]:
    return tuple(n for n in visible_notes_for(store, ticker) if n.parent_id == parent_id)


def hidden_by_author(store: CollaborationStore, ticker: str, user_key: str) -> Tuple[Note, ...]:
    """The notes this account has hidden on this ticker, so it can restore
    them. Nobody else's — a hidden note is not readable by other accounts,
    or hiding it would have done nothing."""
    user_key = (user_key or "").strip()
    if not user_key:
        return ()
    return tuple(n for n in notes_for(store, ticker)
                 if n.hidden and n.author_key == user_key)


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
        if note.authenticated:
            identity_note = COLLABORATION.identity_note_authenticated.format(
                issuer=note.issuer or "their identity provider",
            )
        else:
            identity_note = COLLABORATION.identity_note_self_declared
        body = COLLABORATION.mention_body_template.format(
            name=member.name, author=note.author, ticker=note.ticker, body=note.body,
            identity_note=identity_note,
        )
        ok, err = sender(member.email, subject, body)
        if ok:
            notified.append(name)
        else:
            errors.append(f"{name}: {err}")
    if notified:
        logger.log(logging.INFO, "collaboration.mentions_notified count=%d", len(notified))
    return tuple(notified), errors
