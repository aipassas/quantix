"""A tamper-evident audit trail: who did what, when, in a chain that
cannot be edited without the edit showing.

THE EXISTING LOG IS NOT AN AUDIT TRAIL, and measuring it first is what
shaped this. `logging_setup` already carries 174 `log_event` call sites,
so the ticket's "log every significant user action" is substantially done
already. But measured on 2026-10-01:

    log_event calls that name the ACCOUNT      7 of 174
    oldest surviving record                    2026-08-23  (~32 days)
    retention policy                           a 4 MB byte budget

Ninety-six per cent of the stream cannot answer "who", the file is
truncated by SIZE rather than by any retention decision — everything
before 23 August is simply gone — and it is formatted prose, so reporting
on it means parsing English. None of those is fixable by writing more
log_event calls.

SO THE TRAIL IS ITS OWN FILE, APPEND-ONLY, NEVER ROTATED, and written
through `record()`, whose signature cannot be satisfied without an actor.
A record with no actor is impossible by construction rather than by
discipline — the same signature-as-boundary argument peer_comparison
makes about holdings. `log_event` stays what it is: diagnostics.

"IMMUTABLE" IS NOT ACHIEVABLE HERE AND IS NOT CLAIMED. There is no WORM
storage and no external service; the operator owns the disk and can edit
or delete anything on it. What is achievable is tamper EVIDENCE: each
record carries the hash of the one before, so any edit, deletion or
reordering breaks the chain and `verify()` names the exact record where
it broke. The UI says detects, never prevents. Promising immutability on
a file somebody owns would be the kind of claim this app refuses
everywhere else.

ERASURE AND THE CHAIN ARE RECONCILED BY HASHING THE DIGEST. The chain
covers `actor_digest` — sha256 of the account key — rather than the
readable actor field. A GDPR erasure replaces the actor with a tombstone
and leaves the digest intact, so the chain still verifies, the person is
no longer identifiable in it, and the erasure is itself appended as a
record. `verify()` distinguishes an erased record from a tampered one by
checking that a NON-erased actor still hashes to its stored digest:
change the actor alone and the digest mismatches; change both and the
chain breaks. Deleting the lines instead would break the chain, which
would make honouring a lawful request look exactly like forgery.

NO COMPLIANCE CLAIM IS MADE. `export_rows()` produces evidence an auditor
can review with the verification result attached. It is not an
attestation, it names no control as met, and it maps nothing to SOX,
HIPAA or GDPR clauses — branding.py already locks "No certification is
claimed" against rebranding, and SOC 2 is a separate open ticket. A
framework-named template would imply an assessment nobody performed,
which is the same category error as scoring an ETF on filings it never
makes.
"""
import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import AUDIT
from local_store import shared_path
from logging_setup import get_logger, log_exception

logger = get_logger("audit")

# The first record's predecessor. A literal rather than "" so a truncated
# file cannot be mistaken for a valid chain that merely starts late.
GENESIS = "0" * 64

NO_ACTOR = (
    "An audit record needs the account that caused it. A trail that cannot say "
    "who did something is not an audit trail, which is exactly the gap this "
    "exists to close."
)
UNKNOWN_ACTION = (
    "Unknown audit action. The vocabulary is closed on purpose: a free-text "
    "action makes the trail unreportable and lets a caller invent a category "
    "nobody reviews."
)

NOT_IMMUTABLE = (
    "This trail is tamper-EVIDENT, not tamper-proof. Quantix runs on your own "
    "machine, so nothing here can stop the file being edited or deleted — there "
    "is no write-once storage and no external service holding a copy. What it "
    "does is make any edit, deletion or reordering visible: every record carries "
    "the hash of the one before it, and verification names the first record where "
    "the chain breaks."
)

NO_COMPLIANCE_CLAIM = (
    "This is evidence, not an attestation. The export records what happened and "
    "whether the chain verifies; it does not assess any control, does not map to "
    "SOX, HIPAA or GDPR clauses, and is not a statement that any requirement is "
    "met. No certification is claimed."
)

ERASURE_NOTE = (
    "Erasure replaces the account identifier with a tombstone and leaves the "
    "record and its hash in place, so the trail still verifies and the erasure "
    "itself is recorded. Deleting the lines would break the chain and make a "
    "lawful request indistinguishable from tampering."
)

# --- the closed vocabulary ----------------------------------------------------

AUTH = "auth"
EXPORT = "export"
CONFIG = "config"
SHARING = "sharing"
DELETION = "deletion"

CATEGORIES: Tuple[str, ...] = (AUTH, EXPORT, CONFIG, SHARING, DELETION)


@dataclass(frozen=True)
class ActionSpec:
    id: str
    label: str
    category: str


# The ticket names "logins, exports, config changes". The other two
# categories are the surfaces PHASE 4 built: an opt-in that discloses
# something about a person, and a deletion that destroys their record,
# are both exactly what an auditor asks about.
ACTION_SPECS: Tuple[ActionSpec, ...] = (
    ActionSpec("sign_in", "Signed in", AUTH),
    ActionSpec("sign_in_failed", "Failed sign-in", AUTH),
    ActionSpec("sign_out", "Signed out", AUTH),
    ActionSpec("account_created", "Account created", AUTH),
    ActionSpec("password_reset", "Password reset", AUTH),

    ActionSpec("export_pdf", "Exported a PDF", EXPORT),
    ActionSpec("export_workbook", "Exported a workbook", EXPORT),
    ActionSpec("export_deck", "Exported a deck", EXPORT),
    ActionSpec("report_emailed", "Emailed a report", EXPORT),
    ActionSpec("audit_exported", "Exported the audit trail", EXPORT),

    ActionSpec("thresholds_changed", "Changed analysis thresholds", CONFIG),
    ActionSpec("api_key_issued", "Issued an API key", CONFIG),
    ActionSpec("api_key_revoked", "Revoked an API key", CONFIG),
    ActionSpec("webhook_added", "Registered a webhook", CONFIG),
    ActionSpec("webhook_removed", "Removed a webhook", CONFIG),
    ActionSpec("digest_configured", "Changed digest delivery", CONFIG),

    ActionSpec("sharing_opted_in", "Started sharing", SHARING),
    ActionSpec("sharing_opted_out", "Stopped sharing", SHARING),

    ActionSpec("profile_deleted", "Deleted a profile", DELETION),
    ActionSpec("portfolio_deleted", "Deleted a portfolio", DELETION),
    ActionSpec("data_erased", "Erased an account's audit identity", DELETION),
)

ACTIONS: Dict[str, ActionSpec] = {spec.id: spec for spec in ACTION_SPECS}


@dataclass(frozen=True)
class AuditRecord:
    """One event. `actor_digest` is what the chain covers, so the readable
    `actor` can be tombstoned on erasure without breaking it."""
    seq: int
    at: str
    actor: str
    actor_digest: str
    action: str
    target: str
    detail: str
    prev_hash: str
    record_hash: str

    @property
    def erased(self) -> bool:
        return self.actor == AUDIT.tombstone

    @property
    def category(self) -> str:
        spec = ACTIONS.get(self.action)
        return spec.category if spec else ""

    @property
    def label(self) -> str:
        spec = ACTIONS.get(self.action)
        return spec.label if spec else self.action


@dataclass(frozen=True)
class Verification:
    """The result of walking the chain."""
    ok: bool = True
    checked: int = 0
    erased: int = 0
    broken_at: Optional[int] = None     # seq of the first bad record
    reason: str = ""

    def sentence(self) -> str:
        if not self.checked:
            return "No audit records yet, so there is nothing to verify."
        if self.ok:
            tail = (f" {self.erased} record(s) carry an erased identity."
                    if self.erased else "")
            return (f"All {self.checked} records verify — the chain is intact "
                    f"from the first record to the last.{tail}")
        return (f"The chain BREAKS at record {self.broken_at} of {self.checked}. "
                f"{self.reason}")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _path() -> Path:
    # Shared, never per-user: an audit trail scoped to the account that
    # caused the events is one each person could curate.
    return shared_path(AUDIT.store_filename)


def digest_for(actor: str) -> str:
    return hashlib.sha256((actor or "").encode("utf-8")).hexdigest()


def _hash(seq: int, at: str, actor_digest: str, action: str, target: str,
          detail: str, prev_hash: str) -> str:
    """The chain hash.

    Canonical JSON with sorted keys and no incidental whitespace, so the
    same record always hashes the same way regardless of how it was
    serialised. Note the readable ACTOR is deliberately absent — see the
    module docstring on erasure.
    """
    payload = json.dumps({
        "seq": seq, "at": at, "actor_digest": actor_digest, "action": action,
        "target": target, "detail": detail, "prev_hash": prev_hash,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- reading ------------------------------------------------------------------

def _to_record(raw) -> Optional[AuditRecord]:
    if not isinstance(raw, dict):
        return None
    try:
        return AuditRecord(
            seq=int(raw["seq"]), at=str(raw.get("at") or ""),
            actor=str(raw.get("actor") or ""),
            actor_digest=str(raw.get("actor_digest") or ""),
            action=str(raw.get("action") or ""),
            target=str(raw.get("target") or ""),
            detail=str(raw.get("detail") or ""),
            prev_hash=str(raw.get("prev_hash") or ""),
            record_hash=str(raw.get("record_hash") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_all(path: Optional[Path] = None) -> Tuple[AuditRecord, ...]:
    """Every record, in file order.

    A line that cannot be parsed is KEPT OUT of the list but is not
    silently forgiven — `verify()` walks the raw line count against the
    record count, so a corrupted line shows up as a break rather than as
    a shorter trail.
    """
    path = path or _path()
    if not path.exists():
        return ()
    records: List[AuditRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = _to_record(json.loads(line))
            except Exception:
                parsed = None
            if parsed is not None:
                records.append(parsed)
    except Exception:
        log_exception(logger, "audit.read_failed", section="audit")
        return ()
    return tuple(records)


def line_count(path: Optional[Path] = None) -> int:
    """Non-empty lines on disk, so an unparseable one can be detected."""
    path = path or _path()
    if not path.exists():
        return 0
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
                   if ln.strip())
    except Exception:
        return 0


# --- writing ------------------------------------------------------------------

def record(actor: str, action: str, target: str = "", detail: str = "",
           path: Optional[Path] = None,
           now: Optional[str] = None) -> Tuple[Optional[AuditRecord], str]:
    """Append one record. Returns (record, error).

    THE SIGNATURE IS THE POINT. There is no default for `actor` and no
    code path that writes a record without one, so the gap measured in
    the existing log — 7 of 174 events naming an account — cannot recur
    here by somebody forgetting.

    Appends a single line rather than rewriting the file: a whole-file
    write is how an append-only log stops being append-only, and it would
    also race with the cron processes that share this file.
    """
    actor = (actor or "").strip()
    if not actor:
        return None, NO_ACTOR
    if action not in ACTIONS:
        return None, UNKNOWN_ACTION

    path = path or _path()
    existing = read_all(path)
    prev_hash = existing[-1].record_hash if existing else GENESIS
    seq = (existing[-1].seq + 1) if existing else 1

    at = now or _now_iso()
    target = str(target or "").strip()
    detail = str(detail or "").strip()[:AUDIT.max_detail_chars]
    actor_digest = digest_for(actor)
    record_hash = _hash(seq, at, actor_digest, action, target, detail, prev_hash)

    entry = AuditRecord(seq, at, actor, actor_digest, action, target, detail,
                        prev_hash, record_hash)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "seq": entry.seq, "at": entry.at, "actor": entry.actor,
                "actor_digest": entry.actor_digest, "action": entry.action,
                "target": entry.target, "detail": entry.detail,
                "prev_hash": entry.prev_hash, "record_hash": entry.record_hash,
            }, sort_keys=True) + "\n")
    except Exception:
        log_exception(logger, "audit.write_failed", section="audit")
        return None, "The audit trail could not be written."
    return entry, ""


# --- verification -------------------------------------------------------------

def verify(records: Sequence[AuditRecord],
           raw_lines: Optional[int] = None) -> Verification:
    """Walk the chain and report the FIRST break.

    Four ways a trail can be wrong, and each is distinguished:
      * a record's own hash does not match its contents   — it was edited
      * prev_hash does not match the previous record      — one was removed
                                                            or reordered
      * the sequence numbers are not consecutive          — one was removed
      * a non-erased actor does not match its digest      — the actor alone
                                                            was swapped
    An ERASED actor is not a break: that is the documented erasure path.
    """
    if raw_lines is not None and raw_lines != len(records):
        return Verification(
            ok=False, checked=len(records), broken_at=None,
            reason=(f"{raw_lines} line(s) on disk but only {len(records)} are "
                    "readable records — the file has been edited."))
    if not records:
        return Verification(ok=True, checked=0)

    previous_hash = GENESIS
    erased = 0
    for index, entry in enumerate(records, start=1):
        if entry.seq != index:
            return Verification(
                ok=False, checked=len(records), broken_at=entry.seq,
                reason=(f"Expected record {index} but found {entry.seq} — a "
                        "record has been removed or reordered."))
        if entry.prev_hash != previous_hash:
            return Verification(
                ok=False, checked=len(records), broken_at=entry.seq,
                reason=("Its link to the previous record does not match — "
                        "something before it was changed or removed."))
        expected = _hash(entry.seq, entry.at, entry.actor_digest, entry.action,
                         entry.target, entry.detail, entry.prev_hash)
        if entry.record_hash != expected:
            return Verification(
                ok=False, checked=len(records), broken_at=entry.seq,
                reason="Its contents do not match its own hash — it was edited.")
        if entry.erased:
            erased += 1
        elif digest_for(entry.actor) != entry.actor_digest:
            return Verification(
                ok=False, checked=len(records), broken_at=entry.seq,
                reason=("Its actor does not match the identity recorded in the "
                        "chain — the actor field was swapped."))
        previous_hash = entry.record_hash

    return Verification(ok=True, checked=len(records), erased=erased)


def verify_file(path: Optional[Path] = None) -> Verification:
    path = path or _path()
    return verify(read_all(path), raw_lines=line_count(path))


# --- erasure ------------------------------------------------------------------

def erase_actor(actor: str, path: Optional[Path] = None
                ) -> Tuple[int, str]:
    """Pseudonymise every record belonging to `actor`. (count, error).

    Replaces the readable actor with the tombstone and leaves the digest,
    the contents and the hash untouched — so the chain still verifies and
    the records remain as evidence that the events happened. The erasure
    is appended as its own record by the CALLER, which is why this does
    not write one itself: the caller knows who requested it.
    """
    actor = (actor or "").strip()
    if not actor:
        return 0, NO_ACTOR
    path = path or _path()
    records = read_all(path)
    if not records:
        return 0, ""

    rewritten: List[str] = []
    erased = 0
    for entry in records:
        actor_value = entry.actor
        if actor_value == actor:
            actor_value = AUDIT.tombstone
            erased += 1
        rewritten.append(json.dumps({
            "seq": entry.seq, "at": entry.at, "actor": actor_value,
            "actor_digest": entry.actor_digest, "action": entry.action,
            "target": entry.target, "detail": entry.detail,
            "prev_hash": entry.prev_hash, "record_hash": entry.record_hash,
        }, sort_keys=True))

    if not erased:
        return 0, ""
    try:
        path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    except Exception:
        log_exception(logger, "audit.erase_failed", section="audit")
        return 0, "The audit trail could not be rewritten."
    return erased, ""


# --- reporting ----------------------------------------------------------------

def filtered(records: Sequence[AuditRecord], category: str = "",
             actor: str = "", since: str = "",
             until: str = "") -> Tuple[AuditRecord, ...]:
    """The records an auditor asked for. Pure filtering — never reorders,
    because the sequence IS the evidence."""
    out = list(records)
    if category:
        out = [r for r in out if r.category == category]
    if actor:
        out = [r for r in out if r.actor == actor]
    if since:
        out = [r for r in out if r.at >= since]
    if until:
        out = [r for r in out if r.at <= until]
    return tuple(out)


EXPORT_COLUMNS: Tuple[str, ...] = (
    "seq", "at", "actor", "action", "category", "target", "detail",
    "record_hash",
)


def export_rows(records: Sequence[AuditRecord]) -> Tuple[Dict[str, str], ...]:
    """Rows for a CSV. The hash travels WITH each row so an export can be
    checked back against the trail it came from."""
    return tuple({
        "seq": str(r.seq), "at": r.at, "actor": r.actor, "action": r.action,
        "category": r.category, "target": r.target, "detail": r.detail,
        "record_hash": r.record_hash,
    } for r in records)


def export_payload(records: Sequence[AuditRecord],
                   result: Verification) -> str:
    """The JSON evidence bundle: the records, the verification, and the
    statement that this is not an attestation."""
    return json.dumps({
        "generated_at": _now_iso(),
        "records": list(export_rows(records)),
        "verification": {
            "ok": result.ok, "checked": result.checked,
            "erased": result.erased, "broken_at": result.broken_at,
            "reason": result.reason, "summary": result.sentence(),
        },
        "notice": NO_COMPLIANCE_CLAIM,
        "tamper_evidence": NOT_IMMUTABLE,
    }, indent=2)


def counts_by_category(records: Sequence[AuditRecord]) -> Dict[str, int]:
    out = {name: 0 for name in CATEGORIES}
    for entry in records:
        if entry.category in out:
            out[entry.category] += 1
    return out


def coverage_note() -> str:
    """What the trail covers, in the panel's own words."""
    return (f"{len(ACTION_SPECS)} kinds of event across "
            f"{len(CATEGORIES)} categories: "
            + ", ".join(CATEGORIES) + ".")
