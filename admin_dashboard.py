"""Who is on this instance, what they can do, and what they are using.

THE USER LIST COMES FROM users/, NOT FROM accounts.py, and measuring that
first is what shaped this. accounts.py knows only local email/password
accounts; an OIDC identity never gets an Account record at all. Measured
2026-10-01 on this very instance:

    identities with data in users/      2   (one Google, one local)
    identities accounts.py can see      1

A dashboard built the obvious way would show HALF the users here — and
none at all on a firm signing in through Okta or Google Workspace, which
is precisely the deployment the ticket is written for. Every identity
that has ever signed in has a namespace directory, because `store_path()`
creates it on first write, so that listing is the only complete one. Rows
are enriched with email, name and last sign-in wherever a local Account
happens to exist, and say so plainly where one does not: a key with no
email is an identity this app genuinely knows nothing else about, not a
gap worth hiding.

THERE ARE NO SEATS, BECAUSE THERE IS NO LICENCE MODEL. Nothing in this
codebase has a plan, a quota, a subscription or a bill — grep found
none — so a seat count would be an entitlement measured against nothing.
`NO_LICENCE_MODEL` says that on screen and the panel reports the real
head count instead. Inventing "7 of 10 seats used" would be the same
fabrication as a percentile over a population that was never measured.

NOBODY IS DELETED FROM HERE. An administrator can demote to Viewer, which
removes the ability to write anything shared and covers the real need
("this person has left"). Destroying somebody's journal, portfolio and
watchlists from a dashboard is irreversible, and audit.erase_actor()
already exists as the deliberate, separately-reasoned path for a lawful
erasure request.

THE FUNCTIONS TAKE DATA, NOT PATHS. `members()` is given an already-read
listing, the accounts and the role store, so it is testable without
touching a filesystem — and so the one function that DOES read the disk,
`scan()`, is small enough to see at a glance.
"""
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rbac
from config import ADMIN_DASHBOARD
from logging_setup import get_logger, log_exception

logger = get_logger("admin_dashboard")

# How this app came to know about an identity.
LOCAL_ACCOUNT = "Local account"
SIGN_IN_ONLY = "Signed in (no local account)"

NO_LICENCE_MODEL = (
    "This build has no licensing, billing or plan, so there is nothing to count "
    "seats against and none are shown. What follows is the real number of "
    "identities that have signed in on this instance."
)

UNKNOWN_IDENTITY_NOTE = (
    "Rows without an email signed in through an identity provider rather than with "
    "a password here, so this app holds no address or display name for them — only "
    "the account key, their role, and what they have stored. That is everything it "
    "knows, not everything it is hiding."
)

NO_DELETION_NOTE = (
    "Nobody can be deleted from this panel. Setting someone to Viewer removes their "
    "ability to write anything other people see, which is what leaving usually "
    "means; destroying their journal, portfolio and watchlists is irreversible and "
    "is not a dashboard action. A lawful erasure request is handled in the audit "
    "trail, deliberately and separately."
)


@dataclass(frozen=True)
class Namespace:
    """One directory under users/ — the evidence that an identity exists."""
    user_key: str
    files: int = 0
    bytes_used: int = 0


@dataclass(frozen=True)
class Member:
    """One identity, as completely as this app can describe it."""
    user_key: str
    role: str
    source: str
    email: str = ""
    name: str = ""
    created_at: str = ""
    last_login_at: str = ""
    files: int = 0
    bytes_used: int = 0

    @property
    def known_locally(self) -> bool:
        return self.source == LOCAL_ACCOUNT

    @property
    def display(self) -> str:
        """Never the account key alone when something better exists, and
        never an invented name when it does not."""
        return self.name or self.email or self.user_key

    @property
    def role_label(self) -> str:
        return rbac.ROLE_LABELS.get(self.role, self.role)


@dataclass(frozen=True)
class Overview:
    members: Tuple[Member, ...] = ()
    by_role: Dict[str, int] = None
    local_accounts: int = 0
    sign_in_only: int = 0
    total_bytes: int = 0

    @property
    def total(self) -> int:
        return len(self.members)

    @property
    def admins(self) -> int:
        return (self.by_role or {}).get(rbac.ADMIN, 0)

    def sentence(self) -> str:
        if not self.members:
            return ("Nobody has signed in on this instance yet, so there is "
                    "nothing to show.")
        # The VERB agrees too — "1 identity have signed in" shipped once
        # in this sentence and reads as a bug about the thing it counts.
        people = "identity has" if self.total == 1 else "identities have"
        admins = "administrator" if self.admins == 1 else "administrators"
        return (f"{self.total} {people} signed in: {self.admins} {admins}, "
                f"{self.by_role.get(rbac.ANALYST, 0)} analyst(s), "
                f"{self.by_role.get(rbac.VIEWER, 0)} viewer(s).")


def key_for_account(account) -> str:
    """The namespace key a local Account resolves to.

    Derived through auth.key_for rather than reconstructed here — the
    directory name is a hash with a readable prefix, and a second copy of
    that derivation is how a dashboard starts showing a row that matches
    nobody.
    """
    try:
        import auth
        user_id = getattr(account, "user_id", "") or ""
        if not user_id:
            return ""
        return auth.key_for(auth.LOCAL_ISSUER, user_id)
    except Exception:
        log_exception(logger, "admin_dashboard.key_failed", section="admin")
        return ""


def scan(users_root: Optional[Path] = None) -> Tuple[Namespace, ...]:
    """Read the namespace directories. The only function here that
    touches a filesystem. Never raises: a dashboard that cannot list
    users should say so, not take the page down."""
    if users_root is None:
        try:
            import local_store
            users_root = local_store.app_dir() / "users"
        except Exception:
            log_exception(logger, "admin_dashboard.root_failed", section="admin")
            return ()
    try:
        if not users_root.exists():
            return ()
        out: List[Namespace] = []
        for entry in sorted(users_root.iterdir()):
            if not entry.is_dir():
                continue
            files = 0
            used = 0
            for item in entry.rglob("*"):
                if item.is_file():
                    files += 1
                    try:
                        used += item.stat().st_size
                    except OSError:
                        pass
            out.append(Namespace(entry.name, files, used))
        return tuple(out)
    except Exception:
        log_exception(logger, "admin_dashboard.scan_failed", section="admin")
        return ()


def members(namespaces: Sequence[Namespace], accounts: Sequence,
            role_store) -> Tuple[Member, ...]:
    """Every identity, enriched where this app happens to know more.

    THE NAMESPACES ARE THE SPINE. An Account with no namespace has never
    signed in and has stored nothing, but it still exists and an
    administrator asking "who has access" needs to see it — so it is
    added with zero usage rather than dropped. A namespace with no
    Account is an OIDC identity and is the normal case on the deployment
    this is for.
    """
    by_key: Dict[str, object] = {}
    for account in accounts or ():
        key = key_for_account(account)
        if key:
            by_key[key] = account

    out: List[Member] = []
    seen: set = set()
    for namespace in namespaces or ():
        account = by_key.get(namespace.user_key)
        seen.add(namespace.user_key)
        out.append(Member(
            user_key=namespace.user_key,
            role=rbac.role_for(role_store, namespace.user_key),
            source=LOCAL_ACCOUNT if account is not None else SIGN_IN_ONLY,
            email=str(getattr(account, "email", "") or "") if account else "",
            name=str(getattr(account, "name", "") or "") if account else "",
            created_at=str(getattr(account, "created_at", "") or "") if account else "",
            last_login_at=str(getattr(account, "last_login_at", "") or "") if account else "",
            files=namespace.files,
            bytes_used=namespace.bytes_used,
        ))

    # Accounts that exist but have never written anything.
    for key, account in by_key.items():
        if key in seen:
            continue
        out.append(Member(
            user_key=key, role=rbac.role_for(role_store, key),
            source=LOCAL_ACCOUNT,
            email=str(getattr(account, "email", "") or ""),
            name=str(getattr(account, "name", "") or ""),
            created_at=str(getattr(account, "created_at", "") or ""),
            last_login_at=str(getattr(account, "last_login_at", "") or ""),
        ))

    # Administrators first, then by whatever name we can show, so the
    # people who can change things are at the top and the order is stable.
    out.sort(key=lambda m: (rbac.ROLES.index(m.role)
                            if m.role in rbac.ROLES else len(rbac.ROLES),
                            m.display.lower(), m.user_key))
    return tuple(out)


def overview(people: Sequence[Member]) -> Overview:
    by_role = {role: 0 for role in rbac.ROLES}
    local = 0
    total_bytes = 0
    for member in people:
        if member.role in by_role:
            by_role[member.role] += 1
        if member.known_locally:
            local += 1
        total_bytes += member.bytes_used
    return Overview(tuple(people), by_role, local,
                    len(people) - local, total_bytes)


def format_bytes(count: int) -> str:
    """Never "0 bytes" for an identity that has stored nothing — that
    reads as a measurement when it is an absence."""
    if count <= 0:
        return "nothing stored"
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:,.0f} {unit}" if unit == "B" else f"{count:,.1f} {unit}"
        count /= 1024.0
    return f"{count:,.1f} GB"


def coverage_note(people: Sequence[Member]) -> str:
    """Says how complete the list is, in the panel's own words."""
    unknown = sum(1 for m in people if not m.known_locally)
    if not people:
        return ""
    if not unknown:
        return ("Every identity here has a local account, so an email and a last "
                "sign-in are known for all of them.")
    return (f"{unknown} of {len(people)} signed in through an identity provider, so "
            "this app holds no email or display name for them — see the note below.")
