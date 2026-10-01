"""Admin / Analyst / Viewer roles for a shared workspace.

WHAT THIS IS, PRECISELY, because the word "access control" promises more
than any in-process check can deliver. This is an authorisation boundary
INSIDE the app: every gated action asks `require()` before it acts, and
the UI also hides what the reader cannot do. It is NOT protection against
somebody with shell or filesystem access to the instance — on a laptop
the operator owns every file in `QUANTIX/`, and no amount of Python
changes that. It is a real boundary on the one-deployment-per-firm model
branding.py already describes, where colleagues reach the app and not the
disk. The panel says this in those words rather than leaving a reader to
assume otherwise.

BOTH HALVES ARE NEEDED. Hiding a control is not enforcement: Streamlit
reruns constantly, a session can hold a role that has since changed, and
a cached widget can fire against a stale view. Enforcing without hiding
is worse in the other direction — a control that refuses when pressed
advertises a power the reader does not have, which is the exact mistake
collaboration's ✕ button was fixed to avoid. So: check in the function,
hide in the UI.

THE FIRST ACCOUNT BECOMES ADMIN. On a fresh instance the person setting
it up is the operator and there is nobody to ask. Bootstrapping from the
ROLE STORE rather than from accounts.py is deliberate — OIDC identities
never get an Account record, so a bootstrap keyed on that table would
leave a Google-only instance with no administrator at all. Whoever first
signs in while no Admin exists becomes one, and it is recorded with
`granted_by="bootstrap"` so the trail shows it was not granted by a
person.

THE LAST ADMIN CANNOT BE DEMOTED, by themselves or by anyone. Otherwise
an instance locks itself out of its own administration and the only way
back is editing JSON by hand — which is exactly the kind of recovery an
enterprise feature must not require.

PERMISSIONS ARE A CLOSED VOCABULARY mapped to a minimum role. A call site
names a permission rather than comparing roles inline, so the matrix
lives in one place and a new gated action cannot quietly invent its own
rule. `can_moderate()` in collaboration.py is the one integration the
codebase already predicted: it widens to admins and nothing else changes.

ROLES ARE KEYED ON THE ACCOUNT KEY, never the email or the display name.
Two accounts can share a name, an email is mutable at most providers, and
`auth.key_for()` is the one identifier both sign-in paths produce.
"""
import datetime
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import RBAC
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_exception

logger = get_logger("rbac")

ADMIN = "admin"
ANALYST = "analyst"
VIEWER = "viewer"

# Most privileged first. The ORDER is the privilege ladder — `can()`
# compares positions in this tuple, so inserting a role in the middle
# changes the matrix deliberately rather than by accident.
ROLES: Tuple[str, ...] = (ADMIN, ANALYST, VIEWER)

BOOTSTRAP = "bootstrap"

NOT_SIGNED_IN = "Sign in to do that."
NOT_PERMITTED = ("Your role ({role}) cannot do that. {need} is required — ask an "
                 "administrator on this instance.")
LAST_ADMIN = (
    "This is the only administrator on this instance, so the role cannot be "
    "changed. Promote somebody else first — otherwise nobody could grant roles "
    "again without editing files by hand."
)
NOT_ADMIN_GRANT = "Only an administrator can change roles."
UNKNOWN_ROLE = "Unknown role."

WHAT_THIS_IS = (
    "Roles are enforced inside the app: a gated action checks before it acts, and "
    "controls you cannot use are hidden rather than shown and refused. This is not "
    "protection against someone with shell or file access to the machine Quantix "
    "runs on — whoever owns the disk owns the data, and no in-process check changes "
    "that. It is a real boundary where people reach the app and not the filesystem."
)

BOOTSTRAP_NOTE = (
    "The first account to sign in on an instance with no administrator becomes one, "
    "because there is nobody else to ask. Every account after that starts as a "
    "{default} until an administrator changes it."
)


@dataclass(frozen=True)
class RoleSpec:
    id: str
    label: str
    blurb: str


ROLE_SPECS: Tuple[RoleSpec, ...] = (
    RoleSpec(ADMIN, "Admin",
             "Everything an Analyst can do, plus instance configuration: API "
             "keys, webhooks, branding, delivery settings, the audit trail, and "
             "granting roles. Can also moderate any shared note."),
    RoleSpec(ANALYST, "Analyst",
             "Full research use, and may write to shared surfaces — post notes, "
             "enter the monthly contest, join the leaderboard, share a return."),
    RoleSpec(VIEWER, "Viewer",
             "Reads every analysis and every shared surface, and keeps their own "
             "watchlists, journal and portfolio. Writes nothing other people see."),
)

ROLE_LABELS: Dict[str, str] = {spec.id: spec.label for spec in ROLE_SPECS}


@dataclass(frozen=True)
class Permission:
    id: str
    label: str
    min_role: str


# The matrix. A call site names one of these; it never compares roles.
PERMISSIONS: Tuple[Permission, ...] = (
    # --- instance configuration: Admin ---
    Permission("admin.roles", "Grant and change roles", ADMIN),
    Permission("admin.audit_read", "Read the audit trail", ADMIN),
    Permission("admin.audit_erase", "Erase an account's audit identity", ADMIN),
    Permission("admin.api_keys", "Issue and revoke API keys", ADMIN),
    Permission("admin.webhooks", "Register and remove webhooks", ADMIN),
    Permission("admin.branding", "Change branding", ADMIN),
    Permission("admin.delivery", "Change digest and Slack delivery", ADMIN),
    Permission("admin.moderate_any", "Moderate anyone's shared note", ADMIN),

    # --- shared state: Analyst ---
    Permission("shared.post_note", "Post a team note", ANALYST),
    Permission("shared.contest_enter", "Enter the monthly contest", ANALYST),
    Permission("shared.leaderboard_join", "Join the leaderboard", ANALYST),
    Permission("shared.peer_publish", "Share a monthly return", ANALYST),
    Permission("shared.profile_publish", "Publish a followable profile", ANALYST),

    # --- own work: Viewer ---
    Permission("own.watchlist", "Keep watchlists", VIEWER),
    Permission("own.journal", "Keep an investment journal", VIEWER),
    Permission("own.portfolio", "Record holdings", VIEWER),
    Permission("own.thresholds", "Set personal analysis thresholds", VIEWER),
    Permission("own.export", "Export analysis", VIEWER),
)

PERMISSION_MAP: Dict[str, Permission] = {p.id: p for p in PERMISSIONS}


@dataclass(frozen=True)
class Assignment:
    """One account's role. Keyed on the ACCOUNT KEY — two accounts can
    share a display name and an email is mutable at most providers."""
    user_key: str
    role: str
    granted_by: str = ""
    granted_at: str = ""


@dataclass(frozen=True)
class RoleStore:
    assignments: Tuple[Assignment, ...] = ()
    corrupt: bool = False

    def get(self, user_key: str) -> Optional[Assignment]:
        user_key = (user_key or "").strip()
        if not user_key:
            return None
        return next((a for a in self.assignments if a.user_key == user_key), None)

    @property
    def admin_keys(self) -> Tuple[str, ...]:
        return tuple(a.user_key for a in self.assignments if a.role == ADMIN)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _store_path() -> Path:
    # Shared: roles are a property of the instance, not of one namespace.
    # A per-user role file would let everyone grant themselves Admin.
    return shared_path(RBAC.store_filename)


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None) -> RoleStore:
    """Never raises. Corrupt is distinguished from missing, because
    treating an unreadable role file as empty would re-run the bootstrap
    and hand Admin to whoever happened to be signed in."""
    path = path or _store_path()
    if not path.exists():
        return RoleStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "rbac.store_corrupt", section="rbac")
        return RoleStore(corrupt=True)
    if not isinstance(raw, dict):
        return RoleStore(corrupt=True)

    out: List[Assignment] = []
    seen: set = set()
    for item in raw.get("assignments", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("user_key") or "").strip()
        role = str(item.get("role") or "").strip().lower()
        if not key or key in seen or role not in ROLES:
            continue
        seen.add(key)
        out.append(Assignment(key, role, str(item.get("granted_by") or ""),
                              str(item.get("granted_at") or "")))
    return RoleStore(tuple(out))


def save_store(store: RoleStore, path: Optional[Path] = None) -> bool:
    if store.corrupt:
        return False
    path = path or _store_path()
    payload = {"assignments": [{
        "user_key": a.user_key, "role": a.role,
        "granted_by": a.granted_by, "granted_at": a.granted_at,
    } for a in store.assignments]}
    atomic_write_text(path, json.dumps(payload, indent=2))
    return True


# --- reading a role -----------------------------------------------------------

def role_for(store: RoleStore, user_key: str) -> str:
    """This account's role. Unassigned accounts get the configured
    default, which is the LEAST privileged role by design."""
    assignment = store.get(user_key)
    return assignment.role if assignment else RBAC.default_role


def can(role: str, permission: str) -> bool:
    """Whether `role` satisfies `permission`.

    An UNKNOWN permission is refused rather than allowed. A typo at a
    call site must close the door, not open it — the opposite default is
    how a gate silently stops gating.
    """
    spec = PERMISSION_MAP.get(permission)
    if spec is None:
        return False
    if role not in ROLES:
        return False
    return ROLES.index(role) <= ROLES.index(spec.min_role)


def require(role: str, permission: str) -> str:
    """"" when permitted, else the reason. Call sites check this BEFORE
    acting — hiding the control is the other half, not the whole."""
    if not role:
        return NOT_SIGNED_IN
    if can(role, permission):
        return ""
    spec = PERMISSION_MAP.get(permission)
    need = ROLE_LABELS.get(spec.min_role, "A higher role") if spec else "A known permission"
    return NOT_PERMITTED.format(role=ROLE_LABELS.get(role, role), need=need)


def is_admin(store: RoleStore, user_key: str) -> bool:
    return role_for(store, user_key) == ADMIN


# --- bootstrap ----------------------------------------------------------------

def needs_bootstrap(store: RoleStore) -> bool:
    """True when nobody can administer this instance yet."""
    return not store.corrupt and not store.admin_keys


def bootstrap(store: RoleStore, user_key: str,
              now: Optional[str] = None) -> Tuple[RoleStore, bool]:
    """Make `user_key` Admin if the instance has none. (store, granted).

    Recorded with granted_by="bootstrap" rather than a person's key, so
    the assignment is visibly automatic and an auditor is not left
    wondering which account granted the first one.

    Refuses on a corrupt store: re-running the bootstrap because a file
    could not be read would hand Admin to whoever happened to be signed
    in at the time.
    """
    user_key = (user_key or "").strip()
    if not user_key or store.corrupt or not needs_bootstrap(store):
        return store, False
    kept = tuple(a for a in store.assignments if a.user_key != user_key)
    return replace(store, assignments=kept + (
        Assignment(user_key, ADMIN, BOOTSTRAP, now or _now_iso()),)), True


# --- granting -----------------------------------------------------------------

def grant(store: RoleStore, actor_key: str, target_key: str, role: str,
          now: Optional[str] = None) -> Tuple[RoleStore, str]:
    """Set `target_key`'s role. (store, error).

    Only an Admin may call this, and the LAST Admin cannot be demoted —
    by themselves or by anyone. An instance that can lock itself out of
    its own administration has a recovery path that involves editing JSON
    by hand, which is not a recovery path.
    """
    actor_key = (actor_key or "").strip()
    target_key = (target_key or "").strip()
    role = (role or "").strip().lower()

    if store.corrupt:
        return store, "The role file could not be read, so roles cannot be changed."
    if not actor_key:
        return store, NOT_SIGNED_IN
    if not is_admin(store, actor_key):
        return store, NOT_ADMIN_GRANT
    if not target_key:
        return store, "Name the account to change."
    if role not in ROLES:
        return store, UNKNOWN_ROLE

    current = role_for(store, target_key)
    if current == role:
        return store, ""
    if current == ADMIN and role != ADMIN and store.admin_keys == (target_key,):
        return store, LAST_ADMIN

    kept = tuple(a for a in store.assignments if a.user_key != target_key)
    return replace(store, assignments=kept + (
        Assignment(target_key, role, actor_key, now or _now_iso()),)), ""


def assignments_for_display(store: RoleStore) -> Tuple[Assignment, ...]:
    """Admins first, then by key, so the list is stable between renders
    and the people who can change things are at the top."""
    return tuple(sorted(store.assignments,
                        key=lambda a: (ROLES.index(a.role)
                                       if a.role in ROLES else len(ROLES),
                                       a.user_key)))


def permissions_for(role: str) -> Tuple[Permission, ...]:
    return tuple(p for p in PERMISSIONS if can(role, p.id))


def describe(role: str) -> str:
    spec = next((s for s in ROLE_SPECS if s.id == role), None)
    return spec.blurb if spec else "No role on this instance."


def bootstrap_note() -> str:
    return BOOTSTRAP_NOTE.format(
        default=ROLE_LABELS.get(RBAC.default_role, RBAC.default_role))
