"""Local email + password accounts: storage, lockout, reset tokens.

NO STREAMLIT IN THIS MODULE. It is storage and policy only, so the rules
that matter — lockout arithmetic, token expiry, enumeration resistance —
can be tested without a script run context, and so nothing here can
accidentally depend on session state that a background process lacks.
Session handling lives in auth.py, which already owns identity.

THE STORE IS SHARED, NOT PER-USER, and that is not an oversight. Every
other store in this app resolves through store_path() into the signed-in
user's namespace. This one cannot: it is read while deciding WHO the user
is, before any namespace exists. shared_path() is the deliberate,
documented way to say that (see local_store).

ACCOUNTS ARE KEYED BY AN IMMUTABLE user_id, NEVER BY EMAIL. auth.key_for
already refuses to key namespaces on email because email is mutable at
most providers and re-keying silently orphans a user's entire saved
setup. The same reasoning applies with more force here, where we own the
address-change path: a local account gets a random id at creation, the
email is an index into it, and changing the address moves nothing.

FAILED SIGN-INS ARE RATE LIMITED WITH PROGRESSIVE LOCKOUT, because an
offline-grade KDF does nothing about someone simply guessing against the
live form. Attempts are counted per account and the lock grows with each
further failure. The counter resets on success.

THE STORE NEVER HOLDS ANYTHING REPLAYABLE. Passwords are scrypt records;
reset tokens are stored as SHA-256 digests. Someone who reads this file
learns who has an account, not how to become them.

WRITES ARE SERIALISED THROUGH A LOCK. Sign-in updates the record it just
read (attempt counters, last-login), and Streamlit serves reruns on
worker threads — the same read-modify-write shape that let a revoked API
key verify once (see api_keys). One process-wide lock closes it.
"""
import datetime
import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import local_store
import passwords
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("accounts")

STORE_FILENAME = "accounts_store.json"

# Lockout schedule. Short enough that a real person who fat-fingered a
# password three times is not locked out of their evening; long enough
# that online guessing is hopeless against a KDF this slow.
MAX_ATTEMPTS_BEFORE_LOCK = 5
_LOCK_STEPS_MINUTES = (1, 5, 15, 60)
ATTEMPT_WINDOW_MINUTES = 15

RESET_TOKEN_TTL_MINUTES = 30

# RFC 5322 in full is a famous mistake to attempt. This checks the shape
# that matters — one @, something either side, a dot in the domain — and
# leaves real validation to whether the reset email arrives.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_write_lock = threading.RLock()


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(moment: Optional[datetime.datetime]) -> Optional[str]:
    return moment.isoformat() if moment else None


def _parse_iso(text: Optional[str]) -> Optional[datetime.datetime]:
    if not text:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def normalise_email(email: str) -> str:
    """Lower-cased and trimmed.

    Deliberately NOT doing provider-specific canonicalisation — stripping
    dots or +tags for Gmail would merge addresses their owner considers
    distinct, and would differ from how the OIDC path sees the same person.
    """
    return (email or "").strip().lower()


def looks_like_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalise_email(email)))


@dataclass
class Account:
    user_id: str
    email: str
    name: str = ""
    password_hash: str = ""
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None
    password_changed_at: Optional[str] = None
    failed_attempts: int = 0
    first_failure_at: Optional[str] = None
    locked_until: Optional[str] = None
    lock_level: int = 0
    reset_token_hash: str = ""
    reset_expires_at: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.name or self.email

    @property
    def locked(self) -> bool:
        until = _parse_iso(self.locked_until)
        return bool(until and until > _now())

    def lock_remaining_seconds(self) -> int:
        until = _parse_iso(self.locked_until)
        if not until:
            return 0
        return max(0, int((until - _now()).total_seconds()))


# --- storage ------------------------------------------------------------------

def _path():
    return local_store.shared_path(STORE_FILENAME)


def _load_raw() -> Dict:
    path = _path()
    if not path.exists():
        return {"accounts": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("accounts"), dict):
            return {"accounts": {}}
        return data
    except Exception:
        # A corrupt store must not take the login page down. It does mean
        # nobody can sign in, which is loud enough on its own.
        log_exception(logger, "accounts.store_unreadable", section="accounts")
        return {"accounts": {}}


def _save_raw(data: Dict) -> None:
    local_store.atomic_write_text(_path(), json.dumps(data, indent=2, sort_keys=True))


def _to_account(record: Dict) -> Optional[Account]:
    try:
        allowed = {f for f in Account.__dataclass_fields__}
        return Account(**{k: v for k, v in record.items() if k in allowed})
    except Exception:
        return None


def all_accounts() -> List[Account]:
    out = []
    for record in _load_raw().get("accounts", {}).values():
        account = _to_account(record)
        if account:
            out.append(account)
    return out


def account_count() -> int:
    return len(_load_raw().get("accounts", {}))


def get_by_id(user_id: str) -> Optional[Account]:
    record = _load_raw().get("accounts", {}).get(user_id or "")
    return _to_account(record) if record else None


def get_by_email(email: str) -> Optional[Account]:
    wanted = normalise_email(email)
    if not wanted:
        return None
    for record in _load_raw().get("accounts", {}).values():
        if normalise_email(record.get("email", "")) == wanted:
            return _to_account(record)
    return None


def email_taken(email: str) -> bool:
    return get_by_email(email) is not None


def _persist(account: Account) -> None:
    with _write_lock:
        data = _load_raw()
        data.setdefault("accounts", {})[account.user_id] = asdict(account)
        _save_raw(data)


# --- creation -----------------------------------------------------------------

def create_account(email: str, password: str,
                   name: str = "") -> Tuple[Optional[Account], Optional[str]]:
    """Create an account. Returns (account, error); exactly one is None."""
    email = normalise_email(email)
    name = (name or "").strip()

    if not looks_like_email(email):
        return None, "That doesn't look like an email address."

    problems = passwords.strength_problems(password, email=email, name=name)
    if problems:
        return None, " ".join(problems)

    with _write_lock:
        if get_by_email(email) is not None:
            # Signup DOES reveal that an address is taken, unlike sign-in.
            # There is no way around it — the alternative is silently not
            # creating the account and telling the person it worked, which
            # locks them out of an account they think they have.
            return None, "An account with that email already exists. Sign in instead."

        account = Account(
            user_id=passwords.new_token(8),
            email=email,
            name=name,
            password_hash=passwords.hash_password(password),
            created_at=_iso(_now()),
            password_changed_at=_iso(_now()),
        )
        _persist(account)

    log_event(logger, logging.INFO, "accounts.created", user_id=account.user_id)
    return account, None


# --- authentication -----------------------------------------------------------

def _lock_duration_minutes(level: int) -> int:
    index = min(max(level, 1), len(_LOCK_STEPS_MINUTES)) - 1
    return _LOCK_STEPS_MINUTES[index]


def _register_failure(account: Account) -> Account:
    now = _now()
    first = _parse_iso(account.first_failure_at)
    if first and (now - first) > datetime.timedelta(minutes=ATTEMPT_WINDOW_MINUTES):
        account.failed_attempts = 0
        first = None
    if not first:
        account.first_failure_at = _iso(now)
    account.failed_attempts += 1

    if account.failed_attempts >= MAX_ATTEMPTS_BEFORE_LOCK:
        account.lock_level += 1
        minutes = _lock_duration_minutes(account.lock_level)
        account.locked_until = _iso(now + datetime.timedelta(minutes=minutes))
        account.failed_attempts = 0
        account.first_failure_at = None
        log_event(logger, logging.WARNING, "accounts.locked",
                  user_id=account.user_id, minutes=minutes, level=account.lock_level)
    _persist(account)
    return account


def _lockout_message(account: Account) -> str:
    minutes = max(1, round(account.lock_remaining_seconds() / 60))
    return (
        f"Too many failed attempts. Try again in about {minutes} "
        f"minute{'s' if minutes != 1 else ''}."
    )


def authenticate(email: str, password: str) -> Tuple[Optional[Account], Optional[str]]:
    """Verify credentials. Returns (account, error); exactly one is None.

    The failure message is deliberately identical whether the address is
    unknown or the password is wrong, and the unknown-address path burns
    the same KDF time via verify_dummy(). Together those stop the form
    answering "does this person bank here?" — which is worth protecting
    even though signup necessarily reveals the same fact, because signup
    requires a deliberate action and sign-in can be scripted at volume.
    """
    generic = "That email or password isn't right."
    email = normalise_email(email)

    account = get_by_email(email) if email else None
    if account is None:
        passwords.verify_dummy()
        return None, generic

    if account.locked:
        return None, _lockout_message(account)

    if not passwords.verify_password(password, account.password_hash):
        account = _register_failure(account)
        if account.locked:
            # Say it on the attempt that caused it. Reporting "wrong
            # password" and only revealing the lock on the NEXT try reads
            # as the password having changed under them.
            return None, _lockout_message(account)
        return None, generic

    # Success: clear the counters, and quietly upgrade the stored hash if
    # the work factor has been raised since it was written.
    account.failed_attempts = 0
    account.first_failure_at = None
    account.locked_until = None
    account.lock_level = 0
    account.last_login_at = _iso(_now())
    if passwords.needs_rehash(account.password_hash):
        account.password_hash = passwords.hash_password(password)
        log_event(logger, logging.INFO, "accounts.rehashed", user_id=account.user_id)
    _persist(account)

    log_event(logger, logging.INFO, "accounts.signed_in", user_id=account.user_id)
    return account, None


# --- password change and reset -------------------------------------------------

def change_password(user_id: str, current_password: str,
                    new_password: str) -> Tuple[bool, Optional[str]]:
    account = get_by_id(user_id)
    if account is None:
        return False, "No such account."
    if not passwords.verify_password(current_password, account.password_hash):
        return False, "Your current password isn't right."

    problems = passwords.strength_problems(new_password, email=account.email,
                                           name=account.name)
    if problems:
        return False, " ".join(problems)

    account.password_hash = passwords.hash_password(new_password)
    account.password_changed_at = _iso(_now())
    # Any outstanding reset link stops working the moment the password
    # changes — otherwise a link mailed an hour ago still overrides the
    # password its owner just deliberately set.
    account.reset_token_hash = ""
    account.reset_expires_at = None
    _persist(account)
    log_event(logger, logging.INFO, "accounts.password_changed", user_id=account.user_id)
    return True, None


def begin_reset(email: str) -> Optional[Tuple[Account, str]]:
    """Issue a reset token. Returns (account, plaintext token), or None
    when no such account exists.

    The CALLER must not tell the user which happened — see
    reset_requested_message(). The plaintext token is returned once and
    never stored; only its digest is written.
    """
    account = get_by_email(email)
    if account is None:
        return None
    token = passwords.new_token(32)
    account.reset_token_hash = passwords.hash_token(token)
    account.reset_expires_at = _iso(_now() + datetime.timedelta(minutes=RESET_TOKEN_TTL_MINUTES))
    _persist(account)
    log_event(logger, logging.INFO, "accounts.reset_requested", user_id=account.user_id)
    return account, token


def reset_requested_message(email: str = "") -> str:
    """The one message shown whether or not the address has an account."""
    return (
        "If that address has a Quantix account, a reset link is on its way. "
        f"The link works once and expires in {RESET_TOKEN_TTL_MINUTES} minutes."
    )


def complete_reset(email: str, token: str,
                   new_password: str) -> Tuple[bool, Optional[str]]:
    account = get_by_email(email)
    generic = "That reset link is invalid or has expired. Request a new one."
    if account is None or not account.reset_token_hash:
        return False, generic

    expires = _parse_iso(account.reset_expires_at)
    if not expires or expires <= _now():
        return False, generic
    if not passwords.tokens_match(token, account.reset_token_hash):
        return False, generic

    problems = passwords.strength_problems(new_password, email=account.email,
                                           name=account.name)
    if problems:
        # Not consumed: the token is still valid, so the person can simply
        # pick a better password instead of requesting a whole new link.
        return False, " ".join(problems)

    account.password_hash = passwords.hash_password(new_password)
    account.password_changed_at = _iso(_now())
    account.reset_token_hash = ""          # single use
    account.reset_expires_at = None
    account.failed_attempts = 0
    account.first_failure_at = None
    account.locked_until = None            # a reset also clears a lockout
    account.lock_level = 0
    _persist(account)
    log_event(logger, logging.INFO, "accounts.reset_completed", user_id=account.user_id)
    return True, None
