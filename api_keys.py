"""Scoped API keys for programmatic (robot) access to Quantix.

WHAT A KEY IS. `qtx_<id>_<secret>`. The id is public — it's shown in the
UI so a key can be identified and revoked without anyone ever seeing the
secret again. The secret is 32 bytes of `secrets.token_urlsafe`
randomness. Only `sha256(whole key)` is persisted.

THE SECRET IS SHOWN ONCE AND IS NOT RECOVERABLE. This is the single most
important property here and it is deliberately inconvenient: a store that
cannot reveal a key cannot leak one, and "let me just look up my key
again" is how key systems end up keeping plaintext around. Lose it and
you revoke and reissue, which takes two clicks.

THE STORE IS SHARED, NOT NAMESPACED PER USER — and that is not an
oversight. Every other piece of personal state in this app is scoped to
the signed-in user via local_store.store_path(). Keys cannot be, because
the API server is a SEPARATE PROCESS with no Streamlit session: inside
it, auth.current_user() is None. A key filed under a user's namespace
would be invisible to the exact process whose only job is to verify it.
So the store is shared and each record carries `owner_key`, the auth
namespace of whoever created it. That field is what lets an owner-scoped
endpoint resolve the right user's watchlists without the server needing
a session of its own.

READ-ONLY BY DESIGN. The originating task mentions "headless trading
bots". Quantix has no brokerage integration, and this key system grants
no write or trade capability whatsoever — every scope below is a read.
That is stated in the scope names, in the config, and in the API's own
discovery endpoint, so nobody wires a key up expecting it to place an
order.

Persisted with the same atomic-write, gitignored-local-file pattern every
other store here uses (see local_store.py).
"""
import datetime
import hashlib
import hmac
import json
import logging
import re
import secrets as secrets_module
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import API_KEYS
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("api_keys")

# The scopes a key can carry. Deliberately small and deliberately all
# reads — see the module docstring. Each maps to a group of endpoints in
# api_server.py; a key presented to an endpoint outside its scopes is
# rejected with 403 rather than silently allowed.
SCOPES: Dict[str, str] = {
    "quote:read": "Current price, day change, company name and sector.",
    "fundamentals:read": "Scorecard, Blueprint Alignment, margins, leverage and valuation metrics.",
    "risk:read": "Volatility, Sharpe, Sortino, VaR, CVaR and maximum drawdown.",
    "watchlist:read": "The owning account's saved watchlists and favourites.",
}

DEFAULT_SCOPES: Tuple[str, ...] = ("quote:read", "fundamentals:read")

_KEY_RE = re.compile(r"^([a-z]+)_([A-Za-z0-9]+)_([A-Za-z0-9_\-]+)$")


@dataclass(frozen=True)
class ApiKey:
    """One issued key. Never holds the secret — only its hash."""
    id: str
    name: str
    hashed: str
    scopes: Tuple[str, ...]
    owner_key: str = ""          # auth namespace of the creator; "" = signed-out profile
    created_at: str = ""
    expires_at: str = ""         # "" means no expiry
    last_used_at: str = ""
    revoked_at: str = ""

    @property
    def revoked(self) -> bool:
        return bool(self.revoked_at)

    def is_expired(self, now: Optional[datetime.datetime] = None) -> bool:
        if not self.expires_at:
            return False
        now = now or datetime.datetime.now()
        try:
            return datetime.datetime.fromisoformat(self.expires_at) <= now
        except ValueError:
            # An unparseable expiry is treated as EXPIRED, not as "no
            # expiry". Failing closed is the only safe reading of a
            # corrupt credential record.
            return True

    def is_usable(self, now: Optional[datetime.datetime] = None) -> bool:
        return not self.revoked and not self.is_expired(now)

    @property
    def status(self) -> str:
        if self.revoked:
            return "revoked"
        if self.is_expired():
            return "expired"
        return "active"

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass(frozen=True)
class ApiKeyStore:
    keys: Tuple[ApiKey, ...] = ()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _store_path() -> Path:
    # shared_path, not store_path — see the module docstring.
    return shared_path(API_KEYS.store_filename)


def _usage_path() -> Path:
    """Last-used timestamps, kept SEPARATE from key definitions.

    This split fixes a confirmed revocation-loss race, not a stylistic
    concern. The API server records usage from a background thread while
    the Streamlit UI can revoke a key at any moment. When both lived in
    one file, the usage write did a read-modify-write of the WHOLE store:
    read the keys, get descheduled, then write its stale copy back over
    the revocation that landed in between. Reproduced deterministically —
    the revoked key verified again afterwards.

    Separating them removes the shared mutable state rather than trying
    to coordinate access to it. Two usage writes can still race each
    other, and that is fine: the worst case is one lost timestamp, which
    costs nothing. A revocation can no longer be the casualty.
    """
    return shared_path(API_KEYS.usage_filename)


def load_usage(path: Optional[Path] = None) -> Dict[str, str]:
    """{key_id: last_used_iso}. Never raises — a corrupt or missing usage
    file means timestamps are unknown, which is cosmetic."""
    path = path or _usage_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def save_usage(usage: Dict[str, str], path: Optional[Path] = None) -> None:
    atomic_write_text(path or _usage_path(), json.dumps(usage, indent=2))


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None,
               usage_path: Optional[Path] = None) -> ApiKeyStore:
    """Never raises. A missing file is an empty store; a corrupt one
    degrades to empty rather than taking down the app or the API server.
    Individual malformed records are dropped — but note that dropping a
    key record REVOKES that key in practice, which is the safe direction
    for a credential."""
    path = path or _store_path()
    if not path.exists():
        return ApiKeyStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "api_keys.store_corrupt", section="api_keys")
        return ApiKeyStore()
    if not isinstance(raw, dict):
        return ApiKeyStore()

    # Usage lives in its own file; the value recorded on the key itself
    # is only a fallback for stores written before the split.
    usage = load_usage(usage_path)

    keys: List[ApiKey] = []
    for item in raw.get("keys", []):
        if not isinstance(item, dict):
            continue
        key_id = str(item.get("id", "")).strip()
        hashed = str(item.get("hashed", "")).strip()
        if not key_id or not hashed:
            continue
        scopes = tuple(str(s) for s in (item.get("scopes") or []) if str(s) in SCOPES)
        keys.append(ApiKey(
            id=key_id,
            name=str(item.get("name") or "Unnamed key"),
            hashed=hashed,
            scopes=scopes,
            owner_key=str(item.get("owner_key") or ""),
            created_at=str(item.get("created_at") or ""),
            expires_at=str(item.get("expires_at") or ""),
            last_used_at=usage.get(key_id) or str(item.get("last_used_at") or ""),
            revoked_at=str(item.get("revoked_at") or ""),
        ))
    return ApiKeyStore(keys=tuple(keys))


def save_store(store: ApiKeyStore, path: Optional[Path] = None) -> None:
    path = path or _store_path()
    payload = {"keys": [{
        "id": k.id, "name": k.name, "hashed": k.hashed, "scopes": list(k.scopes),
        "owner_key": k.owner_key, "created_at": k.created_at, "expires_at": k.expires_at,
        "last_used_at": k.last_used_at, "revoked_at": k.revoked_at,
    } for k in store.keys]}
    atomic_write_text(path, json.dumps(payload, indent=2))


# --- issuing ------------------------------------------------------------------

def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _new_id(existing: Tuple[ApiKey, ...]) -> str:
    taken = {k.id for k in existing}
    while True:
        candidate = secrets_module.token_hex(API_KEYS.id_length // 2)
        if candidate not in taken:
            return candidate


def create_key(
    store: ApiKeyStore,
    name: str,
    scopes: Tuple[str, ...] = DEFAULT_SCOPES,
    owner_key: str = "",
    expires_in_days: Optional[int] = None,
) -> Tuple[ApiKeyStore, Optional[ApiKey], Optional[str], Optional[str]]:
    """Issue a key. Returns (store, key, plaintext, error).

    `plaintext` is the ONLY time the full key exists anywhere — it is not
    stored and cannot be recovered. The caller must show it once and then
    let it go.
    """
    name = (name or "").strip()
    if not name:
        return store, None, None, "Give the key a name so you can tell it apart later."
    if len(name) > API_KEYS.max_name_chars:
        return store, None, None, f"Names are capped at {API_KEYS.max_name_chars} characters."

    scopes = tuple(s for s in scopes if s in SCOPES)
    if not scopes:
        return store, None, None, "Select at least one scope — a key with no scopes can do nothing."

    owned = [k for k in store.keys if k.owner_key == owner_key and not k.revoked]
    if len(owned) >= API_KEYS.max_keys_per_owner:
        return store, None, None, (
            f"You already have {API_KEYS.max_keys_per_owner} active keys. "
            f"Revoke one before creating another."
        )

    if expires_in_days is None:
        expires_in_days = API_KEYS.default_expiry_days
    try:
        expires_in_days = int(expires_in_days)
    except (TypeError, ValueError):
        return store, None, None, "Expiry must be a whole number of days."
    if expires_in_days < 0 or expires_in_days > API_KEYS.max_expiry_days:
        return store, None, None, f"Expiry must be between 0 and {API_KEYS.max_expiry_days} days (0 = never)."

    key_id = _new_id(store.keys)
    secret = secrets_module.token_urlsafe(API_KEYS.secret_bytes)
    plaintext = f"{API_KEYS.key_prefix}_{key_id}_{secret}"

    expires_at = ""
    if expires_in_days > 0:
        expires_at = (datetime.datetime.now() + datetime.timedelta(days=expires_in_days)).isoformat(timespec="seconds")

    key = ApiKey(
        id=key_id,
        name=name,
        hashed=hash_key(plaintext),
        scopes=scopes,
        owner_key=owner_key,
        created_at=_now_iso(),
        expires_at=expires_at,
    )
    log_event(logger, logging.INFO, "api_keys.created", key_id=key_id, scopes=len(scopes))
    return replace(store, keys=store.keys + (key,)), key, plaintext, None


def revoke_key(store: ApiKeyStore, key_id: str) -> ApiKeyStore:
    """Revoke rather than delete, so a revoked key's id and name stay
    visible in the UI. Silently deleting the record would make an
    unexplained 401 in a robot's logs much harder to diagnose."""
    keys = tuple(
        replace(k, revoked_at=_now_iso()) if (k.id == key_id and not k.revoked) else k
        for k in store.keys
    )
    log_event(logger, logging.INFO, "api_keys.revoked", key_id=key_id)
    return replace(store, keys=keys)


def delete_key(store: ApiKeyStore, key_id: str) -> ApiKeyStore:
    return replace(store, keys=tuple(k for k in store.keys if k.id != key_id))


def keys_for_owner(store: ApiKeyStore, owner_key: str = "") -> Tuple[ApiKey, ...]:
    return tuple(k for k in store.keys if k.owner_key == owner_key)


# --- verifying ----------------------------------------------------------------

def parse_key(plaintext: str) -> Optional[str]:
    """The key id out of a presented key, or None if it isn't even
    shaped like one of ours. Cheap structural rejection before any store
    lookup."""
    match = _KEY_RE.match((plaintext or "").strip())
    if not match:
        return None
    prefix, key_id, _secret = match.groups()
    if prefix != API_KEYS.key_prefix:
        return None
    return key_id


def verify_key(store: ApiKeyStore, plaintext: str,
               now: Optional[datetime.datetime] = None) -> Tuple[Optional[ApiKey], Optional[str]]:
    """Resolve a presented key. Returns (key, error); exactly one is set.

    Compared with hmac.compare_digest rather than `==`. The comparison is
    against sha256 digests of a 256-bit random secret, so a timing oracle
    here is not a realistic attack — but constant-time comparison of
    credential material is the correct habit, and the cost is nil.

    The error strings are deliberately uniform about WHY a key failed at
    the transport level (api_server maps them all to 401) so a caller
    cannot probe which ids exist.
    """
    key_id = parse_key(plaintext)
    if key_id is None:
        return None, "Malformed API key."

    presented = hash_key(plaintext.strip())
    for key in store.keys:
        if key.id != key_id:
            continue
        if not hmac.compare_digest(key.hashed, presented):
            return None, "Invalid API key."
        if key.revoked:
            return None, "This API key has been revoked."
        if key.is_expired(now):
            return None, "This API key has expired."
        return key, None
    return None, "Invalid API key."


def mark_used(store: ApiKeyStore, key_id: str) -> ApiKeyStore:
    """Record last use, so an unused or forgotten key is visible as such
    in the UI and can be revoked with confidence."""
    return replace(store, keys=tuple(
        replace(k, last_used_at=_now_iso()) if k.id == key_id else k for k in store.keys
    ))


def touch_last_used(key_id: str, path: Optional[Path] = None) -> None:
    """Record that a key was used, best-effort.

    Writes ONLY the usage file — never the key store. That is what makes
    a concurrent revocation safe: this function no longer reads, holds,
    or rewrites key definitions, so it has nothing stale to clobber them
    with. See _usage_path() for the race this replaces.

    `path` is the USAGE path (tests pass one); the signature keeps its
    name for callers that already pass positionally.
    """
    try:
        path = path or _usage_path()
        usage = load_usage(path)
        usage[key_id] = _now_iso()
        save_usage(usage, path)
    except Exception:
        log_exception(logger, "api_keys.touch_failed", section="api_keys")
