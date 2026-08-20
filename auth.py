"""Single-click sign-in (Google / Microsoft / any OIDC provider) and the
per-user data scoping it unlocks.

WHY THIS MODULE IS THIN. Streamlit 1.58 ships native OpenID Connect auth —
`st.login(provider)`, `st.logout()`, `st.user` — with the token exchange,
the signed session cookie and the /oauth2callback route all handled by the
server. So this module deliberately writes no authentication code of its
own. It configures, inspects and degrades; the actual protocol is
Streamlit's. Hand-rolled auth is where security bugs live, and there is no
reason to hand-roll any here.

GITHUB IS NOT DIRECTLY SUPPORTED, AND THAT IS A PROVIDER LIMITATION, NOT A
SHORTCUT TAKEN HERE. Streamlit's auth requires a `server_metadata_url` —
an OIDC discovery document (streamlit/auth_util.py lists it among the
required keys). Google and Microsoft publish one. GitHub does not: it
implements plain OAuth 2.0 for user login and returns no id_token, and
`https://github.com/.well-known/openid-configuration` is a 404. (GitHub
*does* run an OIDC issuer at token.actions.githubusercontent.com, but that
is Actions workload-identity federation — it mints tokens for CI jobs, not
for humans signing in, and cannot be used for this.)

The way to get GitHub is therefore an OIDC broker — Auth0, Okta, Keycloak,
Entra External ID — with GitHub configured as an upstream connection. The
broker publishes discovery, so Streamlit sees an ordinary OIDC provider and
the button works. That is why NOTHING here hardcodes a provider list:
providers are discovered from secrets, so pointing an [auth.github] table
at a broker makes a GitHub button appear with no code change, and the same
is true of any other provider.

EVERYTHING DEGRADES. Auth is optional. With no [auth] section — the state
of a fresh checkout, and of this instance until credentials are added — the
app behaves exactly as it did before sign-in existed: the account panel
explains what is missing, and every store reads and writes the same shared
files it always has. Nothing here can make the app unusable by being
unconfigured, which matters because the credentials can only come from the
person running it.
"""
import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

import local_store
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("auth")

# Keys Streamlit requires in a provider table before st.login() will work.
REQUIRED_PROVIDER_KEYS = ("client_id", "client_secret", "server_metadata_url")

# Keys that live on [auth] itself rather than on a provider sub-table.
_TOP_LEVEL_KEYS = ("redirect_uri", "cookie_secret", *REQUIRED_PROVIDER_KEYS)

# Display names for providers we can guess. Anything not listed is
# title-cased, so a broker table named [auth.auth0] shows as "Auth0" — the
# list is a nicety, never a gate on which providers work.
PROVIDER_LABELS: Dict[str, str] = {
    "google": "Google",
    "microsoft": "Microsoft",
    "entra": "Microsoft Entra",
    "github": "GitHub",
    "okta": "Okta",
    "auth0": "Auth0",
    "keycloak": "Keycloak",
}

# The stores that become per-user when someone signs in.
#
# Chosen, not blanket-applied. These are all "this is *my* setup" state:
# my watchlists, my starred tickers, my theme, my thresholds, my alert
# rules, my saved scenarios, whether I've seen the tour.
PER_USER_STORES: Tuple[str, ...] = (
    "watchlist_store.json",
    "favorites_store.json",
    "theme_state.json",
    "onboarding_state.json",
    "threshold_overrides.json",
    "risk_alert_rules_store.json",
    "alert_rules_store.json",
    "scenario_store.json",
)

# Deliberately NOT per-user, and worth stating so nobody "fixes" it later:
#
#   collaboration_store.json  Team notes are shared BY DESIGN. The whole
#                             point of a thread on AAPL is that teammates
#                             see each other's notes; scoping them per-user
#                             would silently turn the collaboration feature
#                             into eight private diaries.
#   ml_momentum_model.joblib  A trained model and its training history are
#   ml_training_history.json  shared infrastructure, not personal state.
#                             Retraining per user would multiply the cost
#                             for no benefit.
#   api_keys_store.json       Robot credentials. The API server is a
#                             separate process with no Streamlit session,
#                             so current_user() is None inside it — a key
#                             filed under a user's namespace would be
#                             invisible to the process that must verify
#                             it. Each record carries owner_key instead,
#                             which is what still scopes owner-specific
#                             endpoints correctly. See api_keys.py.
SHARED_STORES: Tuple[str, ...] = (
    "collaboration_store.json",
    "ml_momentum_model.joblib",
    "ml_training_history.json",
    "api_keys_store.json",
    # Digest recipients and schedule. Same reason as api_keys: digest.py
    # runs under cron with no Streamlit session, so settings namespaced
    # per user would be unreadable by the process that sends them. Each
    # record carries owner_key, which also lets one scheduled run serve
    # every configured user.
    "digest_store.json",
)


@dataclass(frozen=True)
class AuthUser:
    """An authenticated identity, reduced to what this app actually uses."""
    key: str
    subject: str = ""
    issuer: str = ""
    email: str = ""
    name: str = ""

    @property
    def display_name(self) -> str:
        """Never empty — an identity with no name or email claim still has
        to render as something in the sidebar."""
        return self.name or self.email or "Signed in"


# --- configuration ------------------------------------------------------------

def is_authlib_installed() -> bool:
    """Streamlit's auth needs Authlib; it isn't a hard dependency of
    Streamlit itself, so it can be genuinely absent."""
    try:
        import authlib  # noqa: F401
        return True
    except Exception:
        return False


def _auth_section() -> dict:
    """The [auth] table, or {} if there's no secrets.toml at all.

    st.secrets raises when no secrets file exists anywhere, which is the
    normal state of a fresh checkout — caught here so "unconfigured" is a
    quiet fact rather than an exception every caller has to handle.
    """
    try:
        section = st.secrets.get("auth", {})
    except Exception:
        return {}
    try:
        return dict(section) if section else {}
    except Exception:
        return {}


def _is_complete(table) -> bool:
    try:
        return all(str(table.get(k, "")).strip() for k in REQUIRED_PROVIDER_KEYS)
    except Exception:
        return False


def configured_providers() -> Tuple[str, ...]:
    """Provider names that are fully configured, in secrets order.

    Discovered rather than hardcoded (see the module docstring): any
    sub-table of [auth] carrying client_id, client_secret and
    server_metadata_url is a provider this app will offer. A provider
    missing one of those is skipped rather than shown as a button that
    would fail on click.
    """
    section = _auth_section()
    if not section:
        return ()
    names: List[str] = []
    for name, value in section.items():
        if name in _TOP_LEVEL_KEYS or not hasattr(value, "get"):
            continue
        if _is_complete(value):
            names.append(str(name))
    # An unnamed default provider: credentials sit directly on [auth].
    if not names and _is_complete(section):
        names.append("")
    return tuple(names)


def provider_label(name: str) -> str:
    if not name:
        return "Sign in"
    return PROVIDER_LABELS.get(name.lower(), name.replace("_", " ").title())


def unavailable_reason() -> Optional[str]:
    """Why sign-in can't be offered, phrased for the person who has to fix
    it. None means it's available.

    Ordered by what has to be dealt with first, so following the messages
    in sequence actually gets you there.
    """
    if not is_authlib_installed():
        return (
            "Sign-in needs the Authlib package, which isn't installed. "
            "Run `pip install -r requirements.txt` and restart Streamlit."
        )
    section = _auth_section()
    if not section:
        return (
            "Sign-in isn't configured for this Quantix instance. Add an [auth] section to "
            ".streamlit/secrets.toml — see .streamlit/secrets.toml.example for the exact keys "
            "and where to register the OAuth app."
        )
    missing = [k for k in ("redirect_uri", "cookie_secret") if not str(section.get(k, "")).strip()]
    if missing:
        return f"The [auth] section is missing {' and '.join(missing)}. See .streamlit/secrets.toml.example."
    if not configured_providers():
        return (
            "No identity provider is fully configured. Each provider needs client_id, client_secret "
            "and server_metadata_url — see .streamlit/secrets.toml.example."
        )
    return None


def is_available() -> bool:
    return unavailable_reason() is None


# --- current identity ---------------------------------------------------------

def _safe_claim(name: str) -> str:
    try:
        value = getattr(st.user, name, None)
    except Exception:
        return ""
    return "" if value is None else str(value)


def is_logged_in() -> bool:
    """Never raises — st.user access outside a script run context, or with
    auth unconfigured, must not crash a store trying to resolve its path."""
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def key_for(issuer: str, subject: str, email: str = "") -> str:
    """A stable, filesystem-safe namespace key for one identity.

    Built from issuer + subject, the OIDC pair that is guaranteed unique
    and stable for a user. Deliberately NOT built from email: email is
    mutable at most providers, so keying on it would silently orphan
    someone's entire saved setup the day they change their address.

    Hashed rather than used raw so the directory name can't contain path
    separators or provider-specific punctuation, and so a listing of
    users/ doesn't spell out identities in plain text. The readable
    prefix keeps the directories debuggable.
    """
    basis = f"{issuer}|{subject}".strip("|")
    if not basis:
        basis = (email or "").strip().lower()
    if not basis:
        return ""
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "", _issuer_slug(issuer) or "user")[:12] or "user"
    return f"{slug}-{digest}"


def _issuer_slug(issuer: str) -> str:
    """A short readable hint from an issuer URL: accounts.google.com ->
    'google', login.microsoftonline.com -> 'microsoftonline'."""
    host = re.sub(r"^https?://", "", (issuer or "").strip()).split("/")[0]
    parts = [p for p in host.split(".") if p not in ("com", "org", "net", "www", "io")]
    return (parts[-1] if parts else "").lower()


def current_user() -> Optional[AuthUser]:
    """The signed-in identity, or None. Never raises."""
    if not is_logged_in():
        return None
    subject = _safe_claim("sub")
    issuer = _safe_claim("iss")
    email = _safe_claim("email")
    key = key_for(issuer, subject, email)
    if not key:
        # Authenticated but with no usable identity claim. Refusing to
        # invent a namespace is the safe move: a guessed key could collide
        # with another user's data.
        log_event(logger, logging.WARNING, "auth.no_identity_claim")
        return None
    return AuthUser(key=key, subject=subject, issuer=issuer, email=email, name=_safe_claim("name"))


def user_key() -> Optional[str]:
    """The namespace key for the current user, or None when signed out.

    This is the function handed to local_store — it is called on every
    store read and write, so it must be cheap and must never raise.
    """
    user = current_user()
    return user.key if user else None


# Wire the store layer up to authentication. Importing auth is what turns
# per-user scoping on; a test or script that imports a store module alone
# keeps the shared-file behaviour.
local_store.set_namespace_provider(user_key)


# --- moving existing data into a namespace ------------------------------------

def user_dir(key: str) -> Path:
    return local_store.app_dir() / "users" / key


def has_user_data(key: str) -> bool:
    return any((user_dir(key) / name).exists() for name in PER_USER_STORES)


def shared_data_files() -> Tuple[str, ...]:
    """Which per-user stores exist in the shared/anonymous profile."""
    base = local_store.app_dir()
    return tuple(name for name in PER_USER_STORES if (base / name).exists())


def adopt_shared_data(key: str, overwrite: bool = False) -> Tuple[Tuple[str, ...], List[str]]:
    """Copy the shared profile's stores into `key`'s namespace.

    This exists because of what the first sign-in would otherwise feel
    like: you've built up watchlists, favourites, custom thresholds and
    alert rules on this instance, you sign in, and every one of them
    appears to have vanished. They haven't — they're still the signed-out
    profile's files — but "appears to have vanished" is indistinguishable
    from data loss to the person looking at it.

    COPIES, never moves, and by default never overwrites. The signed-out
    profile is left completely intact, so this is reversible by signing
    out and is safe to run when unsure.

    Returns (copied_filenames, errors); one file failing never stops the
    rest.
    """
    base = local_store.app_dir()
    target = user_dir(key)
    copied: List[str] = []
    errors: List[str] = []
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return (), [f"Couldn't create the profile directory: {e}"]

    for name in PER_USER_STORES:
        source = base / name
        if not source.exists():
            continue
        destination = target / name
        if destination.exists() and not overwrite:
            continue
        try:
            shutil.copy2(source, destination)
            copied.append(name)
        except Exception as e:
            log_exception(logger, "auth.adopt_failed", section="auth")
            errors.append(f"{name}: {type(e).__name__}: {e}")

    if copied:
        log_event(logger, logging.INFO, "auth.adopted_shared_data", files=len(copied))
    return tuple(copied), errors
