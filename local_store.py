"""Shared atomic-write helper for every local JSON store in this app
(onboarding, theme, watchlists, scenarios, ML training history,
real-time alert rules, risk-alert rules, ...).

BUG THIS FIXES: every one of those modules used to write its own
temp-file-then-rename with a FIXED temp filename (`path.with_suffix(".tmp")`).
That's fine for a single writer, but Streamlit doesn't guarantee only one
writer at a time — two browser tabs/sessions open against the same
locally-run instance, or two reruns of the same session overlapping (a
new run starting before Streamlit has fully torn down the previous one),
can both be mid-write to the *same* store at once. Because the temp
filename was fixed, not unique per write, one writer's `tmp.replace(path)`
could consume the temp file the instant before the other writer's own
`tmp.replace(path)` ran — which then raised FileNotFoundError trying to
rename a temp file that had already been renamed away by the other
writer. Exactly this crash was observed live in production (Smart
Risk-Aware Alerts' save_rules(), but every module listed above shared the
identical vulnerable pattern).

FIX: give every write its own unique temp filename (tempfile.mkstemp,
same directory as the target so the final os.replace stays on one
filesystem and stays atomic) instead of a name shared by every writer.
Two concurrent writers now each get their own temp file and never
collide; whichever replace() runs last simply wins, the same "last
write wins" behavior a fixed-name temp file was already going for, just
without the crash in between.
"""
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to path, atomically and safely under concurrent
    writers (see module docstring). On any failure, the orphaned temp
    file is cleaned up and the exception re-raised — callers already
    treat a failed save as something to log and move on from, not
    something to crash the page over."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --- per-user namespacing -----------------------------------------------------
#
# Quantix stores everything in flat JSON files next to the source. That was
# built for "one shared store for whoever runs this instance", which is what
# the app was before it had sign-in. Now that a user can authenticate, their
# data has to be scoped to them.
#
# The scheme is deliberately additive rather than a migration:
#
#   signed out -> QUANTIX/watchlist_store.json        (exactly as before)
#   signed in  -> QUANTIX/users/<key>/watchlist_store.json
#
# Nothing that exists today moves or is rewritten, so an instance that never
# turns on auth keeps behaving byte-for-byte identically and can't be broken
# by this. The signed-out files stay live as the anonymous profile.
#
# The namespace is supplied by an injected callable rather than imported
# from auth.py directly. That keeps this module free of a streamlit import
# (it's the lowest-level thing here and is exercised by tests that have no
# session), and it matches the injectable-dependency pattern already used
# for the email sender and the symbol searcher.

from typing import Callable, Optional

_namespace_provider: Optional[Callable[[], Optional[str]]] = None


def set_namespace_provider(provider: Optional[Callable[[], Optional[str]]]) -> None:
    """Register the callable that names the current user's namespace.

    It must return a filesystem-safe key, or None to mean "not signed in,
    use the shared files". auth.py registers itself on import.
    """
    global _namespace_provider
    _namespace_provider = provider


def current_namespace() -> Optional[str]:
    """The active namespace key, or None for the shared/anonymous profile.

    Never raises. A provider that blows up (no script run context, secrets
    missing, a half-configured OIDC section) degrades to the shared profile
    rather than taking down whichever store happened to be loading — the
    same never-raises contract every loader in this app already keeps.
    """
    if _namespace_provider is None:
        return None
    try:
        key = _namespace_provider()
    except Exception:
        return None
    return key or None


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def store_path(filename: str, namespace: Optional[str] = None) -> Path:
    """Where `filename` lives for the current user.

    Creates the per-user directory on demand, because atomic_write_text
    needs the parent to exist to place its temp file. Falls back to the
    shared path if the directory can't be created, so a permissions
    problem degrades to "shared data" instead of an unusable app.
    """
    base = app_dir()
    key = namespace if namespace is not None else current_namespace()
    if not key:
        return base / filename
    user_dir = base / "users" / key
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return base / filename
    return user_dir / filename
