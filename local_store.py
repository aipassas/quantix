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
