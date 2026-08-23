"""The Quantix logo files, resolved once and shared by every surface.

WHY A MODULE AND NOT FOUR HARD-CODED PATHS. The mark is needed by the
browser tab (favicon), the sidebar, the PDF tear sheet (as base64, since
WeasyPrint renders a detached document with no access to the filesystem
the app is served from) and the PowerPoint title slide. Four literals
would drift the first time a file is renamed, and three of those four
surfaces fail SILENTLY when an image is missing — a favicon just doesn't
appear, an <img> with a dead src renders as nothing.

EVERY ACCESSOR RETURNS None RATHER THAN RAISING. A missing logo must
degrade to no logo, never to a broken analysis: the tear sheet, the deck
and the page config are all things people reach for at the end of a piece
of work, and losing the work to a decorative asset would be absurd.

THE DARK-GROUND LOGO IS THE ONE THIS APP USES. Every Quantix surface is
dark — the sidebar, the slides, and the tear sheet, which is black on
purpose (see export_theme: exports do not follow the viewer's light/dark
preference, because a document that leaves the building should not change
colour with a personal setting). The white-ground file therefore renders
as a white tile everywhere it has been tried, so dark is the default and
light_logo() is kept only because the asset exists and a future white
document might want it. If you reach for it, check the rendered output
first — that is how the white tile was caught.

The transparent mark is separate again and is the only variant safe on a
surface whose colour we do not control, which is exactly the browser tab.
"""
import base64
import functools
from pathlib import Path
from typing import Optional

import local_store


def assets_dir() -> Path:
    """Where the artwork lives.

    Resolved through local_store.app_dir() rather than Path(__file__),
    for the reason tests/test_auth.py enforces: one resolver for every
    path in the app, so redirecting app_dir() genuinely redirects
    everything. These files are read-only and shipped with the code, so
    nothing here can corrupt user data — but a module that quietly
    resolves its own paths is how that guarantee gets eroded, and the
    exception is not worth the precedent. A sandboxed app_dir simply
    finds no logo, which every caller already handles.

    A licensee's own logo does NOT belong here: branding.py owns that,
    via its logo_path field. This directory is Quantix's own artwork.
    """
    return local_store.app_dir() / "Design_Assets"

# Filenames, most-preferred first. Several are listed per role because the
# brief named files (1_2.png, 2_2.png) that do not exist in the directory
# — the shipped assets are 1.png and 2.png. Rather than pick one and have
# it break if the other naming is restored, each role accepts either.
_DARK_LOGO_NAMES = ("1_2.png", "1.png")
_LIGHT_LOGO_NAMES = ("2_2.png", "2.png")
_MARK_NAMES = ("4-removebg-preview.png", "4.png")

# The brand colour, verified against the artwork rather than taken on
# trust: the mark is exactly #00F2FE in all three files.
BRAND_CYAN = "#00f2fe"
BRAND_CYAN_RGB = (0, 242, 254)


def _first_existing(names) -> Optional[Path]:
    for name in names:
        candidate = assets_dir() / name
        if candidate.is_file():
            return candidate
    return None


def dark_logo() -> Optional[Path]:
    """Full logo on a black ground — for the sidebar and dark UI."""
    return _first_existing(_DARK_LOGO_NAMES)


def light_logo() -> Optional[Path]:
    """Full logo on a white ground — for documents and print."""
    return _first_existing(_LIGHT_LOGO_NAMES)


def mark() -> Optional[Path]:
    """The standalone mark with a transparent background — the only
    variant safe on a surface whose colour isn't known, which is exactly
    the browser tab's situation."""
    return _first_existing(_MARK_NAMES)


@functools.lru_cache(maxsize=None)
def _encode(path: Path, _stamp: float) -> str:
    """Cached on (path, mtime) rather than on nothing: the file is ~57 KB
    and the tear sheet re-renders on every interaction, but a cache keyed
    on the path alone would serve stale bytes after the artwork changed."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def data_uri(which: str = "dark") -> Optional[str]:
    """A `data:image/png;base64,...` URI, or None.

    Needed because the PDF is rendered by WeasyPrint from an HTML string
    with no base URL, so a relative <img src> resolves to nothing. Cached
    because the file is ~57 KB and the tear sheet re-renders on every
    interaction with the page.
    """
    path = {"dark": dark_logo, "light": light_logo, "mark": mark}.get(which, dark_logo)()
    if path is None:
        return None
    try:
        encoded = _encode(path, path.stat().st_mtime)
    except Exception:
        return None
    return f"data:image/png;base64,{encoded}"


def missing() -> tuple:
    """Which roles have no file, for a one-line diagnostic in the UI."""
    absent = []
    if dark_logo() is None:
        absent.append(f"dark logo ({' or '.join(_DARK_LOGO_NAMES)})")
    if light_logo() is None:
        absent.append(f"light logo ({' or '.join(_LIGHT_LOGO_NAMES)})")
    if mark() is None:
        absent.append(f"mark ({' or '.join(_MARK_NAMES)})")
    return tuple(absent)
