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
# The horizontal "QUANTIX" wordmark, drawn in black on a white card.
_WORDMARK_NAMES = ("3_2.png", "3.png")

# Above what mean channel value a pixel counts as "the white card the mark
# was exported on" rather than part of the artwork. The mark is saturated
# cyan (#00F2FE) and the ground is #FEFEFE, so anything near-white is
# background — a wide margin, not a delicate threshold.
_WHITE_KEY_CUTOFF = 236

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


def wordmark() -> Optional[Path]:
    """The horizontal "QUANTIX" wordmark, black on a white card."""
    return _first_existing(_WORDMARK_NAMES)


def _keyed_transparent(path: Path, _stamp: float):
    """The mark with its white export card removed, as a PIL image.

    The transparent asset that used to ship (4-removebg-preview.png) is no
    longer in the directory, and a favicon sits on a browser-chrome colour
    we do not control — a white tile there looks like a bug in dark mode.
    Keying the ground out in memory avoids writing a derived file into the
    designer's source folder, where it would be mistaken for an original
    and would drift the next time the artwork is replaced.

    Safe because the artwork is two flat colours: saturated cyan on a
    near-white card. Anything at or above the cutoff becomes transparent.
    """
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            red, green, blue, alpha = pixels[x, y]
            if (red + green + blue) / 3 >= _WHITE_KEY_CUTOFF:
                pixels[x, y] = (red, green, blue, 0)
    return image


@functools.lru_cache(maxsize=None)
def _cached_mark_image(path_text: str, _stamp: float):
    return _keyed_transparent(Path(path_text), _stamp)


def mark_image():
    """The mark for the browser tab, transparent, or None.

    Returns a PIL image rather than a path so nothing derived is written
    to disk. st.set_page_config accepts anything st.image does. If the
    file already has real transparency it is returned untouched; if PIL is
    unavailable the caller falls back to mark().
    """
    path = mark()
    if path is None:
        return None
    try:
        from PIL import Image

        image = Image.open(path)
        if image.mode in ("RGBA", "LA") and image.convert("RGBA").getextrema()[3][0] < 255:
            return image                      # already transparent
        return _cached_mark_image(str(path), path.stat().st_mtime)
    except Exception:
        return None


def _keyed_recoloured(path: Path, _stamp: float, hex_colour: Optional[str]):
    """White card keyed out, and the remaining ink optionally recoloured.

    The wordmark ships as BLACK artwork on white. Keying the card alone
    would leave black letters, invisible on this app's dark surfaces — so
    the ink is repainted. Alpha comes from how dark each pixel was, which
    keeps the letterforms' anti-aliased edges smooth instead of producing
    a jagged one-bit mask.
    """
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    pixels = image.load()
    width, height = image.size
    target = None
    if hex_colour:
        clean = hex_colour.lstrip("#")
        target = tuple(int(clean[i:i + 2], 16) for i in (0, 2, 4))

    for x in range(width):
        for y in range(height):
            red, green, blue, _ = pixels[x, y]
            brightness = (red + green + blue) / 3
            if brightness >= _WHITE_KEY_CUTOFF:
                pixels[x, y] = (red, green, blue, 0)
            elif target is not None:
                alpha = int(max(0, min(255, 255 - brightness)))
                pixels[x, y] = (target[0], target[1], target[2], alpha)

    # Crop to the artwork. Both files are 2000x2000 with the mark or the
    # wordmark occupying a fraction of it, so an uncropped image renders
    # as a small glyph adrift in a large transparent box — impossible to
    # size predictably in CSS, and the reason the first header attempt
    # looked wrong.
    bounds = image.getbbox()
    return image.crop(bounds) if bounds else image


@functools.lru_cache(maxsize=None)
def _cached_recoloured(path_text: str, _stamp: float, hex_colour: Optional[str]):
    return _keyed_recoloured(Path(path_text), _stamp, hex_colour)


def _transformed_data_uri(path: Optional[Path], hex_colour: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    try:
        import io

        image = _cached_recoloured(str(path), path.stat().st_mtime, hex_colour)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def mark_data_uri() -> Optional[str]:
    """The Q mark, transparent, in its own cyan.

    A data URI rather than a path because the login header is raw HTML
    injected through st.markdown, where a filesystem path resolves to
    nothing.
    """
    return _transformed_data_uri(mark(), None)


def wordmark_data_uri(hex_colour: str = "#FFFFFF") -> Optional[str]:
    """The "QUANTIX" wordmark, transparent and recoloured for a dark ground."""
    return _transformed_data_uri(wordmark(), hex_colour)


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
