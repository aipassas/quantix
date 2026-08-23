"""Tests for the Quantix logo assets and the brand colour.

Three of the four surfaces that use these files fail SILENTLY when an
asset is missing — a favicon just doesn't appear, an <img> with a dead
src renders as nothing, a slide loses its logo. So the tests assert the
files resolve, that each role gets the RIGHT variant (a white-ground logo
on a dark slide is a white slab), and that a missing file degrades to
None rather than raising.

The brand colour is checked against the artwork itself rather than taken
on trust: the value in the brief has to be the value in the PNG, or the
UI and the logo disagree by a shade nobody notices until print.
"""
import base64

import pytest

import brand_assets
import local_store


def test_every_role_resolves_to_a_file():
    assert brand_assets.missing() == (), brand_assets.missing()
    for getter in (brand_assets.dark_logo, brand_assets.light_logo, brand_assets.mark):
        path = getter()
        assert path is not None and path.is_file(), getter.__name__


def test_the_brief_filenames_are_accepted_and_so_are_the_shipped_ones():
    """The brief named 1_2.png / 2_2.png; the directory ships 1.png /
    2.png. Both spellings must work, or renaming an asset silently drops
    the logo everywhere."""
    assert "1_2.png" in brand_assets._DARK_LOGO_NAMES
    assert "1.png" in brand_assets._DARK_LOGO_NAMES
    assert "2_2.png" in brand_assets._LIGHT_LOGO_NAMES
    assert "2.png" in brand_assets._LIGHT_LOGO_NAMES


def test_the_favicon_image_is_transparent():
    """The favicon sits on browser chrome whose colour we don't control,
    so a white card behind the mark reads as a bug in dark mode.

    Asserts mark_image(), not the file: the shipped 4.png is exported on
    a white ground and the transparent asset that used to sit beside it
    is gone, so the transparency is produced in memory.
    """
    pytest.importorskip("PIL.Image")

    image = brand_assets.mark_image()
    assert image is not None
    rgba = image.convert("RGBA")
    transparent = sum(1 for pixel in rgba.getdata() if pixel[3] == 0)
    assert transparent / (rgba.width * rgba.height) > 0.5, "favicon is not mostly transparent"


def test_keying_the_mark_keeps_the_artwork_intact():
    """The white key must remove the card and nothing else — an aggressive
    threshold would eat the light edges of the glyph."""
    pytest.importorskip("PIL.Image")
    from collections import Counter

    rgba = brand_assets.mark_image().convert("RGBA")
    opaque = [p for p in rgba.getdata() if p[3] > 200]
    assert len(opaque) > 10_000, "the key removed most of the mark, not just the card"
    (red, green, blue), _ = Counter((p[0], p[1], p[2]) for p in opaque).most_common(1)[0]
    assert (red, green, blue) == brand_assets.BRAND_CYAN_RGB


def test_the_dark_and_light_logos_are_not_the_same_file():
    """Picking one for both roles puts a white slab on a dark slide."""
    assert brand_assets.dark_logo() != brand_assets.light_logo()


def test_the_variants_have_the_grounds_their_names_claim():
    Image = pytest.importorskip("PIL.Image")

    def corner(path):
        image = Image.open(path).convert("RGB")
        return sum(image.getpixel((4, 4))) / 3

    assert corner(brand_assets.dark_logo()) < 40, "the dark logo's ground is not dark"
    assert corner(brand_assets.light_logo()) > 200, "the light logo's ground is not light"


def test_the_brand_colour_matches_the_artwork():
    """#00f2fe has to be the colour actually in the PNG."""
    Image = pytest.importorskip("PIL.Image")
    from collections import Counter

    # The dark lockup, whose ground is black — so the most common non-black
    # colour is the brand cyan. Using the mark's own file would just return
    # its white export card.
    image = Image.open(brand_assets.dark_logo()).convert("RGB")
    coloured = [p for p in image.getdata() if sum(p) > 90 and max(p) - min(p) > 40]
    (red, green, blue), _ = Counter(coloured).most_common(1)[0]
    assert (red, green, blue) == brand_assets.BRAND_CYAN_RGB
    assert brand_assets.BRAND_CYAN.lower() == f"#{red:02x}{green:02x}{blue:02x}"


def test_the_brand_colour_drives_every_surface():
    """One value, or the login page, the exports and the tear sheet drift."""
    import branding
    import export_theme
    import login_page

    assert branding.brand().accent_color.lower() == brand_assets.BRAND_CYAN
    assert export_theme.palette().css("accent").lower() == brand_assets.BRAND_CYAN
    assert login_page.accent().lower() == brand_assets.BRAND_CYAN


def test_a_licensee_accent_still_wins():
    """Defaulting the brand colour must not break white-labelling."""
    import branding

    assert branding.Brand(name="X", accent_color="#AA3366").accent_color == "#AA3366"


def test_the_default_variant_is_the_dark_one():
    """Every Quantix surface is dark, so the white-ground file renders as
    a white tile wherever it has been tried. Nothing should reach for it
    by accident."""
    assert brand_assets.data_uri() == brand_assets.data_uri("dark")
    assert brand_assets.data_uri() != brand_assets.data_uri("light")


def test_nothing_in_the_app_asks_for_the_light_logo():
    """A regression guard, not a style rule: the light variant put a white
    tile in the PDF masthead, caught only by rasterising the page."""
    import pathlib as _pathlib

    root = _pathlib.Path(__file__).resolve().parent.parent
    for module in ("finance.py", "export_deck.py"):
        source = (root / module).read_text()
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith("#"))
        assert 'data_uri("light")' not in code, module
        assert "light_logo()" not in code, module


def test_data_uri_is_a_real_decodable_png():
    uri = brand_assets.data_uri("dark")
    assert uri and uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_missing_assets_directory_degrades_to_none(tmp_path, monkeypatch):
    """A decorative asset must never cost someone their analysis."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    assert brand_assets.dark_logo() is None
    assert brand_assets.light_logo() is None
    assert brand_assets.mark() is None
    assert brand_assets.data_uri("light") is None
    assert brand_assets.data_uri() is None
    assert brand_assets.missing()          # and it says which


def test_paths_are_redirectable(tmp_path, monkeypatch):
    """The whole reason this resolves through app_dir rather than
    Path(__file__) — see tests/test_auth.py."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    assert brand_assets.assets_dir() == tmp_path / "Design_Assets"
