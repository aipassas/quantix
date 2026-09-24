"""Sharing an analysis: composed here, posted by the reader.

THE TEST THAT MATTERS MOST is test_nothing_here_can_post. A share feature
that holds a token and publishes on somebody's behalf is a different
feature with a different consent. This one composes text and opens the
platform's own window; the reader presses Post.

The second is test_an_unavailable_figure_never_reaches_a_post. A failed
DCF rendering as "$0.00" is a bug this codebase has shipped twice, on a
metric card and in prose. A public post cannot be footnoted afterwards,
so it is the worst place for it to appear a third time.
"""
import datetime
import sys
import urllib.parse
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import social_share as ss
from config import SOCIAL_SHARE, BRANDING

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
AS_OF = datetime.date(2026, 9, 24)


def _code_only(source: str) -> str:
    """Source with comments and docstrings stripped, so a ban list checks
    what the module DOES rather than what it says about itself."""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def _facts(*pairs):
    return tuple(ss.Fact(label, display) for label, display in pairs)


def _post(**kwargs):
    defaults = dict(
        ticker="MSFT", headline="High Scorecard alignment (100%)",
        facts=_facts(("Net margin", "40.30%"), ("P/E", "27.73")),
        brand_name="Quantix", as_of=AS_OF,
    )
    defaults.update(kwargs)
    return ss.compose(**defaults)


# --- it never posts -----------------------------------------------------------

def test_nothing_here_can_post():
    """No token, no API call, no publish path. The panel opens the
    platform's own compose window and the reader presses Post."""
    # CODE only. The module's own docstring explains that it stores no
    # OAuth token, and a bare substring check matches that explanation
    # rather than any call — the same declaration-not-substring trap the
    # CSS tests already record.
    src = _code_only(Path(ss.__file__).read_text())
    for banned in ("requests.post", "urlopen", "urllib.request", "oauth",
                   "OAuth", "access_token", "bearer", "api_key", "http.client"):
        assert banned not in src, f"{banned} — this module must not publish"
    assert not any(n for n in dir(ss)
                   if n.lower() in ("post_to_x", "publish", "send", "tweet"))


def test_it_only_builds_urls_for_the_user_to_open():
    src = Path(ss.__file__).read_text()
    assert "urllib.parse" in _code_only(src)
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    assert not any("urllib.request" in ln for ln in imports)


# --- the honesty of the text --------------------------------------------------

def test_an_unavailable_figure_never_reaches_a_post():
    """The $0.00 DCF, which shipped twice elsewhere in this app."""
    for marker in ("Not reported", "Unavailable", "N/A", "—", "", "  ", "None"):
        post = _post(facts=_facts(("Margin of safety", marker),
                                  ("Net margin", "40.30%")))
        assert "Margin of safety" not in post.text, marker
        assert "Net margin: 40.30%" in post.text


def test_a_dropped_figure_is_reported_rather_than_silently_omitted():
    """A reader who is not told would think the post is complete."""
    post = _post(facts=_facts(("DCF", "Not reported"), ("P/E", "27.73")))
    assert post.dropped == ("DCF",)
    assert [f.label for f in post.facts] == ["P/E"]


def test_a_zero_is_not_treated_as_missing():
    """0.00% can be a real measured figure. Only the app's explicit
    unavailable markers are dropped, never a number."""
    post = _post(facts=_facts(("Dividend yield", "0.00%")))
    assert "Dividend yield: 0.00%" in post.text


def test_a_fact_carries_a_formatted_STRING_not_a_number():
    """Taking a float would invite this module to format it, and every
    unit error in this app came from a second place formatting the same
    number."""
    import dataclasses
    fields = {f.name: f.type for f in dataclasses.fields(ss.Fact)}
    assert set(fields) == {"label", "display"}
    assert all("str" in str(t) for t in fields.values())


def test_nothing_is_composed_without_a_headline():
    """A post saying only 'MSFT' and a disclaimer puts the app's name
    against an empty claim."""
    assert _post(headline="") is None
    assert _post(headline="   ") is None
    assert _post(ticker="") is None


def test_the_ticker_is_normalised():
    assert _post(ticker="  msft ").text.startswith("MSFT —")


def test_the_as_of_date_is_carried():
    assert "As of 2026-09-24." in _post().text


# --- the disclaimer -----------------------------------------------------------

def test_every_post_carries_the_disclaimer():
    assert ss.DISCLAIMER in _post().text


def test_the_disclaimer_uses_the_phrase_branding_locks():
    """branding.LOCKED_DISCLOSURE_PHRASES exists so a licensee cannot
    strip it, and a public post is where it matters most."""
    import branding
    assert "not investment advice" in ss.DISCLAIMER.lower()
    assert any(p in ss.DISCLAIMER.lower()
               for p in branding.LOCKED_DISCLOSURE_PHRASES)


def test_rebranding_does_not_remove_the_disclaimer():
    import branding
    assert "not investment advice" in branding.rebrand(ss.DISCLAIMER).lower()


def test_the_brand_name_is_carried_when_there_is_one():
    assert _post(brand_name="Acme Capital").text.endswith(
        f"Acme Capital · {ss.DISCLAIMER}")


def test_an_unbranded_instance_still_gets_the_disclaimer():
    assert _post(brand_name="").text.endswith(ss.DISCLAIMER)


# --- trimming -----------------------------------------------------------------

def test_a_long_post_is_trimmed_to_the_platform_limit():
    post = _post(facts=_facts(*[(f"Metric {i}", f"{i}.00%") for i in range(40)]))
    assert post.length <= SOCIAL_SHARE.max_post_chars


def test_trimming_never_removes_the_disclaimer():
    """The obvious implementation cuts the tail, which removes the one
    line that must survive."""
    post = _post(facts=_facts(*[(f"Metric {i}", f"{i}.00%") for i in range(40)]))
    assert post.text.endswith(ss.DISCLAIMER)
    assert ss.DISCLAIMER in post.text


def test_trimming_drops_whole_facts_rather_than_cutting_one_in_half():
    """Half a figure is a wrong figure.

    A FLOOR ON HOW OFTEN THE CHECK FIRED. Without it this passed a build
    that truncated every fact line to eight characters, because the
    overflow then wiped the middle entirely and no line began with
    "Metric" at all — a conditional assertion that never ran. Same rule
    the ETF suite records for bid/ask.
    """
    supplied = 20
    post = _post(facts=_facts(*[(f"Metric {i}", f"{i}.00%")
                                for i in range(supplied)]))
    kept = [ln for ln in post.text.split("\n") if ln.startswith("Metric")]
    # BOTH bounds. Too few facts and the post fits, so the trimmer never
    # runs and the check is vacuous; too many and everything is wiped,
    # which is vacuous the other way. An earlier version used six and
    # passed a build that truncated every line to eight characters.
    assert 0 < len(kept) < supplied, (
        f"trimming did not actually fire: {len(kept)} of {supplied} kept")
    assert len(kept) >= 2, f"nothing survived to check: {post.text!r}"
    for line in kept:
        assert line.endswith("%"), f"a figure was cut mid-value: {line!r}"


def test_even_an_absurd_headline_cannot_push_out_the_disclaimer():
    post = _post(headline="x" * 5000, facts=())
    assert post.length <= SOCIAL_SHARE.max_post_chars
    assert post.text.endswith(ss.DISCLAIMER)


def test_the_url_is_counted_against_the_budget_before_trimming():
    """The platform counts it too; discovering the overflow afterwards is
    how a post gets truncated mid-figure by the site instead."""
    long_url = "https://example.com/" + "a" * 100
    post = _post(url=long_url,
                 facts=_facts(*[(f"Metric {i}", f"{i}.00%") for i in range(20)]))
    assert post.length + len(long_url) + 1 <= SOCIAL_SHARE.max_post_chars


# --- the public URL -----------------------------------------------------------

def test_a_plain_instance_has_no_public_url():
    """The honest default. This app serves localhost and api_server binds
    loopback, so there is no address anyone else could open."""
    assert BRANDING.public_url == ""
    assert ss.public_url() == ""


def test_the_public_url_is_not_a_disclosure_knob():
    """test_branding forbids a field that could remove a disclosure. This
    one adds an address; it must not be mistaken for one."""
    import dataclasses
    from config import BrandingConfig
    names = {f.name for f in dataclasses.fields(BrandingConfig)}
    assert "public_url" in names
    assert not (names & {"disclosure", "disclaimer", "hide_disclaimer"})


def test_the_public_url_is_read_from_the_branding_secrets(monkeypatch):
    """Declaring the field is not the same as wiring it — the same
    listed-but-not-wired gap test_auth guards for the stores."""
    import branding
    monkeypatch.setattr(branding, "_branding_section",
                        lambda: {"name": "Acme", "public_url": "https://acme.example"})
    assert branding.brand().public_url == "https://acme.example"
    assert ss.public_url() == "https://acme.example"


def test_a_blank_public_url_stays_blank(monkeypatch):
    import branding
    monkeypatch.setattr(branding, "_branding_section", lambda: {"name": "Acme"})
    assert branding.brand().public_url == ""


def test_a_post_with_no_url_carries_none():
    post = _post()
    assert post.url == ""
    assert "http" not in post.text


def test_the_empty_state_explains_why_there_is_no_link():
    assert "no public address" in ss.NO_PUBLIC_URL
    assert "attach" in ss.NO_PUBLIC_URL


# --- X ------------------------------------------------------------------------

def test_the_x_intent_prefills_the_text():
    plan = ss.x_intent(_post())
    assert plan.prefilled is True
    assert plan.url.startswith(SOCIAL_SHARE.x_intent_url + "?")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(plan.url).query)
    assert query["text"][0] == _post().text


def test_the_intent_url_encodes_the_text_safely():
    post = _post(headline="Value & growth, 50% + 20% — \"strong\"")
    plan = ss.x_intent(post)
    assert " " not in plan.url and "\n" not in plan.url
    query = urllib.parse.parse_qs(urllib.parse.urlparse(plan.url).query)
    assert query["text"][0] == post.text, "the text did not survive a round trip"


def test_the_intent_carries_the_url_only_when_there_is_one():
    assert "url=" not in ss.x_intent(_post()).url
    assert "url=" in ss.x_intent(_post(url="https://example.com")).url


def test_no_via_attribution_is_added():
    """A `via` would attribute the post to an account this app does not
    own."""
    assert "via=" not in ss.x_intent(_post()).url
    assert '"via"' not in _code_only(Path(ss.__file__).read_text())


# --- LinkedIn -----------------------------------------------------------------

def test_linkedin_cannot_be_prefilled_and_says_so():
    """Measured against LinkedIn's own documentation: share-offsite takes
    a URL and nothing else, and the old shareArticle title/summary
    parameters are ignored."""
    plan = ss.linkedin_plan(_post())
    assert plan.prefilled is False
    assert plan.note == ss.LINKEDIN_CANNOT_PREFILL
    assert "cannot be handed text" in plan.note


def test_with_no_public_url_linkedin_goes_to_the_composer_not_the_share_box():
    """A share box with no URL comes up empty, which reads as broken."""
    plan = ss.linkedin_plan(_post())
    assert plan.url == SOCIAL_SHARE.linkedin_composer_url
    assert "share-offsite" not in plan.url


def test_with_a_public_url_linkedin_uses_the_real_share_endpoint():
    plan = ss.linkedin_plan(_post(url="https://quantix.example.com/msft"))
    assert plan.url.startswith(SOCIAL_SHARE.linkedin_share_url)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(plan.url).query)
    assert query["url"][0] == "https://quantix.example.com/msft"
    assert plan.prefilled is False, "share-offsite still cannot carry text"


def test_no_text_is_smuggled_into_the_linkedin_url():
    """Passing title/summary would look like it worked and be ignored."""
    plan = ss.linkedin_plan(_post(url="https://example.com"))
    for banned in ("title=", "summary=", "text=", "mini="):
        assert banned not in plan.url


def test_both_platforms_get_a_plan():
    names = [p.platform for p in ss.plans(_post())]
    assert names == ["X", "LinkedIn"]


# --- the card -----------------------------------------------------------------

def test_the_card_renders_a_png():
    data = ss.card_png("MSFT", "High Scorecard alignment",
                       _facts(("Net margin", "40.30%")), "Quantix", as_of=AS_OF)
    assert data and data[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_card_is_the_size_both_platforms_crop_to():
    from PIL import Image
    import io
    data = ss.card_png("MSFT", "High alignment", _facts(("P/E", "27.73")))
    assert Image.open(io.BytesIO(data)).size == SOCIAL_SHARE.card_size


def test_the_card_is_drawn_dark_whatever_the_app_theme_is():
    """An export is not a screenshot of the current view — the rule the
    deck, the workbook and the PDF already follow."""
    from PIL import Image
    import io
    data = ss.card_png("MSFT", "High alignment", _facts(("P/E", "27.73")))
    image = Image.open(io.BytesIO(data)).convert("RGB")
    # Sample well inside the body, below the accent rule at the top.
    corner = image.getpixel((image.width - 20, image.height - 20))
    assert sum(corner) < 120, f"the card background is not dark: {corner}"


def test_the_card_has_ink_on_it():
    """A structural assertion passed a tear sheet that was visibly a black
    card on a white page once. Check something was actually drawn."""
    from PIL import Image
    import io
    data = ss.card_png("MSFT", "High Scorecard alignment",
                       _facts(("Net margin", "40.30%")), "Quantix")
    image = Image.open(io.BytesIO(data)).convert("RGB")
    bright = [p for p in image.getdata() if sum(p) > 200]
    assert len(bright) > 500, f"the card is nearly blank: {len(bright)} lit pixels"


def test_the_card_drops_unavailable_figures_too():
    """The image leaves the building exactly like the text does."""
    from PIL import Image
    import io
    with_missing = ss.card_png("MSFT", "H", _facts(("DCF", "Not reported")))
    without = ss.card_png("MSFT", "H", ())
    assert with_missing == without, "an unavailable figure was drawn on the card"


def test_a_missing_font_does_not_lose_the_card():
    assert ss._font(40) is not None


def test_the_card_returns_none_when_drawing_itself_fails(monkeypatch):
    """The import guard is not the only way this can fail, and an
    `except: raise` inside the draw survived a poison that only the
    import test was watching."""
    from PIL import Image

    def _boom(*args, **kwargs):
        raise RuntimeError("no framebuffer")

    monkeypatch.setattr(Image, "new", _boom)
    assert ss.card_png("MSFT", "H", _facts(("P/E", "27.73"))) is None


def test_the_card_returns_none_rather_than_raising_when_pillow_is_gone(monkeypatch):
    """Same contract as export_deck.chart_png: lose the image, not the
    panel."""
    import builtins
    real_import = builtins.__import__

    def _no_pillow(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pillow)
    assert ss.card_png("MSFT", "H", ()) is None


def test_pillow_is_a_declared_dependency():
    """It arrives transitively today. A feature that renders an image
    must not depend on another package happening to keep it."""
    req = (Path(ss.__file__).resolve().parent / "requirements.txt").read_text()
    # An UNCOMMENTED line. "# Pillow>=10.0" still contains "Pillow" and
    # passed an earlier version of this test.
    declared = [ln.strip() for ln in req.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    assert any(ln.lower().startswith("pillow") for ln in declared), declared


def test_the_filename_carries_the_ticker_and_the_date():
    assert ss.filename("msft", AS_OF) == "MSFT-2026-09-24.png"


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Share this analysis ---")
    end = src.index("# --- end share ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_composes_through_the_module():
    """The ASSIGNMENT, not just a call. A build that called ss_compose
    into a throwaway and rendered its own text still contained the
    substring."""
    panel = _panel()
    assert "_share_post = ss_compose(" in panel
    assert "ss_plans(_share_post)" in panel
    assert "_share_post = None" not in panel


def test_the_panel_offers_the_card_as_a_download():
    panel = _panel()
    assert "ss_card_png(" in panel
    assert "download_button" in panel


def test_the_panel_shows_what_was_left_out():
    """The GUARD, not just a mention. `if False:` left the join below it
    intact, so a substring check still found the name."""
    panel = _panel()
    assert "if _share_post.dropped:" in panel
    assert "join(_share_post.dropped)" in panel


def test_the_panel_states_the_no_public_url_position():
    assert "SS_NO_PUBLIC_URL" in _panel()


def test_the_panel_never_posts():
    panel = _panel()
    for banned in ("requests.post", "urlopen", "oauth", "access_token"):
        assert banned not in panel


def test_the_panel_says_nothing_when_there_is_nothing_honest_to_say():
    assert "SS_NOTHING_TO_SHARE" in _panel()
