"""Composing a post about an analysis, for X or LinkedIn.

THIS APP NEVER POSTS. It composes text, renders a card, and opens the
platform's own compose window with the text pre-filled where the platform
allows it. The reader presses Post. No OAuth token is stored, no API is
called, and there is deliberately no function here that publishes
anything — publishing on somebody's behalf is a different feature needing
a different consent, and this ticket does not ask for it.

THE TICKET'S PREMISE DOES NOT HOLD ON A LOCAL APP, and pretending
otherwise would ship something broken. "A generated preview image" means
an Open Graph card, which requires a public URL that a crawler can fetch;
Streamlit serves localhost and api_server binds loopback by default. So
there is no address to preview, and the "viral acquisition loop" has
nothing to click through to. Measured, not assumed: nothing in this
codebase held any notion of a public address before `BrandingConfig.
public_url` was added for exactly this, blank by default.

WHAT WORKS INSTEAD IS BETTER. The card is rendered to a PNG the reader
downloads and attaches. An attached image needs no public address, shows
on both platforms, and cannot be scraped wrong — whereas an Open Graph
preview depends on a crawler reaching a server that, here, is not
reachable.

THE TWO PLATFORMS ARE NOT SYMMETRIC. Checked against their own
documentation on 2026-09-23. X's web intent accepts `text`, `url`,
`hashtags` and `via`, so a post can be pre-filled. LinkedIn's
share-offsite endpoint accepts a URL AND NOTHING ELSE: headline, summary
and thumbnail are scraped from that page's Open Graph tags, and the older
shareArticle title/summary parameters are ignored. So LinkedIn cannot be
pre-filled at all, and with no public_url it cannot be used at all —
`linkedin_plan()` hands over the text to paste and says why, rather than
opening a share box that would come up empty. Neither platform accepts an
image through a URL, on either path.

EVERY FIGURE IN THE TEXT IS ONE THE APP COMPUTED AND SHOWS. A post leaves
the building and cannot be footnoted afterwards, so `compose()` takes
already-formatted display strings and refuses the app's own
unavailable markers rather than re-deriving anything. A failed DCF
rendering as "$0.00" is the specific bug this codebase has shipped twice;
it must not become somebody's public post.

THE DISCLAIMER IS NOT OPTIONAL. branding.py locks "not investment
advice" against rebranding precisely so a licensee cannot strip it, and a
public post is the single place it matters most. `compose()` appends it
and the trimmer preserves it — text is shortened from the middle, never
from the end.
"""
import datetime
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from config import SOCIAL_SHARE
from logging_setup import get_logger, log_exception

logger = get_logger("social_share")

# Locked by branding.LOCKED_DISCLOSURE_PHRASES. A test asserts this
# sentence carries the phrase and survives rebrand().
DISCLAIMER = "Research, not investment advice."

NO_PUBLIC_URL = (
    "This instance has no public address, so a shared post carries no link — it "
    "runs on your machine and nothing outside could open it. The card below is an "
    "image you attach to the post instead, which works on both platforms and needs "
    "no address at all."
)

LINKEDIN_CANNOT_PREFILL = (
    "LinkedIn's share endpoint accepts a URL and nothing else — the headline and "
    "summary are read from that page's Open Graph tags, and it cannot be handed "
    "text. So copy the post below, open LinkedIn and paste it. That extra step is "
    "LinkedIn's, not a missing piece here."
)

NOTHING_TO_SHARE = (
    "There is nothing to share yet — the Scorecard has not produced a verdict for "
    "this ticker. Nothing is composed from a figure the app could not compute."
)

# The app's own markers for a figure it refused to invent. A post must
# never carry one, and must never carry a zero standing in for one.
UNAVAILABLE_MARKERS: Tuple[str, ...] = (
    "not reported", "unavailable", "n/a", "none", "—", "-", "",
)


@dataclass(frozen=True)
class Fact:
    """One already-formatted figure, as the app displays it.

    `display` is a STRING on purpose. Taking a float here would invite
    this module to format, round or scale it — and every unit error this
    codebase has shipped came from a second place formatting the same
    number. The panel passes what is on screen.
    """
    label: str
    display: str

    @property
    def reportable(self) -> bool:
        return self.display.strip().lower() not in UNAVAILABLE_MARKERS


@dataclass(frozen=True)
class Post:
    """A composed post and everything needed to act on it."""
    text: str
    facts: Tuple[Fact, ...] = ()
    dropped: Tuple[str, ...] = ()      # labels left out, and why they were
    url: str = ""

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class Plan:
    """What a platform button can actually do here."""
    platform: str
    url: str = ""          # where the button goes, "" when there is none
    prefilled: bool = False
    note: str = ""


def _clean(value) -> str:
    return str(value or "").strip()


def public_url() -> str:
    """The configured public address, or "" — the normal state."""
    try:
        from branding import brand
        return _clean(brand().public_url)
    except Exception:
        log_exception(logger, "social_share.brand_failed", section="social_share")
        return ""


def reportable(facts: Sequence[Fact]) -> Tuple[Tuple[Fact, ...], Tuple[str, ...]]:
    """(the facts worth posting, the labels dropped).

    Returned as a pair so the panel can SHOW what was left out. Silently
    dropping a figure would leave the reader thinking the post is
    complete; the app's rule everywhere else is that a missing number is
    an explicit state, not an absence.
    """
    kept = tuple(f for f in facts if f.reportable)
    dropped = tuple(f.label for f in facts if not f.reportable)
    return kept, dropped


def compose(ticker: str, headline: str, facts: Sequence[Fact],
            brand_name: str = "", as_of: Optional[datetime.date] = None,
            url: str = "", max_chars: Optional[int] = None) -> Optional[Post]:
    """The post text, or None when there is nothing honest to say.

    Returns None rather than an empty-ish post when the headline is
    missing: a share that says only "AAPL" and a disclaimer is not a
    share, and composing one would put the app's name against an empty
    claim.
    """
    ticker = _clean(ticker).upper()
    headline = _clean(headline)
    if not ticker or not headline:
        return None

    max_chars = SOCIAL_SHARE.max_post_chars if max_chars is None else max_chars
    kept, dropped = reportable(facts)

    lines: List[str] = [f"{ticker} — {headline}"]
    for fact in kept:
        lines.append(f"{fact.label}: {fact.display}")
    if as_of:
        lines.append(f"As of {as_of.isoformat()}.")

    tail = DISCLAIMER
    if brand_name:
        tail = f"{_clean(brand_name)} · {DISCLAIMER}"
    lines.append(tail)

    text = "\n".join(lines)
    # The platform counts the URL too, so it is reserved before trimming
    # rather than discovered to overflow afterwards.
    budget = max_chars - (len(url) + 1 if url else 0)
    text = _trim(text, budget, keep_last=tail)
    return Post(text=text, facts=kept, dropped=dropped, url=url)


def _trim(text: str, budget: int, keep_last: str) -> str:
    """Shorten from the MIDDLE, never the end.

    Trimming the tail is the obvious implementation and it removes the
    disclaimer — the one line that must survive. Whole fact lines are
    dropped from the bottom up instead, and the first line and the tail
    are kept.
    """
    if len(text) <= budget:
        return text
    lines = text.split("\n")
    head, tail = lines[0], lines[-1]
    middle = lines[1:-1]
    while middle and len("\n".join([head] + middle + [tail])) > budget:
        middle.pop()
    trimmed = "\n".join([head] + middle + [tail])
    if len(trimmed) > budget:
        # Even the headline does not fit. Cut IT, never the disclaimer.
        room = budget - len(tail) - 1
        head = (head[:max(room - 1, 0)] + "…") if room > 1 else ""
        trimmed = "\n".join([p for p in (head, tail) if p])
    return trimmed


def x_intent(post: Post) -> Plan:
    """X's web intent, pre-filled. Documented parameters: text, url,
    hashtags, via — this uses the first two and nothing else, because a
    `via` would attribute the post to an account this app does not own."""
    params = {"text": post.text}
    if post.url:
        params["url"] = post.url
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return Plan("X", f"{SOCIAL_SHARE.x_intent_url}?{query}", prefilled=True)


def linkedin_plan(post: Post) -> Plan:
    """LinkedIn's share endpoint when there is a URL for it, otherwise
    the composer plus an explanation.

    share-offsite takes a URL and reads everything else from that page's
    Open Graph tags, so with no public address it would open an empty
    share box. Handing over the text to paste is the honest answer.
    """
    if post.url:
        query = urllib.parse.urlencode({"url": post.url},
                                       quote_via=urllib.parse.quote)
        return Plan("LinkedIn", f"{SOCIAL_SHARE.linkedin_share_url}?{query}",
                    prefilled=False,
                    note="LinkedIn builds the card from the page's own tags; the "
                         "text above is not carried, so paste it into the post.")
    return Plan("LinkedIn", SOCIAL_SHARE.linkedin_composer_url, prefilled=False,
                note=LINKEDIN_CANNOT_PREFILL)


def plans(post: Post) -> Tuple[Plan, ...]:
    return (x_intent(post), linkedin_plan(post))


# --- the card -----------------------------------------------------------------

def _hex(value, fallback: str) -> str:
    """A colour Pillow will accept.

    export_theme.palette() stores BARE hex — "000000", not "#000000" —
    because python-pptx and openpyxl want it that way. Pillow raises
    "unknown color specifier" on it, and so would any CSS. Measured, not
    assumed: the first render of this card failed on exactly that.
    """
    text = str(value or "").strip()
    if not text:
        return fallback
    return text if text.startswith("#") else f"#{text}"


def card_png(ticker: str, headline: str, facts: Sequence[Fact],
             brand_name: str = "", accent: str = "",
             as_of: Optional[datetime.date] = None) -> Optional[bytes]:
    """The share card as PNG bytes, or None if it cannot be drawn.

    Returns None rather than raising, exactly like export_deck.chart_png:
    a machine missing a usable font should lose the image, not the panel.

    Drawn DARK regardless of the app's theme, via export_theme.palette().
    An export is not a screenshot of the current view — the same rule the
    deck, the workbook and the PDF already follow, and the reason charts
    are pinned before rendering.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        log_exception(logger, "social_share.pillow_missing", section="social_share")
        return None

    try:
        from export_theme import palette
        colours = palette()
        background = _hex(getattr(colours, "background", None), "#000000")
        text_colour = _hex(getattr(colours, "text", None), "#e6e6e6")
        muted = _hex(getattr(colours, "text_muted", None), "#9a9a9a")
        accent = _hex(accent or getattr(colours, "accent", None), "#00f2fe")
    except Exception:
        background, text_colour, muted, accent = (
            "#000000", "#e6e6e6", "#9a9a9a", _hex(accent, "#00f2fe"))

    width, height = SOCIAL_SHARE.card_size
    kept, _ = reportable(facts)

    try:
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        big = _font(64)
        mid = _font(34)
        small = _font(26)

        margin = 72
        # The accent rule, so the card reads as this app's even in a feed.
        draw.rectangle([0, 0, width, 10], fill=accent)

        y = margin
        draw.text((margin, y), _clean(ticker).upper(), font=big, fill=text_colour)
        y += 92
        draw.text((margin, y), _clean(headline), font=mid, fill=accent)
        y += 74

        for fact in kept:
            draw.text((margin, y), fact.label, font=small, fill=muted)
            draw.text((margin + 420, y), fact.display, font=small, fill=text_colour)
            y += 46
            if y > height - margin - 110:
                break

        footer = DISCLAIMER
        if brand_name:
            footer = f"{_clean(brand_name)} · {footer}"
        if as_of:
            footer = f"{footer}  ·  As of {as_of.isoformat()}"
        draw.text((margin, height - margin - 20), footer, font=small, fill=muted)

        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        log_exception(logger, "social_share.card_failed", section="social_share")
        return None


# Tried in order. The first two ship with macOS; DejaVu comes with many
# Linux images and with Pillow's own wheels. A missing font falls back to
# Pillow's bitmap default, which is ugly but legible — losing the card
# entirely over a typeface would be worse.
_FONT_PATHS: Tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int):
    from PIL import ImageFont
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def filename(ticker: str, as_of: Optional[datetime.date] = None) -> str:
    day = (as_of or datetime.date.today()).isoformat()
    return f"{_clean(ticker).upper() or 'SHARE'}-{day}.png"
