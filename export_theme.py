"""One palette for every exported document — deck, tear sheet, PDF.

WHY EXPORTS HAVE THEIR OWN PALETTE RATHER THAN FOLLOWING THE APP TOGGLE.
The in-app light/dark switch is a personal viewing preference. An export
is a document that leaves the building, and its look is a house style: a
client deck should not change colour because whoever generated it
happened to have the app in light mode that morning. So exports are
always dark, and the app's theme toggle does not reach them.

IT IS BUILT FROM THE APP'S OWN DARK PALETTE, not from a second set of
hex codes pasted here. theme.py's dark palette has already had its
contrast tuned (a washed-out palette was a real shipped bug), so
re-deriving one would mean re-making decisions that were already made,
and drifting from them silently. If that palette changes, exports follow.

THE BRAND ACCENT FLOWS THROUGH AUTOMATICALLY. The accent is taken from
branding.apply_accent(), the same path the on-screen app uses, so
configuring a licensee's colour restyles the deck and the tear sheet with
no further work here. Everything else — background, body text, borders —
stays on the tuned dark palette, because a licensee wants their colour,
not a re-tuned document.

CHARTS ARE FORCED DARK HERE TOO, deliberately. The app renders Plotly
figures with a transparent background and whatever template the current
theme selects. Transparent is what makes them sit correctly on a dark
slide — but with the app in LIGHT mode the axis labels come out dark and
would vanish against the background. Exports therefore pin the dark
template rather than inheriting it.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExportPalette:
    """Colours for one exported document. Hex WITHOUT the leading '#',
    because python-pptx's RGBColor.from_string() wants it that way; use
    the css() helper when writing HTML."""
    background: str          # page / slide background
    surface: str             # cards, table body
    surface_alt: str         # table header, subtle banding
    text: str                # body copy
    text_strong: str         # headings
    text_muted: str          # captions, disclosures
    border: str
    accent: str              # brand colour
    positive: str
    negative: str

    def css(self, name: str) -> str:
        return "#" + getattr(self, name)


def _clean(value: Optional[str], fallback: str) -> str:
    """Normalise a hex colour to six upper-case digits, no '#'."""
    text = (value or "").strip().lstrip("#")
    if len(text) == 3:                     # #abc -> #aabbcc
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        return fallback
    try:
        int(text, 16)
    except ValueError:
        return fallback
    return text.upper()


def palette() -> ExportPalette:
    """The export palette, with the licensee's accent applied."""
    background = surface = surface_alt = None
    text = text_strong = text_muted = border = accent = None

    try:
        from theme import PALETTES

        dark = PALETTES["dark"]
        background = getattr(dark, "app_bg", None)
        surface = getattr(dark, "card_bg", None)
        # tab_selected_bg, not table_head_bg: on this palette the table
        # header is the SAME #0a0a0a as the card, which reads as no header
        # at all once both are drawn on black. This field is the tuned
        # "one step lighter than a card" value.
        surface_alt = getattr(dark, "tab_selected_bg", None)
        text = getattr(dark, "app_text", None)
        text_strong = getattr(dark, "header_text", None)
        text_muted = getattr(dark, "tab_inactive_text", None)
        accent = getattr(dark, "card_accent", None)
    except Exception:
        pass

    # The brand accent wins over the palette's own, exactly as on screen.
    try:
        from branding import brand

        configured = brand().accent_color
        if configured:
            accent = configured
    except Exception:
        pass

    return ExportPalette(
        background=_clean(background, "000000"),
        surface=_clean(surface, "0A0A0A"),
        surface_alt=_clean(surface_alt, "141414"),
        text=_clean(text, "E2E8F0"),
        text_strong=_clean(text_strong, "FFFFFF"),
        text_muted=_clean(text_muted, "9CA3AF"),
        # Borders are not a palette field: on a true-black ground a hairline
        # has to be lighter than the surface to read at all, and the app
        # draws its own with rgba it can't hand over as a solid hex.
        border=_clean(border, "2A2F3A"),
        accent=_clean(accent, "00EA77"),
        # Fixed rather than themed. Red/green carry meaning in a financial
        # document, and a licensee restyling "loss" would change what the
        # document says, not how it looks. Both are lifted for contrast
        # against black — the on-screen #1B7F37 / #B42318 pair is tuned for
        # a white page and reads as mud here.
        positive="3FD37E",
        negative="FF6B6B",
    )


def plotly_overrides() -> dict:
    """update_layout() kwargs that pin a figure to the export look.

    Transparent backgrounds on purpose: the slide or page ground shows
    through, so one rendered chart sits correctly on any dark surface
    without a rectangle of slightly-wrong black around it.
    """
    colours = palette()
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": colours.css("text_strong")},
    }
