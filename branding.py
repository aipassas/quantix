"""White-labelling: running this app under a licensee's own name.

ONE DEPLOYMENT PER FIRM, AND THAT IS MEASURED RATHER THAN PREFERRED.
st.context.headers exists, so a single instance COULD resolve a tenant
from the Host header and give each firm its own branding and data
namespace. The blocker is elsewhere: st.secrets is a process-level
singleton, so every tenant sharing one process shares one OAuth client,
one mail sender and one Slack webhook. An asset management firm is not
going to sign its clients in through a competitor's identity provider,
so the shared credentials are the dealbreaker — not the branding.

Per-deployment branding therefore delivers the whole visible product —
their name, their colours, their domain, their credentials — without
claiming a multi-tenancy the runtime cannot actually support.

HOW THE RENAME WORKS. Rather than threading a name parameter through
sixty-odd call sites, rebrand() substitutes at the display boundary. That
is safe here because of a distinction that already holds throughout this
codebase: capital "Quantix" appears only in user-facing text, while
lowercase "quantix" is reserved for internal identifiers — logger names,
CSS class prefixes, log filenames. Verified across every module before
this was written. The substitution is word-boundary and case-sensitive,
so quantix.log and .quantix-symbol-header are untouched.

THE DISCLOSURES ARE NOT BRANDABLE. A licensee can change every name,
colour and logo. There is deliberately NO field that removes "not
investment advice", the unavailable-data notices, or the measured model
accuracy — those are the reason the numbers here can be trusted, and a
rebranded copy that dropped them while showing the same figures to a
client would be materially worse than the original. Enforcement is
structural (no such knob exists) plus tests asserting the phrases survive
a rebrand.
"""
import logging
import re
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import streamlit as st

from config import BRANDING
from logging_setup import get_logger, log_event

logger = get_logger("branding")

# The name in the source. Everything user-facing says this today, so it
# is what rebrand() substitutes away from.
_DEFAULT_NAME = "Quantix"

# Case-sensitive and word-bounded on purpose — see the module docstring.
# "Quantix" is display text; "quantix" is an identifier.
_NAME_RE = re.compile(rf"\b{_DEFAULT_NAME}\b")

# Phrases that must survive rebranding. Not a filter applied to output —
# there is no mechanism to remove them — but a checkable list, so a test
# can prove a rebrand doesn't strip them and a future edit that made them
# configurable would fail.
LOCKED_DISCLOSURE_PHRASES: Tuple[str, ...] = (
    "not investment advice",
    "unavailable",
    "never assumed to be zero",
)

# Palette fields that carry visual identity. Deliberately a handful of
# the 45 in ThemePalette: a licensee wants their accent colour, not to
# re-derive a theme whose contrast has already been tuned (see theme.py,
# where a washed-out palette was a real shipped bug).
_ACCENT_FIELDS: Tuple[str, ...] = (
    "card_accent", "card_hover_accent", "tab_selected_border", "symbol_header_border",
)


@dataclass(frozen=True)
class Brand:
    name: str = BRANDING.name
    tagline: str = BRANDING.tagline
    # Quantix's own brand colour, verified against Design_Assets: the mark
    # is exactly #00F2FE in all three logo files. Defaulting it here rather
    # than leaving it blank is what makes one value drive the login page,
    # the tear sheet, the PDF and the PowerPoint at once — a licensee's
    # [branding] accent_color still overrides it, which is the whole point
    # of this module.
    accent_color: str = "#00f2fe"
    support_email: str = ""
    footer_note: str = ""
    logo_path: str = ""

    @property
    def is_customised(self) -> bool:
        return self.name != _DEFAULT_NAME

    @property
    def title(self) -> str:
        """The page heading. Tagline is optional — a licensee may want
        only their name."""
        return f"{self.name}: {self.tagline}" if self.tagline else self.name


def _branding_section() -> dict:
    """The [branding] table, or {} when there's no secrets file — the
    normal state of an unbranded instance."""
    try:
        section = st.secrets.get("branding", {})
    except Exception:
        return {}
    try:
        return dict(section) if section else {}
    except Exception:
        return {}


def _clean(value, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def brand() -> Brand:
    """The active brand. Never raises — an unbranded or half-configured
    instance falls back to this app's own identity rather than rendering
    a blank title."""
    section = _branding_section()
    if not section:
        return Brand()

    name = _clean(section.get("name"), BRANDING.name)[:BRANDING.max_name_chars]
    return Brand(
        name=name or BRANDING.name,
        # An explicitly empty tagline is a real choice, so "" is
        # preserved rather than falling back to the default.
        tagline=str(section.get("tagline", BRANDING.tagline)).strip(),
        accent_color=_clean(section.get("accent_color")),
        support_email=_clean(section.get("support_email")),
        footer_note=_clean(section.get("footer_note")),
        logo_path=_clean(section.get("logo_path")),
    )


def rebrand(text: str) -> str:
    """Substitute the configured name into user-facing text.

    A no-op on an unbranded instance, so the default path costs a single
    string comparison. Never touches lowercase "quantix" — that is
    reserved for logger names, CSS classes and log filenames.
    """
    if not text:
        return text
    current = brand()
    if not current.is_customised:
        return text
    return _NAME_RE.sub(current.name, text)


def apply_accent(palette):
    """A palette with the licensee's accent colour substituted in.

    Returns the palette unchanged when no accent is set. Only the
    identity-carrying fields move; the contrast-tuned rest of the theme
    is left alone, because a licensee supplying one hex value has not
    re-derived forty-five of them and should not silently be treated as
    though they had.
    """
    accent = brand().accent_color
    if not accent or not is_valid_colour(accent):
        return palette
    return replace(palette, **{field: accent for field in _ACCENT_FIELDS})


def is_valid_colour(value: str) -> bool:
    """A 3- or 6-digit hex colour. Validated because an invalid value
    injected into the CSS block would break the whole stylesheet, not
    just the accent — the palette is interpolated into one f-string."""
    return bool(re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", (value or "").strip()))


def configuration_notes() -> Tuple[str, ...]:
    """Problems with the branding configuration, for the UI to show.

    A licensee who mistypes a colour should be told, not left wondering
    why nothing changed.
    """
    section = _branding_section()
    if not section:
        return ()
    notes = []
    accent = _clean(section.get("accent_color"))
    if accent and not is_valid_colour(accent):
        notes.append(
            f'accent_color "{accent}" isn\'t a hex colour like #1f6feb, so it was ignored.'
        )
    name = _clean(section.get("name"))
    if name and len(name) > BRANDING.max_name_chars:
        notes.append(
            f"The brand name was truncated to {BRANDING.max_name_chars} characters."
        )
    return tuple(notes)


def summary() -> str:
    """One line describing the active branding, for the settings panel."""
    current = brand()
    if not current.is_customised:
        return f"Unbranded — running as {current.name}."
    bits = [f"Branded as **{current.name}**"]
    if current.accent_color:
        bits.append(f"accent {current.accent_color}")
    if current.support_email:
        bits.append(f"support {current.support_email}")
    return " · ".join(bits) + "."
