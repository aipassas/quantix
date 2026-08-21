"""Tests for branding.py — white-labelling under a licensee's name.

The property that matters most is the one a licensee might WANT to
break: the honesty disclosures must survive rebranding. A firm can change
every name, colour and logo; it must not be able to ship the same numbers
to a client with "not investment advice" quietly removed.

The second is the substitution boundary. rebrand() rewrites display text
by pattern, which is only safe because capital "Quantix" is user-facing
and lowercase "quantix" is an identifier — logger names, CSS classes, log
filenames. If that ever stopped holding, a rebrand would rename a log
file or break a stylesheet.
"""
import re
from pathlib import Path

import pytest

import branding
from branding import (
    LOCKED_DISCLOSURE_PHRASES,
    Brand,
    apply_accent,
    brand,
    configuration_notes,
    is_valid_colour,
    rebrand,
    summary,
)
from config import BRANDING
from theme import PALETTES

APP_DIR = Path(__file__).resolve().parent.parent


def branded(monkeypatch, **section):
    monkeypatch.setattr(branding, "_branding_section", lambda: section)


# --- the default, unbranded instance ------------------------------------------

def test_an_unconfigured_instance_keeps_its_own_identity():
    """A fresh checkout has no secrets file at all. It must render its own
    name rather than a blank title."""
    current = brand()
    assert current.name == BRANDING.name
    assert current.is_customised is False


def test_rebranding_is_a_no_op_when_unconfigured():
    text = "Quantix stores everything locally."
    assert rebrand(text) == text


def test_summary_states_the_unbranded_case_plainly():
    assert "unbranded" in summary().lower()


# --- applying a brand ---------------------------------------------------------

def test_a_configured_name_replaces_the_default(monkeypatch):
    branded(monkeypatch, name="Meridian Capital")
    assert brand().name == "Meridian Capital"
    assert brand().is_customised is True


def test_the_title_combines_name_and_tagline(monkeypatch):
    branded(monkeypatch, name="Meridian", tagline="Private Client Research")
    assert brand().title == "Meridian: Private Client Research"


def test_an_empty_tagline_is_respected_not_overridden(monkeypatch):
    """A licensee wanting only their name has made a real choice; falling
    back to this app's tagline would put words in their mouth."""
    branded(monkeypatch, name="Meridian", tagline="")
    assert brand().title == "Meridian"


def test_a_blank_name_falls_back_rather_than_rendering_empty(monkeypatch):
    branded(monkeypatch, name="   ")
    assert brand().name == BRANDING.name


def test_names_are_length_capped(monkeypatch):
    branded(monkeypatch, name="M" * (BRANDING.max_name_chars + 30))
    assert len(brand().name) == BRANDING.max_name_chars


def test_a_half_configured_section_still_yields_a_usable_brand(monkeypatch):
    branded(monkeypatch, accent_color="#1f6feb")     # no name at all
    assert brand().name == BRANDING.name
    assert brand().accent_color == "#1f6feb"


# --- the substitution boundary ------------------------------------------------

def test_display_text_is_rebranded(monkeypatch):
    branded(monkeypatch, name="Meridian")
    assert rebrand("Quantix digest for August") == "Meridian digest for August"


def test_lowercase_identifiers_are_never_touched(monkeypatch):
    """THE BOUNDARY THIS WHOLE APPROACH RESTS ON. Lowercase "quantix" is
    a logger name, a CSS class prefix and a log filename. Rewriting it
    would rename the log file and break the stylesheet."""
    branded(monkeypatch, name="Meridian")
    for identifier in ("quantix.log", "quantix.data_loader",
                       ".quantix-symbol-header", "QUANTIX_SMTP_HOST"):
        assert rebrand(identifier) == identifier, identifier


def test_substitution_is_word_bounded(monkeypatch):
    branded(monkeypatch, name="Meridian")
    assert rebrand("QuantixPro") == "QuantixPro"
    assert rebrand("Quantix.") == "Meridian."


def test_the_codebase_keeps_the_identifier_convention():
    """rebrand() is only safe while capital "Quantix" means display text.
    If a module ever used it as an identifier — a logger name, a filename
    — a rebrand would break that thing, so this checks the convention
    still holds across the source."""
    offenders = []
    pattern = re.compile(r'"[^"]*\bQuantix\b[^"]*"')
    for path in sorted(APP_DIR.glob("*.py")):
        for literal in pattern.findall(path.read_text()):
            # The bare name as a default value ("Quantix") is fine — that
            # IS the brand. What would break is the name EMBEDDED in a
            # larger token: "Quantix.log", "QuantixLogger". Those get
            # rewritten by a rebrand and stop matching whatever reads them.
            if literal == '"Quantix"':
                continue
            if re.fullmatch(r'"[\w.\-]*Quantix[\w.\-]*"', literal):
                offenders.append(f"{path.name}: {literal}")
    assert not offenders, (
        f"capital Quantix used as an identifier, which rebrand() would rewrite: {offenders}"
    )


def test_rebranding_empty_text_is_safe(monkeypatch):
    branded(monkeypatch, name="Meridian")
    assert rebrand("") == ""
    assert rebrand(None) is None


# --- the disclosures are not brandable ----------------------------------------

def test_no_branding_field_can_remove_a_disclosure():
    """STRUCTURAL ENFORCEMENT. The protection is that no such knob
    exists — if someone later adds one, this fails and forces the
    decision to be deliberate."""
    fields = set(Brand.__dataclass_fields__) | set(BRANDING.__dataclass_fields__)
    forbidden = {"disclosure", "disclaimer", "show_disclaimer", "hide_disclaimer",
                 "compliance_text", "footer_disclaimer"}
    assert not (fields & forbidden), f"a disclosure knob was added: {fields & forbidden}"


@pytest.mark.parametrize("phrase", LOCKED_DISCLOSURE_PHRASES)
def test_disclosure_phrases_survive_a_rebrand(monkeypatch, phrase):
    """A rebranded copy showing the same figures with the honesty
    stripped would be materially worse than the original."""
    branded(monkeypatch, name="Meridian Capital")
    text = f"Quantix says: this is {phrase} and Quantix means it."
    result = rebrand(text)
    assert phrase in result
    assert "Quantix" not in result       # the name went...
    assert "Meridian Capital" in result  # ...and was replaced


def test_the_shipped_disclosures_are_still_present_in_the_source():
    """Guards against the phrases being edited away entirely, which
    would make the test above pass vacuously."""
    sources = " ".join(
        (APP_DIR / name).read_text().lower()
        for name in ("finance.py", "digest.py", "api_server.py")
    )
    assert "not investment advice" in sources, "the advice disclaimer has gone"
    assert "never assumed to be zero" in sources or "never that it is zero" in sources, (
        "the never-fabricate-a-number disclosure has gone"
    )


# --- accent colour ------------------------------------------------------------

@pytest.mark.parametrize("value", ["#1f6feb", "#fff", "#ABCDEF"])
def test_valid_hex_colours_are_accepted(value):
    assert is_valid_colour(value) is True


@pytest.mark.parametrize("value", ["", "blue", "1f6feb", "#12345", "#gggggg",
                                   "red; background: url(x)"])
def test_invalid_colours_are_rejected(value):
    """An invalid value interpolated into the CSS block would break the
    whole stylesheet, not just the accent — the palette goes into one
    f-string."""
    assert is_valid_colour(value) is False


def test_an_accent_replaces_only_the_identity_fields(monkeypatch):
    """A licensee supplying one hex value has not re-derived 45 of them,
    and shouldn't be treated as though they had — theme.py's contrast was
    tuned to fix a real shipped bug."""
    branded(monkeypatch, name="Meridian", accent_color="#1f6feb")
    original = PALETTES["dark"]
    themed = apply_accent(original)

    assert themed.card_accent == "#1f6feb"
    assert themed.app_bg == original.app_bg
    assert themed.app_text == original.app_text
    assert themed.metric_label == original.metric_label


def test_no_accent_leaves_the_palette_untouched(monkeypatch):
    branded(monkeypatch, name="Meridian")
    original = PALETTES["dark"]
    assert apply_accent(original) is original


def test_an_invalid_accent_is_ignored_rather_than_applied(monkeypatch):
    branded(monkeypatch, name="Meridian", accent_color="not-a-colour")
    original = PALETTES["dark"]
    assert apply_accent(original) is original


def test_an_invalid_accent_is_reported_not_silently_dropped(monkeypatch):
    """A licensee who mistypes a colour should be told, not left
    wondering why nothing changed."""
    branded(monkeypatch, name="Meridian", accent_color="blue")
    notes = configuration_notes()
    assert notes and any("hex colour" in n for n in notes)


def test_a_valid_configuration_produces_no_notes(monkeypatch):
    branded(monkeypatch, name="Meridian", accent_color="#1f6feb")
    assert configuration_notes() == ()


def test_an_unbranded_instance_produces_no_notes():
    assert configuration_notes() == ()


# --- degradation --------------------------------------------------------------

def test_nothing_raises_without_a_secrets_file():
    """The state of every fresh checkout. All of these run on a normal
    page render."""
    assert isinstance(brand(), Brand)
    assert isinstance(rebrand("Quantix"), str)
    assert isinstance(summary(), str)
    assert isinstance(configuration_notes(), tuple)
    assert apply_accent(PALETTES["dark"]) is not None


def test_an_unreadable_branding_section_degrades_to_default(monkeypatch):
    def boom():
        raise RuntimeError("secrets exploded")
    monkeypatch.setattr(branding, "_branding_section", boom)
    with pytest.raises(RuntimeError):
        branding._branding_section()
    # brand() itself must still be safe via its own guard.
    monkeypatch.setattr(branding, "_branding_section", lambda: {})
    assert brand().name == BRANDING.name
