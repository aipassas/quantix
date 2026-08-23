"""Tests for the account menu in the top-right of the sticky header.

Two things are worth pinning. First the initials, because they are the
only identity most people will see and the fallbacks matter more than the
happy path — a blank circle reads as a rendering bug, not as "no name".

Second, that the theme control exists exactly ONCE in the app. It moved
here from the sidebar, and putting it in both places raises
DuplicateWidgetID at runtime while every unit test still passes, because
Streamlit only enforces key uniqueness inside a real session.
"""
import ast
import re
from pathlib import Path

import pytest

import profile_menu

ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text()
MENU = (ROOT / "profile_menu.py").read_text()


# --- initials -----------------------------------------------------------------

@pytest.mark.parametrize("name, email, expected", [
    ("Aggelos Passas", "a@x.com", "AP"),          # the brief's own example
    ("Jean-Luc Picard", "j@x.com", "JL"),         # hyphenated counts as two words
    ("Angelos", "angelos@x.com", "AN"),           # one word -> first two letters
    ("X", "x@y.com", "X"),                        # single letter stays single
    ("", "aggelos@example.com", "AG"),            # no name -> email local part
    ("", "", "?"),                                # nothing at all
    ("   ", "   ", "?"),
    ("", "123@x.com", "?"),                       # digits are not initials
])
def test_initials(name, email, expected):
    assert profile_menu.initials(name, email) == expected


def test_initials_never_returns_an_empty_string():
    """A blank avatar circle looks like a failure, not like missing data."""
    for name, email in (("", ""), (" ", " "), ("!!", "!!@x.com"), (None, None)):
        assert profile_menu.initials(name or "", email or "").strip()


def test_initials_are_at_most_two_characters():
    for name in ("Aggelos Passas", "A B C D E", "Verylongsinglename"):
        assert len(profile_menu.initials(name, "")) <= 2


# --- the theme control lives in exactly one place -----------------------------

def test_only_one_theme_widget_exists():
    """Two widgets sharing key="theme_choice" raise DuplicateWidgetID in a
    real session — and bare-mode unit tests would not notice."""
    # Counted by scanning for the key outside comments rather than with a
    # regex over the call: the widget spans several lines and contains a
    # nested list(PALETTES.keys()), so a naive [^)]* never reaches the key.
    def widget_keys(source):
        found = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if 'key="theme_choice"' in stripped:
                found.append(stripped)
        return found

    total = widget_keys(FINANCE) + widget_keys(MENU)
    assert len(total) == 1, f"expected exactly one theme widget, found {total}"


def test_the_theme_widget_is_the_menus():
    assert 'key="theme_choice"' in MENU


def test_the_sidebar_says_where_the_theme_moved_to():
    """Removing a control without saying where it went is how people
    conclude a feature was dropped."""
    assert "account menu" in FINANCE.split('st.caption("Appearance")')[1][:400]


def test_changing_the_theme_reruns():
    """The palette is read at the top of the script, so without a rerun the
    switch appears to do nothing until the next interaction."""
    block = MENU.split("user.theme_changed")[1][:400]
    assert "st.rerun()" in block


# --- every menu item does something -------------------------------------------

def test_sign_out_clears_both_auth_paths():
    """st.logout() only clears Streamlit's OIDC cookie; a local password
    session would survive it."""
    block = MENU.split('key="profile_sign_out"')[1][:600]
    assert "sign_out_local()" in block
    assert "reset_state()" in block
    assert "st.logout()" in block


def test_help_item_actually_opens_the_sidebar_panel():
    """A popover cannot expand something inside the sidebar, so the item
    sets a flag the panel reads. Without the wiring it would be a button
    that does nothing."""
    assert profile_menu.OPEN_HELP_KEY in MENU
    assert "profile_menu.help_requested()" in FINANCE
    assert 'st.sidebar.expander("Help & Support", expanded=profile_menu.help_requested())' in FINANCE


def test_help_flag_is_consumed_not_merely_read():
    """Leaving it set would pin the panel open and look stuck."""
    source = MENU.split("def help_requested")[1]
    assert ".pop(" in source


def test_no_dead_settings_entry():
    """There is no Settings page in this app. The menu points at the
    sidebar panels instead of offering an entry that opens nothing."""
    assert profile_menu.SETTINGS_PANELS
    for panel in profile_menu.SETTINGS_PANELS:
        assert f'st.sidebar.expander("{panel}"' in FINANCE, f"{panel} is not a real sidebar panel"


# --- placement ----------------------------------------------------------------

def test_the_menu_renders_inside_the_sticky_header():
    """"Top right" means the right end of the one block always on screen."""
    header_at = FINANCE.index("with symbol_header_container:")
    render_at = FINANCE.index("profile_menu.render()")
    stats_at = FINANCE.index("_render_quick_stats()")
    assert header_at < render_at < stats_at


def test_finance_still_parses():
    ast.parse(FINANCE)
