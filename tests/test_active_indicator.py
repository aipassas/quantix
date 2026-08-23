"""Tests for the active-ticker indicator.

These read finance.py's source, following test_metric_help.py: the
acceptance criterion is "the selected ticker looks different from the
others, everywhere it appears", which is otherwise only checkable by
clicking between tickers in a browser.

The bug this guards is subtle. type="primary" is doing two unrelated jobs
in this app — selection state (watchlist row, header chip, peer switcher)
and genuine call-to-action (Run Screen, Create key, Post note). Before
this change both rendered in Streamlit's stock #FF4B4B, so "you are here"
wore the same red as "send this email", and — worse in a financial app —
the same red that means a losing position. Styling primary at large would
have flattened the two together again from the other direction.
"""
import ast
import re
from pathlib import Path

import pytest

FINANCE_PY = Path(__file__).resolve().parent.parent / "finance.py"
SOURCE = FINANCE_PY.read_text()

# The comments in this file deliberately quote the OLD behaviour — the
# stock #FF4B4B, the too-loose st-key-quick_ selector — so a scan of the
# raw text finds them in the prose that explains why they are gone. Strip
# CSS and Python comments so these tests read the code, not its history.
CODE = re.sub(r"/\*.*?\*/", "", SOURCE, flags=re.S)
CODE = "\n".join(line for line in CODE.splitlines()
                 if not line.lstrip().startswith("#"))

# Widget-key prefixes whose primary state means "this is the current
# ticker" rather than "this button performs an action".
SELECTION_PREFIXES = ("wl_go_", "qa_chip_", "peer_switch_")

# A sample of keys whose primary state is a real call to action. These
# must keep Streamlit's own primary styling.
ACTION_KEYS = ("support_send", "api_key_create", "onboarding_next",
               "collab_post", "screener_save_btn")


def test_every_selection_surface_marks_the_active_ticker():
    """Each of the three places a ticker can be selected must vary its
    button type on whether that ticker is the current one."""
    for prefix in SELECTION_PREFIXES:
        pattern = re.compile(
            r'key=f"' + prefix + r'\{[^}]+\}"[^)]*?type="primary" if ',
            re.S)
        assert pattern.search(SOURCE), f"{prefix} does not switch type on active"


def test_the_active_style_is_scoped_to_selection_keys_only():
    """Styling button[kind="primary"] at large would repaint every
    call-to-action in the app as a selection chip."""
    for prefix in SELECTION_PREFIXES:
        assert f'[class*="st-key-{prefix}"] button[kind="primary"]' in SOURCE, prefix

    # No unscoped primary rule that would catch the action buttons too.
    unscoped = re.findall(r'^\s*button\[kind="primary"\]', CODE, re.M)
    assert not unscoped, f"unscoped primary rule(s) found: {unscoped}"


def test_action_buttons_are_not_restyled_as_selections():
    for key in ACTION_KEYS:
        assert f'[class*="st-key-{key}"] button[kind="primary"]' not in SOURCE, key


def test_no_selector_matches_a_wider_key_prefix_than_intended():
    """Regression: the chips were keyed quick_{ticker}, so a
    [class*="st-key-quick_"] selector also matched quick_stats_save and
    quick_stats_reset — a different control entirely. Renamed to qa_chip_."""
    selectors = re.findall(r'\[class\*="st-key-([a-z_]+)"\]', CODE)
    for selector in set(selectors):
        # Every other widget key in the file that would also be caught.
        others = re.findall(r'key=(?:f)?"([a-z_][a-z0-9_]*)', CODE)
        caught = {k for k in others
                  if k.startswith(selector) and not k.startswith(tuple(SELECTION_PREFIXES))}
        assert not caught, f'selector st-key-{selector} also catches {sorted(caught)}'


def test_the_active_chip_borrows_neither_red_nor_green():
    """Red means loss and green means gain everywhere else in this app, so
    the third state must not reuse either or it says two things at once."""
    block = CODE.split('[class*="st-key-wl_go_"] button[kind="primary"]')[1][:900]
    for banned in ("#ef4444", "#22c55e", "#00ea77", "#FF4B4B", "#ff4b4b"):
        assert banned not in block, f"active-ticker style uses {banned}"


def test_the_active_style_comes_from_the_palette_not_hardcoded_hex():
    """It has to invert with the light theme; a literal colour would not."""
    block = CODE.split('[class*="st-key-wl_go_"] button[kind="primary"]')[1][:900]
    assert "{_theme.tab_selected_bg}" in block
    assert "{_theme.header_text}" in block


def test_both_palettes_keep_the_active_border_legible():
    from theme import PALETTES

    def luminance(value):
        value = value.lstrip("#")
        channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        adjusted = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                    for c in channels]
        return 0.2126 * adjusted[0] + 0.7152 * adjusted[1] + 0.0722 * adjusted[2]

    for name in ("dark", "light"):
        palette = PALETTES[name]
        face, border = palette.tab_selected_bg, palette.header_text
        high, low = max(luminance(face), luminance(border)), min(luminance(face), luminance(border))
        ratio = (high + 0.05) / (low + 0.05)
        assert ratio >= 3.0, f"{name}: active border only {ratio:.2f}:1 against its own face"


def test_the_watchlist_label_keeps_its_own_direction_colour():
    """The chip's colour now means "selected", so up/down has to live on
    the label or a falling stock in the active row loses its red."""
    # Anchored on the whole label-construction block. "_wl_move = "
    # appears twice (the plain string, then the coloured wrap), so
    # splitting on it returns only the gap between the two.
    start = SOURCE.index("_wl_move = ")
    end = SOURCE.index("_wl_label = f\"{_wl_snap.ticker}", start)
    block = SOURCE[start:end]
    assert ":green[" in block, "gains lose their colour on the active chip"
    assert ":red[" in block, "losses lose their colour on the active chip"


def test_finance_still_parses():
    ast.parse(SOURCE)
