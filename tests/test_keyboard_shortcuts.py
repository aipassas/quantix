"""Keyboard shortcuts, the command palette, and the constraints both live under.

The interesting assertions here are about what the PLATFORM allows, since
that is what shaped the feature:

  * onboarding.py records that a <script> injected through st.markdown
    never executes. That is true and unchanged. This feature uses
    st.components.v1.html instead — a real iframe whose scripts do run —
    and the tests below pin that distinction so nobody "simplifies" the
    listener into a markdown call that would silently do nothing.
  * Chrome reserves ⌘N for New Window and the page never receives the
    event, so the task's requested binding is not deliverable. ⌘⇧A was
    probed as a replacement. A test asserts ⌘N is not claimed anywhere,
    because a shortcut documented but never firing is worse than none.
"""
import ast
import json
import re
from pathlib import Path

import pytest

import keyboard_shortcuts as kbd


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


# --- the tab list stays in step with the app ----------------------------------

def test_the_tab_list_matches_the_tabs_finance_actually_creates():
    """Two lists of the same eight labels drift the first time one is
    renamed, and ⌘3 would then open the wrong panel."""
    # The strip is no longer a literal in finance.py: tab LABELS follow
    # the asset class, so asset_views owns them and finance builds from
    # it. The drift this guards against is the same one — MAIN_TABS must
    # stay in step with the panels the app actually creates.
    import asset_views

    assert kbd.MAIN_TABS == asset_views.BASE_TABS, (
        "MAIN_TABS has drifted from the tab strip asset_views builds")
    assert "asset_views.tab_labels(asset_kind)" in FINANCE
    assert "_tab_objects = st.tabs(_tab_labels)" in FINANCE
    # Exactly eight panels are unpacked positionally, so the base strip
    # must stay eight long or the unpacking breaks.
    assert len(asset_views.BASE_TABS) == 8
    assert "_tab_objects[:8]" in FINANCE


def test_every_tab_has_a_numbered_shortcut_the_browser_allows():
    """⌘1–⌘9 are capturable; a ninth tab would be fine, a tenth would not
    have a key and must not silently look like it does."""
    assert len(kbd.MAIN_TABS) <= 9, "a tenth tab has no ⌘N key available"


# --- the platform constraints -------------------------------------------------

def test_the_listener_is_a_component_iframe_not_injected_markdown():
    """st.markdown inserts HTML via innerHTML and a <script> there never
    runs — the wall onboarding.py documents. components.html is a real
    iframe and does execute."""
    assert "components.html(" in FINANCE
    listener_call = FINANCE[FINANCE.index("components.html("):]
    listener_call = listener_call[:listener_call.index(")") + 1]
    assert "keyboard_shortcuts.listener_html" in listener_call

    # ...and the listener is never handed to st.markdown.
    for match in re.finditer(r"st\.markdown\((.{0,120})", FINANCE, re.S):
        assert "listener_html" not in match.group(1)


def test_command_n_is_never_claimed():
    """Chrome reserves it for New Window; the page does not receive the
    event. Probed, not assumed. A shortcut we advertise and cannot
    deliver is worse than not having one."""
    html = kbd.listener_html()
    assert not re.search(r'key\.toLowerCase\(\)\s*===\s*"n"', html)
    for shortcut in kbd.SHORTCUTS:
        assert "⌘N" not in shortcut.keys


def test_the_replacement_binding_is_documented_as_such():
    """The overlay should say why it is not ⌘N, or the next reader
    'fixes' it back."""
    note = " ".join(s.note for s in kbd.SHORTCUTS)
    assert "⌘N" in note and "New Window" in note


def test_bare_keys_yield_to_typing_but_modifier_combos_do_not():
    """"?" is a character someone types into the ticker box."""
    html = kbd.listener_html()
    assert "isTyping" in html
    bare = html[html.index("if (!mod) {"):html.index("if (event.shiftKey")]
    assert "if (typing) return;" in bare
    # The modifier branch comes after the bare-key early return, so it is
    # reached while typing.
    assert html.index("if (typing) return;") < html.index('key === "k"')


def test_the_listener_removes_its_previous_handler():
    """Every rerun mounts a fresh iframe. Without this, one keypress
    fires once per rerun that has happened this session."""
    html = kbd.listener_html()
    assert "removeEventListener" in html
    assert html.index("removeEventListener") < html.index("addEventListener")


def test_the_listener_is_disabled_when_asked():
    assert "return;" in kbd.listener_html(enabled=False)
    assert '"enabled": false' in kbd.listener_html(enabled=False)


def test_the_tab_strip_is_found_by_label_not_position():
    """This page has several nested tab groups, so "the first .stTabs"
    would be a coin flip."""
    html = kbd.listener_html()
    assert "mainTabList" in html
    assert "CONFIG.tabs[0]" in html


def test_config_is_json_encoded_not_string_formatted():
    """A tab label with an apostrophe would otherwise break the script."""
    html = kbd.listener_html()
    payload = re.search(r"const CONFIG = (\{.*?\});", html, re.S).group(1)
    parsed = json.loads(payload)
    assert parsed["tabs"] == list(kbd.MAIN_TABS)
    assert set(parsed["triggers"]) == set(kbd.TRIGGERS)


def test_the_component_waits_for_the_parent_to_paint():
    """This iframe mounts near the top of the script; the tab strip is
    created hundreds of lines later, after every fetch this page does —
    about ten seconds on a cold load. A short retry loop expired first,
    so the palette's "Go to Risk & Technicals" closed the palette and did
    nothing. Observed live, and the reason this is an observer rather
    than a poll."""
    html = kbd.listener_html(pending_tab=1, focus_panel="kbd_palette_panel")
    assert "MutationObserver" in html
    assert "disconnect()" in html, "an observer with no ceiling runs forever"
    budget = int(re.search(r"const WAIT_MS = (\d+);", html).group(1))
    assert budget >= 15000, f"{budget}ms is shorter than a cold page load"


def test_the_pending_tab_survives_the_iframe_being_replaced():
    """Streamlit replaces the component on every rerun, and this page
    reruns constantly — its polling fragments alone see to that. The
    iframe that carried the request was torn down before the tab strip
    rendered, so the request is parked on the parent window instead.
    Verified live: the config arrived with pendingTab 4 and nothing
    happened, while clicking the same tab by hand worked."""
    html = kbd.listener_html(pending_tab=4)
    assert "window.parent[PENDING]" in html


def test_the_tab_request_is_re_applied_until_it_holds():
    """st.tabs holds no state: every rerun rebuilds the strip with the
    FIRST tab selected, and closing the palette is itself a rerun.
    Measured live — the click landed, the flag cleared, and the strip was
    back on Overview a moment later. So the request is enforced until the
    target sticks."""
    html = kbd.listener_html(pending_tab=4)
    assert "setInterval" in html and "clearInterval" in html
    assert 'aria-selected' in html, "it must verify the tab actually took"
    assert "settled" in html, "one successful check is not proof it held"


def test_the_tab_request_stands_down_if_the_reader_moves():
    """Re-applying forever would yank someone off a tab they chose."""
    html = kbd.listener_html(pending_tab=4)
    assert re.search(r"selected > 0 && selected !== pending", html)


def test_a_pending_tab_is_passed_through_and_defaults_to_null():
    """st.tabs cannot be selected from Python, so the palette's "Go to"
    commands are honoured by the component on its next mount."""
    assert '"pendingTab": 3' in kbd.listener_html(pending_tab=3)
    assert '"pendingTab": null' in kbd.listener_html()


# --- the palette --------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("risk", "Go to Risk & Technicals"),
    ("alert", "Create an alert for this ticker"),
    ("overview", "Go to Overview"),
    ("tear", "Go to CIO Tear Sheet"),
    ("shortcut", "Show keyboard shortcuts"),
])
def test_the_palette_finds_what_you_would_type(query, expected):
    assert kbd.search(query)[0].label == expected


def test_an_unmatched_query_returns_nothing_rather_than_everything():
    """Falling back to the full list would make a typo look like a
    successful search."""
    assert kbd.search("zzzznotacommand") == ()


def test_an_empty_query_lists_every_command():
    assert len(kbd.search("")) == len(kbd.commands())
    assert len(kbd.search("   ")) == len(kbd.commands())


def test_a_label_match_outranks_a_keyword_match():
    hits = kbd.search("help")
    assert hits[0].label == "Open Help & Support"


def test_every_command_is_handled_by_the_app():
    """A command that lists itself and does nothing is worse than absent."""
    for command in kbd.commands():
        if command.kind == "tab":
            continue
        assert command.id in FINANCE, f"{command.id} has no handler"


def test_every_trigger_has_a_button_and_a_handler():
    for name, key in kbd.TRIGGERS.items():
        assert f'key=_kbd_key' in FINANCE or key in FINANCE
        assert f'_kbd_name == "{name}"' in FINANCE or name == "palette", name


def test_the_new_alert_shortcut_reuses_the_empty_states_default_trigger():
    """One definition of "a sensible first alert", not two — and it still
    invents no threshold."""
    import realtime_alerts

    # Anchored on the POP, not the first mention: the flag is set in two
    # places (the trigger button and the palette) long before it is
    # consumed, and slicing from index() found a setter.
    block = FINANCE[FINANCE.index('pop("kbd_new_alert_requested"'):]
    block = block[:block.index("st.toast")]
    assert "RT_FIRST_ALERT_TRIGGER" in block
    assert "threshold=" not in block
    assert realtime_alerts.FIRST_ALERT_TRIGGER == "sma_cross_bullish"


def test_a_keystroke_that_stores_state_reports_itself():
    """⌘⇧A writes to the rule store. Doing that silently from a keypress
    is a bad surprise, so it says what was created and where to undo it."""
    block = FINANCE[FINANCE.index('pop("kbd_new_alert_requested"'):]
    block = block[:block.index("\n# ") if "\n# " in block else len(block)]
    assert "st.toast" in block
    assert "Remove it in the Real-Time Alert Engine" in block


def test_an_opened_panel_is_scrolled_to_and_focused():
    """Both panels render where the script reaches them — near the top of
    the main column, which is far below the fold once the symbol header,
    quick stats and tutorial card are on screen. Measured live: the
    palette opened 995px below the viewport, so ⌘K looked inert."""
    html = kbd.listener_html(focus_panel="kbd_palette_panel")
    assert '"focusPanel": "kbd_palette_panel"' in html
    assert "scrollIntoView" in html
    assert ".focus()" in html, "a palette you cannot type into is not a palette"
    assert '"focusPanel": null' in kbd.listener_html()


def test_the_app_asks_for_whichever_panel_just_opened():
    block = FINANCE[FINANCE.index("_kbd_focus = "):]
    block = block[:block.index("components.html(")]
    assert "kbd_palette_open" in block and "kbd_shortcuts_open" in block
    assert "kbd_palette_panel" in block and "kbd_shortcuts_panel" in block


def test_both_panels_sit_above_the_sticky_header():
    """The symbol header is sticky at z-index 100 and rendered straight
    through the palette — the quick-stats chips landed on top of its
    search box. Seen in the browser, not deduced."""
    sticky = int(re.search(r"position: sticky;\s*\n\s*top: 0;\s*\n\s*z-index: (\d+)",
                           FINANCE).group(1))
    block = FINANCE[FINANCE.index('[class*="st-key-kbd_palette_panel"]'):]
    block = block[:block.index("}")]
    panel = int(re.search(r"z-index: (\d+)", block).group(1))
    assert panel > sticky, f"palette z-index {panel} does not clear {sticky}"
    # An opaque background is part of the fix: a merely-raised panel
    # still shows the page through it.
    assert "background:" in block


def test_the_hidden_triggers_are_clipped_not_display_none():
    """A synthetic .click() must still land on them."""
    style = FINANCE[FINANCE.index('[class*="st-key-kbd_trigger_"]'):]
    style = style[:style.index("}")]
    assert "clip:" in style
    assert "display: none" not in style and "display:none" not in style
