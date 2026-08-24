"""Every button in the app has exactly one of three roles.

This is the "every X has a Y" shape, so it is enforced by walking the
app's own source rather than by a hand-written list of examples — a list
stops catching anything the moment someone adds the ninety-first button.

The bug being guarded is specific and was measured on the running page
before any of this was written: with no .streamlit/config.toml, Streamlit's
primary colour is its stock #FF4B4B, and eighteen separate elements were
painted with it — "Run Screen", "Check Alerts", the active tab's
underline, checkbox ticks, radio dots, slider bubbles and multiselect
tags. In an app where red already means a loss, that is not a neutral
default.
"""
import ast
import pathlib
import re
import tomllib

import pytest

import button_roles as roles


ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_FILES = ("finance.py", "login_page.py", "profile_menu.py", "empty_states.py")
STREAMLIT_STOCK_RED = "#ff4b4b"


def _const(node):
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        return "".join(p.value if isinstance(p, ast.Constant) else "{}"
                       for p in p_.values for p_ in [node])
    return None


def _text(node):
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        return "".join(p.value if isinstance(p, ast.Constant) else "{}"
                       for p in node.values)
    return "<expr>"


def buttons():
    """(file, line, label, key, type) for every button the app renders."""
    found = []
    for name in APP_FILES:
        path = ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("button", "form_submit_button"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            found.append({
                "file": name, "line": node.lineno,
                "label": _text(node.args[0]) if node.args else _text(kw.get("label")),
                "key": _text(kw.get("key")),
                "type": _text(kw.get("type")),
            })
    return found


ALL_BUTTONS = buttons()


def _pattern(key: str) -> re.Pattern:
    """A key as read from the source, as a regex.

    Keys come out of the AST with `{}` where an f-string interpolated
    something — "wl_rm_{}", and for the strategy builder "{}_remove_{}",
    whose PREFIX is itself a variable. Turning the placeholders into `.*`
    lets one rule answer "does this danger prefix cover this key" for
    literal, suffixed and dynamically-prefixed keys alike.
    """
    return re.compile(".*".join(re.escape(part) for part in key.split("{}")))


def covered(key: str, prefixes) -> bool:
    # An empty key matches nothing. Without this guard _pattern("") is an
    # empty regex, which matches at position 0 of EVERY prefix and would
    # report every keyless button as covered by every role.
    if not key:
        return False
    rx = _pattern(key)
    return any(rx.match(p) for p in prefixes)


def test_the_coverage_helper_does_not_match_a_keyless_button():
    """An empty regex matches at position 0 of anything, so without a
    guard every keyless button would look covered by every role — and
    "Run Screen" was duly reported as destructive."""
    assert not covered("", roles.DANGER_PREFIXES)
    assert not covered(None, roles.DANGER_PREFIXES)
    assert covered("wl_rm_{}", roles.DANGER_PREFIXES)
    assert covered("{}_remove_{}", roles.DANGER_PREFIXES)
    assert not covered("support_send", roles.DANGER_PREFIXES)


def test_the_scan_actually_found_the_app_s_buttons():
    """A guard on the guard: a broken walk would make everything below
    pass vacuously."""
    assert len(ALL_BUTTONS) > 70, len(ALL_BUTTONS)
    labels = {b["label"] for b in ALL_BUTTONS}
    for expected in ("Run Screen", "Check Alerts", "Sign in"):
        assert expected in labels, expected


# --- the root cause -----------------------------------------------------------

def test_streamlit_has_a_primary_colour_configured():
    """Without this file every widget built from the primary accent —
    not just buttons — falls back to #FF4B4B."""
    config = ROOT / ".streamlit" / "config.toml"
    assert config.exists(), "no .streamlit/config.toml; primary falls back to red"
    theme = tomllib.loads(config.read_text(encoding="utf-8")).get("theme", {})
    assert "primaryColor" in theme
    assert theme["primaryColor"].lower() != STREAMLIT_STOCK_RED


def test_the_configured_primary_is_the_verified_brand_colour():
    import brand_assets

    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml")
                          .read_text(encoding="utf-8"))["theme"]
    assert theme["primaryColor"].lower() == brand_assets.BRAND_CYAN.lower()


def test_the_config_does_not_fight_the_apps_own_theme():
    """The app has a light/dark toggle applied as CSS at runtime, and the
    sidebar deliberately keeps Streamlit's light chrome — finance.py
    scopes its button restyling to stMain for exactly that reason. A
    static base=/backgroundColor here would override both."""
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml")
                          .read_text(encoding="utf-8"))["theme"]
    for forbidden in ("base", "backgroundColor", "secondaryBackgroundColor",
                      "textColor"):
        assert forbidden not in theme, f"{forbidden} would fight theme.PALETTES"


def test_no_source_file_hard_codes_streamlits_stock_red():
    """#FF4B4B appears in finance.py's comments as the colour that was
    REMOVED; it must not appear in a declaration."""
    for name in APP_FILES + ("button_roles.py",):
        text = (ROOT / name).read_text(encoding="utf-8")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))
        for line in text.splitlines():
            if ":" in line and STREAMLIT_STOCK_RED in line.lower():
                pytest.fail(f"{name}: stock red in a declaration: {line.strip()}")


# --- the three roles ----------------------------------------------------------

def test_primary_is_painted_with_the_branded_accent():
    css = roles.css(None, "#00f2fe")
    assert "#00f2fe" in css
    assert STREAMLIT_STOCK_RED not in css.lower()


def test_the_hueless_selection_chips_are_excluded_by_construction():
    """Both selectors compute to (0,2,1), so specificity settles nothing
    and whichever stylesheet loaded last would win. The accent rule
    excludes them with :not() instead, which is order-independent."""
    css = roles.css(None, "#00f2fe")
    primary_rule = css[css.index("PRIMARY"):css.index("DANGER, inline")]
    for prefix in roles.SELECTION_PREFIXES:
        assert f':not([class*="st-key-{prefix}"] *)' in primary_rule, prefix


def test_selection_prefixes_match_the_ones_finance_actually_styles():
    """Two lists of the same three keys would drift."""
    source = (ROOT / "finance.py").read_text(encoding="utf-8")
    for prefix in roles.SELECTION_PREFIXES:
        assert f'[class*="st-key-{prefix}"] button[kind="primary"]' in source, prefix


def test_danger_and_never_danger_never_overlap():
    """Signing out reads destructive and destroys nothing."""
    assert not (set(roles.DANGER_PREFIXES) & set(roles.NEVER_DANGER))


def test_no_navigation_action_is_marked_dangerous():
    for key in ("auth_logout", "profile_sign_out", "reset_back"):
        assert key in roles.NEVER_DANGER
        assert not any(key.startswith(p) for p in roles.DANGER_PREFIXES)


def test_every_danger_key_belongs_to_a_button_that_exists():
    """A prefix matching nothing is dead styling that reads as coverage."""
    keys = [b["key"] for b in ALL_BUTTONS if b["key"]]
    for prefix in roles.DANGER_PREFIXES:
        assert any(_pattern(k).match(prefix) for k in keys), (
            f"{prefix} matches no button in the app")


def test_every_inline_remove_button_is_marked_dangerous():
    """The ✕ controls are the ones a misclick actually costs something."""
    unmarked = []
    for b in ALL_BUTTONS:
        if b["label"] != "✕" or not b["key"]:
            continue
        if not covered(b["key"], roles.DANGER_PREFIXES):
            unmarked.append(f'{b["file"]}:{b["line"]} key={b["key"]}')
    assert not unmarked, "✕ buttons with no danger role: " + ", ".join(unmarked)


def test_an_inline_remove_stays_visible_without_a_pointer():
    """display:none / opacity:0 would make it unreachable on touch, and
    the sidebar is a full-screen overlay on a phone."""
    assert 0.3 < roles.INLINE_REST_OPACITY < 1.0


def test_danger_rules_out_specify_the_main_secondary_rule():
    """finance.py paints main-area secondaries at (0,2,1) with
    !important. A danger rule at (0,1,1) loses that cascade however many
    !importants it carries — measured live, where every strong-danger
    button came back with the ordinary grey border while its opacity
    (which nothing else sets) applied fine."""
    css = roles.css(None, "#00f2fe")
    for prefix in roles.DANGER_PREFIXES:
        qualified = (f'[data-testid="stMain"] [class*="st-key-{prefix}"] '
                     f'button[data-testid="stBaseButton-secondary"]')
        assert qualified in css, f"{prefix} has no rule that beats stMain secondary"
        # ...and the unqualified form too, for the sidebar, where
        # finance.py leaves Streamlit's chrome alone.
        assert f'[class*="st-key-{prefix}"] button' in css


def test_the_two_danger_tiers_differ_at_rest():
    """An inline ✕ recedes; a named "Delete" warns before the click."""
    css = roles.css(None, "#00f2fe")
    inline = css[css.index("DANGER, inline"):css.index("DANGER, named")]
    strong = css[css.index("DANGER, named"):]
    assert "opacity" in inline
    assert roles.DANGER_RED in strong and "border" in strong


def test_the_danger_treatment_lives_in_exactly_one_place(source_finance=None):
    """The watchlist ✕ was styled inline before this module existed. Two
    copies of "how a destructive button looks" is how the watchlist's
    remove and the alert rule's drift apart."""
    text = (ROOT / "finance.py").read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)      # strip comments
    rules = [b for b in text.split("}}") if "st-key-wl_rm_" in b]
    for rule in rules:
        declared = {d.split(":", 1)[0].strip()
                    for d in rule.split("{{")[-1].split(";") if ":" in d}
        assert "border-color" not in declared, (
            "the ✕'s red hover belongs to button_roles, not finance.py")
    # The one genuinely row-specific rule may stay: hovering the ticker
    # lights up its own remove button.
    assert ':has([class*="st-key-wl_rm_"]):hover' in text


def test_the_danger_red_is_the_one_already_used_in_the_app():
    source = (ROOT / "finance.py").read_text(encoding="utf-8")
    assert roles.DANGER_RED in source


# --- coverage over the whole app ----------------------------------------------

def test_every_primary_button_is_a_call_to_action_or_a_selection():
    """type="primary" does two jobs in this app. Each use must be one of
    them — a third meaning would need a third colour nobody defined."""
    for b in ALL_BUTTONS:
        if b["type"] != "primary":
            continue
        key = b["key"] or ""
        is_selection = covered(key, roles.SELECTION_PREFIXES)
        is_danger = covered(key, roles.DANGER_PREFIXES)
        assert not is_danger, (
            f'{b["file"]}:{b["line"]} is both primary and danger — pick one')
        assert not is_selection, (
            f'{b["file"]}:{b["line"]} uses a literal primary for a selection '
            "surface; those switch type conditionally")


def test_no_destructive_button_is_left_as_a_plain_secondary():
    """The point of the task: a destructive action must not look like an
    ordinary one. Checked against the labels, not just the key list, so a
    NEW delete button is caught rather than silently uncovered."""
    destructive_label = re.compile(r"^(delete|remove|revoke|clear|✕)\b", re.I)
    missed = []
    for b in ALL_BUTTONS:
        label = (b["label"] or "").strip()
        if not destructive_label.match(label):
            continue
        key = b["key"] or ""
        if covered(key, roles.DANGER_PREFIXES):
            continue
        if key in roles.NEVER_DANGER:
            continue
        missed.append(f'{b["file"]}:{b["line"]} {label!r} key={key or "None"}')
    assert not missed, "destructive buttons with no danger role: " + "; ".join(missed)
