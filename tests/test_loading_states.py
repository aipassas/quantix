"""Skeletons, the live-data pulse, and real progress.

The judgement worth pinning is WHERE a skeleton is honest. Streamlit runs
the script top to bottom, so content below the current line does not
exist yet and there is nothing to hold a place for. A skeleton only means
something where the app reserved a slot and fills it later — which this
app does in exactly two places. Everywhere else it would be decoration in
front of content that was never going to be late.
"""
import ast
import re
from pathlib import Path

import pytest

import loading_states as ls


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


# --- the reserved slots -------------------------------------------------------

@pytest.mark.parametrize("slot", ["symbol_header_container", "executive_digest_container"])
def test_a_skeleton_slot_is_empty_not_container(slot):
    """st.container APPENDS, so a skeleton written into one stays on
    screen underneath the real content forever. st.empty replaces."""
    assert f"{slot} = st.empty()" in FINANCE
    assert f"{slot} = st.container()" not in FINANCE


@pytest.mark.parametrize("slot", ["symbol_header_container", "executive_digest_container"])
def test_the_real_content_replaces_the_skeleton(slot):
    """Entering the slot with .container() swaps its contents; a bare
    `with slot:` on an st.empty would not."""
    assert f"with {slot}.container():" in FINANCE


def test_both_slots_start_as_a_skeleton():
    # The header's skeleton is drawn after the auth gate — see the
    # ordering test below for why it cannot sit beside the reservation.
    header = FINANCE.index("login_page.require_sign_in()")
    header_block = FINANCE[header:header + 900]
    assert "loading_states.skeleton(" in header_block

    digest = FINANCE.index("executive_digest_container = st.empty()")
    assert "loading_states.skeleton(" in FINANCE[digest:digest + 500]


def test_the_skeleton_is_drawn_after_the_names_it_uses_exist():
    """finance.py is a script, so name order is execution order. The slot
    is RESERVED at the top to hold its position, but _theme is not
    assigned until ~20 lines below it — drawing there raised NameError on
    first paint, which for a loading state is the only paint that
    matters. It is also drawn after the auth gate, since everything above
    that renders behind a signed-out visitor's login page."""
    reserved = FINANCE.index("symbol_header_container = st.empty()")
    theme_at = FINANCE.index("_theme = apply_brand_accent(")
    gate_at = FINANCE.index("login_page.require_sign_in()")
    drawn_at = FINANCE.index("loading_states.css(_theme)")
    assert reserved < theme_at, "the slot must still be reserved early"
    assert drawn_at > theme_at, "_theme does not exist where the skeleton is drawn"
    assert drawn_at > gate_at, "the skeleton would render behind the login page"


def test_the_skeleton_is_drawn_before_the_slow_work():
    """A skeleton that appears after the bundle loads has covered nothing."""
    skeleton_at = FINANCE.index("symbol_header_container = st.empty()")
    bundle_at = FINANCE.index("ticker_bundle = load_ticker_bundle(")
    assert skeleton_at < bundle_at


# --- what a skeleton says -----------------------------------------------------

def test_a_skeleton_names_what_is_coming():
    """A bare grey rectangle does not tell anyone whether to wait or to
    worry."""
    assert "Loading symbol" in ls.skeleton("Loading symbol")
    assert "qx-skeleton-label" in ls.skeleton("Loading symbol")


def test_a_skeleton_announces_itself_to_assistive_tech():
    markup = ls.skeleton("Loading")
    assert 'aria-busy="true"' in markup
    assert "aria-live" in markup


def test_bar_widths_are_clamped_to_something_drawable():
    markup = ls.skeleton("x", rows=(-40, 0, 500))
    widths = [int(w) for w in re.findall(r"width:(\d+)%", markup)]
    assert all(5 <= w <= 100 for w in widths), widths


def test_a_skeleton_has_as_many_bars_as_rows_asked_for():
    assert ls.skeleton("x", rows=(10, 20, 30, 40)).count("qx-skeleton-bar") == 4


# --- the pulse ----------------------------------------------------------------

def test_the_pulse_only_marks_panels_that_really_re_run():
    """Every other figure on the page is as old as the last rerun. A
    pulse elsewhere would imply a liveness this app does not have — it is
    a stateless script with no background worker."""
    uses = FINANCE.count("loading_states.pulse(")
    assert uses == 2, f"{uses} pulses; only the two timer-driven panels qualify"


def test_each_pulse_sits_in_a_fragment_with_a_timer():
    for marker in ("rechecking every", "Refreshes about every"):
        at = FINANCE.index(marker)
        preceding = FINANCE[:at]
        assert "run_every=" in preceding, marker


def test_the_pulse_carries_its_words_not_just_a_dot():
    markup = ls.pulse("rechecking every 60s")
    assert "rechecking every 60s" in markup
    assert "qx-pulse-dot" in markup


# --- motion is optional -------------------------------------------------------

def test_motion_can_be_turned_off(): 
    from theme import PALETTES

    css = ls.css(PALETTES["dark"])
    assert "prefers-reduced-motion: reduce" in css
    reduced = css[css.index("prefers-reduced-motion"):]
    assert "animation: none" in reduced


def test_the_stylesheet_is_balanced():
    from theme import PALETTES

    css = ls.css(PALETTES["dark"])
    assert css.count("{") == css.count("}")


# --- progress -----------------------------------------------------------------

def test_progress_never_reports_more_done_than_exists():
    assert ls.progress_text(99, 16) == "16 of 16"
    assert ls.progress_fraction(99, 16) == 1.0


def test_progress_survives_an_empty_universe():
    """Dividing by zero here would take down the screen it is reporting."""
    assert ls.progress_fraction(0, 0) == 0.0
    assert ls.progress_text(0, 0) == "0 of 0"


def test_progress_names_the_ticker_being_worked_on():
    assert ls.progress_text(7, 16, "NVDA") == "7 of 16 · NVDA"
    assert ls.progress_text(7, 16) == "7 of 16"


# --- the callbacks in the shared modules --------------------------------------

@pytest.mark.parametrize("module,func", [
    ("screener.py", "run_screen"),
    ("risk_alerts.py", "compute_watchlist_snapshots"),
])
def test_the_progress_parameter_is_underscored_so_caching_survives(module, func):
    """st.cache_data does not hash parameters named with a leading
    underscore. Without it, a fresh closure on every run would be a new cache key
    and the cache would never hit."""
    source = (ROOT / module).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == func)
    names = [a.arg for a in node.args.args]
    assert "_on_progress" in names, names
    assert not any(a == "on_progress" for a in names)


@pytest.mark.parametrize("module,func", [
    ("screener.py", "run_screen"),
    ("risk_alerts.py", "compute_watchlist_snapshots"),
])
def test_a_failing_progress_callback_cannot_lose_the_results(module, func):
    """The caller's drawing code breaking is not a reason to drop a
    fifteen-second screen on the floor."""
    source = (ROOT / module).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == func)
    body = ast.get_source_segment(source, node)
    assert "_on_progress(" in body
    guarded = [t for t in ast.walk(node) if isinstance(t, ast.Try)
               and "_on_progress" in ast.unparse(t)]
    assert guarded, "the callback is not wrapped in a try"


def test_both_slow_actions_show_a_bar_rather_than_a_spinner():
    for marker in ("_screener_bar", "_alert_bar"):
        assert f"{marker} = st.progress(" in FINANCE
        assert f"{marker}.empty()" in FINANCE, "the bar must be cleared when done"


def test_the_bars_are_driven_by_the_callback_not_faked():
    assert "_on_progress=_screener_progress" in FINANCE
    assert "_on_progress=_alert_progress" in FINANCE
    assert "loading_states.progress_fraction(done, total)" in FINANCE
