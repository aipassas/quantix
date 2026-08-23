"""The micro-interaction work: the alignment cards' hover panel, and the
transitions/hover states across the app's own CSS.

The CSS tests read finance.py's source rather than a rendered page because
the acceptance criterion is "every interactive surface has a 200ms
transition" — the class of thing a hand-written list of examples stops
catching the moment someone adds a ninth sidebar panel. Three rules in
this app have already been found doing nothing at all (a dead
.streamlit-expanderHeader selector, and two out-specific'd overrides), so
a test that merely asserts "the string appears somewhere" is not enough:
these check the selector and its declaration together.
"""
import re
from pathlib import Path

import pytest

import alignment_card
from quick_stats import NOT_REPORTED


FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


@pytest.fixture(scope="module")
def source() -> str:
    return FINANCE.read_text(encoding="utf-8")


# --- the card's numbers -------------------------------------------------------

FULL = {
    "ticker": "MSFT", "score": 86.0, "status": "Strong",
    "pe": 38.24, "margin": 35.61,
    "eps": 17.93, "earnings_growth": 0.317, "revenue_growth": 0.177,
    "return_on_equity": 0.34039003, "dividend_yield_pct": 0.75,
    "market_cap": 3588320657408,
}


def test_tooltip_carries_the_figures_the_brief_named():
    labels = [label for label, _ in alignment_card.tip_rows(FULL)]
    assert any("EPS" in l for l in labels)
    assert any("growth" in l.lower() for l in labels)


def test_fraction_fields_are_rendered_as_percentages():
    """earnings_growth/ROE are FRACTIONS on StandardizedFinancials while
    dividend_yield_pct is already percent-valued. Getting these the same
    way round is the units bug this codebase keeps shipping."""
    rows = dict(alignment_card.tip_rows(FULL))
    assert rows["Earnings growth"] == "31.70%"     # 0.317, not 0.32%
    assert rows["ROE"] == "34.04%"                 # 0.3404, not 0.34%
    assert rows["Dividend yield"] == "0.75%"       # already percent — NOT 75%


def test_money_fields_are_compacted_not_raw():
    rows = dict(alignment_card.tip_rows(FULL))
    assert rows["Market cap"] == "$3.59T"
    assert rows["EPS (TTM)"] == "$17.93"


@pytest.mark.parametrize("missing", [None, float("nan"), ""])
def test_a_missing_figure_is_never_a_zero(missing):
    """A fabricated 0.00 on a card that recommends a stock is the exact
    failure this project has shipped twice."""
    data = dict(FULL)
    for key in ("eps", "earnings_growth", "revenue_growth",
                "return_on_equity", "dividend_yield_pct", "market_cap"):
        data[key] = missing
    rendered = dict(alignment_card.tip_rows(data))
    assert set(rendered.values()) == {NOT_REPORTED}
    assert "0.00" not in alignment_card.card_html(data)


def test_every_tooltip_row_is_present_even_when_absent():
    """Same shape on every card, so a gap reads as unknown rather than
    letting the reader infer a value from a shorter list."""
    assert len(alignment_card.tip_rows({})) == len(alignment_card.TIP_FIELDS)


def test_alignment_line_survives_a_missing_score():
    assert alignment_card.alignment_line(FULL) == "Strong (86%)"
    assert alignment_card.alignment_line({"status": "Weak"}) == "Weak"
    assert alignment_card.alignment_line({}) == NOT_REPORTED


# --- the card's markup --------------------------------------------------------

def test_card_is_keyboard_reachable_and_described():
    """A tooltip only a mouse can open hides these numbers from keyboard
    and touch users, which is why :focus-within is styled too."""
    html = alignment_card.card_html(FULL)
    assert 'tabindex="0"' in html
    assert 'role="tooltip"' in html
    described = re.search(r'aria-describedby="([^"]+)"', html).group(1)
    assert f'id="{described}"' in html


def test_ticker_is_escaped_into_both_body_and_id():
    """Tickers come from the user's own watchlist text box."""
    html = alignment_card.card_html({"ticker": '<img src=x onerror=alert(1)>'})
    assert "<img" not in html                    # the tag is inert text...
    assert "&lt;img src=x onerror=alert(1)&gt;" in html   # ...still shown verbatim
    ident = re.search(r'aria-describedby="([^"]+)"', html).group(1)
    assert re.fullmatch(r"qx-tip-[A-Za-z0-9_-]+", ident), ident


def test_focus_within_opens_the_panel_not_just_hover():
    style = alignment_card.css(_palette(), "#00f2fe")
    assert ".qx-acard:focus-within .qx-acard-tip" in style
    assert ".qx-acard:hover .qx-acard-tip" in style


def test_card_css_unclips_streamlits_wrappers():
    """The panel is absolutely positioned below the card, so a clipping
    column ancestor makes it invisible rather than merely misplaced."""
    style = alignment_card.css(_palette(), "#00f2fe")
    assert 'stColumn"]:has(.qx-acard)' in style
    assert "overflow: visible" in style


def _palette():
    from theme import PALETTES
    return PALETTES["dark"]


# --- the app's own hover / transition CSS -------------------------------------

def _declarations_for(source: str, selector_fragment: str):
    """Every rule body whose selector list contains the fragment.

    Plural because several of these selectors legitimately appear more
    than once — `button[kind="primary"] {{` ends both the active-chip rule
    and the blanket transition rule — and CSS cascades, so the question is
    whether ANY matching rule carries the declaration, not the first.
    """
    bodies, index = [], source.find(selector_fragment)
    assert index != -1, f"no rule mentions {selector_fragment!r}"
    while index != -1:
        open_brace = source.index("{{", index)
        bodies.append(source[open_brace:source.index("}}", open_brace)])
        index = source.find(selector_fragment, index + 1)
    return bodies


def _declaration_for(source: str, selector_fragment: str) -> str:
    """The single rule body for a selector that appears exactly once."""
    bodies = _declarations_for(source, selector_fragment)
    assert len(bodies) == 1, f"{selector_fragment!r} matched {len(bodies)} rules"
    return bodies[0]


def test_the_cards_are_no_longer_st_info(source):
    """st.info in Streamlit 1.58 takes (body, icon, width, title) — it has
    no help= and no hover surface, so the tooltip is inexpressible while
    the card is a native alert."""
    assert "alignment_card.card_html(data)" in source
    assert "Alignment: {data['status']}" not in source


def test_both_alignment_sections_got_the_card(source):
    """The task names only 'Top Diversified Market Alignments', but the
    Tech section renders the identical card — leaving one behind would be
    an obvious inconsistency."""
    assert source.count("alignment_card.card_html(data)") == 2


def test_the_extra_figures_ride_the_bundle_already_fetched(source):
    """No new network call: process_ticker_data already holds the bundle
    and was discarding these."""
    start = source.index("def process_ticker_data")
    body = source[start:source.index("\n@st.cache_data", start + 1)]
    for field in ("eps", "revenue_growth", "earnings_growth",
                  "return_on_equity", "dividend_yield_pct", "market_cap"):
        assert f'"{field}":' in body, field
    assert body.count("load_ticker_bundle(") == 1   # the call, not the comment


def test_sidebar_panels_highlight_with_more_than_a_text_colour(source):
    """A colour swap on near-black is easy to miss; the header takes a
    surface and a left rail."""
    rule = _declaration_for(source, '[data-testid="stExpander"] summary:hover,')
    assert "background-color" in rule
    assert "box-shadow: inset" in rule


def test_watchlist_remove_recedes_but_stays_reachable(source):
    """Deliberately NOT display:none / opacity:0 — the sidebar is a
    full-screen overlay on a phone, where nothing can be hovered."""
    rest = _declaration_for(source, '[class*="st-key-wl_rm_"] button {{')
    opacity = float(re.search(r"opacity:\s*([0-9.]+)", rest).group(1))
    assert 0.3 < opacity < 1.0, f"rest opacity {opacity} hides the control"


def test_hovering_the_ticker_lights_up_its_remove_button(source):
    """The chip and the ✕ are separate st.columns children, so only a
    :has() rule on the row wrapper connects them."""
    assert ':has([class*="st-key-wl_rm_"]):hover' in source


def test_no_interactive_surface_snaps(source):
    """Before this task the whole app had exactly one transition."""
    for selector in ('[data-testid="stExpander"] summary {{',
                     'button[kind="primary"] {{',
                     '.stTabs [data-baseweb="tab"] {{'):
        assert any("transition:" in body
                   for body in _declarations_for(source, selector)), selector


def test_the_blanket_button_transition_names_every_animated_property(source):
    """The blanket button rule carries !important, so it SHADOWS any
    per-widget `transition:` further down — a rule that looks correct in
    isolation and does nothing in the browser, which is the failure mode
    this codebase has hit three times.

    Measured in-browser first: the watchlist chip's computed transition
    had no `transform` and the remove button's had no `opacity`, so both
    snapped while every neighbour eased. Reading the source alone cannot
    see a cascade, so the invariant checked here is the one that makes the
    cascade safe — whatever the watchlist hover states change must be
    named in the list that actually wins.
    """
    body = next(b for b in _declarations_for(source, 'button[kind="primary"] {{')
                if "transition:" in b)
    assert "!important" in body, "the shadowing analysis below assumes this"
    # The VALUE of the transition declaration, with CSS comments stripped.
    # Checking the whole rule body would match the prose in its own
    # comment, which names every property it is meant to list.
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    blanket = re.search(r"transition:(.*?);", body, re.S).group(1)

    animated = set()
    for fragment in ('[class*="st-key-wl_go_"] button:hover {{',
                     '[class*="st-key-wl_go_"] button[kind="secondary"]:hover {{',
                     '[class*="st-key-wl_rm_"] button:hover {{'):
        for body in _declarations_for(source, fragment):
            for line in body.splitlines():
                name = line.split(":", 1)[0].strip()
                if re.fullmatch(r"[a-z-]+", name):
                    animated.add("transform" if name == "transform" else name)

    assert animated, "expected the watchlist hover rules to change something"
    for prop in animated:
        assert prop in blanket, (
            f"{prop} changes on hover but is not in the winning transition "
            f"list, so it will snap")

    # And the property the resting rule sets, which the hover reverses.
    assert "opacity" in blanket


def test_every_transition_is_200ms(source):
    """The brief specifies 200ms. Any stray 0.3s/300ms means one surface
    settles at a visibly different speed from its neighbour."""
    durations = set(re.findall(r"transition:[^;]*?(\d+(?:\.\d+)?m?s)", source))
    durations |= set(re.findall(r"transition-duration:\s*([0-9.]+m?s)", source))
    allowed = {"200ms", "1ms"}          # 1ms is the reduced-motion stub
    assert durations <= allowed, f"unexpected transition durations: {durations - allowed}"


def test_transitions_are_property_scoped_never_all(source):
    """`transition: all` animates layout and fights Plotly's own resize."""
    assert not re.search(r"transition:\s*all\b", source)


def test_hover_never_repaints_the_active_chips_hueless_border(source):
    """Selection is marked by a WHITE border because red means loss and
    green means gain. A chip-hover rule scoped to all chips out-specifies
    that and turns the current ticker cyan under the pointer."""
    body = _declaration_for(
        source, '[class*="st-key-wl_go_"] button:hover {{')
    assert "border-color" not in body, (
        "the unscoped chip hover must not set a border colour; scope it to "
        'button[kind="secondary"]')


def test_motion_can_be_turned_off(source):
    assert "prefers-reduced-motion: reduce" in source
