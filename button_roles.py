"""Which button means what, and the colours that say so.

THREE ROLES, and Streamlit gives us one and a half of them. st.button
takes type="primary" or "secondary"; there is no "danger". So primary and
secondary come from the widget, and danger is applied here by matching
the widget's key — Streamlit stamps each one onto its container as
st-key-<key>, which is what makes a third role expressible at all.

PRIMARY IS THE BRAND ACCENT, NOT STREAMLIT'S RED. Unset, Streamlit's
primary colour is #FF4B4B, and in an app where red already means a loss
that is not a neutral default — "Run Screen" and "Check Alerts" wore the
same red as a falling price. The accent comes from the palette rather
than a literal so a licensee's branding flows through; .streamlit/
config.toml carries the same value as a static baseline for the widget
families CSS cannot reach (checkbox ticks, radio dots, slider bubbles).

PRIMARY DOES TWO JOBS IN THIS APP and only one of them is a call to
action. The watchlist rows, quick-access chips and peer switcher use
type="primary" to mean "this is the ticker you are looking at", and that
state is deliberately HUELESS — red means loss, green means gain, so
selection borrows neither. Those keys are listed in SELECTION_PREFIXES and are excluded from the
accent rule by construction, not by specificity — both selectors compute
to (0,2,1), so specificity decides nothing and whichever stylesheet
happened to be injected later would win. The accent rule instead carries
a :not() for each selection prefix, which makes the two rule sets
mutually exclusive and the outcome independent of source order. Complex
:not() arguments are Selectors Level 4 and were verified against the
running page before being relied on.

DANGER IS TWO TIERS, because "remove this row" and "revoke this API key"
are not the same risk. An inline ✕ recedes and turns red only under the
pointer — it appears dozens of times and a wall of red chrome would be
noise. A named destructive action ("Delete", "Revoke", "Restore the
built-in screeners") wears a red border at rest, because the warning has
to arrive before the click, not during it.

SIGNING OUT IS NOT DESTRUCTIVE. It is navigation, and nothing is lost.
NEVER_DANGER exists so that a future reader matching on the word "logout"
does not paint it red; a test asserts the two lists never overlap.
"""
from typing import Tuple

# type="primary" here means "currently selected", not "do this". Styled
# hueless elsewhere in finance.py; listed so the accent rule and the
# tests both know these are not calls to action.
SELECTION_PREFIXES: Tuple[str, ...] = ("wl_go_", "qa_chip_", "peer_switch_")

# Inline removals: one row of a list, undone by re-adding. Dim at rest,
# red on hover.
DANGER_INLINE: Tuple[str, ...] = (
    "wl_rm_",              # remove a watchlist ticker
    "rt_remove_",          # remove a real-time alert rule
    "alert_remove_",       # remove a risk alert rule
    "screener_remove_",    # remove a screener criterion
    "etf_remove_",         # remove an ETF screener criterion
    "bond_remove_",        # remove a bond screener criterion
    "crypto_remove_",      # remove a crypto screener criterion
    "collab_rm_",          # remove a note's tag
    "collab_del_",         # delete a team note
    "scenario_delete_",    # delete a saved scenario
    "api_key_revoke_",     # revoke an issued API key
    # The strategy builder renders its condition rows from one helper
    # called twice, so its remove buttons carry the caller's prefix
    # rather than a literal one. Both are listed; the AST test resolves
    # the f-string and would flag either if it were dropped.
    "strategy_entry_remove_",
    "strategy_exit_remove_",
)

# Named destructive actions: the label already says what it does, and the
# thing it destroys is not one row. Red border at rest.
DANGER_STRONG: Tuple[str, ...] = (
    "watchlist_delete_btn",   # deletes a whole named watchlist
    "screener_tpl_del_",      # deletes a saved screener
    "screener_tpl_reset",     # discards saved screeners for the starters
    "thresholds_reset",       # discards customised thresholds
    "pf_delete",              # deletes a portfolio position
    "pf_remove",              # removes a holding
    "clear_recents",          # forgets recently-viewed
    "quick_stats_reset",      # resets the stats strip to defaults
    "notif_clear",            # wipes the persisted alert trigger history
)

DANGER_PREFIXES: Tuple[str, ...] = DANGER_INLINE + DANGER_STRONG

# Actions whose wording invites a red treatment but which destroy
# nothing. Listed so the mistake is caught by a test rather than shipped.
NEVER_DANGER: Tuple[str, ...] = (
    "auth_logout", "profile_sign_out",   # navigation; the account remains
    "reset_back", "forgot_back",         # "← Back to sign in"
    "login_to_forgot",                   # opens the reset form
    "onboarding_skip", "onboarding_back",
)

# The one red this app uses, matching the loss red already in finance.py.
DANGER_RED = "#ef4444"
DANGER_RED_DIM = "rgba(239, 68, 68, 0.55)"

# How faded an inline ✕ sits when the pointer is elsewhere. Not lower:
# the control must stay findable without a mouse, since the sidebar is a
# full-screen overlay on a phone where nothing can be hovered.
INLINE_REST_OPACITY = 0.45


def _not_selection() -> str:
    """One :not() per selection prefix.

    Written as `:not([class*="st-key-X"] *)` — a COMPLEX argument, which
    excludes any button with such an ancestor. That is what makes the
    accent rule and the hueless selection rules mutually exclusive rather
    than a specificity race they would tie.
    """
    return "".join(f':not([class*="st-key-{p}"] *)' for p in SELECTION_PREFIXES)


def _primary(scope: str, suffix: str = "") -> str:
    exclude = _not_selection()
    return ",\n    ".join(
        f'[data-testid="stMain"] button[{attr}="{value}"]{exclude}{suffix}'
        for attr, value in (("kind", "primary"),
                            ("data-testid", "stBaseButton-primary")))


def _selector(prefixes: Tuple[str, ...], suffix: str = "") -> str:
    """Two selectors per prefix, and the second one is the load-bearing one.

    finance.py paints main-area secondary buttons with

        [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]

    which is (0,2,1) and carries !important. A danger rule written as
    `[class*="st-key-X"] button` is only (0,1,1), so it LOSES that
    cascade no matter how many !importants it carries — measured on the
    running page, where every "Clear recents" style button came back with
    the ordinary grey border while the opacity (which nothing else sets)
    applied fine.

    The qualified form below is (0,3,1) and wins outright, rather than
    tying at (0,2,1) and depending on which stylesheet was injected last.
    The plain form is still emitted because destructive controls also
    live in the SIDEBAR, where finance.py deliberately leaves Streamlit's
    own chrome alone and there is nothing to out-specify.
    """
    out = []
    for prefix in prefixes:
        out.append(f'[class*="st-key-{prefix}"] button{suffix}')
        out.append(
            f'[data-testid="stMain"] [class*="st-key-{prefix}"] '
            f'button[data-testid="stBaseButton-secondary"]{suffix}')
    return ",\n    ".join(out)


def css(palette, accent: str) -> str:
    """Every button role, as one stylesheet.

    Emitted from one place so the three roles cannot drift apart the way
    they did when each panel styled its own buttons.
    """
    return f"""
    <style>
    /* --- PRIMARY: the brand accent, never Streamlit's stock red ------ */
    {_primary(".stMain")} {{
        background-color: {accent} !important;
        border: 1px solid {accent} !important;
        color: #00131A !important;
        font-weight: 700 !important;
    }}
    {_primary(".stMain", " p")} {{
        color: #00131A !important;
    }}
    {_primary(".stMain", ":hover")} {{
        filter: brightness(1.12);
        box-shadow: 0 0 0 3px {accent}33;
    }}

    /* --- DANGER, inline: recedes until you reach for it --------------- */
    {_selector(DANGER_INLINE)} {{
        opacity: {INLINE_REST_OPACITY};
    }}
    {_selector(DANGER_INLINE, ":hover")} {{
        opacity: 1;
        border-color: {DANGER_RED} !important;
    }}
    {_selector(DANGER_INLINE, ":hover p")} {{
        color: {DANGER_RED} !important;
    }}
    {_selector(DANGER_INLINE, ":focus-visible")} {{
        opacity: 1;
        border-color: {DANGER_RED} !important;
    }}

    /* --- DANGER, named: the warning arrives before the click ---------- */
    {_selector(DANGER_STRONG)} {{
        border: 1px solid {DANGER_RED_DIM} !important;
        color: {DANGER_RED} !important;
    }}
    {_selector(DANGER_STRONG, " p")} {{
        color: {DANGER_RED} !important;
    }}
    {_selector(DANGER_STRONG, ":hover")} {{
        background-color: {DANGER_RED} !important;
        border-color: {DANGER_RED} !important;
        color: #ffffff !important;
    }}
    {_selector(DANGER_STRONG, ":hover p")} {{
        color: #ffffff !important;
    }}
    </style>
    """
