"""The account menu that sits at the top-right of the sticky header.

WHY A POPOVER AND NOT A DROPDOWN. Streamlit owns the page chrome and gives
an app no top bar to hang a menu off, so "top right corner" here means the
right-hand end of this app's own sticky symbol header — the one element
that is always on screen. st.popover is the only native control that opens
a panel in place; a styled <div> could look like a menu but could never
call back into Streamlit to sign anybody out.

INITIALS, NOT AN AVATAR. The brief offers "initials 'AP' or avatar image"
and only one of those is possible: AuthUser carries key, subject, issuer,
email and name — no picture claim. Streamlit's OIDC integration does not
surface one either. So initials, drawn as a real element rather than
fetched, which also means the menu renders identically for a local
password account that never had a provider profile.

EVERY ITEM DOES SOMETHING. The brief lists Settings, Preferences, Help and
Logout. Preferences and Logout act in place. Help sets a flag that opens
the sidebar's own Help & Support panel on the next run, because a popover
cannot scroll to or expand something inside the sidebar — the flag is the
only honest way to make that item navigate rather than mislead. There is
no separate Settings page in this app; its settings are the sidebar
panels, so the menu says where they are instead of offering a dead entry
that opens nothing.
"""
import logging
from typing import Optional, Tuple

import streamlit as st

import auth
from branding import brand
from logging_setup import get_logger, log_event

logger = get_logger("profile_menu")

# Set by the Help item, read by the sidebar panel on the next run.
OPEN_HELP_KEY = "_profile_open_help"

# Where the app's settings actually live, for the pointer line. Kept here
# rather than inline so the wording stays in one place if the rail changes.
SETTINGS_PANELS: Tuple[str, ...] = (
    "Branding", "Slack Alerts", "Email Digest", "API Keys",
)


def initials(name: str = "", email: str = "") -> str:
    """Up to two letters standing in for a face.

    Prefers the name's word initials ("Aggelos Passas" -> "AP"), falls
    back to the first two letters of a single word, then to the email's
    local part. Returns "?" rather than an empty string when there is
    nothing to work with: a blank circle reads as a rendering failure.
    """
    name = (name or "").strip()
    if name:
        words = [w for w in name.replace("-", " ").split() if w]
        letters = [w[0] for w in words if w[0].isalpha()]
        if len(letters) >= 2:
            return (letters[0] + letters[1]).upper()
        if letters:
            first = words[0]
            return (first[:2] if len(first) >= 2 else first).upper()

    local = (email or "").split("@")[0].strip()
    alpha = "".join(c for c in local if c.isalpha())
    if alpha:
        return alpha[:2].upper()
    return "?"


def sign_in_method() -> str:
    """How this session authenticated, phrased for the menu.

    Worth showing: the two paths behave differently on sign-out and on
    where the account lives, and someone with both a Google identity and a
    local password has no other way to tell which one they are using.
    """
    try:
        if auth.is_logged_in():
            issuer = ""
            user = auth.current_user()
            if user is not None:
                issuer = user.issuer or ""
            label = auth.provider_label(auth._issuer_slug(issuer)) if issuer else ""
            return f"Signed in with {label}" if label else "Signed in with single sign-on"
        if auth.local_user() is not None:
            return "Signed in with email and password"
    except Exception:
        pass
    return "Not signed in"


def _avatar_css(colour: str) -> str:
    return f"""
    <style>
      .qx-avatar {{
          display: inline-flex; align-items: center; justify-content: center;
          width: 34px; height: 34px; border-radius: 50%;
          background: {colour}22; border: 1.5px solid {colour};
          color: {colour}; font-weight: 700; font-size: 0.82rem;
          letter-spacing: 0.5px; font-variant-numeric: tabular-nums;
      }}
      .qx-identity {{ display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }}
      .qx-identity .qx-who {{ line-height: 1.25; min-width: 0; }}
      .qx-identity .qx-name {{ font-weight: 600; }}
      .qx-identity .qx-mail {{
          font-size: 0.85rem; opacity: 0.72;
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      }}
    </style>
    """


def render() -> None:
    """Draw the menu. Safe to call when signed out."""
    user = auth.current_user()
    accent = (brand().accent_color or "#00f2fe").strip()

    label = initials(user.name if user else "", user.email if user else "")
    with st.popover(label, width="stretch",
                    help="Account, preferences and sign out"):
        st.markdown(_avatar_css(accent), unsafe_allow_html=True)

        if user is None:
            st.caption("You are not signed in.")
            return

        display = user.display_name
        email = user.email or ""
        st.markdown(
            f'<div class="qx-identity">'
            f'<span class="qx-avatar">{label}</span>'
            f'<span class="qx-who"><div class="qx-name">{display}</div>'
            f'<div class="qx-mail">{email}</div></span></div>',
            unsafe_allow_html=True,
        )
        st.caption(sign_in_method())
        st.divider()

        # --- Preferences (acts in place) -----------------------------------
        st.caption("**Preferences**")
        # No index= alongside key=: theme_choice is already seeded from the
        # persisted file before this renders, and passing both is the
        # value=/key= conflict this codebase has been bitten by — the
        # sidebar's own theme radio carries the same note.
        from theme import PALETTES

        _chosen = st.radio(
            "Theme", list(PALETTES.keys()), key="theme_choice",
            format_func=lambda name: PALETTES[name].label, horizontal=True,
            label_visibility="collapsed",
            help="Applies to the app's chrome and charts. Exports stay dark "
                 "regardless — a document that leaves the building should not "
                 "change colour with a personal setting.",
        )
        # Persist only on an actual change, matching what the sidebar did.
        from theme import load_theme, save_theme

        if _chosen != load_theme():
            save_theme(_chosen)
            log_event(logger, logging.INFO, "user.theme_changed", theme=_chosen)
            # The palette is read at the top of the script, so the page has
            # already been painted in the OLD theme by the time this runs.
            # The sidebar control reran for the same reason; without it the
            # switch appears to do nothing until the next interaction.
            st.rerun()

        st.divider()

        # --- Help (navigates for real) -------------------------------------
        if st.button("Help & Support", key="profile_help", width="stretch"):
            # A popover cannot expand something inside the sidebar, so the
            # flag is read by that panel on the next run.
            st.session_state[OPEN_HELP_KEY] = True
            st.rerun()

        st.caption(
            "Settings live in the sidebar: " + ", ".join(SETTINGS_PANELS) + "."
        )

        st.divider()

        if st.button("Sign out", key="profile_sign_out", width="stretch"):
            # Both paths, or a password user stays signed in — st.logout()
            # only clears Streamlit's own OIDC cookie.
            import login_page

            auth.sign_out_local()
            login_page.reset_state()
            log_event(logger, logging.INFO, "profile.sign_out")
            if auth.is_logged_in():
                st.logout()
            else:
                st.rerun()


def help_requested() -> bool:
    """Whether the sidebar's Help panel should open on this run.

    Consumed (not merely read) so the panel opens once and then behaves
    normally — leaving the flag set would pin it open and make it look
    stuck.
    """
    return bool(st.session_state.pop(OPEN_HELP_KEY, False))
