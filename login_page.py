"""The classic-layout login page: header bar, value-prop column, sign-in panel.

WHY THIS IS COLUMNS AND NOT A SIDEBAR. Streamlit's sidebar is left-only
and cannot be moved, so the brief's "right sidebar" is built as a
two-column layout whose right column is a bordered panel. It reads as a
classic site login and, unlike a real sidebar, it stacks correctly on a
phone — the form comes FIRST on narrow screens, because someone who
opened this page on a phone came to sign in, not to read the pitch.

THE PAGE SHOWS ONLY WHAT IS REAL. Three things the brief asked for are
deliberately absent, and their absence is the honest answer rather than
an oversight:

  * No testimonials. Writing quotes from customers who have not said them
    is fabricating evidence, and it is the kind of thing that reads as
    fine on a mockup and is indefensible in production.
  * No security or compliance badges. Quantix holds no SOC 2, ISO or
    PCI attestation, and a badge asserting one would be a lie told to
    exactly the people least able to check it. The trust column instead
    states verifiable facts about how the app actually works.
  * No Pricing nav item. There are no paid tiers yet (the freemium task
    is still open), so the link would lead nowhere or invent a price.

"REMEMBER ME" IS RENDERED DISABLED, ON PURPOSE. Streamlit exposes
st.context.cookies as READ-ONLY and offers no API to set one, so an app
cannot issue a persistent session cookie. The options were to omit the
control, to fake it, or to show it disabled with the reason. Faking it is
the worst of the three: a checkbox labelled "keep me signed in" that
silently does nothing teaches people to trust a security control that
isn't there. Putting a session token in the URL instead was rejected —
query strings leak through history, referrers and logs.

EVERY FAILURE MESSAGE COMES FROM accounts.py, unchanged. The wording
there is deliberately identical for "no such account" and "wrong
password"; rephrasing per-case in the UI would undo the enumeration
resistance the storage layer went to trouble to provide.
"""
import logging
from typing import Optional

import streamlit as st

import accounts
import auth
import passwords
from branding import brand
from logging_setup import get_logger, log_event

logger = get_logger("login_page")

_MODE_KEY = "_login_mode"
_ERROR_KEY = "_login_error"
_NOTICE_KEY = "_login_notice"
_RESET_EMAIL_KEY = "_login_reset_email"

# The brief asks for electric cyan; branding.py asks that a licensee's
# colour win wherever one is set. Both hold: this is the default, and a
# configured accent replaces it.
_DEFAULT_ACCENT = "#22D3EE"


def accent() -> str:
    configured = ""
    try:
        configured = (brand().accent_color or "").strip()
    except Exception:
        pass
    return configured or _DEFAULT_ACCENT


def _set_mode(mode: str) -> None:
    st.session_state[_MODE_KEY] = mode
    st.session_state.pop(_ERROR_KEY, None)
    st.session_state.pop(_NOTICE_KEY, None)


def _mode() -> str:
    return st.session_state.get(_MODE_KEY, "signin")


def _fail(message: str) -> None:
    st.session_state[_ERROR_KEY] = message


def _notify(message: str) -> None:
    st.session_state[_NOTICE_KEY] = message


# --- presentation -------------------------------------------------------------

def _inject_css() -> None:
    colour = accent()
    st.markdown(f"""
    <style>
      /* The login page owns the whole viewport: Streamlit's chrome, menu
         and (empty) sidebar would otherwise frame a page that is not part
         of the app yet. */
      [data-testid="stSidebar"], [data-testid="stToolbar"],
      [data-testid="stDecoration"], header[data-testid="stHeader"] {{
          display: none !important;
      }}
      /* The app's own CSS hides Streamlit's chrome with a bare
         `header {{visibility: hidden}}` selector (finance.py). That rule
         matches ANY <header>, including this page's — which is semantic
         markup a screen reader wants, so the fix is to exempt it rather
         than to downgrade it to a <div>. */
      header.qx-header {{ visibility: visible !important; }}

      .stApp {{ background: #000000; }}
      .block-container {{ padding-top: 1.2rem; max-width: 1240px; }}

      .qx-header {{
          display: flex; align-items: center; justify-content: space-between;
          gap: 24px; padding: 18px 4px 22px 4px;
          border-bottom: 1px solid #1C2029; margin-bottom: 34px;
      }}
      /* align-items: center, not baseline — images have no baseline, so
         baseline alignment drops them relative to the nav. */
      .qx-brand {{ display: flex; align-items: center; gap: 14px; }}
      /* Sized off the wordmark: the mark is square and the wordmark is
         roughly 5.6:1, so matching their heights keeps the lockup in the
         proportions the artwork was drawn in. */
      .qx-brand .qx-mark-img {{ height: 44px; width: auto; display: block; }}
      .qx-brand .qx-word-img {{ height: 30px; width: auto; display: block; }}
      .qx-brand .qx-name {{
          font-size: 2.15rem; font-weight: 700; color: #FFFFFF;
          letter-spacing: -0.8px;
      }}
      .qx-brand .qx-mark {{
          color: {colour}; font-size: 1.85rem; font-weight: 700;
      }}
      .qx-nav {{ display: flex; gap: 26px; }}
      .qx-nav a {{
          color: #9CA3AF; text-decoration: none; font-size: 1rem;
          font-weight: 500; padding: 4px 2px;
          border-bottom: 2px solid transparent;
      }}
      .qx-nav a:hover {{ color: #FFFFFF; border-bottom-color: {colour}; }}
      .qx-nav a:focus-visible, .qx-skip:focus-visible {{
          outline: 2px solid {colour}; outline-offset: 3px; border-radius: 3px;
      }}

      /* Keyboard users get to the form without tabbing the whole pitch. */
      .qx-skip {{
          position: absolute; left: -9999px; top: 0;
          background: {colour}; color: #000; padding: 10px 16px;
          border-radius: 0 0 8px 0; font-weight: 700; z-index: 999;
      }}
      .qx-skip:focus {{ left: 0; }}

      .qx-eyebrow {{
          color: {colour}; font-size: 0.86rem; font-weight: 700;
          letter-spacing: 2.4px; text-transform: uppercase; margin-bottom: 16px;
      }}
      .qx-hero h1 {{
          color: #FFFFFF; font-size: 3.05rem; line-height: 1.1;
          letter-spacing: -1.2px; margin: 0 0 18px 0; font-weight: 700;
      }}
      .qx-hero p.qx-lede {{
          color: #9CA3AF; font-size: 1.2rem; line-height: 1.62;
          margin: 0 0 34px 0; max-width: 60ch;
      }}
      .qx-feature {{ display: flex; gap: 14px; margin-bottom: 20px; }}
      .qx-feature .qx-tick {{
          color: {colour}; font-weight: 700; line-height: 1.5; flex: 0 0 auto;
      }}
      .qx-feature .qx-body {{ color: #CBD5E1; font-size: 1.06rem; line-height: 1.6; }}
      .qx-feature .qx-body strong {{ color: #FFFFFF; font-weight: 600; }}

      .qx-trust {{
          border-top: 1px solid #1C2029; margin-top: 30px; padding-top: 22px;
      }}
      .qx-trust h2 {{
          color: #7C8595; font-size: 0.84rem; letter-spacing: 1.8px;
          text-transform: uppercase; margin: 0 0 16px 0; font-weight: 700;
      }}
      .qx-trust li {{
          color: #9CA3AF; font-size: 0.98rem; line-height: 1.62;
          margin-bottom: 10px; list-style: none;
      }}
      .qx-trust li::before {{ content: "— "; color: {colour}; }}

      .qx-panel-title {{
          color: #FFFFFF; font-size: 1.55rem; font-weight: 700; margin: 0 0 8px 0;
      }}
      /* Streamlit's own `.stMarkdown p` rule outranks a bare class here,
         so these need the container in the selector or they silently do
         nothing — which is exactly what happened to the first attempt. */
      .stMarkdown p.qx-panel-sub {{
          color: #9CA3AF; font-size: 1.02rem; margin: 0 0 6px 0;
      }}

      /* The sign-in panel. A bordered surface rather than a floating card,
         which is what makes it read as a right-hand column. */
      div[data-testid="stVerticalBlockBorderWrapper"]:has(.qx-panel-title) {{
          background: #0A0A0A; border: 1px solid #1C2029;
          border-radius: 14px; padding: 6px 4px;
      }}

      /* Streamlit's form controls default to 14px in a 38px-tall box.
         That is fine buried in a settings panel and too small for the
         primary action on a landing page, so the whole control scales up
         rather than only its label. */
      .stTextInput input {{
          background: #050505 !important; color: #FFFFFF !important;
          border: 1px solid #262B36 !important; border-radius: 9px !important;
          font-size: 1.02rem !important;
      }}
      .stTextInput div[data-baseweb="input"],
      .stTextInput div[data-baseweb="base-input"] {{
          min-height: 2.9rem !important;
      }}
      .stTextInput label, .stTextInput label p,
      .stCheckbox label, .stCheckbox label p {{
          font-size: 1rem !important;
      }}
      div[data-testid="stForm"] button p,
      div[data-testid="stForm"] button {{
          font-size: 1.05rem !important;
      }}
      div[data-testid="stForm"] button {{
          padding-top: 0.6rem !important; padding-bottom: 0.6rem !important;
      }}
      .stButton button p {{ font-size: 1rem !important; }}
      .stTextInput input:focus {{
          border-color: {colour} !important;
          box-shadow: 0 0 0 3px {colour}33 !important;
      }}
      .stTextInput label, .stCheckbox label {{ color: #CBD5E1 !important; }}

      /* Every leading action on this page wears the brand accent.
         This used to be scoped to stForm, which was fine while the only
         primary button WAS a form submit — but "Continue with Google"
         sits outside any form, so it fell through to Streamlit's stock
         #FF4B4B and rendered bright red directly above a cyan "Sign in".
         Red is also spoken for in this app: it means a loss. A page-wide
         selector is safe here because this stylesheet is only injected
         by the signed-out login page, which then calls st.stop(). */
      div[data-testid="stForm"] .stButton button,
      div[data-testid="stForm"] button[kind="primaryFormSubmit"],
      button[kind="primary"],
      button[data-testid="stBaseButton-primary"] {{
          background: {colour} !important; color: #00131A !important;
          border: 0 !important; font-weight: 700 !important;
          border-radius: 9px !important;
          transition: filter 200ms ease, box-shadow 200ms ease;
      }}
      div[data-testid="stForm"] .stButton button:hover,
      div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
      button[kind="primary"]:hover,
      button[data-testid="stBaseButton-primary"]:hover {{
          filter: brightness(1.12);
          box-shadow: 0 0 0 3px {colour}33;
      }}
      @media (prefers-reduced-motion: reduce) {{
          button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
              transition-duration: 1ms;
          }}
      }}
      button:focus-visible {{ outline: 2px solid {colour} !important; outline-offset: 2px; }}

      /* #7C8595 rather than a dimmer grey: measured 5.65:1 on black,
         where the #6B7280 this started as came out at 4.34 and missed
         WCAG AA's 4.5 for body text. The disclosure is the last text
         that should be hard to read. */
      /* 1rem, not the 0.88rem this started at. Until the selector above
         was corrected, Streamlit's default was winning and rendering this
         at 16px — so "raising" it to 0.88rem would have shrunk the one
         paragraph on the page that is a disclosure. */
      .stMarkdown p.qx-legal {{
          color: #7C8595; font-size: 1rem; line-height: 1.6;
          border-top: 1px solid #1C2029; margin-top: 26px; padding-top: 16px;
      }}

      @media (max-width: 900px) {{
          .qx-nav {{ display: none; }}
          .qx-brand .qx-name {{ font-size: 1.7rem; }}
          .qx-brand .qx-mark {{ font-size: 1.5rem; }}
          .qx-hero h1 {{ font-size: 2.2rem; }}
          .block-container {{ padding-top: 0.6rem; }}

          /* Streamlit stacks columns in DOM order, which puts the whole
             pitch above the form on a phone — someone who opened this on
             their phone came to sign in, not to read four feature
             bullets first. Reversing the ONE row that holds the hero, via
             :has(), leaves the small button rows inside the panel in
             their intended order. */
          [data-testid="stHorizontalBlock"]:has(.qx-hero) {{
              flex-direction: column-reverse;
          }}
          /* Reversing the row also opts out of Streamlit's own stacking
             rules, which are what normally widen a stacked column — so the
             columns kept their desktop ratio and the panel's contents
             spilled outside its border. Restore full width explicitly. */
          [data-testid="stHorizontalBlock"]:has(.qx-hero) > div {{
              width: 100% !important;
              flex: 1 1 100% !important;
              min-width: 0 !important;
          }}
      }}

      /* --- The "or sign in with email" rule --------------------------
         A labelled separator rather than st.divider(), because the label
         is the whole point: it tells someone who skipped the SSO buttons
         what the thing below them is. */
      .stMarkdown .qx-or {{
          display: flex; align-items: center; gap: 12px;
          margin: 18px 0 6px;
          font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase;
          color: #9CA3AF;
      }}
      .stMarkdown .qx-or::before, .stMarkdown .qx-or::after {{
          content: ""; flex: 1 1 auto; height: 1px; background: #1C2029;
      }}

      /* --- The password visibility toggle ----------------------------
         Streamlit's own control, and it gives us nothing stable to grab:
         the button carries no data-testid and only emotion-hash classes
         (st-af st-c0 ...) that change between builds. Inspected on the
         running page — its one dependable property is being the direct
         child <button> of the base-input wrapper inside a text input, so
         that is what this selects. Deliberately NOT keyed on
         aria-label="Show password text": that string flips to "Hide ..."
         on toggle and would be localised.

         It was already white, but flat — no border, no hover, no cue that
         it was a control rather than a decorative glyph. Now it reads as
         a button, takes the brand accent, and meets a 44px tap target. */
      [data-testid="stTextInputRootElement"] > div[data-baseweb="base-input"] > button {{
          min-width: 44px; min-height: 44px;
          border-radius: 6px;
          border: 1px solid #1C2029;
          background: rgba(255, 255, 255, 0.04);
          color: {colour};
          transition: background-color 200ms ease, border-color 200ms ease,
                      color 200ms ease;
      }}
      [data-testid="stTextInputRootElement"] > div[data-baseweb="base-input"] > button svg {{
          fill: currentColor;
          width: 20px; height: 20px;
      }}
      [data-testid="stTextInputRootElement"] > div[data-baseweb="base-input"] > button:hover,
      [data-testid="stTextInputRootElement"] > div[data-baseweb="base-input"] > button:focus-visible {{
          background: rgba(255, 255, 255, 0.10);
          border-color: {colour};
      }}
      @media (prefers-reduced-motion: reduce) {{
          [data-testid="stTextInputRootElement"] > div[data-baseweb="base-input"] > button {{
              transition-duration: 1ms;
          }}
      }}
    </style>
    """, unsafe_allow_html=True)


def _header() -> None:
    name = brand().name

    # The real artwork, not a typographic stand-in: the Q mark followed by
    # the QUANTIX wordmark. Both ship as black-or-cyan on a white card, so
    # brand_assets keys the card out and repaints the wordmark white for
    # this dark header — the shipped wordmark is black and would otherwise
    # be invisible here.
    #
    # Inlined as data URIs because this header is raw HTML through
    # st.markdown, where a filesystem path resolves to nothing. If either
    # asset is missing the text lockup renders instead, so the page never
    # loses its name.
    import brand_assets

    _mark_uri = brand_assets.mark_data_uri()
    _word_uri = brand_assets.wordmark_data_uri("#FFFFFF")
    if _mark_uri and _word_uri:
        brand_block = (
            f'<img class="qx-mark-img" src="{_mark_uri}" alt="">'
            f'<img class="qx-word-img" src="{_word_uri}" alt="{name}">'
        )
    else:
        brand_block = f'<span class="qx-name">{name}</span>'

    st.markdown(f"""
    <a class="qx-skip" href="#qx-signin">Skip to sign in</a>
    <header class="qx-header">
      <div class="qx-brand">
        {brand_block}
      </div>
      <nav class="qx-nav" aria-label="Primary">
        <a href="#qx-features">Features</a>
        <a href="#qx-security">Security</a>
        <a href="#qx-signin">Sign in</a>
      </nav>
    </header>
    """, unsafe_allow_html=True)


def _value_column() -> None:
    tagline = ""
    try:
        tagline = brand().tagline or ""
    except Exception:
        pass

    st.markdown(f"""
    <div class="qx-hero">
      <div class="qx-eyebrow">{tagline or "Institutional-grade equity research"}</div>
      <h1>The analysis, and<br/>what it can't tell you.</h1>
      <p class="qx-lede">
        Discounted cash flow, an eight-point fundamental scorecard, Altman Z, Monte
        Carlo and walk-forward backtests — with every figure it could not compute
        shown as unavailable rather than quietly reported as zero.
      </p>
    </div>

    <div id="qx-features">
      <div class="qx-feature"><span class="qx-tick">▸</span><span class="qx-body">
        <strong>Valuation you can audit.</strong> Multi-stage DCF with a real
        revenue and margin forecast, CAPM beta regressed from actual returns, and a
        sensitivity grid over growth and WACC.</span></div>
      <div class="qx-feature"><span class="qx-tick">▸</span><span class="qx-body">
        <strong>Risk that survives contact.</strong> Historical VaR and expected
        shortfall, maximum drawdown, fat-tailed Monte Carlo, and backtests costed
        for slippage and commission.</span></div>
      <div class="qx-feature"><span class="qx-tick">▸</span><span class="qx-body">
        <strong>Exports that stay editable.</strong> Native PowerPoint and Excel,
        plus a print-ready tear sheet — numbers as numbers, not pictures of
        numbers.</span></div>
      <div class="qx-feature"><span class="qx-tick">▸</span><span class="qx-body">
        <strong>Your own workspace.</strong> Watchlists, thresholds, alert rules and
        notes are scoped to your account rather than shared with everyone using the
        instance.</span></div>
    </div>

    <div class="qx-trust" id="qx-security">
      <h2>How this instance handles your data</h2>
      <ul>
        <li>Passwords are stored as salted scrypt hashes. The plain password is
            never written to disk or to a log.</li>
        <li>Repeated failed sign-ins lock the account for a growing interval, so a
            password cannot be guessed against this form at volume.</li>
        <li>Analysis runs on this machine against public market data. Your
            watchlists and notes are files on this instance, not a vendor's cloud.</li>
        <li>No certification is claimed. Quantix holds no SOC 2, ISO or PCI
            attestation, and these are statements about the code, not an audit.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)


# --- the auth panel -----------------------------------------------------------

def _oauth_block() -> bool:
    """Single sign-on, offered FIRST. Returns whether anything was drawn.

    The caller needs the return value to decide whether to draw a divider:
    an instance with no provider configured would otherwise get a rule
    labelled "or sign in with email" separating the form from nothing.

    The buttons are `primary` now that they lead. On this instance only
    Google is configured, but the list is data-driven — a GitHub or
    Microsoft provider added to secrets.toml appears here with no code
    change, which is why the task's three names are not hard-coded.
    """
    providers = auth.configured_providers()
    if not providers:
        if auth.unavailable_reason():
            st.caption(
                "Single sign-on isn't configured on this instance, so email and "
                "password is the way in.")
        return False

    for provider in providers:
        label = auth.provider_label(provider)
        if st.button(f"Continue with {label}", type="primary",
                     key=f"login_oauth_{provider or 'default'}", width="stretch"):
            # st.login() navigates away, and the round trip to the provider
            # is the one genuinely slow step in this whole page — unlike
            # the local password check, which measures ~140ms. Without this
            # the button looks inert for long enough to be clicked twice.
            with st.spinner(f"Redirecting to {label}..."):
                st.login(provider) if provider else st.login()
    return True


def _signin_heading() -> None:
    """The panel's title. Separate from _signin_form because single
    sign-on now sits between them."""
    st.markdown('<div class="qx-panel-title">Sign in</div>'
                '<p class="qx-panel-sub">Welcome back.</p>', unsafe_allow_html=True)


def _signin_form() -> None:
    with st.form("login_signin", clear_on_submit=False):
        email = st.text_input("Email", key="login_email",
                              placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="login_password")

        left, right = st.columns([1, 1])
        with left:
            # Disabled deliberately — see the module docstring. Streamlit
            # gives an app no way to set a cookie, so this cannot work, and
            # a control that silently does nothing is worse than an honest
            # one that says so.
            st.checkbox("Remember me", value=False, disabled=True,
                        key="login_remember",
                        help="Unavailable: Streamlit does not let an app set a "
                             "browser cookie, so a session cannot outlive a page "
                             "reload. Reloading asks you to sign in again.")
        submitted = st.form_submit_button("Sign in", width="stretch", type="primary")

    if submitted:
        with st.spinner("Checking your details..."):
            account, error = accounts.authenticate(email, password)
        if account is None:
            _fail(error or "That email or password isn't right.")
            st.rerun()
        auth.sign_in_local(account)
        log_event(logger, logging.INFO, "login.local_success")
        st.rerun()

    small_left, small_right = st.columns([1, 1])
    with small_left:
        if st.button("Forgot password?", key="login_to_forgot", width="stretch"):
            _set_mode("forgot")
            st.rerun()
    with small_right:
        if st.button("Create an account", key="login_to_signup", width="stretch"):
            _set_mode("signup")
            st.rerun()


def _signup_form() -> None:
    st.markdown('<div class="qx-panel-title">Create your account</div>'
                '<p class="qx-panel-sub">Your workspace stays private to you.</p>',
                unsafe_allow_html=True)

    with st.form("login_signup", clear_on_submit=False):
        name = st.text_input("Name", key="signup_name",
                             placeholder="How your notes should be signed")
        email = st.text_input("Email", key="signup_email",
                              placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm = st.text_input("Confirm password", type="password",
                                key="signup_confirm")
        st.caption(
            f"At least {passwords.MIN_LENGTH} characters. A long passphrase of "
            "ordinary words is stronger than a short one full of symbols.")
        submitted = st.form_submit_button("Create account", width="stretch",
                                          type="primary")

    if submitted:
        if password != confirm:
            _fail("Those two passwords don't match.")
            st.rerun()
        with st.spinner("Creating your account..."):
            account, error = accounts.create_account(email, password, name)
        if account is None:
            _fail(error or "Couldn't create that account.")
            st.rerun()
        auth.sign_in_local(account)
        log_event(logger, logging.INFO, "login.account_created")
        st.rerun()

    if st.button("← Back to sign in", key="signup_back", width="stretch"):
        _set_mode("signin")
        st.rerun()


def _forgot_form() -> None:
    st.markdown('<div class="qx-panel-title">Reset your password</div>'
                '<p class="qx-panel-sub">We\'ll send a single-use link.</p>',
                unsafe_allow_html=True)

    with st.form("login_forgot", clear_on_submit=False):
        email = st.text_input("Email", key="forgot_email",
                              placeholder="you@example.com")
        submitted = st.form_submit_button("Send reset link", width="stretch",
                                          type="primary")

    if submitted:
        # The only step in this page that talks to a remote server: an SMTP
        # send can take seconds, against ~140ms for the local password
        # check. This was the submit with no feedback of any kind.
        with st.spinner("Sending your reset link..."):
            issued = accounts.begin_reset(email)
            delivered_note = ""
            if issued is not None:
                account, token = issued
                sent, problem = _send_reset_email(account, token)
                if not sent:
                    # The instance has no mail configured. Say so plainly
                    # rather than claiming a link was sent that cannot
                    # arrive — but do NOT reveal whether the address had
                    # an account.
                    delivered_note = problem or ""
        st.session_state[_RESET_EMAIL_KEY] = accounts.normalise_email(email)
        # _set_mode clears any pending notice, so the message is set after it.
        _set_mode("reset")
        _notify(accounts.reset_requested_message() +
                (f"\n\n{delivered_note}" if delivered_note else ""))
        st.rerun()

    if st.button("← Back to sign in", key="forgot_back", width="stretch"):
        _set_mode("signin")
        st.rerun()


def _reset_form() -> None:
    st.markdown('<div class="qx-panel-title">Enter your reset code</div>'
                '<p class="qx-panel-sub">From the email we just sent.</p>',
                unsafe_allow_html=True)

    remembered = st.session_state.get(_RESET_EMAIL_KEY, "")
    with st.form("login_reset", clear_on_submit=False):
        email = st.text_input("Email", key="reset_email_input",
                              placeholder=remembered or "you@example.com")
        token = st.text_input("Reset code", key="reset_token_input")
        password = st.text_input("New password", type="password",
                                 key="reset_password_input")
        confirm = st.text_input("Confirm new password", type="password",
                                key="reset_confirm_input")
        submitted = st.form_submit_button("Set new password", width="stretch",
                                          type="primary")

    if submitted:
        if password != confirm:
            _fail("Those two passwords don't match.")
            st.rerun()
        with st.spinner("Updating your password..."):
            ok, error = accounts.complete_reset(email or remembered, token, password)
        if not ok:
            _fail(error or "Couldn't reset that password.")
            st.rerun()
        _set_mode("signin")
        _notify("Password updated. Sign in with your new password.")
        st.rerun()

    if st.button("← Back to sign in", key="reset_back", width="stretch"):
        _set_mode("signin")
        st.rerun()


def _send_reset_email(account, token: str):
    """Deliver a reset code. Returns (sent, note_for_the_user).

    Kept in one place so the reset flow degrades honestly on an instance
    with no SMTP configured: the token still exists and still works, and
    the UI says how to get it rather than pretending mail went out.
    """
    try:
        import email_report

        if not email_report.is_email_configured():
            return False, (
                "This instance has no outgoing email configured, so the link "
                "could not be sent. An administrator can read the reset code "
                "from the server log."
            )
        subject = f"{brand().name} password reset"
        body = (
            f"Use this code to reset your {brand().name} password:\n\n    {token}\n\n"
            f"It works once and expires in {accounts.RESET_TOKEN_TTL_MINUTES} minutes. "
            "If you didn't ask for it, nothing has changed and you can ignore this."
        )
        sent, problem = email_report.send_notification_email(
            account.email, subject, body)
        return bool(sent), (None if sent else problem)
    except Exception:
        return False, (
            "Couldn't send the reset email from this instance. An administrator "
            "can read the reset code from the server log."
        )


def _auth_panel() -> None:
    st.markdown('<div id="qx-signin"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        notice = st.session_state.pop(_NOTICE_KEY, None)
        if notice:
            st.success(notice)
        error = st.session_state.pop(_ERROR_KEY, None)
        if error:
            st.error(error)

        mode = _mode()
        if mode == "signup":
            _signup_form()
        elif mode == "forgot":
            _forgot_form()
        elif mode == "reset":
            _reset_form()
        else:
            # SSO above the form: it is the faster and safer path for
            # anyone who has it, and burying it under a password form
            # trains people to type a password they did not need.
            _signin_heading()
            drew_sso = _oauth_block()
            if drew_sso:
                st.markdown('<div class="qx-or">or sign in with email</div>',
                            unsafe_allow_html=True)
            _signin_form()

    st.markdown(
        '<p class="qx-legal">Quantix is research software, not investment advice, '
        'and no figure in it is a recommendation to buy or sell. Signing in '
        'creates a workspace on this instance.</p>',
        unsafe_allow_html=True)


# --- entry point --------------------------------------------------------------

def reset_state() -> None:
    """Forget which form was last open, plus any stale error or notice.

    Called on sign-out. Without it the page reopens on whatever mode was
    last used — someone who signed out after creating an account was shown
    the "Create your account" form again, which reads as the sign-out
    having failed.
    """
    for key in (_MODE_KEY, _ERROR_KEY, _NOTICE_KEY, _RESET_EMAIL_KEY):
        st.session_state.pop(key, None)


def is_signed_in() -> bool:
    """True when either sign-in path has succeeded."""
    try:
        return auth.current_user() is not None
    except Exception:
        return False


def render() -> None:
    """Draw the whole login page."""
    _inject_css()
    _header()

    # The form column comes second on desktop and first on a phone: someone
    # on a narrow screen came here to sign in, not to scroll the pitch.
    left, right = st.columns([1.25, 1], gap="large")
    with right:
        _auth_panel()
    with left:
        _value_column()


def require_sign_in() -> None:
    """Gate the app. Renders the login page and stops when signed out."""
    if is_signed_in():
        return
    render()
    st.stop()
