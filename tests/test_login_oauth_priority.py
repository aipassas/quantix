"""The login page's OAuth-first ordering, validation messages and loading states.

The security-sensitive assertions here are the ones about what a failure
message may say. accounts.authenticate returns a deliberately identical
string whether the address is unknown or the password is wrong, and burns
the same KDF time on both paths, so the form cannot be used to enumerate
who has an account. "Better error messages" must not quietly undo that,
so the tests below pin the property rather than the wording.
"""
import ast
import re
import time
from pathlib import Path

import pytest

import accounts


LOGIN_PY = Path(__file__).resolve().parent.parent / "login_page.py"


def code_only(text: str) -> str:
    """`text` with Python comments, docstrings and CSS comments removed.

    Every assertion below that scans for a string needs this. The comments
    in login_page.py deliberately quote what the code must NOT do — the
    aria-label selector that would be wrong, the provider names that must
    not be hard-coded — so a raw substring scan finds the prose explaining
    the rule and fails on correct code. Three of these tests did exactly
    that on the first run.

    Docstrings are located with ast rather than by matching triple quotes.
    A regex for \"\"\"..\"\"\" also eats the CSS, which lives in an f-string —
    that silently deleted two thirds of the file and turned the toggle
    test into one that passed against anything.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:                      # a fragment, not a whole module
        tree = None
    lines = text.splitlines()
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.FunctionDef,
                                     ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                for i in range(first.lineno - 1, first.end_lineno):
                    lines[i] = ""
    out = "\n".join(l for l in lines if not l.lstrip().startswith("#"))
    return re.sub(r"/\*.*?\*/", "", out, flags=re.S)          # CSS comments


@pytest.fixture(scope="module")
def source() -> str:
    return LOGIN_PY.read_text(encoding="utf-8")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A sandboxed account store. Never the real one."""
    import local_store

    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    account, error = accounts.create_account(
        "real@example.com", "Correct-Horse-9!", "Real Person")
    assert error is None, error
    return account


# --- what a failure is allowed to reveal --------------------------------------

def test_unknown_address_and_wrong_password_are_indistinguishable(store):
    """The whole point of the generic message. If these ever diverge the
    sign-in form answers "does this person have an account here?"."""
    _, wrong_password = accounts.authenticate("real@example.com", "not-the-one")
    _, unknown_email = accounts.authenticate("ghost@example.com", "not-the-one")
    assert wrong_password == unknown_email


def test_both_credential_failures_still_burn_the_kdf(store):
    """A fast rejection on an unknown address is a timing oracle even when
    the text matches, which is why the unknown path calls verify_dummy()."""
    def elapsed(email):
        start = time.perf_counter()
        accounts.authenticate(email, "not-the-one")
        return time.perf_counter() - start

    known, unknown = elapsed("real@example.com"), elapsed("ghost@example.com")
    assert known > 0.02 and unknown > 0.02, (known, unknown)
    # Within a factor of three of each other — generous, because this is
    # about there being no ORDER-of-magnitude tell, not about jitter.
    assert max(known, unknown) / min(known, unknown) < 3, (known, unknown)


def test_a_credential_failure_never_names_the_field(store):
    """"No account with that email" / "wrong password" are the two phrasings
    that leak. Neither may appear once the store has been consulted."""
    _, message = accounts.authenticate("real@example.com", "not-the-one")
    lowered = message.lower()
    for leak in ("no account", "no such", "not registered", "unknown email",
                 "wrong password", "incorrect password", "password is wrong",
                 "email not found", "user not found"):
        assert leak not in lowered, f"{leak!r} in {message!r}"


# --- input validation, which leaks nothing and was simply wrong ---------------

@pytest.mark.parametrize("email,password,expected", [
    ("", "", "Enter your email address and password."),
    ("real@example.com", "", "Enter your password."),
    ("", "Correct-Horse-9!", "Enter your email address."),
    ("   ", "", "Enter your email address and password."),
    # A password of spaces is NOT blank: passwords are never trimmed,
    # because whitespace can legitimately be part of one. Only the email
    # is normalised, so this reports the email alone.
    ("   ", "  ", "Enter your email address."),
    ("notanemail", "whatever", "That doesn't look like an email address."),
])
def test_shape_problems_are_reported_for_what_they_are(store, email, password, expected):
    """Telling someone who left a box empty that their password is wrong
    sent them hunting for a typo they never made."""
    account, message = accounts.authenticate(email, password)
    assert account is None
    assert message == expected


def test_validation_does_not_touch_the_store_or_the_kdf(store):
    """These judgements are about the submitted text, not about any
    account, so there is nothing to be constant-time about — and 122ms of
    scrypt to say "you left the box empty" was pure waste."""
    start = time.perf_counter()
    accounts.authenticate("", "")
    assert time.perf_counter() - start < 0.02


def test_a_valid_sign_in_still_works(store):
    account, error = accounts.authenticate("real@example.com", "Correct-Horse-9!")
    assert error is None and account is not None


def test_lockout_still_reports_itself(store, monkeypatch):
    """The lockout message is the one case that DOES differ, deliberately:
    reporting "wrong password" while locked reads as the password having
    changed underneath you."""
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK + 1):
        _, message = accounts.authenticate("real@example.com", "not-the-one")
    assert "too many failed attempts" in message.lower()


# --- the page's ordering and loading states -----------------------------------

def _panel_body(source: str) -> str:
    """The sign-in branch of _auth_panel."""
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_auth_panel")
    return ast.get_source_segment(source, func)


@pytest.mark.parametrize("panel", ["signin", "signup"])
def test_single_sign_on_is_offered_before_the_password_form(source, panel):
    """The task's headline item, and it applies to BOTH panels: burying
    SSO under a password form trains people to type a password they did
    not need, and that is as true when creating an account as when
    returning to one."""
    body = _panel_body(source)
    assert body.index("_oauth_block()") < body.index(f"_{panel}_form()")


@pytest.mark.parametrize("panel", ["signin", "signup"])
def test_the_heading_still_comes_first(source, panel):
    """Each title had to lift out of its form when SSO moved between them,
    or the panel would open with an unlabelled row of buttons."""
    body = _panel_body(source)
    assert body.index(f"_{panel}_heading()") < body.index(f"_{panel}_form()")
    form = ast.get_source_segment(
        source, next(n for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.FunctionDef) and n.name == f"_{panel}_form"))
    assert "qx-panel-title" not in form, (
        f"_{panel}_form still draws its own title")


@pytest.mark.parametrize("panel,wording", [("signin", "sign in"), ("signup", "sign up")])
def test_the_email_divider_is_suppressed_when_no_provider_exists(source, panel, wording):
    """Otherwise an instance without SSO gets a rule labelled "or sign in
    with email" separating the form from nothing above it. The rule is
    drawn only inside `if _oauth_block():`, whose truthiness is exactly
    "something was drawn"."""
    func = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "_auth_panel")
    guarded = [
        n for n in ast.walk(func)
        if isinstance(n, ast.If) and "_oauth_block()" in ast.unparse(n.test)
        and any(f'"or {wording} with email"' in ast.unparse(c).replace("'", '"')
                for c in n.body)
    ]
    assert guarded, f"the {panel} rule is not guarded by _oauth_block()"

    # And the block reports whether it drew, rather than returning None.
    oauth = next(n for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.FunctionDef) and n.name == "_oauth_block")
    returns = [n for n in ast.walk(oauth) if isinstance(n, ast.Return)]
    assert returns and all(isinstance(r.value, ast.Constant) for r in returns)
    assert {r.value.value for r in returns} == {True, False}


def test_the_sso_button_wording_stays_neutral(source):
    """There is no separate "sign up with Google": the first OIDC sign-in
    creates the workspace. One shared block, one neutral label — only the
    rule beneath it changes between the panels."""
    oauth = ast.get_source_segment(
        source, next(n for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.FunctionDef) and n.name == "_oauth_block"))
    assert "Continue with" in oauth
    for wrong in ("Sign up with", "Sign in with"):
        assert wrong not in code_only(oauth)


def test_providers_are_not_hard_coded(source):
    """The task names Google, GitHub and Microsoft; only Google is
    configured on this instance. The list stays data-driven so adding one
    to secrets.toml needs no code change."""
    oauth = ast.get_source_segment(
        source, next(n for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.FunctionDef) and n.name == "_oauth_block"))
    assert "auth.configured_providers()" in oauth
    for name in ("GitHub", "Microsoft", "Google"):
        assert name not in code_only(oauth), f"{name} is hard-coded in _oauth_block"


@pytest.mark.parametrize("func,needle", [
    ("_signin_form", "st.spinner"),
    ("_signup_form", "st.spinner"),
    ("_forgot_form", "st.spinner"),   # SMTP: the slowest step on the page
    ("_reset_form", "st.spinner"),
    ("_oauth_block", "st.spinner"),   # the provider round trip
])
def test_every_submit_reports_that_it_is_working(source, func, needle):
    body = ast.get_source_segment(
        source, next(n for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.FunctionDef) and n.name == func))
    assert needle in body, f"{func} submits with no loading state"


def test_the_reset_spinner_wraps_the_send_not_just_the_token(source):
    """begin_reset is local and fast; _send_reset_email is the part that
    can take seconds. A spinner around only the former would be theatre."""
    func = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "_forgot_form")

    def calls_spinner(node: ast.With) -> bool:
        return any("st.spinner" in (ast.unparse(item.context_expr) or "")
                   for item in node.items)

    # Structural, not textual: an indentation check passed a version whose
    # spinner wrapped `pass` and left the SMTP send outside it entirely.
    wrapped = [w for w in ast.walk(func)
               if isinstance(w, ast.With) and calls_spinner(w)
               and any("_send_reset_email" in ast.unparse(child)
                       for child in w.body)]
    assert wrapped, "_send_reset_email is not inside the spinner's with-block"


# --- the password toggle ------------------------------------------------------

def test_the_sso_button_is_not_streamlits_stock_red(source):
    """It sits outside any form, so the accent rule scoped to stForm did
    not reach it and it rendered #FF4B4B — bright red directly above a
    cyan "Sign in", in an app where red means a loss."""
    blocks = code_only(source).split("}}")
    filling = [b for b in blocks
               if 'button[kind="primary"]' in b and "background:" in b]
    assert filling, "no rule matching a bare primary button sets a background"
    assert any("{colour}" in b.split("{{", 1)[1] for b in filling), (
        "the primary fill is not the brand accent")


def test_the_eye_toggle_is_selected_structurally_not_by_label(source):
    """Inspected on the running page: the button carries no data-testid and
    only emotion-hash classes. aria-label is the other candidate and is
    wrong — it flips to "Hide password text" on toggle, and is localised."""
    assert '[data-testid="stTextInputRootElement"] > div[data-baseweb="base-input"] > button' in source
    assert "[aria-label" not in code_only(source), (
        "the toggle must not be selected by its label")


def test_the_eye_toggle_reads_as_a_control(source):
    """It was already white but flat: no border, no hover, no cue that it
    was a control at all."""
    match = re.search(
        r'base-input"\] > button \{\{(.*?)\}\}', source, re.S)
    assert match, "no rest-state rule for the toggle"
    # Property NAMES, not substrings: "border" is a substring of
    # "border-radius", so removing the actual border left this passing.
    declared = {d.split(":", 1)[0].strip()
                for d in match.group(1).split(";") if ":" in d}
    for prop in ("border", "min-width", "min-height", "transition"):
        assert prop in declared, f"{prop} not declared; got {sorted(declared)}"
    assert re.search(r'base-input"\] > button:hover', source)


def test_the_toggle_meets_a_44px_tap_target(source):
    match = re.search(r'base-input"\] > button \{\{(.*?)\}\}', source, re.S)
    sizes = re.findall(r"min-(?:width|height):\s*(\d+)px", match.group(1))
    assert sizes and all(int(s) >= 44 for s in sizes), sizes
