"""Tests for auth.py and the per-user store namespacing it drives.

Two properties carry the most weight here and are worth naming, because
both fail silently rather than loudly:

1. NAMESPACE ISOLATION. If store_path() ever returned the same file for
   two identities, one user would be shown another user's watchlists and
   alert rules with nothing on screen to indicate it. Nothing would raise.

2. THE PER-USER / SHARED SPLIT. Team notes are shared on purpose. Scoping
   them per-user would quietly convert the collaboration feature into
   private diaries — again with no error, just a thread that mysteriously
   never shows anyone else's notes. The classification test below fails if
   a new store is added without deciding which side it belongs on.
"""
import ast
import json
import re
from pathlib import Path

import pytest

import auth
import local_store
from auth import (
    PER_USER_STORES,
    SHARED_STORES,
    adopt_shared_data,
    configured_providers,
    key_for,
    provider_label,
    unavailable_reason,
)

APP_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _restore_namespace_provider():
    """auth.py registers itself with local_store on import. Tests that
    swap it out must put it back, or every later test in the session
    silently reads from a stale namespace."""
    original = local_store._namespace_provider
    yield
    local_store.set_namespace_provider(original)


# --- namespace keys -----------------------------------------------------------

def test_key_is_stable_for_the_same_identity():
    assert key_for("https://accounts.google.com", "12345") == key_for("https://accounts.google.com", "12345")


def test_different_subjects_get_different_namespaces():
    a = key_for("https://accounts.google.com", "12345")
    b = key_for("https://accounts.google.com", "67890")
    assert a != b


def test_same_subject_at_different_issuers_does_not_collide():
    """Provider-issued subject IDs are only unique within one issuer, so
    two people could hold the same `sub` at different providers. Keying on
    subject alone would hand them each other's data."""
    assert key_for("https://accounts.google.com", "1") != key_for("https://login.microsoftonline.com", "1")


def test_key_ignores_email_so_changing_it_does_not_orphan_your_data():
    """Email is mutable at most providers. If it were part of the key, the
    day someone changed their address every watchlist, threshold and alert
    rule they had would appear to vanish."""
    with_old = key_for("https://accounts.google.com", "12345", "old@example.com")
    with_new = key_for("https://accounts.google.com", "12345", "new@example.com")
    assert with_old == with_new


def test_email_is_used_only_when_there_is_no_subject():
    assert key_for("", "", "someone@example.com") != ""


def test_no_identity_claims_yields_no_key():
    """Refusing to invent a namespace matters: a guessed key could collide
    with a real user's directory."""
    assert key_for("", "", "") == ""


def test_key_is_filesystem_safe():
    key = key_for("https://login.microsoftonline.com/common/v2.0", "abc-123|xyz")
    assert re.fullmatch(r"[a-z0-9]+-[0-9a-f]{16}", key), key
    assert "/" not in key and ".." not in key


def test_key_carries_a_readable_issuer_hint():
    assert key_for("https://accounts.google.com", "1").startswith("google-")


def test_key_does_not_leak_the_identity_in_plain_text():
    key = key_for("https://accounts.google.com", "12345", "angelos@example.com")
    assert "12345" not in key and "angelos" not in key


# --- store namespacing --------------------------------------------------------

def test_signed_out_paths_are_the_original_shared_files():
    """The whole scheme is additive. An instance that never turns on auth
    must keep reading and writing byte-for-byte the same files."""
    local_store.set_namespace_provider(lambda: None)
    assert local_store.store_path("watchlist_store.json") == APP_DIR / "watchlist_store.json"


# NOTE: store_path() creates the namespace directory on demand, so every
# test below that passes a namespace must redirect app_dir at a tmp_path.
# Without that these tests litter the real application directory with
# users/<fake-key>/ folders — which is exactly what happened the first
# time they were written.

def test_signed_in_paths_are_namespaced(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    local_store.set_namespace_provider(lambda: "google-abc")
    assert local_store.store_path("watchlist_store.json") == tmp_path / "users" / "google-abc" / "watchlist_store.json"


def test_two_identities_never_share_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    for name in PER_USER_STORES:
        a = local_store.store_path(name, namespace="google-aaa")
        b = local_store.store_path(name, namespace="google-bbb")
        assert a != b, name


def test_a_broken_namespace_provider_degrades_to_shared_rather_than_crashing():
    """Called on every store read. A provider that raises must not take
    down whichever store happened to be loading."""
    def boom():
        raise RuntimeError("no script run context")
    local_store.set_namespace_provider(boom)
    assert local_store.current_namespace() is None
    assert local_store.store_path("theme_state.json") == APP_DIR / "theme_state.json"


def test_no_provider_registered_means_shared():
    local_store.set_namespace_provider(None)
    assert local_store.current_namespace() is None


def test_empty_key_is_treated_as_signed_out():
    local_store.set_namespace_provider(lambda: "")
    assert local_store.current_namespace() is None


def test_store_path_creates_the_namespace_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    path = local_store.store_path("theme_state.json", namespace="google-xyz")
    assert path.parent.is_dir()


def test_stores_round_trip_independently_per_user(tmp_path, monkeypatch):
    """The end-to-end isolation property, exercised through a real store."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    import theme

    local_store.set_namespace_provider(lambda: "google-aaa")
    theme.save_theme("light")
    local_store.set_namespace_provider(lambda: "google-bbb")
    theme.save_theme("dark")

    local_store.set_namespace_provider(lambda: "google-aaa")
    assert theme.load_theme() == "light"
    local_store.set_namespace_provider(lambda: "google-bbb")
    assert theme.load_theme() == "dark"


# --- which stores are scoped --------------------------------------------------

def _declared_store_filenames():
    """Every store filename the app declares, scraped from the source so
    this test can't drift from reality."""
    found = set()
    for module in ("config.py", "risk_alerts.py"):
        tree = ast.parse((APP_DIR / module).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name, value = node.target.id, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, node.value
            else:
                continue
            if not re.search(r"(filename|FILENAME)$", name):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.add(value.value)
    return found


def test_every_declared_store_is_classified_as_per_user_or_shared():
    """A new store added without deciding which side it belongs on is a
    silent privacy or silent-breakage bug depending on which way it falls,
    so it fails here instead."""
    classified = set(PER_USER_STORES) | set(SHARED_STORES)
    unclassified = _declared_store_filenames() - classified
    assert not unclassified, (
        f"These stores aren't listed in auth.PER_USER_STORES or auth.SHARED_STORES: "
        f"{sorted(unclassified)}. Decide whether each is personal state or shared "
        f"infrastructure and add it to the right tuple."
    )


def test_no_store_is_in_both_lists():
    assert not (set(PER_USER_STORES) & set(SHARED_STORES))


def test_team_notes_stay_shared():
    """Guarding a design decision, not an implementation detail: notes
    exist so teammates can read each other."""
    assert "collaboration_store.json" in SHARED_STORES
    assert "collaboration_store.json" not in PER_USER_STORES


def test_the_trained_model_stays_shared():
    assert "ml_momentum_model.joblib" in SHARED_STORES
    assert "ml_training_history.json" in SHARED_STORES


def _imports_store_path(text: str) -> bool:
    """Whether a module pulls local_store's namespacing helper in."""
    return bool(re.search(r"^from local_store import .*\bstore_path\b", text, re.M))


# Which module owns which store. Explicit rather than inferred: the
# filenames are declared in config.py, not in the modules that use them,
# so there is nothing reliable to infer ownership from.
STORE_OWNERS = {
    "watchlist_store.json": "watchlist_panel.py",
    "favorites_store.json": "favorites.py",
    "theme_state.json": "theme.py",
    "onboarding_state.json": "onboarding.py",
    "threshold_overrides.json": "user_thresholds.py",
    "risk_alert_rules_store.json": "risk_alerts.py",
    "alert_rules_store.json": "realtime_alerts.py",
    "scenario_store.json": "scenario_modeling.py",
}


def test_the_owner_map_covers_every_per_user_store():
    """Forces this file to be updated when a store is added, which is what
    makes the wiring test below meaningful."""
    assert set(STORE_OWNERS) == set(PER_USER_STORES)


@pytest.mark.parametrize("filename,module", sorted(STORE_OWNERS.items()))
def test_every_per_user_store_module_actually_uses_store_path(filename, module):
    """Listing a file in PER_USER_STORES does nothing on its own — the
    module that owns it has to resolve its path through store_path().
    This catches the listed-but-not-wired case, which would look correct
    in review and share every user's data in production."""
    text = (APP_DIR / module).read_text()
    # Keyed off the IMPORT, not a bare substring: several of these modules
    # have their own private _store_path() helper, so "store_path(" alone
    # matches whether or not local_store's namespacing is actually used.
    assert _imports_store_path(text), (
        f"{module} owns {filename} but doesn't import store_path from local_store, "
        f"so that store is not namespaced per user"
    )
    assert "Path(__file__).resolve().parent /" not in text, (
        f"{module} still resolves a store path directly, bypassing per-user namespacing"
    )


@pytest.mark.parametrize("module", ["collaboration.py", "ml_pipeline.py"])
def test_deliberately_shared_modules_do_not_namespace_their_stores(module):
    """The inverse guard. If someone 'fixes' collaboration.py to use
    store_path() for consistency, the team thread silently becomes a set
    of private diaries — no error, just notes nobody else can see."""
    text = (APP_DIR / module).read_text()
    assert not _imports_store_path(text), (
        f"{module} holds a deliberately SHARED store (see auth.SHARED_STORES) "
        f"but imports store_path, which would scope it per user"
    )


def test_every_store_resolves_through_app_dir():
    """No store module may build its path from Path(__file__) directly.

    This is a REGRESSION TEST FOR A REAL ACCIDENT. A verification script
    pointed local_store.app_dir() at a sandbox and then wrote through the
    store modules, believing that isolated it. collaboration.py did not
    consult app_dir() at all — it built its path from Path(__file__) — so
    the write went straight into the live collaboration store and left a
    junk note in the user's real data.

    Routing every store through app_dir() is what makes sandboxing
    actually work. logging_setup is excluded: it owns the log file, not a
    store, and is initialised before any of this exists.
    """
    offenders = []
    for path in sorted(APP_DIR.glob("*.py")):
        if path.name == "logging_setup.py":
            continue
        if "Path(__file__).resolve().parent /" in path.read_text():
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} resolve a path from Path(__file__) instead of via "
        f"local_store.store_path() / shared_path(), so they cannot be redirected "
        f"and will write into the real app directory during tests"
    )


def test_redirecting_app_dir_sandboxes_shared_stores_too(tmp_path, monkeypatch):
    """The other half of the same accident: a test that redirects app_dir
    must capture the deliberately-shared stores as well, not just the
    namespaced ones."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    import collaboration

    store, _, _ = collaboration.add_note(
        collaboration.CollaborationStore(), "AAPL", "Tester", "sandboxed note")
    collaboration.save_store(store)

    assert (tmp_path / "collaboration_store.json").exists()
    real = APP_DIR / "collaboration_store.json"
    # Absent on a fresh checkout; present on a working instance. Either way
    # the sandboxed note must not be in it.
    assert "sandboxed note" not in (real.read_text() if real.exists() else ""), \
        "the note escaped the sandbox and was written into the real store"


def test_shared_path_never_namespaces(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    local_store.set_namespace_provider(lambda: "google-abc")
    assert local_store.shared_path("collaboration_store.json") == tmp_path / "collaboration_store.json"


# --- provider configuration ---------------------------------------------------

def _secrets(monkeypatch, section):
    monkeypatch.setattr(auth, "_auth_section", lambda: section)


def test_providers_are_discovered_from_secrets_not_hardcoded(monkeypatch):
    """The point of discovery: a provider this module has never heard of
    — an Auth0 broker fronting GitHub, say — must work with no code
    change."""
    _secrets(monkeypatch, {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "x" * 64,
        "acme_broker": {"client_id": "a", "client_secret": "b", "server_metadata_url": "c"},
    })
    assert configured_providers() == ("acme_broker",)


def test_incomplete_providers_are_skipped(monkeypatch):
    """A button that would fail on click is worse than no button."""
    _secrets(monkeypatch, {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "x" * 64,
        "google": {"client_id": "a", "client_secret": "b", "server_metadata_url": "c"},
        "microsoft": {"client_id": "a", "client_secret": "b"},  # no discovery URL
    })
    assert configured_providers() == ("google",)


def test_blank_values_do_not_count_as_configured(monkeypatch):
    _secrets(monkeypatch, {
        "redirect_uri": "u", "cookie_secret": "s",
        "google": {"client_id": "a", "client_secret": "   ", "server_metadata_url": "c"},
    })
    assert configured_providers() == ()


def test_unnamed_default_provider_is_supported(monkeypatch):
    """Streamlit allows credentials directly on [auth] with no sub-table;
    that form calls st.login() with no argument."""
    _secrets(monkeypatch, {
        "redirect_uri": "u", "cookie_secret": "s",
        "client_id": "a", "client_secret": "b", "server_metadata_url": "c",
    })
    assert configured_providers() == ("",)


def test_no_secrets_means_no_providers(monkeypatch):
    _secrets(monkeypatch, {})
    assert configured_providers() == ()


def test_provider_labels():
    assert provider_label("google") == "Google"
    assert provider_label("microsoft") == "Microsoft"
    assert provider_label("") == "Sign in"
    assert provider_label("acme_broker") == "Acme Broker"


# --- unavailability messages --------------------------------------------------

def test_missing_authlib_is_reported_first(monkeypatch):
    """Ordered so following the messages in sequence actually gets you
    there — no point naming a missing client_id to someone who can't run
    the flow at all yet."""
    monkeypatch.setattr(auth, "is_authlib_installed", lambda: False)
    _secrets(monkeypatch, {})
    assert "Authlib" in unavailable_reason()


def test_no_auth_section_explains_where_to_look(monkeypatch):
    monkeypatch.setattr(auth, "is_authlib_installed", lambda: True)
    _secrets(monkeypatch, {})
    reason = unavailable_reason()
    assert "secrets.toml.example" in reason


def test_missing_top_level_keys_are_named(monkeypatch):
    monkeypatch.setattr(auth, "is_authlib_installed", lambda: True)
    _secrets(monkeypatch, {"google": {"client_id": "a", "client_secret": "b", "server_metadata_url": "c"}})
    reason = unavailable_reason()
    assert "redirect_uri" in reason and "cookie_secret" in reason


def test_configured_instance_reports_available(monkeypatch):
    monkeypatch.setattr(auth, "is_authlib_installed", lambda: True)
    _secrets(monkeypatch, {
        "redirect_uri": "u", "cookie_secret": "s",
        "google": {"client_id": "a", "client_secret": "b", "server_metadata_url": "c"},
    })
    assert unavailable_reason() is None
    assert auth.is_available()


# --- adopting the signed-out profile ------------------------------------------

def _seed_shared(tmp_path, names):
    for i, name in enumerate(names):
        (tmp_path / name).write_text(json.dumps({"seeded": i}))


def test_adopt_copies_shared_stores_into_the_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    _seed_shared(tmp_path, ["watchlist_store.json", "favorites_store.json"])
    copied, errors = adopt_shared_data("google-abc")
    assert errors == []
    assert set(copied) == {"watchlist_store.json", "favorites_store.json"}
    assert json.loads((tmp_path / "users" / "google-abc" / "watchlist_store.json").read_text()) == {"seeded": 0}


def test_adopt_copies_rather_than_moves(tmp_path, monkeypatch):
    """The signed-out profile must survive intact — that's what makes this
    reversible by simply signing out, and safe to run when unsure."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    _seed_shared(tmp_path, ["theme_state.json"])
    adopt_shared_data("google-abc")
    assert (tmp_path / "theme_state.json").exists()


def test_adopt_does_not_overwrite_existing_user_data(tmp_path, monkeypatch):
    """Someone who has already customised their account must not have it
    silently replaced by the shared profile."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    _seed_shared(tmp_path, ["theme_state.json"])
    target = tmp_path / "users" / "google-abc"
    target.mkdir(parents=True)
    (target / "theme_state.json").write_text(json.dumps({"mine": True}))

    copied, _ = adopt_shared_data("google-abc")
    assert "theme_state.json" not in copied
    assert json.loads((target / "theme_state.json").read_text()) == {"mine": True}


def test_adopt_ignores_stores_that_are_not_per_user(tmp_path, monkeypatch):
    """Copying team notes into a private namespace would fork the shared
    thread into a divergent private one."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    _seed_shared(tmp_path, ["collaboration_store.json", "watchlist_store.json"])
    copied, _ = adopt_shared_data("google-abc")
    assert copied == ("watchlist_store.json",)
    assert not (tmp_path / "users" / "google-abc" / "collaboration_store.json").exists()


def test_adopt_with_nothing_to_copy_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    assert adopt_shared_data("google-abc") == ((), [])


def test_has_user_data_and_shared_data_files(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    assert auth.shared_data_files() == ()
    assert not auth.has_user_data("google-abc")

    _seed_shared(tmp_path, ["watchlist_store.json"])
    assert auth.shared_data_files() == ("watchlist_store.json",)
    assert not auth.has_user_data("google-abc")

    adopt_shared_data("google-abc")
    assert auth.has_user_data("google-abc")


# --- degradation --------------------------------------------------------------

def test_nothing_raises_when_auth_is_entirely_unconfigured():
    """The state of a fresh checkout, and of this instance until
    credentials are added. Every one of these is called on a normal page
    render, so any of them raising would be a crash on startup."""
    assert auth.current_user() is None
    assert auth.user_key() is None
    assert auth.is_logged_in() is False
    assert isinstance(auth.configured_providers(), tuple)
    assert auth.unavailable_reason() is not None or auth.is_available()


def test_the_authlib_check_tests_the_import_path_st_login_actually_uses():
    """REGRESSION TEST FOR A REAL FALSE POSITIVE.

    is_authlib_installed() originally did a bare `import authlib`. That
    succeeds even when httpx is missing — but Authlib's Starlette
    integration, which is the code path st.login() takes, imports httpx.
    So the check returned True, the Account panel offered a sign-in
    button, and clicking it produced a 500 whose logged error claimed
    "Authentication requires Authlib>=1.3.2" — naming the wrong package.

    Asserting on the source keeps the check pointed at the real
    dependency rather than a proxy for it.
    """
    source = (APP_DIR / "auth.py").read_text()
    body = source.split("def is_authlib_installed")[1].split("def ")[0]
    assert "starlette_client" in body, (
        "is_authlib_installed must import authlib.integrations.starlette_client — "
        "a bare `import authlib` gives a false positive when httpx is absent"
    )


def test_requirements_name_both_auth_dependencies():
    """httpx is a silent, non-obvious requirement: `pip install
    streamlit[auth]` does not pull it, and without it sign-in fails at
    runtime rather than at install time."""
    requirements = (APP_DIR / "requirements.txt").read_text().lower()
    assert "authlib" in requirements
    assert "httpx" in requirements
