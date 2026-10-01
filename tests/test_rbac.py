"""Roles: a boundary inside the app, and honest about where it ends.

THE TEST THAT MATTERS MOST is test_an_unknown_permission_is_refused. A
typo at a call site must CLOSE the door, not open it — the opposite
default is how a gate silently stops gating, and nothing in the UI would
look different.

The second is test_the_last_admin_cannot_be_demoted. An instance that can
lock itself out of its own administration has a recovery path that
involves editing JSON by hand, which is not a recovery path.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rbac
from config import RBAC

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def _store(*pairs):
    return rbac.RoleStore(tuple(
        rbac.Assignment(key, role, "k-granter", "2026-10-01T00:00:00")
        for key, role in pairs))


# --- the ladder ---------------------------------------------------------------

def test_the_roles_are_ordered_most_privileged_first():
    """`can()` compares positions in this tuple, so the order IS the
    matrix."""
    assert rbac.ROLES == (rbac.ADMIN, rbac.ANALYST, rbac.VIEWER)


def test_every_role_has_a_spec_and_a_blurb():
    assert {s.id for s in rbac.ROLE_SPECS} == set(rbac.ROLES)
    for spec in rbac.ROLE_SPECS:
        assert spec.label.strip() and len(spec.blurb) > 40


def test_a_higher_role_can_do_everything_a_lower_one_can():
    """Otherwise the three tiers are not a ladder and an Admin would hit
    refusals an Analyst does not."""
    for permission in rbac.PERMISSIONS:
        allowed = [r for r in rbac.ROLES if rbac.can(r, permission.id)]
        # The allowed roles must be a PREFIX of the ladder.
        assert allowed == list(rbac.ROLES[:len(allowed)]), permission.id


def test_admin_can_do_everything():
    for permission in rbac.PERMISSIONS:
        assert rbac.can(rbac.ADMIN, permission.id), permission.id


# --- the matrix ---------------------------------------------------------------

def test_an_unknown_permission_is_refused():
    """A typo must close the door. Returning True for an unrecognised
    name is how a gate stops gating with nothing on screen to show it."""
    assert rbac.can(rbac.ADMIN, "admin.typo") is False
    assert rbac.can(rbac.ADMIN, "") is False
    assert "A known permission" in rbac.require(rbac.ADMIN, "admin.typo")


def test_an_unknown_role_is_refused():
    assert rbac.can("superuser", "own.watchlist") is False
    assert rbac.can("", "own.watchlist") is False


def test_a_viewer_keeps_their_own_work():
    """A Viewer who cannot even keep a watchlist has no reason to sign
    in."""
    for permission in ("own.watchlist", "own.journal", "own.portfolio",
                       "own.thresholds", "own.export"):
        assert rbac.can(rbac.VIEWER, permission), permission


def test_a_viewer_writes_nothing_shared():
    """The line the product owner drew: shared state needs Analyst."""
    for permission in ("shared.post_note", "shared.contest_enter",
                       "shared.leaderboard_join", "shared.peer_publish",
                       "shared.profile_publish"):
        assert not rbac.can(rbac.VIEWER, permission), permission
        assert rbac.can(rbac.ANALYST, permission), permission


def test_an_analyst_cannot_configure_the_instance():
    for permission in ("admin.roles", "admin.audit_read", "admin.api_keys",
                       "admin.webhooks", "admin.branding", "admin.delivery",
                       "admin.moderate_any", "admin.audit_erase"):
        assert not rbac.can(rbac.ANALYST, permission), permission
        assert rbac.can(rbac.ADMIN, permission), permission


def test_every_permission_id_is_namespaced_by_its_tier():
    """A permission whose prefix disagrees with its minimum role is a
    reader trap — the id is what a call site reads."""
    expected = {"admin": rbac.ADMIN, "shared": rbac.ANALYST, "own": rbac.VIEWER}
    for permission in rbac.PERMISSIONS:
        prefix = permission.id.split(".")[0]
        assert prefix in expected, permission.id
        assert permission.min_role == expected[prefix], permission.id


def test_the_permission_set_has_an_exact_size():
    """Adding one should break this and be extended deliberately."""
    assert len(rbac.PERMISSIONS) == 18


def test_every_permission_id_is_unique():
    ids = [p.id for p in rbac.PERMISSIONS]
    assert len(ids) == len(set(ids))
    assert set(rbac.PERMISSION_MAP) == set(ids)


def test_permissions_for_a_role_grows_down_the_ladder():
    assert (len(rbac.permissions_for(rbac.ADMIN))
            > len(rbac.permissions_for(rbac.ANALYST))
            > len(rbac.permissions_for(rbac.VIEWER)) > 0)


# --- require ------------------------------------------------------------------

def test_require_returns_empty_when_permitted():
    assert rbac.require(rbac.ADMIN, "admin.roles") == ""


def test_require_names_the_role_needed():
    message = rbac.require(rbac.VIEWER, "shared.post_note")
    assert "Viewer" in message and "Analyst" in message
    assert "administrator" in message


def test_require_tells_a_signed_out_reader_to_sign_in():
    assert rbac.require("", "own.watchlist") == rbac.NOT_SIGNED_IN


# --- the default role ---------------------------------------------------------

def test_an_unassigned_account_gets_the_default():
    assert rbac.role_for(rbac.RoleStore(), "k-new") == RBAC.default_role


def test_the_default_is_the_least_privileged_role():
    """A workspace where anyone who reaches the sign-up form can write to
    shared state has roles in name only."""
    assert RBAC.default_role == rbac.VIEWER
    assert rbac.ROLES[-1] == RBAC.default_role


# --- bootstrap ----------------------------------------------------------------

def test_a_fresh_instance_needs_a_bootstrap():
    assert rbac.needs_bootstrap(rbac.RoleStore()) is True


def test_the_first_account_becomes_admin():
    store, granted = rbac.bootstrap(rbac.RoleStore(), "k-ana")
    assert granted is True
    assert rbac.role_for(store, "k-ana") == rbac.ADMIN
    assert rbac.is_admin(store, "k-ana")


def test_the_bootstrap_is_recorded_as_automatic():
    """An auditor must not be left wondering which account granted the
    first one."""
    store, _ = rbac.bootstrap(rbac.RoleStore(), "k-ana")
    assert store.get("k-ana").granted_by == rbac.BOOTSTRAP
    assert store.get("k-ana").granted_at


def test_the_second_account_does_not_also_become_admin():
    store, _ = rbac.bootstrap(rbac.RoleStore(), "k-ana")
    store, granted = rbac.bootstrap(store, "k-ben")
    assert granted is False
    assert rbac.role_for(store, "k-ben") == RBAC.default_role


def test_the_bootstrap_does_not_run_again_once_an_admin_exists():
    store = _store(("k-ana", rbac.ADMIN))
    assert rbac.needs_bootstrap(store) is False
    _, granted = rbac.bootstrap(store, "k-ben")
    assert granted is False


def test_an_instance_with_only_non_admins_still_needs_a_bootstrap():
    """Roles assigned without an admin would otherwise leave nobody able
    to grant one."""
    store = _store(("k-ben", rbac.ANALYST), ("k-cy", rbac.VIEWER))
    assert rbac.needs_bootstrap(store) is True


def test_a_corrupt_store_never_bootstraps():
    """Re-running the bootstrap because a file could not be read would
    hand Admin to whoever happened to be signed in."""
    store = rbac.RoleStore(corrupt=True)
    assert rbac.needs_bootstrap(store) is False
    after, granted = rbac.bootstrap(store, "k-anyone")
    assert granted is False and after.assignments == ()


def test_a_signed_out_reader_cannot_bootstrap():
    _, granted = rbac.bootstrap(rbac.RoleStore(), "")
    assert granted is False


# --- granting -----------------------------------------------------------------

def test_an_admin_can_grant_a_role():
    store = _store(("k-ana", rbac.ADMIN))
    store, err = rbac.grant(store, "k-ana", "k-ben", rbac.ANALYST)
    assert err == "" and rbac.role_for(store, "k-ben") == rbac.ANALYST


def test_an_analyst_cannot_grant():
    store = _store(("k-ana", rbac.ADMIN), ("k-ben", rbac.ANALYST))
    after, err = rbac.grant(store, "k-ben", "k-cy", rbac.ADMIN)
    assert err == rbac.NOT_ADMIN_GRANT
    assert after == store, "a refused grant must change nothing"


def test_a_viewer_cannot_grant_themselves_anything():
    store = _store(("k-ana", rbac.ADMIN), ("k-ben", rbac.VIEWER))
    _, err = rbac.grant(store, "k-ben", "k-ben", rbac.ADMIN)
    assert err == rbac.NOT_ADMIN_GRANT


def test_a_signed_out_reader_cannot_grant():
    store = _store(("k-ana", rbac.ADMIN))
    _, err = rbac.grant(store, "", "k-ben", rbac.ADMIN)
    assert err == rbac.NOT_SIGNED_IN


def test_an_unknown_role_cannot_be_granted():
    store = _store(("k-ana", rbac.ADMIN))
    _, err = rbac.grant(store, "k-ana", "k-ben", "superuser")
    assert err == rbac.UNKNOWN_ROLE


def test_granting_replaces_rather_than_duplicates():
    store = _store(("k-ana", rbac.ADMIN), ("k-ben", rbac.VIEWER))
    store, _ = rbac.grant(store, "k-ana", "k-ben", rbac.ANALYST)
    matching = [a for a in store.assignments if a.user_key == "k-ben"]
    assert len(matching) == 1 and matching[0].role == rbac.ANALYST


def test_a_grant_records_who_made_it():
    store = _store(("k-ana", rbac.ADMIN))
    store, _ = rbac.grant(store, "k-ana", "k-ben", rbac.ANALYST)
    assert store.get("k-ben").granted_by == "k-ana"
    assert store.get("k-ben").granted_at


def test_granting_the_role_somebody_already_has_is_a_no_op():
    store = _store(("k-ana", rbac.ADMIN), ("k-ben", rbac.ANALYST))
    after, err = rbac.grant(store, "k-ana", "k-ben", rbac.ANALYST)
    assert err == "" and after == store


# --- the last admin -----------------------------------------------------------

def test_the_last_admin_cannot_be_demoted():
    """An instance that locks itself out of its own administration has a
    recovery path that means editing JSON by hand."""
    store = _store(("k-ana", rbac.ADMIN), ("k-ben", rbac.ANALYST))
    after, err = rbac.grant(store, "k-ana", "k-ana", rbac.VIEWER)
    assert err == rbac.LAST_ADMIN
    assert after == store
    assert rbac.is_admin(after, "k-ana")


def test_the_last_admin_cannot_be_demoted_by_another_admin_either():
    """There is no second Admin in this fixture, so 'another admin' can
    only be themselves — but the rule is about the COUNT, not about who
    asked."""
    store = _store(("k-ana", rbac.ADMIN))
    _, err = rbac.grant(store, "k-ana", "k-ana", rbac.ANALYST)
    assert err == rbac.LAST_ADMIN


def test_an_admin_can_be_demoted_once_there_are_two():
    store = _store(("k-ana", rbac.ADMIN), ("k-ben", rbac.ADMIN))
    store, err = rbac.grant(store, "k-ben", "k-ana", rbac.ANALYST)
    assert err == ""
    assert rbac.role_for(store, "k-ana") == rbac.ANALYST
    assert store.admin_keys == ("k-ben",)


def test_promoting_a_second_admin_then_demoting_the_first_works():
    store = _store(("k-ana", rbac.ADMIN))
    store, err = rbac.grant(store, "k-ana", "k-ben", rbac.ADMIN)
    assert err == ""
    store, err = rbac.grant(store, "k-ana", "k-ana", rbac.VIEWER)
    assert err == "", "the escape hatch the last-admin rule leaves open"
    assert rbac.role_for(store, "k-ana") == rbac.VIEWER


def test_roles_cannot_be_changed_on_a_corrupt_store():
    store = rbac.RoleStore(corrupt=True)
    _, err = rbac.grant(store, "k-ana", "k-ben", rbac.ADMIN)
    assert "could not be read" in err


# --- identity -----------------------------------------------------------------

def test_roles_are_keyed_on_the_account_key_not_the_email():
    """Two accounts can share a display name and an email is mutable at
    most providers."""
    import dataclasses
    fields = [f.name for f in dataclasses.fields(rbac.Assignment)]
    assert fields == ["user_key", "role", "granted_by", "granted_at"]
    assert "email" not in fields and "name" not in fields


def test_a_blank_key_matches_nobody():
    store = _store(("k-ana", rbac.ADMIN))
    assert store.get("") is None
    assert rbac.role_for(store, "") == RBAC.default_role
    assert rbac.is_admin(store, "") is False


# --- persistence --------------------------------------------------------------

def test_assignments_survive_a_round_trip(tmp_path):
    path = tmp_path / "r.json"
    rbac.save_store(_store(("k-ana", rbac.ADMIN), ("k-ben", rbac.VIEWER)), path)
    loaded = rbac.load_store(path)
    assert rbac.role_for(loaded, "k-ana") == rbac.ADMIN
    assert rbac.role_for(loaded, "k-ben") == rbac.VIEWER


def test_an_unreadable_store_is_corrupt_not_empty(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{not json")
    assert rbac.load_store(path).corrupt is True


def test_a_corrupt_store_is_never_written_over(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{not json")
    assert rbac.save_store(rbac.RoleStore(corrupt=True), path) is False
    assert path.read_text() == "{not json"


def test_an_unknown_role_on_disk_is_dropped(tmp_path):
    """A hand-edited file must not be able to invent a privilege level."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"assignments": [
        {"user_key": "k-ana", "role": "admin"},
        {"user_key": "k-evil", "role": "superuser"},
        {"user_key": "", "role": "admin"}, "nonsense",
    ]}))
    store = rbac.load_store(path)
    assert [a.user_key for a in store.assignments] == ["k-ana"]
    assert store.corrupt is False


def test_a_duplicated_key_on_disk_keeps_the_first(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"assignments": [
        {"user_key": "k-ana", "role": "viewer"},
        {"user_key": "k-ana", "role": "admin"},
    ]}))
    assert rbac.role_for(rbac.load_store(path), "k-ana") == rbac.VIEWER


def test_the_store_is_shared_not_per_user(monkeypatch):
    """A per-user role file would let everyone grant themselves Admin."""
    import local_store
    monkeypatch.setattr(local_store, "_namespace_provider", lambda: "somekey",
                        raising=False)
    assert local_store.current_namespace() == "somekey"
    assert rbac._store_path() == local_store.app_dir() / RBAC.store_filename


def test_the_store_is_declared_shared_in_auth():
    import auth
    assert RBAC.store_filename in auth.SHARED_STORES
    assert RBAC.store_filename not in auth.PER_USER_STORES


def test_the_store_is_gitignored():
    ignored = (Path(rbac.__file__).resolve().parent / ".gitignore").read_text()
    assert RBAC.store_filename in ignored


# --- what is claimed ----------------------------------------------------------

def test_the_limits_of_the_boundary_are_stated():
    """Calling an in-process check 'access control' without qualification
    promises more than any of it can deliver."""
    # Both halves: what it DOES and where it stops. A disclosure with
    # only the limits reads as an apology; only the mechanism reads as a
    # promise.
    assert "checks before it acts" in rbac.WHAT_THIS_IS
    assert "hidden rather than shown and refused" in rbac.WHAT_THIS_IS
    assert "not protection against someone with shell or file access" \
        in rbac.WHAT_THIS_IS
    assert "whoever owns the disk owns the data" in rbac.WHAT_THIS_IS
    assert "real boundary where people reach the app" in rbac.WHAT_THIS_IS


def test_the_bootstrap_rule_is_stated():
    note = rbac.bootstrap_note()
    assert "first account" in note and "administrator" in note
    assert "Viewer" in note


# --- the collaboration widening ----------------------------------------------

def test_an_admin_can_moderate_anyone_s_note():
    """The integration collaboration.can_moderate's own docstring
    predicted: widen it, and nothing else changes."""
    import collaboration
    note = collaboration.Note("n", "AAPL", "Ana", "body", "t",
                              authenticated=True, author_key="k-ana")
    assert collaboration.can_moderate(note, "k-ben") is False
    assert collaboration.can_moderate(note, "k-ben", is_admin=True) is True


def test_the_author_still_moderates_their_own_without_a_role():
    import collaboration
    note = collaboration.Note("n", "AAPL", "Ana", "body", "t",
                              authenticated=True, author_key="k-ana")
    assert collaboration.can_moderate(note, "k-ana") is True


def test_not_even_an_admin_can_moderate_an_unowned_note():
    """A note written signed-out has no owner, so there is no
    accountability either way — and the UI already says so rather than
    drawing a control that would surprise."""
    import collaboration
    legacy = collaboration.Note("old", "AAPL", "Typed", "body", "t")
    assert collaboration.can_moderate(legacy, "k-ana", is_admin=True) is False


def test_collaboration_does_not_import_rbac():
    """It is loaded by the digest and alert cron scripts, which have no
    session and no role to resolve — the caller passes the answer."""
    import collaboration
    src = Path(collaboration.__file__).read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    assert not any("rbac" in ln for ln in imports)


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Roles ---")
    end = src.index("# --- end roles ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_states_what_the_boundary_is_and_is_not():
    assert "RBAC_WHAT_THIS_IS" in _panel()


def test_the_panel_grants_through_the_module():
    panel = _panel()
    assert "rbac_grant(" in panel
    assert "rbac.ROLES" in panel or "ROLE_SPECS" in panel


def test_the_role_panel_is_itself_gated():
    panel = _panel()
    assert "admin.roles" in panel


def test_the_app_bootstraps_the_first_admin():
    src = FINANCE.read_text()
    assert "rbac_bootstrap(" in src


def test_gated_actions_check_before_acting_not_only_in_the_ui():
    """Hiding a control is not enforcement — a stale rerun or a role
    changed mid-session would otherwise slip through."""
    import re
    src = FINANCE.read_text()

    # The check must sit in a GUARD, not merely appear somewhere. An
    # earlier version matched any occurrence, so replacing the guard with
    # `elif False:` still passed because the same call survived inside
    # the branch body that renders the refusal — the `if False:` trap the
    # badges suite already records.
    guards = set(re.findall(
        r'(?:^|\n)\s*(?:if|elif)\s+(?:not\s+)?rbac_require\(\s*_my_role\s*,'
        r'\s*"([a-z_.]+)"', src))
    guards |= set(re.findall(
        r'_\w+_gate\s*=\s*rbac_require\(\s*_my_role\s*,\s*"([a-z_.]+)"', src))

    for required in ("admin.audit_read", "admin.api_keys", "admin.webhooks",
                     "admin.roles", "shared.post_note", "shared.contest_enter",
                     "shared.leaderboard_join", "shared.peer_publish"):
        assert required in guards, f"nothing GUARDS on '{required}' before acting"

    used = set(re.findall(r'rbac_require\(\s*_my_role\s*,\s*"([a-z_.]+)"', src))
    assert used <= set(rbac.PERMISSION_MAP), f"unknown: {used - set(rbac.PERMISSION_MAP)}"


def test_every_admin_permission_the_app_uses_is_declared():
    import re
    src = FINANCE.read_text()
    used = set(re.findall(r'"(admin\.[a-z_]+)"', src))
    assert used <= set(rbac.PERMISSION_MAP), f"undeclared: {used - set(rbac.PERMISSION_MAP)}"
