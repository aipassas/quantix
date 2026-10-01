"""Who is on this instance — completely, and without inventing a licence.

THE TEST THAT MATTERS MOST is test_an_identity_with_no_local_account_is
_still_listed. accounts.py knows only local email/password accounts;
measured on the real instance, 2 identities have data and accounts.py can
see 1. A dashboard built from that table would show half the users here
and NONE on a firm signing in through Okta — the deployment this exists
for.

The second is test_no_seat_count_is_invented. Nothing in this codebase
has a plan, a quota or a bill, so "7 of 10 seats used" would be an
entitlement measured against nothing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import admin_dashboard as ad
import rbac

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def _code_only(source: str) -> str:
    """Source with docstrings stripped, so a ban list checks what the
    module DOES rather than what it says about itself."""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


class _Account:
    def __init__(self, user_id, email="", name="", created_at="2026-01-01",
                 last_login_at="2026-10-01"):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.created_at = created_at
        self.last_login_at = last_login_at


def _key(user_id):
    import auth
    return auth.key_for(auth.LOCAL_ISSUER, user_id)


def _roles(*pairs):
    return rbac.RoleStore(tuple(
        rbac.Assignment(k, r, "k-granter", "2026-10-01T00:00:00")
        for k, r in pairs))


# --- completeness -------------------------------------------------------------

def test_an_identity_with_no_local_account_is_still_listed():
    """The OIDC case, which is the normal one on the deployment this is
    for — and the one accounts.py cannot see at all."""
    people = ad.members([ad.Namespace("google-abc", 8, 4000)], [], rbac.RoleStore())
    assert len(people) == 1
    assert people[0].user_key == "google-abc"
    assert people[0].source == ad.SIGN_IN_ONLY
    assert people[0].known_locally is False


def test_a_local_account_is_matched_to_its_namespace():
    account = _Account("u1", "ana@example.com", "Ana")
    people = ad.members([ad.Namespace(_key("u1"), 4, 2000)], [account],
                        rbac.RoleStore())
    assert len(people) == 1, "the account and the namespace are one person"
    assert people[0].email == "ana@example.com"
    assert people[0].source == ad.LOCAL_ACCOUNT


def test_an_account_that_never_signed_in_is_still_listed():
    """It has no namespace because it has stored nothing — but an
    administrator asking who has access still needs to see it."""
    people = ad.members([], [_Account("u1", "ana@example.com")], rbac.RoleStore())
    assert len(people) == 1
    assert people[0].email == "ana@example.com"
    assert people[0].files == 0 and people[0].bytes_used == 0


def test_the_two_sources_are_merged_not_double_counted():
    account = _Account("u1", "ana@example.com")
    people = ad.members(
        [ad.Namespace(_key("u1"), 4, 2000), ad.Namespace("google-abc", 8, 4000)],
        [account], rbac.RoleStore())
    assert len(people) == 2
    assert {m.user_key for m in people} == {_key("u1"), "google-abc"}


def test_the_account_key_is_derived_through_auth_not_rebuilt():
    """A second copy of that derivation is how a dashboard starts showing
    a row that matches nobody."""
    import auth
    assert ad.key_for_account(_Account("u1")) == auth.key_for(auth.LOCAL_ISSUER, "u1")
    src = Path(ad.__file__).read_text()
    assert "hashlib" not in src and "sha256" not in src


def test_an_account_with_no_id_yields_no_key():
    assert ad.key_for_account(_Account("")) == ""


def test_the_coverage_note_says_how_complete_the_list_is():
    people = ad.members(
        [ad.Namespace(_key("u1"), 1, 1), ad.Namespace("google-abc", 1, 1)],
        [_Account("u1", "ana@example.com")], rbac.RoleStore())
    note = ad.coverage_note(people)
    assert "1 of 2 signed in through an identity provider" in note


def test_the_coverage_note_says_so_when_everyone_is_known():
    people = ad.members([ad.Namespace(_key("u1"), 1, 1)],
                        [_Account("u1", "ana@example.com")], rbac.RoleStore())
    assert "Every identity here has a local account" in ad.coverage_note(people)


def test_an_unknown_identity_is_described_not_hidden():
    """The note must explain the CAUSE, not just the symptom. An earlier
    version checked only the two closing phrases, which survived a build
    whose opening sentence no longer said why the email is missing — so
    a reader was left to assume the app was withholding it."""
    note = ad.UNKNOWN_IDENTITY_NOTE
    assert "identity provider" in note, "the note must say WHY there is no email"
    assert "rather than with" in note and "password" in note
    assert "holds no address or display name" in note
    assert "not everything it is hiding" in note


# --- no invented seats --------------------------------------------------------

def test_no_seat_count_is_invented():
    """Nothing here has a plan, a quota or a bill."""
    # CODE only. The module's own docstring explains why a seat count
    # would be an ENTITLEMENT measured against nothing, and a bare
    # substring check matches that explanation rather than any code —
    # the fourth time this trap has fired in this codebase.
    src = _code_only(Path(ad.__file__).read_text())
    for banned in ("seats_used", "seat_limit", "max_seats", "entitlement",
                   "licence_key", "license_key", "plan_tier"):
        assert banned not in src, banned
    # No numeric seat allowance anywhere in the dataclasses.
    import dataclasses
    for cls in (ad.Member, ad.Overview, ad.Namespace):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not any("seat" in n or "quota" in n or "limit" in n for n in names)


def test_the_absence_of_a_licence_model_is_stated():
    assert "no licensing, billing or plan" in ad.NO_LICENCE_MODEL
    assert "nothing to count seats against" in ad.NO_LICENCE_MODEL
    assert "real number of identities" in ad.NO_LICENCE_MODEL


# --- no deletion --------------------------------------------------------------

def test_the_module_cannot_delete_anybody():
    """Destroying somebody's journal, portfolio and watchlists from a
    dashboard is irreversible."""
    src = Path(ad.__file__).read_text()
    for banned in ("rmtree", "unlink(", "os.remove", "delete_account",
                   "shutil"):
        assert banned not in src, banned
    assert not any(n for n in dir(ad)
                   if n.lower() in ("delete", "remove", "deactivate", "purge"))


def test_the_no_deletion_policy_is_stated():
    assert "Nobody can be deleted from this panel" in ad.NO_DELETION_NOTE
    assert "Setting someone to Viewer" in ad.NO_DELETION_NOTE
    assert "audit trail" in ad.NO_DELETION_NOTE


# --- roles --------------------------------------------------------------------

def test_a_role_is_read_from_the_role_store():
    people = ad.members([ad.Namespace("k-ana", 1, 1)], [],
                        _roles(("k-ana", rbac.ADMIN)))
    assert people[0].role == rbac.ADMIN
    assert people[0].role_label == "Admin"


def test_an_unassigned_identity_shows_the_default_role():
    from config import RBAC
    people = ad.members([ad.Namespace("k-new", 1, 1)], [], rbac.RoleStore())
    assert people[0].role == RBAC.default_role


def test_administrators_are_listed_first():
    """The people who can change things belong at the top."""
    store = _roles(("k-viewer", rbac.VIEWER), ("k-admin", rbac.ADMIN),
                   ("k-analyst", rbac.ANALYST))
    people = ad.members([ad.Namespace(k, 1, 1) for k in
                         ("k-viewer", "k-analyst", "k-admin")], [], store)
    assert [m.role for m in people] == [rbac.ADMIN, rbac.ANALYST, rbac.VIEWER]


def test_the_order_is_stable_between_renders():
    store = _roles(("k-a", rbac.VIEWER), ("k-b", rbac.VIEWER))
    spaces = [ad.Namespace("k-b", 1, 1), ad.Namespace("k-a", 1, 1)]
    first = [m.user_key for m in ad.members(spaces, [], store)]
    second = [m.user_key for m in ad.members(list(reversed(spaces)), [], store)]
    assert first == second == ["k-a", "k-b"]


# --- the overview -------------------------------------------------------------

def test_the_overview_counts_every_role():
    store = _roles(("k-admin", rbac.ADMIN), ("k-analyst", rbac.ANALYST))
    people = ad.members([ad.Namespace(k, 1, 10) for k in
                         ("k-admin", "k-analyst", "k-viewer")], [], store)
    result = ad.overview(people)
    assert result.total == 3
    assert result.by_role == {rbac.ADMIN: 1, rbac.ANALYST: 1, rbac.VIEWER: 1}
    assert result.admins == 1
    assert result.total_bytes == 30


def test_the_overview_splits_local_accounts_from_sign_in_only():
    people = ad.members(
        [ad.Namespace(_key("u1"), 1, 1), ad.Namespace("google-abc", 1, 1)],
        [_Account("u1", "ana@example.com")], rbac.RoleStore())
    result = ad.overview(people)
    assert result.local_accounts == 1 and result.sign_in_only == 1


def test_an_empty_instance_says_so_rather_than_showing_zeroes():
    result = ad.overview(())
    assert result.total == 0
    assert "Nobody has signed in" in result.sentence()


def test_the_sentence_agrees_in_number():
    one = ad.overview(ad.members([ad.Namespace("k", 1, 1)], [],
                                 _roles(("k", rbac.ADMIN))))
    assert "1 identity have" not in one.sentence()
    assert "1 identity has signed in" in one.sentence()
    assert "1 administrator," in one.sentence()


# --- display ------------------------------------------------------------------

def test_a_name_is_preferred_then_an_email_then_the_key():
    assert ad.Member("k", rbac.VIEWER, ad.LOCAL_ACCOUNT,
                     email="a@b.c", name="Ana").display == "Ana"
    assert ad.Member("k", rbac.VIEWER, ad.LOCAL_ACCOUNT,
                     email="a@b.c").display == "a@b.c"
    assert ad.Member("k", rbac.VIEWER, ad.SIGN_IN_ONLY).display == "k"


def test_nothing_stored_is_not_reported_as_zero_bytes():
    """A zero reads as a measurement when it is an absence — the same
    rule the workbook export follows for a blank cell."""
    assert ad.format_bytes(0) == "nothing stored"
    assert ad.format_bytes(-1) == "nothing stored"


def test_bytes_are_formatted_at_a_readable_scale():
    assert ad.format_bytes(512) == "512 B"
    assert "KB" in ad.format_bytes(2381)
    assert "MB" in ad.format_bytes(5 * 1024 * 1024)


# --- robustness ---------------------------------------------------------------

def test_a_missing_users_directory_is_empty_not_a_crash(tmp_path):
    assert ad.scan(tmp_path / "nope") == ()


def test_scan_counts_files_and_bytes(tmp_path):
    root = tmp_path / "users"
    (root / "k-ana").mkdir(parents=True)
    (root / "k-ana" / "a.json").write_text("x" * 100)
    (root / "k-ana" / "b.json").write_text("y" * 50)
    (root / "loose.txt").write_text("not a namespace")
    spaces = ad.scan(root)
    assert len(spaces) == 1, "a loose file is not an identity"
    assert spaces[0].user_key == "k-ana"
    assert spaces[0].files == 2 and spaces[0].bytes_used == 150


def test_scan_is_sorted_so_the_listing_is_stable(tmp_path):
    root = tmp_path / "users"
    for name in ("k-zoe", "k-ana", "k-mia"):
        (root / name).mkdir(parents=True)
    assert [n.user_key for n in ad.scan(root)] == ["k-ana", "k-mia", "k-zoe"]


def test_a_malformed_account_does_not_take_the_panel_down():
    class _Broken:
        pass
    people = ad.members([ad.Namespace("k-ana", 1, 1)], [_Broken()],
                        rbac.RoleStore())
    assert len(people) == 1


def test_the_module_reads_the_disk_in_exactly_one_place():
    """Everything else takes data, so the filesystem surface is small
    enough to see at a glance."""
    src = Path(ad.__file__).read_text()
    assert src.count("iterdir()") == 1
    assert src.count("rglob(") == 1


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Admin dashboard ---")
    end = src.index("# --- end admin dashboard ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_dashboard_is_gated_on_admin():
    panel = _panel()
    assert "admin.roles" in panel


def test_the_dashboard_lists_through_the_module():
    panel = _panel()
    assert "ad_scan(" in panel
    assert "ad_members(" in panel
    assert "ad_overview(" in panel
    assert "sorted(" not in panel, "the ordering belongs in the module"


def test_the_dashboard_states_there_is_no_licence_model():
    assert "AD_NO_LICENCE_MODEL" in _panel()


def test_the_dashboard_states_that_nobody_can_be_deleted():
    assert "AD_NO_DELETION_NOTE" in _panel()


def test_the_dashboard_explains_the_unknown_identities():
    assert "AD_UNKNOWN_IDENTITY_NOTE" in _panel()


def test_the_roles_panel_was_absorbed_rather_than_duplicated():
    """Two controls writing one store is how they drift apart."""
    src = FINANCE.read_text()
    assert src.count('key="rbac_grant"') == 1
    assert src.index("# --- Admin dashboard ---") < src.index("# --- Roles ---")
    assert src.index("# --- end roles ---") < src.index("# --- end admin dashboard ---")
