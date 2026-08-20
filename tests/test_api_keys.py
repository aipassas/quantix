"""Tests for api_keys.py — scoped credentials for robot access.

These are credential tests, so the properties that matter are the ones
that fail SILENTLY and dangerously if they regress:

  - the plaintext secret must never be persisted;
  - a revoked or expired key must stop working, not merely look inactive;
  - scope checks must deny by default;
  - a corrupt record must fail CLOSED (deny) rather than open.

Everything else here is ordinary behaviour coverage.
"""
import datetime
import json

import pytest

from api_keys import (
    DEFAULT_SCOPES,
    SCOPES,
    ApiKey,
    ApiKeyStore,
    create_key,
    delete_key,
    hash_key,
    keys_for_owner,
    load_store,
    mark_used,
    parse_key,
    revoke_key,
    save_store,
    verify_key,
)
from config import API_KEYS


def _issue(store=None, name="robot", scopes=DEFAULT_SCOPES, owner="", days=None):
    store = store if store is not None else ApiKeyStore()
    store, key, plaintext, err = create_key(store, name, scopes, owner_key=owner, expires_in_days=days)
    assert err is None, err
    return store, key, plaintext


# --- the secret ---------------------------------------------------------------

def test_issued_key_has_the_expected_shape():
    _, key, plaintext = _issue()
    prefix, key_id, secret = plaintext.split("_", 2)
    assert prefix == API_KEYS.key_prefix
    assert key_id == key.id
    assert len(secret) >= 40  # 32 random bytes, urlsafe-encoded


def test_the_plaintext_secret_is_never_stored(tmp_path):
    """The single most important property in this module. If the secret
    reaches disk, the store becomes the thing an attacker wants."""
    path = tmp_path / "k.json"
    store, key, plaintext = _issue()
    save_store(store, path)
    raw = path.read_text()
    assert plaintext not in raw
    _, _, secret = plaintext.split("_", 2)
    assert secret not in raw
    assert key.hashed in raw


def test_the_stored_hash_matches_the_plaintext():
    _, key, plaintext = _issue()
    assert key.hashed == hash_key(plaintext)


def test_two_keys_never_collide():
    store, first, first_plain = _issue()
    store, second, second_plain = _issue(store, name="second")
    assert first.id != second.id
    assert first_plain != second_plain
    assert first.hashed != second.hashed


def test_key_ids_are_unique_across_many_issues():
    store = ApiKeyStore()
    for i in range(API_KEYS.max_keys_per_owner):
        store, _, _ = _issue(store, name=f"robot-{i}")
    assert len({k.id for k in store.keys}) == API_KEYS.max_keys_per_owner


# --- verification -------------------------------------------------------------

def test_a_valid_key_verifies():
    store, key, plaintext = _issue()
    resolved, err = verify_key(store, plaintext)
    assert err is None and resolved.id == key.id


def test_an_unknown_key_is_rejected():
    store, _, _ = _issue()
    resolved, err = verify_key(store, "qtx_ffffffff_notarealsecretatall")
    assert resolved is None and err


def test_a_tampered_secret_is_rejected():
    """Same key id, different secret — the hash comparison is what has to
    catch this, not the id lookup."""
    store, key, plaintext = _issue()
    prefix, key_id, secret = plaintext.split("_", 2)
    tampered = f"{prefix}_{key_id}_{secret[:-2]}xy"
    resolved, err = verify_key(store, tampered)
    assert resolved is None and err


def test_malformed_keys_are_rejected_without_a_store_lookup():
    store, _, _ = _issue()
    for junk in ("", "   ", "hello", "qtx_", "qtx__", "notqtx_abc_def", "qtx abc def"):
        resolved, err = verify_key(store, junk)
        assert resolved is None, junk
        assert err


def test_parse_key_rejects_a_foreign_prefix():
    assert parse_key("xyz_abc123_secret") is None
    assert parse_key(f"{API_KEYS.key_prefix}_abc123_secret") == "abc123"


def test_a_revoked_key_stops_working():
    store, key, plaintext = _issue()
    store = revoke_key(store, key.id)
    resolved, err = verify_key(store, plaintext)
    assert resolved is None and "revoked" in err.lower()


def test_an_expired_key_stops_working():
    store, key, plaintext = _issue(days=1)
    later = datetime.datetime.now() + datetime.timedelta(days=2)
    resolved, err = verify_key(store, plaintext, now=later)
    assert resolved is None and "expired" in err.lower()


def test_a_key_is_still_valid_before_its_expiry():
    store, key, plaintext = _issue(days=30)
    soon = datetime.datetime.now() + datetime.timedelta(days=29)
    resolved, err = verify_key(store, plaintext, now=soon)
    assert err is None and resolved.id == key.id


def test_a_never_expiring_key_stays_valid():
    store, key, plaintext = _issue(days=0)
    assert key.expires_at == ""
    far_future = datetime.datetime.now() + datetime.timedelta(days=4000)
    resolved, err = verify_key(store, plaintext, now=far_future)
    assert err is None and resolved.id == key.id


def test_an_unparseable_expiry_fails_closed():
    """A corrupt timestamp on a credential must read as EXPIRED, not as
    'no expiry'. Failing open here would turn file corruption into a
    permanently valid key."""
    key = ApiKey(id="a", name="n", hashed="h", scopes=("quote:read",), expires_at="not-a-date")
    assert key.is_expired() is True
    assert key.is_usable() is False


# --- scopes -------------------------------------------------------------------

def test_scope_checks_are_exact():
    _, key, _ = _issue(scopes=("quote:read",))
    assert key.has_scope("quote:read")
    assert not key.has_scope("fundamentals:read")
    assert not key.has_scope("quote")
    assert not key.has_scope("")


def test_unknown_scopes_are_discarded_at_creation():
    """A typo'd scope must not be silently stored — it would look granted
    in the UI while denying at the endpoint, or worse."""
    store, key, _ = _issue(scopes=("quote:read", "admin:everything"))
    assert key.scopes == ("quote:read",)


def test_a_key_with_no_valid_scopes_is_refused():
    store, key, plaintext, err = create_key(ApiKeyStore(), "robot", ("nonsense:scope",))
    assert key is None and err and "scope" in err.lower()


def test_unknown_scopes_are_dropped_when_loading(tmp_path):
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"keys": [{
        "id": "abc", "name": "n", "hashed": "h",
        "scopes": ["quote:read", "admin:everything"],
    }]}))
    assert load_store(path).keys[0].scopes == ("quote:read",)


def test_every_default_scope_is_a_real_scope():
    assert all(s in SCOPES for s in DEFAULT_SCOPES)


def test_no_scope_grants_writes():
    """Guards the read-only contract at the data level: if someone adds a
    write scope later, this fails and forces the decision to be explicit
    rather than arriving by accident."""
    for name in SCOPES:
        assert name.endswith(":read"), f"{name} is not a read scope"


# --- creation rules -----------------------------------------------------------

def test_a_key_needs_a_name():
    _, key, _, err = create_key(ApiKeyStore(), "   ")
    assert key is None and err


def test_name_length_is_capped():
    _, key, _, err = create_key(ApiKeyStore(), "x" * (API_KEYS.max_name_chars + 1))
    assert key is None and err


def test_key_count_is_capped_per_owner():
    store = ApiKeyStore()
    for i in range(API_KEYS.max_keys_per_owner):
        store, _, _ = _issue(store, name=f"r{i}", owner="alice")
    _, key, _, err = create_key(store, "one-too-many", owner_key="alice")
    assert key is None and err


def test_revoked_keys_do_not_count_against_the_cap():
    store = ApiKeyStore()
    for i in range(API_KEYS.max_keys_per_owner):
        store, _, _ = _issue(store, name=f"r{i}", owner="alice")
    store = revoke_key(store, store.keys[0].id)
    _, key, _, err = create_key(store, "replacement", owner_key="alice")
    assert err is None and key is not None


def test_the_cap_is_per_owner_not_global():
    store = ApiKeyStore()
    for i in range(API_KEYS.max_keys_per_owner):
        store, _, _ = _issue(store, name=f"r{i}", owner="alice")
    _, key, _, err = create_key(store, "bobs-first", owner_key="bob")
    assert err is None and key is not None


@pytest.mark.parametrize("days", [-1, API_KEYS.max_expiry_days + 1, "abc"])
def test_invalid_expiry_is_refused(days):
    _, key, _, err = create_key(ApiKeyStore(), "robot", expires_in_days=days)
    assert key is None and err


# --- ownership ----------------------------------------------------------------

def test_keys_are_listed_per_owner():
    store = ApiKeyStore()
    store, alice_key, _ = _issue(store, name="alice-bot", owner="google-alice")
    store, bob_key, _ = _issue(store, name="bob-bot", owner="google-bob")

    assert [k.id for k in keys_for_owner(store, "google-alice")] == [alice_key.id]
    assert [k.id for k in keys_for_owner(store, "google-bob")] == [bob_key.id]
    assert keys_for_owner(store, "") == ()


def test_a_key_remembers_its_owner_so_the_server_can_scope_data():
    """The server has no session — owner_key on the record is the ONLY
    thing that tells an owner-scoped endpoint whose data to read."""
    _, key, _ = _issue(owner="google-alice")
    assert key.owner_key == "google-alice"


def test_signed_out_keys_have_an_empty_owner():
    _, key, _ = _issue()
    assert key.owner_key == ""


# --- lifecycle ----------------------------------------------------------------

def test_revoking_keeps_the_record_visible():
    """Deleting outright would make an unexplained 401 in a robot's log
    much harder to diagnose."""
    store, key, _ = _issue()
    store = revoke_key(store, key.id)
    assert len(store.keys) == 1
    assert store.keys[0].status == "revoked"


def test_revoking_twice_keeps_the_first_timestamp():
    store, key, _ = _issue()
    store = revoke_key(store, key.id)
    first = store.keys[0].revoked_at
    store = revoke_key(store, key.id)
    assert store.keys[0].revoked_at == first


def test_delete_removes_the_record_entirely():
    store, key, _ = _issue()
    assert delete_key(store, key.id).keys == ()


def test_status_reflects_lifecycle():
    store, key, _ = _issue(days=0)
    assert store.keys[0].status == "active"
    assert revoke_key(store, key.id).keys[0].status == "revoked"


def test_mark_used_records_a_timestamp():
    store, key, _ = _issue()
    assert store.keys[0].last_used_at == ""
    assert mark_used(store, key.id).keys[0].last_used_at != ""


# --- persistence --------------------------------------------------------------

def test_round_trip(tmp_path):
    path = tmp_path / "k.json"
    store, key, plaintext = _issue(name="nightly", scopes=("quote:read", "risk:read"), owner="google-a", days=10)
    save_store(store, path)
    loaded = load_store(path)
    assert len(loaded.keys) == 1
    restored = loaded.keys[0]
    assert (restored.id, restored.name, restored.owner_key) == (key.id, "nightly", "google-a")
    assert restored.scopes == ("quote:read", "risk:read")
    # And it still verifies after a round trip through disk.
    resolved, err = verify_key(loaded, plaintext)
    assert err is None and resolved.id == key.id


def test_missing_store_is_empty(tmp_path):
    assert load_store(tmp_path / "nope.json") == ApiKeyStore()


def test_corrupt_store_degrades_to_empty(tmp_path):
    """Degrading to empty means every key stops working — which is the
    safe direction for a credential store that can't be read."""
    path = tmp_path / "k.json"
    path.write_text("{not json")
    assert load_store(path) == ApiKeyStore()


def test_a_record_missing_its_hash_is_dropped(tmp_path):
    """A record with no hash could never be verified anyway; keeping it
    would just show a phantom key in the UI."""
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"keys": [
        {"id": "good", "name": "n", "hashed": "abc", "scopes": ["quote:read"]},
        {"id": "bad", "name": "n", "scopes": ["quote:read"]},
        {"name": "no-id", "hashed": "abc"},
    ]}))
    assert [k.id for k in load_store(path).keys] == ["good"]


def test_save_leaves_no_leftover_temp_file(tmp_path):
    path = tmp_path / "k.json"
    store, _, _ = _issue()
    save_store(store, path)
    assert [p.name for p in tmp_path.iterdir()] == ["k.json"]
