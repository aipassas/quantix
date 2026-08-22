"""Tests for local email+password accounts.

The properties worth defending are the ones an attacker probes, not the
happy path — signing in with the right password is the easy half.

Every test runs against a SANDBOXED app_dir. accounts.py writes through
local_store.shared_path(), so without the fixture below these would write
real accounts into the developer's own instance, and one careless test
would leave a live credential record behind.
"""
import datetime
import json

import pytest

import local_store
import passwords


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point every store path at a temp dir for the duration of a test."""
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    import accounts as accounts_module
    return tmp_path


@pytest.fixture
def accounts():
    import accounts as module
    return module


def _make(accounts, email="a@example.com", password="Tr0ub4dor&3xx", name="A"):
    account, error = accounts.create_account(email, password, name)
    assert error is None, error
    return account


# --- hashing ------------------------------------------------------------------

def test_password_is_never_stored_in_the_clear(accounts, sandbox):
    secret = "Tr0ub4dor&3xx"
    _make(accounts, password=secret)
    written = (sandbox / accounts.STORE_FILENAME).read_text()
    assert secret not in written
    assert "scrypt$" in written


def test_same_password_hashes_differently_each_time():
    first = passwords.hash_password("Tr0ub4dor&3xx")
    second = passwords.hash_password("Tr0ub4dor&3xx")
    assert first != second                      # per-hash salt
    assert passwords.verify_password("Tr0ub4dor&3xx", first)
    assert passwords.verify_password("Tr0ub4dor&3xx", second)


def test_verify_rejects_a_corrupt_record_instead_of_raising():
    for junk in ("", "not-a-record", "scrypt$x$y$z$q$r", "scrypt$1$2$3"):
        assert passwords.verify_password("anything", junk) is False


def test_work_factor_is_recorded_so_it_can_be_raised_later():
    weak = passwords.hash_password("Tr0ub4dor&3xx", n=2 ** 12)
    # Still verifiable at its own cost — raising the constant must not
    # lock out every existing user.
    assert passwords.verify_password("Tr0ub4dor&3xx", weak)
    assert passwords.needs_rehash(weak)
    assert not passwords.needs_rehash(passwords.hash_password("Tr0ub4dor&3xx"))


def test_a_stronger_hash_is_not_downgraded():
    strong = passwords.hash_password("Tr0ub4dor&3xx", n=passwords.SCRYPT_N * 2)
    assert not passwords.needs_rehash(strong)


def test_unicode_normalisation(accounts):
    """The same passphrase typed on two keyboards must verify."""
    composed = "café-passphrase-9"          # é as one code point
    decomposed = "café-passphrase-9"       # e + combining acute
    record = passwords.hash_password(composed)
    assert passwords.verify_password(decomposed, record)


def test_successful_signin_upgrades_a_weak_hash(accounts):
    account = _make(accounts)
    stale = passwords.hash_password("Tr0ub4dor&3xx", n=2 ** 12)
    account.password_hash = stale
    accounts._persist(account)

    signed, error = accounts.authenticate("a@example.com", "Tr0ub4dor&3xx")
    assert error is None and signed is not None
    assert accounts.get_by_id(account.user_id).password_hash != stale
    assert not passwords.needs_rehash(accounts.get_by_id(account.user_id).password_hash)


# --- enumeration resistance ---------------------------------------------------

def test_wrong_password_and_unknown_account_are_indistinguishable(accounts):
    _make(accounts)
    _, wrong = accounts.authenticate("a@example.com", "definitely-not-it")
    _, missing = accounts.authenticate("nobody@example.com", "definitely-not-it")
    assert wrong == missing


def test_the_unknown_account_path_still_does_the_work(accounts):
    """Otherwise response time answers 'does this address have an account?'"""
    import time

    _make(accounts)
    start = time.perf_counter()
    accounts.authenticate("a@example.com", "wrong")
    real = time.perf_counter() - start

    start = time.perf_counter()
    accounts.authenticate("nobody@example.com", "wrong")
    missing = time.perf_counter() - start

    # Same order of magnitude. Generous bounds because CI machines are
    # noisy; the failure this catches is missing==~0 from an early return.
    assert missing > real * 0.4


# --- lockout ------------------------------------------------------------------

def test_repeated_failures_lock_the_account(accounts):
    _make(accounts)
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK):
        accounts.authenticate("a@example.com", "wrong")
    assert accounts.get_by_email("a@example.com").locked


def test_the_correct_password_is_refused_while_locked(accounts):
    """A lock that the right password walks through is not a lock."""
    _make(accounts)
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK):
        accounts.authenticate("a@example.com", "wrong")

    signed, error = accounts.authenticate("a@example.com", "Tr0ub4dor&3xx")
    assert signed is None
    assert "Too many failed attempts" in error


def test_the_lock_is_announced_on_the_attempt_that_causes_it(accounts):
    _make(accounts)
    last = None
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK):
        _, last = accounts.authenticate("a@example.com", "wrong")
    assert "Too many failed attempts" in last


def test_lockout_lengthens_with_repeated_rounds(accounts, monkeypatch):
    account = _make(accounts)
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK):
        accounts.authenticate("a@example.com", "wrong")
    first = accounts.get_by_email("a@example.com")
    assert first.lock_level == 1

    # Expire the lock, then fail another round.
    first.locked_until = None
    accounts._persist(first)
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK):
        accounts.authenticate("a@example.com", "wrong")
    second = accounts.get_by_email("a@example.com")
    assert second.lock_level == 2
    assert second.lock_remaining_seconds() > first.lock_remaining_seconds()


def test_a_success_clears_the_failure_counter(accounts):
    _make(accounts)
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK - 1):
        accounts.authenticate("a@example.com", "wrong")
    assert accounts.get_by_email("a@example.com").failed_attempts > 0

    accounts.authenticate("a@example.com", "Tr0ub4dor&3xx")
    fresh = accounts.get_by_email("a@example.com")
    assert fresh.failed_attempts == 0
    assert fresh.lock_level == 0


def test_old_failures_age_out_of_the_window(accounts):
    """Three wrong guesses today and two next week is not an attack."""
    _make(accounts)
    accounts.authenticate("a@example.com", "wrong")
    stale = accounts.get_by_email("a@example.com")
    stale.first_failure_at = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(minutes=accounts.ATTEMPT_WINDOW_MINUTES + 5)
    ).isoformat()
    accounts._persist(stale)

    accounts.authenticate("a@example.com", "wrong")
    assert accounts.get_by_email("a@example.com").failed_attempts == 1


# --- accounts and email -------------------------------------------------------

def test_email_is_normalised_and_duplicates_refused(accounts):
    _make(accounts, email="Angelos@Example.COM ")
    assert accounts.get_by_email("angelos@example.com") is not None
    assert accounts.get_by_email("ANGELOS@EXAMPLE.COM") is not None

    duplicate, error = accounts.create_account("angelos@example.com", "Another1Pass!")
    assert duplicate is None and "already exists" in error


def test_plus_tags_and_dots_stay_distinct(accounts):
    """Canonicalising Gmail-style addresses would merge accounts their
    owner considers separate, and would disagree with the OIDC path."""
    _make(accounts, email="a.b+work@example.com")
    other, error = accounts.create_account("ab@example.com", "Another1Pass!")
    assert error is None and other is not None


def test_a_weak_password_is_refused(accounts):
    for weak in ("short", "password123", "aaaaaaaaaaaaaa"):
        account, error = accounts.create_account("x@example.com", weak)
        assert account is None, weak
        assert error


def test_a_long_passphrase_of_plain_words_is_accepted(accounts):
    """Length carries the strength; rejecting this is how people end up
    at Password1!."""
    account, error = accounts.create_account(
        "x@example.com", "correct horse battery staple")
    assert error is None and account is not None


def test_password_may_not_contain_the_email_or_name(accounts):
    account, error = accounts.create_account(
        "angelos@example.com", "angelos-Passw0rd", "Angelos")
    assert account is None and error


def test_malformed_addresses_are_refused(accounts):
    for bad in ("", "  ", "notanemail", "a@b", "a b@example.com", "@example.com"):
        account, error = accounts.create_account(bad, "Tr0ub4dor&3xx")
        assert account is None, bad


def test_user_id_is_stable_and_not_derived_from_email(accounts):
    """Email is mutable; a namespace keyed on it would orphan saved data."""
    account = _make(accounts)
    assert account.user_id
    assert account.email not in account.user_id


# --- reset --------------------------------------------------------------------

def test_reset_token_is_stored_only_as_a_digest(accounts, sandbox):
    _make(accounts)
    _, token = accounts.begin_reset("a@example.com")
    written = (sandbox / accounts.STORE_FILENAME).read_text()
    assert token not in written
    assert passwords.hash_token(token) in written


def test_reset_for_an_unknown_address_reveals_nothing(accounts):
    assert accounts.begin_reset("nobody@example.com") is None
    # and the caller-facing message is the same either way
    assert "If that address has" in accounts.reset_requested_message()


def test_reset_completes_and_the_token_is_single_use(accounts):
    _make(accounts)
    _, token = accounts.begin_reset("a@example.com")

    ok, error = accounts.complete_reset("a@example.com", token, "N3wPassphrase!x")
    assert ok and error is None
    assert accounts.authenticate("a@example.com", "N3wPassphrase!x")[0] is not None
    assert accounts.authenticate("a@example.com", "Tr0ub4dor&3xx")[0] is None

    again, error = accounts.complete_reset("a@example.com", token, "Another1Pass!")
    assert not again


def test_an_expired_token_is_refused(accounts):
    account = _make(accounts)
    _, token = accounts.begin_reset("a@example.com")
    stale = accounts.get_by_id(account.user_id)
    stale.reset_expires_at = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)
    ).isoformat()
    accounts._persist(stale)

    ok, error = accounts.complete_reset("a@example.com", token, "N3wPassphrase!x")
    assert not ok and "expired" in error


def test_a_wrong_token_is_refused(accounts):
    _make(accounts)
    accounts.begin_reset("a@example.com")
    ok, _ = accounts.complete_reset("a@example.com", "not-the-token", "N3wPassphrase!x")
    assert not ok


def test_a_weak_new_password_does_not_burn_the_token(accounts):
    """Otherwise a typo costs the user a whole new reset email."""
    _make(accounts)
    _, token = accounts.begin_reset("a@example.com")
    ok, _ = accounts.complete_reset("a@example.com", token, "short")
    assert not ok
    ok, error = accounts.complete_reset("a@example.com", token, "N3wPassphrase!x")
    assert ok, error


def test_reset_clears_a_lockout(accounts):
    """Someone locked out by an attacker guessing must still get back in."""
    _make(accounts)
    for _ in range(accounts.MAX_ATTEMPTS_BEFORE_LOCK):
        accounts.authenticate("a@example.com", "wrong")
    assert accounts.get_by_email("a@example.com").locked

    _, token = accounts.begin_reset("a@example.com")
    assert accounts.complete_reset("a@example.com", token, "N3wPassphrase!x")[0]
    assert not accounts.get_by_email("a@example.com").locked
    assert accounts.authenticate("a@example.com", "N3wPassphrase!x")[0] is not None


def test_changing_the_password_invalidates_an_outstanding_reset(accounts):
    """A link mailed an hour ago must not override a password just set."""
    account = _make(accounts)
    _, token = accounts.begin_reset("a@example.com")
    ok, error = accounts.change_password(account.user_id, "Tr0ub4dor&3xx",
                                         "N3wPassphrase!x")
    assert ok, error
    used, _ = accounts.complete_reset("a@example.com", token, "Attacker1Pass!")
    assert not used


def test_change_password_requires_the_current_one(accounts):
    account = _make(accounts)
    ok, error = accounts.change_password(account.user_id, "wrong", "N3wPassphrase!x")
    assert not ok and error


# --- store robustness ---------------------------------------------------------

def test_a_corrupt_store_does_not_raise(accounts, sandbox):
    (sandbox / accounts.STORE_FILENAME).write_text("{ this is not json")
    assert accounts.all_accounts() == []
    assert accounts.get_by_email("a@example.com") is None
    signed, error = accounts.authenticate("a@example.com", "whatever")
    assert signed is None and error


def test_unknown_fields_in_a_record_are_ignored(accounts, sandbox):
    """A store written by a newer version must not crash an older one."""
    _make(accounts)
    raw = json.loads((sandbox / accounts.STORE_FILENAME).read_text())
    for record in raw["accounts"].values():
        record["some_future_field"] = 42
    (sandbox / accounts.STORE_FILENAME).write_text(json.dumps(raw))
    assert accounts.get_by_email("a@example.com") is not None


# --- namespace integration ----------------------------------------------------

def test_each_account_gets_its_own_namespace(accounts):
    """Two accounts must not share a workspace."""
    import auth

    first = _make(accounts, email="a@example.com")
    second = _make(accounts, email="b@example.com")
    key_a = auth.key_for(auth.LOCAL_ISSUER, first.user_id, first.email)
    key_b = auth.key_for(auth.LOCAL_ISSUER, second.user_id, second.email)
    assert key_a and key_b and key_a != key_b


def test_a_local_namespace_cannot_collide_with_an_oidc_one(accounts):
    """Even if a provider ever issued a subject equal to one of our ids."""
    import auth

    account = _make(accounts)
    local = auth.key_for(auth.LOCAL_ISSUER, account.user_id)
    google = auth.key_for("https://accounts.google.com", account.user_id)
    assert local != google
    assert local.startswith("local-")


def test_the_namespace_survives_an_email_change(accounts):
    """Keyed on the immutable id, so changing address keeps the workspace."""
    import auth

    account = _make(accounts, email="old@example.com")
    before = auth.key_for(auth.LOCAL_ISSUER, account.user_id, account.email)

    account.email = "new@example.com"
    accounts._persist(account)
    after = auth.key_for(auth.LOCAL_ISSUER, account.user_id, account.email)
    assert before == after
