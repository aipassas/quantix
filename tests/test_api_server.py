"""Tests for api_server.py — the read-only HTTP API.

Driven over REAL HTTP against a real ThreadingHTTPServer on an ephemeral
port, not by calling handlers directly. The whole point of this module is
the layer between the socket and the handler — header parsing, the
credential check, the scope check, the status codes — and calling the
handlers in-process would skip exactly the code under test.

The data-fetching endpoints are stubbed. These tests are about
authorization, not about Yahoo being reachable; a test suite that needs
the network to tell you whether a 401 works is a test suite that stops
working on a train.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import api_keys
import api_server
from api_keys import ApiKeyStore, create_key, revoke_key


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    """Point the whole key system at sandboxed files.

    BOTH paths must be redirected. Usage timestamps live in their own
    file now (see api_keys._usage_path); leaving that one pointed at the
    real application directory would have the suite writing
    api_keys_usage.json into it on every run.
    """
    path = tmp_path / "api_keys_store.json"
    monkeypatch.setattr(api_keys, "_store_path", lambda: path)
    monkeypatch.setattr(api_keys, "_usage_path", lambda: tmp_path / "api_keys_usage.json")
    return path


@pytest.fixture
def server(store_path):
    """A real server on an ephemeral port, torn down after the test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), api_server._Handler)
    # poll_interval drives how long shutdown() blocks; the 0.5s default
    # would add ~15s to this file for no benefit.
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def issue(store_path, scopes=("quote:read",), owner="", days=None):
    store = api_keys.load_store(store_path)
    store, key, plaintext, err = create_key(store, "test-robot", scopes, owner_key=owner, expires_in_days=days)
    assert err is None, err
    api_keys.save_store(store, store_path)
    return key, plaintext


def get(base, path, key=None, header="Authorization"):
    """Returns (status, parsed_json)."""
    request = urllib.request.Request(base + path)
    if key:
        request.add_header(header, f"Bearer {key}" if header == "Authorization" else key)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(autouse=True)
def _stub_data(monkeypatch):
    """Stub the endpoints that would otherwise hit the network."""
    monkeypatch.setattr(api_server, "_handler_quote",
                        lambda key, query: {"ticker": api_server._ticker_arg(query), "stubbed": True})
    monkeypatch.setattr(api_server, "_handler_fundamentals",
                        lambda key, query: {"ticker": api_server._ticker_arg(query), "stubbed": True})
    monkeypatch.setattr(api_server, "_handler_risk",
                        lambda key, query: {"ticker": api_server._ticker_arg(query), "stubbed": True})
    monkeypatch.setattr(api_server, "_handler_watchlists",
                        lambda key, query: {"owner": key.owner_key if key else None})


# --- discovery ----------------------------------------------------------------

def test_index_is_reachable_without_a_key(server):
    """An integrator has to be able to see what exists before wiring up a
    credential."""
    status, body = get(server, "/v1")
    assert status == 200
    assert body["service"] == "Quantix API"


def test_index_states_the_read_only_and_no_trading_contract(server):
    """The originating task said "headless trading bots". The API itself
    has to be the thing that says it cannot trade, so nobody wires a key
    up and discovers it later."""
    _, body = get(server, "/v1")
    assert body["read_only"] is True
    assert body["trading_supported"] is False
    assert "read-only" in body["notice"].lower()


def test_index_lists_every_route_and_its_scope(server):
    _, body = get(server, "/v1")
    assert set(body["endpoints"]) == set(api_server._ROUTES)
    assert body["endpoints"]["/v1/quote"]["scope"] == "quote:read"
    assert body["endpoints"]["/v1"]["scope"] is None


def test_unknown_paths_404(server):
    status, body = get(server, "/v1/nonsense")
    assert status == 404 and "error" in body


# --- authentication -----------------------------------------------------------

def test_a_protected_endpoint_requires_a_key(server):
    status, body = get(server, "/v1/quote?ticker=AAPL")
    assert status == 401
    assert "error" in body


def test_a_valid_key_is_accepted(server, store_path):
    _, plaintext = issue(store_path)
    status, body = get(server, "/v1/quote?ticker=AAPL", plaintext)
    assert status == 200 and body["ticker"] == "AAPL"


def test_the_x_api_key_header_also_works(server, store_path):
    _, plaintext = issue(store_path)
    status, _ = get(server, "/v1/quote?ticker=AAPL", plaintext, header="X-API-Key")
    assert status == 200


def test_a_garbage_key_is_rejected(server, store_path):
    issue(store_path)
    status, _ = get(server, "/v1/quote?ticker=AAPL", "not-even-a-key")
    assert status == 401


def test_a_tampered_key_is_rejected(server, store_path):
    _, plaintext = issue(store_path)
    prefix, key_id, secret = plaintext.split("_", 2)
    status, _ = get(server, "/v1/quote?ticker=AAPL", f"{prefix}_{key_id}_{secret[:-2]}zz")
    assert status == 401


def test_a_revoked_key_is_rejected_immediately(server, store_path):
    """Revocation has to take effect on the NEXT request — the server
    re-reads the store per request rather than caching it, and this is
    what proves that."""
    key, plaintext = issue(store_path)
    assert get(server, "/v1/quote?ticker=AAPL", plaintext)[0] == 200

    api_keys.save_store(revoke_key(api_keys.load_store(store_path), key.id), store_path)
    assert get(server, "/v1/quote?ticker=AAPL", plaintext)[0] == 401


def test_an_expired_key_is_rejected(server, store_path):
    import datetime
    from dataclasses import replace

    key, plaintext = issue(store_path, days=30)
    store = api_keys.load_store(store_path)
    past = (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(timespec="seconds")
    api_keys.save_store(
        replace(store, keys=tuple(replace(k, expires_at=past) for k in store.keys)), store_path)
    assert get(server, "/v1/quote?ticker=AAPL", plaintext)[0] == 401


def test_a_401_carries_a_www_authenticate_header(server):
    request = urllib.request.Request(server + "/v1/quote?ticker=AAPL")
    try:
        urllib.request.urlopen(request, timeout=10)
        pytest.fail("expected 401")
    except urllib.error.HTTPError as e:
        assert e.code == 401
        assert "Bearer" in e.headers.get("WWW-Authenticate", "")


def test_every_auth_failure_uses_the_same_status(server, store_path):
    """Malformed, unknown, revoked and expired all return 401. A different
    status for "this id exists but the secret is wrong" would let someone
    enumerate valid key ids."""
    key, plaintext = issue(store_path)
    prefix, key_id, secret = plaintext.split("_", 2)
    api_keys.save_store(revoke_key(api_keys.load_store(store_path), key.id), store_path)

    for candidate in ("junk", "qtx_00000000_nope", f"{prefix}_{key_id}_{secret[:-2]}zz", plaintext):
        assert get(server, "/v1/quote?ticker=AAPL", candidate)[0] == 401, candidate


# --- authorization (scopes) ---------------------------------------------------

def test_a_key_without_the_scope_is_forbidden(server, store_path):
    _, plaintext = issue(store_path, scopes=("quote:read",))
    status, body = get(server, "/v1/risk?ticker=AAPL", plaintext)
    assert status == 403
    assert body["required_scope"] == "risk:read"


def test_403_is_distinct_from_401(server, store_path):
    """A valid key hitting an out-of-scope endpoint is an authorization
    failure, not an authentication one. Collapsing them would send an
    integrator hunting a credential problem that doesn't exist."""
    _, plaintext = issue(store_path, scopes=("quote:read",))
    assert get(server, "/v1/risk?ticker=AAPL", plaintext)[0] == 403
    assert get(server, "/v1/risk?ticker=AAPL", "qtx_00000000_bogus")[0] == 401


def test_each_scope_unlocks_only_its_own_endpoint(server, store_path):
    paths = {
        "quote:read": "/v1/quote?ticker=AAPL",
        "fundamentals:read": "/v1/fundamentals?ticker=AAPL",
        "risk:read": "/v1/risk?ticker=AAPL",
        "watchlist:read": "/v1/watchlists",
    }
    for granted, granted_path in paths.items():
        path_store = api_keys.load_store(store_path)
        api_keys.save_store(ApiKeyStore(), store_path)
        _, plaintext = issue(store_path, scopes=(granted,))
        for scope, path in paths.items():
            expected = 200 if scope == granted else 403
            assert get(server, path, plaintext)[0] == expected, f"{granted} -> {path}"


def test_a_multi_scope_key_reaches_all_of_them(server, store_path):
    _, plaintext = issue(store_path, scopes=("quote:read", "risk:read"))
    assert get(server, "/v1/quote?ticker=AAPL", plaintext)[0] == 200
    assert get(server, "/v1/risk?ticker=AAPL", plaintext)[0] == 200
    assert get(server, "/v1/fundamentals?ticker=AAPL", plaintext)[0] == 403


# --- owner scoping ------------------------------------------------------------

def test_owner_scoped_endpoints_resolve_the_namespace_from_the_key(server, store_path):
    """The server has no session, so the credential is the only source of
    identity. If this regressed, one user's key would read another's
    watchlists."""
    _, alice = issue(store_path, scopes=("watchlist:read",), owner="google-alice")
    status, body = get(server, "/v1/watchlists", alice)
    assert status == 200 and body["owner"] == "google-alice"


def test_a_signed_out_key_gets_the_shared_profile(server, store_path):
    _, plaintext = issue(store_path, scopes=("watchlist:read",), owner="")
    _, body = get(server, "/v1/watchlists", plaintext)
    assert body["owner"] == ""


# --- request validation -------------------------------------------------------

def test_a_missing_ticker_is_a_400(server, store_path):
    _, plaintext = issue(store_path)
    status, body = get(server, "/v1/quote", plaintext)
    assert status == 400 and "ticker" in body["error"].lower()


def test_an_implausible_ticker_is_rejected(server, store_path):
    _, plaintext = issue(store_path)
    for bad in ("../../etc/passwd", "A" * 40, "drop table"):
        status, _ = get(server, f"/v1/quote?ticker={urllib.parse.quote(bad)}", plaintext)
        assert status == 400, bad


def test_a_handler_exception_does_not_leak_a_traceback(server, store_path, monkeypatch):
    """A stack trace in an error body tells a caller about the filesystem
    and the code. The detail belongs in the log, not the response."""
    def boom(key, query):
        raise RuntimeError("internal detail that must not escape")
    monkeypatch.setattr(api_server, "_handler_quote", boom)

    _, plaintext = issue(store_path)
    status, body = get(server, "/v1/quote?ticker=AAPL", plaintext)
    assert status == 500
    assert "internal detail" not in json.dumps(body)
    assert "Traceback" not in json.dumps(body)


def test_responses_are_json_and_not_cached(server):
    with urllib.request.urlopen(server + "/v1", timeout=10) as response:
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"


# --- the read-only contract ---------------------------------------------------

@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_write_methods_are_not_served(server, method):
    """The read-only guarantee, enforced at the transport. _Handler
    implements do_GET and nothing else, so anything else is a 501."""
    request = urllib.request.Request(server + "/v1", method=method, data=b"{}")
    try:
        urllib.request.urlopen(request, timeout=10)
        pytest.fail(f"{method} should not be served")
    except urllib.error.HTTPError as e:
        assert e.code in (400, 501), f"{method} returned {e.code}"


def test_no_route_is_registered_without_an_explicit_scope_decision():
    """Every route either names a required scope or is deliberately
    public. A route added with a typo'd scope name would silently be
    unreachable, so the scope must be a real one or None."""
    from api_keys import SCOPES
    for path, (scope, _) in api_server._ROUTES.items():
        assert scope is None or scope in SCOPES, f"{path} requires unknown scope {scope!r}"


def test_only_the_discovery_route_is_public():
    public = [p for p, (scope, _) in api_server._ROUTES.items() if scope is None]
    assert public == ["/v1"]
