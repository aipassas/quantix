"""Outbound webhooks: destination safety, signing, and delivery accounting.

The SSRF tests are the point of this file. Both holes they cover were
measured on a real machine before the code was written, not imagined:

  - `localtest.me` and `127.0.0.1.nip.io` are public DNS names that
    resolve to 127.0.0.1, so a hostname blocklist catches neither.
  - urllib's default opener follows a 302 to http://127.0.0.1/... and
    returns the internal body with status 200, which makes any
    registration-time address check decorative on its own.

Tests that need a resolver stub it, so the suite neither depends on
those names continuing to resolve nor touches the network.
"""
import dataclasses
import json
import re
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webhooks as w


@pytest.fixture
def cfg(monkeypatch):
    """WEBHOOKS is a frozen dataclass, so an attribute cannot be patched
    on it — swap the module reference for a modified copy instead."""
    def apply(**overrides):
        monkeypatch.setattr(w, "WEBHOOKS", dataclasses.replace(w.WEBHOOKS, **overrides))
    return apply


@pytest.fixture
def resolver(monkeypatch):
    """Control what every hostname resolves to."""
    table = {}

    def fake(host):
        if host in table:
            value = table[host]
            if isinstance(value, str):
                return (), value
            return tuple(value), ""
        return ("93.184.216.34",), ""

    monkeypatch.setattr(w, "resolve_addresses", fake)
    return table


# --- destination safety -------------------------------------------------------

def test_a_public_hostname_that_resolves_to_loopback_is_refused(resolver):
    """The measured hole: localtest.me is a public name pointing at
    127.0.0.1, so blocking the strings "localhost"/"127.0.0.1" misses it."""
    resolver["localtest.me"] = ["127.0.0.1"]
    ok, error = w.validate_url("https://localtest.me/hook")
    assert ok is False
    assert "127.0.0.1" in error


def test_cloud_metadata_is_refused_even_with_the_local_acknowledgement(resolver):
    """169.254.169.254 hands out instance credentials on a hosted box.
    No checkbox unlocks it."""
    resolver["metadata.example"] = ["169.254.169.254"]
    ok, error = w.validate_url("https://metadata.example/latest/meta-data",
                               allow_private=True)
    assert ok is False
    assert "metadata" in error.lower()


def test_link_local_is_refused_but_loopback_is_opt_in_able(resolver):
    """The opt-in exists for a real local receiver; link-local is never
    that, so the two must not share a switch."""
    resolver["local.example"] = ["127.0.0.1"]
    resolver["ll.example"] = ["169.254.10.10"]
    assert w.validate_url("http://local.example:5678/h", allow_private=True)[0] is True
    assert w.validate_url("https://ll.example/h", allow_private=True)[0] is False


def test_a_name_with_one_public_and_one_private_address_is_refused(resolver):
    """Which address the connection uses is not ours to choose, so any
    private answer disqualifies the name."""
    resolver["mixed.example"] = ["93.184.216.34", "10.0.0.5"]
    ok, error = w.validate_url("https://mixed.example/hook")
    assert ok is False
    assert "10.0.0.5" in error


@pytest.mark.parametrize("address", [
    "10.0.0.1", "192.168.1.10", "172.16.5.4", "169.254.169.254",
    "127.0.0.1", "0.0.0.0", "::1",
])
def test_every_non_public_range_is_recognised(address):
    assert w._address_is_private(address) is True


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "1.1.1.1"])
def test_public_addresses_are_allowed(address):
    assert w._address_is_private(address) is False


def test_an_unparseable_address_is_not_treated_as_public():
    assert w._address_is_private("not-an-ip") is True
    assert w._address_is_never_allowed("not-an-ip") is True


def test_a_non_http_scheme_is_refused():
    for url in ("ftp://example.com/x", "file:///etc/passwd", "gopher://x/1"):
        assert w.validate_url(url)[0] is False


def test_plain_http_to_a_public_host_is_refused(resolver):
    """The signature would cross the network in clear text."""
    ok, error = w.validate_url("http://example.com/hook")
    assert ok is False
    assert "clear text" in error


def test_an_unresolvable_host_is_refused_rather_than_attempted(resolver):
    resolver["nope.invalid"] = "could not resolve 'nope.invalid' (gaierror)"
    ok, error = w.validate_url("https://nope.invalid/hook")
    assert ok is False
    assert "resolve" in error


def test_an_empty_url_says_so():
    assert w.validate_url("")[0] is False
    assert w.validate_url("   ")[0] is False


def test_an_operator_can_switch_the_private_opt_in_off(resolver, cfg):
    """A hosted deployment must be able to remove the escape hatch."""
    resolver["local.example"] = ["127.0.0.1"]
    cfg(allow_private_endpoints=False)
    ok, error = w.validate_url("http://local.example/h", allow_private=True)
    assert ok is False
    assert "does not permit" in error


def test_resolve_addresses_returns_an_error_not_an_exception():
    addresses, error = w.resolve_addresses("this-host-does-not-exist.invalid")
    assert addresses == ()
    assert error


# --- redirects ----------------------------------------------------------------

def test_the_opener_refuses_redirects():
    """The second measured hole. urllib's default opener follows a 302 to
    loopback and returns the body with status 200."""
    handler = w._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {},
                                    "http://127.0.0.1/secret") is None


def test_a_redirect_is_reported_as_a_failure_naming_the_reason(monkeypatch):
    """Exercises the real _post, because the translation from "the opener
    raised a 3xx" to "we do not follow redirects" is what is under test."""
    class FakeOpener:
        def open(self, request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 302, "Found", {}, None)

    monkeypatch.setattr(w, "_no_redirect_opener", lambda: FakeOpener())
    ok, status, error = w._post("https://example.com/h", b"{}", {}, 5)
    assert ok is False
    assert status == 302
    assert "does not follow redirects" in error


def test_a_poster_that_raises_becomes_a_failed_delivery_not_an_exception(resolver):
    """deliver() runs inside the alert path and must never raise there."""
    def exploding(url, body, headers, timeout):
        raise RuntimeError("boom")

    delivery = w.deliver(_endpoint(), "alert.triggered", {}, poster=exploding)
    assert delivery.ok is False
    assert "RuntimeError" in delivery.error


def test_the_module_never_uses_the_default_opener():
    """A bare urlopen would follow redirects and silently reopen the hole."""
    source = Path(w.__file__).read_text()
    body = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    assert "urllib.request.urlopen" not in body
    assert "build_opener(_NoRedirect)" in body


# --- signing ------------------------------------------------------------------

def test_a_signature_verifies_against_its_own_body():
    body = b'{"event":"alert.triggered"}'
    signature = w.sign("whsec_abc", "2026-09-07T12:00:00", body)
    assert signature.startswith(w.SIGNATURE_PREFIX)
    assert w.verify("whsec_abc", "2026-09-07T12:00:00", body, signature) is True


def test_a_tampered_body_fails_verification():
    body = b'{"amount":1}'
    signature = w.sign("s", "t", body)
    assert w.verify("s", "t", b'{"amount":9999}', signature) is False


def test_the_wrong_secret_fails_verification():
    signature = w.sign("right", "t", b"x")
    assert w.verify("wrong", "t", b"x", signature) is False


def test_the_timestamp_is_inside_the_signed_string():
    """Signing the body alone leaves a signature valid forever, so a
    captured delivery could be replayed indefinitely."""
    body = b"same body"
    assert w.sign("s", "2026-01-01T00:00:00", body) != w.sign("s", "2026-06-01T00:00:00", body)
    signature = w.sign("s", "2026-01-01T00:00:00", body)
    assert w.verify("s", "2026-06-01T00:00:00", body, signature) is False


def test_a_missing_signature_does_not_verify():
    assert w.verify("s", "t", b"x", "") is False
    assert w.verify("s", "t", b"x", None) is False


def test_secrets_are_unique_and_prefixed():
    secrets_seen = {w.new_secret() for _ in range(50)}
    assert len(secrets_seen) == 50
    assert all(s.startswith("whsec_") for s in secrets_seen)


def test_the_documented_receiver_code_matches_the_real_scheme():
    """The snippet shown in the UI is generated from the module, so it
    cannot drift away from what sign() actually does."""
    example = w.receiver_example()
    assert w.HEADER_SIGNATURE in example
    assert w.HEADER_TIMESTAMP in example
    assert "hmac.compare_digest" in example
    assert "{timestamp}." in example


def test_redact_removes_a_secret_from_a_message():
    endpoint = _endpoint(secret="whsec_supersecret")
    assert "whsec_supersecret" not in w.redact(
        "failed posting with whsec_supersecret", (endpoint,))


# --- endpoints ----------------------------------------------------------------

def _endpoint(**overrides):
    base = dict(id="e1", url="https://example.com/hook",
                events=("alert.triggered",), secret="whsec_x",
                created_at="2026-09-07T00:00:00")
    base.update(overrides)
    return w.Endpoint(**base)


def test_adding_an_endpoint_mints_a_secret_and_returns_it(resolver):
    store, endpoint, error = w.add_endpoint(
        w.WebhookStore(), "https://example.com/hook", ("alert.triggered",))
    assert error == ""
    assert endpoint.secret.startswith("whsec_")
    assert store.endpoints == (endpoint,)


def test_an_endpoint_with_no_events_is_refused(resolver):
    _, endpoint, error = w.add_endpoint(w.WebhookStore(), "https://example.com/h", ())
    assert endpoint is None
    assert "at least one event" in error


def test_an_unknown_event_name_is_dropped_not_stored(resolver):
    _, endpoint, error = w.add_endpoint(
        w.WebhookStore(), "https://example.com/h", ("alert.triggered", "made.up"))
    assert endpoint.events == ("alert.triggered",)


def test_a_duplicate_url_is_refused(resolver):
    store, _, _ = w.add_endpoint(w.WebhookStore(), "https://example.com/h",
                                 ("alert.triggered",))
    _, endpoint, error = w.add_endpoint(store, "https://example.com/h",
                                        ("alert.triggered",))
    assert endpoint is None
    assert "already registered" in error


def test_the_endpoint_cap_is_enforced(resolver, cfg):
    cfg(max_endpoints=2)
    store = w.WebhookStore()
    for i in range(2):
        store, _, _ = w.add_endpoint(store, f"https://example.com/{i}",
                                     ("alert.triggered",))
    _, endpoint, error = w.add_endpoint(store, "https://example.com/3",
                                        ("alert.triggered",))
    assert endpoint is None and "already have" in error


def test_a_refused_url_never_becomes_an_endpoint(resolver):
    resolver["evil.example"] = ["169.254.169.254"]
    store, endpoint, error = w.add_endpoint(
        w.WebhookStore(), "https://evil.example/h", ("alert.triggered",))
    assert endpoint is None and store.endpoints == ()


def test_removing_an_endpoint_takes_its_deliveries_with_it():
    store = w.WebhookStore(
        endpoints=(_endpoint(),),
        deliveries=(w.Delivery("d1", "e1", "alert.triggered", "t", True, 200),),
    )
    after = w.remove_endpoint(store, "e1")
    assert after.endpoints == () and after.deliveries == ()


def test_pausing_stops_dispatch_and_resuming_clears_the_failure_run():
    store = w.WebhookStore(endpoints=(_endpoint(consecutive_failures=4,
                                                disabled_reason="was failing"),))
    paused = w.set_active(store, "e1", False)
    assert w.endpoints_for(paused, "alert.triggered") == ()
    resumed = w.set_active(paused, "e1", True)
    assert resumed.endpoints[0].consecutive_failures == 0
    assert resumed.endpoints[0].disabled_reason == ""


def test_endpoints_for_matches_only_subscribed_and_active():
    store = w.WebhookStore(endpoints=(
        _endpoint(id="a", events=("alert.triggered",)),
        _endpoint(id="b", events=("screener.match",)),
        _endpoint(id="c", events=("alert.triggered",), active=False),
    ))
    assert [e.id for e in w.endpoints_for(store, "alert.triggered")] == ["a"]


# --- delivery accounting ------------------------------------------------------

def _poster(ok=True, status=200, error=""):
    calls = []

    def post(url, body, headers, timeout):
        calls.append((url, body, headers, timeout))
        return ok, status, error

    post.calls = calls
    return post


def test_a_delivery_signs_the_body_it_actually_sends(resolver):
    endpoint = _endpoint()
    poster = _poster()
    delivery = w.deliver(endpoint, "alert.triggered", {"ticker": "AAPL"},
                         poster=poster)
    assert delivery.ok is True
    _, body, headers, _ = poster.calls[0]
    assert w.verify(endpoint.secret, headers[w.HEADER_TIMESTAMP], body,
                    headers[w.HEADER_SIGNATURE]) is True
    assert json.loads(body)["data"] == {"ticker": "AAPL"}
    assert headers[w.HEADER_EVENT] == "alert.triggered"


def test_the_destination_is_revalidated_at_send_time_not_only_at_registration(resolver):
    """A name that was public when registered can be repointed inward
    afterwards; this is the only check that would notice."""
    resolver["drifted.example"] = ["10.1.2.3"]
    poster = _poster()
    delivery = w.deliver(_endpoint(url="https://drifted.example/h"),
                         "alert.triggered", {}, poster=poster)
    assert delivery.ok is False
    assert poster.calls == [], "nothing may be sent to a refused address"


def test_a_failed_delivery_is_recorded_rather_than_raised(resolver):
    delivery = w.deliver(_endpoint(), "alert.triggered", {},
                         poster=_poster(ok=False, status=500, error="HTTP 500"))
    assert delivery.ok is False and delivery.status == 500


def test_consecutive_failures_disable_the_endpoint_with_a_reason(resolver, cfg):
    cfg(disable_after_failures=3)
    store = w.WebhookStore(endpoints=(_endpoint(),))
    poster = _poster(ok=False, status=500, error="HTTP 500")
    for _ in range(3):
        store, _ = w.dispatch(store, "alert.triggered", {}, poster=poster)
    endpoint = store.endpoints[0]
    assert endpoint.active is False
    assert "3 consecutive failures" in endpoint.disabled_reason
    assert "HTTP 500" in endpoint.disabled_reason


def test_a_success_resets_the_failure_run(resolver):
    store = w.WebhookStore(endpoints=(_endpoint(consecutive_failures=3),))
    store, _ = w.dispatch(store, "alert.triggered", {}, poster=_poster())
    assert store.endpoints[0].consecutive_failures == 0


def test_a_disabled_endpoint_receives_nothing_further(resolver, cfg):
    cfg(disable_after_failures=2)
    store = w.WebhookStore(endpoints=(_endpoint(),))
    poster = _poster(ok=False, status=500, error="HTTP 500")
    for _ in range(2):
        store, _ = w.dispatch(store, "alert.triggered", {}, poster=poster)
    before = len(poster.calls)
    store, results = w.dispatch(store, "alert.triggered", {}, poster=poster)
    assert results == () and len(poster.calls) == before


def test_one_endpoint_failing_does_not_stop_the_others(resolver):
    store = w.WebhookStore(endpoints=(
        _endpoint(id="a", url="https://a.example/h"),
        _endpoint(id="b", url="https://b.example/h"),
    ))

    def flaky(url, body, headers, timeout):
        if "a.example" in url:
            return False, 500, "HTTP 500"
        return True, 200, ""

    store, results = w.dispatch(store, "alert.triggered", {}, poster=flaky)
    assert [r.ok for r in results] == [False, True]


def test_the_delivery_log_is_bounded(resolver, cfg):
    cfg(max_delivery_log=5)
    cfg(disable_after_failures=999)
    store = w.WebhookStore(endpoints=(_endpoint(),))
    for _ in range(20):
        store, _ = w.dispatch(store, "alert.triggered", {}, poster=_poster())
    assert len(store.deliveries) == 5


def test_deliveries_for_returns_newest_first():
    store = w.WebhookStore(endpoints=(_endpoint(),), deliveries=(
        w.Delivery("d1", "e1", "alert.triggered", "2026-01-01T00:00:00", True, 200),
        w.Delivery("d2", "e1", "alert.triggered", "2026-06-01T00:00:00", True, 200),
    ))
    assert [d.id for d in w.deliveries_for(store, "e1")] == ["d2", "d1"]


def test_dispatch_to_nobody_is_a_no_op(resolver):
    store, results = w.dispatch(w.WebhookStore(), "alert.triggered", {})
    assert results == () and store.endpoints == ()


# --- payloads -----------------------------------------------------------------

def test_the_envelope_carries_routing_and_deduplication_fields():
    payload = w.build_payload("alert.triggered", {"a": 1})
    assert payload["event"] == "alert.triggered"
    assert payload["source"] == "quantix"
    assert payload["id"] and payload["created_at"]
    assert payload["data"] == {"a": 1}


def test_screener_payload_reports_the_true_count_when_it_truncates(cfg):
    cfg(max_matches_in_payload=3)
    payload = w.screener_payload("My screen", tuple(f"T{i}" for i in range(10)))
    assert payload["match_count"] == 10
    assert len(payload["matches"]) == 3
    assert payload["truncated"] is True


def test_screener_payload_is_not_marked_truncated_when_it_is_complete():
    payload = w.screener_payload("My screen", ("AAPL", "MSFT"))
    assert payload["truncated"] is False and payload["match_count"] == 2


def test_alert_payload_mirrors_the_trigger_event_field_names():
    class Trigger:
        rule_id, ticker, trigger_type = "r1", "AAPL", "price_below"
        detail, triggered_at = "below 200", "2026-09-07T10:00:00"

    payload = w.alert_payload(Trigger())
    assert payload == {
        "rule_id": "r1", "ticker": "AAPL", "trigger_type": "price_below",
        "detail": "below 200", "triggered_at": "2026-09-07T10:00:00",
    }


# --- persistence --------------------------------------------------------------

def test_a_round_trip_preserves_endpoints_and_deliveries(tmp_path, resolver):
    path = tmp_path / "webhooks_store.json"
    store, endpoint, _ = w.add_endpoint(w.WebhookStore(), "https://example.com/h",
                                        ("alert.triggered",), description="n8n")
    store = w.record(store, w.Delivery("d1", endpoint.id, "alert.triggered",
                                       "2026-09-07T00:00:00", True, 200))
    assert w.save_store(store, path) is True

    loaded = w.load_store(path)
    assert loaded.corrupt is False
    assert loaded.endpoints[0].url == "https://example.com/h"
    assert loaded.endpoints[0].secret == endpoint.secret
    assert loaded.endpoints[0].description == "n8n"
    assert len(loaded.deliveries) == 1


def test_a_missing_store_is_empty_not_corrupt(tmp_path):
    loaded = w.load_store(tmp_path / "nothing.json")
    assert loaded.endpoints == () and loaded.corrupt is False


def test_an_unreadable_store_is_flagged_and_never_overwritten(tmp_path):
    """The file holds signing secrets; replacing it on a parse error
    would destroy them irrecoverably."""
    path = tmp_path / "webhooks_store.json"
    path.write_text("{ this is not json")
    loaded = w.load_store(path)
    assert loaded.corrupt is True
    assert w.save_store(loaded, path) is False
    assert path.read_text() == "{ this is not json"


def test_a_malformed_endpoint_row_is_dropped_not_fatal(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"endpoints": [
        {"url": "https://good.example/h", "events": ["alert.triggered"],
         "secret": "whsec_1", "id": "a"},
        {"no_url": True},
        "not even a dict",
    ]}))
    loaded = w.load_store(path)
    assert len(loaded.endpoints) == 1


def test_the_store_is_shared_not_per_user():
    """alert_watch.py runs under cron with no Streamlit session, so a
    namespaced store would be invisible to the process that delivers."""
    source = Path(w.__file__).read_text()
    assert "shared_path(" in source
    assert "store_path(" not in source.replace("_store_path(", "").replace("shared_path(", "")


# --- wiring -------------------------------------------------------------------

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
ALERT_WATCH = Path(__file__).resolve().parent.parent / "alert_watch.py"


def test_the_panel_is_wired_into_the_app():
    source = FINANCE.read_text()
    assert "import webhooks" in source
    assert "webhooks.add_endpoint(" in source


def test_the_app_dispatches_on_a_real_alert_trigger():
    """Declaring a module is not sourcing it — this project has shipped a
    panel whose data was never handed to it."""
    source = FINANCE.read_text()
    assert 'webhooks.dispatch(_wh_live, "alert.triggered"' in source


def test_the_app_dispatches_screener_matches():
    source = FINANCE.read_text()
    assert '"screener.match"' in source
    assert "webhooks.screener_payload(" in source


def test_the_headless_runner_also_delivers():
    """Alerts fire while the tab is shut via alert_watch under cron; a
    webhook that only worked in-tab would miss most of them."""
    source = ALERT_WATCH.read_text()
    assert "_dispatch_webhooks(" in source
    assert 'webhooks.dispatch(store, "alert.triggered"' in source


def test_in_app_dispatch_cannot_take_the_page_down():
    """The alerts fragment polls constantly; an unreachable automation
    server must not stop the toast, the history or the bell."""
    source = FINANCE.read_text()
    index = source.index('webhooks.dispatch(_wh_live, "alert.triggered"')
    window = source[max(0, index - 1200):index]
    assert "try:" in window


def test_a_muted_rule_does_not_reach_a_webhook():
    """Snooze must mute everywhere or it is cosmetic. The dispatch batch
    is appended after the is_muted() guard's `continue`."""
    source = FINANCE.read_text()
    muted = source.index("if notifications.is_muted(_rt_rid):")
    appended = source.index("_rt_webhook_batch.append(")
    assert muted < appended
