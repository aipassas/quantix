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


# --- queued redelivery --------------------------------------------------------
#
# The queue's whole job is to be honest about at-least-once delivery: it
# keeps the delivery id stable so a receiver can deduplicate, re-signs
# each attempt so a receiver enforcing a timestamp window does not reject
# the retry, and refuses to retry what will never succeed.

import datetime


def _failing(status=500, error="HTTP 500"):
    def post(url, body, headers, timeout):
        return False, status, error
    return post


def _ok():
    def post(url, body, headers, timeout):
        return True, 200, ""
    return post


HALF = (lambda: 0.5)          # deterministic jitter for tests
T0 = datetime.datetime(2026, 9, 8, 12, 0, 0)


def _queued_store(resolver, poster=None):
    store = w.WebhookStore(endpoints=(_endpoint(),))
    return w.dispatch(store, "alert.triggered", {"ticker": "AAPL"},
                      poster=poster or _failing(), jitter=HALF)[0]


def test_a_retryable_failure_is_queued(resolver):
    store = _queued_store(resolver)
    assert len(store.queue) == 1
    assert store.queue[0].attempts == 1
    assert store.queue[0].data == {"ticker": "AAPL"}


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_receiver_rejection_is_never_retried(resolver, status):
    """The classic webhook bug: hammering someone's server with a request
    they have already refused."""
    store = w.WebhookStore(endpoints=(_endpoint(),))
    store, deliveries = w.dispatch(store, "alert.triggered", {},
                                   poster=_failing(status, f"HTTP {status}"),
                                   jitter=HALF)
    assert deliveries[0].retryable is False
    assert store.queue == ()


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
def test_a_transient_failure_is_retried(resolver, status):
    """408 and 429 are the two 4xx that invite a retry."""
    assert w.is_retryable_status(status) is True


def test_no_status_at_all_is_retryable():
    """Connection refused, DNS blip, TLS failure, read timeout."""
    assert w.is_retryable_status(None) is True


def test_a_refused_destination_is_not_retried(resolver):
    """Same shape as a connection error — no status — but a URL the user
    has to change, and retrying would re-resolve a hostile name on a
    schedule."""
    resolver["inward.example"] = ["10.0.0.9"]
    store = w.WebhookStore(endpoints=(_endpoint(url="https://inward.example/h"),))
    store, deliveries = w.dispatch(store, "alert.triggered", {}, jitter=HALF)
    assert deliveries[0].retryable is False
    assert store.queue == ()


def test_the_delivery_id_is_stable_across_retries(resolver):
    """The property the whole queue rests on: a receiver that has seen
    this id can discard the duplicate."""
    seen = []

    def record_id(url, body, headers, timeout):
        seen.append(headers[w.HEADER_DELIVERY])
        return False, 500, "HTTP 500"

    store = w.WebhookStore(endpoints=(_endpoint(),))
    store, first = w.dispatch(store, "alert.triggered", {}, poster=record_id,
                              jitter=HALF)
    for _ in range(2):
        due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
        store, _ = w.drain(store, now=due, poster=record_id, jitter=HALF)
    assert len(seen) == 3
    assert len(set(seen)) == 1
    assert seen[0] == first[0].id


def test_each_retry_is_signed_afresh_with_a_new_timestamp(resolver):
    """Reusing the original signature would be rejected by any receiver
    that checks the timestamp against a tolerance window — which the
    panel's own snippet tells them to do."""
    captured = []

    def capture(url, body, headers, timeout):
        captured.append((headers[w.HEADER_TIMESTAMP],
                         headers[w.HEADER_SIGNATURE], body))
        return False, 500, "HTTP 500"

    store = w.WebhookStore(endpoints=(_endpoint(),))
    store, _ = w.dispatch(store, "alert.triggered", {}, poster=capture, jitter=HALF)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    store, _ = w.drain(store, now=due, poster=capture, jitter=HALF)

    assert len(captured) == 2
    # Each attempt carries a signature that verifies against ITS OWN
    # timestamp — i.e. it was recomputed, not carried over.
    for timestamp, signature, body in captured:
        assert w.verify(_endpoint().secret, timestamp, body, signature) is True
    # The retry is signed for its own moment, so presenting it under a
    # different timestamp fails. (Two attempts can land inside the same
    # second in a test, so the timestamps themselves may match; what is
    # asserted here is that the signature is bound to the timestamp.)
    assert w.verify(_endpoint().secret, "2020-01-01T00:00:00", captured[1][2],
                    captured[1][1]) is False
    # And the retry is the SAME delivery, not a new one.
    assert json.loads(captured[0][2])["id"] == json.loads(captured[1][2])["id"]


def test_an_item_is_not_attempted_before_it_falls_due(resolver):
    store = _queued_store(resolver)
    assert w.due_items(store, T0) == ()
    store, results = w.drain(store, now=T0, poster=_ok(), jitter=HALF)
    assert results == ()
    assert len(store.queue) == 1, "it must still be queued, not consumed"


def test_a_due_item_is_attempted_and_a_success_dequeues_it(resolver):
    store = _queued_store(resolver)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    store, results = w.drain(store, now=due, poster=_ok(), jitter=HALF)
    assert [r.ok for r in results] == [True]
    assert store.queue == ()


def test_a_repeated_failure_reschedules_with_a_longer_delay(resolver):
    store = _queued_store(resolver)
    first_due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    store, _ = w.drain(store, now=first_due, poster=_failing(), jitter=HALF)
    assert store.queue[0].attempts == 2
    second_due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    # The second wait is measured from when the second attempt happened,
    # and must be the longer of the two backoffs. (The first enqueue used
    # the real clock, so T0 is not a valid baseline for it.)
    second_delay = (second_due - first_due).total_seconds()
    assert second_delay >= w.backoff_seconds(1, jitter=HALF)
    assert w.backoff_seconds(2, jitter=HALF) > w.backoff_seconds(1, jitter=HALF)


def test_backoff_grows_exponentially_and_is_capped(cfg):
    delays = [w.backoff_seconds(i, jitter=HALF) for i in range(10)]
    assert delays == sorted(delays), "must be non-decreasing"
    assert delays[0] < delays[3]
    assert max(delays) <= w.WEBHOOKS.backoff_max_seconds


def test_backoff_is_jittered_around_the_computed_delay():
    """Without jitter, everything that failed together becomes due
    together and the receiver coming back gets the whole backlog at once."""
    low = w.backoff_seconds(2, jitter=lambda: 0.0)
    mid = w.backoff_seconds(2, jitter=lambda: 0.5)
    high = w.backoff_seconds(2, jitter=lambda: 1.0)
    assert low < mid < high


def test_backoff_is_never_zero():
    assert w.backoff_seconds(0, jitter=lambda: 0.0) >= 1


def test_it_gives_up_after_max_attempts_and_says_so(resolver, cfg):
    cfg(disable_after_failures=999)          # isolate the per-event budget
    store = _queued_store(resolver)
    for _ in range(w.WEBHOOKS.max_attempts + 2):
        if not store.queue:
            break
        due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
        store, _ = w.drain(store, now=due, poster=_failing(), jitter=HALF)
    assert store.queue == ()
    assert "Gave up after" in store.deliveries[-1].error
    assert store.deliveries[-1].attempt == w.WEBHOOKS.max_attempts


def test_a_disabled_endpoint_holds_its_queue_rather_than_burning_attempts(resolver):
    """Auto-disable fires before the per-event budget with the shipped
    defaults; the queued work must survive so resuming releases it."""
    store = _queued_store(resolver)
    store = w.set_active(store, "e1", False)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    before = store.queue[0].attempts
    store, results = w.drain(store, now=due, poster=_failing(), jitter=HALF)
    assert results == ()
    assert len(store.queue) == 1
    assert store.queue[0].attempts == before, "a paused endpoint must not consume attempts"


def test_resuming_releases_held_items(resolver):
    store = _queued_store(resolver)
    store = w.set_active(store, "e1", False)
    store = w.set_active(store, "e1", True)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    store, results = w.drain(store, now=due, poster=_ok(), jitter=HALF)
    assert [r.ok for r in results] == [True]


def test_a_deleted_endpoint_drops_its_queued_items(resolver):
    store = _queued_store(resolver)
    store = w.remove_endpoint(store, "e1")
    # force=True rather than a hardcoded future date: _queued_store
    # enqueues against the REAL clock, so "T0 + a day" is only in the
    # future while today happens to be T0's date. That made this test
    # pass on one day and fail on the next.
    store, results = w.drain(store, force=True, poster=_ok(), jitter=HALF)
    assert results == ()
    assert store.queue == ()


def test_a_lease_hides_an_item_from_a_second_drainer(resolver):
    """Two drainers exist — cron and an open tab — and without this both
    would send the same item."""
    store = _queued_store(resolver)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    leased = w._lease(store, (store.queue[0].id,), due)
    assert w.due_items(leased, due) == ()


def test_a_stale_lease_expires_so_a_crash_cannot_strand_an_item(resolver):
    store = _queued_store(resolver)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    leased = w._lease(store, (store.queue[0].id,), due)
    after = due + datetime.timedelta(seconds=w.WEBHOOKS.lease_seconds + 1)
    assert len(w.due_items(leased, after)) == 1


def test_the_lease_is_published_before_any_request_goes_out(resolver):
    """A lease only prevents a double send if the other process can SEE
    it, and it cannot see anything still in this process's memory."""
    order = []
    saved_leases = []

    def saver(store):
        order.append("save")
        # Ordering alone is not enough — what matters is that the store
        # HANDED to save already carries the lease. Saving an un-leased
        # store first and leasing afterwards keeps this order and
        # publishes nothing, which is the bug this guards.
        saved_leases.append([bool(q.leased_until) for q in store.queue])

    def poster(url, body, headers, timeout):
        order.append("post")
        return True, 200, ""

    store = _queued_store(resolver)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    w.drain(store, now=due, poster=poster, jitter=HALF, save=saver)
    assert order and order[0] == "save", f"save must precede post, got {order}"
    assert saved_leases and all(saved_leases[0]), (
        "the store published before sending must already hold the lease, "
        f"got {saved_leases}")


def test_a_failing_save_does_not_stop_the_drain(resolver):
    def bad_save(store):
        raise OSError("disk full")

    store = _queued_store(resolver)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    store, results = w.drain(store, now=due, poster=_ok(), jitter=HALF, save=bad_save)
    assert [r.ok for r in results] == [True]


def test_force_ignores_the_clock_but_not_the_lease(resolver):
    store = _queued_store(resolver)
    assert w.due_items(store, T0) == ()
    assert len(w.due_items(store, T0, force=True)) == 1
    leased = w._lease(store, (store.queue[0].id,), T0)
    assert w.due_items(leased, T0, force=True) == ()


def test_a_forced_attempt_that_fails_backs_off_from_real_now(resolver):
    store = _queued_store(resolver)
    store, _ = w.drain(store, now=T0, force=True, poster=_failing(), jitter=HALF)
    due = datetime.datetime.fromisoformat(store.queue[0].next_attempt_at)
    assert T0 < due < T0 + datetime.timedelta(seconds=w.WEBHOOKS.backoff_max_seconds + 60)


def test_a_drain_can_be_limited_to_one_endpoint(resolver):
    store = w.WebhookStore(endpoints=(
        _endpoint(id="a", url="https://a.example/h"),
        _endpoint(id="b", url="https://b.example/h"),
    ))
    store, _ = w.dispatch(store, "alert.triggered", {}, poster=_failing(), jitter=HALF)
    assert len(store.queue) == 2
    store, results = w.drain(store, force=True, endpoint_id="a",
                             poster=_ok(), jitter=HALF)
    assert [r.endpoint_id for r in results] == ["a"]
    assert len(store.queue) == 1


def test_one_pass_is_bounded(resolver, cfg):
    cfg(max_drain_per_pass=2, disable_after_failures=999)
    store = w.WebhookStore(endpoints=(_endpoint(),))
    for i in range(5):
        store, _ = w.dispatch(store, "alert.triggered", {"n": i},
                              poster=_failing(), jitter=HALF)
    assert len(store.queue) == 5
    store, results = w.drain(store, force=True, poster=_ok(), jitter=HALF)
    assert len(results) == 2


def test_the_queue_is_bounded_and_a_drop_is_recorded(resolver, cfg):
    cfg(max_queue_length=2, disable_after_failures=999)
    store = w.WebhookStore(endpoints=(_endpoint(),))
    for i in range(4):
        store, _ = w.dispatch(store, "alert.triggered", {"n": i},
                              poster=_failing(), jitter=HALF)
    assert len(store.queue) == 2
    assert any("Dropped from the retry queue" in d.error for d in store.deliveries), \
        "a silently discarded delivery is the one thing a retry queue must not do"


def test_an_unparseable_due_date_is_treated_as_due_not_stranded(resolver):
    store = _queued_store(resolver)
    store = dataclasses.replace(store, queue=(
        dataclasses.replace(store.queue[0], next_attempt_at="not a date"),))
    assert len(w.due_items(store, T0)) == 1


def test_the_queue_survives_a_store_round_trip(tmp_path, resolver):
    """Durability is the entire point: a queue that died with the process
    would just be a retry loop."""
    store = _queued_store(resolver)
    path = tmp_path / "webhooks_store.json"
    assert w.save_store(store, path) is True

    loaded = w.load_store(path)
    assert len(loaded.queue) == 1
    assert loaded.queue[0].id == store.queue[0].id
    assert loaded.queue[0].data == {"ticker": "AAPL"}
    assert loaded.queue[0].attempts == 1
    assert loaded.queue[0].next_attempt_at == store.queue[0].next_attempt_at


def test_a_malformed_queue_row_is_dropped_not_fatal(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"queue": [
        {"endpoint_id": "e1", "id": "q1", "event": "alert.triggered",
         "data": {}, "attempts": 1},
        {"no_endpoint": True},
        "not a dict",
    ]}))
    loaded = w.load_store(path)
    assert len(loaded.queue) == 1


def test_queue_summary_says_due_not_scheduled(resolver):
    """Nothing here can promise when a drain will happen, so the wording
    must not imply a deadline."""
    store = _queued_store(resolver)
    summary = w.queue_summary(store, "e1")
    assert "waiting to retry" in summary
    assert "due" in summary
    assert "scheduled" not in summary.lower()


def test_queue_summary_is_empty_when_nothing_is_queued():
    assert w.queue_summary(w.WebhookStore()) == ""


def test_queue_summary_flags_a_paused_endpoint(resolver):
    store = _queued_store(resolver)
    store = w.set_active(store, "e1", False)
    assert "Held" in w.queue_summary(store, "e1")


def test_a_short_secret_is_not_blindly_redacted():
    """A one-character secret turned "could not resolve" into "could not
    rewhsec_***olve" — mangling a message the user has to act on."""
    endpoint = _endpoint(secret="s")
    assert w.redact("could not resolve 'x'", (endpoint,)) == "could not resolve 'x'"


# --- redelivery wiring --------------------------------------------------------

def test_the_cron_runner_drains_the_queue():
    """The only drainer that works with the app shut, which is what makes
    redelivery durable rather than incidental."""
    source = ALERT_WATCH.read_text()
    assert "webhooks.drain(" in source


def test_the_cron_runner_drains_even_when_no_alerts_fired():
    """A quiet run is exactly when a backlog should go out."""
    source = ALERT_WATCH.read_text()
    # The phrase appears twice — once in the --check dry run, which must
    # NOT deliver anything, and once on the real quiet run, which must.
    index = source.rindex('messages.append("No newly triggered alerts.")')
    assert "_dispatch_webhooks" in source[index:index + 500]


def test_the_app_drains_opportunistically_and_publishes_the_lease():
    source = FINANCE.read_text()
    assert "webhooks.drain(_wh_live, save=webhooks.save_store)" in source


def test_the_panel_surfaces_the_queue():
    source = FINANCE.read_text()
    assert "queue_summary(" in source
    assert "webhook_drain_" in source
