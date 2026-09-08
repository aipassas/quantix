"""User-registered outbound webhooks: push Quantix events into anything.

WHY THIS IS NOT slack_notify WITH A TEXT BOX. slack_notify posts to ONE
destination, pinned to hooks.slack.com, whose URL is a credential read
from secrets and never typed into the UI. This feature is the opposite on
every axis: many destinations, arbitrary hosts, URLs typed in by the user
and stored on disk. That inverts the threat model, and almost all of the
code below exists because of that inversion rather than because posting
JSON is hard.

TWO SSRF HOLES, BOTH MEASURED ON THIS MACHINE RATHER THAN ASSUMED.

  1. A HOSTNAME BLOCKLIST DOES NOT WORK. `localtest.me` and
     `127.0.0.1.nip.io` are ordinary public DNS names that resolve to
     127.0.0.1 — measured 2026-09-07. Blocking the strings "localhost"
     and "127.0.0.1" catches neither. So the check resolves the host and
     inspects every address the resolver returns, and it runs at DELIVERY
     time, not only at registration: DNS can change between the two, and
     a name that validated as public on Monday can point at 169.254.169.254
     on Tuesday.

  2. urllib FOLLOWS REDIRECTS BY DEFAULT, WHICH DEFEATS ANY PRE-FLIGHT
     CHECK. Measured: a POST to a validated address that answers 302
     Location: http://127.0.0.1:.../secret is followed by the default
     opener, which returns the internal body with status 200. So a public
     URL is enough to reach loopback unless redirects are refused
     outright. `_no_redirect_opener()` refuses them and a 3xx is reported
     to the user as a failed delivery, because a webhook receiver that
     redirects is either misconfigured or hostile and neither deserves a
     silent retry at the new address.

WHOSE PROBLEM SSRF ACTUALLY IS, STATED HONESTLY. On a laptop running
Quantix for one person, "the user can make the app POST to their own
localhost" grants that user nothing they could not do with curl, and
blocking it outright would break the most likely real use — an n8n or
Home Assistant instance on the same machine. The protection matters for
the HOSTED case, which branding.py shows this project takes seriously: a
licensee running Quantix for other people must not let one of them use
it to probe the host's internal network or read cloud metadata. So
private destinations are refused by DEFAULT and allowed only per
endpoint, by an explicit acknowledgement the user has to tick, with the
reason on screen. `WEBHOOKS.allow_private_endpoints` lets an operator
switch that opt-in off entirely for a hosted deployment.

THE SIGNING SECRET IS STORED IN PLAINTEXT, UNLIKE AN API KEY, AND THAT
ASYMMETRY IS FORCED. api_keys.py persists only sha256(key) because it
only ever VERIFIES: a hash is enough to check what someone presents.
This module has to SIGN, and you cannot sign with a hash of the secret.
So the secret is on disk in webhooks_store.json, and pretending otherwise
would be worse than saying it. It is shown once at creation, never
re-rendered in the UI afterwards, and redacted from logs — but the file
is a credential file and the panel says so.

SIGNATURES COVER A TIMESTAMP, NOT JUST THE BODY. `sha256=HMAC(secret,
"<timestamp>.<raw body>")`. Signing the body alone lets anyone who
captures one delivery replay it forever; including the timestamp inside
the signed string lets the receiver reject anything older than its own
tolerance. The header set follows the shape Stripe and GitHub use, so an
integrator's existing verification code mostly works.

DELIVERY NEVER RAISES INTO THE CALLER. An alert must still reach Slack,
the page, and the log when a user's webhook endpoint is down. Every
failure becomes a recorded `Delivery` with a reason, and an endpoint that
fails `WEBHOOKS.disable_after_failures` times in a row is disabled with
that reason stored — a dead endpoint retried forever is a slow leak of
time on every alert evaluation, and the user needs to be told it stopped
rather than quietly discovering it months later.
"""
import datetime
import hashlib
import hmac
import logging
import ipaddress
import json
import random
import secrets
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import WEBHOOKS
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("webhooks")


# The events a user can subscribe to. Deliberately a closed set: an
# endpoint that could subscribe to "everything" would receive events
# added in later versions that its author never designed for.
EVENTS: Dict[str, str] = {
    "alert.triggered": (
        "A real-time alert rule went from not-met to met. Edge-triggered — "
        "a rule that stays breaching does not re-fire."
    ),
    "screener.match": (
        "A saved screen was run and returned at least one match."
    ),
}

DEFAULT_EVENTS: Tuple[str, ...] = ("alert.triggered",)

# Signature scheme identifier, sent in the signature header so the format
# can change later without receivers guessing which one they got.
SIGNATURE_PREFIX = "sha256="

# LOWERCASE ON PURPOSE, AND IT IS NOT A STYLE CHOICE. HTTP header names
# are case-insensitive on the wire, so these behave identically to their
# capitalised spelling for every receiver — but the brand name spelled
# with a capital and embedded in a token is exactly what
# branding.rebrand() rewrites. A white-label licensee who renamed the app
# would have shipped receivers verifying a header named after THEIR
# brand, while their integrators' code still looked for the original,
# breaking every webhook they had. Lowercase is the documented boundary
# rebrand() never touches (see test_branding), which is the stability a
# wire protocol needs.
HEADER_EVENT = "x-quantix-event"
HEADER_DELIVERY = "x-quantix-delivery"
HEADER_TIMESTAMP = "x-quantix-timestamp"
HEADER_SIGNATURE = "x-quantix-signature"


@dataclass(frozen=True)
class Endpoint:
    id: str
    url: str
    events: Tuple[str, ...]
    secret: str                       # plaintext — see the module docstring
    created_at: str
    description: str = ""
    active: bool = True
    # Ticked by the user to accept a private/loopback destination. Stored
    # per endpoint rather than globally so acknowledging a local n8n
    # instance does not silently permit a second endpoint aimed at cloud
    # metadata.
    allow_private: bool = False
    consecutive_failures: int = 0
    disabled_reason: str = ""

    def subscribes_to(self, event: str) -> bool:
        return self.active and event in self.events

    @property
    def host(self) -> str:
        try:
            return urllib.parse.urlparse(self.url).hostname or ""
        except Exception:
            return ""


@dataclass(frozen=True)
class Delivery:
    """One attempt. Recorded whether it succeeded or not — a webhook log
    that only shows successes cannot answer "why didn't my automation
    fire", which is the only question anyone asks it."""
    id: str
    endpoint_id: str
    event: str
    attempted_at: str
    ok: bool
    status: Optional[int] = None
    error: str = ""
    # Whether trying again could plausibly succeed. Set explicitly at
    # every return site rather than inferred from the error text: a
    # refused DESTINATION and a refused REQUEST both arrive with no
    # status, and only one of them is worth another attempt.
    retryable: bool = False
    attempt: int = 1

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.status} OK"
        return self.error or f"HTTP {self.status}"


@dataclass(frozen=True)
class QueuedDelivery:
    """One delivery waiting to be retried.

    `id` IS THE DELIVERY ID AND IT DOES NOT CHANGE ACROSS RETRIES. That
    is the property the whole queue rests on: a receiver that has
    already processed this id can discard the duplicate, so an at-least-
    once queue becomes effectively-once at the far end. Generating a
    fresh id per attempt would make every retry look like a new event
    and fire the automation twice.

    `data` is kept rather than the rendered body, because the body has
    to be REBUILT and RE-SIGNED at each attempt — see drain().
    """
    id: str
    endpoint_id: str
    event: str
    data: dict
    attempts: int
    next_attempt_at: str
    created_at: str
    last_error: str = ""
    # Held by whichever process is currently attempting this item, so a
    # second drainer skips it. Expires, so a crash cannot strand it.
    leased_until: str = ""


@dataclass(frozen=True)
class WebhookStore:
    endpoints: Tuple[Endpoint, ...] = ()
    deliveries: Tuple[Delivery, ...] = ()
    queue: Tuple[QueuedDelivery, ...] = ()
    # True when the file existed but could not be parsed. Distinguished
    # from "no file yet" so a corrupt store is never silently replaced by
    # the next write — the same rule screener_templates follows, and the
    # stakes are higher here because the file holds signing secrets.
    corrupt: bool = False


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _store_path() -> Path:
    # Shared, not per-user, for the same reason as api_keys and the Slack
    # alert state: alert_watch.py runs under cron with no Streamlit
    # session, so auth.current_user() is None inside it and a
    # namespaced store would be invisible to the process that has to
    # deliver from it.
    return shared_path(WEBHOOKS.store_filename)


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None) -> WebhookStore:
    """Never raises. A missing file is an empty store; an unreadable one
    is an EMPTY store flagged corrupt, which save_store() then refuses to
    overwrite."""
    path = path or _store_path()
    if not path.exists():
        return WebhookStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "webhooks.store_corrupt", section="webhooks")
        return WebhookStore(corrupt=True)
    if not isinstance(raw, dict):
        return WebhookStore(corrupt=True)

    endpoints: List[Endpoint] = []
    for item in raw.get("endpoints", []) or []:
        if not isinstance(item, dict) or not str(item.get("url", "")).strip():
            continue
        events = tuple(str(e) for e in (item.get("events") or ()) if str(e) in EVENTS)
        endpoints.append(Endpoint(
            id=str(item.get("id") or uuid.uuid4().hex[:12]),
            url=str(item["url"]).strip(),
            events=events or DEFAULT_EVENTS,
            secret=str(item.get("secret") or ""),
            created_at=str(item.get("created_at") or ""),
            description=str(item.get("description") or ""),
            active=bool(item.get("active", True)),
            allow_private=bool(item.get("allow_private", False)),
            consecutive_failures=int(item.get("consecutive_failures") or 0),
            disabled_reason=str(item.get("disabled_reason") or ""),
        ))

    deliveries: List[Delivery] = []
    for item in raw.get("deliveries", []) or []:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        deliveries.append(Delivery(
            id=str(item.get("id") or uuid.uuid4().hex[:12]),
            endpoint_id=str(item.get("endpoint_id") or ""),
            event=str(item.get("event") or ""),
            attempted_at=str(item.get("attempted_at") or ""),
            ok=bool(item.get("ok", False)),
            status=int(status) if isinstance(status, (int, float)) else None,
            error=str(item.get("error") or ""),
            retryable=bool(item.get("retryable", False)),
            attempt=int(item.get("attempt") or 1),
        ))
    queue: List[QueuedDelivery] = []
    for item in raw.get("queue", []) or []:
        if not isinstance(item, dict) or not str(item.get("endpoint_id", "")).strip():
            continue
        data = item.get("data")
        queue.append(QueuedDelivery(
            id=str(item.get("id") or uuid.uuid4().hex),
            endpoint_id=str(item["endpoint_id"]),
            event=str(item.get("event") or ""),
            data=data if isinstance(data, dict) else {},
            attempts=int(item.get("attempts") or 0),
            next_attempt_at=str(item.get("next_attempt_at") or ""),
            created_at=str(item.get("created_at") or ""),
            last_error=str(item.get("last_error") or ""),
            leased_until=str(item.get("leased_until") or ""),
        ))
    return WebhookStore(tuple(endpoints), tuple(deliveries), tuple(queue))


def save_store(store: WebhookStore, path: Optional[Path] = None) -> bool:
    """Persist. Returns False without writing when the loaded store was
    corrupt — overwriting an unreadable file would destroy endpoints and
    their signing secrets, which cannot be recovered from anywhere."""
    if store.corrupt:
        log_event(logger, logging.ERROR, "webhooks.refused_write_over_corrupt",
                  section="webhooks")
        return False
    path = path or _store_path()
    payload = {
        "endpoints": [{
            "id": e.id, "url": e.url, "events": list(e.events), "secret": e.secret,
            "created_at": e.created_at, "description": e.description,
            "active": e.active, "allow_private": e.allow_private,
            "consecutive_failures": e.consecutive_failures,
            "disabled_reason": e.disabled_reason,
        } for e in store.endpoints],
        "deliveries": [{
            "id": d.id, "endpoint_id": d.endpoint_id, "event": d.event,
            "attempted_at": d.attempted_at, "ok": d.ok, "status": d.status,
            "error": d.error, "retryable": d.retryable, "attempt": d.attempt,
        } for d in store.deliveries],
        # The queue is persisted with the payload DATA, not a rendered
        # body: each retry rebuilds and re-signs, so storing the bytes
        # would freeze a timestamp that a receiver will later reject.
        "queue": [{
            "id": q.id, "endpoint_id": q.endpoint_id, "event": q.event,
            "data": q.data, "attempts": q.attempts,
            "next_attempt_at": q.next_attempt_at, "created_at": q.created_at,
            "last_error": q.last_error, "leased_until": q.leased_until,
        } for q in store.queue],
    }
    atomic_write_text(path, json.dumps(payload, indent=2))
    return True


# --- destination safety -------------------------------------------------------

def _address_is_private(address: str) -> bool:
    """True for anything that is not a public internet destination.

    `is_private` already covers 10/8, 172.16/12, 192.168/16, 169.254/16,
    ::1 and 0.0.0.0 — verified against each of those. The rest are named
    anyway because relying on one property to imply the others is how a
    range gets missed when the stdlib's definitions shift.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True     # unparseable is not provably public
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _address_is_never_allowed(address: str) -> bool:
    """Addresses no acknowledgement can unlock.

    The local-endpoint opt-in exists for a real case — an automation
    runner on localhost or the LAN. LINK-LOCAL IS NOT THAT CASE. Nobody
    hosts their n8n instance at 169.254.169.254; that address is the
    cloud metadata service, and on a hosted deployment reaching it
    returns instance credentials. So link-local is refused even with the
    box ticked, and so are multicast and 0.0.0.0, which are not
    destinations a webhook can meaningfully have.

    Loopback and RFC1918 stay opt-in-able, because those are where the
    legitimate local receivers actually live.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return bool(ip.is_link_local or ip.is_multicast or ip.is_unspecified)


def resolve_addresses(host: str) -> Tuple[Tuple[str, ...], str]:
    """Every address `host` resolves to, or an error.

    ALL of them are returned and all are checked by the caller: a name
    with one public and one loopback address must be refused, because
    which one the connection actually uses is not ours to choose.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception as exc:
        return (), f"could not resolve {host!r} ({type(exc).__name__})"
    addresses = tuple(sorted({info[4][0] for info in infos}))
    if not addresses:
        return (), f"{host!r} resolved to no addresses"
    return addresses, ""


def validate_url(url: str, allow_private: bool = False) -> Tuple[bool, str]:
    """Whether Quantix will POST to this URL, and why not if it won't.

    Called at registration AND before every delivery. The second call is
    the one that matters: registration-time validation alone is defeated
    by a name that starts public and is later repointed at an internal
    address.
    """
    url = (url or "").strip()
    if not url:
        return False, "Enter a URL."
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "That is not a URL Quantix can parse."

    if parsed.scheme not in ("https", "http"):
        return False, "A webhook URL must start with https:// or http://."
    if parsed.scheme == "http" and not allow_private:
        return False, (
            "Plain http:// sends the payload and its signature over the "
            "network in clear text. Use https://, or tick the local-endpoint "
            "box if this is a service on your own machine."
        )
    host = parsed.hostname
    if not host:
        return False, "That URL has no host."

    addresses, error = resolve_addresses(host)
    if error:
        return False, error

    # Checked before the opt-in, because no acknowledgement unlocks these.
    never = [a for a in addresses if _address_is_never_allowed(a)]
    if never:
        return False, (
            f"{host} resolves to {never[0]}, a link-local or non-routable "
            "address. Quantix never posts to these, with or without the "
            "local-endpoint box: 169.254.169.254 is the cloud metadata "
            "service, and on a hosted instance reaching it hands out "
            "credentials."
        )

    private = [a for a in addresses if _address_is_private(a)]
    if private and not allow_private:
        return False, (
            f"{host} resolves to {private[0]}, which is a private, loopback or "
            "link-local address. Quantix refuses those by default: a public "
            "hostname that resolves inward is how a webhook gets used to reach "
            "services that were never meant to be reachable, including cloud "
            "metadata. Tick the local-endpoint box if you meant to reach a "
            "service on your own machine."
        )
    if private and not WEBHOOKS.allow_private_endpoints:
        return False, (
            "This deployment does not permit webhooks to private or loopback "
            "addresses."
        )
    return True, ""


# --- endpoints ----------------------------------------------------------------

def new_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(32)


def add_endpoint(store: WebhookStore, url: str, events: Tuple[str, ...],
                 description: str = "", allow_private: bool = False,
                 secret: str = "") -> Tuple[WebhookStore, Optional[Endpoint], str]:
    """Register a destination. Returns (store, endpoint, error)."""
    events = tuple(e for e in (events or ()) if e in EVENTS)
    if not events:
        return store, None, "Choose at least one event to send."
    if len(store.endpoints) >= WEBHOOKS.max_endpoints:
        return store, None, f"You already have {WEBHOOKS.max_endpoints} endpoints."

    ok, error = validate_url(url, allow_private=allow_private)
    if not ok:
        return store, None, error

    url = url.strip()
    if any(e.url == url for e in store.endpoints):
        return store, None, "That URL is already registered."

    endpoint = Endpoint(
        id=uuid.uuid4().hex[:12],
        url=url,
        events=events,
        secret=secret or new_secret(),
        created_at=_now_iso(),
        description=(description or "").strip(),
        allow_private=bool(allow_private),
    )
    return replace(store, endpoints=store.endpoints + (endpoint,)), endpoint, ""


def remove_endpoint(store: WebhookStore, endpoint_id: str) -> WebhookStore:
    """Drop the endpoint and its delivery history together — leaving
    orphan deliveries would render as rows attributed to nothing."""
    return replace(
        store,
        endpoints=tuple(e for e in store.endpoints if e.id != endpoint_id),
        deliveries=tuple(d for d in store.deliveries if d.endpoint_id != endpoint_id),
    )


def set_active(store: WebhookStore, endpoint_id: str, active: bool) -> WebhookStore:
    """Pause or resume. Re-enabling clears the failure counter and the
    stored reason, so an endpoint the user has fixed gets a clean run
    rather than being disabled again on its next single failure."""
    def _update(e: Endpoint) -> Endpoint:
        if e.id != endpoint_id:
            return e
        if active:
            return replace(e, active=True, consecutive_failures=0, disabled_reason="")
        return replace(e, active=False)
    return replace(store, endpoints=tuple(_update(e) for e in store.endpoints))


def endpoints_for(store: WebhookStore, event: str) -> Tuple[Endpoint, ...]:
    return tuple(e for e in store.endpoints if e.subscribes_to(event))


def deliveries_for(store: WebhookStore, endpoint_id: str) -> Tuple[Delivery, ...]:
    """Newest first — the last attempt is the one being asked about."""
    rows = [d for d in store.deliveries if d.endpoint_id == endpoint_id]
    return tuple(sorted(rows, key=lambda d: d.attempted_at, reverse=True))


# --- payload and signature ----------------------------------------------------

def build_payload(event: str, data: dict, delivery_id: str = "",
                  timestamp: str = "") -> dict:
    """The JSON body. `event`, `id` and `created_at` are envelope fields
    every delivery carries so a receiver can route and deduplicate
    without parsing `data`."""
    return {
        "event": event,
        "id": delivery_id or uuid.uuid4().hex,
        "created_at": timestamp or _now_iso(),
        "source": "quantix",
        "data": data,
    }


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """`sha256=<hex>` over "<timestamp>.<raw body>".

    The timestamp is INSIDE the signed string. Signing the body alone
    produces a signature that stays valid forever, so anyone who captures
    one delivery can replay it; with the timestamp signed, a receiver can
    reject anything outside its own tolerance and the signature cannot be
    lifted onto a newer one.
    """
    mac = hmac.new((secret or "").encode("utf-8"),
                   f"{timestamp}.".encode("utf-8") + body,
                   hashlib.sha256)
    return SIGNATURE_PREFIX + mac.hexdigest()


def verify(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    """The check a receiver performs. Exposed here so the docs in the UI
    can be generated from the real implementation and so the tests prove
    the two halves agree."""
    expected = sign(secret, timestamp, body)
    return hmac.compare_digest(expected, (signature or "").strip())


def redact(text: str, endpoints: Tuple[Endpoint, ...] = ()) -> str:
    """Remove any signing secret from a string before it is logged or
    shown. urllib puts the failing URL into most exception strings, and a
    secret that reached a log is a secret that has to be rotated."""
    out = str(text or "")
    for endpoint in endpoints:
        # A plausible secret only. new_secret() produces ~49 characters,
        # so anything very short is a test fixture or a hand-edited store
        # — and blind-replacing a one-character "secret" rewrites every
        # occurrence of that letter, which turned "could not resolve"
        # into "could not rewhsec_***olve". Mangling the message a user
        # has to act on is a worse outcome than not redacting a string
        # that cannot be a real secret.
        if endpoint.secret and len(endpoint.secret) >= 8:
            out = out.replace(endpoint.secret, "whsec_***")
    return out


# --- delivery -----------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    This is the fix for the second measured hole: the default opener
    follows a 302 to http://127.0.0.1/... and returns the internal body
    with status 200, which makes any pre-flight address check
    decorative. Returning None here turns the 3xx into an HTTPError the
    caller reports as a failure.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _no_redirect_opener():
    return urllib.request.build_opener(_NoRedirect)


def _post(url: str, body: bytes, headers: Dict[str, str],
          timeout: int) -> Tuple[bool, Optional[int], str]:
    """One POST. Returns (ok, status, error). Never raises."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _no_redirect_opener().open(request, timeout=timeout) as response:
            # Read a bounded amount: a receiver answering with a gigabyte
            # must not be able to hold this process.
            response.read(WEBHOOKS.max_response_bytes)
            status = int(getattr(response, "status", 0) or 0)
            if 200 <= status < 300:
                return True, status, ""
            return False, status, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        if 300 <= int(exc.code) < 400:
            return False, int(exc.code), (
                f"HTTP {exc.code} redirect — Quantix does not follow redirects "
                "from a webhook endpoint. Register the final URL instead."
            )
        return False, int(exc.code), f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, None, f"could not reach the endpoint ({exc.reason})"
    except Exception as exc:                       # timeouts, TLS, bad host
        return False, None, f"{type(exc).__name__}: {exc}"


def is_retryable_status(status: Optional[int]) -> bool:
    """Whether another attempt could plausibly succeed.

    THE IMPORTANT HALF IS WHAT IS *NOT* RETRIED. A 4xx means the receiver
    understood the request and rejected it — a bad path, a rejected
    signature, a schema it does not accept. Retrying that six times
    changes nothing and hammers someone else's server with a request they
    have already refused; it is the classic webhook bug. The two
    exceptions are 408 (the receiver timed out reading) and 429 (it is
    explicitly asking us to come back later), which are both invitations
    to retry.

    A 3xx is terminal too: Quantix does not follow redirects, so the
    endpoint is misconfigured and only the user can fix it.

    No status at all — a connection refused, a DNS blip, a TLS handshake
    that died, a read timeout — is the case retrying exists for.
    """
    if status is None:
        return True
    if status in (408, 429):
        return True
    if 500 <= status < 600:
        return True
    return False


def deliver(endpoint: Endpoint, event: str, data: dict,
            poster: Optional[Callable] = None,
            delivery_id: str = "", attempt: int = 1) -> Delivery:
    """Send one event to one endpoint. Never raises.

    `delivery_id` is passed in when this is a RETRY, so the id a receiver
    sees is the same one it saw before and can be deduplicated against.
    The timestamp and signature are still regenerated — see drain().

    `poster` is injected so tests and the UI's "send test event" run the
    whole signing and recording path without a network.
    """
    delivery_id = delivery_id or uuid.uuid4().hex
    timestamp = _now_iso()
    payload = build_payload(event, data, delivery_id, timestamp)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": WEBHOOKS.user_agent,
        HEADER_EVENT: event,
        HEADER_DELIVERY: delivery_id,
        HEADER_TIMESTAMP: timestamp,
        HEADER_SIGNATURE: sign(endpoint.secret, timestamp, body),
    }

    # Re-validated on every send, not just at registration. A hostname
    # that was public when it was added can be repointed at an internal
    # address afterwards, and this is the only check that would see it.
    ok, error = validate_url(endpoint.url, allow_private=endpoint.allow_private)
    if not ok:
        # NOT retryable, and this is exactly why `retryable` is set
        # explicitly rather than sniffed from the error. A refused
        # destination has no status, which looks identical to a refused
        # connection — but one is a network blip worth retrying and the
        # other is a URL the user has to change. Retrying an SSRF
        # refusal would also mean re-resolving a hostile name on a
        # schedule.
        return Delivery(delivery_id, endpoint.id, event, timestamp, False,
                        None, redact(error, (endpoint,)),
                        retryable=False, attempt=attempt)

    send = poster or _post
    try:
        ok, status, error = send(endpoint.url, body, headers,
                                 WEBHOOKS.request_timeout_seconds)
    except Exception as exc:
        # _post itself never raises, but an injected poster might, and
        # "never raises" has to hold for the caller either way: this runs
        # inside the alert path.
        ok, status, error = False, None, f"{type(exc).__name__}: {exc}"
    if not ok:
        log_event(logger, logging.WARNING, "webhooks.delivery_failed",
                  section="webhooks", endpoint=endpoint.id, kind=event,
                  attempt=attempt)
    return Delivery(delivery_id, endpoint.id, event, timestamp, bool(ok),
                    status, redact(error, (endpoint,)),
                    retryable=(not ok and is_retryable_status(status)),
                    attempt=attempt)


def record(store: WebhookStore, delivery: Delivery) -> WebhookStore:
    """Append a delivery, update the endpoint's failure run, and disable
    an endpoint that has failed too many times consecutively.

    Auto-disable is not tidiness. Every alert evaluation would otherwise
    keep paying the timeout for an endpoint that has been dead for
    months, and the user would never be told — the failure only shows if
    they go looking. Disabling records the reason so the panel can say
    what happened and when.
    """
    endpoints: List[Endpoint] = []
    for endpoint in store.endpoints:
        if endpoint.id != delivery.endpoint_id:
            endpoints.append(endpoint)
            continue
        if delivery.ok:
            endpoints.append(replace(endpoint, consecutive_failures=0,
                                     disabled_reason=""))
            continue
        failures = endpoint.consecutive_failures + 1
        if failures >= WEBHOOKS.disable_after_failures:
            endpoints.append(replace(
                endpoint, consecutive_failures=failures, active=False,
                disabled_reason=(
                    f"Disabled after {failures} consecutive failures. "
                    f"Last error: {delivery.summary}"
                ),
            ))
        else:
            endpoints.append(replace(endpoint, consecutive_failures=failures))

    deliveries = (store.deliveries + (delivery,))[-WEBHOOKS.max_delivery_log:]
    return replace(store, endpoints=tuple(endpoints), deliveries=deliveries)


# --- the retry queue ----------------------------------------------------------
#
# WHAT "QUEUED REDELIVERY" HONESTLY MEANS HERE. There is no daemon. The
# queue is durable state in the same store file, and it is DRAINED by
# whoever runs next: alert_watch.py on its cron schedule (the reliable
# path, which works with the app shut) and the running app
# opportunistically (which shortens the wait whenever a tab is open).
#
# So a backoff of 60 seconds does not promise delivery 60 seconds later.
# It makes the item ELIGIBLE at that point; it goes out on the next
# drain. Saying "retry in 60s" while the only drainer runs every 15
# minutes would be a promise the app cannot keep, so the panel words it
# as "due" rather than "scheduled".
#
# DELIVERY IS AT-LEAST-ONCE, MADE EFFECTIVELY-ONCE AT THE RECEIVER. Two
# drainers can overlap, a lease can expire mid-flight, and a receiver can
# process a request whose response never got back to us. The queue does
# not pretend otherwise: it keeps the delivery id STABLE across attempts
# so a receiver that records ids can discard the duplicate, and the
# panel's verification snippet says so.


def backoff_seconds(attempts: int, jitter: Optional[Callable] = None) -> int:
    """Delay before attempt number `attempts` + 1.

    Exponential from `backoff_base_seconds`, capped, then jittered. The
    jitter is not decoration: when a receiver goes down, every queued
    item fails at the same moment and would otherwise become due at the
    same moment, so the machine coming back up gets the whole backlog in
    one burst. Spreading them is the difference between a retry and a
    retry storm.
    """
    attempts = max(0, int(attempts))
    delay = WEBHOOKS.backoff_base_seconds * (WEBHOOKS.backoff_factor ** attempts)
    delay = min(delay, WEBHOOKS.backoff_max_seconds)
    spread = delay * WEBHOOKS.backoff_jitter
    roll = jitter() if jitter is not None else random.random()
    # Jitter is symmetric around the computed delay, and the result is
    # floored at one second so a test with a stubbed roll of 0 cannot
    # produce a zero-delay item that spins.
    return max(1, int(delay - spread + (2 * spread * roll)))


def _parse_iso(value: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None


def enqueue(store: WebhookStore, delivery: Delivery, data: dict,
            now: Optional[datetime.datetime] = None,
            jitter: Optional[Callable] = None) -> WebhookStore:
    """Put a failed delivery in line for another attempt.

    Only called for a delivery that is `retryable` and has attempts left.
    The queue is bounded; when it is full the OLDEST item is dropped and
    the drop is recorded in the delivery log, because a retry queue that
    silently discards is worse than one that never existed.
    """
    now = now or datetime.datetime.now()
    due = now + datetime.timedelta(seconds=backoff_seconds(delivery.attempt, jitter))
    item = QueuedDelivery(
        id=delivery.id,
        endpoint_id=delivery.endpoint_id,
        event=delivery.event,
        data=dict(data or {}),
        attempts=int(delivery.attempt),
        next_attempt_at=due.isoformat(timespec="seconds"),
        created_at=now.isoformat(timespec="seconds"),
        last_error=delivery.summary,
    )
    queue = store.queue + (item,)
    dropped = ()
    if len(queue) > WEBHOOKS.max_queue_length:
        overflow = len(queue) - WEBHOOKS.max_queue_length
        dropped, queue = queue[:overflow], queue[overflow:]

    store = replace(store, queue=queue)
    for lost in dropped:
        store = record(store, Delivery(
            lost.id, lost.endpoint_id, lost.event,
            now.isoformat(timespec="seconds"), False, None,
            f"Dropped from the retry queue — it holds at most "
            f"{WEBHOOKS.max_queue_length} items and older ones go first.",
            retryable=False, attempt=lost.attempts,
        ))
    return store


def queued_for(store: WebhookStore, endpoint_id: str) -> Tuple[QueuedDelivery, ...]:
    return tuple(q for q in store.queue if q.endpoint_id == endpoint_id)


def due_items(store: WebhookStore,
              now: Optional[datetime.datetime] = None,
              force: bool = False) -> Tuple[QueuedDelivery, ...]:
    """Queued items eligible to go out now: due, and not leased by
    another drainer.

    An unparseable date is treated as DUE rather than skipped — a
    corrupt timestamp must not park an item in the queue forever where
    nobody would ever look at it again.
    """
    now = now or datetime.datetime.now()
    out: List[QueuedDelivery] = []
    for item in store.queue:
        lease = _parse_iso(item.leased_until)
        if lease is not None and lease > now:
            continue                       # someone else is attempting it
        if not force:
            due = _parse_iso(item.next_attempt_at)
            if due is not None and due > now:
                continue
        out.append(item)
    return tuple(out)


def _lease(store: WebhookStore, ids: Tuple[str, ...],
           now: datetime.datetime) -> WebhookStore:
    until = (now + datetime.timedelta(seconds=WEBHOOKS.lease_seconds)
             ).isoformat(timespec="seconds")
    return replace(store, queue=tuple(
        replace(q, leased_until=until) if q.id in ids else q
        for q in store.queue))


def drain(store: WebhookStore, now: Optional[datetime.datetime] = None,
          poster: Optional[Callable] = None,
          jitter: Optional[Callable] = None,
          save: Optional[Callable] = None,
          force: bool = False,
          endpoint_id: str = "") -> Tuple[WebhookStore, Tuple[Delivery, ...]]:
    """Attempt every due item. Returns (store, deliveries).

    `save` is called once the leases are taken and before any request
    goes out. That ordering is the point: leases only prevent a double
    send if the OTHER process can see them, and it cannot see anything
    still sitting in this process's memory. Without it, cron and an open
    tab would both find the same item due and both send it.

    Never raises. A drain runs inside the alerts fragment and inside a
    cron job, and neither may be taken down by a receiver.
    """
    now = now or datetime.datetime.now()
    endpoints = {e.id: e for e in store.endpoints}

    # `force` ignores the backoff clock — the panel's "Retry queued now"
    # — but NOT the lease. Overriding both would let the button send an
    # item another drainer is mid-flight on, which is the one thing the
    # lease exists to prevent. Rescheduling still uses the real `now`,
    # so a forced attempt that fails backs off from this moment rather
    # than from whatever the caller pretended the time was.
    ready = due_items(store, now, force=force)
    if endpoint_id:
        ready = tuple(q for q in ready if q.endpoint_id == endpoint_id)
    ready = ready[:WEBHOOKS.max_drain_per_pass]
    if not ready:
        return store, ()

    store = _lease(store, tuple(item.id for item in ready), now)
    if save is not None:
        try:
            save(store)
        except Exception:
            log_exception(logger, "webhooks.lease_save_failed", section="webhooks")

    results: List[Delivery] = []
    for item in ready:
        endpoint = endpoints.get(item.endpoint_id)
        if endpoint is None:
            # The endpoint was deleted while this sat in the queue.
            # Nothing to deliver to and nothing to tell the user about,
            # since remove_endpoint() already took its history.
            store = _dequeue(store, item.id)
            continue
        if not endpoint.active:
            # Paused or auto-disabled. Leave the item queued rather than
            # dropping it: resuming the endpoint should not have silently
            # lost what was waiting. The lease is cleared so it is
            # eligible again the moment the endpoint comes back.
            store = _release(store, item.id)
            continue

        attempt = item.attempts + 1
        # RE-SIGNED WITH A FRESH TIMESTAMP, KEEPING THE ORIGINAL ID. A
        # receiver that checks the timestamp against a tolerance window —
        # which the panel's own snippet tells them to do — would reject a
        # replayed signature from an hour ago, so reusing the original
        # would guarantee every retry failed. The id is what stays
        # constant, and the id is what deduplication uses.
        delivery = deliver(endpoint, item.event, item.data, poster=poster,
                           delivery_id=item.id, attempt=attempt)
        results.append(delivery)
        store = record(store, delivery)

        if delivery.ok:
            store = _dequeue(store, item.id)
            continue
        if not delivery.retryable or attempt >= WEBHOOKS.max_attempts:
            store = _dequeue(store, item.id)
            store = record(store, replace(
                delivery,
                error=_gave_up_reason(delivery, attempt),
                attempted_at=now.isoformat(timespec="seconds"),
            ))
            continue
        store = _reschedule(store, item, delivery, now, jitter)

    return store, tuple(results)


def queue_summary(store: WebhookStore, endpoint_id: str = "",
                  now: Optional[datetime.datetime] = None) -> str:
    """One line describing what is waiting, for the panel.

    Says "due" rather than "scheduled" on purpose — see the section
    comment above. Nothing here can promise when a drain will happen.
    """
    items = queued_for(store, endpoint_id) if endpoint_id else store.queue
    if not items:
        return ""
    now = now or datetime.datetime.now()
    ready = sum(1 for q in items
                if (_parse_iso(q.next_attempt_at) or now) <= now)
    held = ""
    if endpoint_id:
        endpoint = next((e for e in store.endpoints if e.id == endpoint_id), None)
        if endpoint is not None and not endpoint.active:
            held = " Held while this endpoint is paused — resuming releases them."
    word = "delivery" if len(items) == 1 else "deliveries"
    return (f"{len(items)} {word} waiting to retry, {ready} due now.{held}")


# HOW AUTO-DISABLE AND THE RETRY BUDGET INTERACT, because the two limits
# are easy to read as one. `disable_after_failures` counts CONSECUTIVE
# failures on an ENDPOINT; `max_attempts` counts attempts at a single
# EVENT. With the shipped defaults the endpoint limit (5) is reached
# before the per-event one (6), so an endpoint that is simply down gets
# disabled first and its queued items are HELD rather than burning
# through their remaining attempts against a machine nobody is fixing.
# Resuming the endpoint releases them with their attempt counts intact.
# The per-event budget is what bounds an endpoint that fails
# intermittently, where the consecutive counter keeps resetting and the
# endpoint therefore never disables.
def _gave_up_reason(delivery: Delivery, attempt: int) -> str:
    if not delivery.retryable:
        return (f"Gave up after attempt {attempt} — {delivery.summary}. "
                "This will not succeed on a retry, so it was not queued again.")
    return (f"Gave up after {attempt} attempts — {delivery.summary}. "
            "The event was not delivered.")


def _dequeue(store: WebhookStore, item_id: str) -> WebhookStore:
    return replace(store, queue=tuple(q for q in store.queue if q.id != item_id))


def _release(store: WebhookStore, item_id: str) -> WebhookStore:
    """Drop the lease without consuming an attempt."""
    return replace(store, queue=tuple(
        replace(q, leased_until="") if q.id == item_id else q for q in store.queue))


def _reschedule(store: WebhookStore, item: QueuedDelivery, delivery: Delivery,
                now: datetime.datetime,
                jitter: Optional[Callable] = None) -> WebhookStore:
    attempts = item.attempts + 1
    due = now + datetime.timedelta(seconds=backoff_seconds(attempts, jitter))
    return replace(store, queue=tuple(
        replace(q, attempts=attempts,
                next_attempt_at=due.isoformat(timespec="seconds"),
                last_error=delivery.summary, leased_until="")
        if q.id == item.id else q
        for q in store.queue))


def dispatch(store: WebhookStore, event: str, data: dict,
             poster: Optional[Callable] = None,
             jitter: Optional[Callable] = None
             ) -> Tuple[WebhookStore, Tuple[Delivery, ...]]:
    """Send one event to every endpoint subscribed to it.

    One endpoint's failure never stops the others, and nothing here can
    raise into the alert path: an alert must still reach the page, Slack
    and the log when a user's automation server is down.
    """
    deliveries: List[Delivery] = []
    for endpoint in endpoints_for(store, event):
        delivery = deliver(endpoint, event, data, poster=poster)
        store = record(store, delivery)
        deliveries.append(delivery)
        # A first attempt that failed for a reason another attempt could
        # fix goes in the queue. One that failed because the receiver
        # rejected it, or because the destination is refused, does not —
        # see is_retryable_status().
        if (not delivery.ok and delivery.retryable
                and WEBHOOKS.max_attempts > 1):
            store = enqueue(store, delivery, data, jitter=jitter)
    return store, tuple(deliveries)


# --- event payloads -----------------------------------------------------------

def alert_payload(trigger) -> dict:
    """`data` for alert.triggered, built from a realtime_alerts
    TriggerEvent. Field names mirror the dataclass so an integrator
    reading the app and the payload sees the same words."""
    return {
        "rule_id": getattr(trigger, "rule_id", ""),
        "ticker": getattr(trigger, "ticker", ""),
        "trigger_type": getattr(trigger, "trigger_type", ""),
        "detail": getattr(trigger, "detail", ""),
        "triggered_at": getattr(trigger, "triggered_at", ""),
    }


def screener_payload(name: str, matches: Tuple[str, ...],
                     criteria_summary: str = "") -> dict:
    """`data` for screener.match.

    `match_count` is sent alongside the list because the list is capped:
    a screen over a large universe can match hundreds, and a receiver
    should be able to see that it is looking at a truncated view rather
    than infer a small result set from a short array.
    """
    capped = tuple(matches)[:WEBHOOKS.max_matches_in_payload]
    return {
        "screen": name,
        "match_count": len(matches),
        "matches": list(capped),
        "truncated": len(matches) > len(capped),
        "criteria": criteria_summary,
    }


def receiver_example(endpoint: Optional[Endpoint] = None) -> str:
    """Verification code for the receiving end, generated from the real
    signing function rather than typed into a docstring that can drift
    away from it."""
    secret_hint = "YOUR_ENDPOINT_SECRET"
    return (
        "import hmac, hashlib\n\n"
        f"SECRET = {secret_hint!r}\n"
        "TOLERANCE_SECONDS = 300\n\n"
        "def verify(headers, raw_body: bytes) -> bool:\n"
        f"    timestamp = headers[{HEADER_TIMESTAMP!r}]\n"
        f"    signature = headers[{HEADER_SIGNATURE!r}]\n"
        "    mac = hmac.new(SECRET.encode(), f'{timestamp}.'.encode() + raw_body,\n"
        "                   hashlib.sha256)\n"
        f"    expected = {SIGNATURE_PREFIX!r} + mac.hexdigest()\n"
        "    # Reject stale deliveries too — a valid signature on an old\n"
        "    # body is a replay.\n"
        "    return hmac.compare_digest(expected, signature)\n"
    )
