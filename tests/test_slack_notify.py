"""Tests for slack_notify.py and alert_watch.py.

Two properties carry the weight:

  - THE WEBHOOK MUST NEVER APPEAR IN AN ERROR. It is a credential —
    anyone holding it can post to the channel as this app — and urllib
    embeds the failing URL in most of its exception messages, so a single
    404 from Slack would otherwise write it into quantix.log and onto the
    screen.
  - A BREACHING ALERT MUST BE POSTED ONCE, NOT EVERY RUN. "AAPL below
    200" stays true; a checker that posted current state on each tick
    would get the channel muted within a day.

No network: the poster is injected.
"""
import json

import pytest

import alert_watch
import slack_notify
from config import SLACK
from slack_notify import format_alerts, looks_like_a_webhook, post, post_alerts, redact

# Composed at runtime rather than written as one literal. It is entirely
# synthetic, but a webhook-shaped string in a committed file trips
# GitHub's push protection — correctly, since the scanner can't tell a
# fixture from a live credential. Assembling it keeps the test realistic
# without putting the pattern in the repository.
WEBHOOK = "https://hooks.slack.com/" + "services/T00000000/B00000000/" + "x" * 24


class _FakePoster:
    def __init__(self, ok=True, error=None, raises=None):
        self.calls = []
        self.ok = ok
        self.error = error
        self.raises = raises

    def __call__(self, url, payload):
        if self.raises:
            raise self.raises
        self.calls.append((url, payload))
        return self.ok, self.error


@pytest.fixture(autouse=True)
def configured_webhook(monkeypatch):
    """Give the module a webhook for the duration of every test.

    alert_watch.run() calls post_alerts() without an explicit URL — in
    production it comes from secrets — so without this the poster is
    never reached and every delivery test silently passes for the wrong
    reason. Tests that exercise the UNconfigured path override this
    themselves.
    """
    monkeypatch.setattr(slack_notify, "webhook_url", lambda: WEBHOOK)


class _Rule:
    def __init__(self, rule_id, ticker, trigger_type="price_below"):
        self.id = rule_id
        self.ticker = ticker
        self.trigger_type = trigger_type


class _Result:
    def __init__(self, is_met, detail="crossed"):
        self.is_met = is_met
        self.detail = detail
        self.status = "ok"


# --- the webhook is a credential ----------------------------------------------

def test_a_webhook_url_is_redacted():
    """THE PROPERTY THIS MODULE EXISTS TO PROTECT. urllib puts the
    failing URL into its exception text, so an unredacted error writes a
    live credential into the log."""
    leaked = f"HTTP Error 404: Not Found for {WEBHOOK}"
    cleaned = redact(leaked)
    assert WEBHOOK not in cleaned
    assert "abcdefghijklmnopqrstuvwx" not in cleaned
    assert "[redacted]" in cleaned


def test_redaction_leaves_the_rest_of_the_message_intact():
    """A redacted error still has to be diagnosable."""
    cleaned = redact(f"HTTP Error 404: Not Found for {WEBHOOK}")
    assert "404" in cleaned and "Not Found" in cleaned


def test_a_failed_post_never_returns_the_webhook():
    poster = _FakePoster(ok=False, error=f"Slack returned HTTP 404 for {WEBHOOK}")
    ok, error = post("hello", poster=poster, url=WEBHOOK)
    assert ok is False
    assert WEBHOOK not in error


def test_an_exception_from_the_poster_never_returns_the_webhook():
    poster = _FakePoster(raises=RuntimeError(f"connection failed to {WEBHOOK}"))
    ok, error = post("hello", poster=poster, url=WEBHOOK)
    assert ok is False and WEBHOOK not in error


def test_an_unconfigured_instance_says_so_without_leaking(monkeypatch):
    """Overrides the autouse webhook fixture on purpose — this is the
    fresh-checkout path, where nothing is configured at all."""
    monkeypatch.setattr(slack_notify, "webhook_url", lambda: None)
    poster = _FakePoster()
    ok, error = post("hello", poster=poster)
    assert ok is False
    assert poster.calls == [], "it tried to post with no webhook configured"
    assert "isn't configured" in error or "not configured" in error.lower()


# --- configuration ------------------------------------------------------------

def test_a_real_webhook_shape_is_accepted():
    assert looks_like_a_webhook(WEBHOOK) is True


@pytest.mark.parametrize("bad", [
    "", "   ", "not-a-url", "https://example.com/hook",
    "http://hooks.slack.com/services/T/B/x",          # http, not https
])
def test_things_that_are_not_webhooks_are_rejected(bad):
    assert looks_like_a_webhook(bad) is False


def test_a_wrong_looking_url_is_reported_before_any_request(monkeypatch):
    """Better to say "that isn't a webhook" than to POST somewhere
    arbitrary and report whatever comes back."""
    monkeypatch.setattr(slack_notify, "webhook_url", lambda: "https://example.com/hook")
    reason = slack_notify.unavailable_reason()
    assert reason and "hooks.slack.com" in reason


def test_no_webhook_means_unavailable(monkeypatch):
    monkeypatch.setattr(slack_notify, "webhook_url", lambda: None)
    assert slack_notify.is_configured() is False
    assert slack_notify.unavailable_reason()


# --- message formatting -------------------------------------------------------

def test_alerts_are_batched_into_one_message():
    """Five rules tripping at once must not produce five notifications —
    that is how a channel gets muted."""
    text = format_alerts([("AAPL", "price_below", "195.30"),
                          ("MSFT", "rsi_below", "28.1")])
    assert text.count("\n") == 2          # heading plus two rows
    assert "AAPL" in text and "MSFT" in text
    assert "2 alerts triggered" in text


def test_a_single_alert_reads_naturally():
    assert "1 alert triggered" in format_alerts([("AAPL", "price_below", "195.30")])


def test_trigger_types_are_humanised():
    assert "price below" in format_alerts([("AAPL", "price_below", "x")])


def test_long_batches_are_capped_and_say_so():
    alerts = [(f"T{i}", "price_below", "x") for i in range(SLACK.max_alerts_per_message + 5)]
    text = format_alerts(alerts)
    assert "and 5 more" in text
    assert text.count("•") == SLACK.max_alerts_per_message


def test_no_alerts_produces_no_message():
    assert format_alerts([]) == ""


def test_posting_nothing_is_success_not_failure():
    """A scheduled run with no new alerts is the normal case, not an
    error condition."""
    poster = _FakePoster()
    ok, error = post_alerts([], poster=poster)
    assert ok is True and error is None
    assert poster.calls == []


def test_a_successful_post_sends_the_expected_payload():
    poster = _FakePoster()
    ok, _ = post_alerts([("AAPL", "price_below", "195.30")], poster=poster, url=WEBHOOK)
    assert ok
    url, payload = poster.calls[0]
    assert url == WEBHOOK
    assert "AAPL" in payload["text"]
    assert payload["username"] == SLACK.username


def test_empty_text_is_refused():
    poster = _FakePoster()
    ok, error = post("   ", poster=poster, url=WEBHOOK)
    assert ok is False and poster.calls == []


# --- duplicate suppression ----------------------------------------------------

def watcher(rules, results, state_path):
    return dict(rules_loader=lambda: rules, evaluator=lambda r: results, path=state_path)


def test_a_newly_breaching_rule_is_posted(tmp_path):
    state = tmp_path / "state.json"
    poster = _FakePoster()
    posted, _ = alert_watch.run(
        post=True, poster=poster,
        **watcher([_Rule("r1", "AAPL")], {"r1": _Result(True)}, state))
    assert posted == 1
    assert "AAPL" in poster.calls[0][1]["text"]


def test_a_still_breaching_rule_is_not_posted_again(tmp_path):
    """THE CENTRAL PROPERTY. "AAPL below 200" is just as true on the next
    tick. Re-posting it every half hour would get the channel muted."""
    state = tmp_path / "state.json"
    rules, results = [_Rule("r1", "AAPL")], {"r1": _Result(True)}

    first = _FakePoster()
    alert_watch.run(post=True, poster=first, **watcher(rules, results, state))
    assert len(first.calls) == 1

    second = _FakePoster()
    posted, _ = alert_watch.run(post=True, poster=second, **watcher(rules, results, state))
    assert posted == 0
    assert second.calls == []


def test_a_resolved_rule_can_trigger_again_later(tmp_path):
    """If the active set never cleared, a rule that recovered would be
    permanently unable to alert again."""
    state = tmp_path / "state.json"
    rules = [_Rule("r1", "AAPL")]

    alert_watch.run(post=True, poster=_FakePoster(),
                    **watcher(rules, {"r1": _Result(True)}, state))
    # resolves
    alert_watch.run(post=True, poster=_FakePoster(),
                    **watcher(rules, {"r1": _Result(False)}, state))
    # trips again
    again = _FakePoster()
    posted, _ = alert_watch.run(post=True, poster=again,
                                **watcher(rules, {"r1": _Result(True)}, state))
    assert posted == 1 and len(again.calls) == 1


def test_a_failed_post_does_not_mark_the_alert_delivered(tmp_path):
    """Otherwise one unreachable-Slack moment silently swallows the
    alert — the next run would consider it already sent."""
    state = tmp_path / "state.json"
    rules, results = [_Rule("r1", "AAPL")], {"r1": _Result(True)}

    failing = _FakePoster(ok=False, error="Slack unreachable")
    posted, messages = alert_watch.run(post=True, poster=failing, **watcher(rules, results, state))
    assert posted == 0
    assert any("FAILED" in m for m in messages)

    retry = _FakePoster()
    posted, _ = alert_watch.run(post=True, poster=retry, **watcher(rules, results, state))
    assert posted == 1, "the alert was not retried after a failed post"


def test_a_dry_run_posts_nothing_and_writes_nothing(tmp_path):
    """A --check that advanced state would mark alerts seen without
    anyone being told, so the real run would skip them."""
    state = tmp_path / "state.json"
    poster = _FakePoster()
    posted, messages = alert_watch.run(
        post=False, poster=poster,
        **watcher([_Rule("r1", "AAPL")], {"r1": _Result(True)}, state))

    assert posted == 0 and poster.calls == []
    assert not state.exists(), "the dry run wrote state"
    assert any("not posted" in m for m in messages)

    real = _FakePoster()
    assert alert_watch.run(post=True, poster=real,
                           **watcher([_Rule("r1", "AAPL")], {"r1": _Result(True)}, state))[0] == 1


def test_no_rules_configured_is_explained(tmp_path):
    posted, messages = alert_watch.run(
        post=True, poster=_FakePoster(),
        **watcher([], {}, tmp_path / "state.json"))
    assert posted == 0
    assert any("no alert rules" in m.lower() for m in messages)


def test_a_failing_evaluator_degrades_to_a_note(tmp_path):
    def boom(rules):
        raise RuntimeError("market data down")
    posted, messages = alert_watch.run(
        post=True, poster=_FakePoster(), rules_loader=lambda: [_Rule("r1", "AAPL")],
        evaluator=boom, path=tmp_path / "state.json")
    assert posted == 0
    assert any("couldn't evaluate" in m.lower() for m in messages)


def test_corrupt_state_fails_toward_notifying(tmp_path):
    """A duplicate notification is annoying; a silently dropped one is a
    missed alert the user asked for."""
    state = tmp_path / "state.json"
    state.write_text("{not json")
    posted, _ = alert_watch.run(
        post=True, poster=_FakePoster(),
        **watcher([_Rule("r1", "AAPL")], {"r1": _Result(True)}, state))
    assert posted == 1


def test_state_records_every_rule_not_only_the_triggered_ones(tmp_path):
    state = tmp_path / "state.json"
    alert_watch.run(post=True, poster=_FakePoster(), **watcher(
        [_Rule("r1", "AAPL"), _Rule("r2", "MSFT")],
        {"r1": _Result(True), "r2": _Result(False)}, state))
    saved = json.loads(state.read_text())["active"]
    assert saved == {"r1": True, "r2": False}
