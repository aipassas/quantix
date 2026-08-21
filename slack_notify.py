"""Posting Quantix alerts to Slack via an incoming webhook.

THE WEBHOOK URL IS A CREDENTIAL, AND IS TREATED AS ONE. Anyone holding it
can post into that channel as this app. So it is read at send time from
Streamlit's secrets ([slack] webhook_url) or QUANTIX_SLACK_WEBHOOK_URL,
exactly like the SMTP credentials — never entered into the app's UI,
never written to disk by it.

It is also REDACTED FROM EVERY ERROR AND LOG LINE. urllib puts the failing
URL into the string form of most HTTP exceptions, so returning that
straight to the caller would print the webhook into quantix.log and into
the Streamlit UI the first time Slack returned a 404. redact() below is
applied to every message leaving this module, and a test asserts it.

STDLIB URLLIB, NOT requests OR httpx. Both happen to be installed, but a
single JSON POST does not justify a dependency, and urllib is guaranteed
present wherever this app runs.

EVERYTHING DEGRADES. With no webhook configured — the state of a fresh
checkout — is_configured() is False and the UI explains what is missing.
Nothing here can make the app or the scheduled runner fail by being
absent.
"""
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Callable, Optional, Sequence, Tuple

import streamlit as st

from config import SLACK
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("slack_notify")

_ENV_VAR = "QUANTIX_SLACK_WEBHOOK_URL"

# Slack's incoming-webhook host. Matched loosely on purpose — the path
# segments are the secret part, and pinning the exact shape would break
# the moment Slack changed it.
_WEBHOOK_RE = re.compile(r"^https://hooks\.slack\.com/services/\S+", re.I)


def redact(text: str) -> str:
    """Replace any Slack webhook URL in `text` with a placeholder.

    Applied to every string this module returns. urllib embeds the
    failing URL in most of its exception messages, so without this a
    single 404 from Slack would write the webhook — a live credential —
    into quantix.log and onto the screen.
    """
    return re.sub(r"https://hooks\.slack\.com/services/\S+",
                  "https://hooks.slack.com/services/[redacted]", str(text))


def _read_secret() -> Optional[str]:
    """st.secrets first, then the environment. st.secrets raises when no
    secrets file exists at all, which is the normal state of a fresh
    checkout — caught so "unconfigured" is a quiet fact."""
    import os

    try:
        section = st.secrets.get("slack", {})
        value = section.get("webhook_url") if hasattr(section, "get") else None
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(_ENV_VAR) or "").strip() or None


def webhook_url() -> Optional[str]:
    return _read_secret()


def is_configured() -> bool:
    return bool(webhook_url())


def looks_like_a_webhook(url: str) -> bool:
    return bool(_WEBHOOK_RE.match((url or "").strip()))


def unavailable_reason() -> Optional[str]:
    """Why Slack posting can't happen, phrased for whoever has to fix it.
    None means it's available."""
    url = webhook_url()
    if not url:
        return (
            "Slack isn't configured for this Quantix instance. Add a [slack] section with "
            "webhook_url to .streamlit/secrets.toml (see .streamlit/secrets.toml.example), "
            "or set the QUANTIX_SLACK_WEBHOOK_URL environment variable."
        )
    if not looks_like_a_webhook(url):
        return (
            "The configured Slack webhook doesn't look like an incoming-webhook URL — those "
            "start with https://hooks.slack.com/services/. Check you copied the whole thing."
        )
    return None


# --- formatting ---------------------------------------------------------------

def format_alerts(alerts: Sequence[Tuple[str, str, str]]) -> str:
    """Slack mrkdwn for a batch of triggered alerts.

    `alerts` is a sequence of (ticker, trigger_type, detail).

    Batched into ONE message rather than one per alert: a rule set that
    trips on five tickers at once should not produce five notifications,
    which is how a channel gets muted.

    Plain mrkdwn rather than Block Kit — the payload is a headline and a
    list, and Block Kit's schema is one more thing to get subtly wrong
    for no gain in what this actually renders.
    """
    if not alerts:
        return ""
    shown = list(alerts)[:SLACK.max_alerts_per_message]
    count = len(alerts)
    heading = f"*Quantix* — {count} alert{'s' if count != 1 else ''} triggered"
    lines = [heading]
    for ticker, trigger_type, detail in shown:
        readable = trigger_type.replace("_", " ")
        lines.append(f"• *{ticker}* — {readable}: {detail}")
    if count > len(shown):
        lines.append(f"_…and {count - len(shown)} more._")
    return "\n".join(lines)


# --- sending ------------------------------------------------------------------

def _default_poster(url: str, payload: dict) -> Tuple[bool, Optional[str]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=SLACK.request_timeout_seconds) as response:
            text = response.read().decode("utf-8", "replace").strip()
            # Slack answers a successful webhook post with the literal
            # body "ok"; anything else is a failure reported with HTTP 200.
            if text.lower() == "ok":
                return True, None
            return False, f"Slack rejected the message: {text[:200]}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip()[:200] if e.fp else ""
        return False, f"Slack returned HTTP {e.code}. {detail}".strip()
    except Exception as e:
        return False, f"Couldn't reach Slack ({type(e).__name__}: {e})"


def post(text: str, poster: Optional[Callable] = None,
         url: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Post `text` to the configured channel. Returns (ok, error).

    Never raises, and every returned error passes through redact() so a
    failure can't leak the webhook. `poster` is injected in tests so
    nothing here touches the network.
    """
    text = (text or "").strip()
    if not text:
        return False, "Nothing to post."

    url = url or webhook_url()
    if not url:
        return False, redact(unavailable_reason() or "Slack isn't configured.")

    payload = {"text": text, "username": SLACK.username, "icon_emoji": SLACK.icon_emoji}
    try:
        ok, error = (poster or _default_poster)(url, payload)
    except Exception as e:
        log_exception(logger, "slack.post_failed", section="slack")
        return False, redact(f"Couldn't post to Slack ({type(e).__name__}).")

    if ok:
        log_event(logger, logging.INFO, "slack.posted", chars=len(text))
        return True, None
    return False, redact(error or "Slack rejected the message.")


def post_alerts(alerts: Sequence[Tuple[str, str, str]],
                poster: Optional[Callable] = None,
                url: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Format and post a batch. Posting nothing is success, not failure —
    a scheduled run with no new alerts is the normal case."""
    if not alerts:
        return True, None
    return post(format_alerts(alerts), poster=poster, url=url)
