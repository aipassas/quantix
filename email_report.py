"""Emailing the CIO Tear Sheet PDF directly from the app, via SMTP.

Quantix has no accounts or credential storage of its own (the same
disclosed limitation every other local store in this app already
carries), so SMTP credentials are never entered into the app's UI or
written to disk by it. They're read at send time from Streamlit's own
secrets mechanism (.streamlit/secrets.toml, an [smtp] table — see
.streamlit/secrets.toml.example for the exact keys) or, as a fallback,
QUANTIX_SMTP_*-prefixed environment variables — both standard,
already-gitignored places to keep secrets outside source control.

If neither is configured, sending degrades to an explanatory message
rather than crashing the app — the same graceful-unavailable pattern
report_export.py already uses when WeasyPrint's native Pango dependency
is missing.
"""
import logging
import os
import smtplib
from dataclasses import dataclass
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple

import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger("email_report")

_ENV_PREFIX = "QUANTIX_SMTP_"


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    username: str
    password: str
    from_address: str
    use_tls: bool = True


def _read_secret(key: str) -> Optional[str]:
    """st.secrets first, then a QUANTIX_SMTP_-prefixed environment
    variable. st.secrets raises if no secrets.toml file exists at all
    in this instance (the common case for a fresh checkout) — caught
    here so a missing file just falls through to the env-var check
    instead of blowing up the whole settings lookup."""
    try:
        smtp_secrets = st.secrets.get("smtp", {})
        if key in smtp_secrets:
            return str(smtp_secrets[key])
    except Exception:
        pass
    return os.environ.get(_ENV_PREFIX + key.upper())


def load_smtp_settings() -> Optional[SmtpSettings]:
    """None if SMTP isn't configured (missing host/username/password) —
    never raises."""
    host = _read_secret("host")
    username = _read_secret("username")
    password = _read_secret("password")
    from_address = _read_secret("from_address") or username
    if not (host and username and password and from_address):
        return None
    try:
        port = int(_read_secret("port") or "587")
    except ValueError:
        port = 587
    use_tls = (_read_secret("use_tls") or "true").strip().lower() not in ("false", "0", "no")
    return SmtpSettings(host=host, port=port, username=username, password=password, from_address=from_address, use_tls=use_tls)


def is_email_configured() -> bool:
    return load_smtp_settings() is not None


_NOT_CONFIGURED = (
    "Email isn't configured for this Quantix instance. Set [smtp] host/port/username/password/from_address "
    "in .streamlit/secrets.toml (see .streamlit/secrets.toml.example), or the equivalent QUANTIX_SMTP_* "
    "environment variables."
)


def _send(to_address: str, subject: str, body: str,
          attachment: Optional[Tuple[bytes, str]] = None) -> Tuple[bool, Optional[str]]:
    """Shared SMTP path for every outbound mail this app sends. Returns
    (True, None) on success or (False, error_message) on any failure —
    never raises, so a bad config or a network hiccup surfaces as an
    in-app warning instead of crashing the page.

    `attachment` is an optional (bytes, filename) PDF."""
    settings = load_smtp_settings()
    if settings is None:
        return False, _NOT_CONFIGURED
    if not to_address or "@" not in to_address:
        return False, "Enter a valid recipient email address."

    message = MIMEMultipart()
    message["From"] = settings.from_address
    message["To"] = to_address
    message["Subject"] = subject
    message.attach(MIMEText(body))
    if attachment is not None:
        payload, filename = attachment
        part = MIMEApplication(payload, _subtype="pdf")
        part.add_header("Content-Disposition", "attachment", filename=filename)
        message.attach(part)

    try:
        with smtplib.SMTP(settings.host, settings.port, timeout=15) as server:
            if settings.use_tls:
                server.starttls()
            server.login(settings.username, settings.password)
            server.sendmail(settings.from_address, [to_address], message.as_string())
    except Exception as e:
        log_exception(logger, "email_report.send_failed")
        return False, f"Failed to send email: {type(e).__name__}: {e}"

    log_event(logger, logging.INFO, "email_report.sent", to_domain=to_address.rsplit("@", 1)[-1])
    return True, None


def send_report_email(to_address: str, subject: str, body: str, pdf_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """Send pdf_bytes as an attachment to to_address."""
    return _send(to_address, subject, body, attachment=(pdf_bytes, filename))


def send_notification_email(to_address: str, subject: str, body: str) -> Tuple[bool, Optional[str]]:
    """Plain-text notification with no attachment — used for @-mention
    notices. Same never-raises contract as send_report_email."""
    return _send(to_address, subject, body)
