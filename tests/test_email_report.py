"""Tests for email_report.py — SMTP settings resolution and sending the
CIO Tear Sheet PDF as an email attachment.

No real network calls: smtplib.SMTP is monkeypatched with a fake that
records what it was asked to do, the same isolation pattern
test_data_providers.py already uses for requests.get.
"""
import pytest

import email_report
from email_report import is_email_configured, load_smtp_settings, send_report_email

_ENV_KEYS = ("HOST", "PORT", "USERNAME", "PASSWORD", "FROM_ADDRESS", "USE_TLS")


def _clear_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(f"QUANTIX_SMTP_{key}", raising=False)


class _FakeSmtpServer:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.starttls_called = False
        self.login_args = None
        self.sendmail_args = None
        _FakeSmtpServer.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.starttls_called = True

    def login(self, username, password):
        self.login_args = (username, password)

    def sendmail(self, from_addr, to_addrs, message_string):
        self.sendmail_args = (from_addr, to_addrs, message_string)


class _RaisingSmtpServer:
    def __init__(self, *a, **k):
        raise ConnectionRefusedError("no route to host")


# --- settings resolution (env-var fallback path — no secrets.toml in test env) --------

def test_unconfigured_when_no_env_vars_set(monkeypatch):
    _clear_env(monkeypatch)
    assert load_smtp_settings() is None
    assert is_email_configured() is False


def test_configured_from_env_vars(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    settings = load_smtp_settings()
    assert settings is not None
    assert settings.host == "smtp.example.com"
    assert settings.port == 587  # default when QUANTIX_SMTP_PORT isn't set
    assert settings.use_tls is True
    assert is_email_configured() is True


def test_from_address_defaults_to_username_when_unset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    settings = load_smtp_settings()
    assert settings.from_address == "reports@example.com"


def test_missing_password_is_not_configured(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    assert load_smtp_settings() is None


def test_custom_port_and_tls_parsed(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("QUANTIX_SMTP_PORT", "465")
    monkeypatch.setenv("QUANTIX_SMTP_USE_TLS", "false")
    settings = load_smtp_settings()
    assert settings.port == 465
    assert settings.use_tls is False


def test_invalid_port_falls_back_to_587(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("QUANTIX_SMTP_PORT", "not-a-number")
    assert load_smtp_settings().port == 587


# --- send_report_email -----------------------------------------------------------------

def test_send_fails_gracefully_when_unconfigured(monkeypatch):
    _clear_env(monkeypatch)
    sent, error = send_report_email("client@example.com", "Subject", "Body", b"%PDF-fake", "report.pdf")
    assert sent is False
    assert "not configured" in error or "isn't configured" in error


def test_send_rejects_invalid_recipient(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    sent, error = send_report_email("not-an-email", "Subject", "Body", b"%PDF-fake", "report.pdf")
    assert sent is False
    assert "valid" in error.lower()


def test_send_succeeds_and_attaches_pdf_via_fake_smtp(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    _FakeSmtpServer.instances = []
    monkeypatch.setattr(email_report.smtplib, "SMTP", _FakeSmtpServer)

    sent, error = send_report_email("client@example.com", "Quantix Tear Sheet", "See attached.", b"%PDF-fake-bytes", "AAPL_tear_sheet.pdf")

    assert sent is True
    assert error is None
    assert len(_FakeSmtpServer.instances) == 1
    server = _FakeSmtpServer.instances[0]
    assert server.starttls_called is True
    assert server.login_args == ("reports@example.com", "hunter2")
    from_addr, to_addrs, message_string = server.sendmail_args
    assert from_addr == "reports@example.com"
    assert to_addrs == ["client@example.com"]
    assert "AAPL_tear_sheet.pdf" in message_string
    assert "Quantix Tear Sheet" in message_string


def test_send_skips_starttls_when_use_tls_false(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("QUANTIX_SMTP_USE_TLS", "false")
    _FakeSmtpServer.instances = []
    monkeypatch.setattr(email_report.smtplib, "SMTP", _FakeSmtpServer)

    sent, error = send_report_email("client@example.com", "Subject", "Body", b"%PDF-fake", "report.pdf")

    assert sent is True
    assert _FakeSmtpServer.instances[0].starttls_called is False


def test_send_failure_returns_error_not_raise(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QUANTIX_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("QUANTIX_SMTP_USERNAME", "reports@example.com")
    monkeypatch.setenv("QUANTIX_SMTP_PASSWORD", "hunter2")
    monkeypatch.setattr(email_report.smtplib, "SMTP", _RaisingSmtpServer)

    sent, error = send_report_email("client@example.com", "Subject", "Body", b"%PDF-fake", "report.pdf")

    assert sent is False
    assert "no route to host" in error or "ConnectionRefusedError" in error


def test_the_suite_cannot_see_the_real_secrets_file():
    """REGRESSION GUARD FOR A CREDENTIAL LEAK.

    email_report._read_secret checks st.secrets BEFORE environment
    variables. With no secrets.toml on disk that ordering is invisible
    and these tests pass. The moment a real [smtp] section exists, they
    read the developer's LIVE credentials instead of their fixtures,
    fail on the mismatch, and pytest prints the real password into the
    failure output — into terminal scrollback and any CI log.

    conftest.isolate_secrets is what prevents that. This asserts the
    isolation is actually in force rather than trusting it.
    """
    import streamlit as st

    assert st.secrets.get("smtp", None) is None, (
        "the real secrets.toml is visible to the test suite — "
        "conftest.isolate_secrets is not doing its job"
    )
    assert load_smtp_settings() is None, (
        "load_smtp_settings() resolved real credentials during a test run"
    )
