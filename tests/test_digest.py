"""Tests for digest.py — the scheduled email digest.

This module SENDS EMAIL, so the properties that matter most are the ones
governing when it must not:

  - nothing sends without both an explicit opt-in and a recipient;
  - --preview never sends;
  - a failed send must not advance last_sent_at, or one SMTP hiccup
    silently swallows a whole period;
  - a digest that can't price a ticker must SAY so, not drop the row —
    a ticker missing from the digest reads as "it didn't move".

Every dependency is injected, so no test here touches the network, a mail
server, or a real store.
"""
import datetime
import json
from dataclasses import replace

import pandas as pd
import pytest

import digest as dg
from config import DIGEST
from digest import (
    Digest,
    DigestSettings,
    TickerMove,
    build_digest,
    compute_moves,
    cron_line,
    current_risk_breaches,
    fired_alerts_since,
    load_all,
    run_scheduled,
    save_all,
    save_settings,
    send_digest,
    settings_for,
    validate,
)


class _FakeSender:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def __call__(self, to_address, subject, body):
        if self.fail:
            return False, "SMTP refused the connection"
        self.sent.append((to_address, subject, body))
        return True, None


class _FakeEvent:
    def __init__(self, ticker, triggered_at, trigger_type="price", detail="crossed 100"):
        self.ticker = ticker
        self.triggered_at = triggered_at
        self.trigger_type = trigger_type
        self.detail = detail


def _prices(*closes):
    return pd.DataFrame({"Close": list(closes)}), []


def _loader(table):
    def load(ticker, start, end):
        return table[ticker]
    return load


@pytest.fixture
def store(tmp_path):
    return tmp_path / "digest_store.json"


# --- settings -----------------------------------------------------------------

def test_a_fresh_install_has_no_recipients(store):
    assert load_all(store) == ()
    assert settings_for("", store).recipient == ""


def test_settings_round_trip(store):
    save_settings(DigestSettings(owner_key="google-a", recipient="a@example.com",
                                 enabled=True, period_days=14), store)
    restored = settings_for("google-a", store)
    assert restored.recipient == "a@example.com"
    assert restored.enabled is True and restored.period_days == 14


def test_settings_are_per_owner(store):
    save_settings(DigestSettings(owner_key="google-a", recipient="a@example.com"), store)
    save_settings(DigestSettings(owner_key="google-b", recipient="b@example.com"), store)
    assert settings_for("google-a", store).recipient == "a@example.com"
    assert settings_for("google-b", store).recipient == "b@example.com"
    assert len(load_all(store)) == 2


def test_saving_replaces_rather_than_duplicates_an_owner(store):
    save_settings(DigestSettings(owner_key="google-a", recipient="old@example.com"), store)
    save_settings(DigestSettings(owner_key="google-a", recipient="new@example.com"), store)
    assert len(load_all(store)) == 1
    assert settings_for("google-a", store).recipient == "new@example.com"


def test_corrupt_store_degrades_to_nobody_configured(store):
    """Degrading to empty sends nothing, which is the safe direction for
    a store that governs outbound mail."""
    store.write_text("{not json")
    assert load_all(store) == ()


def test_an_out_of_range_period_is_clamped_on_load(store):
    store.write_text(json.dumps({"settings": [
        {"owner_key": "", "recipient": "a@b.com", "period_days": 9999},
    ]}))
    assert load_all(store)[0].period_days == DIGEST.max_period_days


# --- the send-safety gate -----------------------------------------------------

def test_a_digest_is_not_sendable_without_an_opt_in():
    assert DigestSettings(recipient="a@example.com", enabled=False).is_sendable is False


def test_a_digest_is_not_sendable_without_a_recipient():
    """An enabled digest with no address is a scheduled job that fails
    every single week."""
    assert DigestSettings(recipient="", enabled=True).is_sendable is False
    assert DigestSettings(recipient="   ", enabled=True).is_sendable is False


def test_both_together_are_sendable():
    assert DigestSettings(recipient="a@example.com", enabled=True).is_sendable is True


def test_validation_rejects_enabling_without_an_address():
    assert validate(DigestSettings(enabled=True, recipient="")) is not None


def test_validation_rejects_a_malformed_address():
    assert validate(DigestSettings(recipient="not-an-address")) is not None


def test_validation_rejects_enabling_with_every_section_off():
    """Otherwise it mails an empty page every week."""
    error = validate(DigestSettings(
        enabled=True, recipient="a@example.com",
        include_watchlist=False, include_alerts=False, include_risk=False))
    assert error is not None


def test_a_valid_configuration_passes():
    assert validate(DigestSettings(enabled=True, recipient="a@example.com")) is None


def test_sending_without_a_recipient_refuses_rather_than_guesses():
    sender = _FakeSender()
    digest = Digest(owner_key="", period_start=datetime.date(2026, 8, 1),
                    period_end=datetime.date(2026, 8, 8))
    ok, err = send_digest(digest, DigestSettings(recipient=""), sender)
    assert ok is False and err
    assert sender.sent == []


# --- movement -----------------------------------------------------------------

def test_movement_is_computed_from_first_and_last_close():
    moves = compute_moves(("AAPL",), 7, loader=_loader({"AAPL": _prices(100.0, 105.0, 110.0)}))
    assert moves[0].change_pct == pytest.approx(10.0)
    assert moves[0].start_price == 100.0 and moves[0].end_price == 110.0


def test_a_fall_is_negative():
    moves = compute_moves(("X",), 7, loader=_loader({"X": _prices(100.0, 90.0)}))
    assert moves[0].change_pct == pytest.approx(-10.0)


def test_an_unpriceable_ticker_is_reported_not_dropped():
    """A ticker missing from the digest reads as 'it didn't move'. It has
    to appear with a reason instead."""
    moves = compute_moves(("GOOD", "BAD"), 7, loader=_loader({
        "GOOD": _prices(100.0, 110.0),
        "BAD": (pd.DataFrame({"Close": []}), ["Yahoo returned nothing"]),
    }))
    assert len(moves) == 2
    bad = next(m for m in moves if m.ticker == "BAD")
    assert bad.ok is False and bad.unavailable


def test_a_single_close_is_not_enough_to_compute_a_change():
    moves = compute_moves(("X",), 7, loader=_loader({"X": _prices(100.0)}))
    assert moves[0].ok is False and "1 closing price" in moves[0].unavailable


def test_a_loader_that_raises_does_not_stop_the_other_tickers():
    def loader(ticker, start, end):
        if ticker == "BOOM":
            raise RuntimeError("network died")
        return _prices(100.0, 110.0)
    moves = compute_moves(("BOOM", "FINE"), 7, loader=loader)
    assert next(m for m in moves if m.ticker == "BOOM").ok is False
    assert next(m for m in moves if m.ticker == "FINE").ok is True


def test_a_zero_opening_price_does_not_divide_by_zero():
    moves = compute_moves(("X",), 7, loader=_loader({"X": _prices(0.0, 10.0)}))
    assert moves[0].ok is False


def test_movers_are_ranked_by_absolute_size():
    """A 9% fall matters as much as a 9% rise; ranking by signed value
    would bury every crash at the bottom."""
    digest = Digest(
        owner_key="", period_start=datetime.date(2026, 8, 1), period_end=datetime.date(2026, 8, 8),
        moves=(TickerMove("SMALL", 1.0, 1, 1), TickerMove("CRASH", -9.0, 1, 1),
               TickerMove("RISE", 4.0, 1, 1)),
    )
    assert [m.ticker for m in digest.movers_ranked] == ["CRASH", "RISE", "SMALL"]


def test_unpriceable_tickers_are_excluded_from_the_ranking_but_kept_in_moves():
    digest = Digest(
        owner_key="", period_start=datetime.date(2026, 8, 1), period_end=datetime.date(2026, 8, 8),
        moves=(TickerMove("OK", 1.0, 1, 1), TickerMove("BAD", unavailable="no data")),
    )
    assert [m.ticker for m in digest.movers_ranked] == ["OK"]
    assert "BAD" in digest.as_text()
    assert "no data" in digest.as_text()


# --- alerts -------------------------------------------------------------------

def test_only_alerts_inside_the_period_are_included():
    since = datetime.datetime(2026, 8, 10, 0, 0)
    events = [
        _FakeEvent("OLD", "2026-08-01T12:00:00"),
        _FakeEvent("NEW", "2026-08-12T09:30:00"),
    ]
    fired = fired_alerts_since(since, history=events)
    assert len(fired) == 1 and "NEW" in fired[0]


def test_alerts_are_newest_first():
    since = datetime.datetime(2026, 8, 1)
    events = [
        _FakeEvent("FIRST", "2026-08-02T09:00:00"),
        _FakeEvent("SECOND", "2026-08-05T09:00:00"),
    ]
    fired = fired_alerts_since(since, history=events)
    assert "SECOND" in fired[0] and "FIRST" in fired[1]


def test_an_unparseable_timestamp_is_skipped_not_crashed():
    fired = fired_alerts_since(datetime.datetime(2026, 8, 1),
                               history=[_FakeEvent("BAD", "not-a-date")])
    assert fired == ()


def test_no_history_means_no_alerts():
    assert fired_alerts_since(datetime.datetime(2026, 8, 1), history=[]) == ()


def test_risk_breaches_use_the_injected_evaluator():
    breaches = current_risk_breaches(("AAPL",), evaluator=lambda t, o: ["AAPL breached"])
    assert breaches == ("AAPL breached",)


def test_no_tickers_means_no_risk_evaluation():
    """Guards against an empty-watchlist run doing a pointless scan."""
    called = []
    current_risk_breaches((), evaluator=lambda t, o: called.append(1) or [])
    assert called == []


# --- composing ----------------------------------------------------------------

def _built(**kwargs):
    settings = DigestSettings(recipient="a@example.com", **kwargs.pop("settings", {}))
    return build_digest(settings, end=datetime.date(2026, 8, 20),
                        tickers=kwargs.pop("tickers", ("AAPL",)),
                        loader=kwargs.pop("loader", _loader({"AAPL": _prices(100.0, 110.0)})),
                        history=kwargs.pop("history", []),
                        evaluator=kwargs.pop("evaluator", lambda t, o: []))


def test_the_period_ends_today_and_spans_the_configured_days():
    digest = _built()
    assert digest.period_end == datetime.date(2026, 8, 20)
    assert (digest.period_end - digest.period_start).days == DIGEST.default_period_days


def test_sections_can_be_switched_off():
    digest = _built(settings={"include_watchlist": False})
    assert digest.moves == ()


def test_every_digest_discloses_that_holdings_are_not_tracked():
    """The task asked for portfolio changes and Quantix has no holdings.
    Saying so in every digest is what stops the watchlist being mistaken
    for a portfolio."""
    text = _built().as_text()
    assert "holdings" in text.lower()
    assert "watch" in text.lower()


def test_an_empty_watchlist_is_explained_rather_than_shown_blank():
    digest = _built(tickers=())
    assert any("watchlist is empty" in n.lower() for n in digest.notes)


def test_an_entirely_empty_digest_says_so():
    digest = _built(tickers=())
    assert digest.is_empty
    assert "nothing moved" in digest.as_text().lower()


# --- the subject line ---------------------------------------------------------

def test_the_headline_leads_with_fired_alerts():
    """Ordered by what the user asked to be told about, not by which
    number is biggest."""
    digest = _built(history=[_FakeEvent("X", "2026-08-19T10:00:00")])
    assert "alert" in digest.headline


def test_risk_breaches_outrank_a_big_mover():
    digest = _built(evaluator=lambda t, o: ["AAPL breached"])
    assert "breach" in digest.headline


def test_movement_is_the_headline_when_nothing_was_configured_to_fire():
    digest = _built()
    assert "AAPL" in digest.headline and "%" in digest.headline


def test_a_quiet_period_still_produces_a_sensible_subject():
    digest = _built(tickers=())
    assert digest.subject() and "no material movement" in digest.subject()


def test_the_subject_carries_the_period():
    assert "20 Aug" in _built().subject()


# --- scheduled runs -----------------------------------------------------------

def test_a_disabled_digest_is_never_sent(store, monkeypatch):
    save_settings(DigestSettings(recipient="a@example.com", enabled=False), store)
    sender = _FakeSender()
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    sent, _ = run_scheduled(sender=sender, path=store)
    assert sent == 0 and sender.sent == []


def test_an_enabled_digest_with_no_recipient_is_never_sent(store, monkeypatch):
    save_settings(DigestSettings(recipient="", enabled=True), store)
    sender = _FakeSender()
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    sent, _ = run_scheduled(sender=sender, path=store)
    assert sent == 0 and sender.sent == []


def test_an_enabled_addressed_digest_sends(store, monkeypatch):
    save_settings(DigestSettings(recipient="a@example.com", enabled=True), store)
    sender = _FakeSender()
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    sent, _ = run_scheduled(sender=sender, path=store)
    assert sent == 1
    assert sender.sent[0][0] == "a@example.com"


def test_a_successful_send_records_when(store, monkeypatch):
    save_settings(DigestSettings(recipient="a@example.com", enabled=True), store)
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    run_scheduled(sender=_FakeSender(), path=store)
    assert settings_for("", store).last_sent_at != ""


def test_a_failed_send_does_not_advance_the_clock(store, monkeypatch):
    """Otherwise one SMTP hiccup silently swallows an entire period —
    the next run would consider it already delivered."""
    save_settings(DigestSettings(recipient="a@example.com", enabled=True), store)
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    sent, messages = run_scheduled(sender=_FakeSender(fail=True), path=store)
    assert sent == 0
    assert settings_for("", store).last_sent_at == ""
    assert any("FAILED" in m for m in messages)


def test_a_digest_sent_recently_is_not_due_again(store, monkeypatch):
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    now = datetime.datetime(2026, 8, 20, 8, 0)
    save_settings(DigestSettings(
        recipient="a@example.com", enabled=True, period_days=7,
        last_sent_at=(now - datetime.timedelta(days=2)).isoformat()), store)
    sender = _FakeSender()
    sent, _ = run_scheduled(now=now, sender=sender, path=store)
    assert sent == 0 and sender.sent == []


def test_a_digest_becomes_due_once_the_period_has_elapsed(store, monkeypatch):
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    now = datetime.datetime(2026, 8, 20, 8, 0)
    save_settings(DigestSettings(
        recipient="a@example.com", enabled=True, period_days=7,
        last_sent_at=(now - datetime.timedelta(days=8)).isoformat()), store)
    sent, _ = run_scheduled(now=now, sender=_FakeSender(), path=store)
    assert sent == 1


def test_lateness_does_not_skip_a_period(store, monkeypatch):
    """Due-ness is elapsed time, not day-of-week: a machine asleep on
    Monday should still get its digest on Tuesday."""
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    now = datetime.datetime(2026, 8, 20, 8, 0)   # a Thursday
    save_settings(DigestSettings(
        recipient="a@example.com", enabled=True, period_days=7,
        last_sent_at=(now - datetime.timedelta(days=30)).isoformat()), store)
    assert run_scheduled(now=now, sender=_FakeSender(), path=store)[0] == 1


def test_force_overrides_the_due_check(store, monkeypatch):
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    now = datetime.datetime(2026, 8, 20, 8, 0)
    save_settings(DigestSettings(
        recipient="a@example.com", enabled=True,
        last_sent_at=now.isoformat()), store)
    assert run_scheduled(now=now, sender=_FakeSender(), path=store, force=True)[0] == 1


def test_one_run_serves_every_configured_owner(store, monkeypatch):
    """The reason the store is shared and keyed by owner rather than
    namespaced per user — one cron invocation covers everyone."""
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    save_settings(DigestSettings(owner_key="a", recipient="a@example.com", enabled=True), store)
    save_settings(DigestSettings(owner_key="b", recipient="b@example.com", enabled=True), store)
    sender = _FakeSender()
    sent, _ = run_scheduled(sender=sender, path=store)
    assert sent == 2
    assert sorted(a for a, _, _ in sender.sent) == ["a@example.com", "b@example.com"]


def test_a_run_can_be_limited_to_one_owner(store, monkeypatch):
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    save_settings(DigestSettings(owner_key="a", recipient="a@example.com", enabled=True), store)
    save_settings(DigestSettings(owner_key="b", recipient="b@example.com", enabled=True), store)
    sender = _FakeSender()
    run_scheduled(owner_key="a", sender=sender, path=store)
    assert [a for a, _, _ in sender.sent] == ["a@example.com"]


def test_other_owners_survive_a_targeted_run(store, monkeypatch):
    """A single-owner run rewrites the store; the others must still be
    there afterwards."""
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    save_settings(DigestSettings(owner_key="a", recipient="a@example.com", enabled=True), store)
    save_settings(DigestSettings(owner_key="b", recipient="b@example.com", enabled=True), store)
    run_scheduled(owner_key="a", sender=_FakeSender(), path=store)
    assert len(load_all(store)) == 2
    assert settings_for("b", store).recipient == "b@example.com"


# --- scheduling helper --------------------------------------------------------

def test_cron_line_is_runnable_and_targets_this_script():
    line = cron_line(python_executable="/usr/bin/python3", script_dir="/opt/quantix")
    assert line.startswith("0 8 * * 1 ")
    assert "cd /opt/quantix" in line
    assert "/usr/bin/python3 digest.py --send" in line


def test_cron_line_sends_rather_than_previews():
    """A schedule wired to --preview would run silently forever and never
    deliver anything."""
    assert "--send" in cron_line()
    assert "--preview" not in cron_line()


def test_cron_line_captures_output_so_a_silent_failure_is_diagnosable():
    assert ">>" in cron_line() and "2>&1" in cron_line()


def test_a_run_with_nothing_configured_does_not_create_a_store(tmp_path):
    """A --send on a fresh install should leave no trace. Writing an empty
    store makes "does this file exist" a misleading signal about whether
    anyone has configured a digest."""
    path = tmp_path / "digest_store.json"
    sent, _ = run_scheduled(sender=_FakeSender(), path=path)
    assert sent == 0
    assert not path.exists()


def test_a_run_that_sends_nothing_leaves_the_store_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "build_digest", lambda s, **k: _built())
    path = tmp_path / "digest_store.json"
    save_settings(DigestSettings(recipient="a@example.com", enabled=False), path)
    before = path.read_text()
    run_scheduled(sender=_FakeSender(), path=path)
    assert path.read_text() == before


# --- portfolio summary --------------------------------------------------------

class _FakePerf:
    def __init__(self, market_value=1000.0, twr_pct=12.5, excess=3.0, holdings=("A",)):
        self.market_value = market_value
        self.twr_pct = twr_pct
        self.excess_vs_benchmark_pct = excess
        self.holdings = holdings


def test_no_holdings_means_no_portfolio_line():
    assert dg.portfolio_summary(builder=lambda o: None) == ""
    assert dg.portfolio_summary(builder=lambda o: _FakePerf(holdings=())) == ""


def test_the_portfolio_line_reports_value_and_time_weighted_return():
    line = dg.portfolio_summary(builder=lambda o: _FakePerf())
    assert "1,000.00" in line and "+12.50%" in line and "+3.00%" in line


def test_a_failing_portfolio_drops_the_section_rather_than_the_digest():
    """Losing a section beats losing the email."""
    def boom(owner):
        raise RuntimeError("pricing died")
    assert dg.portfolio_summary(builder=boom) == ""


def test_the_digest_no_longer_claims_holdings_cannot_be_tracked():
    """That statement was true when the digest was built and became false
    the moment portfolio_holdings shipped. A stale disclosure is a wrong
    disclosure."""
    digest = _built()
    text = digest.as_text().lower()
    assert "isn't available" not in text
    assert "does not track holdings" not in text


def test_the_note_explains_how_to_get_a_portfolio_section():
    digest = _built()
    assert any("portfolio tab" in n.lower() for n in digest.notes)


def test_a_portfolio_line_renders_as_its_own_section():
    digest = replace(_built(), portfolio_line="value 1,000.00  ·  time-weighted +12.50%")
    text = digest.as_text()
    assert "PORTFOLIO" in text and "time-weighted +12.50%" in text
