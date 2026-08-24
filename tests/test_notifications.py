"""The notification bell: unread counting, and snoozing a rule.

The two decisions worth pinning were settled with the user before
building, and both are the kind a later reader would "simplify" back:

  * The bell counts the real-time engine's persisted TriggerEvents ONLY.
    The Smart Risk-Aware Alerts are an on-demand snapshot with no event
    log and no timestamps, so counting them means inventing an occurrence
    time and a dedup key.
  * Snooze mutes the RULE, not the notification. The event has already
    happened; hiding it changes nothing and the rule fires again on the
    next poll.
"""
import datetime
import pathlib

import pytest

import notifications as notif


ROOT = pathlib.Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")

NOW = datetime.datetime(2026, 8, 24, 12, 0, 0)


class Event:
    def __init__(self, when, rule_id="r1", ticker="AAPL"):
        self.triggered_at = when
        self.rule_id = rule_id
        self.ticker = ticker
        self.trigger_type = "sma_cross_bullish"
        self.detail = "crossed"


@pytest.fixture(autouse=True)
def sandboxed_store(tmp_path, monkeypatch):
    """Never the real notifications.json."""
    import local_store

    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    yield tmp_path


# --- unread -------------------------------------------------------------------

def test_with_no_watermark_everything_is_unread():
    """A first-time user genuinely has not seen any of them; defaulting to
    "all read" hides the feature the first time it matters."""
    history = [Event("2026-08-24T09:00:00"), Event("2026-08-24T11:00:00")]
    assert len(notif.unread(history, None)) == 2


def test_only_events_after_the_watermark_count():
    history = [Event("2026-08-24T09:00:00"), Event("2026-08-24T11:00:00"),
               Event("2026-08-24T11:55:00")]
    seen = datetime.datetime(2026, 8, 24, 10, 0, 0)
    assert len(notif.unread(history, seen)) == 2


def test_unread_is_newest_first():
    history = [Event("2026-08-24T09:00:00"), Event("2026-08-24T11:00:00")]
    assert notif.unread(history, None)[0].triggered_at == "2026-08-24T11:00:00"


def test_an_unparseable_timestamp_does_not_crash_the_count():
    """A store written by another version, or hand-edited."""
    history = [Event("not-a-date"), Event("2026-08-24T11:00:00")]
    seen = datetime.datetime(2026, 8, 24, 10, 0, 0)
    assert len(notif.unread(history, seen)) == 1


def test_marking_read_moves_the_watermark():
    history = [Event("2026-08-24T11:00:00")]
    assert len(notif.unread(history, notif.last_seen())) == 1
    notif.mark_all_read(NOW)
    assert notif.unread(history, notif.last_seen()) == []


def test_the_watermark_survives_history_being_trimmed():
    """History is capped at 100 and rewritten on every save. A per-event
    read flag would be dropped with the events it described, and old
    alerts would silently become unread again."""
    notif.mark_all_read(NOW)
    seen = notif.last_seen()
    # The whole history is replaced by newer events; the watermark holds.
    assert notif.unread([Event("2026-08-24T11:00:00")], seen) == []
    assert len(notif.unread([Event("2026-08-24T13:00:00")], seen)) == 1


@pytest.mark.parametrize("count,expected", [(0, ""), (1, "1"), (9, "9"), (10, "9+"), (99, "9+")])
def test_the_badge_caps_rather_than_growing(count, expected):
    assert notif.badge_text(count) == expected


def test_a_zero_badge_is_blank_not_zero():
    """So the control reads "Alerts" rather than "Alerts 0"."""
    assert notif.badge_text(0) == ""


# --- snoozing -----------------------------------------------------------------

def test_snoozing_mutes_the_rule_for_the_requested_period():
    until = notif.snooze("r1", 4, NOW)
    assert until == NOW + datetime.timedelta(hours=4)
    assert notif.is_muted("r1", NOW)


def test_a_mute_lapses_on_its_own():
    """Nothing writes an "unmute" event, so is_muted compares against the
    clock every time — a store carried across a restart cannot leave a
    rule permanently silent."""
    notif.snooze("r1", 1, NOW)
    assert notif.is_muted("r1", NOW)
    assert not notif.is_muted("r1", NOW + datetime.timedelta(hours=2))


def test_expired_mutes_are_not_reported_as_active():
    notif.snooze("r1", 1, NOW)
    assert notif.mutes(NOW + datetime.timedelta(hours=2)) == {}


def test_unsnoozing_takes_effect_immediately():
    notif.snooze("r1", 4, NOW)
    assert notif.unsnooze("r1") is True
    assert not notif.is_muted("r1", NOW)


def test_unsnoozing_something_not_muted_is_a_no_op():
    assert notif.unsnooze("nope") is False


def test_snoozing_only_affects_the_rule_asked_for():
    notif.snooze("r1", 4, NOW)
    assert notif.is_muted("r1", NOW)
    assert not notif.is_muted("r2", NOW)


@pytest.mark.parametrize("hours", [0, -1])
def test_a_nonsense_duration_is_refused(hours):
    assert notif.snooze("r1", hours, NOW) is None
    assert not notif.is_muted("r1", NOW)


def test_the_mute_is_described_as_a_duration():
    """An absolute time makes the reader do the subtraction."""
    until = notif.snooze("r1", 4, NOW)
    assert notif.describe_mute(until, NOW) == "muted for another 4h 00m"
    soon = NOW + datetime.timedelta(minutes=47)
    assert notif.describe_mute(soon, NOW) == "muted for another 47 min"


# --- a corrupt store ----------------------------------------------------------

def test_a_corrupt_store_is_not_silently_overwritten(sandboxed_store):
    """Treating it as empty would un-snooze every muted rule on the next
    write — the same distinction screener_templates makes."""
    (sandboxed_store / notif.STORE_FILENAME).write_text("{not json", encoding="utf-8")
    assert notif.store_is_corrupt()
    assert notif.mark_all_read(NOW) is False
    assert notif.snooze("r1", 4, NOW) is None
    assert (sandboxed_store / notif.STORE_FILENAME).read_text(encoding="utf-8") == "{not json"


# --- ages ---------------------------------------------------------------------

@pytest.mark.parametrize("delta,expected", [
    (datetime.timedelta(seconds=5), "5s ago"),
    (datetime.timedelta(minutes=4), "4 min ago"),
    (datetime.timedelta(hours=3), "3h ago"),
    (datetime.timedelta(days=2), "2d ago"),
])
def test_ages_read_as_durations(delta, expected):
    assert notif.describe_age((NOW - delta).isoformat(timespec="seconds"), NOW) == expected


def test_a_missing_timestamp_says_so_rather_than_guessing():
    assert notif.describe_age(None) == "time unknown"
    assert notif.describe_age("garbage") == "time unknown"


# --- the wiring ---------------------------------------------------------------

def test_the_mute_suppresses_the_EVENT_not_just_the_display():
    """The only place snooze can meaningfully act. If the event is still
    recorded, the badge still counts it and the mute is cosmetic."""
    block = FINANCE[FINANCE.index("_rt_newly_triggered:"):]
    block = block[:block.index("rt_save_store")]
    assert "notifications.is_muted(_rt_rid)" in block
    assert "continue" in block
    # ...and it must come BEFORE the append, or it suppresses nothing.
    assert block.index("is_muted") < block.index("rt_alert_history\"].append")


def test_the_bell_counts_only_the_persisted_event_log():
    """The risk-alert snapshot has no event log and no timestamps."""
    block = FINANCE[FINANCE.index("# --- Notification bell"):]
    block = block[:block.index("with _pm_slot:")]
    assert 'st.session_state.get("rt_alert_history"' in block
    assert "risk_alert_rules" not in block
    assert "evaluate_alerts" not in block


def test_the_bell_sits_in_the_sticky_header_beside_the_profile_menu():
    assert "_nb_slot" in FINANCE
    header = FINANCE.index("with symbol_header_container:")
    bell = FINANCE.index("# --- Notification bell")
    tabs = FINANCE.index("= st.tabs([")
    assert header < bell < tabs


def test_clearing_history_also_resets_the_watermark():
    """Otherwise a restored backup resurrects the badge count."""
    block = FINANCE[FINANCE.index('key="notif_clear"'):]
    block = block[:block.index("st.rerun()")]
    assert "notifications.clear_history()" in block
    assert "rt_save_store" in block


def test_clearing_history_is_marked_destructive():
    import button_roles

    assert "notif_clear" in button_roles.DANGER_PREFIXES
