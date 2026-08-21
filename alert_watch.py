"""Headless alert checking, for posting to Slack while Quantix is shut.

RUN IT:  python3 alert_watch.py --check     evaluate and print, post nothing
         python3 alert_watch.py --post      evaluate and post new triggers

WHY THIS EXISTS SEPARATELY FROM THE APP. realtime_alerts evaluates rules
inside the page, via st.fragment(run_every=...) polling — so alerts only
fire while a browser tab is open. A Slack message about something already
on your screen is not what Slack delivery is for. This runs the same
evaluation headlessly under cron, sharing the digest's schedule so there
is still only one thing to install.

DUPLICATE SUPPRESSION IS THE WHOLE PROBLEM. A rule that is breaching stays
breaching: "AAPL below 200" is just as true fifteen minutes later. A
checker that posted current state every run would send the same alert
until the price moved, and that channel would be muted within a day.

realtime_alerts.detect_new_triggers is already edge-triggered — it
returns only rules that went from not-met to met — but it needs the
PREVIOUS run's active set to compare against, and a cron process holds
nothing between invocations. So the active set is persisted here.

The state file is shared rather than namespaced per user, for the same
reason as api_keys and the digest: a cron process has no Streamlit
session, so auth.current_user() is None inside it and per-user state
would be invisible to the process that has to read it.

STATE IS ONLY ADVANCED ON A SUCCESSFUL POST. If Slack is unreachable, the
previous active set is left untouched so the next run retries rather than
silently swallowing the alert — the same rule the digest follows for a
failed send.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import SLACK
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_event, log_exception, setup_logging

logger = get_logger("alert_watch")


def _state_path() -> Path:
    return shared_path(SLACK.state_filename)


def load_active(path: Optional[Path] = None) -> Dict[str, bool]:
    """Which rule ids were breaching on the previous run.

    Never raises. A missing or corrupt file means "nothing was active",
    which makes the next run treat every currently-met rule as new. That
    is the safe direction: a duplicate notification is annoying, a
    silently dropped one is a missed alert the user asked for.
    """
    path = path or _state_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "alert_watch.state_corrupt", section="alert_watch")
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): bool(v) for k, v in (raw.get("active") or {}).items()}


def save_active(active: Dict[str, bool], path: Optional[Path] = None) -> None:
    atomic_write_text(path or _state_path(),
                      json.dumps({"active": {k: bool(v) for k, v in active.items()}}, indent=2))


def check(evaluator: Optional[Callable] = None,
          rules_loader: Optional[Callable] = None,
          path: Optional[Path] = None) -> Tuple[List[Tuple[str, str, str]], Dict[str, bool], List[str]]:
    """Evaluate every rule and return (new_alerts, current_active, notes).

    `new_alerts` is a list of (ticker, trigger_type, detail) for rules
    that JUST transitioned to met. `current_active` is the full state to
    persist once the alerts have actually been delivered.

    Dependencies are injected so tests never touch the network.
    """
    notes: List[str] = []

    if rules_loader is None:
        def rules_loader():
            from realtime_alerts import load_store
            rules, _history = load_store()
            return rules
    if evaluator is None:
        from realtime_alerts import evaluate_all as evaluator

    try:
        rules = rules_loader()
    except Exception:
        log_exception(logger, "alert_watch.rules_unreadable", section="alert_watch")
        return [], {}, ["Couldn't read the alert rules."]

    if not rules:
        return [], {}, ["No alert rules are configured."]

    try:
        results = evaluator(rules)
    except Exception as e:
        log_exception(logger, "alert_watch.evaluation_failed", section="alert_watch")
        return [], {}, [f"Couldn't evaluate the rules ({type(e).__name__})."]

    previously_active = load_active(path)

    from realtime_alerts import detect_new_triggers
    newly = detect_new_triggers(results, previously_active)

    by_id = {rule.id: rule for rule in rules}
    alerts: List[Tuple[str, str, str]] = []
    for rule_id in newly:
        rule = by_id.get(rule_id)
        result = results.get(rule_id)
        if rule is None or result is None:
            continue
        alerts.append((rule.ticker, rule.trigger_type, getattr(result, "detail", "") or "met"))

    current_active = {rule_id: bool(getattr(r, "is_met", False)) for rule_id, r in results.items()}
    log_event(logger, logging.INFO, "alert_watch.checked",
              rules=len(rules), active=sum(current_active.values()), new=len(alerts))
    return alerts, current_active, notes


def run(post: bool, poster: Optional[Callable] = None,
        evaluator: Optional[Callable] = None,
        rules_loader: Optional[Callable] = None,
        path: Optional[Path] = None) -> Tuple[int, List[str]]:
    """Check, and post if asked. Returns (posted_count, messages)."""
    alerts, current_active, notes = check(
        evaluator=evaluator, rules_loader=rules_loader, path=path)
    messages = list(notes)

    if not post:
        # A dry run writes NOTHING. Advancing state here would mark
        # alerts as seen without anyone being told about them, so the
        # real run would then skip them — a --check that silently
        # swallows the next notification is a trap.
        if alerts:
            messages.append(f"{len(alerts)} new alert(s) — not posted (--check).")
            for ticker, trigger_type, detail in alerts:
                messages.append(f"  {ticker} {trigger_type}: {detail}")
        else:
            messages.append("No newly triggered alerts.")
        return 0, messages

    if not alerts:
        # Nothing new, but the active set still has to advance: a rule
        # that RESOLVED must drop out of the active set, or it would
        # never register as newly-met the next time it trips.
        if current_active:
            save_active(current_active, path)
        messages.append("No newly triggered alerts.")
        return 0, messages

    import slack_notify

    ok, error = slack_notify.post_alerts(alerts, poster=poster)
    if ok:
        # Advance state ONLY after a successful post. On failure the
        # previous set is left alone so the next run retries, rather than
        # marking an alert delivered that never arrived.
        save_active(current_active, path)
        messages.append(f"Posted {len(alerts)} alert(s) to Slack.")
        return len(alerts), messages

    messages.append(f"FAILED to post — {error}")
    return 0, messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Quantix alert rules and post new triggers to Slack.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="Evaluate and print. Posts nothing and does not advance state.")
    group.add_argument("--post", action="store_true",
                       help="Evaluate and post newly triggered alerts to Slack.")
    args = parser.parse_args()

    setup_logging()
    posted, messages = run(post=args.post)
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
