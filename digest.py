"""Weekly email digest — watchlist movement, alerts that fired, and risk
breaches, composed and sent from outside the running app.

RUN IT:  python3 digest.py --preview          print the digest, send nothing
         python3 digest.py --send             send to every enabled recipient
         python3 digest.py --send --owner KEY send just one person's

WHY THIS IS A SEPARATE SCRIPT. The task asks for a digest that reaches an
inactive user "without requiring them to check manually". That means it
has to run while Streamlit is shut — and Streamlit executes nothing when
no browser tab is open. realtime_alerts.py already documents the same
limit for in-tab alert polling. So the schedule belongs to the operating
system: cron on Linux, launchd or cron on macOS. cron_line() below
generates the exact entry, and the app shows it, but INSTALLING it stays
the user's deliberate act. A job that mails out on its own is not
something to arrange on somebody's behalf without them choosing it.

THE SETTINGS STORE IS SHARED AND KEYED BY OWNER, which is the same
lesson api_keys.py records: a script run by cron has no Streamlit
session, so auth.current_user() is None inside it, and settings filed
under a user's namespace would be invisible to the very process that has
to read them. Each record carries owner_key instead — which also means a
single scheduled run can send every configured user's digest, and each
one reads its own namespaced watchlist and alert stores.

THE PORTFOLIO SECTION IS ONE LINE, AND CONDITIONAL. When this was first
built nothing in Quantix stored holdings, so the digest said so plainly
rather than relabelling the watchlist as a portfolio. portfolio_holdings
now exists, so the digest reports value and time-weighted return when
holdings are recorded and keeps the honest note when they are not. One
line by choice: a digest is skimmed, and the dashboard is where the
position detail belongs.

NOTHING IS SENT UNLESS SOMEONE ASKED FOR IT. A recipient must be set and
the digest explicitly enabled; --preview never sends; and with SMTP
unconfigured every path degrades to printing the digest rather than
failing silently or half-sending.
"""
import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import DIGEST
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_event, log_exception, setup_logging

logger = get_logger("digest")


@dataclass(frozen=True)
class DigestSettings:
    owner_key: str = ""
    recipient: str = ""
    enabled: bool = False
    period_days: int = DIGEST.default_period_days
    include_watchlist: bool = True
    include_alerts: bool = True
    include_risk: bool = True
    last_sent_at: str = ""

    @property
    def is_sendable(self) -> bool:
        """Enabled AND addressed. Both, because an enabled digest with no
        recipient is a scheduled job that fails every week."""
        return bool(self.enabled and self.recipient.strip())


@dataclass(frozen=True)
class TickerMove:
    ticker: str
    change_pct: Optional[float] = None
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    unavailable: str = ""       # why, when it couldn't be computed

    @property
    def ok(self) -> bool:
        return self.change_pct is not None


@dataclass(frozen=True)
class Digest:
    owner_key: str
    period_start: datetime.date
    period_end: datetime.date
    moves: Tuple[TickerMove, ...] = ()
    fired_alerts: Tuple[str, ...] = ()
    risk_breaches: Tuple[str, ...] = ()
    portfolio_line: str = ""     # "" when no holdings are recorded
    notes: Tuple[str, ...] = ()

    @property
    def movers_ranked(self) -> Tuple[TickerMove, ...]:
        """Biggest absolute movers first — a digest is skimmed, so the
        thing most worth knowing has to be in the first line, not sorted
        alphabetically halfway down."""
        usable = [m for m in self.moves if m.ok]
        usable.sort(key=lambda m: abs(m.change_pct), reverse=True)
        return tuple(usable)

    @property
    def headline(self) -> str:
        """One line summarising why this email is worth opening.

        Ordered by what the user ASKED to be told about, not by which
        number is largest: a fired alert and a breached threshold are
        both signals someone deliberately configured, whereas "biggest
        mover" is just whatever moved most. A subject line that leads
        with a 3% move while two configured thresholds are breached
        buries the part that was actually requested.
        """
        if self.fired_alerts:
            return f"{len(self.fired_alerts)} alert{'s' if len(self.fired_alerts) != 1 else ''} fired"
        if self.risk_breaches:
            count = len(self.risk_breaches)
            return f"{count} risk threshold{'s' if count != 1 else ''} breached"
        ranked = self.movers_ranked
        if ranked:
            top = ranked[0]
            return f"{top.ticker} {top.change_pct:+.1f}%"
        return "no material movement"

    @property
    def is_empty(self) -> bool:
        return not (self.movers_ranked or self.fired_alerts or self.risk_breaches)

    def subject(self) -> str:
        period = f"{self.period_start:%d %b} – {self.period_end:%d %b}"
        return DIGEST.subject_template.format(period=period, headline=self.headline)

    def as_text(self) -> str:
        lines = [
            f"Quantix digest for {self.period_start:%d %B %Y} to {self.period_end:%d %B %Y}",
            "",
        ]

        ranked = self.movers_ranked
        unavailable = [m for m in self.moves if not m.ok]
        if ranked or unavailable:
            lines.append(f"WATCHLIST MOVEMENT ({len(ranked)} of {len(self.moves)} priced)")
            for move in ranked[:DIGEST.max_movers_shown]:
                lines.append(
                    f"  {move.ticker:<8} {move.change_pct:+7.2f}%   "
                    f"{move.start_price:,.2f} -> {move.end_price:,.2f}"
                )
            if len(ranked) > DIGEST.max_movers_shown:
                lines.append(f"  ... and {len(ranked) - DIGEST.max_movers_shown} more")
            for move in unavailable:
                # Named rather than silently dropped: a ticker vanishing
                # from the digest looks like it didn't move.
                lines.append(f"  {move.ticker:<8} unavailable — {move.unavailable}")
            lines.append("")

        if self.fired_alerts:
            lines.append(f"ALERTS THAT FIRED ({len(self.fired_alerts)})")
            for entry in self.fired_alerts[:DIGEST.max_alerts_shown]:
                lines.append(f"  {entry}")
            if len(self.fired_alerts) > DIGEST.max_alerts_shown:
                lines.append(f"  ... and {len(self.fired_alerts) - DIGEST.max_alerts_shown} more")
            lines.append("")

        if self.portfolio_line:
            lines.append("PORTFOLIO")
            lines.append(f"  {self.portfolio_line}")
            lines.append("")

        if self.risk_breaches:
            lines.append(f"RISK THRESHOLDS BREACHED NOW ({len(self.risk_breaches)})")
            for entry in self.risk_breaches[:DIGEST.max_alerts_shown]:
                lines.append(f"  {entry}")
            lines.append("")

        if self.is_empty:
            lines += ["Nothing moved enough to report over this period.", ""]

        for note in self.notes:
            lines.append(note)
        if self.notes:
            lines.append("")

        lines += [
            "---",
            "Prices are the closing values Quantix could fetch for the period; a figure shown",
            "as unavailable was not reported, never assumed to be zero. Portfolio return is",
            "time-weighted, so money you added is not counted as a gain.",
        ]
        return "\n".join(lines)


# --- settings store -----------------------------------------------------------

def _store_path() -> Path:
    # shared_path, not store_path — see the module docstring.
    return shared_path(DIGEST.store_filename)


def load_all(path: Optional[Path] = None) -> Tuple[DigestSettings, ...]:
    """Every configured recipient. Never raises; a corrupt file degrades
    to none configured, which sends nothing — the safe direction."""
    path = path or _store_path()
    if not path.exists():
        return ()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "digest.store_corrupt", section="digest")
        return ()
    if not isinstance(raw, dict):
        return ()

    out: List[DigestSettings] = []
    for item in raw.get("settings", []):
        if not isinstance(item, dict):
            continue
        try:
            period = int(item.get("period_days", DIGEST.default_period_days))
        except (TypeError, ValueError):
            period = DIGEST.default_period_days
        period = max(DIGEST.min_period_days, min(DIGEST.max_period_days, period))
        out.append(DigestSettings(
            owner_key=str(item.get("owner_key") or ""),
            recipient=str(item.get("recipient") or ""),
            enabled=bool(item.get("enabled", False)),
            period_days=period,
            include_watchlist=bool(item.get("include_watchlist", True)),
            include_alerts=bool(item.get("include_alerts", True)),
            include_risk=bool(item.get("include_risk", True)),
            last_sent_at=str(item.get("last_sent_at") or ""),
        ))
    return tuple(out)


def save_all(settings: Tuple[DigestSettings, ...], path: Optional[Path] = None) -> None:
    path = path or _store_path()
    payload = {"settings": [{
        "owner_key": s.owner_key, "recipient": s.recipient, "enabled": s.enabled,
        "period_days": s.period_days, "include_watchlist": s.include_watchlist,
        "include_alerts": s.include_alerts, "include_risk": s.include_risk,
        "last_sent_at": s.last_sent_at,
    } for s in settings]}
    atomic_write_text(path, json.dumps(payload, indent=2))


def settings_for(owner_key: str = "", path: Optional[Path] = None) -> DigestSettings:
    """This owner's settings, or unconfigured defaults."""
    for entry in load_all(path):
        if entry.owner_key == owner_key:
            return entry
    return DigestSettings(owner_key=owner_key)


def save_settings(settings: DigestSettings, path: Optional[Path] = None) -> Tuple[DigestSettings, ...]:
    existing = [s for s in load_all(path) if s.owner_key != settings.owner_key]
    updated = tuple(existing + [settings])
    save_all(updated, path)
    return updated


def validate(settings: DigestSettings) -> Optional[str]:
    if settings.enabled and not settings.recipient.strip():
        return "Add an address to send the digest to, or leave it switched off."
    if settings.recipient.strip() and "@" not in settings.recipient:
        return "That doesn't look like an email address."
    if not (DIGEST.min_period_days <= settings.period_days <= DIGEST.max_period_days):
        return f"The period must be between {DIGEST.min_period_days} and {DIGEST.max_period_days} days."
    if settings.enabled and not any(
        (settings.include_watchlist, settings.include_alerts, settings.include_risk)
    ):
        return "Include at least one section, or the digest would be empty every week."
    return None


# --- gathering ----------------------------------------------------------------

def _namespaced(filename: str, owner_key: str) -> Path:
    """A per-user store's path for a given owner, resolved without a
    session — the script has none."""
    import local_store
    return local_store.store_path(filename, namespace=owner_key or None)


def watchlist_tickers_for(owner_key: str = "") -> Tuple[str, ...]:
    from config import WATCHLIST_PANEL
    from watchlist_panel import load_watchlist_store

    try:
        store = load_watchlist_store(_namespaced(WATCHLIST_PANEL.store_filename, owner_key))
    except Exception:
        log_exception(logger, "digest.watchlist_unreadable", section="digest")
        return ()
    seen: List[str] = []
    for entry in store.lists.values():
        for ticker in entry.tickers:
            if ticker not in seen:
                seen.append(ticker)
    return tuple(seen)


def compute_moves(tickers: Tuple[str, ...], period_days: int,
                  end: Optional[datetime.date] = None,
                  loader: Optional[Callable] = None) -> Tuple[TickerMove, ...]:
    """Percentage change per ticker over the period, from closing prices.

    `loader` is injectable so tests never touch the network. A ticker that
    can't be priced comes back as an explicit unavailable row rather than
    being dropped — silently omitting it would read as "it didn't move".
    """
    end = end or datetime.date.today()
    start = end - datetime.timedelta(days=period_days)
    if loader is None:
        from data_loader import load_price_history_only
        loader = load_price_history_only

    moves: List[TickerMove] = []
    for ticker in tickers:
        try:
            history, errors = loader(ticker, start, end)
        except Exception as e:
            moves.append(TickerMove(ticker=ticker, unavailable=f"fetch failed ({type(e).__name__})"))
            continue
        if history is None or getattr(history, "empty", True):
            moves.append(TickerMove(ticker=ticker, unavailable=(errors or ["no price data"])[0]))
            continue
        closes = history["Close"].dropna()
        if len(closes) < 2:
            moves.append(TickerMove(
                ticker=ticker,
                unavailable=f"only {len(closes)} closing price(s) in this period",
            ))
            continue
        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        if first == 0:
            moves.append(TickerMove(ticker=ticker, unavailable="opening price was zero"))
            continue
        moves.append(TickerMove(
            ticker=ticker,
            change_pct=((last - first) / first) * 100,
            start_price=first,
            end_price=last,
        ))
    return tuple(moves)


def fired_alerts_since(since: datetime.datetime, owner_key: str = "",
                       history: Optional[List] = None) -> Tuple[str, ...]:
    """Real-time alert triggers recorded since `since`, newest first."""
    if history is None:
        from config import REALTIME_ALERTS
        from realtime_alerts import load_store

        try:
            _rules, history = load_store(_namespaced(REALTIME_ALERTS.store_filename, owner_key))
        except Exception:
            log_exception(logger, "digest.alert_history_unreadable", section="digest")
            return ()

    out: List[str] = []
    for event in history or []:
        stamp = getattr(event, "triggered_at", "")
        try:
            when = datetime.datetime.fromisoformat(str(stamp))
        except ValueError:
            continue
        if when < since:
            continue
        out.append(
            f"{when:%d %b %H:%M}  {getattr(event, 'ticker', '?')}  "
            f"{getattr(event, 'trigger_type', '')}  {getattr(event, 'detail', '')}".rstrip()
        )
    out.reverse()
    return tuple(out)


def current_risk_breaches(tickers: Tuple[str, ...], owner_key: str = "",
                          evaluator: Optional[Callable] = None) -> Tuple[str, ...]:
    """Risk rules breaching right now.

    Evaluated fresh rather than read from history, because "your watchlist
    is breaching a threshold today" is what a weekly summary should tell
    you — a stale breach from Tuesday that has since resolved is noise.
    """
    if not tickers:
        return ()
    if evaluator is not None:
        return tuple(evaluator(tickers, owner_key))
    try:
        import risk_alerts as ra

        rules = ra.load_rules(_namespaced("risk_alert_rules_store.json", owner_key))
        if not rules:
            return ()
        snapshots = ra.compute_watchlist_snapshots(tickers)
        triggered = ra.evaluate_alerts(snapshots, tuple(ra.AlertRule(**r) for r in rules))
    except Exception:
        log_exception(logger, "digest.risk_evaluation_failed", section="digest")
        return ()
    return tuple(
        f"{t.ticker}  {t.rule.metric} {t.rule.operator} {t.rule.threshold}  (now {t.value})"
        for t in triggered
    )


def portfolio_summary(owner_key: str = "", builder: Optional[Callable] = None) -> str:
    """One line on the portfolio, or "" when no holdings are recorded.

    Deliberately one line. A digest is skimmed, and the dashboard is
    where the detail lives — repeating the full position table here would
    make the email long enough to stop being a digest.

    Never raises: a portfolio that can't be priced simply drops out of the
    email rather than failing the send. Losing a section beats losing the
    digest.
    """
    try:
        if builder is not None:
            performance = builder(owner_key)
        else:
            import local_store
            import portfolio_holdings as ph
            from config import PORTFOLIO

            store = ph.load_store(
                local_store.store_path(PORTFOLIO.store_filename, namespace=owner_key or None))
            holdings = store.holdings()
            if not holdings:
                return ""

            def loader(ticker, start, end):
                from data_loader import load_price_history_only
                history, _ = load_price_history_only(ticker, start, end)
                if history is None or history.empty or "Close" not in history:
                    return None
                closes = history["Close"].dropna()
                return closes if not closes.empty else None

            performance = ph.build_performance(holdings, loader)
        if performance is None or not performance.holdings:
            return ""
    except Exception:
        log_exception(logger, "digest.portfolio_summary_failed", section="digest")
        return ""

    parts = [f"value {performance.market_value:,.2f}"]
    if performance.twr_pct is not None:
        parts.append(f"time-weighted {performance.twr_pct:+.2f}%")
    if performance.excess_vs_benchmark_pct is not None:
        parts.append(f"vs benchmark {performance.excess_vs_benchmark_pct:+.2f}%")
    return "  ·  ".join(parts)


def build_digest(settings: DigestSettings, end: Optional[datetime.date] = None,
                 loader: Optional[Callable] = None,
                 history: Optional[List] = None,
                 evaluator: Optional[Callable] = None,
                 tickers: Optional[Tuple[str, ...]] = None) -> Digest:
    """Assemble one digest. Every dependency is injectable so this is
    fully testable without the network or any store on disk."""
    end = end or datetime.date.today()
    start = end - datetime.timedelta(days=settings.period_days)
    owner = settings.owner_key

    if tickers is None:
        tickers = watchlist_tickers_for(owner)

    notes: List[str] = []
    moves: Tuple[TickerMove, ...] = ()
    fired: Tuple[str, ...] = ()
    breaches: Tuple[str, ...] = ()

    if settings.include_watchlist:
        if tickers:
            moves = compute_moves(tickers, settings.period_days, end=end, loader=loader)
        else:
            notes.append("Your watchlist is empty, so there was no movement to report.")
    if settings.include_alerts:
        fired = fired_alerts_since(
            datetime.datetime.combine(start, datetime.time.min), owner, history=history)
    if settings.include_risk:
        breaches = current_risk_breaches(tickers, owner, evaluator=evaluator)

    portfolio_line = portfolio_summary(owner)
    if not portfolio_line:
        notes.append(
            "No holdings are recorded, so this digest covers your watchlist rather than "
            "positions. Add holdings in the Portfolio tab to have them summarised here."
        )
    return Digest(owner_key=owner, period_start=start, period_end=end,
                  moves=moves, fired_alerts=fired, risk_breaches=breaches,
                  portfolio_line=portfolio_line, notes=tuple(notes))


# --- sending ------------------------------------------------------------------

def send_digest(digest: Digest, settings: DigestSettings,
                sender: Optional[Callable] = None) -> Tuple[bool, Optional[str]]:
    """Email one digest. Returns (ok, error); never raises.

    Refuses rather than guesses when there is no recipient: a digest with
    nowhere to go is a configuration mistake, not something to paper over.
    """
    if not settings.recipient.strip():
        return False, "No recipient is configured for this digest."
    if sender is None:
        from email_report import send_notification_email
        sender = send_notification_email
    ok, err = sender(settings.recipient.strip(), digest.subject(), digest.as_text())
    if ok:
        log_event(logger, logging.INFO, "digest.sent",
                  movers=len(digest.movers_ranked), alerts=len(digest.fired_alerts))
    return ok, err


def run_scheduled(now: Optional[datetime.datetime] = None, owner_key: Optional[str] = None,
                  sender: Optional[Callable] = None, path: Optional[Path] = None,
                  force: bool = False) -> Tuple[int, List[str]]:
    """Send every digest that is enabled and due. Returns (sent, messages).

    Due-checking is by elapsed time since the last successful send, not by
    day-of-week: a machine that was asleep on Monday should still get its
    digest on Tuesday rather than skipping the week entirely.
    """
    now = now or datetime.datetime.now()
    messages: List[str] = []
    sent = 0
    remaining: List[DigestSettings] = []
    # Only rewrite the store if a timestamp actually moved. Without this,
    # a --send on an instance with nothing configured creates an empty
    # store file for no reason, which makes "does the store exist" a
    # misleading signal about whether anyone has set a digest up.
    changed = False

    for settings in load_all(path):
        if owner_key is not None and settings.owner_key != owner_key:
            remaining.append(settings)
            continue
        if not settings.is_sendable:
            remaining.append(settings)
            continue
        if not force and not _is_due(settings, now):
            remaining.append(settings)
            messages.append(f"{settings.recipient}: not due yet")
            continue

        digest = build_digest(settings, end=now.date())
        ok, err = send_digest(digest, settings, sender=sender)
        if ok:
            sent += 1
            messages.append(f"{settings.recipient}: sent — {digest.headline}")
            remaining.append(replace(settings, last_sent_at=now.isoformat(timespec="seconds")))
            changed = True
        else:
            # last_sent_at is NOT advanced on failure, so the next run
            # retries rather than skipping the period entirely.
            messages.append(f"{settings.recipient}: FAILED — {err}")
            remaining.append(settings)

    if changed:
        save_all(tuple(remaining), path)
    return sent, messages


def _is_due(settings: DigestSettings, now: datetime.datetime) -> bool:
    if not settings.last_sent_at:
        return True
    try:
        last = datetime.datetime.fromisoformat(settings.last_sent_at)
    except ValueError:
        return True
    return (now - last) >= datetime.timedelta(days=settings.period_days)


# --- scheduling helper --------------------------------------------------------

def cron_line(python_executable: Optional[str] = None, script_dir: Optional[Path] = None,
              weekday: int = 1, hour: int = 8) -> str:
    """The crontab entry that runs this weekly.

    Generated rather than hand-written into the docs so the paths are
    this machine's actual ones. Installing it is deliberately left to the
    user — see the module docstring.
    """
    python_executable = python_executable or sys.executable
    script_dir = script_dir or Path(__file__).resolve().parent
    return (
        f"{0} {hour} * * {weekday} cd {script_dir} && "
        f"{python_executable} digest.py --send >> digest_cron.log 2>&1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantix weekly email digest.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true",
                       help="Print the digest without sending anything.")
    group.add_argument("--send", action="store_true", help="Send to enabled recipients that are due.")
    group.add_argument("--cron-line", action="store_true",
                       help="Print the crontab entry for a weekly run, then exit.")
    parser.add_argument("--owner", default=None, help="Limit to one owner key.")
    parser.add_argument("--force", action="store_true",
                        help="With --send, ignore the due check and send now.")
    args = parser.parse_args()

    if args.cron_line:
        print(cron_line())
        return 0

    setup_logging()

    if args.preview:
        configured = load_all()
        if args.owner is not None:
            configured = tuple(s for s in configured if s.owner_key == args.owner)
        if not configured:
            # Preview must still show something useful on a fresh install,
            # or there is no way to see what the digest looks like before
            # committing to a schedule.
            configured = (DigestSettings(recipient="(nobody configured)"),)
        for settings in configured:
            digest = build_digest(settings)
            print("=" * 72)
            print(f"To: {settings.recipient or '(unset)'}")
            print(f"Subject: {digest.subject()}")
            print("=" * 72)
            print(digest.as_text())
        return 0

    sent, messages = run_scheduled(owner_key=args.owner, force=args.force)
    for message in messages:
        print(message)
    print(f"{sent} digest(s) sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
