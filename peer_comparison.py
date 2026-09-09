"""How your portfolio return compares with other people on this instance.

THE TICKET ASKED FOR "YOU BEAT 72% OF USERS" AND THAT POPULATION DOES NOT
EXIST. Measured on 2026-09-10: this instance has ONE local account, no
portfolio stores at all, and no backend — `api_server.py` is a read-only
API bound to loopback, not a service that collects anything. Quantix runs
on your machine. There is no fleet of users to be in the 28th percentile
of, and inventing one would mean printing a percentile against a
population that was never measured, which is the single thing this
codebase refuses to do.

SO "OTHER USERS" MEANS THE OTHER ACCOUNTS ON THIS INSTANCE, and that is a
real case rather than a consolation prize: branding.py exists because a
licensee runs Quantix for a team, and on such an instance the accounts
are colleagues. On a laptop with one account the honest answer is "there
is no cohort", and that is what this returns — not a fabricated 72%.

WHAT IS SHARED IS ONE NUMBER, AND ONLY IF YOU SAY SO. An entry holds the
period, a time-weighted return, and the account key that already exists
as a hash in `users/`. It does NOT hold holdings, tickers, market value
or cost. Someone's rate of return is a very different disclosure from
what they own and how much of it, and a comparison feature needs only the
first. `publish()` cannot write the others because it is never given
them.

THE COMPARISON IS ONLY VALID OVER AN IDENTICAL WINDOW, and the obvious
implementation gets this wrong. `build_performance` starts each portfolio
at `min(purchase_date)` — its OWN earliest purchase — so one person's
figure covers three years and another's three months. Ranking those
against each other would reward whoever has held longest and call it
skill. Every entry therefore carries a calendar-month label and only
entries sharing that label are compared; `window_return()` recomputes the
return over that month rather than reusing the dashboard's
inception-to-date number.

TIME-WEIGHTED, NEVER MONEY-WEIGHTED. portfolio_holdings says it plainly
about the benchmark and the same applies here: MWR depends on when money
arrived, so a well-timed deposit flatters it. TWR is the figure it is
fair to difference between people.

A COHORT SMALLER THAN `min_cohort` GETS NO PERCENTILE. With one other
participant, "you beat 100%" discloses that person's return exactly; with
two it narrows it to a half. A percentile over a handful of people is
both statistically empty and a deanonymisation, so below the floor this
reports the cohort size and nothing else.
"""
import datetime
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import PEER_COMPARISON
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_exception

logger = get_logger("peer_comparison")


@dataclass(frozen=True)
class PeerEntry:
    """One participant's return for one month.

    Deliberately four fields. Anything richer — a ticker, a market value,
    a holdings count — would turn an opt-in comparison into a disclosure
    of what someone owns.
    """
    user_key: str
    period: str            # "2026-09"
    twr_pct: float
    updated_at: str


@dataclass(frozen=True)
class PeerStore:
    entries: Tuple[PeerEntry, ...] = ()
    corrupt: bool = False

    def for_period(self, period: str) -> Tuple[PeerEntry, ...]:
        return tuple(e for e in self.entries if e.period == period)

    def entry_for(self, user_key: str, period: str) -> Optional[PeerEntry]:
        return next((e for e in self.entries
                     if e.user_key == user_key and e.period == period), None)


@dataclass(frozen=True)
class Comparison:
    """The answer, or the reason there isn't one."""
    period: str
    your_return_pct: Optional[float] = None
    cohort_size: int = 0                 # OTHER participants, not counting you
    beat_count: int = 0
    percentile: Optional[float] = None   # None whenever it must not be shown
    reason: str = ""

    @property
    def has_result(self) -> bool:
        return self.percentile is not None

    def sentence(self) -> str:
        if not self.has_result:
            return self.reason
        return (f"You beat {self.percentile:.0f}% of the {self.cohort_size} "
                f"other people sharing a return for {self.period}.")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _store_path() -> Path:
    # Shared by definition: a comparison across accounts cannot live in
    # one account's namespace. This is the ONLY portfolio-derived figure
    # that leaves a user's own directory, which is why it is opt-in.
    return shared_path(PEER_COMPARISON.store_filename)


def period_for(day: Optional[datetime.date] = None) -> str:
    """The calendar month label every entry is filed under.

    A calendar month rather than a trailing 30 days because it is
    identical for everyone by definition — a trailing window depends on
    when each person happened to open the app, which would quietly
    reintroduce the mismatched-window problem this label exists to fix.
    """
    day = day or datetime.date.today()
    return f"{day.year:04d}-{day.month:02d}"


def period_bounds(period: str) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    """(first day, last day) of a period label, or (None, None)."""
    try:
        year, month = (int(part) for part in period.split("-"))
        start = datetime.date(year, month, 1)
    except Exception:
        return None, None
    if month == 12:
        end = datetime.date(year, 12, 31)
    else:
        end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    return start, end


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None) -> PeerStore:
    """Never raises. A corrupt file is flagged rather than treated as
    empty, so the next publish cannot quietly replace other people's
    entries with one."""
    path = path or _store_path()
    if not path.exists():
        return PeerStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "peer_comparison.store_corrupt",
                      section="peer_comparison")
        return PeerStore(corrupt=True)
    if not isinstance(raw, dict):
        return PeerStore(corrupt=True)

    entries: List[PeerEntry] = []
    for item in raw.get("entries", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("user_key") or "").strip()
        period = str(item.get("period") or "").strip()
        value = item.get("twr_pct")
        if not key or not period or not isinstance(value, (int, float)):
            continue
        entries.append(PeerEntry(key, period, float(value),
                                 str(item.get("updated_at") or "")))
    return PeerStore(tuple(entries))


def save_store(store: PeerStore, path: Optional[Path] = None) -> bool:
    """Persist. Refuses to write over a store that could not be read —
    this file holds other people's entries, not just yours."""
    if store.corrupt:
        return False
    path = path or _store_path()
    # Written field by field rather than by asdict(), so a field added to
    # PeerEntry later cannot silently start being shared.
    payload = {"entries": [{
        "user_key": e.user_key, "period": e.period,
        "twr_pct": e.twr_pct, "updated_at": e.updated_at,
    } for e in store.entries]}
    atomic_write_text(path, json.dumps(payload, indent=2))
    return True


# --- opting in and out --------------------------------------------------------

def is_participating(store: PeerStore, user_key: str,
                     period: Optional[str] = None) -> bool:
    period = period or period_for()
    return store.entry_for(user_key, period) is not None


def publish(store: PeerStore, user_key: str, twr_pct: Optional[float],
            period: Optional[str] = None) -> Tuple[PeerStore, str]:
    """Record this account's return for the period. Returns (store, error).

    Takes the RETURN, not the portfolio — the signature is the privacy
    boundary. There is deliberately no parameter here through which
    holdings or a market value could arrive.
    """
    user_key = (user_key or "").strip()
    if not user_key:
        return store, "Sign in to compare with other people on this instance."
    if twr_pct is None:
        return store, ("There is no return to share yet — a portfolio needs "
                       "at least two days of value in this month.")
    period = period or period_for()

    kept = tuple(e for e in store.entries
                 if not (e.user_key == user_key and e.period == period))
    entry = PeerEntry(user_key, period, float(twr_pct), _now_iso())
    return replace(store, entries=kept + (entry,)), ""


def withdraw(store: PeerStore, user_key: str) -> PeerStore:
    """Remove EVERY entry for this account, not just the current month.

    Opting out has to mean gone. Leaving last month's figure behind would
    make the choice cosmetic, and someone withdrawing is usually
    withdrawing from the whole idea rather than from one period.
    """
    return replace(store, entries=tuple(e for e in store.entries
                                        if e.user_key != user_key))


# --- the comparison -----------------------------------------------------------

def compare(store: PeerStore, user_key: str,
            period: Optional[str] = None) -> Comparison:
    """Where this account sits, or why it cannot be said.

    Every refusal path returns a `reason` rather than a number. A
    percentile that appears without a cohort behind it is the failure
    this whole module is arranged to avoid.
    """
    period = period or period_for()
    mine = store.entry_for(user_key, period)
    if mine is None:
        return Comparison(period, reason=(
            "You are not sharing a return for this month, so there is "
            "nothing to compare."))

    others = tuple(e for e in store.for_period(period) if e.user_key != user_key)
    cohort = len(others)
    if cohort < PEER_COMPARISON.min_cohort:
        return Comparison(
            period, your_return_pct=mine.twr_pct, cohort_size=cohort,
            reason=(
                f"{cohort} other account(s) on this instance are sharing a "
                f"return for {period}. Quantix shows a percentile only once "
                f"{PEER_COMPARISON.min_cohort} are: below that a percentage "
                "would say more about the handful of people in it than about "
                "you, and would come close to revealing their individual "
                "returns."
            ))

    beat = sum(1 for e in others if e.twr_pct < mine.twr_pct)
    return Comparison(
        period, your_return_pct=mine.twr_pct, cohort_size=cohort,
        beat_count=beat, percentile=100.0 * beat / cohort)


# --- the window-scoped return -------------------------------------------------

def window_return(value: pd.Series, flows: pd.Series,
                  period: Optional[str] = None) -> Optional[float]:
    """Time-weighted return over the calendar month, or None.

    THIS IS THE FUNCTION THAT MAKES THE COMPARISON MEAN ANYTHING. The
    portfolio dashboard's headline return runs from each portfolio's own
    earliest purchase, so two people's figures cover different spans and
    ranking them measures who started sooner. Slicing both the value and
    the flow series to the same calendar month, and chaining the return
    over that slice, is what puts everyone on one window.

    Returns None rather than 0.0 when the month holds fewer than two
    valuations — a portfolio opened yesterday has no monthly return, and
    saying 0% would enter it in the ranking as though it had.
    """
    from portfolio_holdings import time_weighted_return

    if value is None or getattr(value, "empty", True):
        return None
    start, end = period_bounds(period or period_for())
    if start is None:
        return None

    try:
        index = pd.to_datetime(value.index)
        mask = (index.date >= start) & (index.date <= end)
        window_value = value[mask]
        if flows is None or getattr(flows, "empty", True):
            window_flows = pd.Series(0.0, index=window_value.index)
        else:
            flow_index = pd.to_datetime(flows.index)
            window_flows = flows[(flow_index.date >= start) & (flow_index.date <= end)]
    except Exception:
        log_exception(logger, "peer_comparison.window_failed",
                      section="peer_comparison")
        return None

    if len(window_value) < 2:
        return None
    return time_weighted_return(window_value, window_flows)


def flows_from_holdings(holdings: Sequence, index) -> pd.Series:
    """The external cash flow series, rebuilt from the holdings.

    `PortfolioPerformance` keeps the value series but not the flows, and
    the flows are not optional: time_weighted_return subtracts them so
    that money ARRIVING is not counted as a gain. Without this, a
    purchase made inside the month would register as a spectacular return
    and the person who deposited most would top the ranking.

    Built the same way build_value_series builds it — each purchase's
    cost on its purchase date — rather than re-deriving prices, which the
    dashboard does not hand back.
    """
    flows = pd.Series(0.0, index=index)
    if flows.empty:
        return flows
    stamps = pd.to_datetime(flows.index)
    for holding in holdings or ():
        try:
            when = pd.Timestamp(holding.purchase_date)
        except Exception:
            continue
        matches = stamps == when
        if matches.any():
            flows.loc[matches] += float(getattr(holding, "cost_total", 0.0) or 0.0)
    return flows


def cohort_note(store: PeerStore, period: Optional[str] = None) -> str:
    """One line about who is in the pool, for the panel."""
    period = period or period_for()
    count = len(store.for_period(period))
    if count == 0:
        return (f"Nobody on this instance is sharing a return for {period} yet.")
    return (f"{count} account(s) on this instance are sharing a return for "
            f"{period}. Only the percentage is shared — never your holdings, "
            "their values, or what you paid.")
