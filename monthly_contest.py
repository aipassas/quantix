"""The monthly stock-picking contest — entered before the month starts,
scored as a binary against the benchmark, ranked over many months.

IT DOES NOT RANK THE MONTH'S RETURN, AND THAT IS A MEASUREMENT. Measured
over 30 large caps and 60 real months before any of this was designed:

    median volatility rank of the month's winner   6 of 30
    months won by a top-quartile-volatility name   58%   (chance 25%)
    same name wins back-to-back                     5%   (chance ~3%)
    last month's winner beats the median next       56%   (coin flip 50%)

The winners were MSTR (91% annualised vol), SMCI (89%), RIVN, AMD and
GME (94%). A highest-return contest ranks VOLATILITY, and winning it
carries no information about next month. Handing out a monthly prize for
that would teach people that the way to win is to pick the wildest
ticker on the board — in an app whose entire character is refusing to
present noise as insight.

SO THE MONTH IS A FACT AND THE RECORD IS THE RANKING. Each month's
result is reported plainly, with no prize and no ordering. What the
ladder ranks is the HIT RATE across months: how often your pick beat the
benchmark, a binary that does not reward volatility because a wild pick
that loses counts exactly like a quiet one that loses.

THE NULL IS 50.7%, NOT 50%. Measured over 3,435 stock-months across ten
years: 50.7% of stock-months beat SPY. That universe is large-cap
survivors, so the figure flatters a random pick — a broad universe sits
below 50% because index returns come from a right tail. It is used
anyway, because it is the number actually measured here and erring that
way sets a HIGHER bar, not a lower one.

A SHORT RECORD MEANS NOTHING AND SAYS SO. Exact binomial, one-sided,
p<0.05 against that null: three months cannot reach significance at all
(even 3 for 3), six months needs 6/6, twelve needs 10/12, twenty-four
needs 17/24. So a record is always shown, and it is marked as separable
from chance only when it genuinely is. Ranking by hit rate with two
entries at the top would otherwise crown whoever had the shortest
record — the same trap peer_comparison's cohort floor exists for.

ENTRIES CLOSE BEFORE THE MONTH BEGINS. A pick made on the 20th has seen
twenty days of the return it is predicting, and measuring from the entry
date instead would give every entrant a different window — the mismatched
-window problem peer_comparison was built to fix. So during September you
enter for October, everyone is scored over the identical calendar month,
and `open_period()` is what the panel offers.

THE MONTH LABEL IS peer_comparison's. Imported rather than reimplemented:
two definitions of "this month" in one app is how two panels start
disagreeing about which month it is. Note the direction of the
dependency — this module reads that one's calendar and nothing else.
"""
import datetime
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scipy import stats

from config import CONTEST
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_exception
from peer_comparison import period_bounds, period_for

logger = get_logger("monthly_contest")

NOT_SIGNED_IN = "Sign in to enter the monthly contest."
NEEDS_PROFILE = (
    "Set a display name in your profile first — the contest lists entrants by "
    "the name they chose, never by an account key."
)
NEEDS_THESIS = (
    "Say in a sentence why. An entry with no reasoning teaches nobody anything "
    "afterwards, including you."
)
NEEDS_TICKER = "Name the stock you are picking."

WHY_NOT_RANKED_BY_RETURN = (
    "The month's returns are reported, not ranked, and there is no prize for the "
    "biggest one. Measured over 30 large caps and 60 months, the best performer in "
    "a month sits at a median volatility rank of 6 out of 30, 58% of months are won "
    "by a top-quartile-volatility name against 25% by chance, and last month's "
    "winner beats the median the next month 56% of the time — a coin flip. A "
    "highest-return contest ranks volatility, so this ranks how often a pick beat "
    "the benchmark instead."
)

SHORT_RECORD = (
    "A few months of picks cannot be told apart from chance. Against the measured "
    "base rate of {null:.1f}%, an exact binomial test needs 6 of 6 to clear p<0.05 "
    "at six months, 10 of 12 at twelve, and 17 of 24 at twenty-four. Records below "
    "that are shown but not marked as skill."
)


@dataclass(frozen=True)
class Entry:
    """One account's pick for one month.

    A ticker and a sentence — never a position size, a price target or a
    direction, because none of those is what is being scored and each
    would read as advice to whoever saw it.
    """
    user_key: str
    period: str            # "2026-10"
    ticker: str
    thesis: str
    entered_at: str


@dataclass(frozen=True)
class ContestStore:
    entries: Tuple[Entry, ...] = ()
    corrupt: bool = False

    def for_period(self, period: str) -> Tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.period == period)

    def entry_for(self, user_key: str, period: str) -> Optional[Entry]:
        return next((e for e in self.entries
                     if e.user_key == user_key and e.period == period), None)

    def for_user(self, user_key: str) -> Tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.user_key == user_key)


@dataclass(frozen=True)
class Result:
    """How one entry did. `beat` is None when it could not be measured —
    never False, which would silently count an unmeasurable month as a
    loss and drag a record down for a data problem."""
    entry: Entry
    pick_pct: Optional[float] = None
    benchmark_pct: Optional[float] = None
    beat: Optional[bool] = None
    reason: str = ""

    @property
    def measured(self) -> bool:
        return self.beat is not None


@dataclass(frozen=True)
class Record:
    """One entrant's multi-month record."""
    user_key: str
    name: str
    months: int = 0            # months actually MEASURED, not months entered
    hits: int = 0
    entered: int = 0
    significant: bool = False  # separable from the measured null at alpha

    @property
    def hit_rate(self) -> Optional[float]:
        return (100.0 * self.hits / self.months) if self.months else None


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _store_path() -> Path:
    # Shared: a contest across accounts cannot live in one namespace.
    return shared_path(CONTEST.store_filename)


# --- the entry window ---------------------------------------------------------

def open_period(day: Optional[datetime.date] = None) -> str:
    """The month entries are currently open for — the NEXT one.

    A pick made inside the month it predicts has already seen part of
    that month's return. Scoring from the entry date instead would give
    every entrant a different window, which is exactly the mismatched
    -span problem peer_comparison exists to fix.
    """
    day = day or datetime.date.today()
    first = datetime.date(day.year, day.month, 1)
    following = (first + datetime.timedelta(days=32)).replace(day=1)
    return f"{following.year:04d}-{following.month:02d}"


def entries_close(period: str) -> Optional[datetime.date]:
    """The last day on which `period` can still be entered — the day
    before it starts."""
    start, _ = period_bounds(period)
    return (start - datetime.timedelta(days=1)) if start else None


def is_open(period: str, day: Optional[datetime.date] = None) -> bool:
    return period == open_period(day)


def scored_period(day: Optional[datetime.date] = None) -> str:
    """The month whose results are currently being shown — the one in
    progress, whose entries closed before it began."""
    return period_for(day or datetime.date.today())


# --- persistence --------------------------------------------------------------

def load_store(path: Optional[Path] = None) -> ContestStore:
    """Never raises. Corrupt is distinguished from missing, because the
    next entry must not overwrite everyone else's picks."""
    path = path or _store_path()
    if not path.exists():
        return ContestStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "contest.store_corrupt", section="monthly_contest")
        return ContestStore(corrupt=True)
    if not isinstance(raw, dict):
        return ContestStore(corrupt=True)

    entries: List[Entry] = []
    for item in raw.get("entries", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("user_key") or "").strip()
        period = str(item.get("period") or "").strip()
        ticker = str(item.get("ticker") or "").strip().upper()
        if not key or not period or not ticker:
            continue
        entries.append(Entry(key, period, ticker,
                             str(item.get("thesis") or ""),
                             str(item.get("entered_at") or "")))
    return ContestStore(tuple(entries))


def save_store(store: ContestStore, path: Optional[Path] = None) -> bool:
    if store.corrupt:
        return False
    path = path or _store_path()
    payload = {"entries": [{
        "user_key": e.user_key, "period": e.period, "ticker": e.ticker,
        "thesis": e.thesis, "entered_at": e.entered_at,
    } for e in store.entries]}
    atomic_write_text(path, json.dumps(payload, indent=2))
    return True


# --- entering -----------------------------------------------------------------

def enter(store: ContestStore, user_key: str, period: str, ticker: str,
          thesis: str, day: Optional[datetime.date] = None
          ) -> Tuple[ContestStore, str]:
    """Record or replace this account's pick for `period`. (store, error).

    Replacing is allowed while entries are open — the window has not
    begun, so nothing has been seen yet — and refused once it closes,
    which is the whole anti-hindsight guarantee.
    """
    user_key = (user_key or "").strip()
    ticker = (ticker or "").strip().upper()
    thesis = (thesis or "").strip()

    if not user_key:
        return store, NOT_SIGNED_IN
    if not ticker:
        return store, NEEDS_TICKER
    if not thesis:
        return store, NEEDS_THESIS
    if not is_open(period, day):
        closed = entries_close(period)
        return store, (
            f"Entries for {period} are closed"
            + (f" — they closed on {closed.isoformat()}, before the month began."
               if closed else ".")
            + " A pick made inside the month it predicts has already seen part of "
              "the answer.")

    thesis = thesis[:CONTEST.max_thesis_chars]
    kept = tuple(e for e in store.entries
                 if not (e.user_key == user_key and e.period == period))
    entry = Entry(user_key, period, ticker, thesis, _now_iso())
    return replace(store, entries=kept + (entry,)), ""


def withdraw(store: ContestStore, user_key: str, period: str,
             day: Optional[datetime.date] = None) -> Tuple[ContestStore, str]:
    """Remove this account's entry for a period that is still open.

    A closed month cannot be withdrawn from. Letting someone delete an
    entry once the month has begun would let them drop the picks that
    went badly and keep the ones that went well, which would make every
    hit rate on the ladder meaningless.
    """
    if not is_open(period, day):
        return store, ("That month has begun, so its entry stays on the record. "
                       "Withdrawing losing months would make every hit rate here "
                       "meaningless.")
    return replace(store, entries=tuple(
        e for e in store.entries
        if not (e.user_key == user_key and e.period == period))), ""


# --- scoring ------------------------------------------------------------------

def _window_return(prices, start: datetime.date,
                   end: datetime.date) -> Optional[float]:
    """Percent change across a date window, or None.

    Needs two bars INSIDE the window. Falling back to the nearest bar
    outside it would score a month partly on days that belong to another
    one.
    """
    if prices is None or getattr(prices, "empty", True):
        return None
    try:
        import pandas as pd
        index = pd.to_datetime(prices.index)
        window = prices[(index.date >= start) & (index.date <= end)]
    except Exception:
        log_exception(logger, "contest.window_failed", section="monthly_contest")
        return None
    values = [float(v) for v in window.values if v is not None and v == v]
    if len(values) < 2 or values[0] == 0:
        return None
    return ((values[-1] - values[0]) / values[0]) * 100.0


def measure(entry: Entry, pick_prices, benchmark_prices) -> Result:
    """Score one entry over its own month.

    Takes the price series rather than a ticker so this is testable
    without the network — the same shape investment_journal.measure_outcome
    uses.
    """
    start, end = period_bounds(entry.period)
    if start is None:
        return Result(entry, reason="That month could not be read.")
    if end >= datetime.date.today():
        end = min(end, datetime.date.today())

    pick = _window_return(pick_prices, start, end)
    bench = _window_return(benchmark_prices, start, end)
    if pick is None or bench is None:
        return Result(entry, pick_pct=pick, benchmark_pct=bench, reason=(
            f"Not enough price history inside {entry.period} to score this yet."))
    return Result(entry, pick_pct=pick, benchmark_pct=bench, beat=pick > bench)


def is_significant(hits: int, months: int,
                   null: Optional[float] = None,
                   alpha: Optional[float] = None) -> bool:
    """Whether a record is separable from the measured base rate.

    One-sided exact binomial — not a normal approximation, because the
    counts here are single digits and the approximation is worst exactly
    there. At three months nothing can pass, including 3 out of 3, which
    is the correct answer rather than an edge case to work around.
    """
    null = CONTEST.null_hit_rate if null is None else null
    alpha = CONTEST.significance_alpha if alpha is None else alpha
    if months <= 0 or hits > months or hits < 0:
        return False
    # bool() is load-bearing: the comparison yields a numpy bool, which
    # is not `is False`, serialises as a numpy scalar, and would put a
    # numpy type into a dataclass that is otherwise plain Python.
    return bool(
        stats.binomtest(hits, months, null, alternative="greater").pvalue < alpha)


def records(results: Sequence[Result], profiles,
            viewer_key: str = "") -> Tuple[Record, ...]:
    """Every entrant's multi-month record, best hit rate first.

    Only MEASURED months count toward the rate. An entry whose prices
    could not be read is carried in `entered` so the reader can see the
    difference, but it is not scored as a loss — a data gap is not a
    wrong call.

    A name comes from the profile and is re-read here rather than stored
    with the entry, so deleting a profile drops the row, exactly as the
    leaderboard does.
    """
    tally: Dict[str, List[int]] = {}
    for result in results:
        key = result.entry.user_key
        counts = tally.setdefault(key, [0, 0, 0])   # months, hits, entered
        counts[2] += 1
        if result.measured:
            counts[0] += 1
            counts[1] += 1 if result.beat else 0

    out: List[Record] = []
    for key, (months, hits, entered) in tally.items():
        profile = profiles.get(key) if profiles else None
        name = (getattr(profile, "name", "") or "").strip()
        if not name:
            continue
        out.append(Record(key, name, months, hits, entered,
                          is_significant(hits, months)))

    # Significant records first, then by hit rate, then by how many
    # months back it goes — a 2-of-2 must not outrank a 17-of-24. Ties
    # resolve by name so two readers see the same order.
    out.sort(key=lambda r: (not r.significant, -(r.hit_rate or -1), -r.months,
                            r.name.lower()))
    return tuple(out)


def short_record_note() -> str:
    return SHORT_RECORD.format(null=CONTEST.null_hit_rate * 100.0)
