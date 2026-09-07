"""Why you bought it, and what happened next.

THE POINT IS THE GAP BETWEEN THE TWO. A journal that only stores the
reasoning is a diary; one that only stores the outcome is a P&L. What
makes this worth returning to is holding a decision's stated reasoning
beside what the price actually did since, so the reader can see where
their thinking was right, where it was lucky, and where it was neither.

OUTCOMES ARE MEASURED, NEVER STORED. The return since a decision is
recomputed from price history every time it is shown, not written into
the entry at save time. A stored number would be frozen at the moment of
writing and quietly wrong forever after — and this is precisely the
feature whose value depends on the number being current.

CONVICTION IS RECORDED BECAUSE IT IS THE LEARNABLE PART. Whether a call
was right is mostly noise over one quarter. Whether you were more often
right when you felt certain than when you did not is a pattern about
YOU, and it needs the confidence to have been written down BEFORE the
outcome was known. That is the whole reason the field exists and the
reason it cannot be edited once the outcome is in — see `lock_reason`.

WHAT THIS DOES NOT DO. It does not score a decision as good or bad. A
buy that fell is not a mistake and a buy that rose is not a skill: over
one holding period those are mostly the market, and a scoreboard would
teach exactly the wrong lesson. The panel reports the move, the holding
period and what the market did over the same window, and leaves the
judgement to the person who wrote the reasoning.
"""
import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_event, log_exception

logger = get_logger(__name__)

STORE_FILENAME = "investment_journal.json"

BUY, SELL, HOLD, WATCH = "Buy", "Sell", "Hold", "Watch"
ACTIONS: Tuple[str, ...] = (BUY, SELL, HOLD, WATCH)

# The directions each action is a bet on, used to phrase the outcome from
# the decision's own point of view: a Sell that was followed by a fall
# went the way the writer expected, and calling that "-12%" without
# saying so reads as a loss.
BULLISH_ACTIONS: Tuple[str, ...] = (BUY, HOLD)
BEARISH_ACTIONS: Tuple[str, ...] = (SELL,)

CONVICTION_LEVELS: Tuple[str, ...] = ("Low", "Medium", "High")

MAX_REASONING_CHARS = 4000

# Below this many days the "outcome" is noise: a week's move says nothing
# about a thesis, and showing it invites reading one anyway.
MIN_DAYS_FOR_OUTCOME = 21

NOT_A_SCOREBOARD = (
    "This does not score a decision as right or wrong. Over one holding "
    "period a rise or a fall is mostly the market, and a scoreboard "
    "would teach the wrong lesson. What is shown is the move, the "
    "holding period and the benchmark over the same window — the "
    "judgement stays with whoever wrote the reasoning."
)

CONVICTION_IS_THE_LEARNABLE_PART = (
    "Conviction is recorded before the outcome is known, and cannot be "
    "edited afterwards. Whether a single call was right is mostly noise; "
    "whether you are more often right when you felt certain is a pattern "
    "about you — and it only means anything if the confidence was "
    "written down first."
)


# --- entries ------------------------------------------------------------------

@dataclass(frozen=True)
class JournalEntry:
    id: str
    ticker: str
    action: str
    reasoning: str
    created_at: str                 # ISO date-time, when it was written
    decided_on: str                 # ISO date the decision applies to
    conviction: str = "Medium"
    price_at_decision: Optional[float] = None
    # Set the first time an outcome is displayed for this entry, so the
    # conviction can be frozen from that moment. Editing confidence after
    # seeing the result is the one change that would make the whole
    # feature lie.
    outcome_seen_at: str = ""
    review_note: str = ""           # written later, deliberately separate
    reviewed_at: str = ""

    @property
    def locked(self) -> bool:
        return bool(self.outcome_seen_at)

    @property
    def lock_reason(self) -> str:
        if not self.locked:
            return ""
        return ("Conviction was frozen when this entry's outcome was "
                "first shown. Changing it now would let hindsight edit "
                "the record it exists to test.")

    @property
    def decided_date(self) -> Optional[datetime.date]:
        return _parse_date(self.decided_on)


def _parse_date(value) -> Optional[datetime.date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class JournalStore:
    entries: Tuple[JournalEntry, ...] = ()
    corrupt: bool = False

    def for_ticker(self, ticker: str) -> Tuple[JournalEntry, ...]:
        wanted = str(ticker or "").strip().upper()
        return tuple(e for e in self.entries if e.ticker == wanted)

    @property
    def tickers(self) -> Tuple[str, ...]:
        return tuple(sorted({e.ticker for e in self.entries}))


def _path() -> Path:
    return store_path(STORE_FILENAME)


def load_store(path: Optional[Path] = None) -> JournalStore:
    """A missing file is an empty journal; an unreadable one is REPORTED.

    The distinction matters here more than almost anywhere else in the
    app: treating a corrupt journal as empty means the next entry
    overwrites everything the user has written, and this is the one
    store whose contents cannot be re-fetched from anywhere.
    """
    path = path or _path()
    if not path.exists():
        return JournalStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:                              # noqa: BLE001
        log_exception(logger, "investment_journal.store_corrupt")
        return JournalStore(corrupt=True)
    if not isinstance(raw, dict):
        return JournalStore(corrupt=True)

    entries: List[JournalEntry] = []
    for item in raw.get("entries") or []:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        reasoning = str(item.get("reasoning") or "").strip()
        if not ticker or not reasoning:
            continue
        action = str(item.get("action") or BUY)
        entries.append(JournalEntry(
            id=str(item.get("id") or uuid.uuid4().hex[:12]),
            ticker=ticker,
            action=action if action in ACTIONS else BUY,
            reasoning=reasoning[:MAX_REASONING_CHARS],
            created_at=str(item.get("created_at") or ""),
            decided_on=str(item.get("decided_on") or "")[:10],
            conviction=(str(item.get("conviction"))
                        if item.get("conviction") in CONVICTION_LEVELS
                        else "Medium"),
            price_at_decision=_float(item.get("price_at_decision")),
            outcome_seen_at=str(item.get("outcome_seen_at") or ""),
            review_note=str(item.get("review_note") or ""),
            reviewed_at=str(item.get("reviewed_at") or ""),
        ))
    entries.sort(key=lambda e: (e.decided_on, e.created_at), reverse=True)
    return JournalStore(tuple(entries))


def _float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        return None if number != number else number
    except (TypeError, ValueError):
        return None


def save_store(store: JournalStore, path: Optional[Path] = None) -> Optional[str]:
    """Refuses to write over a store that could not be read.

    Returning an error rather than raising keeps the caller's panel
    alive, and refusing the write is what stops one bad parse erasing a
    journal that exists nowhere else.
    """
    if store.corrupt:
        return ("The journal file could not be read, so nothing was "
                "saved — writing now would overwrite entries that are "
                "still in the file. Move or delete "
                f"{STORE_FILENAME} and reload.")
    path = path or _path()
    payload = {"entries": [{
        "id": e.id, "ticker": e.ticker, "action": e.action,
        "reasoning": e.reasoning, "created_at": e.created_at,
        "decided_on": e.decided_on, "conviction": e.conviction,
        "price_at_decision": e.price_at_decision,
        "outcome_seen_at": e.outcome_seen_at,
        "review_note": e.review_note, "reviewed_at": e.reviewed_at,
    } for e in store.entries]}
    atomic_write_text(path, json.dumps(payload, indent=2))
    return None


# --- writing ------------------------------------------------------------------

def add_entry(store: JournalStore, ticker: str, action: str, reasoning: str,
              conviction: str = "Medium",
              decided_on: Optional[datetime.date] = None,
              price_at_decision: Optional[float] = None
              ) -> Tuple[JournalStore, Optional[str]]:
    """A new decision. Reasoning is required — that is the whole point."""
    ticker = str(ticker or "").strip().upper()
    reasoning = str(reasoning or "").strip()
    if not ticker:
        return store, "Name the ticker this decision is about."
    if not reasoning:
        return store, ("Write the reasoning. An entry without it records "
                       "that you acted but not why, which is the part "
                       "worth coming back to.")
    if action not in ACTIONS:
        return store, f"Action must be one of {', '.join(ACTIONS)}."
    if conviction not in CONVICTION_LEVELS:
        return store, f"Conviction must be one of {', '.join(CONVICTION_LEVELS)}."
    decided = decided_on or datetime.date.today()
    if decided > datetime.date.today():
        return store, "A decision cannot be dated in the future."

    entry = JournalEntry(
        id=uuid.uuid4().hex[:12], ticker=ticker, action=action,
        reasoning=reasoning[:MAX_REASONING_CHARS], created_at=_now_iso(),
        decided_on=decided.isoformat(), conviction=conviction,
        price_at_decision=_float(price_at_decision))
    log_event(logger, logging.INFO, "journal.entry_added",
              ticker=ticker, action=action, conviction=conviction)
    return JournalStore(tuple([entry] + list(store.entries))), None


def add_review(store: JournalStore, entry_id: str, note: str
               ) -> Tuple[JournalStore, Optional[str]]:
    """A later reflection, kept SEPARATE from the original reasoning.

    Editing the original would destroy the record this feature exists to
    keep — what you thought at the time, not what you now wish you had
    thought.
    """
    note = str(note or "").strip()
    if not note:
        return store, "Write something to review."
    updated = []
    found = False
    for entry in store.entries:
        if entry.id == entry_id:
            found = True
            updated.append(JournalEntry(
                **{**entry.__dict__, "review_note": note[:MAX_REASONING_CHARS],
                   "reviewed_at": _now_iso()}))
        else:
            updated.append(entry)
    if not found:
        return store, "That entry no longer exists."
    return JournalStore(tuple(updated)), None


def mark_outcome_seen(store: JournalStore, entry_ids: Sequence[str]
                      ) -> Tuple[JournalStore, bool]:
    """Freeze conviction on entries whose outcome has now been shown.

    Returns whether anything changed, so the caller only writes the
    store when it did.
    """
    wanted = {str(i) for i in entry_ids}
    changed = False
    updated = []
    for entry in store.entries:
        if entry.id in wanted and not entry.outcome_seen_at:
            changed = True
            updated.append(JournalEntry(
                **{**entry.__dict__, "outcome_seen_at": _now_iso()}))
        else:
            updated.append(entry)
    return JournalStore(tuple(updated), store.corrupt), changed


def delete_entry(store: JournalStore, entry_id: str) -> JournalStore:
    return JournalStore(
        tuple(e for e in store.entries if e.id != entry_id), store.corrupt)


def update_conviction(store: JournalStore, entry_id: str, conviction: str
                      ) -> Tuple[JournalStore, Optional[str]]:
    """Only while the outcome is still unknown.

    This is the single edit the feature must refuse, because allowing it
    lets hindsight rewrite the record that gives the pattern its
    meaning.
    """
    if conviction not in CONVICTION_LEVELS:
        return store, f"Conviction must be one of {', '.join(CONVICTION_LEVELS)}."
    updated = []
    for entry in store.entries:
        if entry.id == entry_id:
            if entry.locked:
                return store, entry.lock_reason
            updated.append(JournalEntry(
                **{**entry.__dict__, "conviction": conviction}))
        else:
            updated.append(entry)
    return JournalStore(tuple(updated), store.corrupt), None


# --- outcomes -----------------------------------------------------------------

@dataclass(frozen=True)
class Outcome:
    entry_id: str
    ticker: str
    action: str
    days_held: int = 0
    price_then: Optional[float] = None
    price_now: Optional[float] = None
    change_pct: Optional[float] = None
    benchmark_change_pct: Optional[float] = None
    benchmark: str = ""
    error: str = ""
    # A decision written today sits AFTER the last close — the market has
    # not printed a bar for it yet. That is not a missing-data failure,
    # it is an entry with nothing to measure, and saying "the price
    # history does not reach the decision date" reads as the former.
    too_new: bool = False

    @property
    def ok(self) -> bool:
        return self.change_pct is not None and not self.error

    @property
    def mature(self) -> bool:
        """Whether enough time has passed for the move to mean anything."""
        return self.ok and self.days_held >= MIN_DAYS_FOR_OUTCOME

    @property
    def went_as_expected(self) -> Optional[bool]:
        """Did the price move the way the decision was betting?

        A Sell followed by a fall went the writer's way; reporting that
        as "-12%" without saying so reads as a loss on their record.
        """
        if self.change_pct is None:
            return None
        if self.action in BULLISH_ACTIONS:
            return self.change_pct > 0
        if self.action in BEARISH_ACTIONS:
            return self.change_pct < 0
        return None            # Watch is not a bet

    @property
    def versus_benchmark_pct(self) -> Optional[float]:
        """The gap, but only where subtracting means something.

        For a Buy or a Hold, "the stock against the index" is the
        natural comparison: both are things you could have held. For a
        Sell it is not — the difference depends entirely on what the
        proceeds did, and this app does not know whether they went into
        the index or into cash. Reporting -20 points on a short thesis
        that WORKED reads as underperformance and is a counterfactual
        nobody supplied.
        """
        if self.change_pct is None or self.benchmark_change_pct is None:
            return None
        if self.action not in BULLISH_ACTIONS:
            return None
        return self.change_pct - self.benchmark_change_pct


def measure_outcome(entry: JournalEntry, prices, benchmark_prices=None,
                    benchmark: str = "") -> Outcome:
    """Recomputed from price history every time, never stored.

    `prices` is a date-indexed close series for the ticker. The entry's
    own recorded price is used as the starting point when it exists —
    it is what the writer actually saw — and the series' close on the
    decision date otherwise.
    """
    import pandas as pd

    decided = entry.decided_date
    if decided is None:
        return Outcome(entry.id, entry.ticker, entry.action,
                       error="This entry has no decision date.")
    if prices is None or len(prices) == 0:
        return Outcome(entry.id, entry.ticker, entry.action,
                       error="No price history for this ticker.")

    series = pd.Series(prices).dropna()
    if series.empty:
        return Outcome(entry.id, entry.ticker, entry.action,
                       error="No price history for this ticker.")
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    stamp = pd.Timestamp(decided)

    at_or_after = series[series.index >= stamp]
    if at_or_after.empty:
        # No bar at or after the decision date. If the decision is simply
        # newer than the last close — which every entry written today is,
        # before the session settles — that is "nothing to measure yet",
        # not a broken fetch.
        elapsed = (datetime.date.today() - decided).days
        # BOTH conditions: after the last bar, AND recent enough that we
        # would not be showing an outcome anyway. A decision six years
        # old with a series that stops before it is a data gap, not a
        # fresh entry — and an earlier draft called that "too new".
        if (decided >= series.index[-1].date()
                and 0 <= elapsed <= MIN_DAYS_FOR_OUTCOME):
            return Outcome(entry.id, entry.ticker, entry.action,
                           days_held=max(0, elapsed), too_new=True)
        return Outcome(entry.id, entry.ticker, entry.action,
                       error=("The price history does not reach the "
                              "decision date."))
    then = (entry.price_at_decision
            if entry.price_at_decision else float(at_or_after.iloc[0]))
    now = float(series.iloc[-1])
    if not then:
        return Outcome(entry.id, entry.ticker, entry.action,
                       error="No usable starting price.")

    days = int((series.index[-1] - stamp).days)
    change = 100.0 * (now / then - 1.0)

    bench_change = None
    if benchmark_prices is not None and len(benchmark_prices):
        bench = pd.Series(benchmark_prices).dropna()
        if not bench.empty:
            bench.index = pd.DatetimeIndex(
                bench.index).tz_localize(None).normalize()
            bench_after = bench[bench.index >= stamp]
            if not bench_after.empty and float(bench_after.iloc[0]):
                bench_change = 100.0 * (float(bench.iloc[-1])
                                        / float(bench_after.iloc[0]) - 1.0)

    return Outcome(entry.id, entry.ticker, entry.action, days, then, now,
                   change, bench_change, benchmark)


def describe_outcome(outcome: Outcome) -> str:
    if outcome.too_new:
        return ("No price has printed since this decision yet — there is "
                f"nothing to measure. Shown from "
                f"{MIN_DAYS_FOR_OUTCOME} days.")
    if not outcome.ok:
        return outcome.error or "No outcome could be measured."
    if not outcome.mature:
        return (f"{outcome.days_held} days since this decision — too "
                f"early to read anything into the move. Shown from "
                f"{MIN_DAYS_FOR_OUTCOME} days.")
    direction = "up" if outcome.change_pct >= 0 else "down"
    text = (f"{outcome.ticker} is {direction} "
            f"{abs(outcome.change_pct):.1f}% over the "
            f"{outcome.days_held} days since.")
    expected = outcome.went_as_expected
    if expected is not None:
        text += (" That is the direction this decision was betting on."
                 if expected else
                 " That is against the direction this decision was "
                 "betting on.")
    if outcome.versus_benchmark_pct is not None:
        text += (f" {outcome.benchmark} moved "
                 f"{outcome.benchmark_change_pct:+.1f}% over the same "
                 f"window, so the gap is "
                 f"{outcome.versus_benchmark_pct:+.1f} points.")
    elif outcome.benchmark_change_pct is not None:
        # Context, not a gap — see versus_benchmark_pct.
        text += (f" {outcome.benchmark} moved "
                 f"{outcome.benchmark_change_pct:+.1f}% over the same "
                 f"window; what this decision was worth against that "
                 f"depends on what the proceeds did, which is not "
                 f"recorded here.")
    return text


# --- the pattern across entries -----------------------------------------------

@dataclass(frozen=True)
class ConvictionRow:
    conviction: str
    decisions: int = 0
    matured: int = 0
    went_as_expected: int = 0

    @property
    def hit_rate_pct(self) -> Optional[float]:
        if self.matured < MIN_DECISIONS_FOR_PATTERN:
            return None
        return 100.0 * self.went_as_expected / self.matured

    @property
    def scored(self) -> bool:
        return self.hit_rate_pct is not None


# A hit rate over three decisions is a coin flip with a percentage sign.
# Ten is still small, and the panel says so rather than implying the
# number is stable.
MIN_DECISIONS_FOR_PATTERN = 10


def conviction_pattern(entries: Sequence[JournalEntry],
                       outcomes: Dict[str, Outcome]) -> Tuple[ConvictionRow, ...]:
    """Hit rate by stated conviction — the one pattern worth surfacing.

    Withheld below MIN_DECISIONS_FOR_PATTERN matured decisions per band,
    because a hit rate over a handful of calls is noise wearing a
    percentage sign, and this panel's whole claim is that it shows you
    something real about your own judgement.
    """
    rows = []
    for level in CONVICTION_LEVELS:
        band = [e for e in entries if e.conviction == level]
        matured = [outcomes[e.id] for e in band
                   if e.id in outcomes and outcomes[e.id].mature
                   and outcomes[e.id].went_as_expected is not None]
        rows.append(ConvictionRow(
            level, len(band), len(matured),
            sum(1 for o in matured if o.went_as_expected)))
    return tuple(rows)


def describe_pattern(rows: Sequence[ConvictionRow]) -> str:
    scored = [r for r in rows if r.scored]
    if not scored:
        total = sum(r.matured for r in rows)
        return (f"Not enough decisions have matured to read a pattern — "
                f"{total} so far, and a hit rate needs at least "
                f"{MIN_DECISIONS_FOR_PATTERN} in a band before it says "
                f"more than chance does.")
    parts = [f"{r.conviction}: {r.hit_rate_pct:.0f}% of {r.matured}"
             for r in scored]
    text = "Went as expected — " + "; ".join(parts) + "."
    if len(scored) > 1:
        high = max(scored, key=lambda r: r.hit_rate_pct)
        low = min(scored, key=lambda r: r.hit_rate_pct)
        if high.conviction != low.conviction:
            text += (f" Your {high.conviction.lower()}-conviction calls "
                     f"have gone your way more often than your "
                     f"{low.conviction.lower()}-conviction ones.")
    return text
