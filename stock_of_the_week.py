"""The weekly highlight card — one name from the Institutional basket,
frozen for the ISO week, with a writeup built only from figures the app
already computed.

THE SCORE CANNOT RANK, SO THE WEEK DOES. The obvious build is "take the
highest-scoring name in the basket". Measured on 2026-09-23 before
anything was written, that does not work: `screen_watchlist` awards one
point per passed check out of four, so its range is {0, 25, 50, 75, 100}
over sixteen names, and the live distribution was

    100  MSFT GOOGL NVDA META
     75  AAPL AMZN V LLY AVGO ASML CAT
     50  MA UNH WMT
     25  JPM COST

A four-way tie at the top, broken alphabetically, makes GOOGL the "Stock
of the Week" every week forever — these are annual-statement figures that
move once a quarter at most. A feature whose whole purpose is to give
someone a reason to open the app cannot show the same card indefinitely.

So the score is used as a GATE and the week as the CHOOSER. Everything at
or above `min_score` is equally eligible, and among the eligible the name
featured longest ago goes first; a name never featured goes before all of
them. That is deterministic, needs no tiebreaker the data cannot support,
and cycles the whole qualifying cohort before repeating — eleven weeks on
the distribution above. The card says it rotates, because a reader who
thinks the app picked a WINNER has been told something false.

NOTHING IS PROMOTED BELOW THE GATE. In a week when no name clears it,
`choose()` returns no pick and the reason, and the panel says so. The
alternative — featuring the least-bad name — would attach the word
"noteworthy" to something the app's own pre-screen just rejected.

THE PICK IS FROZEN AND SHARED. Once a week's pick is recorded it is
reused for the rest of that ISO week, so the card does not change under a
reader who refreshes, and the store is shared rather than per-user so
everyone on the instance sees the same name. That is what makes it
discussable; a per-account pick would be a recommendation engine, which
this app deliberately does not have (see recommendations.py).

THE WRITEUP MAKES NO PREDICTION. Every sentence carries a figure computed
elsewhere in the app and shown elsewhere in the app: the four pre-screen
checks with their thresholds, profitability and growth off the same
cached bundle the alignment cards already paid for, and the trailing
week's price move as a stated fact. There is no target, no rating and no
"expected to" anywhere in it.

A CHECK THAT COULD NOT BE EVALUATED IS NOT A CHECK THAT FAILED. The score
counts an unreported debt-to-equity as a miss — pre-existing behaviour of
`screen_watchlist`, preserved deliberately because changing it would move
every basket score on the page above. But the writeup distinguishes them:
`WatchlistCheck.evaluable` drives "not reported" rather than "misses", so
the card never tells a reader a company failed a test nobody ran.
"""
import datetime
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import STOCK_OF_THE_WEEK, WATCHLIST
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_exception
from quick_stats import NOT_REPORTED, StatSpec, format_value

logger = get_logger("stock_of_the_week")

# What "noteworthy" is allowed to mean here, in the UI's own words. Kept
# as a constant so the panel and the tests assert the same sentence.
DISCLOSURE = (
    "Noteworthy means it passes the published pre-screen checks below — not that it "
    "will rise. No price prediction, target or rating is made here, and the feature "
    "rotates between every qualifying name rather than ranking them."
)

NO_CANDIDATES = (
    "The Institutional basket could not be screened this week, so there is no "
    "highlight — every name was missing the price/earnings or margin figure the "
    "pre-screen needs."
)

# Phrased as an absence on purpose: a week with nothing above the gate is
# a real answer about the basket, not a failure of the panel.
NO_QUALIFIER = (
    "No name in the Institutional basket reached {gate:.0f}% on the pre-screen this "
    "week, so nothing is featured. The strongest was {best} at {best_score:.0f}%."
)

# The extra figures, in display order. Units come from quick_stats for the
# reason alignment_card records: StandardizedFinancials mixes fractions
# (return_on_equity, earnings_growth) with already-percent values
# (dividend_yield_pct), and re-deriving that here is how the two copies
# start disagreeing.
FIGURE_FIELDS: Tuple[StatSpec, ...] = (
    StatSpec("return_on_equity", "ROE", "fundamental", "fraction_percent"),
    StatSpec("revenue_growth", "revenue growth", "fundamental", "fraction_percent"),
    StatSpec("earnings_growth", "earnings growth", "fundamental", "fraction_percent"),
    StatSpec("market_cap", "market cap", "fundamental", "money"),
)


@dataclass(frozen=True)
class Candidate:
    """One basket name as the chooser sees it.

    Built by the caller from the hourly basket scan that already runs for
    the alignment cards, so selecting a pick costs no fetch of its own.
    `checks` comes straight off `WatchlistScore` rather than being
    recomputed against a second copy of the WATCHLIST thresholds.
    """
    ticker: str
    score: float
    status: str
    checks: Tuple = ()                        # fundamental_analysis.WatchlistCheck
    figures: Dict[str, Optional[float]] = None  # keys of FIGURE_FIELDS
    sector: str = ""

    @property
    def passed(self) -> Tuple:
        return tuple(c for c in self.checks if c.passed)

    @property
    def missed(self) -> Tuple:
        """Checks the company was measured on and did not clear."""
        return tuple(c for c in self.checks if not c.passed and c.evaluable)

    @property
    def unreported(self) -> Tuple:
        return tuple(c for c in self.checks if not c.evaluable)


@dataclass(frozen=True)
class Pick:
    """A recorded pick. `score`/`status` are the figures AS OF the moment
    it was chosen — the card reports what qualified it, not what the same
    screen would say three days later."""
    week: str            # "2026-W39"
    ticker: str
    score: float
    status: str
    chosen_at: str


@dataclass(frozen=True)
class PickStore:
    picks: Tuple[Pick, ...] = ()
    corrupt: bool = False

    def for_week(self, week: str) -> Optional[Pick]:
        return next((p for p in self.picks if p.week == week), None)


@dataclass(frozen=True)
class WeeklyMove:
    """The trailing week's price move, or the reason there isn't one."""
    pct: Optional[float] = None
    start_date: str = ""
    end_date: str = ""
    sessions: int = 0
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.pct is not None


# --- the week -----------------------------------------------------------------

def week_for(day: Optional[datetime.date] = None) -> str:
    """The ISO week label a pick is filed under.

    ISO rather than "the last seven days" for the same reason
    peer_comparison uses a calendar month: it is identical for everyone by
    definition, where a trailing window would depend on when each reader
    happened to open the app and two people would see different cards.
    """
    day = day or datetime.date.today()
    year, week, _ = day.isocalendar()
    return f"{year:04d}-W{week:02d}"


def week_bounds(week: str) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    """(Monday, Sunday) of a week label, or (None, None) if unparseable."""
    try:
        year_part, week_part = week.split("-W")
        monday = datetime.date.fromisocalendar(int(year_part), int(week_part), 1)
    except Exception:
        return None, None
    return monday, monday + datetime.timedelta(days=6)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# --- persistence --------------------------------------------------------------

def _store_path() -> Path:
    # Shared by definition: one pick per week for the whole instance. A
    # per-user copy would let two people on the same team see different
    # "Stock of the Week", which is not what the phrase means.
    return shared_path(STOCK_OF_THE_WEEK.store_filename)


def load_store(path: Optional[Path] = None) -> PickStore:
    """Never raises. A file that exists but cannot be read is flagged
    corrupt rather than reported empty — treating it as empty would let
    the next save replace the whole rotation history, and the rotation IS
    that history."""
    path = path or _store_path()
    if not path.exists():
        return PickStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "stock_of_the_week.store_corrupt",
                      section="stock_of_the_week")
        return PickStore(corrupt=True)
    if not isinstance(raw, dict):
        return PickStore(corrupt=True)

    picks: List[Pick] = []
    for item in raw.get("picks", []) or []:
        if not isinstance(item, dict):
            continue
        week = str(item.get("week") or "").strip()
        ticker = str(item.get("ticker") or "").strip().upper()
        if not week or not ticker:
            continue
        score = item.get("score")
        picks.append(Pick(
            week=week,
            ticker=ticker,
            score=float(score) if isinstance(score, (int, float)) else 0.0,
            status=str(item.get("status") or ""),
            chosen_at=str(item.get("chosen_at") or ""),
        ))
    return PickStore(tuple(picks))


def save_store(store: PickStore, path: Optional[Path] = None) -> bool:
    """Persist. Refuses to write over a store that could not be read."""
    if store.corrupt:
        return False
    path = path or _store_path()
    payload = {"picks": [{
        "week": p.week, "ticker": p.ticker, "score": p.score,
        "status": p.status, "chosen_at": p.chosen_at,
    } for p in store.picks]}
    atomic_write_text(path, json.dumps(payload, indent=2))
    return True


def record(store: PickStore, pick: Pick) -> PickStore:
    """Add `pick`, replacing any existing pick for the same week and
    trimming the history to `max_history`. Kept in chronological order,
    because the rotation reads position in this list as recency."""
    kept = [p for p in store.picks if p.week != pick.week]
    kept.append(pick)
    kept.sort(key=lambda p: p.week)
    return replace(store, picks=tuple(kept[-STOCK_OF_THE_WEEK.max_history:]))


# --- choosing -----------------------------------------------------------------

def eligible(candidates: Sequence[Candidate]) -> Tuple[Candidate, ...]:
    """Everything at or above the quality gate. Not an ordering — see the
    module docstring: the score has five possible values, so this is a
    set, not a ranking."""
    return tuple(c for c in candidates
                 if c.score >= STOCK_OF_THE_WEEK.min_score)


def _recency(store: PickStore) -> Dict[str, int]:
    """ticker -> position of its most recent pick. Later position means
    more recently featured; a ticker absent from the history has none."""
    return {p.ticker: i for i, p in enumerate(store.picks)}


def rotation_order(candidates: Sequence[Candidate],
                   store: PickStore) -> Tuple[Candidate, ...]:
    """Eligible candidates, least-recently-featured first.

    A name never featured sorts ahead of every name that has been (-1).
    Remaining ties go to the higher score and then alphabetically, so the
    order is fully determined by the store and the candidates — two
    readers on the same instance in the same week compute the same list.
    """
    seen = _recency(store)
    return tuple(sorted(eligible(candidates),
                        key=lambda c: (seen.get(c.ticker, -1), -c.score, c.ticker)))


def choose(candidates: Sequence[Candidate], store: PickStore,
           week: Optional[str] = None) -> Tuple[Optional[Pick], str]:
    """The pick for `week`, or (None, reason).

    An already-recorded week is returned unchanged — that is the freeze,
    and it is checked before anything is screened so a mid-week change in
    the underlying figures cannot swap the card out from under a reader.
    """
    week = week or week_for()

    existing = store.for_week(week)
    if existing is not None:
        return existing, ""

    if not candidates:
        return None, NO_CANDIDATES

    order = rotation_order(candidates, store)
    if not order:
        best = max(candidates, key=lambda c: c.score)
        return None, NO_QUALIFIER.format(gate=STOCK_OF_THE_WEEK.min_score,
                                         best=best.ticker, best_score=best.score)

    winner = order[0]
    return Pick(week=week, ticker=winner.ticker, score=winner.score,
                status=winner.status, chosen_at=_now_iso()), ""


# --- the writeup --------------------------------------------------------------

def _phrase(label: str) -> str:
    """A check's label mid-sentence.

    Lowercased only when the label is an ordinary capitalised word —
    "Net margin" reads badly as "Net margin" inside a clause, but a blunt
    .lower() turned "P/E ratio" into "p/e ratio" on the live card. Any
    label carrying an internal capital is an acronym and stays as written.
    """
    return label.lower() if label[1:] == label[1:].lower() else label


def _checks_sentence(candidate: Candidate) -> str:
    passed, missed, unreported = (candidate.passed, candidate.missed,
                                  candidate.unreported)
    total = len(candidate.checks)
    if not total:
        return ""

    detail = ", ".join(f"{_phrase(c.label)} {c.display} ({c.benchmark})"
                       for c in passed)
    if len(passed) == total:
        head = (f"{candidate.ticker} clears all {total} of the basket pre-screen "
                f"checks — {detail}.")
    elif passed:
        head = (f"{candidate.ticker} clears {len(passed)} of the {total} basket "
                f"pre-screen checks — {detail}.")
    else:
        head = f"{candidate.ticker} clears none of the {total} basket pre-screen checks."

    tail = []
    if missed:
        tail.append("It misses " + ", ".join(
            f"{_phrase(c.label)} at {c.display} against {c.benchmark}" for c in missed) + ".")
    if unreported:
        # Never "misses": the figure was never available to judge.
        tail.append(", ".join(c.label for c in unreported)
                    + (" is" if len(unreported) == 1 else " are")
                    + " not reported for it, which the pre-screen counts as a miss.")
    return " ".join([head] + tail)


def _figures_sentence(candidate: Candidate) -> str:
    figures = candidate.figures or {}
    parts = []
    for spec in FIGURE_FIELDS:
        shown = format_value(spec, figures.get(spec.key))
        if shown != NOT_REPORTED:
            parts.append(f"{spec.label} {shown}")
    if not parts:
        return ""
    where = f" ({candidate.sector})" if candidate.sector else ""
    return f"Also reported{where}: " + ", ".join(parts) + "."


def _move_sentence(move: Optional[WeeklyMove]) -> str:
    if move is None:
        return ""
    if not move.available:
        return move.reason or "Its move over the past week could not be measured."
    direction = "up" if move.pct >= 0 else "down"
    window = (f" ({move.start_date} to {move.end_date})"
              if move.start_date and move.end_date else "")
    return (f"Over the last {move.sessions} trading sessions{window} it is "
            f"{direction} {abs(move.pct):.2f}%. That is a statement of what "
            f"happened, not a reason it was chosen.")


def _rotation_sentence(cohort_size: int, store: PickStore,
                       ticker: str) -> str:
    previous = [p for p in store.picks if p.ticker == ticker]
    again = (f" It was last featured in {previous[-2].week}."
             if len(previous) >= 2 else "")
    if cohort_size <= 1:
        return ("It is the only name in the Institutional basket at or above "
                f"{STOCK_OF_THE_WEEK.min_score:.0f}% on the pre-screen this week." + again)
    return (f"It is one of {cohort_size} names in the Institutional basket at or above "
            f"{STOCK_OF_THE_WEEK.min_score:.0f}% on the pre-screen this week; the "
            f"highlight rotates through them, longest-unfeatured first, rather than "
            f"ranking them." + again)


def writeup(candidate: Candidate, store: PickStore,
            cohort_size: int, move: Optional[WeeklyMove] = None) -> Tuple[str, ...]:
    """The "why it's noteworthy" paragraph, as sentences.

    Every one of them is either a figure computed elsewhere in the app or
    a statement about how the pick was made. An empty sentence is dropped
    rather than padded, so a thinly-covered company gets a shorter
    writeup instead of an invented one.
    """
    sentences = (
        _checks_sentence(candidate),
        _rotation_sentence(cohort_size, store, candidate.ticker),
        _figures_sentence(candidate),
        _move_sentence(move),
    )
    return tuple(s for s in sentences if s)


def weekly_move(closes: Sequence[float], dates: Sequence[str],
                sessions: Optional[int] = None) -> WeeklyMove:
    """Percent change across the last `sessions` bars.

    Takes the series rather than a ticker so it is testable without the
    network; the caller supplies it from the price history the app
    already loads. Needs sessions+1 bars — a five-session move is measured
    against the close BEFORE those five, not against the first of them,
    which would silently report a four-session move.
    """
    sessions = sessions or STOCK_OF_THE_WEEK.week_price_sessions
    values = [float(c) for c in closes if c is not None and c == c]
    if len(values) < sessions + 1:
        return WeeklyMove(reason=(
            f"Its move over the past week could not be measured — {len(values)} "
            f"usable closes available where {sessions + 1} are needed."))
    first, last = values[-(sessions + 1)], values[-1]
    if first == 0:
        return WeeklyMove(reason="Its move over the past week could not be measured.")
    window_dates = list(dates)[-(sessions + 1):]
    return WeeklyMove(
        pct=((last - first) / first) * 100,
        start_date=str(window_dates[0]) if window_dates else "",
        end_date=str(window_dates[-1]) if window_dates else "",
        sessions=sessions,
    )


def basket() -> Tuple[str, ...]:
    """The candidate universe. Deliberately the SAME tuple the alignment
    cards already scan — a second universe here would mean two lists to
    keep in step and a second hourly fetch."""
    return tuple(WATCHLIST.tech_basket) + tuple(WATCHLIST.diversified_basket)
